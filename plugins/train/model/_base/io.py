#!/usr/bin/env python3
"""IO handling for the model base plugin.

The objects in this module should not be called directly, but are called from
:class:`~plugins.train.model._base.ModelBase`

This module handles:
    - The loading, saving and backing up of keras models to and from disk.
    - The loading and freezing of weights for model plugins.
"""
from __future__ import annotations
import gc
import io
import json
import logging
import os
import sys
import typing as T
import zipfile

from keras import layers, models as k_models, Variable
import numpy as np
import torch

from lib.logger import parse_class_init
from lib.model.backup_restore import Backup
from lib.training.optimizer import get_parameter_group_ids
from lib.utils import get_module_objects, FaceswapError

from .update import Legacy, PatchKerasConfig

if T.TYPE_CHECKING:
    from keras.optimizers import Optimizer as K_Optimizer, LossScaleOptimizer
    from lib.training.optimizer import Optimizer
    from .model import ModelBase

logger = logging.getLogger(__name__)


def get_all_sub_models(
        model: k_models.Model,
        models: list[k_models.Model] | None = None) -> list[k_models.Model]:
    """For a given model, return all sub-models that occur (recursively) as children.

    Parameters
    ----------
    model
        A Keras model to scan for sub models
    models
        Do not provide this parameter. It is used for recursion

    Returns
    -------
    A list of all :class:`keras.models.Model` objects found within the given model. The provided
    model will always be returned in the first position
    """
    if models is None:
        models = [model]
    else:
        models.append(model)
    for layer in model.layers:
        if isinstance(layer, k_models.Model):
            get_all_sub_models(layer, models=models)
    return models


class IO():
    """Model saving and loading functions.

    Handles the loading and saving of the plugin model from disk as well as the model backup and
    snapshot functions.

    Parameters
    ----------
    plugin
        The parent plugin class that owns the IO functions.
    model_dir
        The full path to the model save location
    is_predict
        ``True`` if the model is being loaded for inference. ``False`` if the model is being loaded
        for training.
    save_optimizer
        When to save the optimizer weights. `"never"` never saves the optimizer weights. `"always"`
        always saves the optimizer weights. `"exit"` only saves the optimizer weights on an exit
        request.
    """
    def __init__(self,
                 plugin: ModelBase,
                 model_dir: str,
                 is_predict: bool,
                 save_optimizer: T.Literal["never", "always", "exit"]) -> None:
        logger.debug(parse_class_init(locals()))
        self._plugin = plugin
        self._is_predict = is_predict
        self._model_dir = model_dir
        self._do_save_optimizer = save_optimizer
        self._history: list[float] = []
        """Loss history for current save iteration"""
        self._backup = Backup(self._model_dir, self._plugin.name)
        self._update_legacy()

    @property
    def model_dir(self) -> str:
        """The full path to the model folder"""
        return self._model_dir

    @property
    def filename(self) -> str:
        """The filename for this model."""
        return os.path.join(self._model_dir, f"{self._plugin.name}.keras")

    @property
    def model_exists(self) -> bool:
        """``True`` if a model of the type being loaded exists within the model folder location
        otherwise ``False``."""
        return os.path.isfile(self.filename)

    @property
    def history(self) -> list[float]:
        """list of loss history for the current save iteration."""
        return self._history

    @property
    def multiple_models_in_folder(self) -> list[str] | None:
        """If there are multiple model types in the requested folder, or model types that don't
        correspond to the requested plugin type, then returns the list of plugin names that exist
        in the folder, otherwise returns ``None``"""
        plugins = [fname.replace(".keras", "")
                   for fname in os.listdir(self._model_dir)
                   if fname.endswith(".keras")]
        test_names = plugins + [self._plugin.name]
        test = False if not test_names else os.path.commonprefix(test_names) == ""
        retval = None if not test else plugins
        logger.debug("[IO] plugin name: %s, plugins: %s, test result: %s, retval: %s",
                     self._plugin.name, plugins, test, retval)
        return retval

    def _update_legacy(self) -> None:
        """Look for faceswap 2.x .h5 files in the model folder. If exists, then update to Faceswap
        3 .keras file and backup the original model .h5 file"""
        if self.model_exists:
            logger.debug("[IO] Existing model file is current: '%s'",
                         os.path.basename(self.filename))
            return

        old_fname = f"{os.path.splitext(self.filename)[0]}.h5"
        if not os.path.isfile(old_fname):
            logger.debug("[IO] No legacy model file to update")
            return

        Legacy(old_fname)

    def load(self) -> k_models.Model:
        """Loads the model from disk

        If the predict function is to be called and the model cannot be found in the model folder
        then an error is logged and the process exits.

        When loading the model, the plugin model folder is scanned for custom layers which are
        added to Keras' custom objects.

        Returns
        -------
        The saved model loaded from disk
        """
        logger.debug("[IO] Loading model: %s", self.filename)
        if self._is_predict and not self.model_exists:
            logger.error("Model could not be found in folder '%s'. Exiting", self._model_dir)
            sys.exit(1)

        try:
            model = k_models.load_model(self.filename, compile=False)
        except RuntimeError as err:
            if "unable to get link info" in str(err).lower():
                msg = (f"Unable to load the model from '{self.filename}'. This may be a "
                       "temporary error but most likely means that your model has corrupted.\n"
                       "You can try to load the model again but if the problem persists you "
                       "should use the Restore Tool to restore your model from backup.\n"
                       f"Original error: {str(err)}")
                raise FaceswapError(msg) from err
            raise err
        except KeyError as err:
            if "unable to open object" in str(err).lower():
                msg = (f"Unable to load the model from '{self.filename}'. This may be a "
                       "temporary error but most likely means that your model has corrupted.\n"
                       "You can try to load the model again but if the problem persists you "
                       "should use the Restore Tool to restore your model from backup.\n"
                       f"Original error: {str(err)}")
                raise FaceswapError(msg) from err
            if 'parameter name can\\\'t contain "."' in str(err).lower():
                PatchKerasConfig(self.filename)()
                return self.load()
            raise err
        except TypeError as err:
            if any(x in str(err) for x in ("Could not locate class 'Conv2D'",
                                           "Could not locate class 'DepthwiseConv2D'")):
                PatchKerasConfig(self.filename)()
                return self.load()
            raise err

        logger.info("Loaded model from disk: '%s'", self.filename)
        return model  # pyright:ignore[reportReturnType]

    def load_optimizer(self) -> dict[str, T.Any] | None:
        """Load the optimizer's state_dict from the .keras model file

        Returns
        -------
        The saved optimizer state_dict or ``None`` if it does not exist
        """
        logger.debug("[IO] Loading optimizer state_dict")
        opt_file = "optimizer.pt"
        keras_conf = "config.json"
        with zipfile.ZipFile(self.filename, "r") as z_file:
            f_list = z_file.namelist()
            if opt_file in f_list:  # Saved torch optimizer
                retval = torch.load(io.BytesIO(z_file.read(opt_file)))
            elif keras_conf in f_list:  # convert legacy keras optimizer
                conf = json.loads(z_file.read(keras_conf))
                retval = OptimizerMigrate(conf, self.filename).convert()
            else:
                retval = None

        if retval is None:
            logger.debug("[IO] No optimizer in .keras file")
            return None

        logger.debug("[IO] Loaded optimizer state_dict: %s",
                     {k: list(v) if isinstance(v, dict) else v for k, v in retval.items()})
        return retval

    def _save_optimizer(self, optimizer: Optimizer) -> None:
        """Inject the optimizer's state_dict into the .keras model file

        Parameters
        ----------
        optimizer
            The current optimizer in use for the model that is to be injected
        """
        logger.debug("[IO] Saving optimizer: %s", optimizer)
        buf = io.BytesIO()
        torch.save(optimizer.state_dict(), buf)
        opt_bytes = buf.getvalue()
        with zipfile.ZipFile(self.filename, "a") as z_file:
            z_file.writestr("optimizer.pt",
                            opt_bytes,
                            compress_type=zipfile.ZIP_DEFLATED,
                            compresslevel=1)

    def _save_model(self, optimizer: Optimizer | None, is_exit: bool) -> None:
        """Save the model either with or without the optimizer weights

        Parameters
        ----------
        optimizer
            The current optimizer in use for the model if it should be saved
        is_exit
            ``True`` if the save request has come from an exit process request otherwise ``False``.
        """
        include_optimizer = (self._do_save_optimizer == "always" or
                             (self._do_save_optimizer == "exit" and is_exit))

        self._plugin.model.save(self.filename)
        if include_optimizer and optimizer is not None:
            self._save_optimizer(optimizer)
        self._plugin.state.save()

    def _get_save_average(self) -> float:
        """Return the average loss since the last save iteration and reset historical loss

        Returns
        -------
        The average loss since the last save iteration
        """
        logger.debug("[IO] Getting save averages")
        if not self._history:
            logger.debug("[IO] No loss in history")
            retval = 0.0
        else:
            retval = sum(self._history) / len(self._history)
            self._history = []  # Reset historical loss
        logger.debug("[IO] Average loss since last save: %s", round(retval, 5))
        return retval

    def _should_backup(self, save_average: float) -> bool:
        """Check whether the loss average for this save iteration is the lowest that has been
        seen.

        This protects against model corruption by only backing up the model if the sum of all loss
        functions has fallen.

        Notes
        -----
        This is by no means a perfect system. If the model corrupts at an iteration close
        to a save iteration, then the averages may still be pushed lower than a previous
        save average, resulting in backing up a corrupted model. Changing loss weighting can also
        artificially impact this

        Parameters
        ----------
        save_average
            The average loss since the last save iteration
        """
        if not self._plugin.state.lowest_avg_loss:
            logger.debug("[IO] Set initial save iteration loss average: %s", save_average)
            self._plugin.state.lowest_avg_loss = save_average
            return False

        old_average = self._plugin.state.lowest_avg_loss
        backup = save_average < old_average

        if backup:  # Update lowest loss values to the state file
            self._plugin.state.lowest_avg_loss = save_average
            logger.debug("[IO] Updated lowest historical save iteration average from: %s to: %s",
                         old_average, save_average)

        logger.debug("[IO] Should backup: %s", backup)
        return backup

    def _maybe_backup(self) -> tuple[float, bool]:
        """Backup the model if total average loss has dropped for the save iteration

        Returns
        -------
        average_loss
            The total loss average since the last save iteration
        backed_up
            ``True`` if the model was backed up
        """
        save_average = self._get_save_average()
        should_backup = self._should_backup(save_average)
        if not save_average or not should_backup:
            logger.debug("[IO] Not backing up model (save_average: %s, should_backup: %s)",
                         save_average, should_backup)
            return save_average, False

        logger.debug("[IO] Backing up model")
        self._backup.backup_model(self.filename)
        self._backup.backup_model(self._plugin.state.filename)
        return save_average, True

    def save(self, optimizer: Optimizer | None = None, is_exit: bool = False) -> None:
        """Backup and save the model and state file.

        Parameters
        ----------
        optimizer
            The current optimizer in use for the model if it should be saved. Default: ``None``
        is_exit
            ``True`` if the save request has come from an exit process request otherwise ``False``.
            Default: ``False``
        """
        logger.debug("[IO] Backing up and saving models")
        print("\x1b[2K", end="\r")  # Clear last line
        logger.info("Saving Model...")

        self._save_model(optimizer, is_exit)
        save_average, backed_up = self._maybe_backup()

        msg = "[Saved model]"
        if save_average:
            msg += f" - Average total loss since last save: {save_average:.5f}"
        if backed_up:
            msg += " [Model backed up]"
        logger.info(msg)

    def snapshot(self) -> None:
        """Perform a model snapshot.

        Notes
        -----
        Snapshot function is called 1 iteration after the model was saved, so that it is built from
        the latest save, hence iteration being reduced by 1.
        """
        logger.debug("[IO] Performing snapshot. Iterations: %s", self._plugin.iterations)
        self._backup.snapshot_models(self._plugin.iterations - 1)
        logger.debug("[IO] Performed snapshot")


class Weights():
    """Handling of freezing and loading model weights

    Parameters
    ----------
    plugin
        The parent plugin class that owns the IO functions.
    """
    def __init__(self, plugin: ModelBase) -> None:
        logger.debug(parse_class_init(locals()))
        self._model = plugin.model
        self._name = plugin.model_name
        self._do_freeze = plugin._args.freeze_weights
        self._weights_file = self._check_weights_file(plugin._args.load_weights)

        self._freeze_layers = plugin.freeze_layers
        self._load_layers = plugin.load_layers

    @classmethod
    def _check_weights_file(cls, weights_file: str) -> str | None:
        """Validate that we have a valid path to a .keras file.

        Parameters
        ----------
        weights_file
            The full path to a weights file

        Returns
        -------
        The full path to a weights file
        """
        if not weights_file:
            logger.debug("[Weights] No weights file selected.")
            return None

        msg = ""
        if not os.path.exists(weights_file):
            msg = f"Load weights selected, but the path '{weights_file}' does not exist."
        elif not os.path.splitext(weights_file)[-1].lower() == ".keras":
            msg = (f"Load weights selected, but the path '{weights_file}' is not a valid Keras "
                   f"model (.keras) file.")

        if msg:
            msg += " Please check and try again."
            raise FaceswapError(msg)

        logger.verbose("Using weights file: %s", weights_file)  # type:ignore
        return weights_file

    def freeze(self) -> None:
        """If freeze has been selected in the cli arguments, then freeze those models indicated
        in the plugin's configuration. """
        # Blanket unfreeze layers, as checking the value of :attr:`layer.trainable` appears to
        # return ``True`` even when the weights have been frozen
        for layer in get_all_sub_models(self._model):
            layer.trainable = True

        if not self._do_freeze:
            logger.debug("[Weights] Freeze weights deselected. Not freezing")
            return

        for layer in get_all_sub_models(self._model):
            if layer.name in self._freeze_layers:
                logger.info("Freezing weights for '%s' in model '%s'", layer.name, self._name)
                layer.trainable = False
                self._freeze_layers.remove(layer.name)
        if self._freeze_layers:
            logger.warning("The following layers were set to be frozen but do not exist in the "
                           "model: %s", self._freeze_layers)

    def load(self, model_exists: bool) -> None:
        """Load weights for newly created models, or output warning for pre-existing models.

        Parameters
        ----------
        model_exists
            ``True`` if a model pre-exists and is being resumed, ``False`` if this is a new model
        """
        if not self._weights_file:
            logger.debug("[Weights] No weights file provided. Not loading weights.")
            return
        if model_exists and self._weights_file:
            logger.warning("Ignoring weights file '%s' as this model is resuming.",
                           self._weights_file)
            return

        weights_models = self._get_weights_model()
        all_models = get_all_sub_models(self._model)
        loaded_ops = 0
        skipped_ops = 0

        for model_name in self._load_layers:
            sub_model = next((lyr for lyr in all_models if lyr.name == model_name), None)
            sub_weights = next((lyr for lyr in weights_models if lyr.name == model_name), None)

            if not sub_model or not sub_weights:
                msg = f"Skipping layer {model_name} as not in "
                msg += "current_model." if not sub_model else f"weights '{self._weights_file}.'"
                logger.warning(msg)
                continue

            logger.info("Loading weights for layer '%s'", model_name)
            skipped_ops = 0
            loaded_ops = 0
            for layer in sub_model.layers:
                success = self._load_layer_weights(layer, sub_weights, model_name)
                if success == 0:
                    skipped_ops += 1
                elif success == 1:
                    loaded_ops += 1

        del weights_models

        if loaded_ops == 0:
            raise FaceswapError(f"No weights were successfully loaded from your weights file: "
                                f"'{self._weights_file}'. Please check and try again.")
        if skipped_ops > 0:
            logger.warning("%s weight(s) were unable to be loaded for your model. This is most "
                           "likely because the weights you are trying to load were trained with "
                           "different settings than you have set for your current model.",
                           skipped_ops)

    def _get_weights_model(self) -> list[k_models.Model]:
        """Obtain a list of all sub-models contained within the weights model.

        Returns
        -------
        List of all models contained within the .keras file

        Raises
        ------
        FaceswapError
            In the event of a failure to load the weights, or the weights belonging to a different
            model
        """
        retval = get_all_sub_models(k_models.load_model(    # pyright:ignore[reportArgumentType]
            self._weights_file,
            compile=False))
        if not retval:
            raise FaceswapError(f"Error loading weights file {self._weights_file}.")

        if retval[0].name != self._name:
            raise FaceswapError(f"You are attempting to load weights from a '{retval[0].name}' "
                                f"model into a '{self._name}' model. This is not supported.")
        return retval

    def _load_layer_weights(self,
                            layer: layers.Layer,
                            sub_weights: layers.Layer,
                            model_name: str) -> T.Literal[-1, 0, 1]:
        """Load the weights for a single layer.

        Parameters
        ----------
        layer
            The layer to set the weights for
        sub_weights
            The list of layers in the weights model to load weights from
        model_name
            The name of the current sub-model that is having it's weights loaded

        Returns
        -------
        `-1` if the layer has no weights to load. `0` if weights loading was unsuccessful. `1` if
        weights loading was successful
        """
        old_weights = layer.get_weights()
        if not old_weights:
            logger.debug("[Weights] Skipping layer without weights: %s", layer.name)
            return -1

        layer_weights = next((lyr for lyr in sub_weights.layers
                             if lyr.name == layer.name), None)
        if not layer_weights:
            logger.warning("The weights file '%s' for layer '%s' does not contain weights for "
                           "'%s'. Skipping", self._weights_file, model_name, layer.name)
            return 0

        new_weights = layer_weights.get_weights()
        if old_weights[0].shape != new_weights[0].shape:
            logger.warning("The weights for layer '%s' are of incompatible shapes. Skipping.",
                           layer.name)
            return 0
        logger.verbose("Setting weights for '%s'", layer.name)  # type:ignore
        layer.set_weights(layer_weights.get_weights())
        return 1


class OptimizerMigrate:
    """Migrates weights from a keras optimizer to a torch optimizer's state dict"""
    def __init__(self, config: dict[str, T.Any], model_path: str):
        logger.debug(parse_class_init(locals()))
        self._config = config
        self._model_path = model_path
        ada_map = (("_momentums", "_velocities"), ("exp_avg", "exp_avg_sq"))
        self._mapping: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
            "AdaBeliefOptimizer": (ada_map[0], ("exp_avg", "exp_avg_var")),
            "adam": ada_map,
            "adamax": (("_m", "_u"), ("exp_avg", "exp_inf")),
            "adamw": ada_map,
            "lion": (("_momentums", ), ("exp_avg", )),
            "nadam": (ada_map[0] + ("_u_product", ), ada_map[1] + ("mu_product", )),
            "rmsprop": (("_velocities", ), ("square_avg",))
            }

    def _get_optimizer_and_group_ids(self) -> tuple[K_Optimizer,
                                                    dict[int,
                                                         T.Literal["decay", "no_decay"]]] | None:
        """Obtain the optimizer from the saved .keras model

        Returns
        -------
        optimizer
            The saved keras optimizer if it exists or ``None`` if it does not
        group_ids
            dictionary of keras model's trainable weight index to the name of the parameter group
        """
        compile_conf = self._config.get("compile_config", {}).get("optimizer")
        if not compile_conf:
            logger.debug("[OptimizerMigrate] No saved keras optimizer in model file")
            return None
        tmp_model = T.cast(k_models.Model, k_models.load_model(self._model_path, compile=True))
        opt = T.cast("K_Optimizer", tmp_model.optimizer)
        group_ids = get_parameter_group_ids(tmp_model.trainable_variables)
        del tmp_model
        gc.collect()
        logger.debug("[OptimizerMigrate] keras optimizer from model file: %s", opt)
        return opt, group_ids

    def _build_optimizer_state(self,
                               optimizer: K_Optimizer,
                               decay_indices: list[int],
                               no_decay_indices: list[int]) -> dict[int, dict[str, torch.Tensor]]:
        """Build the "state" item for the optimizer state_dict

        Parameters
        ----------
        optimizer
            The loaded keras optimizer
        decay_indices
            The list of keras variable indices that belong to the decay parameter group
        no_decay_indices
            The list of keras variable indices that belong to the no_decay parameter group

        Returns
        -------
        The populated, ordered, state item in torch format from the keras optimizer
        """
        mapping = self._mapping[optimizer.name]
        logger.debug("[OptimizerMigrate] mapping for '%s': %s -> %s",
                     optimizer.name, mapping[0], mapping[1])
        if not all(hasattr(optimizer, x) for x in mapping[0]):
            raise RuntimeError(
                f"Cannot extract {mapping[0]} from keras optimizer. Keras version may have "
                "changed internal structure.")

        if optimizer.name == "lion":
            step = {}
        else:
            step = {"step": torch.from_numpy(
                T.cast(np.ndarray, optimizer.iterations.numpy()).astype(np.float32))}
        ordered = decay_indices + no_decay_indices

        # pylint:disable=protected-access
        vars_ = {mapping[1][idx]: getattr(optimizer, x)._value.data
                 for idx, x in enumerate(mapping[0])
                 if isinstance(getattr(optimizer, x), Variable)}
        weights = {x: getattr(optimizer, mapping[0][idx])
                   for idx, x in enumerate(mapping[1])
                   if x not in vars_}

        retval: dict[int, dict[str, torch.Tensor]] = {}
        for dst_idx, src_idx in enumerate(ordered):
            layer = {k: v[src_idx] for k, v in weights.items()}
            if not all(hasattr(v, "_value") for v in layer.values()):
                logger.debug("[OptimizerMigrate] Skipping variable without torch param: %s",
                             list(layer.values())[0].name.rsplit("_", maxsplit=1)[0])
                continue
            c_step = {k: v.clone() for k, v in step.items()}
            c_vars = {k: v.clone() for k, v in vars_.items()}
            retval[dst_idx] = c_step | c_vars | {k: v._value.data for k, v in layer.items()}

        return retval

    @classmethod
    def _get_parameter_groups(cls,
                              optimizer: K_Optimizer,
                              weight_indices: list[int],
                              bias_indices: list[int]) -> list[dict[str, T.Any]]:
        """Obtain the fixed config optimizer value and param ids for each parameter group

        Parameters
        ----------
        optimizer
            The loaded keras optimizer
        weight_indices
            The list of keras variable indices that belong to the weight parameter group
        bias_indices
            The list of keras variable indices that belong to the bias parameter group

        Returns
        -------
        The parameter group fixed config items and parameter ids
        """
        fixed = {}
        if hasattr(optimizer, "beta_1") and hasattr(optimizer, "beta_2"):
            fixed["betas"] = (optimizer.beta_1, optimizer.beta_2)
        if hasattr(optimizer, "amsgrad"):
            fixed["amsgrad"] = optimizer.amsgrad

        g1_len = len(weight_indices)
        params = [{"params": list(range(g1_len))},
                  {"params": list(range(g1_len, g1_len + len(bias_indices)))}]

        retval = [fixed | params[0], fixed | params[1]]
        logger.debug("[OptimizerMigrate] param_groups: %s", retval)
        return retval

    @classmethod
    def _get_scaler_state(cls,
                          optimizer: LossScaleOptimizer | None) -> dict[str, float | int] | None:
        """Build the scaler state_dict from Keras' LossScaleOptimizer

        Parameters
        ----------
        optimizer
            The Keras LossScaleOptimizer or ``None`` if the optimizer is not scaled

        Returns
        -------
        The state dict for Torch scaler or ``None`` if the optimizer is not scaled
        """
        if optimizer is None:
            logger.debug("[OptimizerMigrate] No scaler to migrate")
            return None

        if (not hasattr(optimizer, "dynamic_growth_steps")
                or not hasattr(optimizer, "dynamic_scale")
                or not hasattr(optimizer, "step_counter")):
            logger.warning("Unable to migrate Loss Scaler parameters. Scaler will be reset")
            return None

        retval = {"scale": float(optimizer.dynamic_scale.numpy()),
                  "growth_factor": 2.0,
                  "backoff_factor": 0.5,
                  "growth_interval": optimizer.dynamic_growth_steps,
                  "_growth_tracker": int(optimizer.step_counter.numpy())}
        logger.debug("[OptimizerMigrate] scaler: %s", retval)
        return retval

    def convert(self) -> dict[str, T.Any] | None:
        """Convert the keras optimizer from a keras model file into a torch optimizer state dict

        Returns
        -------
        The optimizer state dict for loading into a torch optimizer or ``None`` if no saved
        optimizer exists
        """
        optimizer_group_ids = self._get_optimizer_and_group_ids()
        if optimizer_group_ids is None:
            return None
        optimizer, index_map = optimizer_group_ids

        scaler_opt: LossScaleOptimizer | None = None
        if hasattr(optimizer, "inner_optimizer"):
            logger.debug("[OptimizerMigrate] Extracting inner optimizer %s from %s",
                         optimizer.inner_optimizer, optimizer)
            scaler_opt = optimizer
            optimizer = optimizer.inner_optimizer

        logger.info("Migrating optimizer weights to Torch")

        weight_indices = [k for k, v in index_map.items() if v == "decay"]
        bias_indices = [k for k, v in index_map.items() if v == "no_decay"]

        opt_state = self._build_optimizer_state(optimizer, weight_indices, bias_indices)
        if not opt_state:
            logger.warning("Unable to migrate optimizer weights. Optimizer will be reset")
            return None

        param_groups = self._get_parameter_groups(optimizer, weight_indices, bias_indices)
        scaler_state = self._get_scaler_state(scaler_opt)

        retval = {"version": 0.5,
                  "optimizer": {"state": opt_state, "param_groups": param_groups},
                  "scaler": scaler_state}
        return retval


__all__ = get_module_objects(__name__)
