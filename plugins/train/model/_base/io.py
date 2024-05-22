#!/usr/bin/env python3
"""
IO handling for the model base plugin.

The objects in this module should not be called directly, but are called from
:class:`~plugins.train.model._base.ModelBase`

This module handles:
    - The loading, saving and backing up of keras models to and from disk.
    - The loading and freezing of weights for model plugins.
"""
from __future__ import annotations
import json
import logging
import os
import sys
import typing as T
from shutil import copyfile, copytree

import h5py
import numpy as np
from keras import layers, models as kmodels

from lib.logger import parse_class_init
from lib.model.backup_restore import Backup
from lib.model.layers import ScalarOp
from lib.model.networks import TypeModelsViT, ViT
from lib.utils import FaceswapError

if T.TYPE_CHECKING:
    from .model import ModelBase
    from keras import Optimizer

logger = logging.getLogger(__name__)


def get_all_sub_models(
        model: kmodels.Model,
        models: list[kmodels.Model] | None = None) -> list[kmodels.Model]:
    """ For a given model, return all sub-models that occur (recursively) as children.

    Parameters
    ----------
    model: :class:`keras.models.Model`
        A Keras model to scan for sub models
    models: `None`
        Do not provide this parameter. It is used for recursion

    Returns
    -------
    list
        A list of all :class:`keras.models.Model` objects found within the given model.
        The provided model will always be returned in the first position
    """
    if models is None:
        models = [model]
    else:
        models.append(model)
    for layer in model.layers:
        if isinstance(layer, kmodels.Model):
            get_all_sub_models(layer, models=models)
    return models


class IO():
    """ Model saving and loading functions.

    Handles the loading and saving of the plugin model from disk as well as the model backup and
    snapshot functions.

    Parameters
    ----------
    plugin: :class:`Model`
        The parent plugin class that owns the IO functions.
    model_dir: str
        The full path to the model save location
    is_predict: bool
        ``True`` if the model is being loaded for inference. ``False`` if the model is being loaded
        for training.
    save_optimizer: ["never", "always", "exit"]
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
        self._save_optimizer = save_optimizer
        self._history: list[float] = []
        """list[float]: Loss history for current save iteration """
        self._backup = Backup(self._model_dir, self._plugin.name)
        self._update_legacy()
        logger.debug("Initialized %s", self.__class__.__name__)

    @property
    def model_dir(self) -> str:
        """ str: The full path to the model folder """
        return self._model_dir

    @property
    def filename(self) -> str:
        """str: The filename for this model."""
        return os.path.join(self._model_dir, f"{self._plugin.name}.keras")

    @property
    def model_exists(self) -> bool:
        """ bool: ``True`` if a model of the type being loaded exists within the model folder
        location otherwise ``False``.
        """
        return os.path.isfile(self.filename)

    @property
    def history(self) -> list[float]:
        """ list[float]: list of loss history for the current save iteration. """
        return self._history

    @property
    def multiple_models_in_folder(self) -> list[str] | None:
        """ :list: or ``None`` If there are multiple model types in the requested folder, or model
        types that don't correspond to the requested plugin type, then returns the list of plugin
        names that exist in the folder, otherwise returns ``None`` """
        plugins = [fname.replace(".keras", "")
                   for fname in os.listdir(self._model_dir)
                   if fname.endswith(".keras")]
        test_names = plugins + [self._plugin.name]
        test = False if not test_names else os.path.commonprefix(test_names) == ""
        retval = None if not test else plugins
        logger.debug("plugin name: %s, plugins: %s, test result: %s, retval: %s",
                     self._plugin.name, plugins, test, retval)
        return retval

    def _update_legacy(self) -> None:
        """ Look for faceswap 2.x .h5 files in the model folder. If exists, then update to Faceswap
        3 .keras file and backup the original model .h5 file

        Note: Currently disabled as keras hangs trying to load old faceswap models
        """
        if self.model_exists:
            logger.debug("Existing model file is current: '%s'", os.path.basename(self.filename))
            return

        old_fname = f"{os.path.splitext(self.filename)[0]}.h5"
        if not os.path.isfile(old_fname):
            logger.debug("No legacy model file to update")
            return

        Legacy(old_fname)

    def load(self) -> kmodels.Model:
        """ Loads the model from disk

        If the predict function is to be called and the model cannot be found in the model folder
        then an error is logged and the process exits.

        When loading the model, the plugin model folder is scanned for custom layers which are
        added to Keras' custom objects.

        Returns
        -------
        :class:`keras.models.Model`
            The saved model loaded from disk
        """
        logger.debug("Loading model: %s", self.filename)
        if self._is_predict and not self.model_exists:
            logger.error("Model could not be found in folder '%s'. Exiting", self._model_dir)
            sys.exit(1)

        try:
            model = kmodels.load_model(self.filename, compile=False)
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
            raise err

        logger.info("Loaded model from disk: '%s'", self.filename)
        return model

    def _remove_optimizer(self) -> Optimizer:
        """ Keras 3 `.keras` format ignores the `save_optimizer` kwarg. To hack around this we
        remove the optimizer from the model prior to saving and then re-attach it to the model

        Returns
        -------
        :class:`keras.optimizers.Optimizer` | None
            The optimizer for the model, if it should not be saved. ``None`` if it should be saved
        """
        retval = self._plugin.model.optimizer
        del self._plugin.model.optimizer
        logger.debug("Removed optimizer for saving: %s", retval)
        return retval

    def _save_model(self, is_exit: bool, force_save_optimizer: bool) -> None:
        """ Save the model either with or without the optimizer weights

        Keras 3 ignores 'save_optimizer` so if it should not be saved, we remove it from
        the model for saving, then re-attach it

        Parameters
        ----------
        is_exit: bool
            ``True`` if the save request has come from an exit process request otherwise ``False``.
        force_save_optimizer: bool
            ``True`` to force saving the optimizer weights with the model, otherwise ``False``.
        """
        include_optimizer = (force_save_optimizer or
                             self._save_optimizer == "always" or
                             (self._save_optimizer == "exit" and is_exit))

        if not include_optimizer:
            optimizer = self._remove_optimizer()

        self._plugin.model.save(self.filename)
        self._plugin.state.save()

        if not include_optimizer:
            logger.debug("Re-attaching optimizer: %s", optimizer)
            setattr(self._plugin.model, "optimizer", optimizer)

    def _get_save_average(self) -> float:
        """ Return the average loss since the last save iteration and reset historical loss

        Returns
        -------
        float
            The average loss since the last save iteration
        """
        logger.debug("Getting save averages")
        if not self._history:
            logger.debug("No loss in history")
            retval = 0.0
        else:
            retval = sum(self._history) / len(self._history)
            self._history = []  # Reset historical loss
        logger.debug("Average loss since last save: %s", round(retval, 5))
        return retval

    def _should_backup(self, save_average: float) -> bool:
        """ Check whether the loss average for this save iteration is the lowest that has been
        seen.

        This protects against model corruption by only backing up the model if the sum of all loss
        functions has fallen.

        Notes
        -----
        This is by no means a perfect system. If the model corrupts at an iteration close
        to a save iteration, then the averages may still be pushed lower than a previous
        save average, resulting in backing up a corrupted model. Changing loss weighting can also
        arteficially impact this

        Parameters
        ----------
        save_average: float
            The average loss since the last save iteration
        """
        if not self._plugin.state.lowest_avg_loss:
            logger.debug("Set initial save iteration loss average: %s", save_average)
            self._plugin.state.lowest_avg_loss = save_average
            return False

        old_average = self._plugin.state.lowest_avg_loss
        backup = save_average < old_average

        if backup:  # Update lowest loss values to the state file
            self._plugin.state.lowest_avg_loss = save_average
            logger.debug("Updated lowest historical save iteration average from: %s to: %s",
                         old_average, save_average)

        logger.debug("Should backup: %s", backup)
        return backup

    def _maybe_backup(self) -> tuple[float, bool]:
        """ Backup the model if total average loss has dropped for the save iteration

        Returns
        -------
        float
            The total loss average since the last save iteration
        bool
            ``True`` if the model was backed up
        """
        save_average = self._get_save_average()
        should_backup = self._should_backup(save_average)
        if not save_average or not should_backup:
            logger.debug("Not backing up model (save_average: %s, should_backup: %s)",
                         save_average, should_backup)
            return save_average, False

        logger.debug("Backing up model")
        self._backup.backup_model(self.filename)
        self._backup.backup_model(self._plugin.state.filename)
        return save_average, True

    def save(self,
             is_exit: bool = False,
             force_save_optimizer: bool = False) -> None:
        """ Backup and save the model and state file.

        Parameters
        ----------
        is_exit: bool, optional
            ``True`` if the save request has come from an exit process request otherwise ``False``.
            Default: ``False``
        force_save_optimizer: bool, optional
            ``True`` to force saving the optimizer weights with the model, otherwise ``False``.
            Default:``False``
        """
        logger.debug("Backing up and saving models")
        print("\x1b[2K", end="\r")  # Clear last line
        logger.info("Saving Model...")

        self._save_model(is_exit, force_save_optimizer)
        save_average, backed_up = self._maybe_backup()

        msg = "[Saved optimizer state for Snapshot]" if force_save_optimizer else "[Saved model]"
        if save_average:
            msg += f" - Average total loss since last save: {save_average:.5f}"
        if backed_up:
            msg += " [Model backed up]"
        logger.info(msg)

    def snapshot(self) -> None:
        """ Perform a model snapshot.

        Notes
        -----
        Snapshot function is called 1 iteration after the model was saved, so that it is built from
        the latest save, hence iteration being reduced by 1.
        """
        logger.debug("Performing snapshot. Iterations: %s", self._plugin.iterations)
        self._backup.snapshot_models(self._plugin.iterations - 1)
        logger.debug("Performed snapshot")


class Weights():
    """ Handling of freezing and loading model weights

    Parameters
    ----------
    plugin: :class:`Model`
        The parent plugin class that owns the IO functions.
    """
    def __init__(self, plugin: ModelBase) -> None:
        logger.debug("Initializing %s: (plugin: %s)", self.__class__.__name__, plugin)
        self._model = plugin.model
        self._name = plugin.model_name
        self._do_freeze = plugin._args.freeze_weights
        self._weights_file = self._check_weights_file(plugin._args.load_weights)

        freeze_layers = plugin.config.get("freeze_layers")  # Standardized config for freezing
        load_layers = plugin.config.get("load_layers")  # Standardized config for loading
        self._freeze_layers = freeze_layers if freeze_layers else ["encoder"]  # No plugin config
        self._load_layers = load_layers if load_layers else ["encoder"]  # No plugin config
        logger.debug("Initialized %s", self.__class__.__name__)

    @classmethod
    def _check_weights_file(cls, weights_file: str) -> str | None:
        """ Validate that we have a valid path to a .keras file.

        Parameters
        ----------
        weights_file: str
            The full path to a weights file

        Returns
        -------
        str
            The full path to a weights file
        """
        if not weights_file:
            logger.debug("No weights file selected.")
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
        """ If freeze has been selected in the cli arguments, then freeze those models indicated
        in the plugin's configuration. """
        # Blanket unfreeze layers, as checking the value of :attr:`layer.trainable` appears to
        # return ``True`` even when the weights have been frozen
        for layer in get_all_sub_models(self._model):
            layer.trainable = True

        if not self._do_freeze:
            logger.debug("Freeze weights deselected. Not freezing")
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
        """ Load weights for newly created models, or output warning for pre-existing models.

        Parameters
        ----------
        model_exists: bool
            ``True`` if a model pre-exists and is being resumed, ``False`` if this is a new model
        """
        if not self._weights_file:
            logger.debug("No weights file provided. Not loading weights.")
            return
        if model_exists and self._weights_file:
            logger.warning("Ignoring weights file '%s' as this model is resuming.",
                           self._weights_file)
            return

        weights_models = self._get_weights_model()
        all_models = get_all_sub_models(self._model)

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
            raise FaceswapError(f"No weights were succesfully loaded from your weights file: "
                                f"'{self._weights_file}'. Please check and try again.")
        if skipped_ops > 0:
            logger.warning("%s weight(s) were unable to be loaded for your model. This is most "
                           "likely because the weights you are trying to load were trained with "
                           "different settings than you have set for your current model.",
                           skipped_ops)

    def _get_weights_model(self) -> list[kmodels.Model]:
        """ Obtain a list of all sub-models contained within the weights model.

        Returns
        -------
        list
            List of all models contained within the .keras file

        Raises
        ------
        FaceswapError
            In the event of a failure to load the weights, or the weights belonging to a different
            model
        """
        retval = get_all_sub_models(kmodels.load_model(self._weights_file, compile=False))
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
        """ Load the weights for a single layer.

        Parameters
        ----------
        layer: :class:`keras.layers.Layer`
            The layer to set the weights for
        sub_weights: list
            The list of layers in the weights model to load weights from
        model_name: str
            The name of the current sub-model that is having it's weights loaded

        Returns
        -------
        int
            `-1` if the layer has no weights to load. `0` if weights loading was unsuccessful. `1`
            if weights loading was successful
        """
        old_weights = layer.get_weights()
        if not old_weights:
            logger.debug("Skipping layer without weights: %s", layer.name)
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


class Legacy:  # pylint:disable=too-few-public-methods
    """ Handles the updating of Keras 2.x models to Keras 3.x

    Generally Keras 2.x models will open in Keras 3.x. There are a couple of bugs in Keras 3
    legacy loading code which impacts Faceswap models:
    - When a model receives a shared functional model as an inbound node, the node index needs
    reducing by 1 (non-trivial to fix upstream)
    - Keras 3 does not accept nested outputs, so Keras 2 FS models need to have the outputs
    flattened

    Parameters
    ----------
    model_path: str
        Full path to the legacy Keras 2.x model h5 file to upgrade
    """
    def __init__(self, model_path: str):
        parse_class_init(locals())
        self._old_model_file = model_path
        """str: Full path to the old .h5 model file"""
        self._new_model_file = f"{os.path.splitext(model_path)[0]}.keras"
        """str: Full path to the new .keras model file"""
        self._functionals: set[str] = set()
        """set[str]: The name of any Functional models discovered in the keras 2 model config"""

        self._upgrade_model()
        logger.debug("Initialized %s", self.__class__.__name__)

    def _get_model_config(self) -> dict[str, T.Any]:
        """ Obtain a keras 2.x config from a keras 2.x .h5 file.

        As keras 3.x will error out loading the file, we collect it directly from the .h5 file

        Returns
        -------
        dict[str, Any]
            A keras 2.x model configuration dictionary

        Raises
        ------
        FaceswapError
            If the file is not a valid Faceswap 2 .h5 model file
        """
        h5file = h5py.File(self._old_model_file, "r")
        s_version = T.cast(str | None, h5file.attrs.get("keras_version"))
        s_config = T.cast(str | None, h5file.attrs.get("model_config"))
        if not s_version or not s_config:
            raise FaceswapError(f"'{self._old_model_file}' is not a valid Faceswap 2 model file")

        version = s_version.split(".")[:2]
        if len(version) != 2 or version[0] != "2":
            raise FaceswapError(f"'{self._old_model_file}' is not a valid Faceswap 2 model file")

        retval = json.loads(s_config)
        logger.debug("Loaded keras 2.x model config: %s", retval)
        return retval

    @classmethod
    def _unwrap_outputs(cls, outputs: list[list[T.Any]]) -> list[list[str | int]]:
        """ Unwrap nested output tensors from a config dict to be a single list of output tensor

        Parameters
        ----------
        outputs: list[list[Any]]
            The outputs that exist within the Keras 2 config dict that may be nested

        Returns
        -------
        list[list[str | int]]
            The output configuration formatted to be compatible with Keras 3
        """
        retval = np.array(outputs).reshape(-1, 3).tolist()
        for item in retval:
            item[1] = int(item[1])
            item[2] = int(item[2])
        logger.debug("Unwrapped outputs: %s to: %s", outputs, retval)
        return retval

    def _get_clip_config(self) -> dict[str, T.Any]:
        """ Build a clip model from the configuration information stored in the legacy state file

        Returns
        -------
        dict[str, T.Any]
            The new keras configuration for a Clip model

        Raises
        ------
        FaceswapError
            If the clip model cannot be built
        """
        state_file = f"{os.path.splitext(self._old_model_file)[0]}_state.json"
        if not os.path.isfile(state_file):
            raise FaceswapError(
                f"The state file '{state_file}' does not exist. This model cannot be ported")

        with open(state_file, "r", encoding="utf-8") as ifile:
            config = json.load(ifile)

        logger.debug("Loaded legacy config '%s': %s", state_file, config)
        net_name = config.get("config", {}).get("enc_architecture", "")
        scaling = config.get("config", {}).get("enc_scaling", 0) / 100

        # Import here to prevent circular imports
        from plugins.train.model.phaze_a import _MODEL_MAPPING  # pylint:disable=C0415
        vit_info = _MODEL_MAPPING.get(net_name)

        if not scaling or not vit_info:
            raise FaceswapError(
                f"Clip network could not be found in '{state_file}'. Discovered network is "
                f"'{net_name}' with encoder scaling: {scaling}. This model cannot be ported")

        input_size = int(max(vit_info.min_size, ((vit_info.default_size * scaling) // 16) * 16))
        vit_model = ViT(T.cast(TypeModelsViT, vit_info.keras_name), input_size=input_size)()

        retval = vit_model.get_config()
        del vit_model
        logger.debug("Got new config for '%s' at input size: %s: %s", net_name, input_size, retval)
        return retval

    def _convert_lambda_config(self, layer: dict[str, T.Any]):
        """ Keras 2 TFLambdaOps are not compatible with Keras 3. Scalar operations can be
        relatively easily substituted with a :class:`~lib.model.layers.ScalarOp` layer

        Parameters
        ----------
        layer: dict[str, Any]
            An existing Keras 2 TFLambdaOp layer

        Raises
        ------
        FaceswapError
            If the TFLambdaOp is not currently supported
        """
        name = layer["config"]["name"]
        operation = name.rsplit(".", maxsplit=1)[-1]
        if operation not in ("multiply", "truediv", "add", "subtract"):
            raise FaceswapError(f"The TFLambdaOp '{name}' is not supported")
        value = layer["inbound_nodes"][0][-1]["y"]
        new_layer = ScalarOp(operation, value, name=name, dtype=layer["config"]["dtype"])

        logger.debug("Converting legacy TFLambdaOp: %s", layer)

        layer["class_name"] = "ScalarOp"
        layer["config"] = new_layer.get_config()
        for n in layer["inbound_nodes"]:
            n[-1] = {}
        layer["inbound_nodes"] = [layer["inbound_nodes"]]
        logger.debug("Converted legacy TFLambdaOp to %s", layer)

    def _process_deprecations(self, layer: dict[str, T.Any]) -> None:
        """ Some layer kwargs are deprecated between Keras 2 and Keras 3. Some are not mission
        critical, but updating these here prevents Keras from outputting warnings about deprecated
        arguments. Others will fail to load the legacy model (eg Clip) so are replaced with a new
        config. Operation is performed in place

        Parameters
        ----------
        layer: dict[str, T.Any]
            A keras model config item representing a keras layer
        """
        if layer["class_name"] == "LeakyReLU":
            # Non mission-critical, but prevents scary deprecation messages
            config = layer["config"]
            old, new = "alpha", "negative_slope"
            if old in config:
                logger.debug("Updating '%s' kwarg '%s' to '%s'", layer["name"], old, new)
                config[new] = config[old]
                del config[old]

        if layer["name"] == "visual":
            # MultiHeadAttention is not backwards compatible, so get new config for Clip models
            logger.debug("Getting new config for 'visual' model")
            layer["config"] = self._get_clip_config()

        if layer["class_name"] == "TFOpLambda":
            # TFLambdaOp are not supported
            self._convert_lambda_config(layer)

        if layer["class_name"] == "DepthwiseConv2D" and "groups" in layer["config"]:
            # groups parameter doesn't exist in Keras 3. Hopefully it still works the same
            logger.debug("Removing groups from DepthwiseConv2D '%s'", layer["name"])
            del layer["config"]["groups"]

    def _process_inbounds(self,
                          layer_name: str,
                          inbound_nodes: list[list[list[str | int]]] | list[list[str | int]]
                          ) -> None:
        """ If the inbound nodes are from a shared functional model, decrement the node index by
        one. Operation is performed in place

        Parameters
        ----------
        layer_name: str
            The name of the layer (for logging)
        inbound_nodes: list[list[list[str | int]]] | list[list[str | int]]
            The inbound nodes from a Keras 2 config dict to process
        """
        to_process = T.cast(
            list[list[list[str | int]]],
            inbound_nodes if isinstance(inbound_nodes[0][0], list) else [inbound_nodes])

        for inbound in to_process:
            for node in inbound:
                name, node_index = node[0], node[1]
                assert isinstance(name, str) and isinstance(node_index, int)
                if name in self._functionals and node_index > 0:
                    logger.debug("Updating '%s' inbound node index for '%s' from %s to %s",
                                 layer_name, name, node_index, node_index - 1)
                    node[1] = node_index - 1

    def _update_layers(self, layer_list: list[dict[str, T.Any]]) -> None:
        """ Given a list of keras layers from a keras 2 config dict, increment the indices for
        any inbound nodes that come from a shared Functional model. Flatten any nested output
        tensor lists. Operations are performed in place

        Parameters
        ----------
        layers: list[dict[str, Any]]
            A list of layers that belong to a keras 2 functional model config dictionary
        """
        for layer in layer_list:
            if layer["class_name"] == "Functional":
                logger.debug("Found Functional layer. Keys: %s", list(layer))

                if layer.get("name"):
                    logger.debug("Storing layer: '%s'", layer["name"])
                    self._functionals.add(layer["name"])

                layer["config"]["output_layers"] = self._unwrap_outputs(
                    layer["config"]["output_layers"])

                self._update_layers(layer["config"]["layers"])

            if not layer.get("inbound_nodes"):
                continue

            self._process_deprecations(layer)
            self._process_inbounds(layer["name"], layer["inbound_nodes"])

    def _archive_model(self) -> str:
        """ Archive an existing Keras 2 model to a new archive location

        Raises
        ------
        FaceswapError
            If the destination archive folder exists and is not empty

        Returns
        -------
        str
            The path to the archived keras 2 model folder
        """
        model_dir = os.path.dirname(self._old_model_file)
        dst_path = f"{model_dir}_fs2_backup"
        if os.path.exists(dst_path) and os.listdir(dst_path):
            raise FaceswapError(
                f"The destination archive folder '{dst_path}' already exists. Either delete this "
                "folder, select a different model folder, or remove the legacy model files from "
                f"your model folder '{model_dir}'.")

        if os.path.exists(dst_path):
            logger.info("Removing pre-existing empty folder '%s'", dst_path)
            os.rmdir(dst_path)

        logger.info("Archiving model folder '%s' to '%s'", model_dir, dst_path)
        os.rename(model_dir, dst_path)
        return dst_path

    def _restore_files(self, archive_dir: str) -> None:
        """ Copy the state.json file and the logs folder from the archive folder to the new model
        folder

        Parameters
        ----------
        archive_dir: str
            The full path to the archived Keras 2 model
        """
        model_dir = os.path.dirname(self._new_model_file)
        model_name = os.path.splitext(os.path.basename(self._new_model_file))[0]
        logger.debug("Restoring required '%s 'files from '%s' to '%s'",
                     model_name, archive_dir, model_dir)

        for fname in os.listdir(archive_dir):
            fullpath = os.path.join(archive_dir, fname)
            new_path = os.path.join(model_dir, fname)

            if fname == f"{model_name}_logs" and os.path.isdir(fullpath):
                logger.debug("Restoring '%s' to '%s'", fullpath, new_path)
                copytree(fullpath, new_path)
                continue

            if fname == f"{model_name}_state.json" and os.path.isfile(fullpath):
                logger.debug("Restoring '%s' to '%s'", fullpath, new_path)
                copyfile(fullpath, new_path)
                continue

            logger.debug("Skipping file: '%s'", fname)

    def _upgrade_model(self) -> None:
        """ Get the model configuration of a Faceswap 2 model and upgrade it to Faceswap 3
        compatible """
        logger.info("Upgrading model file from Faceswap 2 to Faceswap 3...")
        config = self._get_model_config()
        self._update_layers([config])

        logger.debug("Migrating data to new model...")
        model = kmodels.Model.from_config(config["config"])
        model.load_weights(self._old_model_file)

        archive_dir = self._archive_model()

        dirname = os.path.dirname(self._new_model_file)
        logger.debug("Saving model '%s'", self._new_model_file)
        os.mkdir(dirname)
        model.save(self._new_model_file)
        logger.debug("Saved model '%s'", self._new_model_file)

        self._restore_files(archive_dir)
        logger.info("Model upgraded: '%s'", dirname)
