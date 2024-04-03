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
import logging
import os
import sys
import typing as T

import tensorflow as tf

from lib.model.backup_restore import Backup
from lib.utils import FaceswapError

if T.TYPE_CHECKING:
    from .model import ModelBase

kmodels = tf.keras.models
logger = logging.getLogger(__name__)


def get_all_sub_models(
        model: tf.keras.models.Model,
        models: list[tf.keras.models.Model] | None = None) -> list[tf.keras.models.Model]:
    """ For a given model, return all sub-models that occur (recursively) as children.

    Parameters
    ----------
    model: :class:`tensorflow.keras.models.Model`
        A Keras model to scan for sub models
    models: `None`
        Do not provide this parameter. It is used for recursion

    Returns
    -------
    list
        A list of all :class:`tensorflow.keras.models.Model` objects found within the given model.
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
        self._plugin = plugin
        self._is_predict = is_predict
        self._model_dir = model_dir
        self._save_optimizer = save_optimizer
        self._history: list[list[float]] = [[], []]  # Loss histories per save iteration
        self._backup = Backup(self._model_dir, self._plugin.name)

    @property
    def model_dir(self) -> str:
        """ str: The full path to the model folder """
        return self._model_dir

    @property
    def filename(self) -> str:
        """str: The filename for this model."""
        return os.path.join(self._model_dir, f"{self._plugin.name}.h5")

    @property
    def model_exists(self) -> bool:
        """ bool: ``True`` if a model of the type being loaded exists within the model folder
        location otherwise ``False``.
        """
        return os.path.isfile(self.filename)

    @property
    def history(self) -> list[list[float]]:
        """ list: list of loss histories per side for the current save iteration. """
        return self._history

    @property
    def multiple_models_in_folder(self) -> list[str] | None:
        """ :list: or ``None`` If there are multiple model types in the requested folder, or model
        types that don't correspond to the requested plugin type, then returns the list of plugin
        names that exist in the folder, otherwise returns ``None`` """
        plugins = [fname.replace(".h5", "")
                   for fname in os.listdir(self._model_dir)
                   if fname.endswith(".h5")]
        test_names = plugins + [self._plugin.name]
        test = False if not test_names else os.path.commonprefix(test_names) == ""
        retval = None if not test else plugins
        logger.debug("plugin name: %s, plugins: %s, test result: %s, retval: %s",
                     self._plugin.name, plugins, test, retval)
        return retval

    def load(self) -> tf.keras.models.Model:
        """ Loads the model from disk

        If the predict function is to be called and the model cannot be found in the model folder
        then an error is logged and the process exits.

        When loading the model, the plugin model folder is scanned for custom layers which are
        added to Keras' custom objects.

        Returns
        -------
        :class:`tensorflow.keras.models.Model`
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

        Notes
        -----
        The backup function actually backups the model from the previous save iteration rather than
        the current save iteration. This is not a bug, but protection against long save times, as
        models can get quite large, so renaming the current model file rather than copying it can
        save substantial amount of time.
        """
        logger.debug("Backing up and saving models")
        print("")  # Insert a new line to avoid spamming the same row as loss output
        save_averages = self._get_save_averages()
        if save_averages and self._should_backup(save_averages):
            self._backup.backup_model(self.filename)
            self._backup.backup_model(self._plugin.state.filename)

        include_optimizer = (force_save_optimizer or
                             self._save_optimizer == "always" or
                             (self._save_optimizer == "exit" and is_exit))

        try:
            self._plugin.model.save(self.filename, include_optimizer=include_optimizer)
        except ValueError as err:
            if include_optimizer and "name already exists" in str(err):
                logger.warning("Due to a bug in older versions of Tensorflow, optimizer state "
                               "cannot be saved for this model.")
                self._plugin.model.save(self.filename, include_optimizer=False)
            else:
                raise

        self._plugin.state.save()

        msg = "[Saved optimizer state for Snapshot]" if force_save_optimizer else "[Saved model]"
        if save_averages:
            lossmsg = [f"face_{side}: {avg:.5f}"
                       for side, avg in zip(("a", "b"), save_averages)]
            msg += f" - Average loss since last save: {', '.join(lossmsg)}"
        logger.info(msg)

    def _get_save_averages(self) -> list[float]:
        """ Return the average loss since the last save iteration and reset historical loss """
        logger.debug("Getting save averages")
        if not all(loss for loss in self._history):
            logger.debug("No loss in history")
            retval = []
        else:
            retval = [sum(loss) / len(loss) for loss in self._history]
            self._history = [[], []]  # Reset historical loss
        logger.debug("Average losses since last save: %s", retval)
        return retval

    def _should_backup(self, save_averages: list[float]) -> bool:
        """ Check whether the loss averages for this save iteration is the lowest that has been
        seen.

        This protects against model corruption by only backing up the model if both sides have
        seen a total fall in loss.

        Notes
        -----
        This is by no means a perfect system. If the model corrupts at an iteration close
        to a save iteration, then the averages may still be pushed lower than a previous
        save average, resulting in backing up a corrupted model.

        Parameters
        ----------
        save_averages: list
            The average loss for each side for this save iteration
        """
        backup = True
        for side, loss in zip(("a", "b"), save_averages):
            if not self._plugin.state.lowest_avg_loss.get(side, None):
                logger.debug("Set initial save iteration loss average for '%s': %s", side, loss)
                self._plugin.state.lowest_avg_loss[side] = loss
                continue
            backup = loss < self._plugin.state.lowest_avg_loss[side] if backup else backup

        if backup:  # Update lowest loss values to the state file
            # pylint:disable=unnecessary-comprehension
            old_avgs = {key: val for key, val in self._plugin.state.lowest_avg_loss.items()}
            self._plugin.state.lowest_avg_loss["a"] = save_averages[0]
            self._plugin.state.lowest_avg_loss["b"] = save_averages[1]
            logger.debug("Updated lowest historical save iteration averages from: %s to: %s",
                         old_avgs, self._plugin.state.lowest_avg_loss)

        logger.debug("Should backup: %s", backup)
        return backup

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
        """ Validate that we have a valid path to a .h5 file.

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
        elif not os.path.splitext(weights_file)[-1].lower() == ".h5":
            msg = (f"Load weights selected, but the path '{weights_file}' is not a valid Keras "
                   f"model (.h5) file.")

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

    def _get_weights_model(self) -> list[tf.keras.models.Model]:
        """ Obtain a list of all sub-models contained within the weights model.

        Returns
        -------
        list
            List of all models contained within the .h5 file

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
                            layer: tf.keras.layers.Layer,
                            sub_weights: tf.keras.layers.Layer,
                            model_name: str) -> T.Literal[-1, 0, 1]:
        """ Load the weights for a single layer.

        Parameters
        ----------
        layer: :class:`tensorflow.keras.layers.Layer`
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
