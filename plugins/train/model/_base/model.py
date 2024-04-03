#!/usr/bin/env python3
"""
Base class for Models. ALL Models should at least inherit from this class.

See :mod:`~plugins.train.model.original` for an annotated example for how to create model plugins.
"""
from __future__ import annotations
import logging
import os
import sys
import time
import typing as T

from collections import OrderedDict

import numpy as np
import tensorflow as tf

from lib.serializer import get_serializer
from lib.model.nn_blocks import set_config as set_nnblock_config
from lib.utils import FaceswapError
from plugins.train._config import Config

from .io import IO, get_all_sub_models, Weights
from .settings import Loss, Optimizer, Settings

if T.TYPE_CHECKING:
    import argparse
    from lib.config import ConfigValueType

keras = tf.keras
K = tf.keras.backend


logger = logging.getLogger(__name__)
_CONFIG: dict[str, ConfigValueType] = {}


class ModelBase():
    """ Base class that all model plugins should inherit from.

    Parameters
    ----------
    model_dir: str
        The full path to the model save location
    arguments: :class:`argparse.Namespace`
        The arguments that were passed to the train or convert process as generated from
        Faceswap's command line arguments
    predict: bool, optional
        ``True`` if the model is being loaded for inference, ``False`` if the model is being loaded
        for training. Default: ``False``

    Attributes
    ----------
    input_shape: tuple or list
        A `tuple` of `ints` defining the shape of the faces that the model takes as input. This
        should be overridden by model plugins in their :func:`__init__` function. If the input size
        is the same for both sides of the model, then this can be a single 3 dimensional `tuple`.
        If the inputs have different sizes for `"A"` and `"B"` this should be a `list` of 2 3
        dimensional shape `tuples`, 1 for each side respectively.
    trainer: str
        Currently there is only one trainer available (`"original"`), so at present this attribute
        can be ignored. If/when more trainers are added, then this attribute should be overridden
        with the trainer name that a model requires in the model plugin's
        :func:`__init__` function.
    """
    def __init__(self,
                 model_dir: str,
                 arguments: argparse.Namespace,
                 predict: bool = False) -> None:
        logger.debug("Initializing ModelBase (%s): (model_dir: '%s', arguments: %s, predict: %s)",
                     self.__class__.__name__, model_dir, arguments, predict)

        # Input shape must be set within the plugin after initializing
        self.input_shape: tuple[int, ...] = ()
        self.trainer = "original"  # Override for plugin specific trainer
        self.color_order: T.Literal["bgr", "rgb"] = "bgr"  # Override for image color channel order

        self._args = arguments
        self._is_predict = predict
        self._model: tf.keras.models.Model | None = None

        self._configfile = arguments.configfile if hasattr(arguments, "configfile") else None
        self._load_config()

        if self.config["penalized_mask_loss"] and self.config["mask_type"] is None:
            raise FaceswapError("Penalized Mask Loss has been selected but you have not chosen a "
                                "Mask to use. Please select a mask or disable Penalized Mask "
                                "Loss.")

        if self.config["learn_mask"] and self.config["mask_type"] is None:
            raise FaceswapError("'Learn Mask' has been selected but you have not chosen a Mask to "
                                "use. Please select a mask or disable 'Learn Mask'.")

        self._mixed_precision = self.config["mixed_precision"]
        self._io = IO(self, model_dir, self._is_predict, self.config["save_optimizer"])
        self._check_multiple_models()

        self._state = State(model_dir,
                            self.name,
                            self._config_changeable_items,
                            False if self._is_predict else self._args.no_logs)
        self._settings = Settings(self._args,
                                  self._mixed_precision,
                                  self.config["allow_growth"],
                                  self._is_predict)
        self._loss = Loss(self.config, self.color_order)

        logger.debug("Initialized ModelBase (%s)", self.__class__.__name__)

    @property
    def model(self) -> tf.keras.models.Model:
        """:class:`Keras.models.Model`: The compiled model for this plugin. """
        return self._model

    @property
    def command_line_arguments(self) -> argparse.Namespace:
        """ :class:`argparse.Namespace`: The command line arguments passed to the model plugin from
        either the train or convert script """
        return self._args

    @property
    def coverage_ratio(self) -> float:
        """ float: The ratio of the training image to crop out and train on as defined in user
        configuration options.

        NB: The coverage ratio is a raw float, but will be applied to integer pixel images.

        To ensure consistent rounding and guaranteed even image size, the calculation for coverage
        should always be: :math:`(original_size * coverage_ratio // 2) * 2`
        """
        return self.config.get("coverage", 62.5) / 100

    @property
    def io(self) -> IO:  # pylint:disable=invalid-name
        """ :class:`~plugins.train.model.io.IO`: Input/Output operations for the model """
        return self._io

    @property
    def config(self) -> dict:
        """ dict: The configuration dictionary for current plugin, as set by the user's
        configuration settings. """
        global _CONFIG  # pylint:disable=global-statement
        if not _CONFIG:
            model_name = self._config_section
            logger.debug("Loading config for: %s", model_name)
            _CONFIG = Config(model_name, configfile=self._configfile).config_dict
        return _CONFIG

    @property
    def name(self) -> str:
        """ str: The name of this model based on the plugin name. """
        _name = sys.modules[self.__module__].__file__
        assert isinstance(_name, str)
        return os.path.splitext(os.path.basename(_name))[0].lower()

    @property
    def model_name(self) -> str:
        """ str: The name of the keras model. Generally this will be the same as :attr:`name`
        but some plugins will override this when they contain multiple architectures """
        return self.name

    @property
    def input_shapes(self) -> list[tuple[None, int, int, int]]:
        """ list: A flattened list corresponding to all of the inputs to the model. """
        shapes = [T.cast(tuple[None, int, int, int], K.int_shape(inputs))
                  for inputs in self.model.inputs]
        return shapes

    @property
    def output_shapes(self) -> list[tuple[None, int, int, int]]:
        """ list: A flattened list corresponding to all of the outputs of the model. """
        shapes = [T.cast(tuple[None, int, int, int], K.int_shape(output))
                  for output in self.model.outputs]
        return shapes

    @property
    def iterations(self) -> int:
        """ int: The total number of iterations that the model has trained. """
        return self._state.iterations

    # Private properties
    @property
    def _config_section(self) -> str:
        """ str: The section name for the current plugin for loading configuration options from the
        config file. """
        return ".".join(self.__module__.split(".")[-2:])

    @property
    def _config_changeable_items(self) -> dict:
        """ dict: The configuration options that can be updated after the model has already been
            created. """
        return Config(self._config_section, configfile=self._configfile).changeable_items

    @property
    def state(self) -> "State":
        """:class:`State`: The state settings for the current plugin. """
        return self._state

    def _load_config(self) -> None:
        """ Load the global config for reference in :attr:`config` and set the faceswap blocks
        configuration options in `lib.model.nn_blocks` """
        global _CONFIG  # pylint:disable=global-statement
        if not _CONFIG:
            model_name = self._config_section
            logger.debug("Loading config for: %s", model_name)
            _CONFIG = Config(model_name, configfile=self._configfile).config_dict

        nn_block_keys = ['icnr_init', 'conv_aware_init', 'reflect_padding']
        set_nnblock_config({key: _CONFIG.pop(key)
                            for key in nn_block_keys})

    def _check_multiple_models(self) -> None:
        """ Check whether multiple models exist in the model folder, and that no models exist that
        were trained with a different plugin than the requested plugin.

        Raises
        ------
        FaceswapError
            If multiple model files, or models for a different plugin from that requested exists
            within the model folder
        """
        multiple_models = self._io.multiple_models_in_folder
        if multiple_models is None:
            logger.debug("Contents of model folder are valid")
            return

        if len(multiple_models) == 1:
            msg = (f"You have requested to train with the '{self.name}' plugin, but a model file "
                   f"for the '{multiple_models[0]}' plugin already exists in the folder "
                   f"'{self.io.model_dir}'.\nPlease select a different model folder.")
        else:
            ptypes = "', '".join(multiple_models)
            msg = (f"There are multiple plugin types ('{ptypes}') stored in the model folder '"
                   f"{self.io.model_dir}'. This is not supported.\nPlease split the model files "
                   "into their own folders before proceeding")
        raise FaceswapError(msg)

    def build(self) -> None:
        """ Build the model and assign to :attr:`model`.

        Within the defined strategy scope, either builds the model from scratch or loads an
        existing model if one exists.

        If running inference, then the model is built only for the required side to perform the
        swap function, otherwise  the model is then compiled with the optimizer and chosen
        loss function(s).

        Finally, a model summary is outputted to the logger at verbose level.
        """
        self._update_legacy_models()
        is_summary = hasattr(self._args, "summary") and self._args.summary
        with self._settings.strategy_scope():
            if self._io.model_exists:
                model = self.io.load()
                if self._is_predict:
                    inference = _Inference(model, self._args.swap_model)
                    self._model = inference.model
                else:
                    self._model = model
            else:
                self._validate_input_shape()
                inputs = self._get_inputs()
                if not self._settings.use_mixed_precision and not is_summary:
                    # Store layer names which can be switched to mixed precision
                    model, mp_layers = self._settings.get_mixed_precision_layers(self.build_model,
                                                                                 inputs)
                    self._state.add_mixed_precision_layers(mp_layers)
                    self._model = model
                else:
                    self._model = self.build_model(inputs)
            if not is_summary and not self._is_predict:
                self._compile_model()
            self._output_summary()

    def _update_legacy_models(self) -> None:
        """ Load weights from legacy split models into new unified model, archiving old model files
        to a new folder. """
        legacy_mapping = self._legacy_mapping()  # pylint:disable=assignment-from-none
        if legacy_mapping is None:
            return

        if not all(os.path.isfile(os.path.join(self.io.model_dir, fname))
                   for fname in legacy_mapping):
            return
        archive_dir = f"{self.io.model_dir}_TF1_Archived"
        if os.path.exists(archive_dir):
            raise FaceswapError("We need to update your model files for use with Tensorflow 2.x, "
                                "but the archive folder already exists. Please remove the "
                                f"following folder to continue: '{archive_dir}'")

        logger.info("Updating legacy models for Tensorflow 2.x")
        logger.info("Your Tensorflow 1.x models will be archived in the following location: '%s'",
                    archive_dir)
        os.rename(self.io.model_dir, archive_dir)
        os.mkdir(self.io.model_dir)
        new_model = self.build_model(self._get_inputs())
        for model_name, layer_name in legacy_mapping.items():
            old_model: tf.keras.models.Model = keras.models.load_model(
                os.path.join(archive_dir, model_name),
                compile=False)
            layer = [layer for layer in new_model.layers if layer.name == layer_name]
            if not layer:
                logger.warning("Skipping legacy weights from '%s'...", model_name)
                continue
            klayer: tf.keras.layers.Layer = layer[0]
            logger.info("Updating legacy weights from '%s'...", model_name)
            klayer.set_weights(old_model.get_weights())
        filename = self._io.filename
        logger.info("Saving Tensorflow 2.x model to '%s'", filename)
        new_model.save(filename)
        # Penalized Loss and Learn Mask used to be disabled automatically if a mask wasn't
        # selected, so disable it if enabled, but mask_type is None
        if self.config["mask_type"] is None:
            self.config["penalized_mask_loss"] = False
            self.config["learn_mask"] = False
            self.config["eye_multiplier"] = 1
            self.config["mouth_multiplier"] = 1
        self._state.save()

    def _validate_input_shape(self) -> None:
        """ Validate that the input shape is either a single shape tuple of 3 dimensions or
        a list of 2 shape tuples of 3 dimensions. """
        assert len(self.input_shape) == 3, "Input shape should be a 3 dimensional shape tuple"

    def _get_inputs(self) -> list[tf.keras.layers.Input]:
        """ Obtain the standardized inputs for the model.

        The inputs will be returned for the "A" and "B" sides in the shape as defined by
        :attr:`input_shape`.

        Returns
        -------
        list
            A list of :class:`keras.layers.Input` tensors. This will be a list of 2 tensors (one
            for each side) each of shapes :attr:`input_shape`.
        """
        logger.debug("Getting inputs")
        input_shapes = [self.input_shape, self.input_shape]
        inputs = [keras.layers.Input(shape=shape, name=f"face_in_{side}")
                  for side, shape in zip(("a", "b"), input_shapes)]
        logger.debug("inputs: %s", inputs)
        return inputs

    def build_model(self, inputs: list[tf.keras.layers.Input]) -> tf.keras.models.Model:
        """ Override for Model Specific autoencoder builds.

        Parameters
        ----------
        inputs: list
            A list of :class:`keras.layers.Input` tensors. This will be a list of 2 tensors (one
            for each side) each of shapes :attr:`input_shape`.

        Returns
        -------
        :class:`keras.models.Model`
            See Keras documentation for the correct structure, but note that parameter :attr:`name`
            is a required rather than an optional argument in Faceswap. You should assign this to
            the attribute ``self.name`` that is automatically generated from the plugin's filename.
        """
        raise NotImplementedError

    def _output_summary(self) -> None:
        """ Output the summary of the model and all sub-models to the verbose logger. """
        if hasattr(self._args, "summary") and self._args.summary:
            print_fn = None  # Print straight to stdout
        else:
            # print to logger
            print_fn = lambda x: logger.verbose("%s", x)  #type:ignore[attr-defined]  # noqa[E731]  # pylint:disable=C3001
        for idx, model in enumerate(get_all_sub_models(self.model)):
            if idx == 0:
                parent = model
                continue
            model.summary(line_length=100, print_fn=print_fn)
        parent.summary(line_length=100, print_fn=print_fn)

    def _compile_model(self) -> None:
        """ Compile the model to include the Optimizer and Loss Function(s). """
        logger.debug("Compiling Model")

        if self.state.model_needs_rebuild:
            self._model = self._settings.check_model_precision(self._model, self._state)

        optimizer = Optimizer(self.config["optimizer"],
                              self.config["learning_rate"],
                              self.config["autoclip"],
                              10 ** int(self.config["epsilon_exponent"])).optimizer
        if self._settings.use_mixed_precision:
            optimizer = self._settings.loss_scale_optimizer(optimizer)

        weights = Weights(self)
        weights.load(self._io.model_exists)
        weights.freeze()

        self._loss.configure(self.model)
        self.model.compile(optimizer=optimizer, loss=self._loss.functions)
        self._state.add_session_loss_names(self._loss.names)
        logger.debug("Compiled Model: %s", self.model)

    def _legacy_mapping(self) -> dict | None:
        """ The mapping of separate model files to single model layers for transferring of legacy
        weights.

        Returns
        -------
        dict or ``None``
            Dictionary of original H5 filenames for legacy models mapped to new layer names or
            ``None`` if the model did not exist in Faceswap prior to Tensorflow 2
        """
        return None

    def add_history(self, loss: list[float]) -> None:
        """ Add the current iteration's loss history to :attr:`_io.history`.

        Called from the trainer after each iteration, for tracking loss drop over time between
        save iterations.

        Parameters
        ----------
        loss: list
            The loss values for the A and B side for the current iteration. This should be the
            collated loss values for each side.
        """
        self._io.history[0].append(loss[0])
        self._io.history[1].append(loss[1])


class State():
    """ Holds state information relating to the plugin's saved model.

    Parameters
    ----------
    model_dir: str
        The full path to the model save location
    model_name: str
        The name of the model plugin
    config_changeable_items: dict
        Configuration options that can be altered when resuming a model, and their current values
    no_logs: bool
        ``True`` if Tensorboard logs should not be generated, otherwise ``False``
    """
    def __init__(self,
                 model_dir: str,
                 model_name: str,
                 config_changeable_items: dict,
                 no_logs: bool) -> None:
        logger.debug("Initializing %s: (model_dir: '%s', model_name: '%s', "
                     "config_changeable_items: '%s', no_logs: %s", self.__class__.__name__,
                     model_dir, model_name, config_changeable_items, no_logs)
        self._serializer = get_serializer("json")
        filename = f"{model_name}_state.{self._serializer.file_extension}"
        self._filename = os.path.join(model_dir, filename)
        self._name = model_name
        self._iterations = 0
        self._mixed_precision_layers: list[str] = []
        self._rebuild_model = False
        self._sessions: dict[int, dict] = {}
        self._lowest_avg_loss: dict[str, float] = {}
        self._config: dict[str, ConfigValueType] = {}
        self._load(config_changeable_items)
        self._session_id = self._new_session_id()
        self._create_new_session(no_logs, config_changeable_items)
        logger.debug("Initialized %s:", self.__class__.__name__)

    @property
    def filename(self) -> str:
        """ str: Full path to the state filename """
        return self._filename

    @property
    def loss_names(self) -> list[str]:
        """ list: The loss names for the current session """
        return self._sessions[self._session_id]["loss_names"]

    @property
    def current_session(self) -> dict:
        """ dict: The state dictionary for the current :attr:`session_id`. """
        return self._sessions[self._session_id]

    @property
    def iterations(self) -> int:
        """ int: The total number of iterations that the model has trained. """
        return self._iterations

    @property
    def lowest_avg_loss(self) -> dict:
        """dict: The lowest average save interval loss seen for each side. """
        return self._lowest_avg_loss

    @property
    def session_id(self) -> int:
        """ int: The current training session id. """
        return self._session_id

    @property
    def sessions(self) -> dict[int, dict[str, T.Any]]:
        """ dict[int, dict[str, Any]]: The session information for each session in the state
        file """
        return {int(k): v for k, v in self._sessions.items()}

    @property
    def mixed_precision_layers(self) -> list[str]:
        """list: Layers that can be switched between mixed-float16 and float32. """
        return self._mixed_precision_layers

    @property
    def model_needs_rebuild(self) -> bool:
        """bool: ``True`` if mixed precision policy has changed so model needs to be rebuilt
        otherwise ``False`` """
        return self._rebuild_model

    def _new_session_id(self) -> int:
        """ Generate a new session id. Returns 1 if this is a new model, or the last session id + 1
        if it is a pre-existing model.

        Returns
        -------
        int
            The newly generated session id
        """
        if not self._sessions:
            session_id = 1
        else:
            session_id = max(int(key) for key in self._sessions.keys()) + 1
        logger.debug(session_id)
        return session_id

    def _create_new_session(self, no_logs: bool, config_changeable_items: dict) -> None:
        """ Initialize a new session, creating the dictionary entry for the session in
        :attr:`_sessions`.

        Parameters
        ----------
        no_logs: bool
            ``True`` if Tensorboard logs should not be generated, otherwise ``False``
        config_changeable_items: dict
            Configuration options that can be altered when resuming a model, and their current
            values
        """
        logger.debug("Creating new session. id: %s", self._session_id)
        self._sessions[self._session_id] = {"timestamp": time.time(),
                                            "no_logs": no_logs,
                                            "loss_names": [],
                                            "batchsize": 0,
                                            "iterations": 0,
                                            "config": config_changeable_items}

    def update_session_config(self, key: str, value: T.Any) -> None:
        """ Update a configuration item of the currently loaded session.

        Parameters
        ----------
        key: str
            The configuration item to update for the current session
        value: any
            The value to update to
        """
        old_val = self.current_session["config"][key]
        assert isinstance(value, type(old_val))
        logger.debug("Updating configuration item '%s' from '%s' to '%s'", key, old_val, value)
        self.current_session["config"][key] = value

    def add_session_loss_names(self, loss_names: list[str]) -> None:
        """ Add the session loss names to the sessions dictionary.

        The loss names are used for Tensorboard logging

        Parameters
        ----------
        loss_names: list
            The list of loss names for this session.
        """
        logger.debug("Adding session loss_names: %s", loss_names)
        self._sessions[self._session_id]["loss_names"] = loss_names

    def add_session_batchsize(self, batch_size: int) -> None:
        """ Add the session batch size to the sessions dictionary.

        Parameters
        ----------
        batch_size: int
            The batch size for the current training session
        """
        logger.debug("Adding session batch size: %s", batch_size)
        self._sessions[self._session_id]["batchsize"] = batch_size

    def increment_iterations(self) -> None:
        """ Increment :attr:`iterations` and session iterations by 1. """
        self._iterations += 1
        self._sessions[self._session_id]["iterations"] += 1

    def add_mixed_precision_layers(self, layers: list[str]) -> None:
        """ Add the list of model's layers that are compatible for mixed precision to the
        state dictionary """
        logger.debug("Storing mixed precision layers: %s", layers)
        self._mixed_precision_layers = layers

    def _load(self, config_changeable_items: dict) -> None:
        """ Load a state file and set the serialized values to the class instance.

        Updates the model's config with the values stored in the state file.

        Parameters
        ----------
        config_changeable_items: dict
            Configuration options that can be altered when resuming a model, and their current
            values
        """
        logger.debug("Loading State")
        if not os.path.exists(self._filename):
            logger.info("No existing state file found. Generating.")
            return
        state = self._serializer.load(self._filename)
        self._name = state.get("name", self._name)
        self._sessions = state.get("sessions", {})
        self._lowest_avg_loss = state.get("lowest_avg_loss", {})
        self._iterations = state.get("iterations", 0)
        self._mixed_precision_layers = state.get("mixed_precision_layers", [])
        self._config = state.get("config", {})
        logger.debug("Loaded state: %s", state)
        self._replace_config(config_changeable_items)

    def save(self) -> None:
        """ Save the state values to the serialized state file. """
        logger.debug("Saving State")
        state = {"name": self._name,
                 "sessions": self._sessions,
                 "lowest_avg_loss": self._lowest_avg_loss,
                 "iterations": self._iterations,
                 "mixed_precision_layers": self._mixed_precision_layers,
                 "config": _CONFIG}
        self._serializer.save(self._filename, state)
        logger.debug("Saved State")

    def _replace_config(self, config_changeable_items) -> None:
        """ Replace the loaded config with the one contained within the state file.

        Check for any `fixed`=``False`` parameter changes and log info changes.

        Update any legacy config items to their current versions.

        Parameters
        ----------
        config_changeable_items: dict
            Configuration options that can be altered when resuming a model, and their current
            values
        """
        global _CONFIG  # pylint:disable=global-statement
        if _CONFIG is None:
            return
        legacy_update = self._update_legacy_config()
        # Add any new items to state config for legacy purposes where the new default may be
        # detrimental to an existing model.
        legacy_defaults: dict[str, str | int | bool] = {"centering": "legacy",
                                                        "mask_loss_function": "mse",
                                                        "l2_reg_term": 100,
                                                        "optimizer": "adam",
                                                        "mixed_precision": False}
        for key, val in _CONFIG.items():
            if key not in self._config.keys():
                setting: ConfigValueType = legacy_defaults.get(key, val)
                logger.info("Adding new config item to state file: '%s': '%s'", key, setting)
                self._config[key] = setting
        self._update_changed_config_items(config_changeable_items)
        logger.debug("Replacing config. Old config: %s", _CONFIG)
        _CONFIG = self._config
        if legacy_update:
            self.save()
        logger.debug("Replaced config. New config: %s", _CONFIG)
        logger.info("Using configuration saved in state file")

    def _update_legacy_config(self) -> bool:
        """ Legacy updates for new config additions.

        When new config items are added to the Faceswap code, existing model state files need to be
        updated to handle these new items.

        Current existing legacy update items:

            * loss - If old `dssim_loss` is ``true`` set new `loss_function` to `ssim` otherwise
            set it to `mae`. Remove old `dssim_loss` item

            * l2_reg_term - If this exists, set loss_function_2 to ``mse`` and loss_weight_2 to
            the value held in the old ``l2_reg_term`` item

            * masks - If `learn_mask` does not exist then it is set to ``True`` if `mask_type` is
            not ``None`` otherwise it is set to ``False``.

            * masks type - Replace removed masks 'dfl_full' and 'facehull' with `components` mask

            * clipnorm - Only existed in 2 models (DFL-SAE + Unbalanced). Replaced with global
            option autoclip

        Returns
        -------
        bool
            ``True`` if legacy items exist and state file has been updated, otherwise ``False``
        """
        logger.debug("Checking for legacy state file update")
        priors = ["dssim_loss", "mask_type", "mask_type", "l2_reg_term", "clipnorm"]
        new_items = ["loss_function", "learn_mask", "mask_type", "loss_function_2",
                     "autoclip"]
        updated = False
        for old, new in zip(priors, new_items):
            if old not in self._config:
                logger.debug("Legacy item '%s' not in config. Skipping update", old)
                continue

            # dssim_loss > loss_function
            if old == "dssim_loss":
                self._config[new] = "ssim" if self._config[old] else "mae"
                del self._config[old]
                updated = True
                logger.info("Updated config from legacy dssim format. New config loss "
                            "function: '%s'", self._config[new])
                continue

            # Add learn mask option and set to True if model has "penalized_mask_loss" specified
            if old == "mask_type" and new == "learn_mask" and new not in self._config:
                self._config[new] = self._config["mask_type"] is not None
                updated = True
                logger.info("Added new 'learn_mask' config item for this model. Value set to: %s",
                            self._config[new])
                continue

            # Replace removed masks with most similar equivalent
            if old == "mask_type" and new == "mask_type" and self._config[old] in ("facehull",
                                                                                   "dfl_full"):
                old_mask = self._config[old]
                self._config[new] = "components"
                updated = True
                logger.info("Updated 'mask_type' from '%s' to '%s' for this model",
                            old_mask, self._config[new])

            # Replace l2_reg_term with the correct loss_2_function and update the value of
            # loss_2_weight
            if old == "l2_reg_term":
                self._config[new] = "mse"
                self._config["loss_weight_2"] = self._config[old]
                del self._config[old]
                updated = True
                logger.info("Updated config from legacy 'l2_reg_term' to 'loss_function_2'")

            # Replace clipnorm with correct gradient clipping type and value
            if old == "clipnorm":
                self._config[new] = self._config[old]
                del self._config[old]
                updated = True
                logger.info("Updated config from legacy '%s' to '%s'", old, new)

        logger.debug("State file updated for legacy config: %s", updated)
        return updated

    def _update_changed_config_items(self, config_changeable_items: dict) -> None:
        """ Update any parameters which are not fixed and have been changed.

        Set the :attr:`model_needs_rebuild` to ``True`` if mixed precision state has changed

        Parameters
        ----------
        config_changeable_items: dict
            Configuration options that can be altered when resuming a model, and their current
            values
        """
        rebuild_tasks = ["mixed_precision"]
        if not config_changeable_items:
            logger.debug("No changeable parameters have been updated")
            return
        for key, val in config_changeable_items.items():
            old_val = self._config[key]
            if old_val == val:
                continue
            self._config[key] = val
            logger.info("Config item: '%s' has been updated from '%s' to '%s'", key, old_val, val)
            self._rebuild_model = self._rebuild_model or key in rebuild_tasks


class _Inference():  # pylint:disable=too-few-public-methods
    """ Calculates required layers and compiles a saved model for inference.

    Parameters
    ----------
    saved_model: :class:`keras.models.Model`
        The saved trained Faceswap model
    switch_sides: bool
        ``True`` if the swap should be performed "B" > "A" ``False`` if the swap should be
        "A" > "B"
    """
    def __init__(self, saved_model: tf.keras.models.Model, switch_sides: bool) -> None:
        logger.debug("Initializing: %s (saved_model: %s, switch_sides: %s)",
                     self.__class__.__name__, saved_model, switch_sides)
        self._config = saved_model.get_config()

        self._input_idx = 1 if switch_sides else 0
        self._output_idx = 0 if switch_sides else 1

        self._input_names = [inp[0] for inp in self._config["input_layers"]]
        self._model = self._make_inference_model(saved_model)
        logger.debug("Initialized: %s", self.__class__.__name__)

    @property
    def model(self) -> tf.keras.models.Model:
        """ :class:`keras.models.Model`: The Faceswap model, compiled for inference. """
        return self._model

    def _get_nodes(self, nodes: np.ndarray) -> list[tuple[str, int]]:
        """ Given in input list of nodes from a :attr:`keras.models.Model.get_config` dictionary,
        filters the layer name(s) and output index of the node, splitting to the correct output
        index in the event of multiple inputs.

        Parameters
        ----------
        nodes: list
            A node entry from the :attr:`keras.models.Model.get_config` dictionary

        Returns
        -------
        list
            The (node name, output index) for each node passed in
        """
        anodes = np.array(nodes, dtype="object")[..., :3]
        num_layers = anodes.shape[0]
        anodes = anodes[self._output_idx] if num_layers == 2 else anodes[0]

        # Probably better checks for this, but this occurs when DNY preset is used and learn
        # mask is enabled (i.e. the mask is created in fully connected layers)
        anodes = anodes.squeeze() if anodes.ndim == 3 else anodes

        retval = [(node[0], node[2]) for node in anodes]
        return retval

    def _make_inference_model(self, saved_model: tf.keras.models.Model) -> tf.keras.models.Model:
        """ Extract the sub-models from the saved model that are required for inference.

        Parameters
        ----------
        saved_model: :class:`keras.models.Model`
            The saved trained Faceswap model

        Returns
        -------
        :class:`keras.models.Model`
            The model compiled for inference
        """
        logger.debug("Compiling inference model. saved_model: %s", saved_model)
        struct = self._get_filtered_structure()
        model_inputs = self._get_inputs(saved_model.inputs)
        compiled_layers: dict[str, tf.keras.layers.Layer] = {}
        for layer in saved_model.layers:
            if layer.name not in struct:
                logger.debug("Skipping unused layer: '%s'", layer.name)
                continue
            inbound = struct[layer.name]
            logger.debug("Processing layer '%s': (layer: %s, inbound_nodes: %s)",
                         layer.name, layer, inbound)
            if not inbound:
                model = model_inputs
                logger.debug("Adding model inputs %s: %s", layer.name, model)
            else:
                layer_inputs = []
                for inp in inbound:
                    inbound_layer = compiled_layers[inp[0]]
                    if isinstance(inbound_layer, list) and len(inbound_layer) > 1:
                        # Multi output inputs
                        inbound_output_idx = inp[1]
                        next_input = inbound_layer[inbound_output_idx]
                        logger.debug("Selecting output index %s from multi output inbound layer: "
                                     "%s (using: %s)", inbound_output_idx, inbound_layer,
                                     next_input)
                    else:
                        next_input = inbound_layer

                    layer_inputs.append(next_input)

                logger.debug("Compiling layer '%s': layer inputs: %s", layer.name, layer_inputs)
                model = layer(layer_inputs)
            compiled_layers[layer.name] = model
            retval = keras.models.Model(model_inputs, model, name=f"{saved_model.name}_inference")
        logger.debug("Compiled inference model '%s': %s", retval.name, retval)
        return retval

    def _get_filtered_structure(self) -> OrderedDict:
        """ Obtain the structure of the inference model.

        This parses the model config (in reverse) to obtain the required layers for an inference
        model.

        Returns
        -------
        :class:`collections.OrderedDict`
            The layer name as key with the input name and output index as value.
        """
        # Filter output layer
        out = np.array(self._config["output_layers"], dtype="object")
        if out.ndim == 2:
            out = np.expand_dims(out, axis=1)  # Needs to be expanded for _get_nodes
        outputs = self._get_nodes(out)

        # Iterate backwards from the required output to get the reversed model structure
        current_layers = [outputs[0]]
        next_layers = []
        struct = OrderedDict()
        drop_input = self._input_names[abs(self._input_idx - 1)]
        switch_input = self._input_names[self._input_idx]
        while True:
            layer_info = current_layers.pop(0)
            current_layer = next(lyr for lyr in self._config["layers"]
                                 if lyr["name"] == layer_info[0])
            inbound = current_layer["inbound_nodes"]

            if not inbound:
                break

            inbound_info = self._get_nodes(inbound)

            if any(inb[0] == drop_input for inb in inbound_info):  # Switch inputs
                inbound_info = [(switch_input if inb[0] == drop_input else inb[0], inb[1])
                                for inb in inbound_info]
            struct[layer_info[0]] = inbound_info
            next_layers.extend(inbound_info)

            if not current_layers:
                current_layers = next_layers
                next_layers = []

        struct[switch_input] = []  # Add the input layer
        logger.debug("Model structure: %s", struct)
        return struct

    def _get_inputs(self, inputs: list) -> list:
        """ Obtain the inputs for the requested swap direction.

        Parameters
        ----------
        inputs: list
            The full list of input tensors to the saved faceswap training model

        Returns
        -------
        list
            List of input tensors to feed the model for the requested swap direction
        """
        input_split = len(inputs) // 2
        start_idx = input_split * self._input_idx
        retval = inputs[start_idx: start_idx + input_split]
        logger.debug("model inputs: %s, input_split: %s, start_idx: %s, inference_inputs: %s",
                     inputs, input_split, start_idx, retval)
        return retval
