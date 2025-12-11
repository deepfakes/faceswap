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

import keras

from lib.logger import parse_class_init
from lib.serializer import get_serializer
from lib.model.nn_blocks import set_config as set_nnblock_config
from lib.utils import get_module_objects, FaceswapError
from plugins.train._config import Config

from .io import IO, get_all_sub_models, Weights
from .settings import Loss, Optimizer, Settings

if T.TYPE_CHECKING:
    import argparse
    import keras.src.ops.node
    from lib.config import ConfigValueType


logger = logging.getLogger(__name__)
_CONFIG: dict[str, ConfigValueType] = {}


class ModelBase():  # pylint:disable=too-many-instance-attributes
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
        logger.debug(parse_class_init(locals()))
        # Input shape must be set within the plugin after initializing
        self.input_shape: tuple[int, ...] = ()
        self.trainer = "original"  # Override for plugin specific trainer
        self.color_order: T.Literal["bgr", "rgb"] = "bgr"  # Override for image color channel order

        self._args = arguments
        self._is_predict = predict
        self._model: keras.Model | None = None

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
                                  self._is_predict)
        self._loss = Loss(self.config, self.color_order)

        logger.debug("Initialized ModelBase (%s)", self.__class__.__name__)

    @property
    def model(self) -> keras.Model:
        """:class:`keras.Model`: The compiled model for this plugin. """
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
        shapes = [T.cast(tuple[None, int, int, int], inputs.shape)
                  for inputs in self.model.inputs]
        return shapes

    @property
    def output_shapes(self) -> list[tuple[None, int, int, int]]:
        """ list: A flattened list corresponding to all of the outputs of the model. """
        shapes = [T.cast(tuple[None, int, int, int], output.shape)
                  for output in self.model.outputs]
        return shapes

    @property
    def iterations(self) -> int:
        """ int: The total number of iterations that the model has trained. """
        return self._state.iterations

    @property
    def warmup_steps(self) -> int:
        """ int : The number of steps to perform learning rate warmup """
        return self._args.warmup

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
        is_summary = hasattr(self._args, "summary") and self._args.summary
        with self._settings.strategy_scope():
            if self._io.model_exists:
                model = self.io.load()
                if self._is_predict:
                    inference = Inference(model, self._args.swap_model)
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

    def _validate_input_shape(self) -> None:
        """ Validate that the input shape is either a single shape tuple of 3 dimensions or
        a list of 2 shape tuples of 3 dimensions. """
        assert len(self.input_shape) == 3, "Input shape should be a 3 dimensional shape tuple"

    def _get_inputs(self) -> list[keras.layers.Input]:
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

    def build_model(self, inputs: list[keras.layers.Input]) -> keras.Model:
        """ Override for Model Specific autoencoder builds.

        Parameters
        ----------
        inputs: list
            A list of :class:`keras.layers.Input` tensors. This will be a list of 2 tensors (one
            for each side) each of shapes :attr:`input_shape`.

        Returns
        -------
        :class:`keras.Model`
            See Keras documentation for the correct structure, but note that parameter :attr:`name`
            is a required rather than an optional argument in Faceswap. You should assign this to
            the attribute ``self.name`` that is automatically generated from the plugin's filename.
        """
        raise NotImplementedError

    def _summary_to_log(self, summary: str) -> None:
        """ Function to output Keras model summary to log file at verbose log level

        Parameters
        ----------
        summary, str
            The model summary output from keras
        """
        for line in summary.splitlines():
            logger.verbose(line)  # type:ignore[attr-defined]

    def _output_summary(self) -> None:
        """ Output the summary of the model and all sub-models to the verbose logger. """
        if hasattr(self._args, "summary") and self._args.summary:
            print_fn = None  # Print straight to stdout
        else:
            # print to logger
            print_fn = self._summary_to_log
        parent = self.model
        for idx, model in enumerate(get_all_sub_models(self.model)):
            if idx == 0:
                parent = model
                continue
            model.summary(print_fn=print_fn)
        parent.summary(print_fn=print_fn)

    def _compile_model(self) -> None:
        """ Compile the model to include the Optimizer and Loss Function(s). """
        logger.debug("Compiling Model")

        if self.state.model_needs_rebuild:
            self._model = self._settings.check_model_precision(self._model, self._state)

        optimizer = Optimizer(self.config).optimizer
        if self._settings.use_mixed_precision:
            optimizer = self._settings.loss_scale_optimizer(optimizer)

        weights = Weights(self)
        weights.load(self._io.model_exists)
        weights.freeze()

        self._loss.configure(self.model)
        losses = list(self._loss.functions.values())
        self.model.compile(optimizer=optimizer, loss=losses)
        self._state.add_session_loss_names(self._loss.names)
        logger.debug("Compiled Model: %s", self.model)

    def add_history(self, loss: list[float]) -> None:
        """ Add the current iteration's loss history to :attr:`_io.history`.

        Called from the trainer after each iteration, for tracking loss drop over time between
        save iterations.

        Parameters
        ----------
        loss: list[float]
            The loss values for the A and B side for the current iteration. This should be the
            collated loss values for each side.
        """
        self._io.history.append(sum(loss))


class State():  # pylint:disable=too-many-instance-attributes
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
        logger.debug(parse_class_init(locals()))
        self._serializer = get_serializer("json")
        filename = f"{model_name}_state.{self._serializer.file_extension}"
        self._filename = os.path.join(model_dir, filename)
        self._name = model_name
        self._iterations = 0
        self._mixed_precision_layers: list[str] = []
        self._lr_finder = -1.0
        self._rebuild_model = False
        self._sessions: dict[int, dict] = {}
        self.lowest_avg_loss: float = 0.0
        """float: The lowest average loss seen between save intervals. """

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
    def lr_finder(self) -> float:
        """ The value discovered from the learning rate finder. -1 if no value stored """
        return self._lr_finder

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

    def add_lr_finder(self, learning_rate: float) -> None:
        """ Add the optimal discovered learning rate from the learning rate finder

        Parameters
        ----------
        learning_rate : float
            The discovered learning rate
        """
        logger.debug("Storing learning rate from LR Finder: %s", learning_rate)
        self._lr_finder = learning_rate

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

        self.lowest_avg_loss = state.get("lowest_avg_loss", 0.0)
        if isinstance(self.lowest_avg_loss, dict):
            lowest_avg_loss = sum(self.lowest_avg_loss.values())
            logger.debug("Collating legacy lowest_avg_loss from %s to %s",
                         self.lowest_avg_loss, lowest_avg_loss)
            self.lowest_avg_loss = lowest_avg_loss

        self._iterations = state.get("iterations", 0)
        self._mixed_precision_layers = state.get("mixed_precision_layers", [])
        self._lr_finder = state.get("lr_finder", -1.0)
        self._config = state.get("config", {})
        logger.debug("Loaded state: %s", state)
        self._replace_config(config_changeable_items)

    def save(self) -> None:
        """ Save the state values to the serialized state file. """
        logger.debug("Saving State")
        state = {"name": self._name,
                 "sessions": {k: v for k, v in self._sessions.items()
                              if v.get("iterations", 0) > 0},
                 "lowest_avg_loss": self.lowest_avg_loss,
                 "iterations": self._iterations,
                 "mixed_precision_layers": self._mixed_precision_layers,
                 "lr_finder": self._lr_finder,
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
        priors = ["dssim_loss", "mask_type", "mask_type", "l2_reg_term", "clipnorm", "autoclip"]
        new_items = ["loss_function", "learn_mask", "mask_type", "loss_function_2", "clipping",
                     "clipping"]
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
                self._config[new] = "norm"
                del self._config[old]
                updated = True
                logger.info("Updated config from legacy '%s' to  '%s: %s'", old, new, old)

            # Replace autoclip with correct gradient clipping type
            if old == "autoclip":
                self._config[new] = old
                del self._config[old]
                updated = True
                logger.info("Updated config from legacy '%s' to '%s: %s'", old, new, old)

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


class Inference():
    """ Calculates required layers and compiles a saved model for inference.

    Parameters
    ----------
    saved_model: :class:`keras.Model`
        The saved trained Faceswap model
    switch_sides: bool
        ``True`` if the swap should be performed "B" > "A" ``False`` if the swap should be
        "A" > "B"
    """
    def __init__(self, saved_model: keras.Model, switch_sides: bool) -> None:
        logger.debug(parse_class_init(locals()))

        self._layers: list[keras.Layer] = [lyr for lyr in saved_model.layers
                                           if not isinstance(lyr, keras.layers.InputLayer)]
        """list[:class:`keras.layers.Layer]: All the layers that exist within the model excluding
        input layers """

        self._input = self._get_model_input(saved_model, switch_sides)
        """:class:`keras.KerasTensor`: The correct input for the inference model """

        self._name = f"{saved_model.name}_inference"
        """str: The name for the final inference model"""

        self._model = self._build()
        logger.debug("Initialized: %s", self.__class__.__name__)

    @property
    def model(self) -> keras.Model:
        """ :class:`keras.Model`: The Faceswap model, compiled for inference. """
        return self._model

    def _get_model_input(self, model: keras.Model, switch_sides: bool) -> list[keras.KerasTensor]:
        """ Obtain the inputs for the requested swap direction.

        Parameters
        ----------
        saved_model: :class:`keras.Model`
            The saved trained Faceswap model
        switch_sides: bool
            ``True`` if the swap should be performed "B" > "A" ``False`` if the swap should be
            "A" > "B"

        Returns
        -------
        list[]:class:`keras.KerasTensor`]
            The input tensor to feed the model for the requested swap direction
        """
        inputs: list[keras.KerasTensor] = model.input
        assert len(inputs) == 2, "Faceswap models should have exactly 2 inputs"
        idx = 0 if switch_sides else 1
        retval = inputs[idx]
        logger.debug("model inputs: %s, idx: %s, inference_input: '%s'",
                     [(i.name, i.shape[1:]) for i in inputs], idx, retval.name)
        return [retval]

    def _get_candidates(self, input_tensors: list[keras.KerasTensor | keras.Layer]
                        ) -> T.Generator[tuple[keras.Layer, list[keras.src.ops.node.KerasHistory]],
                                         None, None]:
        """ Given a list of input tensors, get all layers from the main model which have the given
        input tensors marked as Inbound nodes for the model

        Parameters
        ----------
        input_tensors: list[:class:`keras.KerasTensor` | :class:`keras.Layer`]
            List of Tensors that act as an input to a layer within the model

        Yields
        ------
        tuple[:class:`keras.KerasLayer`, list[:class:`keras.src.ops.node.KerasHistory']
            Any layer in the main model that use the given input tensors as an input along with the
            corresponding keras inbound history
        """
        unique_input_names = set(i.name for i in input_tensors)
        for layer in self._layers:

            history = [tensor._keras_history  # pylint:disable=protected-access
                       for node in layer._inbound_nodes  # pylint:disable=protected-access
                       for parent in node.parent_nodes
                       for tensor in parent.outputs]

            unique_inbound_names = set(h.operation.name for h in history)
            if not unique_input_names.issubset(unique_inbound_names):
                logger.debug("%s: Skipping candidate '%s' unmatched inputs: %s",
                             unique_input_names, layer.name, unique_inbound_names)
                continue

            logger.debug("%s: Yielding candidate '%s'. History: %s",
                         unique_input_names, layer.name, [(h.operation.name, h.node_index)
                                                          for h in history])
            yield layer, history

    @T.overload
    def _group_inputs(self, layer: keras.Layer, inputs: list[tuple[keras.Layer, int]]
                      ) -> list[list[tuple[keras.Layer, int]]]:
        ...

    @T.overload
    def _group_inputs(self, layer: keras.Layer, inputs: list[keras.src.ops.node.KerasHistory]
                      ) -> list[list[keras.src.ops.node.KerasHistory]]:
        ...

    def _group_inputs(self, layer, inputs):
        """ Layers can have more than one input. In these instances we need to group the inputs
        and the layers' inbound nodes to correspond to inputs per instance.

        Parameters
        ----------
        layer: :class:`keras.Layer`
            The current layer being processed
        inputs: list[:class:`keras.KerasTensor`] | list[:class:`keras.src.ops.node.KerasHistory`]
            List of input tensors or inbound keras histories to be grouped per layer input

        Returns
        -------
        list[list[tuple[:class:`keras.Layer`, int]]] |
        list[list[:class:`keras.src.ops.node.KerasHistory`]
            A list of list of input layers  and the corresponding node index or inbound keras
            histories
        """
        layer_inputs = 1 if isinstance(layer.input, keras.KerasTensor) else len(layer.input)
        num_inputs = len(inputs)

        total_calls = num_inputs / layer_inputs
        assert total_calls.is_integer()
        total_calls = int(total_calls)

        retval = [inputs[i * layer_inputs: i * layer_inputs + layer_inputs]
                  for i in range(total_calls)]

        return retval

    def _layers_from_inputs(self,
                            input_tensors: list[keras.KerasTensor | keras.Layer],
                            node_indices: list[int]
                            ) -> tuple[list[keras.Layer],
                                       list[keras.src.ops.node.KerasHistory],
                                       list[int]]:
        """ Given a list of input tensors and their corresponding inbound node ids, return all of
        the layers for the model that uses the given nodes as their input

        Parameters
        ----------
        input_tensors: list[:class:`keras.KerasTensor` | :class:`keras.Layer`]
            List of Tensors that act as an input to a layer within the model
        node_indices: list[int]
            The list of node indices corresponding to the inbound node index of the given layers

        Returns
        -------
        list[:class:`keras.layers.Layer`]
            Any layers from the model that use the given inputs as its input. Empty list if there
            are no matches
        list[:class:`keras.src.ops.node.KerasHistory`]
            The keras inbound history for the layers
        list[int]
            The output node index for the layer, used for the inbound node index of the next layer
        """
        retval: tuple[list[keras.Layer],
                      list[keras.src.ops.node.KerasHistory],
                      list[int]] = ([], [], [])
        for layer, history in self._get_candidates(input_tensors):
            grp_inputs = self._group_inputs(layer, list(zip(input_tensors, node_indices)))
            grp_hist = self._group_inputs(layer, history)

            for input_group in grp_inputs:  # pylint:disable=not-an-iterable
                have = [(i[0].name, i[1]) for i in input_group]
                for out_idx, hist in enumerate(grp_hist):
                    requires = [(h.operation.name, h.node_index) for h in hist]
                    if sorted(have) != sorted(requires):
                        logger.debug("%s: Skipping '%s'. Requires %s. Output node index: %s",
                                     have, layer.name, requires, out_idx)
                        continue
                    retval[0].append(layer)
                    retval[1].append(hist)
                    retval[2].append(out_idx)

        logger.debug("Got layers %s for input_tensors: %s",
                     [x.name for x in retval[0]], [t.name for t in input_tensors])
        return retval

    def _build_layers(self,
                      layers: list[keras.Layer],
                      history: list[keras.src.ops.node.KerasHistory],
                      inputs: list[keras.KerasTensor]) -> list[keras.KerasTensor]:
        """ Compile the given layers with the given inputs

        Parameters
        ----------
        layers: list[:class:`keras.Layer`]
            The layers to be called with the given inputs
        history: list[:class:`keras.src.ops.node.KerasHistory`]
            The corresponding keras inbound history for the layers
        inputs: list[:class:`keras.KerasTensor]
            The inputs for the given layers

        Returns
        -------
        list[:class:`keras.KerasTensor`]
            The list of compiled layers
        """
        retval = []
        given_order = [i._keras_history.operation.name  # pylint:disable=protected-access
                       for i in inputs]
        for layer, hist in zip(layers, history):
            layer_input = [inputs[given_order.index(h.operation.name)]
                           for h in hist if h.operation.name in given_order]
            if layer_input != inputs:
                logger.debug("Sorted layer inputs %s to %s",
                             given_order,
                             [i._keras_history.operation.name  # pylint:disable=protected-access
                              for i in layer_input])

            if isinstance(layer_input, list) and len(layer_input) == 1:
                # Flatten single inputs to stop Keras warnings
                actual_input = layer_input[0]
            else:
                actual_input = layer_input

            built = layer(actual_input)
            built = built if isinstance(built, list) else [built]
            logger.debug(
                "Compiled layer '%s' from input(s) %s",
                layer.name,
                [i._keras_history.operation.name  # pylint:disable=protected-access
                 for i in layer_input])
            retval.extend(built)

        logger.debug(
            "Compiled layers %s from input %s",
            [x._keras_history.operation.name for x in retval],  # pylint:disable=protected-access
            [x._keras_history.operation.name for x in inputs])  # pylint:disable=protected-access
        return retval

    def _build(self):
        """ Extract the sub-models from the saved model that are required for inference.

        Returns
        -------
        :class:`keras.Model`
            The model compiled for inference
        """
        logger.debug("Compiling inference model")

        layers = self._input
        node_index = [0]
        built = layers

        while True:
            layers, history, node_index = self._layers_from_inputs(layers, node_index)
            if not layers:
                break

            built = self._build_layers(layers, history, built)

        assert len(self._input) == 1
        assert len(built) == 1
        retval = keras.Model(inputs=self._input[0], outputs=built[0], name=self._name)
        logger.debug("Compiled inference model '%s': %s", retval.name, retval)

        return retval


__all__ = get_module_objects(__name__)
