#!/usr/bin/env python3
"""
Base class for Models. ALL Models should at least inherit from this class.

See :mod:`~plugins.train.model.original` for an annotated example for how to create model plugins.
"""
from __future__ import annotations
import logging
import os
import sys
import typing as T

import keras

from lib.logger import parse_class_init
from lib.utils import get_module_objects, FaceswapError
from plugins.train import train_config as cfg

from .inference import Inference
from .io import IO, get_all_sub_models, Weights
from .settings import Loss, Optimizer, Settings
from .state import State

if T.TYPE_CHECKING:
    import argparse
    import numpy as np


logger = logging.getLogger(__name__)


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
    """
    def __init__(self,
                 model_dir: str,
                 arguments: argparse.Namespace,
                 predict: bool = False) -> None:
        logger.debug(parse_class_init(locals()))
        # Input shape must be set within the plugin after initializing
        self.input_shape: tuple[int, ...] = ()
        self.color_order: T.Literal["bgr", "rgb"] = "bgr"  # Override for image color channel order

        self._args = arguments
        self._is_predict = predict
        self._model: keras.Model | None = None

        cfg.load_config(config_file=arguments.configfile)

        if cfg.Loss.penalized_mask_loss() and cfg.Loss.mask_type() == "none":
            raise FaceswapError("Penalized Mask Loss has been selected but you have not chosen a "
                                "Mask to use. Please select a mask or disable Penalized Mask "
                                "Loss.")

        if cfg.Loss.learn_mask() and cfg.Loss.mask_type() == "none":
            raise FaceswapError("'Learn Mask' has been selected but you have not chosen a Mask to "
                                "use. Please select a mask or disable 'Learn Mask'.")

        self._mixed_precision = cfg.mixed_precision()
        self._io = IO(self, model_dir,
                      self._is_predict,
                      T.cast(T.Literal["never", "always", "exit"], cfg.Optimizer.save_optimizer()))
        self._check_multiple_models()

        self._state = State(model_dir,
                            self.name,
                            False if self._is_predict else self._args.no_logs)
        self._settings = Settings(self._args,
                                  self._mixed_precision,
                                  self._is_predict)
        self._loss = Loss(self.color_order)

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
        return cfg.coverage() / 100.

    @property
    def io(self) -> IO:  # pylint:disable=invalid-name
        """ :class:`~plugins.train.model.io.IO`: Input/Output operations for the model """
        return self._io

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

    @property
    def freeze_layers(self) -> list[str]:
        """ list[str] : Override to set plugin specific layers that can be frozen. Defaults to
        ["encoder"] """
        return ["encoder"]

    @property
    def load_layers(self) -> list[str]:
        """ list[str] : Override to set plugin specific layers that can be loaded. Defaults to
        ["encoder"] """
        return ["encoder"]

    # Private properties
    @property
    def _config_section(self) -> str:
        """ str: The section name for the current plugin for loading configuration options from the
        config file. """
        return ".".join(self.__module__.split(".")[-2:])

    @property
    def state(self) -> "State":
        """:class:`State`: The state settings for the current plugin. """
        return self._state

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

        optimizer = Optimizer().optimizer
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

    def add_history(self, loss: np.ndarray) -> None:
        """ Add the current iteration's loss history to :attr:`_io.history`.

        Called from the trainer after each iteration, for tracking loss drop over time between
        save iterations.

        Parameters
        ----------
        loss : :class:`numpy.ndarray`
            The loss values for the A and B side for the current iteration. This should be the
            collated loss values for each side.
        """
        self._io.history.append(float(sum(loss)))


__all__ = get_module_objects(__name__)
