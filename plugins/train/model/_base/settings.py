#!/usr/bin/env python3
"""
Settings for the model base plugins.

The objects in this module should not be called directly, but are called from
:class:`~plugins.train.model._base.ModelBase`

Handles configuration of model plugins for:
    - Loss configuration
    - Optimizer settings
    - General global model configuration settings
"""
from __future__ import annotations
from dataclasses import dataclass, field
import logging
import platform
import typing as T

from contextlib import nullcontext

import tensorflow as tf
# Ignore linting errors from Tensorflow's thoroughly broken import system
from tensorflow.keras import losses as k_losses  # pylint:disable=import-error
import tensorflow.keras.mixed_precision as mixedprecision  # noqa pylint:disable=import-error

from lib.model import losses, optimizers
from lib.model.autoclip import AutoClipper
from lib.utils import get_backend

if T.TYPE_CHECKING:
    from collections.abc import Callable
    from contextlib import AbstractContextManager as ContextManager
    from argparse import Namespace
    from .model import State

keras = tf.keras
K = keras.backend

logger = logging.getLogger(__name__)


@dataclass
class LossClass:
    """ Typing class for holding loss functions.

    Parameters
    ----------
    function: Callable
        The function that takes in the true/predicted images and returns the loss
    init: bool, Optional
        Whether the loss object ``True`` needs to be initialized (i.e. it's a class) or
        ``False`` it does not require initialization (i.e. it's a function).
        Default ``True``
    kwargs: dict
        Any keyword arguments to supply to the loss function at initialization.
    """
    function: Callable[[tf.Tensor, tf.Tensor], tf.Tensor] | T.Any = k_losses.mae
    init: bool = True
    kwargs: dict[str, T.Any] = field(default_factory=dict)


class Loss():
    """ Holds loss names and functions for an Autoencoder.

    Parameters
    ----------
    config: dict
        The configuration options for the current model plugin
    color_order: str
        Color order of the model. One of `"BGR"` or `"RGB"`
    """
    def __init__(self, config: dict, color_order: T.Literal["bgr", "rgb"]) -> None:
        logger.debug("Initializing %s: (color_order: %s)", self.__class__.__name__, color_order)
        self._config = config
        self._mask_channels = self._get_mask_channels()
        self._inputs: list[tf.keras.layers.Layer] = []
        self._names: list[str] = []
        self._funcs: dict[str, Callable] = {}

        self._loss_dict = {"ffl": LossClass(function=losses.FocalFrequencyLoss),
                           "flip": LossClass(function=losses.LDRFLIPLoss,
                                             kwargs={"color_order": color_order}),
                           "gmsd": LossClass(function=losses.GMSDLoss),
                           "l_inf_norm": LossClass(function=losses.LInfNorm),
                           "laploss": LossClass(function=losses.LaplacianPyramidLoss),
                           "logcosh": LossClass(function=k_losses.logcosh, init=False),
                           "lpips_alex": LossClass(function=losses.LPIPSLoss,
                                                   kwargs={"trunk_network": "alex"}),
                           "lpips_squeeze": LossClass(function=losses.LPIPSLoss,
                                                      kwargs={"trunk_network": "squeeze"}),
                           "lpips_vgg16": LossClass(function=losses.LPIPSLoss,
                                                    kwargs={"trunk_network": "vgg16"}),
                           "ms_ssim": LossClass(function=losses.MSSIMLoss),
                           "mae": LossClass(function=k_losses.mean_absolute_error, init=False),
                           "mse": LossClass(function=k_losses.mean_squared_error, init=False),
                           "pixel_gradient_diff": LossClass(function=losses.GradientLoss),
                           "ssim": LossClass(function=losses.DSSIMObjective),
                           "smooth_loss": LossClass(function=losses.GeneralizedLoss)}

        logger.debug("Initialized: %s", self.__class__.__name__)

    @property
    def names(self) -> list[str]:
        """ list: The list of loss names for the model. """
        return self._names

    @property
    def functions(self) -> dict:
        """ dict: The loss functions that apply to each model output. """
        return self._funcs

    @property
    def _mask_inputs(self) -> list | None:
        """ list: The list of input tensors to the model that contain the mask. Returns ``None``
        if there is no mask input to the model. """
        mask_inputs = [inp for inp in self._inputs if inp.name.startswith("mask")]
        return None if not mask_inputs else mask_inputs

    @property
    def _mask_shapes(self) -> list[tuple] | None:
        """ list: The list of shape tuples for the mask input tensors for the model. Returns
        ``None`` if there is no mask input. """
        if self._mask_inputs is None:
            return None
        return [K.int_shape(mask_input) for mask_input in self._mask_inputs]

    def configure(self, model: tf.keras.models.Model) -> None:
        """ Configure the loss functions for the given inputs and outputs.

        Parameters
        ----------
        model: :class:`keras.models.Model`
            The model that is to be trained
        """
        self._inputs = model.inputs
        self._set_loss_names(model.outputs)
        self._set_loss_functions(model.output_names)
        self._names.insert(0, "total")

    def _set_loss_names(self, outputs: list[tf.Tensor]) -> None:
        """ Name the losses based on model output.

        This is used for correct naming in the state file, for display purposes only.

        Adds the loss names to :attr:`names`

        Notes
        -----
        TODO Currently there is an issue in Tensorflow that wraps all outputs in an Identity layer
        when running in Eager Execution mode, which means we cannot use the name of the output
        layers to name the losses (https://github.com/tensorflow/tensorflow/issues/32180).
        With this in mind, losses are named based on their shapes

        Parameters
        ----------
        outputs: list
            A list of output tensors from the model plugin
        """
        # TODO Use output names if/when these are fixed upstream
        split_outputs = [outputs[:len(outputs) // 2], outputs[len(outputs) // 2:]]
        for side, side_output in zip(("a", "b"), split_outputs):
            output_names = [output.name for output in side_output]
            output_shapes = [K.int_shape(output)[1:] for output in side_output]
            output_types = ["mask" if shape[-1] == 1 else "face" for shape in output_shapes]
            logger.debug("side: %s, output names: %s, output_shapes: %s, output_types: %s",
                         side, output_names, output_shapes, output_types)
            for idx, name in enumerate(output_types):
                suffix = "" if output_types.count(name) == 1 else f"_{idx}"
                self._names.append(f"{name}_{side}{suffix}")
        logger.debug(self._names)

    def _get_function(self, name: str) -> Callable[[tf.Tensor, tf.Tensor], tf.Tensor]:
        """ Obtain the requested Loss function

        Parameters
        ----------
        name: str
            The name of the loss function from the training configuration file

        Returns
        -------
        Keras Loss Function
            The requested loss function
        """
        func = self._loss_dict[name]
        retval = func.function(**func.kwargs) if func.init else func.function  # type:ignore
        logger.debug("Obtained loss function `%s` (%s)", name, retval)
        return retval

    def _set_loss_functions(self, output_names: list[str]):
        """ Set the loss functions and their associated weights.

        Adds the loss functions to the :attr:`functions` dictionary.

        Parameters
        ----------
        output_names: list
            The output names from the model
        """
        face_losses = [(lossname, self._config.get(f"loss_weight_{k[-1]}", 100))
                       for k, lossname in sorted(self._config.items())
                       if k.startswith("loss_function")
                       and self._config.get(f"loss_weight_{k[-1]}", 100) != 0
                       and lossname is not None]

        for name, output_name in zip(self._names, output_names):
            if name.startswith("mask"):
                loss_func = self._get_function(self._config["mask_loss_function"])
            else:
                loss_func = losses.LossWrapper()
                for func, weight in face_losses:
                    self._add_face_loss_function(loss_func, func, weight / 100.)

            logger.debug("%s: (output_name: '%s', function: %s)", name, output_name, loss_func)
            self._funcs[output_name] = loss_func
        logger.debug("functions: %s", self._funcs)

    def _add_face_loss_function(self,
                                loss_wrapper: losses.LossWrapper,
                                loss_function: str,
                                weight: float) -> None:
        """ Add the given face loss function at the given weight and apply any mouth and eye
        multipliers

        Parameters
        ----------
        loss_wrapper: :class:`lib.model.losses.LossWrapper`
            The wrapper loss function that holds the face losses
        loss_function: str
            The loss function to add to the loss wrapper
        weight: float
            The amount of weight to apply to the given loss function
        """
        logger.debug("Adding loss function: %s, weight: %s", loss_function, weight)
        loss_wrapper.add_loss(self._get_function(loss_function),
                              weight=weight,
                              mask_channel=self._mask_channels[0])

        channel_idx = 1
        for section in ("eye_multiplier", "mouth_multiplier"):
            mask_channel = self._mask_channels[channel_idx]
            multiplier = self._config[section] * 1.
            if multiplier > 1.:
                logger.debug("Adding section loss %s: %s", section, multiplier)
                loss_wrapper.add_loss(self._get_function(loss_function),
                                      weight=weight * multiplier,
                                      mask_channel=mask_channel)
            channel_idx += 1

    def _get_mask_channels(self) -> list[int]:
        """ Obtain the channels from the face targets that the masks reside in from the training
        data generator.

        Returns
        -------
        list:
            A list of channel indices that contain the mask for the corresponding config item
        """
        eye_multiplier = self._config["eye_multiplier"]
        mouth_multiplier = self._config["mouth_multiplier"]
        if not self._config["penalized_mask_loss"] and (eye_multiplier > 1 or
                                                        mouth_multiplier > 1):
            logger.warning("You have selected eye/mouth loss multipliers greater than 1x, but "
                           "Penalized Mask Loss is disabled. Disabling all multipliers.")
            eye_multiplier = 1
            mouth_multiplier = 1
        uses_masks = (self._config["penalized_mask_loss"],
                      eye_multiplier > 1,
                      mouth_multiplier > 1)
        mask_channels = [-1 for _ in range(len(uses_masks))]
        current_channel = 3
        for idx, mask_required in enumerate(uses_masks):
            if mask_required:
                mask_channels[idx] = current_channel
                current_channel += 1
        logger.debug("uses_masks: %s, mask_channels: %s", uses_masks, mask_channels)
        return mask_channels


class Optimizer():  # pylint:disable=too-few-public-methods
    """ Obtain the selected optimizer with the appropriate keyword arguments.

    Parameters
    ----------
    optimizer: str
        The selected optimizer name for the plugin
    learning_rate: float
        The selected learning rate to use
    autoclip: bool
        ``True`` if AutoClip should be enabled otherwise ``False``
    epsilon: float
        The value to use for the epsilon of the optimizer
    """
    def __init__(self,
                 optimizer: str,
                 learning_rate: float,
                 autoclip: bool,
                 epsilon: float) -> None:
        logger.debug("Initializing %s: (optimizer: %s, learning_rate: %s, autoclip: %s, "
                     ", epsilon: %s)", self.__class__.__name__, optimizer, learning_rate,
                     autoclip, epsilon)
        valid_optimizers = {"adabelief": (optimizers.AdaBelief,
                                          {"beta_1": 0.5, "beta_2": 0.99, "epsilon": epsilon}),
                            "adam": (optimizers.Adam,
                                     {"beta_1": 0.5, "beta_2": 0.99, "epsilon": epsilon}),
                            "nadam": (optimizers.Nadam,
                                      {"beta_1": 0.5, "beta_2": 0.99, "epsilon": epsilon}),
                            "rms-prop": (optimizers.RMSprop, {"epsilon": epsilon})}
        optimizer_info = valid_optimizers[optimizer]
        self._optimizer: Callable = optimizer_info[0]
        self._kwargs: dict[str, T.Any] = optimizer_info[1]

        self._configure(learning_rate, autoclip)
        logger.verbose("Using %s optimizer", optimizer.title())  # type:ignore[attr-defined]
        logger.debug("Initialized: %s", self.__class__.__name__)

    @property
    def optimizer(self) -> tf.keras.optimizers.Optimizer:
        """ :class:`keras.optimizers.Optimizer`: The requested optimizer. """
        return self._optimizer(**self._kwargs)

    def _configure(self,
                   learning_rate: float,
                   autoclip: bool) -> None:
        """ Configure the optimizer based on user settings.

        Parameters
        ----------
        learning_rate: float
            The selected learning rate to use
        autoclip: bool
            ``True`` if AutoClip should be enabled otherwise ``False``
        """
        self._kwargs["learning_rate"] = learning_rate
        if not autoclip:
            return

        logger.info("Enabling AutoClip")
        self._kwargs["gradient_transformers"] = [AutoClipper(10, history_size=10000)]
        logger.debug("optimizer kwargs: %s", self._kwargs)


class Settings():
    """ Tensorflow core training settings.

    Sets backend tensorflow settings prior to launching the model.

    Tensorflow 2 uses distribution strategies for multi-GPU/system training. These are context
    managers.

    Parameters
    ----------
    arguments: :class:`argparse.Namespace`
        The arguments that were passed to the train or convert process as generated from
        Faceswap's command line arguments
    mixed_precision: bool
        ``True`` if Mixed Precision training should be used otherwise ``False``
    allow_growth: bool
        ``True`` if the Tensorflow allow_growth parameter should be set otherwise ``False``
    is_predict: bool, optional
        ``True`` if the model is being loaded for inference, ``False`` if the model is being loaded
        for training. Default: ``False``
    """
    def __init__(self,
                 arguments: Namespace,
                 mixed_precision: bool,
                 allow_growth: bool,
                 is_predict: bool) -> None:
        logger.debug("Initializing %s: (arguments: %s, mixed_precision: %s, allow_growth: %s, "
                     "is_predict: %s)", self.__class__.__name__, arguments, mixed_precision,
                     allow_growth, is_predict)
        self._set_tf_settings(allow_growth, arguments.exclude_gpus)

        use_mixed_precision = not is_predict and mixed_precision
        self._use_mixed_precision = self._set_keras_mixed_precision(use_mixed_precision)
        if self._use_mixed_precision:
            logger.info("Enabling Mixed Precision Training.")

        if hasattr(arguments, "distribution_strategy"):
            strategy = arguments.distribution_strategy
        else:
            strategy = "default"
        self._strategy = self._get_strategy(strategy)
        logger.debug("Initialized %s", self.__class__.__name__)

    @property
    def use_mixed_precision(self) -> bool:
        """ bool: ``True`` if mixed precision training has been enabled, otherwise ``False``. """
        return self._use_mixed_precision

    @classmethod
    def loss_scale_optimizer(
            cls,
            optimizer: tf.keras.optimizers.Optimizer) -> mixedprecision.LossScaleOptimizer:
        """ Optimize loss scaling for mixed precision training.

        Parameters
        ----------
        optimizer: :class:`tf.keras.optimizers.Optimizer`
            The optimizer instance to wrap

        Returns
        --------
        :class:`tf.keras.mixed_precision.loss_scale_optimizer.LossScaleOptimizer`
            The original optimizer with loss scaling applied
        """
        return mixedprecision.LossScaleOptimizer(optimizer)  # pylint:disable=no-member

    @classmethod
    def _set_tf_settings(cls, allow_growth: bool, exclude_devices: list[int]) -> None:
        """ Specify Devices to place operations on and Allow TensorFlow to manage VRAM growth.

        Enables the Tensorflow allow_growth option if requested in the command line arguments

        Parameters
        ----------
        allow_growth: bool
            ``True`` if the Tensorflow allow_growth parameter should be set otherwise ``False``
        exclude_devices: list or ``None``
            List of GPU device indices that should not be made available to Tensorflow. Pass
            ``None`` if all devices should be made available
        """
        backend = get_backend()
        if backend == "cpu":
            logger.verbose("Hiding GPUs from Tensorflow")  # type:ignore[attr-defined]
            tf.config.set_visible_devices([], "GPU")
            return

        if not exclude_devices and not allow_growth:
            logger.debug("Not setting any specific Tensorflow settings")
            return

        gpus = tf.config.list_physical_devices('GPU')
        if exclude_devices:
            gpus = [gpu for idx, gpu in enumerate(gpus) if idx not in exclude_devices]
            logger.debug("Filtering devices to: %s", gpus)
            tf.config.set_visible_devices(gpus, "GPU")

        if allow_growth and backend == "nvidia":
            logger.debug("Setting Tensorflow 'allow_growth' option")
            for gpu in gpus:
                logger.info("Setting allow growth for GPU: %s", gpu)
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.debug("Set Tensorflow 'allow_growth' option")

    @classmethod
    def _set_keras_mixed_precision(cls, use_mixed_precision: bool) -> bool:
        """ Enable the Keras experimental Mixed Precision API.

        Enables the Keras experimental Mixed Precision API if requested in the user configuration
        file.

        Parameters
        ----------
        use_mixed_precision: bool
            ``True`` if experimental mixed precision support should be enabled for Nvidia GPUs
            otherwise ``False``.

        Returns
        -------
        bool
            ``True`` if mixed precision has been enabled otherwise ``False``
        """
        logger.debug("use_mixed_precision: %s", use_mixed_precision)
        if not use_mixed_precision:
            policy = mixedprecision.Policy('float32')  # pylint:disable=no-member
            mixedprecision.set_global_policy(policy)  # pylint:disable=no-member
            logger.debug("Disabling mixed precision. (Compute dtype: %s, variable_dtype: %s)",
                         policy.compute_dtype, policy.variable_dtype)
            return False

        policy = mixedprecision.Policy('mixed_float16')  # pylint:disable=no-member
        mixedprecision.set_global_policy(policy)  # pylint:disable=no-member
        logger.debug("Enabled mixed precision. (Compute dtype: %s, variable_dtype: %s)",
                     policy.compute_dtype, policy.variable_dtype)
        return True

    def _get_strategy(self,
                      strategy: T.Literal["default", "central-storage", "mirrored"]
                      ) -> tf.distribute.Strategy | None:
        """ If we are running on Nvidia backend and the strategy is not ``None`` then return
        the correct tensorflow distribution strategy, otherwise return ``None``.

        Notes
        -----
        By default Tensorflow defaults mirrored strategy to use the Nvidia NCCL method for
        reductions, however this is only available in Linux, so the method used falls back to
        `Hierarchical Copy All Reduce` if the OS is not Linux.

        Central Storage strategy is not compatible with Mixed Precision. However, in testing it
        worked fine when using a single GPU, so we monkey-patch out the tests for Mixed-Precision
        when using this strategy with a single GPU

        Parameters
        ----------
        strategy: str
            One of 'default', 'central-storage' or 'mirrored'.

        Returns
        -------
        :class:`tensorflow.distribute.Strategy` or `None`
            The request Tensorflow Strategy if the backend is Nvidia and the strategy is not
            `"Default"` otherwise ``None``
        """
        if get_backend() not in ("nvidia", "directml", "rocm"):
            retval = None
        elif strategy == "mirrored":
            retval = self._get_mirrored_strategy()
        elif strategy == "central-storage":
            retval = self._get_central_storage_strategy()
        else:
            retval = tf.distribute.get_strategy()
        logger.debug("Using strategy: %s", retval)
        return retval

    @classmethod
    def _get_mirrored_strategy(cls) -> tf.distribute.MirroredStrategy:
        """ Obtain an instance of a Tensorflow Mirrored Strategy, setting the cross device
        operations appropriate for the OS in use.

        Returns
        -------
        :class:`tensorflow.distribute.MirroredStrategy`
            The Mirrored Distribution Strategy object with correct cross device operations set
        """
        if platform.system().lower() == "linux":
            cross_device_ops = tf.distribute.NcclAllReduce()
        else:
            cross_device_ops = tf.distribute.HierarchicalCopyAllReduce()
        logger.debug("cross_device_ops: %s", cross_device_ops)
        return tf.distribute.MirroredStrategy(cross_device_ops=cross_device_ops)

    @classmethod
    def _get_central_storage_strategy(cls) -> tf.distribute.experimental.CentralStorageStrategy:
        """ Obtain an instance of a Tensorflow Central Storage Strategy. If the strategy is being
        run on a single GPU then monkey patch Tensorflows mixed-precision strategy checks to pass
        successfully.

        Returns
        -------
        :class:`tensorflow.distribute.experimental.CentralStorageStrategy`
            The Central Storage Distribution Strategy object
        """
        gpus = tf.config.get_visible_devices("GPU")
        if len(gpus) == 1:
            # TODO Remove these monkey patches when Strategy supports mixed-precision
            from keras.mixed_precision import loss_scale_optimizer  # noqa pylint:disable=import-outside-toplevel

            # Force a return of True on Loss Scale Optimizer Stategy check
            loss_scale_optimizer.strategy_supports_loss_scaling = lambda: True

            # As LossScaleOptimizer aggregates gradients internally, it passes `False` as the value
            # for `experimental_aggregate_gradients` in `OptimizerV2.apply_gradients`. This causes
            # the optimizer to fail when checking against this strategy. We could monkey patch
            # `Optimizer.apply_gradients`, but it is a lot more code to check, so we just switch
            # the `experimental_aggregate_gradients` back to `True`. In brief testing this does not
            # appear to have a negative impact.
            func = lambda s, grads, wvars, name: s._optimizer.apply_gradients(  # noqa pylint:disable=protected-access,unnecessary-lambda-assignment
                 list(zip(grads, wvars.value)), name, experimental_aggregate_gradients=True)
            loss_scale_optimizer.LossScaleOptimizer._apply_gradients = func  # noqa pylint:disable=protected-access

        return tf.distribute.experimental.CentralStorageStrategy(parameter_device="/cpu:0")

    def _get_mixed_precision_layers(self, layers: list[dict]) -> list[str]:
        """ Obtain the names of the layers in a mixed precision model that have their dtype policy
        explicitly set to mixed-float16.

        Parameters
        ----------
        layers: List
            The list of layers that appear in a keras's model configuration `dict`

        Returns
        -------
        list
            A list of layer names within the model that are assigned a float16 policy
        """
        retval = []
        for layer in layers:
            config = layer["config"]

            if layer["class_name"] in ("Functional", "Sequential"):  # Recurse into sub-models
                retval.extend(self._get_mixed_precision_layers(config["layers"]))
                continue

            dtype = config["dtype"]
            if isinstance(dtype, dict) and dtype["config"]["name"] == "mixed_float16":
                logger.debug("Adding supported mixed precision layer: %s %s", layer["name"], dtype)
                retval.append(layer["name"])
            else:
                logger.debug("Skipping unsupported layer: %s %s",
                             layer.get("name", f"class_name: {layer['class_name']}"), dtype)
        return retval

    def _switch_precision(self, layers: list[dict], compatible: list[str]) -> None:
        """ Switch a model's datatype between mixed-float16 and float32.

        Parameters
        ----------
        layers: List
            The list of layers that appear in a keras's model configuration `dict`
        compatible: List
            A list of layer names that are compatible to have their datatype switched
        """
        dtype = "mixed_float16" if self.use_mixed_precision else "float32"
        policy = {"class_name": "Policy", "config": {"name": dtype}}

        for layer in layers:
            config = layer["config"]

            if layer["class_name"] in ["Functional", "Sequential"]:  # Recurse into sub-models
                self._switch_precision(config["layers"], compatible)
                continue

            if layer["name"] not in compatible:
                logger.debug("Skipping incompatible layer: %s", layer["name"])
                continue

            logger.debug("Updating dtype for %s from: %s to: %s",
                         layer["name"], config["dtype"], policy)
            config["dtype"] = policy

    def get_mixed_precision_layers(self,
                                   build_func: Callable[[list[tf.keras.layers.Layer]],
                                                        tf.keras.models.Model],
                                   inputs: list[tf.keras.layers.Layer]
                                   ) -> tuple[tf.keras.models.Model, list[str]]:
        """ Get and store the mixed precision layers from a full precision enabled model.

        Parameters
        ----------
        build_func: Callable
            The function to be called to compile the newly created model
        inputs:
            The inputs to the model to be compiled

        Returns
        -------
        model: :class:`tensorflow.keras.model`
            The built model in fp32
        list
            The list of layer names within the full precision model that can be switched
            to mixed precision
        """
        logger.info("Storing Mixed Precision compatible layers. Please ignore any following "
                    "warnings about using mixed precision.")
        self._set_keras_mixed_precision(True)
        with tf.device("CPU"):
            model = build_func(inputs)
            layers = self._get_mixed_precision_layers(model.get_config()["layers"])

        tf.keras.backend.clear_session()
        self._set_keras_mixed_precision(False)

        config = model.get_config()
        self._switch_precision(config["layers"], layers)
        new_model = model.from_config(config)
        del model
        return new_model, layers

    def check_model_precision(self,
                              model: tf.keras.models.Model,
                              state: "State") -> tf.keras.models.Model:
        """ Check the model's precision.

        If this is a new model, then
        Rewrite an existing model's training precsion mode from mixed-float16 to float32 or
        vice versa.

        This is not easy to do in keras, so we edit the model's config to change the dtype policy
        for compatible layers. Create a new model from this config, then port the weights from the
        old model to the new model.

        Parameters
        ----------
        model: :class:`keras.models.Model`
            The original saved keras model to rewrite the dtype
        state: ~:class:`plugins.train.model._base.model.State`
            The State information for the model

        Returns
        -------
        :class:`keras.models.Model`
            The original model with the datatype updated
        """
        if self.use_mixed_precision and not state.mixed_precision_layers:
            # Switching to mixed precision on a model which was started in FP32 prior to the
            # ability to switch between precisions on a saved model is not supported as we
            # do not have the compatible layer names
            logger.warning("Switching from Full Precision to Mixed Precision is not supported on "
                           "older model files. Reverting to Full Precision.")
            return model

        config = model.get_config()

        if not self.use_mixed_precision and not state.mixed_precision_layers:
            # Switched to Full Precision, get compatible layers from model if not already stored
            state.add_mixed_precision_layers(self._get_mixed_precision_layers(config["layers"]))

        self._switch_precision(config["layers"], state.mixed_precision_layers)

        new_model = keras.models.Model().from_config(config)
        new_model.set_weights(model.get_weights())
        logger.info("Mixed precision has been updated from '%s' to '%s'",
                    not self.use_mixed_precision, self.use_mixed_precision)
        del model
        return new_model

    def strategy_scope(self) -> ContextManager:
        """ Return the strategy scope if we have set a strategy, otherwise return a null
        context.

        Returns
        -------
        :func:`tensorflow.python.distribute.Strategy.scope` or :func:`contextlib.nullcontext`
            The tensorflow strategy scope if a strategy is valid in the current scenario. A null
            context manager if the strategy is not valid in the current scenario
        """
        retval = nullcontext() if self._strategy is None else self._strategy.scope()
        logger.debug("Using strategy scope: %s", retval)
        return retval
