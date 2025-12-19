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
import typing as T

import keras
from keras import config as k_config, dtype_policies, losses as k_losses, optimizers

from lib.model import losses
from lib.model.optimizers import AdaBelief
from lib.model.autoclip import AutoClipper
from lib.model.nn_blocks import reset_naming
from lib.logger import parse_class_init
from lib.utils import get_module_objects
from plugins.train.train_config import Loss as cfg_loss, Optimizer as cfg_opt

if T.TYPE_CHECKING:
    from collections.abc import Callable
    from argparse import Namespace
    from keras import KerasTensor
    from .state import State

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
    function: Callable[[KerasTensor, KerasTensor],
                       KerasTensor] | T.Any = k_losses.MeanSquaredError
    init: bool = True
    kwargs: dict[str, T.Any] = field(default_factory=dict)


class Loss():
    """ Holds loss names and functions for an Autoencoder.

    Parameters
    ----------
    color_order: str
        Color order of the model. One of `"BGR"` or `"RGB"`
    """
    def __init__(self, color_order: T.Literal["bgr", "rgb"]) -> None:
        logger.debug(parse_class_init(locals()))
        self._mask_channels = self._get_mask_channels()
        self._inputs: list[keras.layers.Layer] = []
        self._names: list[str] = []
        self._funcs: dict[str, losses.LossWrapper | T.Callable[[KerasTensor, KerasTensor],
                                                               KerasTensor]] = {}

        self._loss_dict = {"ffl": LossClass(function=losses.FocalFrequencyLoss),
                           "flip": LossClass(function=losses.LDRFLIPLoss,
                                             kwargs={"color_order": color_order}),
                           "gmsd": LossClass(function=losses.GMSDLoss),
                           "l_inf_norm": LossClass(function=losses.LInfNorm),
                           "laploss": LossClass(function=losses.LaplacianPyramidLoss),
                           "logcosh": LossClass(function=k_losses.LogCosh),
                           "lpips_alex": LossClass(function=losses.LPIPSLoss,
                                                   kwargs={"trunk_network": "alex"}),
                           "lpips_squeeze": LossClass(function=losses.LPIPSLoss,
                                                      kwargs={"trunk_network": "squeeze"}),
                           "lpips_vgg16": LossClass(function=losses.LPIPSLoss,
                                                    kwargs={"trunk_network": "vgg16"}),
                           "ms_ssim": LossClass(function=losses.MSSIMLoss),
                           "mae": LossClass(function=k_losses.MeanAbsoluteError),
                           "mse": LossClass(function=k_losses.MeanSquaredError),
                           "pixel_gradient_diff": LossClass(function=losses.GradientLoss),
                           "ssim": LossClass(function=losses.DSSIMObjective),
                           "smooth_loss": LossClass(function=losses.GeneralizedLoss)}

        logger.debug("Initialized: %s", self.__class__.__name__)

    @property
    def names(self) -> list[str]:
        """ list: The list of loss names for the model. """
        return self._names

    @property
    def functions(self) -> dict[str, losses.LossWrapper | T.Callable[[KerasTensor, KerasTensor],
                                                                     KerasTensor]]:
        """ dict[str, :class:`~lib.model.losses.LossWrapper` | | Callable[[KerasTensor,
        KerasTensor], KerasTensor]]]: The loss functions that apply to each model output. """
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
        return [mask_input.shape for mask_input in self._mask_inputs]

    def configure(self, model: keras.models.Model) -> None:
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

    def _set_loss_names(self, outputs: list[KerasTensor]) -> None:
        """ Name the losses based on model output.

        This is used for correct naming in the state file, for display purposes only.

        Adds the loss names to :attr:`names`

        Parameters
        ----------
        outputs: list[:class:`keras.KerasTensor`]
            A list of output tensors from the model plugin
        """
        # TODO Use output names if/when these are fixed upstream
        split_outputs = [outputs[:len(outputs) // 2], outputs[len(outputs) // 2:]]
        for side, side_output in zip(("a", "b"), split_outputs):
            output_names = [output.name for output in side_output]
            output_shapes = [output.shape[1:] for output in side_output]
            output_types = ["mask" if shape[-1] == 1 else "face" for shape in output_shapes]
            logger.debug("side: %s, output names: %s, output_shapes: %s, output_types: %s",
                         side, output_names, output_shapes, output_types)
            for idx, name in enumerate(output_types):
                suffix = "" if output_types.count(name) == 1 else f"_{idx}"
                self._names.append(f"{name}_{side}{suffix}")
        logger.debug(self._names)

    def _get_function(self, name: str) -> Callable[[KerasTensor, KerasTensor], KerasTensor]:
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

    def _set_loss_functions(self, output_names: list[str]) -> None:
        """ Set the loss functions and their associated weights.

        Adds the loss functions to the :attr:`functions` dictionary.

        Parameters
        ----------
        output_names: list[str]
            The output names from the model
        """
        loss_funcs = [cfg_loss.loss_function(),
                      cfg_loss.loss_function_2(),
                      cfg_loss.loss_function_3(),
                      cfg_loss.loss_function_4()]
        loss_amount = [100,
                       cfg_loss.loss_weight_2(),
                       cfg_loss.loss_weight_3(),
                       cfg_loss.loss_weight_4()]
        face_losses = [(name, weight) for name, weight in zip(loss_funcs, loss_amount)
                       if name != "none" and weight > 0]

        for name, output_name in zip(self._names, output_names):
            if name.startswith("mask"):
                loss_func = self._get_function(cfg_loss.mask_loss_function())
            else:
                loss_func = losses.LossWrapper()
                for func, weight in face_losses:
                    self._add_face_loss_function(loss_func, func, weight / 100.)

            logger.debug("%s: (output_name: '%s', function: %s)", name, output_name, loss_func)
            self._funcs[name] = loss_func
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
        for section, multiplier in zip(
                ("eye_multiplier", "mouth_multiplier"),
                (float(cfg_loss.eye_multiplier()), float(cfg_loss.mouth_multiplier()))):
            mask_channel = self._mask_channels[channel_idx]
            multiplier *= 1.
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
        eye_multiplier = cfg_loss.eye_multiplier()
        mouth_multiplier = cfg_loss.mouth_multiplier()
        if not cfg_loss.penalized_mask_loss() and (eye_multiplier > 1 or mouth_multiplier > 1):
            logger.warning("You have selected eye/mouth loss multipliers greater than 1x, but "
                           "Penalized Mask Loss is disabled. Disabling all multipliers.")
            eye_multiplier = 1
            mouth_multiplier = 1
        uses_masks = (cfg_loss.penalized_mask_loss(), eye_multiplier > 1, mouth_multiplier > 1)
        mask_channels = [-1 for _ in range(len(uses_masks))]
        current_channel = 3
        for idx, mask_required in enumerate(uses_masks):
            if mask_required:
                mask_channels[idx] = current_channel
                current_channel += 1
        logger.debug("uses_masks: %s, mask_channels: %s", uses_masks, mask_channels)
        return mask_channels


class Optimizer():
    """ Obtain the selected optimizer with the appropriate keyword arguments. """
    def __init__(self) -> None:
        logger.debug(parse_class_init(locals()))
        betas = {"ada_beta_1": "beta_1", "ada_beta_2": "beta_2"}
        amsgrad = {"ada_amsgrad": "amsgrad"}
        self._valid: dict[str, tuple[T.Type[Optimizer], dict[str, T.Any]]] = {
            "adabelief": (AdaBelief, betas | amsgrad),
            "adam": (optimizers.Adam, betas | amsgrad),
            "adamax": (optimizers.Adamax, betas),
            "adamw": (optimizers.AdamW, betas | amsgrad),
            "lion": (optimizers.Lion, betas),
            "nadam": (optimizers.Nadam, betas),
            "rms-prop": (optimizers.RMSprop, {})}

        self._optimizer = self._valid[cfg_opt.optimizer()][0]
        self._kwargs: dict[str, T.Any] = {"learning_rate": cfg_opt.learning_rate()}
        if cfg_opt.optimizer() != "lion":
            self._kwargs["epsilon"] = 10 ** int(cfg_opt.epsilon_exponent())

        self._configure()
        logger.info("Using %s optimizer", self._optimizer.__name__)
        logger.debug("Initialized: %s", self.__class__.__name__)

    @property
    def optimizer(self) -> optimizers.Optimizer:
        """ :class:`keras.optimizers.Optimizer`: The requested optimizer. """
        return T.cast(optimizers.Optimizer, self._optimizer(**self._kwargs))

    def _configure_clipping(self,
                            method: T.Literal["autoclip", "norm", "value", "none"],
                            value: float,
                            history: int) -> None:
        """ Configure optimizer clipping related kwargs, if selected

        Parameters
        ----------
        method: Literal["autoclip", "norm", "value", "none"]
            The clipping method to use. ``None`` for no clipping
        value: float
            The value to clip by norm/value by. For autoclip, this is the clip percentile
            (a value of 1.0 is a clip percentile of 10%)
        history: int
            autoclip only: The number of iterations to keep for calculating the normalized value
        """
        logger.debug("method: '%s', value: %s, history: %s", method, value, history)
        if method == "none":
            logger.debug("clipping disabled")
            return

        logger.info("Enabling Clipping: %s", method.replace("_", " ").replace("_", " ").title())
        clip_types = {"global_norm": "global_clipnorm", "norm": "clipnorm", "value": "clipvalue"}
        if method in clip_types:
            self._kwargs[clip_types[method]] = value
            logger.debug("Setting clipping kwargs for '%s': %s",
                         method, {k: v for k, v in self._kwargs.items()
                                  if k == clip_types[method]})
            return

        assert method == "autoclip"
        # Test for if keras optimizer changes its structure to no longer have _clip_gradients.
        # Ensures any tests fails in this situation
        assert hasattr(self._optimizer,
                       "_clip_gradients"), "keras.BaseOptimizer._clip_gradients no longer exists"

        # TODO Keras3 has removed the ""gradient_transformers" kwarg, and there now appears to be
        # no standardised method to add custom gradent transformers. Currently, we monkey patch its
        # _clip_gradients function, which feels hacky and potentially problematic
        setattr(self._optimizer, "_clip_gradients", AutoClipper(int(value * 10),
                                                                history_size=history))

    def _configure_ema(self, enable: bool, momentum: float, frequency: int) -> None:
        """ Confihure the optimizer kwargs for exponential moving average updates

        Parameters
        ----------
        enable: bool
            ``False`` to disable
        momentum: float
            the momentum to use when computing the EMA of the model's weights: new_average =
            momentum * old_average + (1 - momentum) * current_variable_value
        frequency: int
            the number of iterations, to overwrite the model variable by its moving average.
        """
        self._kwargs["use_ema"] = enable
        if not enable:
            logger.debug("ema disabled.")
            return

        logger.info("Enabling EMA")
        self._kwargs["ema_momentum"] = momentum
        self._kwargs["ema_overwrite_frequency"] = frequency
        logger.debug("ema enabled (momentum: %s, frequency: %s)", momentum, frequency)

    def _configure_kwargs(self, weight_decay: float, gradient_accumulation_steps: int) -> None:
        """ Configure the remaining global optimizer kwargs

        Parameters
        ----------
        weight_decay: float
            The amount of weight decay to apply
        gradient_accumulation_steps: int
            The number of steps to accumulate gradients for before applying the average
        """
        if weight_decay > 0.0:
            logger.info("Enabling Weight Decay: %s", weight_decay)
            self._kwargs["weight_decay"] = weight_decay
        else:
            logger.debug("weight decay disabled")

        if gradient_accumulation_steps > 1:
            logger.info("Enabling Gradient Accumulation: %s", gradient_accumulation_steps)
            self._kwargs["gradient_accumulation_steps"] = gradient_accumulation_steps
        else:
            logger.debug("gradient accumulation disabled")

    def _configure_specific(self) -> None:
        """ Configure keyword optimizer specific keyword arguments based on user settings. """
        opts = self._valid[cfg_opt.optimizer()][1]
        if not opts:
            logger.debug("No additional kwargs to set for '%s'", cfg_opt.optimizer())
            return

        for key, val in opts.items():
            opt_val = getattr(cfg_opt, key)()
            logger.debug("Setting kwarg '%s' from '%s' to: %s", val, key, opt_val)
            self._kwargs[val] = opt_val

    def _configure(self) -> None:
        """ Process the user configuration options into Keras Optimizer kwargs. """
        self._configure_clipping(T.cast(T.Literal["autoclip", "norm", "value", "none"],
                                        cfg_opt.gradient_clipping()),
                                 cfg_opt.clipping_value(),
                                 cfg_opt.autoclip_history())

        self._configure_ema(cfg_opt.use_ema(),
                            cfg_opt.ema_momentum(),
                            cfg_opt.ema_frequency())

        self._configure_kwargs(cfg_opt.weight_decay(),
                               cfg_opt.gradient_accumulation())

        self._configure_specific()

        logger.debug("Configured '%s' optimizer. kwargs: %s", cfg_opt.optimizer(), self._kwargs)


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
    is_predict: bool, optional
        ``True`` if the model is being loaded for inference, ``False`` if the model is being loaded
        for training. Default: ``False``
    """
    def __init__(self,
                 arguments: Namespace,
                 mixed_precision: bool,
                 is_predict: bool) -> None:
        logger.debug("Initializing %s: (arguments: %s, mixed_precision: %s, is_predict: %s)",
                     self.__class__.__name__, arguments, mixed_precision, is_predict)
        use_mixed_precision = not is_predict and mixed_precision
        self._use_mixed_precision = use_mixed_precision
        if use_mixed_precision:
            logger.info("Enabling Mixed Precision Training.")

        self._set_keras_mixed_precision(use_mixed_precision)
        logger.debug("Initialized %s", self.__class__.__name__)

    @property
    def use_mixed_precision(self) -> bool:
        """ bool: ``True`` if mixed precision training has been enabled, otherwise ``False``. """
        return self._use_mixed_precision

    @classmethod
    def loss_scale_optimizer(
            cls,
            optimizer: optimizers.Optimizer) -> optimizers.LossScaleOptimizer:
        """ Optimize loss scaling for mixed precision training.

        Parameters
        ----------
        optimizer: :class:`keras.optimizers.Optimizer`
            The optimizer instance to wrap

        Returns
        --------
        :class:`keras.optimizers.LossScaleOptimizer`
            The original optimizer with loss scaling applied
        """
        return optimizers.LossScaleOptimizer(optimizer)

    @classmethod
    def _set_keras_mixed_precision(cls, enable: bool) -> None:
        """ Enable or disable Keras Mixed Precision.

        Parameters
        ----------
        enable: bool
            ``True`` to enable mixed precision. ``False`` to disable.

        Enables or disables the Keras Mixed Precision API if requested in the user configuration
        file.
        """
        policy = dtype_policies.DTypePolicy("mixed_float16" if enable else "float32")
        k_config.set_dtype_policy(policy)
        logger.debug("%s mixed precision. (Compute dtype: %s, variable_dtype: %s)",
                     "Enabling" if enable else "Disabling",
                     policy.compute_dtype, policy.variable_dtype)

#    def _get_strategy(self,
#                      strategy: T.Literal["default", "central-storage", "mirrored"]
#                      ) -> tf.distribute.Strategy | None:
#        """ If we are running on Nvidia backend and the strategy is not ``None`` then return
#        the correct tensorflow distribution strategy, otherwise return ``None``.
#
#        Notes
#        -----
#        By default Tensorflow defaults mirrored strategy to use the Nvidia NCCL method for
#        reductions, however this is only available in Linux, so the method used falls back to
#        `Hierarchical Copy All Reduce` if the OS is not Linux.
#
#        Central Storage strategy is not compatible with Mixed Precision. However, in testing it
#        worked fine when using a single GPU, so we monkey-patch out the tests for Mixed-Precision
#        when using this strategy with a single GPU
#
#        Parameters
#        ----------
#        strategy: str
#            One of 'default', 'central-storage' or 'mirrored'.
#
#        Returns
#        -------
#        :class:`tensorflow.distribute.Strategy` or `None`
#            The request Tensorflow Strategy if the backend is Nvidia and the strategy is not
#            `"Default"` otherwise ``None``
#        """
#        if get_backend() not in ("nvidia", "rocm"):
#            retval = None
#        elif strategy == "mirrored":
#            retval = self._get_mirrored_strategy()
#        elif strategy == "central-storage":
#            retval = self._get_central_storage_strategy()
#        else:
#            retval = tf.distribute.get_strategy()
#        logger.debug("Using strategy: %s", retval)
#        return retval

#    @classmethod
#    def _get_mirrored_strategy(cls) -> tf.distribute.MirroredStrategy:
#        """ Obtain an instance of a Tensorflow Mirrored Strategy, setting the cross device
#        operations appropriate for the OS in use.
#
#        Returns
#        -------
#        :class:`tensorflow.distribute.MirroredStrategy`
#            The Mirrored Distribution Strategy object with correct cross device operations set
#        """
#        if platform.system().lower() == "linux":
#            cross_device_ops = tf.distribute.NcclAllReduce()
#        else:
#            cross_device_ops = tf.distribute.HierarchicalCopyAllReduce()
#        logger.debug("cross_device_ops: %s", cross_device_ops)
#        return tf.distribute.MirroredStrategy(cross_device_ops=cross_device_ops)

#    @classmethod
#    def _get_central_storage_strategy(cls) -> tf.distribute.experimental.CentralStorageStrategy:
#        """ Obtain an instance of a Tensorflow Central Storage Strategy. If the strategy is being
#        run on a single GPU then monkey patch Tensorflows mixed-precision strategy checks to pass
#        successfully.
#
#        Returns
#        -------
#        :class:`tensorflow.distribute.experimental.CentralStorageStrategy`
#            The Central Storage Distribution Strategy object
#        """
#        gpus = tf.config.get_visible_devices("GPU")
#        if len(gpus) == 1:
#            # TODO Remove these monkey patches when Strategy supports mixed-precision
#            # pylint:disable=import-outside-toplevel
#            from keras.mixed_precision import loss_scale_optimizer
#
#            # Force a return of True on Loss Scale Optimizer Stategy check
#            loss_scale_optimizer.strategy_supports_loss_scaling = lambda: True
#
#           # As LossScaleOptimizer aggregates gradients internally, it passes `False` as the value
#           # for `experimental_aggregate_gradients` in `OptimizerV2.apply_gradients`. This causes
#           # the optimizer to fail when checking against this strategy. We could monkey patch
#           # `Optimizer.apply_gradients`, but it is a lot more code to check, so we just switch
#           # the `experimental_aggregate_gradients` back to `True`. In brief testing this does not
#           # appear to have a negative impact.
#            func = lambda s, grads, wvars, name: s._optimizer.apply_gradients(  # noqa pylint:disable=protected-access,unnecessary-lambda-assignment
#                 list(zip(grads, wvars.value)), name, experimental_aggregate_gradients=True)
#            loss_scale_optimizer.LossScaleOptimizer._apply_gradients = func  # noqa pylint:disable=protected-access

#        return tf.distribute.experimental.CentralStorageStrategy(parameter_device="/cpu:0")

    @classmethod
    def _dtype_from_config(cls, config: dict[str, T.Any]) -> str:
        """ Obtain the dtype of a layer from the given layer config

        Parameters
        ----------
        config: dict[str, Any] : The Keras layer configuration dictionary

        Returns
        -------
        str
            The datatype of the layer
        """
        dtype = config["dtype"]
        logger.debug("Obtaining layer dtype from config: %s", dtype)
        if isinstance(dtype, str):
            return dtype
        # Fail tests if Keras changes the way it stores dtypes
        assert isinstance(dtype, dict) and "config" in dtype, (
            "Keras config dtype storage method has changed")

        dtype_conf = dtype["config"]
        # Fail tests if Keras changes the way it stores dtypes
        assert isinstance(dtype_conf, dict) and "name" in dtype_conf, (
            "Keras config dtype storage method has changed")

        retval = dtype_conf["name"]
        return retval

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

            if "dtype" not in config:
                logger.debug("Skipping unsupported layer: %s %s",
                             layer.get("name", f"class_name: {layer['class_name']}"), config)
                continue
            dtype = self._dtype_from_config(config)
            logger.debug("layer: '%s', dtype: '%s'", config["name"], dtype)

            if dtype == "mixed_float16":
                logger.debug("Adding supported mixed precision layer: %s %s",
                             layer["config"]["name"], dtype)
                retval.append(layer["config"]["name"])
            else:
                logger.debug("Skipping unsupported layer: %s %s",
                             layer["config"].get("name", f"class_name: {layer['class_name']}"),
                             dtype)
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

        for layer in layers:
            config = layer["config"]

            if layer["class_name"] in ["Functional", "Sequential"]:  # Recurse into sub-models
                self._switch_precision(config["layers"], compatible)
                continue

            if layer["config"]["name"] not in compatible:
                logger.debug("Skipping incompatible layer: %s", layer["config"]["name"])
                continue

            logger.debug("Updating dtype for %s from: %s to: %s",
                         layer["config"]["name"], config["dtype"], dtype)
            config["dtype"] = dtype

    def get_mixed_precision_layers(self,
                                   build_func: Callable[[list[keras.layers.Layer]],
                                                        keras.models.Model],
                                   inputs: list[keras.layers.Layer]
                                   ) -> tuple[keras.models.Model, list[str]]:
        """ Get and store the mixed precision layers from a full precision enabled model.

        Parameters
        ----------
        build_func: Callable
            The function to be called to compile the newly created model
        inputs:
            The inputs to the model to be compiled

        Returns
        -------
        model: :class:`keras.model`
            The built model in fp32
        list
            The list of layer names within the full precision model that can be switched
            to mixed precision
        """
        logger.debug("Storing Mixed Precision compatible layers.")
        self._set_keras_mixed_precision(True)
        with keras.device("CPU"):
            model = build_func(inputs)
            layers = self._get_mixed_precision_layers(model.get_config()["layers"])

        del model
        keras.backend.clear_session()

        self._set_keras_mixed_precision(False)
        reset_naming()
        model = build_func(inputs)

        logger.debug("model: %s, mixed precision layers: %s", model, layers)
        return model, layers

    def check_model_precision(self,
                              model: keras.models.Model,
                              state: "State") -> keras.models.Model:
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
        weights = model.get_weights()

        if not self.use_mixed_precision and not state.mixed_precision_layers:
            # Switched to Full Precision, get compatible layers from model if not already stored
            state.add_mixed_precision_layers(self._get_mixed_precision_layers(config["layers"]))

        self._switch_precision(config["layers"], state.mixed_precision_layers)

        del model
        keras.backend.clear_session()
        new_model = keras.models.Model().from_config(config)

        new_model.set_weights(weights)
        logger.info("Mixed precision has been %s",
                    "enabled" if self.use_mixed_precision else "disabled")
        return new_model


__all__ = get_module_objects(__name__)
