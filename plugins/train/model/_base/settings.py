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
import logging
import platform

from contextlib import nullcontext
from typing import Callable, ContextManager, Dict, List, Optional, TYPE_CHECKING

import tensorflow as tf

from lib.model import losses, optimizers
from lib.utils import get_backend, get_tf_version

if get_backend() == "amd":
    import keras
    from keras import losses as k_losses
    from keras import backend as K
else:
    # Ignore linting errors from Tensorflow's thoroughly broken import system
    from tensorflow import keras
    from tensorflow.keras import losses as k_losses  # pylint:disable=import-error
    from tensorflow.keras import backend as K  # pylint:disable=import-error

if get_tf_version() < 2.4:
    import tensorflow.keras.mixed_precision.experimental as mixedprecision  # noqa pylint:disable=import-error,no-name-in-module
else:
    import tensorflow.keras.mixed_precision as mixedprecision  # noqa pylint:disable=import-error,no-name-in-module

if TYPE_CHECKING:
    from argparse import Namespace

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class Loss():
    """ Holds loss names and functions for an Autoencoder.

    Parameters
    ----------
    config: dict
        The configuration options for the current model plugin
    """
    def __init__(self, config: dict) -> None:
        logger.debug("Initializing %s", self.__class__.__name__)
        self._config = config
        self._loss_dict = dict(gmsd=losses.GMSDLoss(),
                               l_inf_norm=losses.LInfNorm(),
                               laploss=losses.LaplacianPyramidLoss(),
                               logcosh=k_losses.logcosh,
                               ms_ssim=losses.MSSIMLoss(),
                               mae=k_losses.mean_absolute_error,
                               mse=k_losses.mean_squared_error,
                               pixel_gradient_diff=losses.GradientLoss(),
                               ssim=losses.DSSIMObjective(),
                               smooth_loss=losses.GeneralizedLoss(),)
        self._mask_channels = self._get_mask_channels()
        self._inputs: List[keras.layers.Layer] = []
        self._names: List[str] = []
        self._funcs: Dict[str, Callable] = {}
        logger.debug("Initialized: %s", self.__class__.__name__)

    @property
    def names(self) -> List[str]:
        """ list: The list of loss names for the model. """
        return self._names

    @property
    def functions(self) -> dict:
        """ dict: The loss functions that apply to each model output. """
        return self._funcs

    @property
    def _mask_inputs(self) -> Optional[list]:
        """ list: The list of input tensors to the model that contain the mask. Returns ``None``
        if there is no mask input to the model. """
        mask_inputs = [inp for inp in self._inputs if inp.name.startswith("mask")]
        return None if not mask_inputs else mask_inputs

    @property
    def _mask_shapes(self) -> Optional[List[tuple]]:
        """ list: The list of shape tuples for the mask input tensors for the model. Returns
        ``None`` if there is no mask input. """
        if self._mask_inputs is None:
            return None
        return [K.int_shape(mask_input) for mask_input in self._mask_inputs]

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

    def _set_loss_names(self, outputs: List[tf.Tensor]) -> None:
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

    def _set_loss_functions(self, output_names: List[str]):
        """ Set the loss functions and their associated weights.

        Adds the loss functions to the :attr:`functions` dictionary.

        Parameters
        ----------
        output_names: list
            The output names from the model
        """
        face_losses = [(self._loss_dict[v], self._config.get(f"loss_weight_{k[-1]}", 100))
                       for k, v in sorted(self._config.items())
                       if k.startswith("loss_function")
                       and self._config.get(f"loss_weight_{k[-1]}", 100) != 0
                       and v is not None]

        for name, output_name in zip(self._names, output_names):
            if name.startswith("mask"):
                loss_func = self._loss_dict[self._config["mask_loss_function"]]
            else:
                loss_func = losses.LossWrapper()
                for func, weight in face_losses:
                    self._add_face_loss_function(loss_func, func, weight / 100.)

            logger.debug("%s: (output_name: '%s', function: %s)", name, output_name, loss_func)
            self._funcs[output_name] = loss_func
        logger.debug("functions: %s", self._funcs)

    def _add_face_loss_function(self,
                                loss_wrapper: losses.LossWrapper,
                                loss_function: Callable,
                                weight: float) -> None:
        """ Add the given face loss function at the given weight and apply any mouth and eye
        multipliers

        Parameters
        ----------
        loss_wrapper: :class:`lib.model.losses.LossWrapper`
            The wrapper loss function that holds the face losses
        loss_func: :class:`keras.losses.Loss`
            The loss function to add to the loss wrapper
        weight: float
            The amount of weight to apply to the given loss function
        """
        logger.debug("Adding loss function: %s, weight: %s", loss_function, weight)
        loss_wrapper.add_loss(loss_function,
                              weight=weight,
                              mask_channel=self._mask_channels[0])

        channel_idx = 1
        for section in ("eye_multiplier", "mouth_multiplier"):
            mask_channel = self._mask_channels[channel_idx]
            multiplier = self._config[section] * 1.
            if multiplier > 1.:
                logger.debug("Adding section loss %s: %s", section, multiplier)
                loss_wrapper.add_loss(loss_function,
                                      weight=weight * multiplier,
                                      mask_channel=mask_channel)
            channel_idx += 1

    def _get_mask_channels(self) -> List[int]:
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
    clipnorm: bool
        Whether to clip gradients to avoid exploding/vanishing gradients
    epsilon: float
        The value to use for the epsilon of the optimizer
    mixed_precision: bool
        ``True`` if mixed precision training is to be enabled otherwise ``False``
    arguments: :class:`argparse.Namespace`
        The arguments that were passed to the train or convert process as generated from
        Faceswap's command line arguments
    """
    def __init__(self,
                 optimizer: str,
                 learning_rate: float,
                 clipnorm: bool,
                 epsilon: float,
                 mixed_precision: bool,
                 arguments: "Namespace") -> None:
        logger.debug("Initializing %s: (optimizer: %s, learning_rate: %s, clipnorm: %s, "
                     "epsilon: %s, mixed_precision: %s, arguments: %s)", self.__class__.__name__,
                     optimizer, learning_rate, clipnorm, epsilon, mixed_precision, arguments)
        valid_optimizers = {"adabelief": (optimizers.AdaBelief,
                                          dict(beta_1=0.5, beta_2=0.99, epsilon=epsilon)),
                            "adam": (optimizers.Adam,
                                     dict(beta_1=0.5, beta_2=0.99, epsilon=epsilon)),
                            "nadam": (optimizers.Nadam,
                                      dict(beta_1=0.5, beta_2=0.99, epsilon=epsilon)),
                            "rms-prop": (optimizers.RMSprop, dict(epsilon=epsilon))}
        self._optimizer, self._kwargs = valid_optimizers[optimizer]

        self._configure(learning_rate, clipnorm, mixed_precision, arguments)
        logger.verbose("Using %s optimizer", optimizer.title())  # type:ignore
        logger.debug("Initialized: %s", self.__class__.__name__)

    @property
    def optimizer(self) -> keras.optimizers.Optimizer:
        """ :class:`keras.optimizers.Optimizer`: The requested optimizer. """
        return self._optimizer(**self._kwargs)

    def _configure(self,
                   learning_rate: float,
                   clipnorm: bool,
                   mixed_precision: bool,
                   arguments: "Namespace") -> None:
        """ Configure the optimizer based on user settings.

        Parameters
        ----------
        learning_rate: float
            The selected learning rate to use
        clipnorm: bool
            Whether to clip gradients to avoid exploding/vanishing gradients
        mixed_precision: bool
            ``True`` if mixed precision training is to be enabled otherwise ``False``
        arguments: :class:`argparse.Namespace`
            The arguments that were passed to the train or convert process as generated from
            Faceswap's command line arguments

        Notes
        -----
        Clip-norm is ballooning VRAM usage, which is not expected behavior and may be a bug in
        Keras/Tensorflow.

        PlaidML has a bug regarding the clip-norm parameter See:
        https://github.com/plaidml/plaidml/issues/228. We workaround by simply not adding this
        parameter for AMD backend users.
        """
        lr_key = "lr" if get_backend() == "amd" else "learning_rate"
        self._kwargs[lr_key] = learning_rate

        if clipnorm and (arguments.distributed or mixed_precision):
            logger.warning("Clipnorm has been selected, but is unsupported when using distributed "
                           "or mixed_precision training, so has been disabled. If you wish to "
                           "enable clipnorm, then you must disable these other options.")
            clipnorm = False
        if clipnorm and get_backend() == "amd":
            # TODO add clipnorm in for plaidML when it is fixed upstream. Still not fixed in
            # release 0.7.0.
            logger.warning("Due to a bug in plaidML, clipnorm cannot be used on AMD backends so "
                           "has been disabled")
            clipnorm = False
        if clipnorm:
            self._kwargs["clipnorm"] = 1.0

        logger.debug("optimizer kwargs: %s", self._kwargs)


class Settings():
    """ Tensorflow core training settings.

    Sets backend tensorflow settings prior to launching the model.

    Tensorflow 2 uses distribution strategies for multi-GPU/system training. These are context
    managers. To enable the code to be more readable, we handle strategies the same way for Nvidia
    and AMD backends. PlaidML does not support strategies, but we need to still create a context
    manager so that we don't need branching logic.

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
                 arguments: "Namespace",
                 mixed_precision: bool,
                 allow_growth: bool,
                 is_predict: bool) -> None:
        logger.debug("Initializing %s: (arguments: %s, mixed_precision: %s, allow_growth: %s, "
                     "is_predict: %s)", self.__class__.__name__, arguments, mixed_precision,
                     allow_growth, is_predict)
        self._set_tf_settings(allow_growth, arguments.exclude_gpus)

        use_mixed_precision = not is_predict and mixed_precision and get_backend() == "nvidia"
        self._use_mixed_precision = self._set_keras_mixed_precision(use_mixed_precision,
                                                                    bool(arguments.exclude_gpus))

        distributed = False if not hasattr(arguments, "distributed") else arguments.distributed
        self._strategy = self._get_strategy(distributed)
        logger.debug("Initialized %s", self.__class__.__name__)

    @property
    def use_strategy(self) -> bool:
        """ bool: ``True`` if a distribution strategy is to be used otherwise ``False``. """
        return self._strategy is not None

    @property
    def use_mixed_precision(self) -> bool:
        """ bool: ``True`` if mixed precision training has been enabled, otherwise ``False``. """
        return self._use_mixed_precision

    @classmethod
    def loss_scale_optimizer(
            cls,
            optimizer: keras.optimizers.Optimizer) -> mixedprecision.LossScaleOptimizer:
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
        return mixedprecision.LossScaleOptimizer(optimizer)

    @classmethod
    def _set_tf_settings(cls, allow_growth: bool, exclude_devices: List[int]) -> None:
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
        if get_backend() == "amd":
            return  # No settings for AMD
        if get_backend() == "cpu":
            logger.verbose("Hiding GPUs from Tensorflow")  # type:ignore
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

        if allow_growth:
            logger.debug("Setting Tensorflow 'allow_growth' option")
            for gpu in gpus:
                logger.info("Setting allow growth for GPU: %s", gpu)
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.debug("Set Tensorflow 'allow_growth' option")

    @classmethod
    def _set_keras_mixed_precision(cls, use_mixed_precision: bool, exclude_gpus: bool) -> bool:
        """ Enable the Keras experimental Mixed Precision API.

        Enables the Keras experimental Mixed Precision API if requested in the user configuration
        file.

        Parameters
        ----------
        use_mixed_precision: bool
            ``True`` if experimental mixed precision support should be enabled for Nvidia GPUs
            otherwise ``False``.
        exclude_gpus: bool
            ``True`` If connected GPUs are being excluded otherwise ``False``.

        Returns
        -------
        bool
            ``True`` if mixed precision has been enabled otherwise ``False``
        """
        logger.debug("use_mixed_precision: %s, exclude_gpus: %s",
                     use_mixed_precision, exclude_gpus)
        if not use_mixed_precision:
            logger.debug("Not enabling 'mixed_precision' (backend: %s, use_mixed_precision: %s)",
                         get_backend(), use_mixed_precision)
            return False
        logger.info("Enabling Mixed Precision Training.")

        policy = mixedprecision.Policy('mixed_float16')
        mixedprecision.set_global_policy(policy)
        logger.debug("Enabled mixed precision. (Compute dtype: %s, variable_dtype: %s)",
                     policy.compute_dtype, policy.variable_dtype)
        return True

    @classmethod
    def _get_strategy(cls, distributed: bool) -> Optional[tf.distribute.Strategy]:
        """ If we are running on Nvidia backend and the strategy is not `"default"` then return
        the correct tensorflow distribution strategy, otherwise return ``None``.

        Notes
        -----
        By default Tensorflow defaults mirrored strategy to use the Nvidia NCCL method for
        reductions, however this is only available in Linux, so the method used falls back to
        `Hierarchical Copy All Reduce` if the OS is not Linux.

        Parameters
        ----------
        distributed: bool
            ``True`` if Tensorflow mirrored strategy should be used for multiple GPU training.
            ``False`` if the default strategy should be used.

        Returns
        -------
        :class:`tensorflow.distribute.Strategy` or `None`
            The request Tensorflow Strategy if the backend is Nvidia and the strategy is not
            `"Default"` otherwise ``None``
        """
        if get_backend() != "nvidia":
            retval = None
        elif distributed:
            if platform.system().lower() == "linux":
                cross_device_ops = tf.distribute.NcclAllReduce()
            else:
                cross_device_ops = tf.distribute.HierarchicalCopyAllReduce()
            logger.debug("cross_device_ops: %s", cross_device_ops)
            retval = tf.distribute.MirroredStrategy(cross_device_ops=cross_device_ops)
        else:
            retval = tf.distribute.get_strategy()
        logger.debug("Using strategy: %s", retval)
        return retval

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
