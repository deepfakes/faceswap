#!/usr/bin/env python3
"""
Settings for the model base plugins.

The objects in this module should not be called directly, but are called from
:class:`~plugins.train.model._base.ModelBase`

Handles configuration of model plugins for:
    - Optimizer settings
    - General global model configuration settings
"""
from __future__ import annotations
import logging
import typing as T

import keras
from keras import config as k_config, dtype_policies, optimizers

from lib.model.optimizers import AdaBelief
from lib.model.autoclip import AutoClipper
from lib.model.nn_blocks import reset_naming
from lib.logger import parse_class_init
from lib.utils import get_module_objects
from plugins.train.train_config import Optimizer as cfg_opt

if T.TYPE_CHECKING:
    from collections.abc import Callable
    from argparse import Namespace
    from .state import State

logger = logging.getLogger(__name__)


class Optimizer():
    """Obtain the selected optimizer with the appropriate keyword arguments."""
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
        """The requested optimizer."""
        return T.cast(optimizers.Optimizer, self._optimizer(**self._kwargs))

    def _configure_clipping(self,
                            method: T.Literal["autoclip", "norm", "value", "none"],
                            value: float,
                            history: int) -> None:
        """Configure optimizer clipping related kwargs, if selected

        Parameters
        ----------
        method
            The clipping method to use. ``None`` for no clipping
        value
            The value to clip by norm/value by. For autoclip, this is the clip percentile
            (a value of 1.0 is a clip percentile of 10%)
        history
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
        # no standardized method to add custom gradient transformers. Currently, we monkey patch
        # its _clip_gradients function, which feels hacky and potentially problematic
        setattr(self._optimizer, "_clip_gradients", AutoClipper(int(value * 10),
                                                                history_size=history))

    def _configure_ema(self, enable: bool, momentum: float, frequency: int) -> None:
        """configure the optimizer kwargs for exponential moving average updates

        Parameters
        ----------
        enable
            ``False`` to disable
        momentum
            the momentum to use when computing the EMA of the model's weights: new_average =
            momentum * old_average + (1 - momentum) * current_variable_value
        frequency
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
        """Configure the remaining global optimizer kwargs

        Parameters
        ----------
        weight_decay
            The amount of weight decay to apply
        gradient_accumulation_steps
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
        """Configure keyword optimizer specific keyword arguments based on user settings."""
        opts = self._valid[cfg_opt.optimizer()][1]
        if not opts:
            logger.debug("No additional kwargs to set for '%s'", cfg_opt.optimizer())
            return

        for key, val in opts.items():
            opt_val = getattr(cfg_opt, key)()
            logger.debug("Setting kwarg '%s' from '%s' to: %s", val, key, opt_val)
            self._kwargs[val] = opt_val

    def _configure(self) -> None:
        """Process the user configuration options into Keras Optimizer kwargs."""
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
    """Core training settings.

    Sets backend settings prior to launching the model.

    Parameters
    ----------
    arguments
        The arguments that were passed to the train or convert process as generated from
        Faceswap's command line arguments
    mixed_precision
        ``True`` if Mixed Precision training should be used otherwise ``False``
    is_predict
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
        """``True`` if mixed precision training has been enabled, otherwise ``False``. """
        return self._use_mixed_precision

    @classmethod
    def loss_scale_optimizer(
            cls,
            optimizer: optimizers.Optimizer) -> optimizers.LossScaleOptimizer:
        """Optimize loss scaling for mixed precision training.

        Parameters
        ----------
        optimizer
            The optimizer instance to wrap

        Returns
        --------
        The original optimizer with loss scaling applied
        """
        return optimizers.LossScaleOptimizer(optimizer)

    @classmethod
    def _set_keras_mixed_precision(cls, enable: bool) -> None:
        """Enable or disable Keras Mixed Precision.

        Parameters
        ----------
        enable
            ``True`` to enable mixed precision. ``False`` to disable.
        """
        policy = dtype_policies.DTypePolicy("mixed_float16" if enable else "float32")
        k_config.set_dtype_policy(policy)
        logger.debug("%s mixed precision. (Compute dtype: %s, variable_dtype: %s)",
                     "Enabling" if enable else "Disabling",
                     policy.compute_dtype, policy.variable_dtype)

    @classmethod
    def _dtype_from_config(cls, config: dict[str, T.Any]) -> str:
        """Obtain the dtype of a layer from the given layer config

        Parameters
        ----------
        config

        Returns
        -------
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
        """Obtain the names of the layers in a mixed precision model that have their dtype policy
        explicitly set to mixed-float16.

        Parameters
        ----------
        layers
            The list of layers that appear in a keras's model configuration `dict`

        Returns
        -------
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
        """Switch a model's datatype between mixed-float16 and float32.

        Parameters
        ----------
        layers
            The list of layers that appear in a keras's model configuration `dict`
        compatible
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
        """Get and store the mixed precision layers from a full precision enabled model.

        Parameters
        ----------
        build_func
            The function to be called to compile the newly created model
        inputs
            The inputs to the model to be compiled

        Returns
        -------
        model
            The built model in fp32
        names
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
        """Check the model's precision.

        If this is a new model, then
        Rewrite an existing model's training precision mode from mixed-float16 to float32 or
        vice versa.

        This is not easy to do in keras, so we edit the model's config to change the dtype policy
        for compatible layers. Create a new model from this config, then port the weights from the
        old model to the new model.

        Parameters
        ----------
        model
            The original saved keras model to rewrite the dtype
        state
            The State information for the model

        Returns
        -------
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
