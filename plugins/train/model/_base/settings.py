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

from lib.model.nn_blocks import reset_naming
from lib.utils import get_module_objects

if T.TYPE_CHECKING:
    from collections.abc import Callable
    from argparse import Namespace
    from .state import State

logger = logging.getLogger(__name__)


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
