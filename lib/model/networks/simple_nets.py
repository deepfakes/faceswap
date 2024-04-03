#!/usr/bin/env python3
""" Ports of existing NN Architecture for use in faceswap.py """
from __future__ import annotations
import logging
import typing as T

import tensorflow as tf

# Fix intellisense/linting for tf.keras' thoroughly broken import system
keras = tf.keras
layers = keras.layers
Model = keras.models.Model

if T.TYPE_CHECKING:
    from tensorflow import Tensor


logger = logging.getLogger(__name__)


class _net():  # pylint:disable=too-few-public-methods
    """ Base class for existing NeuralNet architecture

    Notes
    -----
    All architectures assume channels_last format

    Parameters
    ----------
    input_shape, Tuple, optional
        The input shape for the model. Default: ``None``
    """
    def __init__(self,
                 input_shape: tuple[int, int, int] | None = None) -> None:
        logger.debug("Initializing: %s (input_shape: %s)", self.__class__.__name__, input_shape)
        self._input_shape = (None, None, 3) if input_shape is None else input_shape
        assert len(self._input_shape) == 3 and self._input_shape[-1] == 3, (
            "Input shape must be in the format (height, width, channels) and the number of "
            f"channels must equal 3. Received: {self._input_shape}")
        logger.debug("Initialized: %s", self.__class__.__name__)


class AlexNet(_net):
    """ AlexNet ported from torchvision version.

    Notes
    -----
    This port only contains the features portion of the model.

    References
    ----------
    https://papers.nips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf

    Parameters
    ----------
    input_shape, Tuple, optional
        The input shape for the model. Default: ``None``
    """
    def __init__(self, input_shape: tuple[int, int, int] | None = None) -> None:
        super().__init__(input_shape)
        self._feature_indices = [0, 3, 6, 8, 10]  # For naming equivalent to PyTorch
        self._filters = [64, 192, 384, 256, 256]  # Filters at each block

    @classmethod
    def _conv_block(cls,
                    inputs: Tensor,
                    padding: int,
                    filters: int,
                    kernel_size: int,
                    strides: int,
                    block_idx: int,
                    max_pool: bool) -> Tensor:
        """
        The Convolutional block for AlexNet

        Parameters
        ----------
        inputs: :class:`tf.Tensor`
            The input tensor to the block
        padding: int
            The amount of zero paddin to apply prior to convolution
        filters: int
            The number of filters to apply during convolution
        kernel_size: int
            The kernel size of the convolution
        strides: int
            The number of strides for the convolution
        block_idx: int
            The index of the current block (for standardized naming convention)
        max_pool: bool
            ``True`` to apply a max pooling layer at the beginning of the block otherwise ``False``

        Returns
        -------
        :class:`tf.Tensor`
            The output of the Convolutional block
        """
        name = f"features.{block_idx}"
        var_x = inputs
        if max_pool:
            var_x = layers.MaxPool2D(pool_size=3, strides=2, name=f"{name}.pool")(var_x)
        var_x = layers.ZeroPadding2D(padding=padding, name=f"{name}.pad")(var_x)
        var_x = layers.Conv2D(filters,
                              kernel_size=kernel_size,
                              strides=strides,
                              padding="valid",
                              activation="relu",
                              name=name)(var_x)
        return var_x

    def __call__(self) -> tf.keras.models.Model:
        """ Create the AlexNet Model

        Returns
        -------
        :class:`keras.models.Model`
            The compiled AlexNet model
        """
        inputs = layers.Input(self._input_shape)
        var_x = inputs
        kernel_size = 11
        strides = 4

        for idx, (filters, block_idx) in enumerate(zip(self._filters, self._feature_indices)):
            padding = 2 if idx < 2 else 1
            do_max_pool = 0 < idx < 3
            var_x = self._conv_block(var_x,
                                     padding,
                                     filters,
                                     kernel_size,
                                     strides,
                                     block_idx,
                                     do_max_pool)
            kernel_size = max(3, kernel_size // 2)
            strides = 1
        return Model(inputs=inputs, outputs=[var_x])


class SqueezeNet(_net):
    """ SqueezeNet ported from torchvision version.

    Notes
    -----
    This port only contains the features portion of the model.

    References
    ----------
    https://arxiv.org/abs/1602.07360

    Parameters
    ----------
    input_shape, Tuple, optional
        The input shape for the model. Default: ``None``
    """

    @classmethod
    def _fire(cls,
              inputs: Tensor,
              squeeze_planes: int,
              expand_planes: int,
              block_idx: int) -> Tensor:
        """ The fire block for SqueezeNet.

        Parameters
        ----------
        inputs: :class:`tf.Tensor`
            The input to the fire block
        squeeze_planes: int
            The number of filters for the squeeze convolution
        expand_planes: int
            The number of filters for the expand convolutions
        block_idx: int
            The index of the current block (for standardized naming convention)

        Returns
        -------
        :class:`tf.Tensor`
            The output of the SqueezeNet fire block
        """
        name = f"features.{block_idx}"
        squeezed = layers.Conv2D(squeeze_planes, 1,
                                 activation="relu", name=f"{name}.squeeze")(inputs)
        expand1 = layers.Conv2D(expand_planes, 1,
                                activation="relu", name=f"{name}.expand1x1")(squeezed)
        expand3 = layers.Conv2D(expand_planes,
                                3,
                                activation="relu",
                                padding="same",
                                name=f"{name}.expand3x3")(squeezed)
        return layers.Concatenate(axis=-1, name=name)([expand1, expand3])

    def __call__(self) -> tf.keras.models.Model:
        """ Create the SqueezeNet Model

        Returns
        -------
        :class:`keras.models.Model`
            The compiled SqueezeNet model
        """
        inputs = layers.Input(self._input_shape)
        var_x = layers.Conv2D(64, 3, strides=2, activation="relu", name="features.0")(inputs)

        block_idx = 2
        squeeze = 16
        expand = 64
        for idx in range(4):
            if idx < 3:
                var_x = layers.MaxPool2D(pool_size=3, strides=2)(var_x)
                block_idx += 1
            var_x = self._fire(var_x, squeeze, expand, block_idx)
            block_idx += 1
            var_x = self._fire(var_x, squeeze, expand, block_idx)
            block_idx += 1
            squeeze += 16
            expand += 64
        return Model(inputs=inputs, outputs=[var_x])
