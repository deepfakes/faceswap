#!/usr/bin/env python3
""" Custom Initializers for faceswap.py """

import logging
import sys
import inspect

import numpy as np
import tensorflow as tf
from keras import backend as K
from keras import initializers
from keras.utils.generic_utils import get_custom_objects

from lib.utils import get_backend

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class ICNR(initializers.Initializer):  # pylint: disable=invalid-name
    """ ICNR initializer for checkerboard artifact free sub pixel convolution

    Parameters
    ----------
    initializer: :class:`keras.initializers.Initializer`
        The initializer used for sub kernels (orthogonal, glorot uniform, etc.)
    scale: int
        scaling factor of sub pixel convolution (up sampling from 8x8 to 16x16 is scale 2)

    Returns
    -------
    tensor
        The modified kernel weights

    Example
    -------
    >>> x = conv2d(... weights_initializer=ICNR(initializer=he_uniform(), scale=2))

    References
    ----------
    Andrew Aitken et al. Checkerboard artifact free sub-pixel convolution
    https://arxiv.org/pdf/1707.02937.pdf,  https://distill.pub/2016/deconv-checkerboard/
    """

    def __init__(self, initializer, scale=2):
        self.scale = scale
        self.initializer = initializer

    def __call__(self, shape, dtype="float32"):
        """ Call function for the ICNR initializer.

        Parameters
        ----------
        shape: tuple or list
            The required resized shape for the output tensor
        dtype: str
            The data type for the tensor

        Returns
        -------
        tensor
            The modified kernel weights
        """
        shape = list(shape)
        if self.scale == 1:
            return self.initializer(shape)
        new_shape = shape[:3] + [shape[3] // (self.scale ** 2)]
        if isinstance(self.initializer, dict):
            self.initializer = initializers.deserialize(self.initializer)
        var_x = self.initializer(new_shape, dtype)
        var_x = K.permute_dimensions(var_x, [2, 0, 1, 3])
        var_x = self._resize_nearest_neighbour(var_x,
                                               (shape[0] * self.scale, shape[1] * self.scale))
        var_x = self._space_to_depth(var_x)
        var_x = K.permute_dimensions(var_x, [1, 2, 0, 3])
        logger.debug("Output: %s", var_x)
        return var_x

    def _resize_nearest_neighbour(self, input_tensor, size):
        """ Resize a tensor using nearest neighbor interpolation.

        Notes
        -----
        Tensorflow has a bug that resizes the image incorrectly if :attr:`align_corners` is not set
        to ``True``. Keras Backend does not set this flag, so we explicitly call the Tensorflow
        operation for non-amd backends.

        Parameters
        ----------
        input_tensor: tensor
            The tensor to be resized
        tuple: int
            The (`h`, `w`) that the tensor should be resized to (used for non-amd backends only)

        Returns
        -------
        tensor
            The input tensor resized to the given size
        """
        if get_backend() == "amd":
            retval = K.resize_images(input_tensor, self.scale, self.scale, "channels_last",
                                     interpolation="nearest")
        else:
            retval = tf.image.resize_nearest_neighbor(input_tensor, size=size, align_corners=True)
        logger.debug("Input Tensor: %s, Output Tensor: %s", input_tensor, retval)
        return retval

    def _space_to_depth(self, input_tensor):
        """ Space to depth implementation.

        PlaidML does not have a space to depth operation, so calculate if backend is amd
        otherwise returns the :func:`tensorflow.space_to_depth` operation.

        Parameters
        ----------
        input_tensor: tensor
            The tensor to be manipulated

        Returns
        -------
        tensor
            The manipulated input tensor
        """
        if get_backend() == "amd":
            batch, height, width, depth = input_tensor.shape.dims
            new_height = height // self.scale
            new_width = width // self.scale
            reshaped = K.reshape(input_tensor,
                                 (batch, new_height, self.scale, new_width, self.scale, depth))
            retval = K.reshape(K.permute_dimensions(reshaped, [0, 1, 3, 2, 4, 5]),
                               (batch, new_height, new_width, -1))
        else:
            retval = tf.space_to_depth(input_tensor, block_size=self.scale, data_format="NHWC")
        logger.debug("Input Tensor: %s, Output Tensor: %s", input_tensor, retval)
        return retval

    def get_config(self):
        """ Return the ICNR Initializer configuration.

        Returns
        -------
        dict
            The configuration for ICNR Initialization
        """
        config = {"scale": self.scale,
                  "initializer": self.initializer
                  }
        base_config = super(ICNR, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class ConvolutionAware(initializers.Initializer):
    """
    Initializer that generates orthogonal convolution filters in the Fourier space. If this
    initializer is passed a shape that is not 3D or 4D, orthogonal initialization will be used.

    Adapted, fixed and optimized from:
    https://github.com/keras-team/keras-contrib/blob/master/keras_contrib/initializers/convaware.py

    Parameters
    ----------
    eps_std: float
        The Standard deviation for the random normal noise used to break symmetry in the inverse
        Fourier transform.
    seed: int, optional
        Used to seed the random generator. Default: ``None``

    Returns
    -------
    tensor
        The modified kernel weights

    References
    ----------
    Armen Aghajanyan, https://arxiv.org/abs/1702.06295

    Notes
    -----
    Convolutional Aware Initialization takes a long time. Keras model loading loads a model,
    performs initialization and then loads weights, which is an unnecessary waste of time.
    init defaults to False so that this is bypassed when loading a saved model passing zeros.
    """

    def __init__(self, eps_std=0.05, seed=None, init=False):
        self._init = init
        self.eps_std = eps_std
        self.seed = seed
        self.orthogonal = initializers.Orthogonal()
        self.he_uniform = initializers.he_uniform()

    def __call__(self, shape, dtype=None):
        """ Call function for the ICNR initializer.

        Parameters
        ----------
        shape: tuple or list
            The required shape for the output tensor
        dtype: str
            The data type for the tensor

        Returns
        -------
        tensor
            The modified kernel weights
        """
        dtype = K.floatx() if dtype is None else dtype
        if self._init:
            logger.info("Calculating Convolution Aware Initializer for shape: %s", shape)
        else:
            logger.debug("Bypassing Convolutional Aware Initializer for saved model")
            # Dummy in he_uniform just in case there aren't any weighs being loaded
            # and it needs some kind of initialization
            return self.he_uniform(shape, dtype=dtype)

        rank = len(shape)
        if self.seed is not None:
            np.random.seed(self.seed)

        fan_in, _ = initializers._compute_fans(shape)  # pylint:disable=protected-access
        variance = 2 / fan_in

        if rank == 3:
            row, stack_size, filters_size = shape

            transpose_dimensions = (2, 1, 0)
            kernel_shape = (row,)
            correct_ifft = lambda shape, s=[None]: np.fft.irfft(shape, s[0])  # noqa
            correct_fft = np.fft.rfft

        elif rank == 4:
            row, column, stack_size, filters_size = shape

            transpose_dimensions = (2, 3, 1, 0)
            kernel_shape = (row, column)
            correct_ifft = np.fft.irfft2
            correct_fft = np.fft.rfft2

        elif rank == 5:
            var_x, var_y, var_z, stack_size, filters_size = shape

            transpose_dimensions = (3, 4, 0, 1, 2)
            kernel_shape = (var_x, var_y, var_z)
            correct_fft = np.fft.rfftn
            correct_ifft = np.fft.irfftn

        else:
            return K.variable(self.orthogonal(shape), dtype=dtype)

        kernel_fourier_shape = correct_fft(np.zeros(kernel_shape)).shape

        basis = self._create_basis(filters_size, stack_size, np.prod(kernel_fourier_shape), dtype)
        basis = basis.reshape((filters_size, stack_size,) + kernel_fourier_shape)
        randoms = np.random.normal(0, self.eps_std, basis.shape[:-2] + kernel_shape)
        init = correct_ifft(basis, kernel_shape) + randoms
        init = self._scale_filters(init, variance)
        return K.variable(init.transpose(transpose_dimensions), dtype=dtype, name="conv_aware")

    def _create_basis(self, filters_size, filters, size, dtype):
        """ Create the basis for convolutional aware initialization """
        if size == 1:
            return np.random.normal(0.0, self.eps_std, (filters_size, filters, size))
        nbb = filters // size + 1
        var_a = np.random.normal(0.0, 1.0, (filters_size, nbb, size, size))
        var_a = self._symmetrize(var_a)
        var_u = np.linalg.svd(var_a)[0].transpose(0, 1, 3, 2)
        var_p = np.reshape(var_u, (filters_size, nbb * size, size))[:, :filters, :].astype(dtype)
        return var_p

    @staticmethod
    def _symmetrize(var_a):
        """ Make the given tensor symmetrical. """
        var_b = np.transpose(var_a, axes=(0, 1, 3, 2))
        diag = var_a.diagonal(axis1=2, axis2=3)
        var_c = np.array([[np.diag(arr) for arr in batch] for batch in diag])
        return var_a + var_b - var_c

    @staticmethod
    def _scale_filters(filters, variance):
        """ Scale the given filters. """
        c_var = np.var(filters)
        var_p = np.sqrt(variance / c_var)
        return filters * var_p

    def get_config(self):
        """ Return the Convolutional Aware Initializer configuration.

        Returns
        -------
        dict
            The configuration for ICNR Initialization
        """
        return {
            "eps_std": self.eps_std,
            "seed": self.seed
        }


# Update initializers into Keras custom objects
for name, obj in inspect.getmembers(sys.modules[__name__]):
    if inspect.isclass(obj) and obj.__module__ == __name__:
        get_custom_objects().update({name: obj})
