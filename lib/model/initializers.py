#!/usr/bin/env python3
""" Custom Initializers for faceswap.py """

import logging
import sys
import inspect

import numpy as np
import tensorflow as tf

# Fix intellisense/linting for tf.keras' thoroughly broken import system
keras = tf.keras
K = keras.backend


logger = logging.getLogger(__name__)


def compute_fans(shape, data_format='channels_last'):
    """Computes the number of input and output units for a weight shape.

    Ported directly from Keras as the location moves between keras and tensorflow-keras

    Parameters
    ----------
    shape: tuple
        shape tuple of integers
    data_format: str
        Image data format to use for convolution kernels. Note that all kernels in Keras are
        standardized on the `"channels_last"` ordering (even when inputs are set to
        `"channels_first"`).

    Returns
    -------
    tuple
            A tuple of scalars, `(fan_in, fan_out)`.

    Raises
    ------
    ValueError
        In case of invalid `data_format` argument.
    """
    if len(shape) == 2:
        fan_in = shape[0]
        fan_out = shape[1]
    elif len(shape) in {3, 4, 5}:
        # Assuming convolution kernels (1D, 2D or 3D).
        # Theano kernel shape: (depth, input_depth, ...)
        # Tensorflow kernel shape: (..., input_depth, depth)
        if data_format == 'channels_first':
            receptive_field_size = np.prod(shape[2:])
            fan_in = shape[1] * receptive_field_size
            fan_out = shape[0] * receptive_field_size
        elif data_format == 'channels_last':
            receptive_field_size = np.prod(shape[:-2])
            fan_in = shape[-2] * receptive_field_size
            fan_out = shape[-1] * receptive_field_size
        else:
            raise ValueError('Invalid data_format: ' + data_format)
    else:
        # No specific assumptions.
        fan_in = np.sqrt(np.prod(shape))
        fan_out = np.sqrt(np.prod(shape))
    return fan_in, fan_out


class ICNR(keras.initializers.Initializer):  # type:ignore[name-defined]
    """ ICNR initializer for checkerboard artifact free sub pixel convolution

    Parameters
    ----------
    initializer: :class:`keras.initializers.Initializer`
        The initializer used for sub kernels (orthogonal, glorot uniform, etc.)
    scale: int, optional
        scaling factor of sub pixel convolution (up sampling from 8x8 to 16x16 is scale 2).
        Default: `2`

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

    def __call__(self, shape, dtype="float32", **kwargs):
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
            self.initializer = keras.initializers.deserialize(self.initializer)
        var_x = self.initializer(new_shape, dtype)
        var_x = K.permute_dimensions(var_x, [2, 0, 1, 3])
        var_x = K.resize_images(var_x,
                                self.scale,
                                self.scale,
                                "channels_last",
                                interpolation="nearest")
        var_x = self._space_to_depth(var_x)
        var_x = K.permute_dimensions(var_x, [1, 2, 0, 3])
        logger.debug("Output shape: %s", var_x.shape)
        return var_x

    def _space_to_depth(self, input_tensor):
        """ Space to depth implementation.

        Parameters
        ----------
        input_tensor: tensor
            The tensor to be manipulated

        Returns
        -------
        tensor
            The manipulated input tensor
        """
        retval = tf.nn.space_to_depth(input_tensor, block_size=self.scale, data_format="NHWC")
        logger.debug("Input shape: %s, Output shape: %s", input_tensor.shape, retval.shape)
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
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


class ConvolutionAware(keras.initializers.Initializer):  # type:ignore[name-defined]
    """
    Initializer that generates orthogonal convolution filters in the Fourier space. If this
    initializer is passed a shape that is not 3D or 4D, orthogonal initialization will be used.

    Adapted, fixed and optimized from:
    https://github.com/keras-team/keras-contrib/blob/master/keras_contrib/initializers/convaware.py

    Parameters
    ----------
    eps_std: float, optional
        The Standard deviation for the random normal noise used to break symmetry in the inverse
        Fourier transform. Default: 0.05
    seed: int, optional
        Used to seed the random generator. Default: ``None``
    initialized: bool, optional
        This should always be set to ``False``. To avoid Keras re-calculating the values every time
        the model is loaded, this parameter is internally set on first time initialization.
        Default:``False``

    Returns
    -------
    tensor
        The modified kernel weights

    References
    ----------
    Armen Aghajanyan, https://arxiv.org/abs/1702.06295
    """

    def __init__(self, eps_std=0.05, seed=None, initialized=False):
        self.eps_std = eps_std
        self.seed = seed
        self.orthogonal = keras.initializers.Orthogonal()
        self.he_uniform = keras.initializers.he_uniform()
        self.initialized = initialized

    def __call__(self, shape, dtype=None, **kwargs):
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
        # TODO Tensorflow appears to pass in a :class:`tensorflow.python.framework.dtypes.DType`
        # object which causes this to error, so currently just reverts to default dtype if a string
        # is not passed in.
        if self.initialized:   # Avoid re-calculating initializer when loading a saved model
            return self.he_uniform(shape, dtype=dtype)
        dtype = K.floatx() if not isinstance(dtype, str) else dtype
        logger.info("Calculating Convolution Aware Initializer for shape: %s", shape)
        rank = len(shape)
        if self.seed is not None:
            np.random.seed(self.seed)

        fan_in, _ = compute_fans(shape)  # pylint:disable=protected-access
        variance = 2 / fan_in

        if rank == 3:
            row, stack_size, filters_size = shape

            transpose_dimensions = (2, 1, 0)
            kernel_shape = (row,)
            correct_ifft = lambda shape, s=[None]: np.fft.irfft(shape, s[0])  # noqa:E501,E731 # pylint:disable=unnecessary-lambda-assignment
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
            self.initialized = True
            return K.variable(self.orthogonal(shape), dtype=dtype)

        kernel_fourier_shape = correct_fft(np.zeros(kernel_shape)).shape

        basis = self._create_basis(filters_size, stack_size, np.prod(kernel_fourier_shape), dtype)
        basis = basis.reshape((filters_size, stack_size,) + kernel_fourier_shape)
        randoms = np.random.normal(0, self.eps_std, basis.shape[:-2] + kernel_shape)
        init = correct_ifft(basis, kernel_shape) + randoms
        init = self._scale_filters(init, variance)
        self.initialized = True
        return K.variable(init.transpose(transpose_dimensions), dtype=dtype, name="conv_aware")

    def _create_basis(self, filters_size, filters, size, dtype):
        """ Create the basis for convolutional aware initialization """
        logger.debug("filters_size: %s, filters: %s, size: %s, dtype: %s",
                     filters_size, filters, size, dtype)
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
        return {"eps_std": self.eps_std,
                "seed": self.seed,
                "initialized": self.initialized}


# Update initializers into Keras custom objects
for name, obj in inspect.getmembers(sys.modules[__name__]):
    if inspect.isclass(obj) and obj.__module__ == __name__:
        keras.utils.get_custom_objects().update({name: obj})
