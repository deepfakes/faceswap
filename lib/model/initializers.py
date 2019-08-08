#!/usr/bin/env python3
""" Custom Initializers for faceswap.py
    Initializers from:
        shoanlu GAN: https://github.com/shaoanlu/faceswap-GAN"""

import logging
import sys
import inspect

import numpy as np
import tensorflow as tf
from keras import backend as K
from keras import initializers
from keras.utils.generic_utils import get_custom_objects

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def icnr_keras(shape, dtype=None):
    """
    Custom initializer for subpix upscaling
    From https://github.com/kostyaev/ICNR
    Note: upscale factor is fixed to 2, and the base initializer is fixed to random normal.
    """
    # TODO Roll this into ICNR_init when porting GAN 2.2
    shape = list(shape)
    scale = 2
    initializer = tf.keras.initializers.RandomNormal(0, 0.02)

    new_shape = shape[:3] + [int(shape[3] / (scale ** 2))]
    var_x = initializer(new_shape, dtype)
    var_x = tf.transpose(var_x, perm=[2, 0, 1, 3])
    var_x = tf.image.resize_nearest_neighbor(var_x, size=(shape[0] * scale, shape[1] * scale))
    var_x = tf.space_to_depth(var_x, block_size=scale)
    var_x = tf.transpose(var_x, perm=[1, 2, 0, 3])
    return var_x


class ICNR(initializers.Initializer):  # pylint: disable=invalid-name
    '''
    ICNR initializer for checkerboard artifact free sub pixel convolution

    Andrew Aitken et al. Checkerboard artifact free sub-pixel convolution
    https://arxiv.org/pdf/1707.02937.pdf	https://distill.pub/2016/deconv-checkerboard/

    Parameters:
        initializer: initializer used for sub kernels (orthogonal, glorot uniform, etc.)
        scale: scale factor of sub pixel convolution (upsampling from 8x8 to 16x16 is scale 2)
    Return:
        The modified kernel weights
    Example:
        x = conv2d(... weights_initializer=ICNR(initializer=he_uniform(), scale=2))
    '''

    def __init__(self, initializer, scale=2):
        self.scale = scale
        self.initializer = initializer

    def __call__(self, shape, dtype='float32'):  # tf needs partition_info=None
        shape = list(shape)
        if self.scale == 1:
            return self.initializer(shape)
        new_shape = shape[:3] + [shape[3] // (self.scale ** 2)]
        if isinstance(self.initializer, dict):
            self.initializer = initializers.deserialize(self.initializer)
        var_x = self.initializer(new_shape, dtype)
        var_x = tf.transpose(var_x, perm=[2, 0, 1, 3])
        var_x = tf.image.resize_nearest_neighbor(
            var_x,
            size=(shape[0] * self.scale, shape[1] * self.scale),
            align_corners=True)
        var_x = tf.space_to_depth(var_x, block_size=self.scale, data_format='NHWC')
        var_x = tf.transpose(var_x, perm=[1, 2, 0, 3])
        return var_x

    def get_config(self):
        config = {'scale': self.scale,
                  'initializer': self.initializer
                  }
        base_config = super(ICNR, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class ConvolutionAware(initializers.Initializer):
    """
    Initializer that generates orthogonal convolution filters in the fourier
    space. If this initializer is passed a shape that is not 3D or 4D,
    orthogonal initialization will be used.
    # Arguments
        eps_std: Standard deviation for the random normal noise used to break
        symmetry in the inverse fourier transform.
        seed: A Python integer. Used to seed the random generator.
    # References
        Armen Aghajanyan, https://arxiv.org/abs/1702.06295
    # Adapted and fixed from:
    https://github.com/keras-team/keras-contrib/blob/master/keras_contrib/initializers/convaware.py
    """

    def __init__(self, eps_std=0.05, seed=None, init=False):
        # Convolutional Aware Initialization takes a long time.
        # Keras model loading loads a model, performs initialization and then
        # loads weights, which is an unnecessary waste of time.
        # init defaults to False so that this is bypassed when loading a saved model
        # passing zeros
        self._init = init
        self.eps_std = eps_std
        self.seed = seed
        self.orthogonal = initializers.Orthogonal()
        self.he_uniform = initializers.he_uniform()

    def __call__(self, shape, dtype=None):
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
        init = []
        for _ in range(filters_size):
            basis = self._create_basis(
                stack_size, np.prod(kernel_fourier_shape), dtype)
            basis = basis.reshape((stack_size,) + kernel_fourier_shape)

            filters = [correct_ifft(x, kernel_shape) +
                       np.random.normal(0, self.eps_std, kernel_shape) for
                       x in basis]

            init.append(filters)

        # Format of array is now: filters, stack, row, column
        init = np.array(init)
        init = self._scale_filters(init, variance)
        return K.variable(init.transpose(transpose_dimensions), dtype=dtype, name="conv_aware")

    def _create_basis(self, filters, size, dtype):
        if size == 1:
            return np.random.normal(0.0, self.eps_std, (filters, size))

        nbb = filters // size + 1
        lst = []
        for _ in range(nbb):
            var_a = np.random.normal(0.0, 1.0, (size, size))
            var_a = self._symmetrize(var_a)
            var_u, _, _ = np.linalg.svd(var_a)
            lst.extend(var_u.T.tolist())
        var_p = np.array(lst[:filters], dtype=dtype)
        return var_p

    @staticmethod
    def _symmetrize(var_a):
        return var_a + var_a.T - np.diag(var_a.diagonal())

    @staticmethod
    def _scale_filters(filters, variance):
        c_var = np.var(filters)
        var_p = np.sqrt(variance / c_var)
        return filters * var_p

    def get_config(self):
        return {
            'eps_std': self.eps_std,
            'seed': self.seed
        }


# Update initializers into Keras custom objects
for name, obj in inspect.getmembers(sys.modules[__name__]):
    if inspect.isclass(obj) and obj.__module__ == __name__:
        get_custom_objects().update({name: obj})
