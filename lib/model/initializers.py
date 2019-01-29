#!/usr/bin/env python3
""" Custom Initializers for faceswap.py
    Initializers from:
        shoanlu GAN: https://github.com/shaoanlu/faceswap-GAN"""

import sys
import inspect
import tensorflow as tf
from keras import initializers
from keras.utils.generic_utils import get_custom_objects


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
        if type(self.initializer) is dict:
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


# Update initializers into Keras custom objects
for name, obj in inspect.getmembers(sys.modules[__name__]):
    if inspect.isclass(obj) and obj.__module__ == __name__:
        get_custom_objects().update({name: obj})
