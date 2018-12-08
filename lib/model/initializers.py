#!/usr/bin/env python3
""" Custom Initializers for faceswap.py
    Initializers from:
        shoanlu GAN: https://github.com/shaoanlu/faceswap-GAN"""

import tensorflow as tf


def icnr_keras(shape, dtype=None):
    """
    Custom initializer for subpix upscaling
    From https://github.com/kostyaev/ICNR
    Note: upscale factor is fixed to 2, and the base initializer is fixed to random normal.
    """
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
