#!/usr/bin/env python3
""" Tests for Faceswap Losses.

Adapted from Keras tests.
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose

from keras import backend as K
from keras.layers import Conv2D
from keras.models import Sequential
from keras.optimizers import Adam

from lib.model import losses
from lib.utils import get_backend


_PARAMS = [(losses.gradient_loss, (1, 5, 6, 7), (1, 5, 6)),
           (losses.generalized_loss, (5, 6, 7), (5, 6)),
           # TODO Make sure these output dimensions are correct
           (losses.l_inf_norm, (1, 5, 6, 7), (1, 1, 1)),
           # TODO Make sure these output dimensions are correct
           (losses.gmsd_loss, (1, 5, 6, 7), (1, 1, 1))]
_IDS = ["gradient_loss", "generalized_loss", "l_inf_norm", "gmsd_loss"]
_IDS = ["{}[{}]".format(loss, get_backend().upper()) for loss in _IDS]


@pytest.mark.parametrize(["loss_func", "input_shape", "output_shape"], _PARAMS, ids=_IDS)
def test_objective_shapes(loss_func, input_shape, output_shape):
    """ Basic shape tests for loss functions. """
    y_a = K.variable(np.random.random(input_shape))
    y_b = K.variable(np.random.random(input_shape))
    objective_output = loss_func(y_a, y_b)
    assert K.eval(objective_output).shape == output_shape


@pytest.mark.parametrize('dummy', [None], ids=[get_backend().upper()])
@pytest.mark.xfail(get_backend() == "amd", reason="plaidML generates NaNs")
def test_dssim_channels_last(dummy):  # pylint:disable=unused-argument
    """ Basic test for DSSIM Loss """
    prev_data = K.image_data_format()
    K.set_image_data_format('channels_last')
    for input_dim, kernel_size in zip([32, 33], [2, 3]):
        input_shape = [input_dim, input_dim, 3]
        var_x = np.random.random_sample(4 * input_dim * input_dim * 3)
        var_x = var_x.reshape([4] + input_shape)
        var_y = np.random.random_sample(4 * input_dim * input_dim * 3)
        var_y = var_y.reshape([4] + input_shape)

        model = Sequential()
        model.add(Conv2D(32, (3, 3), padding='same', input_shape=input_shape,
                         activation='relu'))
        model.add(Conv2D(3, (3, 3), padding='same', input_shape=input_shape,
                         activation='relu'))
        adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
        model.compile(loss=losses.DSSIMObjective(kernel_size=kernel_size),
                      metrics=['mse'],
                      optimizer=adam)
        model.fit(var_x, var_y, batch_size=2, epochs=1, shuffle='batch')

        # Test same
        x_1 = K.constant(var_x, 'float32')
        x_2 = K.constant(var_x, 'float32')
        dssim = losses.DSSIMObjective(kernel_size=kernel_size)
        assert_allclose(0.0, K.eval(dssim(x_1, x_2)), atol=1e-4)

        # Test opposite
        x_1 = K.zeros([4] + input_shape)
        x_2 = K.ones([4] + input_shape)
        dssim = losses.DSSIMObjective(kernel_size=kernel_size)
        assert_allclose(0.5, K.eval(dssim(x_1, x_2)), atol=1e-4)

    K.set_image_data_format(prev_data)
