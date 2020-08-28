#!/usr/bin/env python3
""" Tests for Faceswap Losses.

Adapted from Keras tests.
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose

from keras import backend as K
from keras import losses as k_losses
from keras.layers import Conv2D
from keras.models import Sequential
from keras.optimizers import Adam

from lib.model import losses
from lib.utils import get_backend


_PARAMS = [(losses.GeneralizedLoss(), (2, 16, 16)),
           (losses.GradientLoss(), (2, 16, 16)),
           # TODO Make sure these output dimensions are correct
           (losses.GMSDLoss(), (2, 1, 1)),
           # TODO Make sure these output dimensions are correct
           (losses.LInfNorm(), (2, 1, 1))]
_IDS = ["GeneralizedLoss", "GradientLoss", "GMSDLoss", "LInfNorm"]
_IDS = ["{}[{}]".format(loss, get_backend().upper()) for loss in _IDS]


@pytest.mark.parametrize(["loss_func", "output_shape"], _PARAMS, ids=_IDS)
def test_loss_output(loss_func, output_shape):
    """ Basic shape tests for loss functions. """
    if get_backend() == "amd" and isinstance(loss_func, losses.GMSDLoss):
        pytest.skip("GMSD Loss is not currently compatible with PlaidML")
    y_a = K.variable(np.random.random((2, 16, 16, 3)))
    y_b = K.variable(np.random.random((2, 16, 16, 3)))
    objective_output = loss_func(y_a, y_b)
    if get_backend() == "amd":
        assert K.eval(objective_output).shape == output_shape
    else:
        output = objective_output.numpy()
        assert output.dtype == "float32" and not np.isnan(output)


_LWPARAMS = [losses.GeneralizedLoss(), losses.GradientLoss(), losses.GMSDLoss(),
             losses.LInfNorm(), k_losses.mean_absolute_error, k_losses.mean_squared_error,
             k_losses.logcosh, losses.DSSIMObjective()]
_LWIDS = ["GeneralizedLoss", "GradientLoss", "GMSDLoss", "LInfNorm", "mae", "mse", "logcosh",
          "DSSIMObjective"]
_LWIDS = ["{}[{}]".format(loss, get_backend().upper()) for loss in _LWIDS]


@pytest.mark.parametrize("loss_func", _LWPARAMS, ids=_LWIDS)
def test_loss_wrapper(loss_func):
    """ Test penalized loss wrapper works as expected """
    if get_backend() == "amd":
        if isinstance(loss_func, losses.GMSDLoss):
            pytest.skip("GMSD Loss is not currently compatible with PlaidML")
        if hasattr(loss_func, "__name__") and loss_func.__name__ == "logcosh":
            pytest.skip("LogCosh Loss is not currently compatible with PlaidML")
    y_a = K.variable(np.random.random((2, 16, 16, 4)))
    y_b = K.variable(np.random.random((2, 16, 16, 3)))
    p_loss = losses.LossWrapper()
    p_loss.add_loss(loss_func, 1.0, -1)
    p_loss.add_loss(k_losses.mean_squared_error, 2.0, 3)
    output = p_loss(y_a, y_b)
    if get_backend() == "amd":
        assert K.dtype(output) == "float32" and K.eval(output).shape == (2, )
    else:
        output = output.numpy()
        assert output.dtype == "float32" and not np.isnan(output)


@pytest.mark.parametrize('dummy', [None], ids=[get_backend().upper()])
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
