#!/usr/bin/env python3
""" Tests for Faceswap Losses.

Adapted from Keras tests.
"""

import pytest
import numpy as np

from lib.model import losses
from lib.utils import get_backend

if get_backend() == "amd":
    from keras import backend as K, losses as k_losses
else:
    # Ignore linting errors from Tensorflow's thoroughly broken import system
    from tensorflow.keras import backend as K, losses as k_losses  # pylint:disable=import-error

_PARAMS = [(losses.GeneralizedLoss(), (2, 16, 16)),
           (losses.GradientLoss(), (2, 16, 16)),
           # TODO Make sure these output dimensions are correct
           (losses.GMSDLoss(), (2, 1, 1)),
           # TODO Make sure these output dimensions are correct
           (losses.LInfNorm(), (2, 1, 1))]
_IDS = ["GeneralizedLoss", "GradientLoss", "GMSDLoss", "LInfNorm"]
_IDS = [f"{loss}[{get_backend().upper()}]" for loss in _IDS]


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
        assert output.dtype == "float32" and not np.any(np.isnan(output))


_LWPARAMS = [losses.GeneralizedLoss(), losses.GradientLoss(), losses.GMSDLoss(),
             losses.LInfNorm(), k_losses.mean_absolute_error, k_losses.mean_squared_error,
             k_losses.logcosh, losses.DSSIMObjective(), losses.MSSIMLoss()]
_LWIDS = ["GeneralizedLoss", "GradientLoss", "GMSDLoss", "LInfNorm", "mae", "mse", "logcosh",
          "DSSIMObjective", "MS-SSIM"]
_LWIDS = [f"{loss}[{get_backend().upper()}]" for loss in _LWIDS]


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
        assert output.dtype == "float32" and not np.any(np.isnan(output))
