#!/usr/bin/env python3
""" Tests for Faceswap Losses.

Adapted from Keras tests.
"""

import pytest
import numpy as np

from keras import device, losses as k_losses, Variable


from lib.model import losses
from lib.utils import get_backend

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
    with device("cpu"):
        y_a = Variable(np.random.random((2, 16, 16, 3)))
        y_b = Variable(np.random.random((2, 16, 16, 3)))
        objective_output = loss_func(y_a, y_b)
        output = objective_output.detach().numpy()
    assert output.dtype == "float32" and not np.any(np.isnan(output))


with device("cpu"):
    _LWPARAMS = [losses.DSSIMObjective(),
                 losses.FocalFrequencyLoss(),
                 losses.GeneralizedLoss(),
                 losses.GMSDLoss(),
                 losses.GradientLoss(),
                 losses.LaplacianPyramidLoss(),
                 losses.LDRFLIPLoss(),
                 losses.LInfNorm(),
                 k_losses.LogCosh(),
                 k_losses.MeanAbsoluteError(),
                 k_losses.MeanSquaredError(),
                 losses.MSSIMLoss()]
_LWIDS = ["DSSIMObjective", "FocalFrequencyLoss", "GeneralizedLoss", "GMSDLoss", "GradientLoss",
          "LaplacianPyramidLoss", "LInfNorm", "LDRFlipLoss", "logcosh", "mae", "mse", "MS-SSIM"]
_LWIDS = [f"{loss}[{get_backend().upper()}]" for loss in _LWIDS]


@pytest.mark.parametrize("loss_func", _LWPARAMS, ids=_LWIDS)
def test_loss_wrapper(loss_func):
    """ Test penalized loss wrapper works as expected """
    with device("cpu"):
        p_loss = losses.LossWrapper()
        p_loss.add_loss(loss_func, 1.0, -1)
        p_loss.add_loss(k_losses.MeanSquaredError(), 2.0, 3)
        y_a = Variable(np.random.random((2, 64, 64, 4)))
        y_b = Variable(np.random.random((2, 64, 64, 3)))

        output = p_loss(y_a, y_b)
        output = output.detach().numpy()
    assert output.dtype == "float32" and not np.any(np.isnan(output))
