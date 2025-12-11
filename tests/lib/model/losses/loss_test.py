#!/usr/bin/env python3
""" Tests for Faceswap Losses.

Adapted from Keras tests.
"""

import pytest
import numpy as np

from keras import device, losses as k_losses, Variable

from lib.model.losses.loss import (FocalFrequencyLoss, GeneralizedLoss, GradientLoss,
                                   LaplacianPyramidLoss, LInfNorm, LossWrapper)
from lib.model.losses.feature_loss import LPIPSLoss
from lib.model.losses.perceptual_loss import DSSIMObjective, GMSDLoss, LDRFLIPLoss, MSSIMLoss

from lib.utils import get_backend


_PARAMS = ((FocalFrequencyLoss, 1.0),
           (GeneralizedLoss, 1.0),
           (GradientLoss, 200.0),
           (LaplacianPyramidLoss, 1.0),
           (LInfNorm, 1.0))
_IDS = [f"{x[0].__name__}[{get_backend().upper()}]" for x in _PARAMS]


@pytest.mark.parametrize(["loss_func", "max_target"], _PARAMS, ids=_IDS)
def test_loss_output(loss_func, max_target):
    """ Basic dtype and value tests for loss functions. """
    with device("cpu"):
        y_a = Variable(np.random.random((2, 32, 32, 3)))
        y_b = Variable(np.random.random((2, 32, 32, 3)))
        objective_output = loss_func()(y_a, y_b)
    output = objective_output.detach().numpy()
    assert output.dtype == "float32" and not np.any(np.isnan(output))
    assert output < max_target


_LWPARAMS = [(FocalFrequencyLoss, ()),
             (GeneralizedLoss, ()),
             (GradientLoss, ()),
             (LaplacianPyramidLoss, ()),
             (LInfNorm, ()),
             (LPIPSLoss, ("squeeze", )),
             (DSSIMObjective, ()),
             (GMSDLoss, ()),
             (LDRFLIPLoss, ()),
             (MSSIMLoss, ()),
             (k_losses.LogCosh, ()),
             (k_losses.MeanAbsoluteError, ()),
             (k_losses.MeanSquaredError, ())]
_LWIDS = [f"{x[0].__name__}[{get_backend().upper()}]" for x in _LWPARAMS]


@pytest.mark.parametrize(["loss_func", "func_args"], _LWPARAMS, ids=_LWIDS)
def test_loss_wrapper(loss_func, func_args):
    """ Test penalized loss wrapper works as expected """
    with device("cpu"):
        p_loss = LossWrapper()
        p_loss.add_loss(loss_func(*func_args), 1.0, -1)
        p_loss.add_loss(k_losses.MeanSquaredError(), 2.0, 3)
        y_a = Variable(np.random.random((2, 32, 32, 4)))
        y_b = Variable(np.random.random((2, 32, 32, 3)))

        output = p_loss(y_a, y_b)
    output = output.detach().numpy()  # type:ignore
    assert output.dtype == "float32" and not np.any(np.isnan(output))
