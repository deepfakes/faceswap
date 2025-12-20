#!/usr/bin/env python3
""" Tests for Faceswap Feature Losses. Adapted from Keras tests. """
import pytest
import numpy as np
from keras import device, Variable

# pylint:disable=import-error
from lib.model.losses.perceptual_loss import DSSIMObjective, GMSDLoss, LDRFLIPLoss, MSSIMLoss
from lib.utils import get_backend


_PARAMS = [DSSIMObjective, GMSDLoss, LDRFLIPLoss, MSSIMLoss]
_IDS = [f"{x.__name__}[{get_backend().upper()}]" for x in _PARAMS]


@pytest.mark.parametrize("loss_func", _PARAMS, ids=_IDS)
def test_loss_output(loss_func):
    """ Basic dtype and value tests for loss functions. """
    with device("cpu"):
        y_a = Variable(np.random.random((2, 32, 32, 3)))
        y_b = Variable(np.random.random((2, 32, 32, 3)))
        objective_output = loss_func()(y_a, y_b)
    output = objective_output.detach().numpy()  # type:ignore
    assert output.dtype == "float32" and not np.any(np.isnan(output))
    assert output < 1.0
