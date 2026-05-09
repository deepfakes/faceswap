#!/usr/bin/env python3
""" Tests for Faceswap Feature Losses. Adapted from Keras tests. """
import pytest
import numpy as np
import torch

# pylint:disable=import-error,duplicate-code
from lib.model.losses.perceptual_loss import GMSDLoss, SSIMLoss, MSSIMLoss
from lib.model.losses.flip import LDRFLIPLoss
from lib.utils import get_backend


_PARAMS = [SSIMLoss, GMSDLoss, LDRFLIPLoss, MSSIMLoss]
_IDS = [f"{x.__name__}[{get_backend().upper()}]" for x in _PARAMS]


@pytest.mark.parametrize("loss_func", _PARAMS, ids=_IDS)
def test_loss_output(loss_func):
    """ Basic dtype and value tests for loss functions. """
    y_a = torch.Tensor(np.random.random((2, 3, 128, 128))).cpu()
    y_b = torch.Tensor(np.random.random((2, 3, 128, 128))).cpu()
    metric = loss_func().cpu()
    objective_output = metric(y_a, y_b)
    output = objective_output.detach().numpy()  # type:ignore
    assert output.dtype == "float32" and not np.any(np.isnan(output))
    assert output.mean() <= 1.0
