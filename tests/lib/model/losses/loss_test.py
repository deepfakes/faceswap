#!/usr/bin/env python3
""" Tests for Faceswap Losses.

Adapted from Keras tests.
"""

import pytest
import numpy as np

import torch

from lib.model.losses.loss import (FocalFrequencyLoss, GeneralizedLoss, GradientLoss,
                                   LaplacianPyramidLoss, LInfNorm)

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
    y_a = torch.Tensor(np.random.random((2, 3, 32, 32))).cpu()
    y_b = torch.Tensor(np.random.random((2, 3, 32, 32))).cpu()
    metric = loss_func().cpu()
    objective_output = metric(y_a, y_b)
    output = objective_output.detach().numpy()
    assert output.dtype == "float32" and not np.any(np.isnan(output))
    assert output.mean() <= max_target
