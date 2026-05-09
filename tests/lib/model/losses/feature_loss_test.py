#!/usr/bin/env python3
""" Tests for Faceswap Feature Losses. Adapted from Keras tests. """
import pytest
import numpy as np
import torch

# pylint:disable=import-error
from lib.model.losses.feature_loss import LPIPSLoss
from lib.utils import get_backend


_NETS = ("alex", "squeeze", "vgg16")
_IDS = [f"LPIPS_{x}[{get_backend().upper()}]" for x in _NETS]


@pytest.mark.parametrize("net", _NETS, ids=_IDS)
def test_loss_output(net):
    """ Basic dtype and value tests for loss functions. """
    y_a = torch.Tensor(np.random.random((2, 3, 32, 32))).cpu()
    y_b = torch.Tensor(np.random.random((2, 3, 32, 32))).cpu()
    lpips = LPIPSLoss(net).cpu()
    objective_output = lpips(y_a, y_b)
    output = objective_output.detach().numpy()  # type:ignore
    assert output.dtype == "float32" and not np.any(np.isnan(output))
    assert output.mean() <= 0.1  # LPIPS loss is reduced 10x
