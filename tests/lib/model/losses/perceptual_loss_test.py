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


_DEVICE_PARAMS = [(SSIMLoss, 64), (MSSIMLoss, 384)]
_DEVICE_IDS = [f"{x[0].__name__}[{get_backend().upper()}]" for x in _DEVICE_PARAMS]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires a CUDA device")
@pytest.mark.parametrize(["loss_func", "size"], _DEVICE_PARAMS, ids=_DEVICE_IDS)
def test_ssim_buffers_follow_input_device(loss_func, size):
    """ SSIM/MS-SSIM must work when the module itself has not been moved to the device that the
    images are on. The registered buffers (gaussian kernel and the MS-SSIM power factors) should
    be moved to the image device on the first call and remain registered buffers afterwards. """
    y_a = torch.rand((2, 3, size, size), device="cuda")
    y_b = torch.rand((2, 3, size, size), device="cuda")
    metric = loss_func()  # Deliberately not moved to the GPU
    assert all(buf.device.type == "cpu" for _, buf in metric.named_buffers())
    output = metric(y_a, y_b)
    assert output.device.type == "cuda"
    assert not torch.isnan(output).any()
    buffers = dict(metric.named_buffers())
    assert "_kernel" in buffers
    assert all(buf.device.type == "cuda" for buf in buffers.values())


def test_mssim_rebuilt_kernel_is_buffer():
    """ When MS-SSIM rebuilds its kernel for small images, the new kernel must still be a
    registered buffer so that it is moved along with the module. """
    metric = MSSIMLoss().cpu()
    original_size = dict(metric.named_buffers())["_kernel"].shape[-1]
    y_a = torch.rand((2, 3, 64, 64))
    y_b = torch.rand((2, 3, 64, 64))
    metric(y_a, y_b)
    buffers = dict(metric.named_buffers())
    assert "_kernel" in buffers
    assert buffers["_kernel"].shape[-1] == 3 < original_size
