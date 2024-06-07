#!/usr/bin/env python3
""" Tests for Faceswap Initializers.

Adapted from Keras tests.
"""

import pytest
import numpy as np

from keras import device, initializers as k_initializers, Variable

from lib.model import initializers
from lib.utils import get_backend

CONV_SHAPE = (3, 3, 256, 2048)
CONV_ID = get_backend().upper()


def _runner(init, shape, target_mean=None, target_std=None,
            target_max=None, target_min=None):
    with device("cpu"):
        variable = Variable(init(shape))
    output = variable.numpy()
    lim = 3e-2
    if target_std is not None:
        assert abs(output.std() - target_std) < lim
    if target_mean is not None:
        assert abs(output.mean() - target_mean) < lim
    if target_max is not None:
        assert abs(output.max() - target_max) < lim
    if target_min is not None:
        assert abs(output.min() - target_min) < lim


@pytest.mark.parametrize('tensor_shape', [CONV_SHAPE], ids=[CONV_ID])
def test_icnr(tensor_shape):
    """ ICNR Initialization Test

    Parameters
    ----------
    tensor_shape: tuple
        The shape of the tensor to feed to the initializer
    """
    with device("cpu"):
        fan_in, _ = initializers.compute_fans(tensor_shape)
        std = np.sqrt(2. / fan_in)
        _runner(initializers.ICNR(initializer=k_initializers.he_uniform(), scale=2),
                tensor_shape,
                target_mean=0,
                target_std=std)


@pytest.mark.parametrize('tensor_shape', [CONV_SHAPE], ids=[CONV_ID])
def test_convolution_aware(tensor_shape):
    """ Convolution Aware Initialization Test

    Parameters
    ----------
    tensor_shape: tuple
        The shape of the tensor to feed to the initializer
    """
    with device("cpu"):
        fan_in, _ = initializers.compute_fans(tensor_shape)
        std = np.sqrt(2. / fan_in)
        _runner(initializers.ConvolutionAware(seed=123), tensor_shape,
                target_mean=0, target_std=std)
