#!/usr/bin/env python3
""" Tests for Faceswap Normalization.

Adapted from Keras tests.
"""

from keras import regularizers
import pytest

from lib.model import normalization
from lib.utils import get_backend

from .layers_test import layer_test


@pytest.mark.parametrize('dummy', [None], ids=[get_backend().upper()])
def test_instance_normalization(dummy):  # pylint:disable=unused-argument
    """ Basic test for instance normalization. """
    layer_test(normalization.InstanceNormalization,
               kwargs={'epsilon': 0.1,
                       'gamma_regularizer': regularizers.l2(0.01),
                       'beta_regularizer': regularizers.l2(0.01)},
               input_shape=(3, 4, 2))
    layer_test(normalization.InstanceNormalization,
               kwargs={'epsilon': 0.1,
                       'axis': 1},
               input_shape=(1, 4, 1))
    layer_test(normalization.InstanceNormalization,
               kwargs={'gamma_initializer': 'ones',
                       'beta_initializer': 'ones'},
               input_shape=(3, 4, 2, 4))
    layer_test(normalization.InstanceNormalization,
               kwargs={'epsilon': 0.1,
                       'axis': 1,
                       'scale': False,
                       'center': False},
               input_shape=(3, 4, 2, 4))
