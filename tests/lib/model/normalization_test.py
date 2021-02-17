#!/usr/bin/env python3
""" Tests for Faceswap Normalization.

Adapted from Keras tests.
"""
from itertools import product

from keras import regularizers
import pytest

from lib.model import normalization
from lib.utils import get_backend

from tests.lib.model.layers_test import layer_test


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


_PARAMS = ["center", "scale"]
_VALUES = list(product([True, False], repeat=len(_PARAMS)))
_IDS = ["{}[{}]".format("|".join([_PARAMS[idx] for idx, b in enumerate(v) if b]),
                        get_backend().upper()) for v in _VALUES]


@pytest.mark.parametrize(_PARAMS, _VALUES, ids=_IDS)
def test_layer_normalization(center, scale):  # pylint:disable=unused-argument
    """ Basic test for layer normalization. """
    layer_test(normalization.LayerNormalization,
               kwargs={"center": center, "scale": scale},
               input_shape=(4, 512))


_PARAMS = ["partial", "bias"]
_VALUES = [(0.0, False), (0.25, False), (0.5, True), (0.75, False), (1.0, True)]
_IDS = ["partial={}|bias={}[{}]".format(v[0], v[1], get_backend().upper())
        for v in _VALUES]


@pytest.mark.parametrize(_PARAMS, _VALUES, ids=_IDS)
def test_rms_normalization(partial, bias):  # pylint:disable=unused-argument
    """ Basic test for RMS Layer normalization. """
    layer_test(normalization.RMSNormalization,
               kwargs={"partial": partial, "bias": bias},
               input_shape=(4, 512))
