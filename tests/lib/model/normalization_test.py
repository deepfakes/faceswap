#!/usr/bin/env python3
""" Tests for Faceswap Normalization.

Adapted from Keras tests.
"""
from itertools import product

import numpy as np
import pytest

from tensorflow.keras import regularizers, models, layers  # noqa:E501  # pylint:disable=import-error

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


@pytest.mark.parametrize('dummy', [None], ids=[get_backend().upper()])
def test_group_normalization(dummy):  # pylint:disable=unused-argument
    """ Basic test for instance normalization. """
    layer_test(normalization.GroupNormalization,
               kwargs={'epsilon': 0.1,
                       'gamma_regularizer': regularizers.l2(0.01),
                       'beta_regularizer': regularizers.l2(0.01)},
               input_shape=(4, 3, 4, 128))
    layer_test(normalization.GroupNormalization,
               kwargs={'epsilon': 0.1,
                       'axis': 1},
               input_shape=(4, 1, 4, 256))
    layer_test(normalization.GroupNormalization,
               kwargs={'gamma_init': 'ones',
                       'beta_init': 'ones'},
               input_shape=(4, 64))
    layer_test(normalization.GroupNormalization,
               kwargs={'epsilon': 0.1,
                       'axis': 1,
                       'group': 16},
               input_shape=(3, 64))


_PARAMS = ["center", "scale"]
_VALUES = list(product([True, False], repeat=len(_PARAMS)))
_IDS = [f"{'|'.join([_PARAMS[idx] for idx, b in enumerate(v) if b])}[{get_backend().upper()}]"
        for v in _VALUES]


@pytest.mark.parametrize(_PARAMS, _VALUES, ids=_IDS)
def test_adain_normalization(center, scale):
    """ Basic test for Ada Instance Normalization. """
    norm = normalization.AdaInstanceNormalization(center=center, scale=scale)
    shapes = [(4, 8, 8, 1280), (4, 1, 1, 1280), (4, 1, 1, 1280)]
    norm.build(shapes)
    expected_output_shape = norm.compute_output_shape(shapes)
    inputs = [layers.Input(shape=shapes[0][1:]),
              layers.Input(shape=shapes[1][1:]),
              layers.Input(shape=shapes[2][1:])]
    model = models.Model(inputs, norm(inputs))
    data = [10 * np.random.random(shape) for shape in shapes]

    actual_output = model.predict(data, verbose=0)
    actual_output_shape = actual_output.shape

    for expected_dim, actual_dim in zip(expected_output_shape,
                                        actual_output_shape):
        if expected_dim is not None:
            assert expected_dim == actual_dim


_PARAMS = ["partial", "bias"]
_VALUES = [(0.0, False), (0.25, False), (0.5, True), (0.75, False), (1.0, True)]  # type:ignore
_IDS = [f"partial={v[0]}|bias={v[1]}[{get_backend().upper()}]" for v in _VALUES]


@pytest.mark.parametrize(_PARAMS, _VALUES, ids=_IDS)
def test_rms_normalization(partial, bias):  # pylint:disable=unused-argument
    """ Basic test for RMS Layer normalization. """
    layer_test(normalization.RMSNormalization,
               kwargs={"partial": partial, "bias": bias},
               input_shape=(4, 512))
