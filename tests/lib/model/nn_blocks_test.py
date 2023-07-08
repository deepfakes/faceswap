#!/usr/bin/env python3
""" Tests for Faceswap Custom Layers.

Adapted from Keras tests.
"""

from itertools import product

import pytest
import numpy as np

from numpy.testing import assert_allclose

# Ignore linting errors from Tensorflow's thoroughly broken import system
from tensorflow.keras import Input, Model, backend as K  # pylint:disable=import-error

from lib.model import nn_blocks
from lib.utils import get_backend


def block_test(layer_func, kwargs={}, input_shape=None):
    """Test routine for faceswap neural network blocks. """
    # generate input data
    assert input_shape
    input_dtype = K.floatx()
    input_data_shape = list(input_shape)
    for i, var_e in enumerate(input_data_shape):
        if var_e is None:
            input_data_shape[i] = np.random.randint(1, 4)
    input_data = 10 * np.random.random(input_data_shape)
    input_data = input_data.astype(input_dtype)
    expected_output_dtype = input_dtype

    # test in functional API
    inp = Input(shape=input_shape[1:], dtype=input_dtype)
    outp = layer_func(inp, **kwargs)
    assert K.dtype(outp) == expected_output_dtype

    # check with the functional API
    model = Model(inp, outp)

    actual_output = model.predict(input_data, verbose=0)

    # test serialization, weight setting at model level
    model_config = model.get_config()
    recovered_model = model.__class__.from_config(model_config)
    if model.weights:
        weights = model.get_weights()
        recovered_model.set_weights(weights)
        _output = recovered_model.predict(input_data, verbose=0)
        assert_allclose(_output, actual_output, rtol=1e-3)

    # for further checks in the caller function
    return actual_output


_PARAMS = ["use_icnr_init", "use_convaware_init", "use_reflect_padding"]
_VALUES = list(product([True, False], repeat=len(_PARAMS)))
_IDS = [f"{'|'.join([_PARAMS[idx] for idx, b in enumerate(v) if b])}[{get_backend().upper()}]"
        for v in _VALUES]


@pytest.mark.parametrize(_PARAMS, _VALUES, ids=_IDS)
def test_blocks(use_icnr_init, use_convaware_init, use_reflect_padding):
    """ Test for all blocks contained within the NNBlocks Class """
    config = {"icnr_init": use_icnr_init,
              "conv_aware_init": use_convaware_init,
              "reflect_padding": use_reflect_padding}
    nn_blocks.set_config(config)
    block_test(nn_blocks.Conv2DOutput(64, 3), input_shape=(2, 8, 8, 32))
    block_test(nn_blocks.Conv2DBlock(64), input_shape=(2, 8, 8, 32))
    block_test(nn_blocks.SeparableConv2DBlock(64), input_shape=(2, 8, 8, 32))
    block_test(nn_blocks.UpscaleBlock(64), input_shape=(2, 4, 4, 128))
    block_test(nn_blocks.Upscale2xBlock(64, fast=True), input_shape=(2, 4, 4, 128))
    block_test(nn_blocks.Upscale2xBlock(64, fast=False), input_shape=(2, 4, 4, 128))
    block_test(nn_blocks.ResidualBlock(64), input_shape=(2, 4, 4, 64))
