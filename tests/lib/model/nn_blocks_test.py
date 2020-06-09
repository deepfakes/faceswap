#!/usr/bin/env python3
""" Tests for Faceswap Custom Layers.

Adapted from Keras tests.
"""

from itertools import product

import pytest
import numpy as np

from keras import Input, Model, backend as K
from numpy.testing import assert_allclose

from lib.model.nn_blocks import NNBlocks
from lib.utils import get_backend

_PARAMS = ["use_icnr_init", "use_convaware_init", "use_reflect_padding"]
_VALUES = list(product([True, False], repeat=len(_PARAMS)))
_IDS = ["{}[{}]".format("|".join([_PARAMS[idx] for idx, b in enumerate(v) if b]),
                        get_backend().upper()) for v in _VALUES]


def block_test(layer_func, kwargs={}, input_shape=None):
    """Test routine for faceswap neural network blocks.

    Tests are simple and are to ensure that the blocks compile on both tensorflow
    and plaidml backends
    """
    # generate input data
    assert input_shape
    input_dtype = K.floatx()
    input_data_shape = list(input_shape)
    for i, var_e in enumerate(input_data_shape):
        if var_e is None:
            input_data_shape[i] = np.random.randint(1, 4)
    input_data = (10 * np.random.random(input_data_shape))
    input_data = input_data.astype(input_dtype)
    expected_output_dtype = input_dtype

    # test in functional API
    inp = Input(shape=input_shape[1:], dtype=input_dtype)
    outp = layer_func(inp, **kwargs)
    assert K.dtype(outp) == expected_output_dtype

    # check with the functional API
    model = Model(inp, outp)

    actual_output = model.predict(input_data)

    # test serialization, weight setting at model level
    model_config = model.get_config()
    recovered_model = model.__class__.from_config(model_config)
    if model.weights:
        weights = model.get_weights()
        recovered_model.set_weights(weights)
        _output = recovered_model.predict(input_data)
        assert_allclose(_output, actual_output, rtol=1e-3)

    # for further checks in the caller function
    return actual_output


@pytest.mark.parametrize(_PARAMS, _VALUES, ids=_IDS)
def test_blocks(use_icnr_init, use_convaware_init, use_reflect_padding):
    """ Test for all blocks contained within the NNBlocks Class """
    cls_ = NNBlocks(use_icnr_init=use_icnr_init,
                    use_convaware_init=use_convaware_init,
                    use_reflect_padding=use_reflect_padding)
    block_test(cls_.conv2d, input_shape=(2, 5, 5, 128), kwargs=dict(filters=1024, kernel_size=3))
    block_test(cls_.conv, input_shape=(2, 8, 8, 32), kwargs=dict(filters=64))
    block_test(cls_.conv_sep, input_shape=(2, 8, 8, 32), kwargs=dict(filters=64))
    block_test(cls_.upscale, input_shape=(2, 4, 4, 128), kwargs=dict(filters=64))
    block_test(cls_.res_block, input_shape=(2, 2, 2, 64), kwargs=dict(filters=64))
    block_test(cls_.upscale2x, input_shape=(2, 4, 4, 128), kwargs=dict(filters=64, fast=False))
    block_test(cls_.upscale2x, input_shape=(2, 4, 4, 128), kwargs=dict(filters=64, fast=True))
