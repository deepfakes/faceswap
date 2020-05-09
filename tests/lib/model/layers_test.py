#!/usr/bin/env python3
""" Tests for Faceswap Custom Layers.

Adapted from Keras tests.
"""


import pytest
import numpy as np
from keras import Input, Model, backend as K
from keras.utils.generic_utils import has_arg

from numpy.testing import assert_allclose

from lib.model import layers
from lib.utils import get_backend


CONV_SHAPE = (3, 3, 256, 2048)
CONV_ID = get_backend().upper()


def layer_test(layer_cls, kwargs={}, input_shape=None, input_dtype=None,
               input_data=None, expected_output=None,
               expected_output_dtype=None, fixed_batch_size=False):
    """Test routine for a layer with a single input tensor
    and single output tensor.
    """
    # generate input data
    if input_data is None:
        assert input_shape
        if not input_dtype:
            input_dtype = K.floatx()
        input_data_shape = list(input_shape)
        for i, var_e in enumerate(input_data_shape):
            if var_e is None:
                input_data_shape[i] = np.random.randint(1, 4)
        input_data = (10 * np.random.random(input_data_shape))
        input_data = input_data.astype(input_dtype)
    else:
        if input_shape is None:
            input_shape = input_data.shape
        if input_dtype is None:
            input_dtype = input_data.dtype
    if expected_output_dtype is None:
        expected_output_dtype = input_dtype

    # instantiation
    layer = layer_cls(**kwargs)

    # test get_weights , set_weights at layer level
    weights = layer.get_weights()
    layer.set_weights(weights)

    if isinstance(layer, layers.ReflectionPadding2D):
        layer.build(input_shape)
    expected_output_shape = layer.compute_output_shape(input_shape)

    # test in functional API
    if fixed_batch_size:
        inp = Input(batch_shape=input_shape, dtype=input_dtype)
    else:
        inp = Input(shape=input_shape[1:], dtype=input_dtype)
    outp = layer(inp)
    assert K.dtype(outp) == expected_output_dtype

    # check with the functional API
    model = Model(inp, outp)

    actual_output = model.predict(input_data)
    actual_output_shape = actual_output.shape
    for expected_dim, actual_dim in zip(expected_output_shape,
                                        actual_output_shape):
        if expected_dim is not None:
            assert expected_dim == actual_dim

    if expected_output is not None:
        assert_allclose(actual_output, expected_output, rtol=1e-3)

    # test serialization, weight setting at model level
    model_config = model.get_config()
    recovered_model = model.__class__.from_config(model_config)
    if model.weights:
        weights = model.get_weights()
        recovered_model.set_weights(weights)
        _output = recovered_model.predict(input_data)
        assert_allclose(_output, actual_output, rtol=1e-3)

    # test training mode (e.g. useful when the layer has a
    # different behavior at training and testing time).
    if has_arg(layer.call, 'training'):
        model.compile('rmsprop', 'mse')
        model.train_on_batch(input_data, actual_output)

    # test instantiation from layer config
    layer_config = layer.get_config()
    layer_config['batch_input_shape'] = input_shape
    layer = layer.__class__.from_config(layer_config)

    # for further checks in the caller function
    return actual_output


@pytest.mark.parametrize('dummy', [None], ids=[get_backend().upper()])
def test_pixel_shuffler(dummy):  # pylint:disable=unused-argument
    """ Pixel Shuffler layer test """
    layer_test(layers.PixelShuffler, input_shape=(2, 4, 4, 1024))


@pytest.mark.skipif(get_backend() == "amd", reason="amd does not support this layer")
@pytest.mark.parametrize('dummy', [None], ids=[get_backend().upper()])
def test_subpixel_upscaling(dummy):  # pylint:disable=unused-argument
    """ Sub Pixel Upscaling layer test """
    layer_test(layers.SubPixelUpscaling, input_shape=(2, 4, 4, 1024))


@pytest.mark.parametrize('dummy', [None], ids=[get_backend().upper()])
def test_reflection_padding_2d(dummy):  # pylint:disable=unused-argument
    """ Reflection Padding 2D layer test """
    layer_test(layers.ReflectionPadding2D, input_shape=(2, 4, 4, 512))


@pytest.mark.parametrize('dummy', [None], ids=[get_backend().upper()])
def test_global_min_pooling_2d(dummy):  # pylint:disable=unused-argument
    """ Global Min Pooling 2D layer test """
    layer_test(layers.GlobalMinPooling2D, input_shape=(2, 4, 4, 1024))


@pytest.mark.parametrize('dummy', [None], ids=[get_backend().upper()])
def test_global_std_pooling_2d(dummy):  # pylint:disable=unused-argument
    """ Global Standard Deviation Pooling 2D layer test """
    layer_test(layers.GlobalStdDevPooling2D, input_shape=(2, 4, 4, 1024))


@pytest.mark.parametrize('dummy', [None], ids=[get_backend().upper()])
def test_l2_normalize(dummy):  # pylint:disable=unused-argument
    """ L2 Normalize layer test """
    layer_test(layers.L2_normalize, kwargs={"axis": 1}, input_shape=(2, 4, 4, 1024))
