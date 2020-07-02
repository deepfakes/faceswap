#!/usr/bin/env python3
""" Tests for Faceswap Initializers.

Adapted from Keras tests.
"""
import pytest

from keras import optimizers as k_optimizers
from keras.layers import Dense, Activation
from keras.models import Sequential
from keras.utils import test_utils
from keras.utils.np_utils import to_categorical
import numpy as np
from numpy.testing import assert_allclose

from lib.model import optimizers
from lib.utils import get_backend


def get_test_data():
    """ Obtain radomized test data for training """
    np.random.seed(1337)
    (x_train, y_train), _ = test_utils.get_test_data(num_train=1000,
                                                     num_test=200,
                                                     input_shape=(10,),
                                                     classification=True,
                                                     num_classes=2)
    y_train = to_categorical(y_train)
    return x_train, y_train


def _test_optimizer(optimizer, target=0.75):
    x_train, y_train = get_test_data()

    model = Sequential()
    model.add(Dense(10, input_shape=(x_train.shape[1],)))
    model.add(Activation('relu'))
    model.add(Dense(y_train.shape[1]))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])

    history = model.fit(x_train, y_train, epochs=2, batch_size=16, verbose=0)
    # TODO PlaidML fails this test
    assert history.history['acc'][-1] >= target
    config = k_optimizers.serialize(optimizer)
    optim = k_optimizers.deserialize(config)
    new_config = k_optimizers.serialize(optim)
    new_config['class_name'] = new_config['class_name'].lower()
    assert config == new_config

    # Test constraints.
    model = Sequential()
    dense = Dense(10,
                  input_shape=(x_train.shape[1],),
                  kernel_constraint=lambda x: 0. * x + 1.,
                  bias_constraint=lambda x: 0. * x + 2.,)
    model.add(dense)
    model.add(Activation('relu'))
    model.add(Dense(y_train.shape[1]))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    model.train_on_batch(x_train[:10], y_train[:10])
    kernel, bias = dense.get_weights()
    assert_allclose(kernel, 1.)
    assert_allclose(bias, 2.)


@pytest.mark.parametrize('dummy', [None], ids=[get_backend().upper()])
@pytest.mark.xfail(get_backend() == "amd", reason="plaidML fails the standard accuracy test")
def test_adam(dummy):  # pylint:disable=unused-argument
    """ Test for custom adam optimizer """
    _test_optimizer(optimizers.Adam())
    _test_optimizer(optimizers.Adam(decay=1e-3))
