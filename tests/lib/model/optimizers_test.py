#!/usr/bin/env python3
""" Tests for Faceswap Initializers.

Adapted from Keras tests.
"""
import pytest

import numpy as np

from keras import device, layers as kl, optimizers as k_optimizers, Sequential

from lib.model import optimizers
from lib.utils import get_backend

from tests.utils import generate_test_data, to_categorical


def get_test_data():
    """ Obtain randomized test data for training """
    np.random.seed(1337)
    (x_train, y_train), _ = generate_test_data(num_train=1000,
                                               num_test=200,
                                               input_shape=(10,),
                                               classification=True,
                                               num_classes=2)
    y_train = to_categorical(y_train)
    return x_train, y_train


def _test_optimizer(optimizer, target=0.75):
    x_train, y_train = get_test_data()

    model = Sequential()
    model.add(kl.Input((x_train.shape[1], )))
    model.add(kl.Dense(10))
    model.add(kl.Activation("relu"))
    model.add(kl.Dense(y_train.shape[1]))
    model.add(kl.Activation("softmax"))
    model.compile(loss="categorical_crossentropy",
                  optimizer=optimizer,
                  metrics=["accuracy"])

    history = model.fit(x_train, y_train, epochs=2, batch_size=16, verbose=0)  # type:ignore
    assert history.history["accuracy"][-1] >= target
    config = k_optimizers.serialize(optimizer)
    optim = k_optimizers.deserialize(config)
    new_config = k_optimizers.serialize(optim)
    config["class_name"] = config["class_name"].lower()  # type:ignore
    new_config["class_name"] = new_config["class_name"].lower()  # type:ignore
    assert config == new_config


# TODO remove the next line that supresses a weird pytest bug when it tears down the tempdir
@pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning")
@pytest.mark.parametrize("dummy", [None], ids=[get_backend().upper()])
def test_adabelief(dummy):  # pylint:disable=unused-argument
    """ Test for custom Adam optimizer """
    with device("cpu"):
        _test_optimizer(optimizers.AdaBelief(), target=0.20)
