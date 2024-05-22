#!/usr/bin/env python3
""" Sanity checks for Faceswap. """

import inspect
import pytest

import keras

from lib.utils import get_backend

_BACKEND = get_backend()


@pytest.mark.parametrize('dummy', [None], ids=[get_backend().upper()])
def test_backend(dummy):  # pylint:disable=unused-argument
    """ Sanity check to ensure that Keras backend is returning the correct object type. """
    with keras.device("cpu"):
        test_var = keras.Variable((1, 1, 4, 4))
    lib = inspect.getmodule(test_var).__name__.split(".")[0]
    assert _BACKEND in ("cpu", ) and lib == "keras"


@pytest.mark.parametrize('dummy', [None], ids=[get_backend().upper()])
def test_keras(dummy):  # pylint:disable=unused-argument
    """ Sanity check to ensure that tensorflow keras is being used for CPU """
    assert (_BACKEND in ("cpu", )
            and keras.__version__ in ("3.3.3", ))
