#!/usr/bin/env python3
""" Sanity checks for Faceswap. """

import inspect
import pytest

# Ignore linting errors from Tensorflow's thoroughly broken import system
from tensorflow import keras
from tensorflow.keras import backend as K  # pylint:disable=import-error

from lib.utils import get_backend

_BACKEND = get_backend()


@pytest.mark.parametrize('dummy', [None], ids=[get_backend().upper()])
def test_backend(dummy):  # pylint:disable=unused-argument
    """ Sanity check to ensure that Keras backend is returning the correct object type. """
    test_var = K.variable((1, 1, 4, 4))
    lib = inspect.getmodule(test_var).__name__.split(".")[0]
    assert _BACKEND in ("cpu", "directml") and lib == "tensorflow"


@pytest.mark.parametrize('dummy', [None], ids=[get_backend().upper()])
def test_keras(dummy):  # pylint:disable=unused-argument
    """ Sanity check to ensure that tensorflow keras is being used for CPU """
    assert (_BACKEND in ("cpu", "directml")
            and keras.__version__ in ("2.7.0", "2.8.0", "2.9.0", "2.10.0"))
