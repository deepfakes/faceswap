#!/usr/bin/env python3
""" Sanity checks for Faceswap. """

import inspect

import pytest

import keras
from keras import backend as K

from lib.utils import get_backend

_BACKEND = get_backend()


@pytest.mark.parametrize('dummy', [None], ids=[get_backend().upper()])
def test_backend(dummy):  # pylint:disable=unused-argument
    """ Sanity check to ensure that Keras backend is returning the correct object type. """
    test_var = K.variable((1, 1, 4, 4))
    lib = inspect.getmodule(test_var).__name__.split(".")[0]
    assert (_BACKEND == "cpu" and lib == "tensorflow") or (_BACKEND == "amd" and lib == "plaidml")


@pytest.mark.parametrize('dummy', [None], ids=[get_backend().upper()])
def test_keras(dummy):  # pylint:disable=unused-argument
    """ Sanity check to ensure that tensorflow keras is being used for CPU and standard
    keras for AMD. """
    assert ((_BACKEND == "cpu" and keras.__version__.endswith("-tf")) or
            (_BACKEND == "amd" and not keras.__version__.endswith("-tf")))
