#!/usr/bin/env python3
""" Sanity checks for Faceswap. """

import inspect

import pytest
from keras import backend as K

from lib.utils import get_backend


@pytest.mark.parametrize('dummy', [None], ids=[get_backend().upper()])
def test_backend(dummy):  # pylint:disable=unused-argument
    """ Sanity check to ensure that Keras backend is returning the correct object type. """
    backend = get_backend()
    test_var = K.variable((1, 1, 4, 4))
    lib = inspect.getmodule(test_var).__name__.split(".")[0]
    assert (backend == "cpu" and lib == "tensorflow") or (backend == "amd" and lib == "plaidml")
