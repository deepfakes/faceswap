#!/usr/bin/env python3
""" Sanity checks for Faceswap. """

import inspect

from keras import backend as K

from lib.utils import get_backend


def test_backend():
    """ Sanity check to ensure that Keras backend is returning the correct object type. """
    backend = get_backend()
    test_var = K.variable((1, 1, 4, 4))
    lib = inspect.getmodule(test_var).__name__.split(".")[0]
    assert (backend == "cpu" and lib == "tensorflow") or (backend == "amd" and lib == "plaidml")
