#!/usr/bin/env python3
""" Sanity checks for Faceswap. """

import inspect

import pytest

from lib.utils import get_backend

if get_backend() == "amd":
    import keras
    from keras import backend as K
else:
    # Ignore linting errors from Tensorflow's thoroughly broken import system
    from tensorflow import keras
    from tensorflow.keras import backend as K  # pylint:disable=import-error


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
    assert ((_BACKEND == "cpu" and keras.__version__ in ("2.4.0", "2.6.0", "2.7.0", "2.8.0")) or
            (_BACKEND == "amd" and keras.__version__ == "2.2.4"))
