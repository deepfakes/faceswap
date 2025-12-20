#!/usr/bin/env python3
""" Sanity checks for Faceswap. """

import inspect
import sys

import pytest
import keras
import torch

from lib.utils import get_backend
from lib.system.system import VALID_KERAS, VALID_PYTHON, VALID_TORCH

_BACKEND = get_backend().upper()

_LIBS = (VALID_KERAS + (keras.__version__, ),
         VALID_PYTHON + (sys.version, ),
         VALID_TORCH + (torch.__version__, ))
_IDS = [f"{x}[{_BACKEND}" for x in ("keras", "python", "torch")]


@pytest.mark.parametrize(["min_vers", "max_vers", "installed_vers"], _LIBS, ids=_IDS)
def test_libraries(min_vers: tuple[int, int],
                   max_vers: tuple[int, int],
                   installed_vers: str) -> None:
    """ Sanity check to ensure that we are running on a valid libraries """
    installed = tuple(int(x) for x in installed_vers.split(".")[:2])
    assert min_vers <= installed <= max_vers


@pytest.mark.parametrize('dummy', [None], ids=[_BACKEND])
def test_backend(dummy):  # pylint:disable=unused-argument
    """ Sanity check to ensure that Keras backend is returning the correct object type. """
    with keras.device("cpu"):
        test_var = keras.Variable((1, 1, 4, 4), trainable=False)
    mod = inspect.getmodule(test_var)
    assert mod is not None
    lib = mod.__name__.split(".")[0]
    assert lib == "keras"
