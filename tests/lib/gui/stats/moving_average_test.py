#!/usr/bin python3
""" Pytest unit tests for :mod:`lib.gui.stats.moving_average` """

import numpy as np
import pytest

from lib.gui.analysis.moving_average import ExponentialMovingAverage as EMA

# pylint:disable=[protected-access,invalid-name]


_INIT_PARAMS = ((np.array([1, 2, 3], dtype="float32"), 0.0),
                (np.array([4, 5, 6], dtype="float64"), 0.25),
                (np.array([7, 8, 9], dtype="uint8"), 1.0),
                (np.array([0, np.nan, 1], dtype="float32"), 0.74),
                (np.array([2, 3, np.inf], dtype="float32"), 0.33),
                (np.array([4, 5, 6], dtype="float32"), -1.0),
                (np.array([7, 8, 9], dtype="float32"), 99.0))
_INIT_IDS = ["float32", "float64", "uint8", "nan", "inf", "amount:-1", "amount:99"]


@pytest.mark.parametrize(("data", "amount"), _INIT_PARAMS, ids=_INIT_IDS)
def test_ExponentialMovingAverage_init(data: np.ndarray, amount: float):
    """ Test that moving_average.MovingAverage correctly initializes """
    attrs = {"_data": np.ndarray,
             "_alpha": float,
             "_dtype": str,
             "_row_size": int,
             "_out": np.ndarray}

    instance = EMA(data, amount)
    # Verify required attributes exist and are of the correct type
    for attr, attr_type in attrs.items():
        assert attr in instance.__dict__
        assert isinstance(getattr(instance, attr), attr_type)
    # Verify we are testing all existing attributes
    for key in instance.__dict__:
        assert key in attrs

    # Verify numeric sanitization
    assert not np.any(np.isnan(instance._data))
    assert not np.any(np.isinf(instance._data))

    # Check alpha clamp logic
    expected_alpha = 1. - min(0.999, max(0.001, amount))
    assert instance._alpha == expected_alpha

    # dtype assignment logic
    expected_dtype = "float32" if data.dtype == np.float32 else "float64"
    assert instance._dtype == expected_dtype

    # ensure row size is positive and output matches shape and dtype
    assert instance._row_size > 0
    assert instance._out.shape == data.shape
    assert instance._out.dtype == expected_dtype


def naive_ewma(data: np.ndarray, alpha: float) -> np.ndarray:
    """ A simple ewma implementation to test for correctness """
    out = np.empty_like(data, dtype=data.dtype)
    out[0] = data[0]
    for i in range(1, len(data)):
        out[i] = alpha * data[i] + (1 - alpha) * out[i - 1]
    return out


@pytest.mark.parametrize("alpha", [0.001, 0.01, 0.25, 0.33, 0.5, 0.66, 0.75, 0.90, 0.999])
@pytest.mark.parametrize("dtype", ("float32", "float64"))
def test_ExponentialMovingAverage_matches_naive(alpha: float, dtype: str) -> None:
    """ Make sure that we get sane results out for various data sizes against our reference
    for various amounts """
    rows = max(5, int(np.random.random() * 25000))
    data = np.random.rand(rows).astype(dtype)
    instance = EMA(data, 1 - alpha)
    out = instance()

    ref = naive_ewma(data, alpha)
    np.testing.assert_allclose(out, ref, rtol=3e-6, atol=3e-6)


@pytest.mark.parametrize("dtype", ("float32", "float64"))
def test_ExponentialMovingAverage_small_data(dtype: str) -> None:
    """ Make sure we get sane results out of our small path """
    data = np.array([1., 2., 3.], dtype=dtype)
    instance = EMA(data, 0.5)
    out = instance()
    ref = naive_ewma(data, instance._alpha)
    np.testing.assert_allclose(out, ref)


@pytest.mark.parametrize("dtype", ("float32", "float64"))
def test_ExponentialMovingAverage_large_data_safe_path(dtype: str) -> None:
    """ Make sure we get sane results out of our safe path """
    data = np.random.rand(50000).astype(dtype)
    instance = EMA(data, 0.1)
    # Force safe path
    instance._row_size = 10

    out = instance()
    ref = naive_ewma(data, instance._alpha)

    np.testing.assert_allclose(out, ref, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("dtype", ("float32", "float64"))
def test_ExponentialMovingAverage_empty_input(dtype: str) -> None:
    """ Test that we get no data on an empty input """
    data = np.array([], dtype=dtype)
    instance = EMA(data, 0.5)
    out = instance()
    assert out.size == 0
