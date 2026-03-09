"""Tests for tollama.preprocess.window."""

from __future__ import annotations

import numpy as np
import pytest

from tollama.preprocess.window import make_windows, make_windows_multivariate

# ---------------------------------------------------------------------------
# make_windows
# ---------------------------------------------------------------------------


def test_make_windows_shapes() -> None:
    s = np.arange(20, dtype=float)
    X, y = make_windows(s, lookback=5, horizon=2)
    assert X.shape == (14, 5, 1)
    assert y.shape == (14, 2)


def test_make_windows_values() -> None:
    s = np.arange(10, dtype=float)
    X, y = make_windows(s, lookback=3, horizon=1)
    np.testing.assert_array_equal(X[0, :, 0], [0.0, 1.0, 2.0])
    np.testing.assert_array_equal(y[0], [3.0])
    np.testing.assert_array_equal(X[-1, :, 0], [6.0, 7.0, 8.0])
    np.testing.assert_array_equal(y[-1], [9.0])


def test_make_windows_nan_raises() -> None:
    s = np.array([1.0, np.nan, 3.0])
    with pytest.raises(ValueError, match="NaN"):
        make_windows(s, lookback=1, horizon=1)


def test_make_windows_too_short_raises() -> None:
    s = np.array([1.0, 2.0])
    with pytest.raises(ValueError, match="not enough"):
        make_windows(s, lookback=3, horizon=1)


def test_make_windows_exact_fit() -> None:
    s = np.arange(5, dtype=float)
    X, y = make_windows(s, lookback=3, horizon=2)
    assert X.shape[0] == 1


# ---------------------------------------------------------------------------
# make_windows_multivariate
# ---------------------------------------------------------------------------


def test_multivariate_shapes() -> None:
    features = np.random.randn(30, 3)
    target = np.random.randn(30)
    X, y = make_windows_multivariate(features, target, lookback=5, horizon=2)
    assert X.shape == (24, 5, 3)
    assert y.shape == (24, 2)


def test_multivariate_with_future() -> None:
    features = np.random.randn(30, 2)
    target = np.random.randn(30)
    future = np.random.randn(30, 1)
    X, y, X_fut = make_windows_multivariate(
        features, target, lookback=5, horizon=2, future_features=future
    )
    assert X.shape == (24, 5, 2)
    assert y.shape == (24, 2)
    assert X_fut.shape == (24, 2, 1)


def test_multivariate_shape_mismatch_raises() -> None:
    features = np.random.randn(20, 2)
    target = np.random.randn(15)
    with pytest.raises(ValueError, match="same time length"):
        make_windows_multivariate(features, target, lookback=5)
