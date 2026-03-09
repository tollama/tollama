"""Tests for tollama.preprocess.spline."""

from __future__ import annotations

import numpy as np
import pytest

scipy = pytest.importorskip("scipy")

from tollama.preprocess.spline import SplinePreprocessor  # noqa: E402


def _synth(n: int = 100) -> tuple[np.ndarray, np.ndarray]:
    x = np.arange(n, dtype=float)
    y = np.sin(x * 0.1) + np.random.default_rng(42).normal(0, 0.05, n)
    return x, y


# ---------------------------------------------------------------------------
# Fit / transform
# ---------------------------------------------------------------------------


def test_fit_transform_round_trip() -> None:
    x, y = _synth()
    sp = SplinePreprocessor()
    sp.fit(x, y)
    fitted = sp.transform(x)
    assert fitted.shape == y.shape
    # Spline should be close to noisy data
    assert np.mean(np.abs(fitted - y)) < 0.5


def test_fit_transform_combined() -> None:
    x, y = _synth()
    sp = SplinePreprocessor()
    result = sp.fit_transform(x, y)
    assert result.shape == y.shape


def test_unfitted_transform_raises() -> None:
    sp = SplinePreprocessor()
    with pytest.raises(RuntimeError, match="not fitted"):
        sp.transform(np.arange(10, dtype=float))


# ---------------------------------------------------------------------------
# Interpolation
# ---------------------------------------------------------------------------


def test_interpolate_missing_fills_nans() -> None:
    y = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
    sp = SplinePreprocessor()
    result = sp.interpolate_missing(y)
    assert not np.any(np.isnan(result))
    # Interpolated value should be roughly 3
    assert abs(result[2] - 3.0) < 1.0


def test_interpolate_no_missing_returns_copy() -> None:
    y = np.array([1.0, 2.0, 3.0])
    sp = SplinePreprocessor()
    result = sp.interpolate_missing(y)
    np.testing.assert_array_equal(result, y)


def test_interpolate_custom_mask() -> None:
    y = np.array([1.0, 999.0, 3.0, 4.0])
    mask = np.array([False, True, False, False])
    sp = SplinePreprocessor()
    result = sp.interpolate_missing(y, missing_mask=mask)
    assert result[1] != 999.0


# ---------------------------------------------------------------------------
# Smoothing
# ---------------------------------------------------------------------------


def test_smooth_reduces_noise() -> None:
    rng = np.random.default_rng(0)
    y = np.sin(np.linspace(0, 4 * np.pi, 100)) + rng.normal(0, 0.3, 100)
    sp = SplinePreprocessor()
    smoothed = sp.smooth(y, window=7)
    # Smoothed should have less variance than noisy
    assert np.std(smoothed) < np.std(y)


def test_smooth_pspline_passthrough() -> None:
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    sp = SplinePreprocessor(smoothing_method="pspline")
    result = sp.smooth(y)
    np.testing.assert_array_equal(result, y)


# ---------------------------------------------------------------------------
# Knot strategies
# ---------------------------------------------------------------------------


def test_uniform_knot_strategy() -> None:
    x, y = _synth()
    sp = SplinePreprocessor(knot_strategy="uniform")
    sp.fit(x, y)
    assert sp._fitted


def test_curvature_knot_strategy() -> None:
    x, y = _synth()
    sp = SplinePreprocessor(knot_strategy="curvature")
    sp.fit(x, y)
    assert sp._fitted


# ---------------------------------------------------------------------------
# Advanced methods
# ---------------------------------------------------------------------------


def test_extrapolate() -> None:
    x, y = _synth(50)
    sp = SplinePreprocessor()
    sp.fit(x, y)
    future = np.array([50.0, 51.0, 52.0])
    result = sp.extrapolate(future)
    assert result.shape == (3,)


def test_evaluate_derivatives_shape() -> None:
    x, y = _synth(50)
    sp = SplinePreprocessor()
    sp.fit(x, y)
    d1 = sp.evaluate_derivatives(x, order=1)
    assert d1.shape == x.shape


def test_compute_residuals() -> None:
    x, y = _synth(50)
    sp = SplinePreprocessor()
    sp.fit(x, y)
    residuals = sp.compute_residuals(x, y)
    assert residuals.shape == y.shape
    # Residuals should be small relative to y range
    assert np.std(residuals) < np.std(y)


def test_extract_features() -> None:
    _, y = _synth(50)
    sp = SplinePreprocessor()
    features = sp.extract_features(y)
    assert "mean" in features
    assert "std" in features
    assert "min" in features
    assert "max" in features


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_short_series_fallback() -> None:
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([1.0, 2.0, 3.0])
    sp = SplinePreprocessor()
    sp.fit(x, y)
    assert sp._fitted


def test_nan_in_x_raises() -> None:
    x = np.array([0.0, np.nan, 2.0])
    y = np.array([1.0, 2.0, 3.0])
    sp = SplinePreprocessor()
    with pytest.raises(ValueError, match="NaN"):
        sp.fit(x, y)
