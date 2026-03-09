"""Tests for tollama.preprocess.pipeline."""

from __future__ import annotations

import numpy as np
import pytest

scipy = pytest.importorskip("scipy")

from tollama.preprocess.pipeline import PreprocessResult, run_pipeline  # noqa: E402
from tollama.preprocess.schemas import PreprocessConfig, SplineConfig  # noqa: E402


def _synth(n: int = 200) -> tuple[np.ndarray, np.ndarray]:
    x = np.arange(n, dtype=float)
    rng = np.random.default_rng(42)
    y = np.sin(x * 0.05) * 10 + rng.normal(0, 0.5, n)
    return x, y


def test_run_pipeline_returns_result() -> None:
    x, y = _synth()
    result = run_pipeline(x, y)
    assert isinstance(result, PreprocessResult)
    assert result.X.ndim == 3
    assert result.y.ndim == 2


def test_result_shapes() -> None:
    x, y = _synth()
    cfg = PreprocessConfig(lookback=10, horizon=3)
    result = run_pipeline(x, y, config=cfg)
    assert result.X.shape[1] == 10
    assert result.X.shape[2] == 1
    assert result.y.shape[1] == 3
    assert result.X.shape[0] == result.y.shape[0]


def test_scaler_fitted_on_train_only() -> None:
    x, y = _synth()
    result = run_pipeline(x, y)
    assert result.scaler is not None
    assert result.scaler.fitted_
    assert result.train_end > 0
    assert result.val_end > result.train_end


def test_scaling_none_skips_scaler() -> None:
    x, y = _synth()
    cfg = PreprocessConfig(scaling="none")
    result = run_pipeline(x, y, config=cfg)
    assert result.scaler is None
    np.testing.assert_array_equal(result.scaled, result.smoothed)


def test_pipeline_with_nan_interpolates() -> None:
    x, y = _synth()
    y[10] = np.nan
    y[20] = np.nan
    result = run_pipeline(x, y)
    assert not np.any(np.isnan(result.interpolated))


def test_default_config_works() -> None:
    x, y = _synth()
    result = run_pipeline(x, y)
    assert result.X.shape[0] > 0


def test_custom_config_propagates() -> None:
    x, y = _synth()
    cfg = PreprocessConfig(
        lookback=5,
        horizon=2,
        scaling="minmax",
        spline=SplineConfig(smoothing_window=7),
    )
    result = run_pipeline(x, y, config=cfg)
    assert result.X.shape[1] == 5
    assert result.y.shape[1] == 2


def test_pipeline_too_short_raises() -> None:
    x = np.array([0.0, 1.0])
    y = np.array([1.0, 2.0])
    with pytest.raises(ValueError):
        run_pipeline(x, y)
