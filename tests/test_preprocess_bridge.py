"""Tests for tollama.preprocess.bridge."""

from __future__ import annotations

import numpy as np
import pytest

scipy = pytest.importorskip("scipy")

from tollama.core.schemas import SeriesInput  # noqa: E402
from tollama.preprocess.bridge import (  # noqa: E402
    preprocess_series_input,
    series_input_to_arrays,
)


def _make_series(n: int = 100) -> SeriesInput:
    rng = np.random.default_rng(42)
    timestamps = [f"2025-01-{i + 1:02d}T00:00:00Z" for i in range(min(n, 28))]
    if n > 28:
        timestamps = [f"2025-01-01T{i:04d}:00Z".replace(":", "", 1) for i in range(n)]
        # Simpler: just use sequential hour timestamps
        timestamps = []
        for i in range(n):
            day = i // 24 + 1
            hour = i % 24
            timestamps.append(f"2025-01-{day:02d}T{hour:02d}:00:00Z")

    target = (np.sin(np.arange(n) * 0.1) * 10 + rng.normal(0, 0.5, n)).tolist()
    return SeriesInput(
        id="test-series",
        freq="H",
        timestamps=timestamps,
        target=target,
    )


def test_series_input_to_arrays() -> None:
    series = _make_series(50)
    x, target, covariates = series_input_to_arrays(series)
    assert x.shape == (50,)
    assert target.shape == (50,)
    assert isinstance(covariates, dict)
    assert len(covariates) == 0


def test_series_input_to_arrays_with_covariates() -> None:
    series = _make_series(50)
    series = SeriesInput(
        id="test",
        freq="H",
        timestamps=series.timestamps,
        target=series.target,
        past_covariates={"temp": [float(i) for i in range(50)]},
    )
    x, target, covariates = series_input_to_arrays(series)
    assert "temp" in covariates
    assert covariates["temp"].shape == (50,)


def test_timestamps_are_zero_based() -> None:
    series = _make_series(10)
    x, _, _ = series_input_to_arrays(series)
    np.testing.assert_array_equal(x, np.arange(10, dtype=float))


def test_preprocess_series_input_e2e() -> None:
    series = _make_series(100)
    result = preprocess_series_input(series)
    assert result.X.ndim == 3
    assert result.y.ndim == 2
    assert result.X.shape[0] > 0
