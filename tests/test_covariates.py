"""Tests for daemon covariate normalization helpers."""

from __future__ import annotations

import builtins

import pytest

from tollama.core.schemas import SeriesInput
from tollama.daemon.covariates import normalize_covariates


def _series(*, timestamps: list[str], freq: str = "auto") -> SeriesInput:
    target = [float(index + 1) for index in range(len(timestamps))]
    return SeriesInput.model_validate(
        {
            "id": "s1",
            "freq": freq,
            "timestamps": timestamps,
            "target": target,
        },
    )


def test_normalize_covariates_infers_daily_frequency_when_freq_auto() -> None:
    series = _series(
        timestamps=["2025-01-01", "2025-01-02", "2025-01-03"],
        freq="auto",
    )

    normalized, warnings = normalize_covariates([series], prediction_length=2)

    assert warnings == []
    assert normalized[0].freq.lower() == "d"


def test_normalize_covariates_infers_hourly_frequency_when_freq_auto() -> None:
    series = _series(
        timestamps=[
            "2025-01-01T00:00:00",
            "2025-01-01T01:00:00",
            "2025-01-01T02:00:00",
        ],
        freq="auto",
    )

    normalized, _ = normalize_covariates([series], prediction_length=1)

    assert normalized[0].freq.lower() == "h"


def test_normalize_covariates_fails_when_freq_auto_for_irregular_timestamps() -> None:
    series = _series(
        timestamps=["2025-01-01", "2025-01-03", "2025-01-04"],
        freq="auto",
    )

    with pytest.raises(ValueError, match="could not infer frequency"):
        normalize_covariates([series], prediction_length=1)


def test_normalize_covariates_fails_when_freq_auto_with_single_timestamp() -> None:
    series = _series(timestamps=["2025-01-01"], freq="auto")

    with pytest.raises(ValueError, match="could not infer frequency"):
        normalize_covariates([series], prediction_length=1)


def test_normalize_covariates_fails_when_pandas_is_unavailable(monkeypatch) -> None:
    series = _series(
        timestamps=["2025-01-01", "2025-01-02", "2025-01-03"],
        freq="auto",
    )
    original_import = builtins.__import__

    def _fake_import(name: str, *args: object, **kwargs: object):
        if name == "pandas":
            raise ImportError("pandas unavailable for test")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)

    with pytest.raises(ValueError, match="pandas is required"):
        normalize_covariates([series], prediction_length=1)


def test_normalize_covariates_preserves_explicit_freq() -> None:
    series = _series(
        timestamps=["2025-01-01", "2025-01-02", "2025-01-03"],
        freq="D",
    )

    normalized, _ = normalize_covariates([series], prediction_length=1)

    assert normalized[0].freq == "D"
