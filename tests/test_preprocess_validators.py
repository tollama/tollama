"""Tests for tollama.preprocess.validators."""

from __future__ import annotations

import numpy as np
import pytest

from tollama.preprocess.schemas import ValidationConfig
from tollama.preprocess.validators import validate_series


def _ts(n: int) -> np.ndarray:
    return np.arange(n, dtype=float)


def test_valid_series_passes() -> None:
    validate_series(_ts(50), np.sin(np.linspace(0, 6, 50)))


def test_non_monotonic_timestamps_raises() -> None:
    ts = np.array([0.0, 2.0, 1.0, 3.0])
    with pytest.raises(ValueError, match="strictly increasing"):
        validate_series(ts, np.array([1.0, 2.0, 3.0, 4.0]))


def test_duplicate_timestamps_raises() -> None:
    ts = np.array([0.0, 1.0, 1.0, 2.0])
    with pytest.raises(ValueError, match="strictly increasing"):
        validate_series(ts, np.array([1.0, 2.0, 3.0, 4.0]))


def test_inf_target_raises() -> None:
    with pytest.raises(ValueError, match="Inf"):
        validate_series(_ts(3), np.array([1.0, np.inf, 3.0]))


def test_missing_ratio_exceeded_raises() -> None:
    target = np.array([1.0, np.nan, np.nan, np.nan, 5.0])
    cfg = ValidationConfig(missing_ratio_max=0.5)
    with pytest.raises(ValueError, match="missing ratio"):
        validate_series(_ts(5), target, config=cfg)


def test_max_gap_exceeded_raises() -> None:
    target = np.full(10, np.nan)
    target[0] = 1.0
    target[9] = 10.0
    cfg = ValidationConfig(missing_ratio_max=0.9, max_gap=5)
    with pytest.raises(ValueError, match="max missing gap"):
        validate_series(_ts(10), target, config=cfg)


def test_constant_target_raises() -> None:
    with pytest.raises(ValueError, match="constant"):
        validate_series(_ts(5), np.array([3.0, 3.0, 3.0, 3.0, 3.0]))


def test_all_nan_target_raises() -> None:
    with pytest.raises(ValueError, match="no valid"):
        validate_series(
            _ts(3),
            np.array([np.nan, np.nan, np.nan]),
            config=ValidationConfig(missing_ratio_max=1.0),
        )


def test_min_length_check() -> None:
    cfg = ValidationConfig(min_length=10)
    with pytest.raises(ValueError, match="below minimum"):
        validate_series(_ts(5), np.arange(5, dtype=float) + 1.0, config=cfg)


def test_empty_raises() -> None:
    with pytest.raises(ValueError, match="empty"):
        validate_series(np.array([]), np.array([]))


def test_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="match"):
        validate_series(_ts(3), np.array([1.0, 2.0]))


def test_custom_config_allows_more_missing() -> None:
    target = np.array([1.0, np.nan, 3.0, np.nan, 5.0])
    cfg = ValidationConfig(missing_ratio_max=0.5, max_gap=2)
    validate_series(_ts(5), target, config=cfg)
