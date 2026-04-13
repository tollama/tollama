"""Tests for automated feature engineering."""

from __future__ import annotations

import numpy as np

from tollama.preprocess.feature_engineering import (
    FeatureConfig,
    auto_engineer_features,
    extract_calendar_features,
    extract_fourier_features,
    extract_lag_features,
    extract_rolling_features,
)


def test_extract_lag_features_auto() -> None:
    values = np.arange(100, dtype=float)
    features = extract_lag_features(values)
    assert "lag_1" in features
    assert "lag_7" in features
    assert len(features["lag_1"]) == 100
    assert np.isnan(features["lag_1"][0])
    assert features["lag_1"][1] == 0.0  # lag_1[1] = values[0] = 0


def test_extract_lag_features_custom() -> None:
    values = np.arange(50, dtype=float)
    features = extract_lag_features(values, lags=[1, 5])
    assert set(features.keys()) == {"lag_1", "lag_5"}


def test_extract_rolling_features() -> None:
    values = np.ones(100)
    features = extract_rolling_features(values)
    assert "rolling_mean_7" in features
    assert "rolling_std_7" in features
    # Constant series should have zero std
    assert features["rolling_std_7"][6] == 0.0


def test_extract_fourier_features() -> None:
    features = extract_fourier_features(100, periods=[7.0], num_pairs=2)
    assert "sin_7.0_1" in features
    assert "cos_7.0_1" in features
    assert "sin_7.0_2" in features
    assert len(features["sin_7.0_1"]) == 100


def test_extract_calendar_features() -> None:
    timestamps = [
        "2025-01-01T00:00:00",
        "2025-01-02T06:00:00",
        "2025-01-03T12:00:00",
    ]
    features = extract_calendar_features(timestamps)
    # Should have some features (not all may be present if constant)
    assert len(features) > 0
    for name, arr in features.items():
        assert len(arr) == 3
        assert np.all(arr >= 0)
        assert np.all(arr <= 1)


def test_auto_engineer_features_basic() -> None:
    values = np.random.randn(200)
    features = auto_engineer_features(values)
    assert len(features) > 0
    for arr in features.values():
        assert len(arr) == 200


def test_auto_engineer_features_with_timestamps() -> None:
    values = np.arange(10, dtype=float)
    timestamps = [f"2025-01-{i + 1:02d}" for i in range(10)]
    config = FeatureConfig(
        calendar=True,
        lags=[1],
        rolling=[3],
        fourier=1,
        differences=True,
    )
    features = auto_engineer_features(values, timestamps=timestamps, config=config)
    assert "lag_1" in features
    assert "diff_1" in features
    assert np.isnan(features["diff_1"][0])
    assert features["diff_1"][1] == 1.0  # diff of [0,1,...] = 1


def test_feature_config_defaults() -> None:
    cfg = FeatureConfig()
    assert cfg.calendar is True
    assert cfg.lags is None
    assert cfg.rolling is None
    assert cfg.fourier == 0
    assert cfg.differences is False
