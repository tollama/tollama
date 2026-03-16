"""Tests for conformal prediction module."""

from __future__ import annotations

import numpy as np
import pytest

from tollama.core.conformal import (
    ConformalCalibration,
    apply_conformal_to_response,
    calibrate,
    predict_intervals,
)
from tollama.core.schemas import ForecastResponse, SeriesForecast


def test_calibrate_split_basic() -> None:
    actuals = np.array([[10.0, 11.0, 12.0], [20.0, 21.0, 22.0]])
    predictions = np.array([[9.5, 11.5, 12.0], [19.5, 21.5, 22.0]])
    cal = calibrate(actuals, predictions, coverage=0.9, method="split")
    assert cal.method == "split"
    assert cal.coverage == 0.9
    assert cal.horizon == 3
    assert cal.q_hat > 0


def test_calibrate_adaptive_basic() -> None:
    actuals = np.array([[10.0, 20.0], [30.0, 40.0]])
    predictions = np.array([[9.0, 19.0], [29.0, 39.0]])
    cal = calibrate(actuals, predictions, coverage=0.9, method="adaptive")
    assert cal.method == "adaptive"
    assert cal.q_hat > 0


def test_calibrate_validates_shapes() -> None:
    with pytest.raises(ValueError, match="Shape mismatch"):
        calibrate(np.array([1, 2, 3]), np.array([1, 2]))


def test_calibrate_validates_coverage() -> None:
    with pytest.raises(ValueError, match="Coverage must be"):
        calibrate(np.array([1.0]), np.array([1.0]), coverage=1.5)


def test_calibrate_validates_empty() -> None:
    with pytest.raises(ValueError, match="empty"):
        calibrate(np.array([]), np.array([]))


def test_predict_intervals_split() -> None:
    cal = ConformalCalibration(
        scores=np.array([0.5, 1.0, 1.5]),
        method="split",
        coverage=0.9,
        horizon=3,
        q_hat=1.5,
    )
    intervals = predict_intervals([10.0, 20.0, 30.0], cal)
    assert "0.05" in intervals
    assert "0.95" in intervals
    assert len(intervals["0.05"]) == 3
    assert len(intervals["0.95"]) == 3
    # Lower should be below mean, upper above
    for i in range(3):
        assert intervals["0.05"][i] < [10.0, 20.0, 30.0][i]
        assert intervals["0.95"][i] > [10.0, 20.0, 30.0][i]


def test_predict_intervals_adaptive() -> None:
    cal = ConformalCalibration(
        scores=np.array([0.1, 0.2]),
        method="adaptive",
        coverage=0.9,
        horizon=2,
        q_hat=0.15,
    )
    intervals = predict_intervals([100.0, 200.0], cal)
    # Adaptive intervals scale with prediction magnitude
    width_100 = intervals["0.95"][0] - intervals["0.05"][0]
    width_200 = intervals["0.95"][1] - intervals["0.05"][1]
    assert width_200 > width_100  # Wider interval for larger prediction


def test_apply_conformal_to_response() -> None:
    response = ForecastResponse(
        model="test",
        forecasts=[
            SeriesForecast(
                id="s1", freq="D", start_timestamp="2025-01-01",
                mean=[10.0, 20.0, 30.0],
            ),
        ],
    )
    cal = calibrate(
        np.array([[10.0, 20.0, 30.0], [11.0, 21.0, 31.0]]),
        np.array([[9.5, 19.5, 29.5], [10.5, 20.5, 30.5]]),
        coverage=0.9,
    )
    result = apply_conformal_to_response(response, cal)
    assert result.model == "test"
    assert len(result.forecasts) == 1
    assert result.forecasts[0].quantiles is not None
    assert "0.05" in result.forecasts[0].quantiles
    assert "0.95" in result.forecasts[0].quantiles
