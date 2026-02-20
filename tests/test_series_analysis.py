"""Tests for deterministic core series analysis helpers."""

from __future__ import annotations

import warnings
from datetime import date, timedelta

from tollama.core.schemas import AnalyzeRequest
from tollama.core.series_analysis import (
    analyze_series_request,
    detect_anomalies,
    detect_seasonality,
    detect_stationarity,
    detect_trend,
)


def _base_series_payload() -> dict[str, object]:
    start = date(2025, 1, 1)
    return {
        "id": "s1",
        "freq": "D",
        "timestamps": [str(start + timedelta(days=offset)) for offset in range(40)],
        "target": [10.0 + (day % 4) for day in range(40)],
    }


def test_analyze_series_request_returns_structured_results() -> None:
    payload = {
        "series": [_base_series_payload()],
        "parameters": {"max_lag": 16, "top_k_seasonality": 2},
    }
    request = AnalyzeRequest.model_validate(payload)

    response = analyze_series_request(request)

    assert len(response.results) == 1
    result = response.results[0]
    assert result.id == "s1"
    assert result.detected_frequency in {"D", "24H", "1D"}
    assert 0.0 <= result.data_quality_score <= 1.0


def test_detect_trend_direction_is_up_for_rising_series() -> None:
    trend, warnings = detect_trend([1.0, 2.0, 3.0, 4.0, 5.0])

    assert warnings == []
    assert trend.direction == "up"
    assert trend.slope > 0.0
    assert 0.0 <= trend.r2 <= 1.0


def test_detect_seasonality_finds_repeating_period() -> None:
    values = [float((idx % 4) - 1) for idx in range(48)]
    periods, warnings = detect_seasonality(values, max_lag=24, top_k=3)

    assert warnings == []
    assert 4 in periods


def test_detect_anomalies_flags_large_outlier_index() -> None:
    anomalies, warnings = detect_anomalies([1.0, 1.2, 1.1, 1.0, 9.0, 1.2, 1.1], iqr_k=1.5)

    assert warnings == []
    assert anomalies == [4]


def test_detect_stationarity_returns_none_for_short_series() -> None:
    stationary, warnings = detect_stationarity([1.0, 1.1, 0.9, 1.0])

    assert stationary is None
    assert warnings


def test_analyze_series_request_downsamples_when_max_points_is_small() -> None:
    payload = {
        "series": [_base_series_payload()],
        "parameters": {"max_points": 8, "max_lag": 4},
    }
    request = AnalyzeRequest.model_validate(payload)

    response = analyze_series_request(request)

    warnings = response.results[0].warnings or []
    assert any("downsampled" in warning for warning in warnings)


def test_analyze_series_request_applies_chronological_order_for_unsorted_timestamps() -> None:
    payload = {
        "series": [
            {
                "id": "s1",
                "freq": "D",
                "timestamps": ["2025-01-03", "2025-01-01", "2025-01-02"],
                "target": [3.0, 1.0, 2.0],
            }
        ]
    }
    request = AnalyzeRequest.model_validate(payload)

    response = analyze_series_request(request)

    result = response.results[0]
    assert result.trend.direction == "up"
    result_warnings = result.warnings or []
    assert any("not monotonic" in warning for warning in result_warnings)


def test_analyze_series_request_penalizes_unparseable_timestamps_in_quality_score() -> None:
    payload = {
        "series": [
            {
                "id": "s1",
                "freq": "auto",
                "timestamps": ["foo", "bar", "baz"],
                "target": [1.0, 2.0, 3.0],
            }
        ]
    }
    request = AnalyzeRequest.model_validate(payload)

    response = analyze_series_request(request)

    assert response.results[0].data_quality_score < 0.5


def test_analyze_series_request_suppresses_parse_user_warnings() -> None:
    payload = {
        "series": [
            {
                "id": "s1",
                "timestamps": ["not-a-date", "still-not-a-date", "2025-01-03"],
                "target": [1.0, 2.0, 3.0],
            }
        ]
    }
    request = AnalyzeRequest.model_validate(payload)

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        analyze_series_request(request)

    user_warnings = [item for item in captured if issubclass(item.category, UserWarning)]
    assert not user_warnings
