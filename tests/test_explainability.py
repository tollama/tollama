"""Tests for deterministic forecast explainability helpers."""

from __future__ import annotations

from tollama.core.explainability import generate_explanation
from tollama.core.schemas import ForecastRequest, ForecastResponse


def test_generate_explanation_returns_structured_series_payload() -> None:
    request = ForecastRequest.model_validate(
        {
            "model": "mock",
            "horizon": 3,
            "quantiles": [0.1, 0.9],
            "series": [
                {
                    "id": "s1",
                    "freq": "D",
                    "timestamps": ["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-04"],
                    "target": [10.0, 11.0, 12.0, 13.0],
                }
            ],
            "options": {},
        },
    )
    response = ForecastResponse.model_validate(
        {
            "model": "mock",
            "forecasts": [
                {
                    "id": "s1",
                    "freq": "D",
                    "start_timestamp": "2025-01-05",
                    "mean": [14.0, 15.0, 16.0],
                    "quantiles": {
                        "0.1": [13.5, 14.2, 15.0],
                        "0.9": [14.5, 15.8, 17.0],
                    },
                }
            ],
        },
    )

    explanation = generate_explanation(request=request, response=response)

    assert explanation is not None
    assert len(explanation.series) == 1
    series = explanation.series[0]
    assert series.id == "s1"
    assert series.trend_direction == "up"
    assert "confidence" in series.confidence_assessment
    assert "above historical mean" in series.historical_comparison
    assert "upward trajectory" in series.notable_patterns[0]


def test_generate_explanation_handles_missing_quantile_forecasts() -> None:
    request = ForecastRequest.model_validate(
        {
            "model": "mock",
            "horizon": 2,
            "quantiles": [],
            "series": [
                {
                    "id": "s1",
                    "freq": "D",
                    "timestamps": ["2025-01-01", "2025-01-02"],
                    "target": [5.0, 5.0],
                }
            ],
            "options": {},
        },
    )
    response = ForecastResponse.model_validate(
        {
            "model": "mock",
            "forecasts": [
                {
                    "id": "s1",
                    "freq": "D",
                    "start_timestamp": "2025-01-03",
                    "mean": [5.0, 5.0],
                }
            ],
        },
    )

    explanation = generate_explanation(request=request, response=response)

    assert explanation is not None
    series = explanation.series[0]
    assert series.trend_direction == "flat"
    assert series.confidence_assessment == "quantile uncertainty unavailable"


def test_generate_explanation_returns_none_when_series_do_not_match_request() -> None:
    request = ForecastRequest.model_validate(
        {
            "model": "mock",
            "horizon": 1,
            "quantiles": [],
            "series": [
                {
                    "id": "s1",
                    "freq": "D",
                    "timestamps": ["2025-01-01"],
                    "target": [1.0],
                }
            ],
            "options": {},
        },
    )
    response = ForecastResponse.model_validate(
        {
            "model": "mock",
            "forecasts": [
                {
                    "id": "other",
                    "freq": "D",
                    "start_timestamp": "2025-01-02",
                    "mean": [1.0],
                }
            ],
        },
    )

    explanation = generate_explanation(request=request, response=response)

    assert explanation is None
