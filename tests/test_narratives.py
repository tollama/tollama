"""Tests for deterministic narrative builder helpers."""

from __future__ import annotations

from tollama.core.narratives import (
    build_analysis_narrative,
    build_comparison_narrative,
    build_forecast_narrative,
    build_pipeline_narrative,
)
from tollama.core.schemas import (
    AnalyzeResponse,
    AutoForecastResponse,
    CompareResponse,
    ForecastRequest,
    ForecastResponse,
    PipelineResponse,
)


def test_build_forecast_narrative_returns_structured_series_payload() -> None:
    request = ForecastRequest.model_validate(
        {
            "model": "mock",
            "horizon": 2,
            "series": [
                {
                    "id": "s1",
                    "freq": "D",
                    "timestamps": ["2025-01-01", "2025-01-02", "2025-01-03"],
                    "target": [1.0, 2.0, 3.0],
                }
            ],
            "options": {},
        }
    )
    response = ForecastResponse.model_validate(
        {
            "model": "mock",
            "forecasts": [
                {
                    "id": "s1",
                    "freq": "D",
                    "start_timestamp": "2025-01-04",
                    "mean": [3.0, 4.0],
                    "quantiles": {"0.1": [2.0, 3.0], "0.9": [4.0, 5.0]},
                }
            ],
        }
    )

    narrative = build_forecast_narrative(request=request, response=response)

    assert narrative is not None
    assert narrative.series[0].id == "s1"
    assert narrative.series[0].trend.direction in {"up", "down", "flat"}
    assert narrative.series[0].confidence.level in {"high", "medium", "low", "unknown"}


def test_build_analysis_narrative_returns_risk_flags() -> None:
    response = AnalyzeResponse.model_validate(
        {
            "results": [
                {
                    "id": "s1",
                    "detected_frequency": "D",
                    "seasonality_periods": [7],
                    "trend": {"direction": "up", "slope": 0.2, "r2": 0.8},
                    "anomaly_indices": [2],
                    "stationarity_flag": False,
                    "data_quality_score": 0.6,
                }
            ]
        }
    )

    narrative = build_analysis_narrative(response=response)

    assert narrative is not None
    assert narrative.series[0].id == "s1"
    assert narrative.series[0].anomaly_count == 1
    assert "anomalies_detected" in narrative.series[0].key_risks
    assert "low_data_quality" in narrative.series[0].key_risks


def test_build_comparison_narrative_ranks_by_metric() -> None:
    response = CompareResponse.model_validate(
        {
            "models": ["mock", "chronos2"],
            "horizon": 2,
            "results": [
                {
                    "model": "mock",
                    "ok": True,
                    "response": {
                        "model": "mock",
                        "forecasts": [
                            {
                                "id": "s1",
                                "freq": "D",
                                "start_timestamp": "2025-01-04",
                                "mean": [3.0, 4.0],
                            }
                        ],
                        "metrics": {
                            "aggregate": {"smape": 12.0},
                            "series": [{"id": "s1", "values": {"smape": 12.0}}],
                        },
                    },
                },
                {
                    "model": "chronos2",
                    "ok": True,
                    "response": {
                        "model": "chronos2",
                        "forecasts": [
                            {
                                "id": "s1",
                                "freq": "D",
                                "start_timestamp": "2025-01-04",
                                "mean": [3.0, 4.0],
                            }
                        ],
                        "metrics": {
                            "aggregate": {"smape": 8.0},
                            "series": [{"id": "s1", "values": {"smape": 8.0}}],
                        },
                    },
                },
            ],
            "summary": {"requested_models": 2, "succeeded": 2, "failed": 0},
        }
    )

    narrative = build_comparison_narrative(response=response)

    assert narrative.best_model == "chronos2"
    assert narrative.criterion == "metrics.aggregate.smape"
    assert narrative.rankings[0].model == "chronos2"


def test_build_pipeline_narrative_uses_auto_selection() -> None:
    response = PipelineResponse.model_validate(
        {
            "analysis": {
                "results": [
                    {
                        "id": "s1",
                        "detected_frequency": "D",
                        "seasonality_periods": [],
                        "trend": {"direction": "up", "slope": 0.2, "r2": 0.8},
                        "anomaly_indices": [],
                        "stationarity_flag": True,
                        "data_quality_score": 0.9,
                    }
                ]
            },
            "recommendation": {
                "request": {"horizon": 2, "freq": "D", "top_k": 3},
                "recommendations": [{"model": "mock", "family": "mock", "rank": 1, "score": 10.0}],
                "excluded": [],
                "total_candidates": 1,
                "compatible_candidates": 1,
            },
            "auto_forecast": AutoForecastResponse.model_validate(
                {
                    "strategy": "auto",
                    "selection": {
                        "strategy": "auto",
                        "chosen_model": "mock",
                        "selected_models": ["mock"],
                        "candidates": [
                            {
                                "model": "mock",
                                "family": "mock",
                                "rank": 1,
                                "score": 10.0,
                                "reasons": ["selected"],
                            }
                        ],
                        "rationale": ["selected"],
                        "fallback_used": False,
                    },
                    "response": {
                        "model": "mock",
                        "forecasts": [
                            {
                                "id": "s1",
                                "freq": "D",
                                "start_timestamp": "2025-01-04",
                                "mean": [3.0, 4.0],
                            }
                        ],
                    },
                }
            ).model_dump(mode="python"),
            "warnings": ["one warning"],
        }
    )

    narrative = build_pipeline_narrative(response=response)

    assert narrative.chosen_model == "mock"
    assert narrative.warnings_count == 1
