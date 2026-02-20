"""Tests for forecast request/response schemas."""

import pytest
from pydantic import ValidationError

from tollama.core.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    AutoForecastRequest,
    AutoForecastResponse,
    CounterfactualRequest,
    CounterfactualResponse,
    ForecastReport,
    ForecastRequest,
    ForecastResponse,
    GenerateRequest,
    GenerateResponse,
    PipelineRequest,
    PipelineResponse,
    ProgressiveForecastEvent,
    ReportRequest,
    ScenarioTreeRequest,
    ScenarioTreeResponse,
    SeriesForecast,
    WhatIfRequest,
    WhatIfResponse,
)


def _example_request_payload() -> dict[str, object]:
    return {
        "model": "naive",
        "horizon": 3,
        "quantiles": [0.1, 0.5, 0.9],
        "series": [
            {
                "id": "series-1",
                "freq": "D",
                "timestamps": ["2025-01-01", "2025-01-02", "2025-01-03"],
                "target": [10.0, 11.0, 12.0],
                "past_covariates": {"promo": [0.0, 1.0, 0.0]},
                "future_covariates": {"promo": [1.1, 1.2, 1.3]},
                "static_covariates": {"region": "us-east", "tier": 2},
            }
        ],
        "options": {"temperature": 0.1, "seed": 7},
    }


def _example_analyze_request_payload() -> dict[str, object]:
    return {
        "series": [
            {
                "id": "series-1",
                "freq": "D",
                "timestamps": ["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-04"],
                "target": [10.0, 11.0, 12.0, 11.5],
            }
        ],
        "parameters": {
            "max_points": 100,
            "max_lag": 16,
            "top_k_seasonality": 2,
            "anomaly_iqr_k": 1.5,
        },
    }


def _example_auto_forecast_request_payload() -> dict[str, object]:
    return {
        "horizon": 3,
        "strategy": "auto",
        "series": [
            {
                "id": "series-1",
                "freq": "D",
                "timestamps": ["2025-01-01", "2025-01-02", "2025-01-03"],
                "target": [10.0, 11.0, 12.0],
            }
        ],
        "options": {},
    }


def _example_what_if_request_payload() -> dict[str, object]:
    return {
        "model": "naive",
        "horizon": 3,
        "series": [
            {
                "id": "series-1",
                "freq": "D",
                "timestamps": ["2025-01-01", "2025-01-02", "2025-01-03"],
                "target": [10.0, 11.0, 12.0],
                "past_covariates": {"temperature": [20.0, 21.0, 22.0]},
                "future_covariates": {"temperature": [23.0, 24.0, 25.0]},
            }
        ],
        "scenarios": [
            {
                "name": "high_demand",
                "transforms": [
                    {"operation": "multiply", "field": "target", "value": 1.2},
                ],
            }
        ],
        "options": {},
    }


def _example_pipeline_request_payload() -> dict[str, object]:
    return {
        "horizon": 3,
        "strategy": "auto",
        "series": [
            {
                "id": "series-1",
                "freq": "D",
                "timestamps": ["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-04"],
                "target": [10.0, 11.0, 12.0, 11.5],
            }
        ],
        "options": {},
        "pull_if_missing": True,
        "recommend_top_k": 3,
    }


def _example_generate_request_payload() -> dict[str, object]:
    return {
        "series": [
            {
                "id": "series-1",
                "freq": "D",
                "timestamps": [
                    "2025-01-01",
                    "2025-01-02",
                    "2025-01-03",
                    "2025-01-04",
                    "2025-01-05",
                ],
                "target": [10.0, 11.0, 12.0, 11.5, 12.5],
            }
        ],
        "count": 2,
        "length": 5,
        "seed": 7,
        "method": "statistical",
        "variation": {
            "level_jitter": 0.1,
            "trend_jitter": 0.1,
            "seasonality_jitter": 0.1,
            "noise_scale": 1.0,
            "respect_non_negative": True,
        },
    }


def _example_counterfactual_request_payload() -> dict[str, object]:
    return {
        "model": "naive",
        "intervention_index": 3,
        "series": [
            {
                "id": "series-1",
                "freq": "D",
                "timestamps": [
                    "2025-01-01",
                    "2025-01-02",
                    "2025-01-03",
                    "2025-01-04",
                    "2025-01-05",
                ],
                "target": [10.0, 11.0, 12.0, 20.0, 22.0],
            }
        ],
        "options": {},
    }


def _example_scenario_tree_request_payload() -> dict[str, object]:
    return {
        "model": "naive",
        "horizon": 4,
        "depth": 2,
        "branch_quantiles": [0.1, 0.5, 0.9],
        "series": [
            {
                "id": "series-1",
                "freq": "D",
                "timestamps": ["2025-01-01", "2025-01-02", "2025-01-03"],
                "target": [10.0, 11.0, 12.0],
            }
        ],
        "options": {},
    }


def _example_report_request_payload() -> dict[str, object]:
    payload = _example_pipeline_request_payload()
    payload["include_baseline"] = True
    return payload


def test_forecast_request_roundtrip_is_lossless() -> None:
    request = ForecastRequest.model_validate(_example_request_payload())
    encoded = request.to_json()
    decoded = ForecastRequest.model_validate_json(encoded)
    assert decoded == request
    assert decoded.model_dump(mode="json", exclude_none=True) == request.model_dump(
        mode="json",
        exclude_none=True,
    )


def test_forecast_request_accepts_response_options_narrative() -> None:
    payload = _example_request_payload()
    payload["response_options"] = {"narrative": True}
    request = ForecastRequest.model_validate(payload)
    assert request.response_options.narrative is True


def test_analyze_request_roundtrip_is_lossless() -> None:
    request = AnalyzeRequest.model_validate(_example_analyze_request_payload())
    encoded = request.to_json()
    decoded = AnalyzeRequest.model_validate_json(encoded)
    assert decoded == request
    assert decoded.parameters.max_points == 100


def test_auto_forecast_request_roundtrip_is_lossless() -> None:
    request = AutoForecastRequest.model_validate(_example_auto_forecast_request_payload())
    encoded = request.to_json()
    decoded = AutoForecastRequest.model_validate_json(encoded)
    assert decoded == request
    assert decoded.model is None
    assert decoded.strategy == "auto"


def test_what_if_request_roundtrip_is_lossless() -> None:
    request = WhatIfRequest.model_validate(_example_what_if_request_payload())
    encoded = request.to_json()
    decoded = WhatIfRequest.model_validate_json(encoded)
    assert decoded == request
    assert decoded.scenarios[0].name == "high_demand"
    assert decoded.scenarios[0].transforms[0].operation == "multiply"


def test_pipeline_request_roundtrip_is_lossless() -> None:
    request = PipelineRequest.model_validate(_example_pipeline_request_payload())
    encoded = request.to_json()
    decoded = PipelineRequest.model_validate_json(encoded)
    assert decoded == request
    assert decoded.recommend_top_k == 3
    assert decoded.pull_if_missing is True


def test_generate_request_roundtrip_is_lossless() -> None:
    request = GenerateRequest.model_validate(_example_generate_request_payload())
    encoded = request.to_json()
    decoded = GenerateRequest.model_validate_json(encoded)
    assert decoded == request
    assert decoded.count == 2
    assert decoded.method == "statistical"


def test_generate_request_rejects_short_series() -> None:
    payload = _example_generate_request_payload()
    payload["series"][0]["timestamps"] = ["2025-01-01", "2025-01-02"]
    payload["series"][0]["target"] = [1.0, 2.0]
    with pytest.raises(ValidationError):
        GenerateRequest.model_validate(payload)


def test_counterfactual_request_roundtrip_is_lossless() -> None:
    request = CounterfactualRequest.model_validate(_example_counterfactual_request_payload())
    encoded = request.to_json()
    decoded = CounterfactualRequest.model_validate_json(encoded)
    assert decoded == request
    assert decoded.intervention_index == 3


def test_counterfactual_request_rejects_invalid_intervention_index() -> None:
    payload = _example_counterfactual_request_payload()
    payload["intervention_index"] = 5
    with pytest.raises(ValidationError):
        CounterfactualRequest.model_validate(payload)


def test_scenario_tree_request_roundtrip_is_lossless() -> None:
    request = ScenarioTreeRequest.model_validate(_example_scenario_tree_request_payload())
    encoded = request.to_json()
    decoded = ScenarioTreeRequest.model_validate_json(encoded)
    assert decoded == request
    assert decoded.depth == 2


def test_report_request_roundtrip_is_lossless() -> None:
    request = ReportRequest.model_validate(_example_report_request_payload())
    encoded = request.to_json()
    decoded = ReportRequest.model_validate_json(encoded)
    assert decoded == request
    assert decoded.include_baseline is True


def test_what_if_request_rejects_duplicate_scenario_names() -> None:
    payload = _example_what_if_request_payload()
    payload["scenarios"] = [payload["scenarios"][0], payload["scenarios"][0]]
    with pytest.raises(ValidationError):
        WhatIfRequest.model_validate(payload)


def test_forecast_request_freq_defaults_to_auto_and_preserves_explicit_freq() -> None:
    payload = _example_request_payload()
    del payload["series"][0]["freq"]
    request = ForecastRequest.model_validate(payload)
    assert request.series[0].freq == "auto"

    payload = _example_request_payload()
    payload["series"][0]["freq"] = "H"
    request = ForecastRequest.model_validate(payload)
    assert request.series[0].freq == "H"


def test_forecast_request_rejects_invalid_payloads() -> None:
    invalid_horizon = _example_request_payload()
    invalid_horizon["horizon"] = "3"
    with pytest.raises(ValidationError):
        ForecastRequest.model_validate(invalid_horizon)

    invalid_series = _example_request_payload()
    series = invalid_series["series"][0]
    series["target"] = [1.0, 2.0]
    with pytest.raises(ValidationError):
        ForecastRequest.model_validate(invalid_series)

    invalid_quantiles = _example_request_payload()
    invalid_quantiles["quantiles"] = [0.9, 0.1]
    with pytest.raises(ValidationError):
        ForecastRequest.model_validate(invalid_quantiles)


def test_forecast_request_accepts_data_url_without_series() -> None:
    payload = {
        "model": "naive",
        "horizon": 3,
        "data_url": "file:///tmp/history.csv",
        "ingest": {"format": "csv"},
        "options": {},
    }
    request = ForecastRequest.model_validate(payload)
    assert request.data_url == "file:///tmp/history.csv"
    assert request.series == []


def test_forecast_request_requires_series_or_data_url() -> None:
    payload = {
        "model": "naive",
        "horizon": 3,
        "options": {},
    }
    with pytest.raises(ValidationError):
        ForecastRequest.model_validate(payload)


def test_forecast_request_rejects_series_and_data_url_together() -> None:
    payload = _example_request_payload()
    payload["data_url"] = "file:///tmp/history.csv"
    with pytest.raises(ValidationError):
        ForecastRequest.model_validate(payload)


def test_analyze_request_rejects_invalid_parameters() -> None:
    payload = _example_analyze_request_payload()
    payload["parameters"]["max_points"] = 0  # type: ignore[index]
    with pytest.raises(ValidationError):
        AnalyzeRequest.model_validate(payload)


def test_auto_forecast_request_rejects_invalid_strategy() -> None:
    payload = _example_auto_forecast_request_payload()
    payload["strategy"] = "unknown"
    with pytest.raises(ValidationError):
        AutoForecastRequest.model_validate(payload)


def test_analyze_request_requires_unique_series_ids() -> None:
    payload = _example_analyze_request_payload()
    payload["series"].append(  # type: ignore[union-attr]
        {
            "id": "series-1",
            "freq": "D",
            "timestamps": ["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-04"],
            "target": [1.0, 1.1, 1.2, 1.3],
        }
    )
    with pytest.raises(ValidationError):
        AnalyzeRequest.model_validate(payload)


def test_forecast_response_rejects_mismatched_quantile_lengths() -> None:
    payload = {
        "model": "naive",
        "forecasts": [
            {
                "id": "series-1",
                "freq": "D",
                "start_timestamp": "2025-01-04",
                "mean": [12.0, 13.0],
                "quantiles": {"0.5": [12.0]},
            }
        ],
    }

    with pytest.raises(ValidationError):
        ForecastResponse.model_validate(payload)


def test_analyze_response_rejects_unsorted_anomaly_indices() -> None:
    payload = {
        "results": [
            {
                "id": "series-1",
                "detected_frequency": "D",
                "seasonality_periods": [2, 4],
                "trend": {"direction": "up", "slope": 0.2, "r2": 0.8},
                "anomaly_indices": [3, 1],
                "stationarity_flag": False,
                "data_quality_score": 0.9,
            }
        ],
    }
    with pytest.raises(ValidationError):
        AnalyzeResponse.model_validate(payload)


def test_auto_forecast_response_rejects_strategy_mismatch() -> None:
    payload = {
        "strategy": "fastest",
        "selection": {
            "strategy": "auto",
            "chosen_model": "mock",
            "selected_models": ["mock"],
            "candidates": [
                {
                    "model": "mock",
                    "family": "mock",
                    "rank": 1,
                    "score": 1.0,
                    "reasons": ["test"],
                }
            ],
            "rationale": ["test"],
            "fallback_used": False,
        },
        "response": {
            "model": "mock",
            "forecasts": [
                {
                    "id": "series-1",
                    "freq": "D",
                    "start_timestamp": "2025-01-04",
                    "mean": [12.0, 13.0],
                }
            ],
        },
    }
    with pytest.raises(ValidationError):
        AutoForecastResponse.model_validate(payload)


def test_progressive_forecast_event_requires_response_on_completed_status() -> None:
    payload = {
        "event": "forecast.complete",
        "stage": 1,
        "strategy": "fastest",
        "model": "mock",
        "status": "completed",
        "final": True,
        "response": {
            "model": "mock",
            "forecasts": [
                {
                    "id": "series-1",
                    "freq": "D",
                    "start_timestamp": "2025-01-05",
                    "mean": [1.0, 1.0],
                }
            ],
        },
    }
    event = ProgressiveForecastEvent.model_validate(payload)
    assert event.status == "completed"
    assert event.response is not None

    invalid = dict(payload)
    invalid.pop("response")
    with pytest.raises(ValidationError):
        ProgressiveForecastEvent.model_validate(invalid)


def test_series_forecast_accepts_valid_quantiles() -> None:
    forecast = SeriesForecast.model_validate(
        {
            "id": "series-1",
            "freq": "D",
            "start_timestamp": "2025-01-04",
            "mean": [12.0, 13.0],
            "quantiles": {"0.1": [11.0, 12.0], "0.9": [13.0, 14.0]},
        }
    )
    assert forecast.quantiles is not None
    assert set(forecast.quantiles) == {"0.1", "0.9"}


def test_forecast_request_rejects_mixed_covariate_value_types() -> None:
    payload = _example_request_payload()
    series = payload["series"][0]
    series["past_covariates"] = {"promo": [1.0, "bad", 3.0]}
    with pytest.raises(ValidationError):
        ForecastRequest.model_validate(payload)


def test_forecast_request_rejects_future_only_covariates() -> None:
    payload = _example_request_payload()
    series = payload["series"][0]
    series["future_covariates"] = {"future_only": [1.0, 2.0, 3.0]}
    with pytest.raises(ValidationError):
        ForecastRequest.model_validate(payload)


def test_forecast_request_metrics_requires_actuals_and_horizon_alignment() -> None:
    payload = _example_request_payload()
    payload["parameters"] = {"metrics": {"names": ["mae", "rmse", "smape", "mape", "mase"]}}

    with pytest.raises(ValidationError):
        ForecastRequest.model_validate(payload)

    series = payload["series"][0]
    series["actuals"] = [12.0, 13.0]
    with pytest.raises(ValidationError):
        ForecastRequest.model_validate(payload)

    series["actuals"] = [12.0, 13.0, 14.0]
    request = ForecastRequest.model_validate(payload)
    assert request.parameters.metrics is not None
    assert request.parameters.metrics.names == ["mae", "rmse", "smape", "mape", "mase"]
    assert request.parameters.metrics.mase_seasonality == 1


def test_forecast_request_rejects_duplicate_metric_names() -> None:
    payload = _example_request_payload()
    payload["series"][0]["actuals"] = [12.0, 13.0, 14.0]
    payload["parameters"] = {"metrics": {"names": ["mape", "mape"]}}
    with pytest.raises(ValidationError):
        ForecastRequest.model_validate(payload)


def test_forecast_request_rejects_unknown_metric_names() -> None:
    payload = _example_request_payload()
    payload["series"][0]["actuals"] = [12.0, 13.0, 14.0]
    payload["parameters"] = {"metrics": {"names": ["mae", "unknown"]}}
    with pytest.raises(ValidationError):
        ForecastRequest.model_validate(payload)


def test_forecast_response_accepts_metrics_payload() -> None:
    payload = {
        "model": "naive",
        "forecasts": [
            {
                "id": "series-1",
                "freq": "D",
                "start_timestamp": "2025-01-04",
                "mean": [12.0, 13.0],
            }
        ],
        "metrics": {
            "aggregate": {"mape": 12.5, "mase": 0.8, "mae": 0.2, "rmse": 0.3, "smape": 14.0},
            "series": [
                {
                    "id": "series-1",
                    "values": {
                        "mape": 12.5,
                        "mase": 0.8,
                        "mae": 0.2,
                        "rmse": 0.3,
                        "smape": 14.0,
                    },
                }
            ],
        },
        "timing": {
            "model_load_ms": 1.2,
            "inference_ms": 4.8,
            "total_ms": 7.5,
        },
        "usage": {
            "runner": "tollama-mock",
            "device": "cpu",
            "peak_memory_mb": 42.0,
        },
    }
    response = ForecastResponse.model_validate(payload)
    assert response.metrics is not None
    assert response.metrics.aggregate["mape"] == 12.5
    assert response.metrics.aggregate["smape"] == 14.0
    assert response.timing is not None
    assert response.timing.total_ms == 7.5
    assert response.usage is not None
    assert response.usage["runner"] == "tollama-mock"


def test_what_if_response_accepts_baseline_and_results() -> None:
    payload = {
        "model": "naive",
        "horizon": 2,
        "baseline": {
            "model": "naive",
            "forecasts": [
                {
                    "id": "series-1",
                    "freq": "D",
                    "start_timestamp": "2025-01-04",
                    "mean": [12.0, 13.0],
                }
            ],
        },
        "results": [
            {
                "scenario": "high_demand",
                "ok": True,
                "response": {
                    "model": "naive",
                    "forecasts": [
                        {
                            "id": "series-1",
                            "freq": "D",
                            "start_timestamp": "2025-01-04",
                            "mean": [13.0, 14.0],
                        }
                    ],
                },
            }
        ],
        "summary": {"requested_scenarios": 1, "succeeded": 1, "failed": 0},
    }
    response = WhatIfResponse.model_validate(payload)
    assert response.summary.requested_scenarios == 1
    assert response.results[0].scenario == "high_demand"


def test_pipeline_response_accepts_analysis_recommendation_and_auto_forecast() -> None:
    payload = {
        "analysis": {
            "results": [
                {
                    "id": "series-1",
                    "detected_frequency": "D",
                    "seasonality_periods": [2],
                    "trend": {"direction": "up", "slope": 0.2, "r2": 0.8},
                    "anomaly_indices": [],
                    "stationarity_flag": False,
                    "data_quality_score": 0.9,
                }
            ]
        },
        "recommendation": {
            "request": {"horizon": 3, "freq": "D", "top_k": 3},
            "recommendations": [{"model": "mock", "family": "mock", "rank": 1, "score": 100}],
            "excluded": [],
            "total_candidates": 1,
            "compatible_candidates": 1,
        },
        "auto_forecast": {
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
                        "score": 1.0,
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
                        "id": "series-1",
                        "freq": "D",
                        "start_timestamp": "2025-01-04",
                        "mean": [12.0, 13.0],
                    }
                ],
            },
        },
    }
    response = PipelineResponse.model_validate(payload)
    assert response.analysis.results[0].id == "series-1"
    assert response.auto_forecast.response.model == "mock"


def test_pipeline_response_accepts_top_level_narrative() -> None:
    payload = {
        "analysis": {
            "results": [
                {
                    "id": "series-1",
                    "detected_frequency": "D",
                    "seasonality_periods": [],
                    "trend": {"direction": "up", "slope": 0.2, "r2": 0.8},
                    "anomaly_indices": [],
                    "stationarity_flag": True,
                    "data_quality_score": 0.9,
                }
            ]
        },
        "recommendation": {"request": {"horizon": 3}},
        "auto_forecast": {
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
                        "score": 1.0,
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
                        "id": "series-1",
                        "freq": "D",
                        "start_timestamp": "2025-01-04",
                        "mean": [12.0, 13.0],
                    }
                ],
            },
        },
        "narrative": {
            "summary": "pipeline summary",
            "chosen_model": "mock",
            "warnings_count": 0,
        },
    }
    response = PipelineResponse.model_validate(payload)
    assert response.narrative is not None
    assert response.narrative.chosen_model == "mock"


def test_generate_response_accepts_generated_series_payload() -> None:
    payload = {
        "method": "statistical",
        "generated": [
            {
                "id": "series-1_synthetic_1",
                "source_id": "series-1",
                "freq": "D",
                "timestamps": ["2025-01-01", "2025-01-02", "2025-01-03"],
                "target": [10.0, 11.0, 12.0],
            }
        ],
    }
    response = GenerateResponse.model_validate(payload)
    assert response.generated[0].source_id == "series-1"


def test_counterfactual_response_accepts_results_payload() -> None:
    payload = {
        "model": "mock",
        "horizon": 2,
        "intervention_index": 3,
        "baseline": {
            "model": "mock",
            "forecasts": [
                {
                    "id": "series-1",
                    "freq": "D",
                    "start_timestamp": "2025-01-04",
                    "mean": [12.0, 13.0],
                }
            ],
        },
        "results": [
            {
                "id": "series-1",
                "actual": [20.0, 22.0],
                "counterfactual": [12.0, 13.0],
                "delta": [8.0, 9.0],
                "absolute_delta": [8.0, 9.0],
                "mean_absolute_delta": 8.5,
                "total_delta": 17.0,
                "average_delta_pct": 40.0,
                "direction": "above_counterfactual",
            }
        ],
    }
    response = CounterfactualResponse.model_validate(payload)
    assert response.results[0].id == "series-1"
    assert response.results[0].direction == "above_counterfactual"


def test_scenario_tree_response_accepts_flattened_nodes() -> None:
    payload = {
        "model": "mock",
        "depth": 2,
        "branch_quantiles": [0.1, 0.9],
        "nodes": [
            {
                "node_id": "node_1",
                "parent_id": None,
                "series_id": "series-1",
                "depth": 0,
                "step": 0,
                "branch": "root",
                "value": 12.0,
                "probability": 1.0,
            },
            {
                "node_id": "node_2",
                "parent_id": "node_1",
                "series_id": "series-1",
                "depth": 1,
                "step": 1,
                "branch": "q0.1",
                "quantile": 0.1,
                "value": 11.0,
                "probability": 0.5,
            },
        ],
    }
    response = ScenarioTreeResponse.model_validate(payload)
    assert response.nodes[0].parent_id is None
    assert response.nodes[1].quantile == pytest.approx(0.1)


def test_forecast_report_accepts_composite_payload() -> None:
    payload = {
        "analysis": {
            "results": [
                {
                    "id": "series-1",
                    "detected_frequency": "D",
                    "seasonality_periods": [2],
                    "trend": {"direction": "up", "slope": 0.1, "r2": 0.5},
                    "anomaly_indices": [],
                    "anomalies": [],
                    "stationarity_flag": True,
                    "data_quality_score": 0.9,
                }
            ]
        },
        "recommendation": {"request": {"horizon": 2, "freq": "D"}},
        "forecast": {
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
                        "score": 1.0,
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
                        "id": "series-1",
                        "freq": "D",
                        "start_timestamp": "2025-01-04",
                        "mean": [12.0, 13.0],
                    }
                ],
            },
        },
        "baseline": {
            "model": "mock",
            "forecasts": [
                {
                    "id": "series-1",
                    "freq": "D",
                    "start_timestamp": "2025-01-04",
                    "mean": [12.0, 13.0],
                }
            ],
        },
    }
    report = ForecastReport.model_validate(payload)
    assert report.forecast.selection.chosen_model == "mock"
