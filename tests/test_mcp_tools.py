"""Tests for MCP tool handlers independent of MCP transport runtime."""

from __future__ import annotations

from typing import Any

import pytest

from tollama.client import ModelMissingError
from tollama.core.schemas import (
    AnalyzeResponse,
    AutoForecastResponse,
    CounterfactualResponse,
    ForecastReport,
    ForecastResponse,
    GenerateResponse,
    PipelineResponse,
    ScenarioTreeResponse,
    WhatIfResponse,
)
from tollama.mcp.tools import (
    MCPToolError,
    tollama_analyze,
    tollama_auto_forecast,
    tollama_counterfactual,
    tollama_forecast,
    tollama_generate,
    tollama_health,
    tollama_models,
    tollama_pipeline,
    tollama_report,
    tollama_scenario_tree,
    tollama_show,
    tollama_what_if,
)


def _forecast_request() -> dict[str, Any]:
    return {
        "model": "mock",
        "horizon": 2,
        "series": [
            {
                "id": "s1",
                "freq": "D",
                "timestamps": ["2025-01-01", "2025-01-02"],
                "target": [1.0, 2.0],
            }
        ],
        "options": {},
    }


def _forecast_response() -> ForecastResponse:
    return ForecastResponse.model_validate(
        {
            "model": "mock",
            "forecasts": [
                {
                    "id": "s1",
                    "freq": "D",
                    "start_timestamp": "2025-01-03",
                    "mean": [3.0, 4.0],
                }
            ],
        }
    )


def _analyze_request() -> dict[str, Any]:
    return {
        "series": [
            {
                "id": "s1",
                "freq": "D",
                "timestamps": ["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-04"],
                "target": [1.0, 2.0, 1.5, 2.5],
            }
        ]
    }


def _auto_forecast_request() -> dict[str, Any]:
    return {
        "horizon": 2,
        "strategy": "auto",
        "series": _forecast_request()["series"],
        "options": {},
    }


def _generate_request() -> dict[str, Any]:
    return {
        "series": [
            {
                "id": "s1",
                "freq": "D",
                "timestamps": [
                    "2025-01-01",
                    "2025-01-02",
                    "2025-01-03",
                    "2025-01-04",
                    "2025-01-05",
                ],
                "target": [1.0, 2.0, 3.0, 2.5, 3.5],
            }
        ],
        "count": 2,
        "length": 5,
        "seed": 7,
        "method": "statistical",
    }


def _what_if_request() -> dict[str, Any]:
    return {
        "model": "mock",
        "horizon": 2,
        "series": _forecast_request()["series"],
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


def _counterfactual_request() -> dict[str, Any]:
    return {
        "model": "mock",
        "intervention_index": 3,
        "series": [
            {
                "id": "s1",
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


def _scenario_tree_request() -> dict[str, Any]:
    return {
        "model": "mock",
        "horizon": 4,
        "depth": 2,
        "branch_quantiles": [0.1, 0.9],
        "series": _forecast_request()["series"],
        "options": {},
    }


def _report_request() -> dict[str, Any]:
    return {
        "horizon": 2,
        "strategy": "auto",
        "series": _forecast_request()["series"],
        "options": {},
        "pull_if_missing": True,
        "include_baseline": True,
    }


def _pipeline_request() -> dict[str, Any]:
    return {
        "horizon": 2,
        "strategy": "auto",
        "series": _forecast_request()["series"],
        "options": {},
        "pull_if_missing": True,
    }


def test_tollama_health_success(monkeypatch) -> None:
    class _FakeClient:
        def health(self) -> dict[str, Any]:
            return {"health": {"status": "ok"}, "version": {"version": "0.1.0"}}

    monkeypatch.setattr("tollama.mcp.tools._make_client", lambda **_: _FakeClient())

    payload = tollama_health()

    assert payload["healthy"] is True
    assert payload["version"]["version"] == "0.1.0"


def test_tollama_models_available_success(monkeypatch) -> None:
    class _FakeClient:
        def models(self, mode: str = "installed") -> list[dict[str, Any]]:
            return [{"name": "mock", "mode": mode}]

    monkeypatch.setattr("tollama.mcp.tools._make_client", lambda **_: _FakeClient())

    payload = tollama_models(mode="available")

    assert payload["mode"] == "available"
    assert payload["items"] == [{"name": "mock", "mode": "available"}]


def test_tollama_forecast_success(monkeypatch) -> None:
    class _FakeClient:
        def forecast_response(self, _request: Any) -> ForecastResponse:
            return _forecast_response()

    monkeypatch.setattr("tollama.mcp.tools._make_client", lambda **_: _FakeClient())

    payload = tollama_forecast(request=_forecast_request())

    assert payload["model"] == "mock"
    assert payload["forecasts"][0]["id"] == "s1"


def test_tollama_forecast_invalid_request_maps_to_invalid_request() -> None:
    with pytest.raises(MCPToolError) as exc_info:
        tollama_forecast(request={"model": "mock"})

    assert exc_info.value.exit_code == 2
    assert exc_info.value.category == "INVALID_REQUEST"


def test_tollama_analyze_success(monkeypatch) -> None:
    class _FakeClient:
        def analyze(self, _request: Any) -> AnalyzeResponse:
            return AnalyzeResponse.model_validate(
                {
                    "results": [
                        {
                            "id": "s1",
                            "detected_frequency": "D",
                            "seasonality_periods": [2],
                            "trend": {"direction": "up", "slope": 0.2, "r2": 0.7},
                            "anomaly_indices": [],
                            "stationarity_flag": False,
                            "data_quality_score": 0.9,
                        }
                    ],
                },
            )

    monkeypatch.setattr("tollama.mcp.tools._make_client", lambda **_: _FakeClient())

    payload = tollama_analyze(request=_analyze_request())

    assert payload["results"][0]["id"] == "s1"


def test_tollama_analyze_invalid_request_maps_to_invalid_request() -> None:
    with pytest.raises(MCPToolError) as exc_info:
        tollama_analyze(request={"series": []})

    assert exc_info.value.exit_code == 2
    assert exc_info.value.category == "INVALID_REQUEST"


def test_tollama_generate_success(monkeypatch) -> None:
    class _FakeClient:
        def generate(self, _request: Any) -> GenerateResponse:
            return GenerateResponse.model_validate(
                {
                    "method": "statistical",
                    "generated": [
                        {
                            "id": "s1_synthetic_1",
                            "source_id": "s1",
                            "freq": "D",
                            "timestamps": [
                                "2025-01-01",
                                "2025-01-02",
                                "2025-01-03",
                                "2025-01-04",
                                "2025-01-05",
                            ],
                            "target": [1.1, 1.9, 3.1, 2.4, 3.6],
                        }
                    ],
                }
            )

    monkeypatch.setattr("tollama.mcp.tools._make_client", lambda **_: _FakeClient())

    payload = tollama_generate(request=_generate_request())

    assert payload["method"] == "statistical"
    assert payload["generated"][0]["source_id"] == "s1"


def test_tollama_generate_invalid_request_maps_to_invalid_request() -> None:
    with pytest.raises(MCPToolError) as exc_info:
        tollama_generate(request={"series": []})

    assert exc_info.value.exit_code == 2
    assert exc_info.value.category == "INVALID_REQUEST"


def test_tollama_counterfactual_success(monkeypatch) -> None:
    class _FakeClient:
        def counterfactual(self, _request: Any) -> CounterfactualResponse:
            return CounterfactualResponse.model_validate(
                {
                    "model": "mock",
                    "horizon": 2,
                    "intervention_index": 3,
                    "baseline": _forecast_response().model_dump(mode="json", exclude_none=True),
                    "results": [
                        {
                            "id": "s1",
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
            )

    monkeypatch.setattr("tollama.mcp.tools._make_client", lambda **_: _FakeClient())

    payload = tollama_counterfactual(request=_counterfactual_request())

    assert payload["model"] == "mock"
    assert payload["results"][0]["direction"] == "above_counterfactual"


def test_tollama_counterfactual_invalid_request_maps_to_invalid_request() -> None:
    with pytest.raises(MCPToolError) as exc_info:
        tollama_counterfactual(request={"model": "mock"})

    assert exc_info.value.exit_code == 2
    assert exc_info.value.category == "INVALID_REQUEST"


def test_tollama_scenario_tree_success(monkeypatch) -> None:
    class _FakeClient:
        def scenario_tree(self, _request: Any) -> ScenarioTreeResponse:
            return ScenarioTreeResponse.model_validate(
                {
                    "model": "mock",
                    "depth": 2,
                    "branch_quantiles": [0.1, 0.9],
                    "nodes": [
                        {
                            "node_id": "node_1",
                            "parent_id": None,
                            "series_id": "s1",
                            "depth": 0,
                            "step": 0,
                            "branch": "root",
                            "value": 2.0,
                            "probability": 1.0,
                        },
                        {
                            "node_id": "node_2",
                            "parent_id": "node_1",
                            "series_id": "s1",
                            "depth": 1,
                            "step": 1,
                            "branch": "q0.1",
                            "quantile": 0.1,
                            "value": 2.0,
                            "probability": 0.5,
                        },
                    ],
                }
            )

    monkeypatch.setattr("tollama.mcp.tools._make_client", lambda **_: _FakeClient())

    payload = tollama_scenario_tree(request=_scenario_tree_request())

    assert payload["model"] == "mock"
    assert payload["nodes"][1]["parent_id"] == "node_1"


def test_tollama_scenario_tree_invalid_request_maps_to_invalid_request() -> None:
    with pytest.raises(MCPToolError) as exc_info:
        tollama_scenario_tree(request={"model": "mock"})

    assert exc_info.value.exit_code == 2
    assert exc_info.value.category == "INVALID_REQUEST"


def test_tollama_report_success(monkeypatch) -> None:
    class _FakeClient:
        def report(self, _request: Any) -> ForecastReport:
            return ForecastReport.model_validate(
                {
                    "analysis": {
                        "results": [
                            {
                                "id": "s1",
                                "detected_frequency": "D",
                                "seasonality_periods": [2],
                                "trend": {"direction": "up", "slope": 0.2, "r2": 0.7},
                                "anomaly_indices": [],
                                "stationarity_flag": False,
                                "data_quality_score": 0.9,
                            }
                        ],
                    },
                    "recommendation": {"request": {"horizon": 2, "freq": "D", "top_k": 3}},
                    "forecast": AutoForecastResponse.model_validate(
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
                                        "score": 100.0,
                                        "reasons": ["selected"],
                                    }
                                ],
                                "rationale": ["selected"],
                                "fallback_used": False,
                            },
                            "response": _forecast_response().model_dump(
                                mode="json",
                                exclude_none=True,
                            ),
                        }
                    ).model_dump(mode="json", exclude_none=True),
                    "baseline": _forecast_response().model_dump(mode="json", exclude_none=True),
                }
            )

    monkeypatch.setattr("tollama.mcp.tools._make_client", lambda **_: _FakeClient())

    payload = tollama_report(request=_report_request())

    assert payload["forecast"]["selection"]["chosen_model"] == "mock"
    assert payload["analysis"]["results"][0]["id"] == "s1"


def test_tollama_report_invalid_request_maps_to_invalid_request() -> None:
    with pytest.raises(MCPToolError) as exc_info:
        tollama_report(request={"horizon": 2})

    assert exc_info.value.exit_code == 2
    assert exc_info.value.category == "INVALID_REQUEST"


def test_tollama_auto_forecast_success(monkeypatch) -> None:
    class _FakeClient:
        def auto_forecast(self, _request: Any) -> AutoForecastResponse:
            return AutoForecastResponse.model_validate(
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
                                "score": 100.0,
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
                                "start_timestamp": "2025-01-03",
                                "mean": [3.0, 4.0],
                            }
                        ],
                    },
                },
            )

    monkeypatch.setattr("tollama.mcp.tools._make_client", lambda **_: _FakeClient())

    payload = tollama_auto_forecast(request=_auto_forecast_request())

    assert payload["strategy"] == "auto"
    assert payload["selection"]["chosen_model"] == "mock"
    assert payload["response"]["model"] == "mock"


def test_tollama_auto_forecast_invalid_request_maps_to_invalid_request() -> None:
    with pytest.raises(MCPToolError) as exc_info:
        tollama_auto_forecast(request={"horizon": 2})

    assert exc_info.value.exit_code == 2
    assert exc_info.value.category == "INVALID_REQUEST"


def test_tollama_what_if_success(monkeypatch) -> None:
    class _FakeClient:
        def what_if(self, _request: Any) -> WhatIfResponse:
            return WhatIfResponse.model_validate(
                {
                    "model": "mock",
                    "horizon": 2,
                    "baseline": {
                        "model": "mock",
                        "forecasts": [
                            {
                                "id": "s1",
                                "freq": "D",
                                "start_timestamp": "2025-01-03",
                                "mean": [3.0, 4.0],
                            }
                        ],
                    },
                    "results": [
                        {
                            "scenario": "high_demand",
                            "ok": True,
                            "response": {
                                "model": "mock",
                                "forecasts": [
                                    {
                                        "id": "s1",
                                        "freq": "D",
                                        "start_timestamp": "2025-01-03",
                                        "mean": [3.6, 4.8],
                                    }
                                ],
                            },
                        }
                    ],
                    "summary": {
                        "requested_scenarios": 1,
                        "succeeded": 1,
                        "failed": 0,
                    },
                },
            )

    monkeypatch.setattr("tollama.mcp.tools._make_client", lambda **_: _FakeClient())

    payload = tollama_what_if(request=_what_if_request())

    assert payload["model"] == "mock"
    assert payload["summary"]["succeeded"] == 1
    assert payload["results"][0]["scenario"] == "high_demand"


def test_tollama_what_if_invalid_request_maps_to_invalid_request() -> None:
    with pytest.raises(MCPToolError) as exc_info:
        tollama_what_if(request={"model": "mock"})

    assert exc_info.value.exit_code == 2
    assert exc_info.value.category == "INVALID_REQUEST"


def test_tollama_pipeline_success(monkeypatch) -> None:
    class _FakeClient:
        def pipeline(self, _request: Any) -> PipelineResponse:
            return PipelineResponse.model_validate(
                {
                    "analysis": {
                        "results": [
                            {
                                "id": "s1",
                                "detected_frequency": "D",
                                "seasonality_periods": [2],
                                "trend": {"direction": "up", "slope": 0.2, "r2": 0.7},
                                "anomaly_indices": [],
                                "stationarity_flag": False,
                                "data_quality_score": 0.9,
                            }
                        ],
                    },
                    "recommendation": {
                        "request": {"horizon": 2, "freq": "D", "top_k": 3},
                        "recommendations": [
                            {"model": "mock", "family": "mock", "rank": 1, "score": 100}
                        ],
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
                                    "score": 100.0,
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
                                    "start_timestamp": "2025-01-03",
                                    "mean": [3.0, 4.0],
                                }
                            ],
                        },
                    },
                },
            )

    monkeypatch.setattr("tollama.mcp.tools._make_client", lambda **_: _FakeClient())

    payload = tollama_pipeline(request=_pipeline_request())

    assert payload["analysis"]["results"][0]["id"] == "s1"
    assert payload["auto_forecast"]["selection"]["chosen_model"] == "mock"


def test_tollama_pipeline_invalid_request_maps_to_invalid_request() -> None:
    with pytest.raises(MCPToolError) as exc_info:
        tollama_pipeline(request={"horizon": 2})

    assert exc_info.value.exit_code == 2
    assert exc_info.value.category == "INVALID_REQUEST"


def test_tollama_show_model_missing_maps_to_mcp_error(monkeypatch) -> None:
    class _FakeClient:
        def show(self, _model: str) -> dict[str, Any]:
            raise ModelMissingError(action="show model", status_code=404, detail="missing")

    monkeypatch.setattr("tollama.mcp.tools._make_client", lambda **_: _FakeClient())

    with pytest.raises(MCPToolError) as exc_info:
        tollama_show(model="missing")

    assert exc_info.value.exit_code == 4
    assert exc_info.value.category == "MODEL_MISSING"
