"""Tests for MCP tool handlers independent of MCP transport runtime."""

from __future__ import annotations

from typing import Any

import pytest

from tollama.client import ModelMissingError
from tollama.core.schemas import (
    AnalyzeResponse,
    AutoForecastResponse,
    ForecastResponse,
    PipelineResponse,
    WhatIfResponse,
)
from tollama.mcp.tools import (
    MCPToolError,
    tollama_analyze,
    tollama_auto_forecast,
    tollama_forecast,
    tollama_health,
    tollama_models,
    tollama_pipeline,
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
