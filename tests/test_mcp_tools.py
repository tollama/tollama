"""Tests for MCP tool handlers independent of MCP transport runtime."""

from __future__ import annotations

from typing import Any

import pytest

from tollama.client import ModelMissingError
from tollama.core.schemas import ForecastResponse
from tollama.mcp.tools import (
    MCPToolError,
    tollama_forecast,
    tollama_health,
    tollama_models,
    tollama_show,
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


def test_tollama_show_model_missing_maps_to_mcp_error(monkeypatch) -> None:
    class _FakeClient:
        def show(self, _model: str) -> dict[str, Any]:
            raise ModelMissingError(action="show model", status_code=404, detail="missing")

    monkeypatch.setattr("tollama.mcp.tools._make_client", lambda **_: _FakeClient())

    with pytest.raises(MCPToolError) as exc_info:
        tollama_show(model="missing")

    assert exc_info.value.exit_code == 4
    assert exc_info.value.category == "MODEL_MISSING"
