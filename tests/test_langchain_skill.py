"""Tests for LangChain tool wrappers."""

from __future__ import annotations

import importlib
import importlib.util
from typing import Any

import pytest

from tollama.client import ModelMissingError
from tollama.core.schemas import ForecastResponse

_HAS_LANGCHAIN = importlib.util.find_spec("langchain_core") is not None
if _HAS_LANGCHAIN:
    _HAS_LANGCHAIN = importlib.util.find_spec("langchain_core.tools") is not None


@pytest.fixture
def langchain_tools():
    if not _HAS_LANGCHAIN:
        pytest.skip("langchain_core.tools is not installed")
    return importlib.import_module("tollama.skill.langchain")


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


def test_get_tollama_tools_returns_preconfigured_tools(langchain_tools) -> None:
    tools = langchain_tools.get_tollama_tools(base_url="http://daemon.test", timeout=7.0)

    assert [tool.name for tool in tools] == [
        "tollama_forecast",
        "tollama_health",
        "tollama_models",
    ]
    for tool in tools:
        assert tool.base_url == "http://daemon.test"
        assert tool.timeout == 7.0


def test_tollama_health_tool_success(monkeypatch, langchain_tools) -> None:
    class _FakeClient:
        def health(self) -> dict[str, Any]:
            return {"health": {"status": "ok"}, "version": {"version": "0.1.0"}}

    monkeypatch.setattr("tollama.skill.langchain._make_client", lambda **_: _FakeClient())
    tool = langchain_tools.TollamaHealthTool()

    payload = tool._run()

    assert payload["healthy"] is True
    assert payload["version"]["version"] == "0.1.0"


def test_tollama_models_tool_success(monkeypatch, langchain_tools) -> None:
    class _FakeClient:
        def models(self, mode: str = "installed") -> list[dict[str, Any]]:
            return [{"name": "mock", "mode": mode}]

    monkeypatch.setattr("tollama.skill.langchain._make_client", lambda **_: _FakeClient())
    tool = langchain_tools.TollamaModelsTool()

    payload = tool._run(mode="available")

    assert payload["mode"] == "available"
    assert payload["items"] == [{"name": "mock", "mode": "available"}]


def test_tollama_forecast_tool_success(monkeypatch, langchain_tools) -> None:
    class _FakeClient:
        def forecast_response(self, _request: Any) -> ForecastResponse:
            return _forecast_response()

    monkeypatch.setattr("tollama.skill.langchain._make_client", lambda **_: _FakeClient())
    tool = langchain_tools.TollamaForecastTool()

    payload = tool._run(request=_forecast_request())

    assert payload["model"] == "mock"
    assert payload["forecasts"][0]["id"] == "s1"


def test_tollama_forecast_tool_invalid_request_maps_to_invalid_request(langchain_tools) -> None:
    tool = langchain_tools.TollamaForecastTool()

    payload = tool._run(request={"model": "mock"})

    assert payload["error"]["category"] == "INVALID_REQUEST"
    assert payload["error"]["exit_code"] == 2


def test_tollama_forecast_tool_client_error_maps_to_error_payload(
    monkeypatch,
    langchain_tools,
) -> None:
    class _FakeClient:
        def forecast_response(self, _request: Any) -> ForecastResponse:
            raise ModelMissingError(action="forecast model", status_code=404, detail="missing")

    monkeypatch.setattr("tollama.skill.langchain._make_client", lambda **_: _FakeClient())
    tool = langchain_tools.TollamaForecastTool()

    payload = tool._run(request=_forecast_request())

    assert payload["error"]["category"] == "MODEL_MISSING"
    assert payload["error"]["exit_code"] == 4
