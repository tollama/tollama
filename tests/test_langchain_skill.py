"""Tests for LangChain tool wrappers."""

from __future__ import annotations

import importlib
import importlib.util
from typing import Any

import pytest

from tollama.client import ModelMissingError
from tollama.core.schemas import (
    AnalyzeResponse,
    AutoForecastResponse,
    CompareResponse,
    ForecastResponse,
)

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


def test_get_tollama_tools_returns_preconfigured_tools(langchain_tools) -> None:
    tools = langchain_tools.get_tollama_tools(base_url="http://daemon.test", timeout=7.0)

    assert [tool.name for tool in tools] == [
        "tollama_forecast",
        "tollama_auto_forecast",
        "tollama_analyze",
        "tollama_compare",
        "tollama_recommend",
        "tollama_health",
        "tollama_models",
    ]
    for tool in tools:
        assert tool.base_url == "http://daemon.test"
        assert tool.timeout == 7.0


def test_tool_descriptions_include_usage_guidance(langchain_tools) -> None:
    forecast_tool = langchain_tools.TollamaForecastTool()
    auto_forecast_tool = langchain_tools.TollamaAutoForecastTool()
    analyze_tool = langchain_tools.TollamaAnalyzeTool()
    compare_tool = langchain_tools.TollamaCompareTool()
    recommend_tool = langchain_tools.TollamaRecommendTool()
    models_tool = langchain_tools.TollamaModelsTool()
    health_tool = langchain_tools.TollamaHealthTool()

    assert "request" in forecast_tool.description
    assert "horizon" in forecast_tool.description
    assert "series" in forecast_tool.description
    assert "Example:" in forecast_tool.description

    assert "auto-forecast" in auto_forecast_tool.description
    assert "strategy" in auto_forecast_tool.description
    assert "Example:" in auto_forecast_tool.description

    assert "series" in analyze_tool.description
    assert "seasonality" in analyze_tool.description
    assert "Example:" in analyze_tool.description

    assert "models" in compare_tool.description
    assert "ok=true/false" in compare_tool.description
    assert "Example:" in compare_tool.description

    assert "horizon" in recommend_tool.description
    assert "covariates" in recommend_tool.description
    assert "Example:" in recommend_tool.description

    assert "mode" in models_tool.description
    assert "available" in models_tool.description
    assert "Example:" in models_tool.description

    assert "status" in health_tool.description
    assert "Example:" in health_tool.description


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


def test_tollama_auto_forecast_tool_success(monkeypatch, langchain_tools) -> None:
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
                    "response": _forecast_response().model_dump(mode="json", exclude_none=True),
                },
            )

    monkeypatch.setattr("tollama.skill.langchain._make_client", lambda **_: _FakeClient())
    tool = langchain_tools.TollamaAutoForecastTool()

    payload = tool._run(request=_auto_forecast_request())

    assert payload["selection"]["chosen_model"] == "mock"
    assert payload["response"]["model"] == "mock"


def test_tollama_auto_forecast_tool_invalid_request_maps_to_invalid_request(
    langchain_tools,
) -> None:
    tool = langchain_tools.TollamaAutoForecastTool()

    payload = tool._run(request={"horizon": 2})

    assert payload["error"]["category"] == "INVALID_REQUEST"
    assert payload["error"]["exit_code"] == 2


def test_tollama_analyze_tool_success(monkeypatch, langchain_tools) -> None:
    class _FakeClient:
        def analyze(self, _request: Any) -> AnalyzeResponse:
            return AnalyzeResponse.model_validate(
                {
                    "results": [
                        {
                            "id": "s1",
                            "detected_frequency": "D",
                            "seasonality_periods": [2],
                            "trend": {"direction": "up", "slope": 0.1, "r2": 0.4},
                            "anomaly_indices": [],
                            "stationarity_flag": False,
                            "data_quality_score": 0.9,
                        }
                    ],
                },
            )

    monkeypatch.setattr("tollama.skill.langchain._make_client", lambda **_: _FakeClient())
    tool = langchain_tools.TollamaAnalyzeTool()

    payload = tool._run(request=_analyze_request())

    assert payload["results"][0]["id"] == "s1"


def test_tollama_analyze_tool_invalid_request_maps_to_invalid_request(langchain_tools) -> None:
    tool = langchain_tools.TollamaAnalyzeTool()

    payload = tool._run(request={"series": []})

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


@pytest.mark.asyncio
async def test_tollama_health_tool_arun_uses_async_client(monkeypatch, langchain_tools) -> None:
    class _FakeAsyncClient:
        async def health(self) -> dict[str, Any]:
            return {"health": {"status": "ok"}, "version": {"version": "0.2.0"}}

    monkeypatch.setattr(
        "tollama.skill.langchain._make_async_client",
        lambda **_: _FakeAsyncClient(),
    )
    tool = langchain_tools.TollamaHealthTool()

    payload = await tool._arun()

    assert payload["healthy"] is True
    assert payload["version"]["version"] == "0.2.0"


@pytest.mark.asyncio
async def test_tollama_forecast_tool_arun_uses_async_client(monkeypatch, langchain_tools) -> None:
    class _FakeAsyncClient:
        async def forecast_response(self, _request: Any) -> ForecastResponse:
            return _forecast_response()

    monkeypatch.setattr(
        "tollama.skill.langchain._make_async_client",
        lambda **_: _FakeAsyncClient(),
    )
    tool = langchain_tools.TollamaForecastTool()

    payload = await tool._arun(request=_forecast_request())

    assert payload["model"] == "mock"
    assert payload["forecasts"][0]["id"] == "s1"


@pytest.mark.asyncio
async def test_tollama_auto_forecast_tool_arun_uses_async_client(
    monkeypatch,
    langchain_tools,
) -> None:
    class _FakeAsyncClient:
        async def auto_forecast(self, _request: Any) -> AutoForecastResponse:
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
                    "response": _forecast_response().model_dump(mode="json", exclude_none=True),
                },
            )

    monkeypatch.setattr(
        "tollama.skill.langchain._make_async_client",
        lambda **_: _FakeAsyncClient(),
    )
    tool = langchain_tools.TollamaAutoForecastTool()

    payload = await tool._arun(request=_auto_forecast_request())

    assert payload["selection"]["chosen_model"] == "mock"
    assert payload["response"]["forecasts"][0]["id"] == "s1"


@pytest.mark.asyncio
async def test_tollama_analyze_tool_arun_uses_async_client(monkeypatch, langchain_tools) -> None:
    class _FakeAsyncClient:
        async def analyze(self, _request: Any) -> AnalyzeResponse:
            return AnalyzeResponse.model_validate(
                {
                    "results": [
                        {
                            "id": "s1",
                            "detected_frequency": "D",
                            "seasonality_periods": [2],
                            "trend": {"direction": "up", "slope": 0.1, "r2": 0.4},
                            "anomaly_indices": [],
                            "stationarity_flag": False,
                            "data_quality_score": 0.9,
                        }
                    ],
                },
            )

    monkeypatch.setattr(
        "tollama.skill.langchain._make_async_client",
        lambda **_: _FakeAsyncClient(),
    )
    tool = langchain_tools.TollamaAnalyzeTool()

    payload = await tool._arun(request=_analyze_request())

    assert payload["results"][0]["detected_frequency"] == "D"


@pytest.mark.asyncio
async def test_tollama_compare_tool_arun_uses_async_client(monkeypatch, langchain_tools) -> None:
    class _FakeAsyncClient:
        async def compare(self, _request: Any) -> Any:
            return CompareResponse.model_validate(
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
                                        "start_timestamp": "2025-01-03",
                                        "mean": [3.0, 4.0],
                                    }
                                ],
                            },
                        },
                        {
                            "model": "chronos2",
                            "ok": False,
                            "error": {
                                "category": "RUNNER_UNAVAILABLE",
                                "status_code": 503,
                                "message": "runner unavailable",
                            },
                        },
                    ],
                    "summary": {"requested_models": 2, "succeeded": 1, "failed": 1},
                },
            )

    monkeypatch.setattr(
        "tollama.skill.langchain._make_async_client",
        lambda **_: _FakeAsyncClient(),
    )
    tool = langchain_tools.TollamaCompareTool()
    payload = await tool._arun(
        request={
            "models": ["mock", "chronos2"],
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
        },
    )

    assert payload["summary"]["succeeded"] == 1


@pytest.mark.asyncio
async def test_tollama_models_tool_arun_uses_async_client(monkeypatch, langchain_tools) -> None:
    class _FakeAsyncClient:
        async def models(self, mode: str = "installed") -> list[dict[str, Any]]:
            return [{"name": "mock", "mode": mode}]

    monkeypatch.setattr(
        "tollama.skill.langchain._make_async_client",
        lambda **_: _FakeAsyncClient(),
    )
    tool = langchain_tools.TollamaModelsTool()

    payload = await tool._arun(mode="available")

    assert payload["mode"] == "available"
    assert payload["items"] == [{"name": "mock", "mode": "available"}]


@pytest.mark.asyncio
async def test_tollama_recommend_tool_arun_success(langchain_tools) -> None:
    tool = langchain_tools.TollamaRecommendTool()

    payload = await tool._arun(
        horizon=24,
        has_past_covariates=True,
        has_future_covariates=True,
        covariates_type="numeric",
        top_k=3,
    )

    assert payload["request"]["horizon"] == 24
    assert payload["recommendations"]
