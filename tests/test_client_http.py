"""Tests for shared tollama HTTP client behavior and error mapping."""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from tollama.client import (
    AsyncTollamaClient,
    DaemonUnreachableError,
    ForecastTimeoutError,
    InvalidRequestError,
    LicenseRequiredError,
    ModelMissingError,
    TollamaClient,
)


def _request_payload() -> dict[str, Any]:
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


def _response_payload() -> dict[str, Any]:
    return {
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


def _analyze_request_payload() -> dict[str, Any]:
    return {
        "series": [
            {
                "id": "s1",
                "freq": "D",
                "timestamps": ["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-04"],
                "target": [1.0, 2.0, 1.5, 2.5],
            }
        ],
        "parameters": {"max_lag": 2, "top_k_seasonality": 1},
    }


def _analyze_response_payload() -> dict[str, Any]:
    return {
        "results": [
            {
                "id": "s1",
                "detected_frequency": "D",
                "seasonality_periods": [2],
                "trend": {"direction": "up", "slope": 0.2, "r2": 0.4},
                "anomaly_indices": [],
                "stationarity_flag": False,
                "data_quality_score": 0.9,
            }
        ]
    }


def _auto_forecast_request_payload() -> dict[str, Any]:
    return {
        "horizon": 2,
        "strategy": "auto",
        "series": _request_payload()["series"],
        "options": {},
    }


def _auto_forecast_response_payload() -> dict[str, Any]:
    return {
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
                    "score": 120.0,
                    "reasons": ["fallback"],
                }
            ],
            "rationale": ["selected mock"],
            "fallback_used": False,
        },
        "response": _response_payload(),
    }


def _what_if_request_payload() -> dict[str, Any]:
    return {
        "model": "mock",
        "horizon": 2,
        "series": _request_payload()["series"],
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


def _what_if_response_payload() -> dict[str, Any]:
    return {
        "model": "mock",
        "horizon": 2,
        "baseline": _response_payload(),
        "results": [
            {
                "scenario": "high_demand",
                "ok": True,
                "response": _response_payload(),
            }
        ],
        "summary": {"requested_scenarios": 1, "succeeded": 1, "failed": 0},
    }


def _pipeline_request_payload() -> dict[str, Any]:
    return {
        "horizon": 2,
        "strategy": "auto",
        "series": _request_payload()["series"],
        "options": {},
        "pull_if_missing": True,
        "recommend_top_k": 3,
    }


def _pipeline_response_payload() -> dict[str, Any]:
    return {
        "analysis": _analyze_response_payload(),
        "recommendation": {
            "request": {"horizon": 2, "freq": "D", "top_k": 3},
            "recommendations": [{"model": "mock", "family": "mock", "rank": 1, "score": 100}],
            "excluded": [],
            "total_candidates": 1,
            "compatible_candidates": 1,
        },
        "pulled_model": None,
        "auto_forecast": _auto_forecast_response_payload(),
    }


def _client(handler: httpx.MockTransport) -> TollamaClient:
    return TollamaClient(base_url="http://daemon.test", timeout=3.0, transport=handler)


def _async_client(handler: httpx.MockTransport) -> AsyncTollamaClient:
    return AsyncTollamaClient(base_url="http://daemon.test", timeout=3.0, transport=handler)


def test_forecast_non_stream_falls_back_to_v1_on_api_404() -> None:
    paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        if request.url.path == "/api/forecast":
            return httpx.Response(404, json={"detail": "not found"})
        if request.url.path == "/v1/forecast":
            return httpx.Response(200, json=_response_payload())
        return httpx.Response(500, json={"detail": "unexpected path"})

    client = _client(httpx.MockTransport(handler))
    result = client.forecast(_request_payload(), stream=False)

    assert isinstance(result, dict)
    assert result["model"] == "mock"
    assert paths == ["/api/forecast", "/v1/forecast"]


def test_models_available_reads_api_info() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/info"
        return httpx.Response(
            200,
            json={"models": {"available": [{"name": "mock", "family": "mock"}]}},
        )

    client = _client(httpx.MockTransport(handler))
    models = client.models("available")

    assert models == [{"name": "mock", "family": "mock"}]


def test_analyze_returns_typed_response() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/analyze"
        return httpx.Response(200, json=_analyze_response_payload())

    client = _client(httpx.MockTransport(handler))
    response = client.analyze(_analyze_request_payload())

    assert len(response.results) == 1
    assert response.results[0].id == "s1"


def test_auto_forecast_returns_typed_response() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/auto-forecast"
        return httpx.Response(200, json=_auto_forecast_response_payload())

    client = _client(httpx.MockTransport(handler))
    response = client.auto_forecast(_auto_forecast_request_payload())

    assert response.strategy == "auto"
    assert response.selection.chosen_model == "mock"
    assert response.response.model == "mock"


def test_what_if_returns_typed_response() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/what-if"
        return httpx.Response(200, json=_what_if_response_payload())

    client = _client(httpx.MockTransport(handler))
    response = client.what_if(_what_if_request_payload())

    assert response.model == "mock"
    assert response.summary.succeeded == 1
    assert response.results[0].scenario == "high_demand"


def test_pipeline_returns_typed_response() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/pipeline"
        return httpx.Response(200, json=_pipeline_response_payload())

    client = _client(httpx.MockTransport(handler))
    response = client.pipeline(_pipeline_request_payload())

    assert response.analysis.results[0].id == "s1"
    assert response.auto_forecast.selection.chosen_model == "mock"


def test_client_includes_api_key_header_when_configured() -> None:
    seen_header = {"authorization": None}

    def handler(request: httpx.Request) -> httpx.Response:
        seen_header["authorization"] = request.headers.get("Authorization")
        return httpx.Response(200, json={"models": []})

    client = TollamaClient(
        base_url="http://daemon.test",
        timeout=3.0,
        api_key="top-secret",
        transport=httpx.MockTransport(handler),
    )
    client.list_tags()

    assert seen_header["authorization"] == "Bearer top-secret"


def test_show_404_maps_to_model_missing_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/show"
        return httpx.Response(404, json={"detail": "model missing"})

    client = _client(httpx.MockTransport(handler))

    with pytest.raises(ModelMissingError) as exc_info:
        client.show_model("missing")

    assert exc_info.value.exit_code == 4


def test_pull_409_license_maps_to_license_required_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/pull"
        return httpx.Response(409, json={"detail": "license requires accept_license"})

    client = _client(httpx.MockTransport(handler))

    with pytest.raises(LicenseRequiredError) as exc_info:
        client.pull("moirai", accept_license=False)

    assert exc_info.value.exit_code == 5


def test_validate_400_maps_to_invalid_request_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/validate"
        return httpx.Response(400, json={"detail": "invalid request"})

    client = _client(httpx.MockTransport(handler))

    with pytest.raises(InvalidRequestError) as exc_info:
        client.validate_request({})

    assert exc_info.value.exit_code == 2


def test_connect_error_maps_to_daemon_unreachable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    client = _client(httpx.MockTransport(handler))

    with pytest.raises(DaemonUnreachableError) as exc_info:
        client.list_tags()

    assert exc_info.value.exit_code == 3


def test_timeout_error_maps_to_forecast_timeout() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=request)

    client = _client(httpx.MockTransport(handler))

    with pytest.raises(ForecastTimeoutError) as exc_info:
        client.forecast(_request_payload(), stream=False)

    assert exc_info.value.exit_code == 6


@pytest.mark.asyncio
async def test_async_forecast_non_stream_falls_back_to_v1_on_api_404() -> None:
    paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        if request.url.path == "/api/forecast":
            return httpx.Response(404, json={"detail": "not found"})
        if request.url.path == "/v1/forecast":
            return httpx.Response(200, json=_response_payload())
        return httpx.Response(500, json={"detail": "unexpected path"})

    client = _async_client(httpx.MockTransport(handler))
    result = await client.forecast(_request_payload(), stream=False)

    assert isinstance(result, dict)
    assert result["model"] == "mock"
    assert paths == ["/api/forecast", "/v1/forecast"]


@pytest.mark.asyncio
async def test_async_models_available_reads_api_info() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/info"
        return httpx.Response(
            200,
            json={"models": {"available": [{"name": "mock", "family": "mock"}]}},
        )

    client = _async_client(httpx.MockTransport(handler))
    models = await client.models("available")

    assert models == [{"name": "mock", "family": "mock"}]


@pytest.mark.asyncio
async def test_async_analyze_returns_typed_response() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/analyze"
        return httpx.Response(200, json=_analyze_response_payload())

    client = _async_client(httpx.MockTransport(handler))
    response = await client.analyze(_analyze_request_payload())

    assert len(response.results) == 1
    assert response.results[0].detected_frequency == "D"


@pytest.mark.asyncio
async def test_async_auto_forecast_returns_typed_response() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/auto-forecast"
        return httpx.Response(200, json=_auto_forecast_response_payload())

    client = _async_client(httpx.MockTransport(handler))
    response = await client.auto_forecast(_auto_forecast_request_payload())

    assert response.selection.chosen_model == "mock"
    assert response.response.forecasts[0].id == "s1"


@pytest.mark.asyncio
async def test_async_what_if_returns_typed_response() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/what-if"
        return httpx.Response(200, json=_what_if_response_payload())

    client = _async_client(httpx.MockTransport(handler))
    response = await client.what_if(_what_if_request_payload())

    assert response.model == "mock"
    assert response.summary.requested_scenarios == 1
    assert response.results[0].ok is True


@pytest.mark.asyncio
async def test_async_pipeline_returns_typed_response() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/pipeline"
        return httpx.Response(200, json=_pipeline_response_payload())

    client = _async_client(httpx.MockTransport(handler))
    response = await client.pipeline(_pipeline_request_payload())

    assert response.analysis.results[0].id == "s1"
    assert response.auto_forecast.selection.chosen_model == "mock"


@pytest.mark.asyncio
async def test_async_client_includes_api_key_header_when_configured() -> None:
    seen_header = {"authorization": None}

    def handler(request: httpx.Request) -> httpx.Response:
        seen_header["authorization"] = request.headers.get("Authorization")
        return httpx.Response(200, json={"models": []})

    client = AsyncTollamaClient(
        base_url="http://daemon.test",
        timeout=3.0,
        api_key="top-secret",
        transport=httpx.MockTransport(handler),
    )
    await client.list_tags()

    assert seen_header["authorization"] == "Bearer top-secret"
