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
