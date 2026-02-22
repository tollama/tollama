"""Tests for TUI forecast/model client helpers."""

from __future__ import annotations

import json

import httpx
import pytest

from tollama.tui.client import DashboardAPIClient, DashboardAPIError


@pytest.mark.asyncio
async def test_dashboard_api_client_forecast_and_model_actions() -> None:
    def _handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/forecast" and request.method == "POST":
            return httpx.Response(
                status_code=200,
                json={
                    "model": "mock",
                    "created_at": "2025-01-01T00:00:00Z",
                    "forecasts": [{"id": "s1", "mean": [1.0, 2.0, 3.0]}],
                },
            )
        if request.url.path == "/api/show" and request.method == "POST":
            return httpx.Response(status_code=200, json={"name": "mock", "family": "mock"})
        if request.url.path == "/api/delete" and request.method == "DELETE":
            return httpx.Response(status_code=200, json={"deleted": True, "model": "mock"})
        return httpx.Response(status_code=404, json={"detail": "not found"})

    transport = httpx.MockTransport(_handler)
    client = DashboardAPIClient(
        base_url="http://localhost:11435",
        timeout=10.0,
        api_key="secret",
        transport=transport,
    )

    forecast = await client.forecast(
        {
            "model": "mock",
            "horizon": 3,
            "series": [{"id": "s1", "freq": "D", "timestamps": ["2025-01-01"], "target": [1.0]}],
            "options": {},
        }
    )
    show = await client.show_model("mock")
    removed = await client.delete_model("mock")

    assert forecast["model"] == "mock"
    assert show["name"] == "mock"
    assert removed["deleted"] is True


@pytest.mark.asyncio
async def test_dashboard_api_client_pull_model_events_supports_accept_license() -> None:
    captured_payloads: list[dict[str, object]] = []

    def _handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/pull" and request.method == "POST":
            payload = request.read().decode("utf-8")
            captured_payloads.append(json.loads(payload))
            return httpx.Response(
                status_code=200,
                text='{"status":"pulling manifest"}\n{"status":"success","done":true}\n',
            )
        return httpx.Response(status_code=404, json={"detail": "not found"})

    transport = httpx.MockTransport(_handler)
    client = DashboardAPIClient(
        base_url="http://localhost:11435",
        timeout=10.0,
        api_key=None,
        transport=transport,
    )

    events = await client.pull_model_events("chronos2", accept_license=True)

    assert len(captured_payloads) == 1
    assert captured_payloads[0]["accept_license"] is True
    assert captured_payloads[0]["stream"] is True
    assert events[-1]["done"] is True


@pytest.mark.asyncio
async def test_dashboard_api_client_surfaces_structured_api_errors() -> None:
    def _handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/pull" and request.method == "POST":
            return httpx.Response(
                status_code=409,
                json={
                    "detail": "model 'x' requires license acceptance; pass accept_license=True",
                    "hint": "Re-run with --accept-license",
                },
            )
        return httpx.Response(status_code=404, json={"detail": "not found"})

    transport = httpx.MockTransport(_handler)
    client = DashboardAPIClient(
        base_url="http://localhost:11435",
        timeout=10.0,
        api_key=None,
        transport=transport,
    )

    with pytest.raises(DashboardAPIError) as exc_info:
        await client.pull_model_events("chronos2")

    error = exc_info.value
    assert error.status_code == 409
    assert "license acceptance" in error.detail
    assert error.hint == "Re-run with --accept-license"
