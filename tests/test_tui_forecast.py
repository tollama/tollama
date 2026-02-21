"""Tests for TUI forecast/model client helpers."""

from __future__ import annotations

import httpx
import pytest

from tollama.tui.client import DashboardAPIClient


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
