"""Tests for Textual dashboard entrypoints and snapshot client helpers."""

from __future__ import annotations

import httpx
import pytest

from tollama.tui.app import run_dashboard_app
from tollama.tui.client import DashboardAPIClient


class _FakeApp:
    def __init__(self, *, base_url: str, timeout: float, api_key: str | None) -> None:
        self.base_url = base_url
        self.timeout = timeout
        self.api_key = api_key
        self.ran = False

    def run(self) -> None:
        self.ran = True


def test_run_dashboard_app_invokes_app_runner(monkeypatch) -> None:
    instance: dict[str, _FakeApp] = {}

    def _factory(*, base_url: str, timeout: float, api_key: str | None) -> _FakeApp:
        app = _FakeApp(base_url=base_url, timeout=timeout, api_key=api_key)
        instance["app"] = app
        return app

    monkeypatch.setattr("tollama.tui.app.TollamaDashboardApp", _factory)

    run_dashboard_app(base_url="http://localhost:11435", timeout=10.0, api_key="secret")

    assert instance["app"].ran is True


@pytest.mark.asyncio
async def test_dashboard_api_client_fetches_snapshot() -> None:
    def _handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/dashboard/state":
            return httpx.Response(
                status_code=200,
                json={
                    "info": {"daemon": {"version": "0.1.0"}},
                    "ps": {"models": []},
                    "usage": {"summary": {"request_count": 0}},
                    "warnings": [],
                },
            )
        if request.url.path == "/api/tags":
            return httpx.Response(status_code=200, json={"models": [{"name": "mock"}]})
        return httpx.Response(status_code=404, json={"detail": "not found"})

    transport = httpx.MockTransport(_handler)
    client = DashboardAPIClient(
        base_url="http://localhost:11435",
        timeout=10.0,
        api_key=None,
        transport=transport,
    )

    snapshot = await client.dashboard_snapshot()

    assert snapshot.state["info"]["daemon"]["version"] == "0.1.0"
    assert snapshot.tags["models"][0]["name"] == "mock"
