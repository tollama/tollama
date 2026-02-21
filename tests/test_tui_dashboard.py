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


@pytest.mark.asyncio
async def test_stream_events_uses_unbounded_read_timeout(monkeypatch) -> None:
    captured_timeouts: list[object] = []

    class _FakeResponse:
        def raise_for_status(self) -> None:
            return None

        async def aiter_lines(self):
            yield "event: status"
            yield "data: ok"
            yield ""

    class _FakeStreamContext:
        def __init__(self) -> None:
            self._response = _FakeResponse()

        async def __aenter__(self) -> _FakeResponse:
            return self._response

        async def __aexit__(self, *_args: object) -> None:
            return None

    class _FakeAsyncClient:
        def __init__(self, *, base_url: str, timeout: object, transport: object) -> None:
            del base_url, transport
            captured_timeouts.append(timeout)

        async def __aenter__(self) -> _FakeAsyncClient:
            return self

        async def __aexit__(self, *_args: object) -> None:
            return None

        def stream(
            self,
            method: str,
            path: str,
            headers: dict[str, str] | None = None,
        ) -> _FakeStreamContext:
            assert method == "GET"
            assert path == "/api/events?heartbeat=15"
            assert headers == {"Accept": "text/event-stream"}
            return _FakeStreamContext()

    monkeypatch.setattr("tollama.tui.client.httpx.AsyncClient", _FakeAsyncClient)

    client = DashboardAPIClient(
        base_url="http://localhost:11435",
        timeout=10.0,
        api_key=None,
        transport=None,
    )

    events = [event async for event in client.stream_events()]
    assert events == [{"event": "status", "data": "ok"}]
    assert len(captured_timeouts) == 1
    timeout = captured_timeouts[0]
    assert isinstance(timeout, httpx.Timeout)
    assert timeout.read is None
    assert timeout.connect == pytest.approx(10.0)
