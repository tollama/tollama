"""Async client helpers for the Textual dashboard."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import httpx

from tollama.client import AsyncTollamaClient


@dataclass(frozen=True)
class DashboardSnapshot:
    """Aggregate dashboard payload consumed by TUI screens."""

    state: dict[str, Any]
    tags: dict[str, Any]


class DashboardAPIClient:
    """Small async wrapper around daemon APIs and SSE streams."""

    def __init__(
        self,
        *,
        base_url: str,
        timeout: float,
        api_key: str | None = None,
        transport: httpx.AsyncBaseTransport | httpx.BaseTransport | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._api_key = api_key
        self._transport = transport
        self._client = AsyncTollamaClient(
            base_url=base_url,
            timeout=timeout,
            api_key=api_key,
            transport=transport,
        )

    async def dashboard_snapshot(self) -> DashboardSnapshot:
        state = await self._request_json("GET", "/api/dashboard/state")
        tags = await self._client.list_tags()
        return DashboardSnapshot(state=state, tags=tags)

    async def forecast(self, payload: dict[str, Any]) -> dict[str, Any]:
        response = await self._request_json(
            "POST",
            "/api/forecast",
            json_payload={**payload, "stream": False},
        )
        return response

    async def show_model(self, model: str) -> dict[str, Any]:
        return await self._request_json(
            "POST",
            "/api/show",
            json_payload={"model": model},
        )

    async def delete_model(self, model: str) -> dict[str, Any]:
        return await self._request_json(
            "DELETE",
            "/api/delete",
            json_payload={"model": model},
        )

    async def pull_model_events(self, model: str) -> list[dict[str, Any]]:
        response = await self._request(
            "POST",
            "/api/pull",
            json_payload={"model": model, "stream": True, "accept_license": False},
            accept="application/x-ndjson",
        )
        events: list[dict[str, Any]] = []
        for raw_line in response.text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                events.append(payload)
        return events

    async def stream_events(self) -> AsyncIterator[dict[str, Any]]:
        headers = self._headers(accept="text/event-stream")
        async with httpx.AsyncClient(
            base_url=self._base_url,
            timeout=self._timeout,
            transport=self._transport,
        ) as client:
            async with client.stream(
                "GET",
                "/api/events?heartbeat=15",
                headers=headers,
            ) as response:
                response.raise_for_status()
                event = "message"
                data_lines: list[str] = []
                async for raw_line in response.aiter_lines():
                    line = raw_line.strip()
                    if not line:
                        if data_lines:
                            payload = {
                                "event": event,
                                "data": "\n".join(data_lines),
                            }
                            yield payload
                        event = "message"
                        data_lines = []
                        continue
                    if line.startswith("event:"):
                        event = line.partition(":")[2].strip() or "message"
                    elif line.startswith("data:"):
                        data_lines.append(line.partition(":")[2].strip())

    async def _request_json(
        self,
        method: str,
        path: str,
        *,
        json_payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        response = await self._request(method, path, json_payload=json_payload)
        payload = response.json()
        if not isinstance(payload, dict):
            raise RuntimeError(f"unexpected {path} payload shape")
        return payload

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json_payload: dict[str, Any] | None = None,
        accept: str = "application/json",
    ) -> httpx.Response:
        async with httpx.AsyncClient(
            base_url=self._base_url,
            timeout=self._timeout,
            transport=self._transport,
        ) as client:
            response = await client.request(
                method,
                path,
                json=json_payload,
                headers=self._headers(accept=accept),
            )
        response.raise_for_status()
        return response

    def _headers(self, *, accept: str) -> dict[str, str] | None:
        headers: dict[str, str] = {"Accept": accept}
        token = (self._api_key or "").strip()
        if token:
            headers["Authorization"] = f"Bearer {token}"
        return headers
