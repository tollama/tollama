"""Async client helpers for the Textual dashboard."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import httpx

from tollama.client import AsyncTollamaClient


@dataclass(frozen=True)
class DashboardAPIError(RuntimeError):
    """Structured API error used by TUI dashboard workflows."""

    status_code: int
    detail: str
    hint: str | None = None
    path: str | None = None

    def __str__(self) -> str:
        location = f" ({self.path})" if self.path else ""
        hint = f" Hint: {self.hint}" if self.hint else ""
        return f"HTTP {self.status_code}{location}: {self.detail}{hint}"


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

    async def pull_model_events(
        self,
        model: str,
        *,
        accept_license: bool = False,
    ) -> list[dict[str, Any]]:
        response = await self._request(
            "POST",
            "/api/pull",
            json_payload={
                "model": model,
                "stream": True,
                "accept_license": accept_license,
            },
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
            timeout=self._stream_timeout(),
            transport=self._transport,
        ) as client:
            async with client.stream(
                "GET",
                "/api/events?heartbeat=15",
                headers=headers,
            ) as response:
                if bool(getattr(response, "is_error", False)):
                    raise self._build_api_error(path="/api/events", response=response)
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

    def _stream_timeout(self) -> httpx.Timeout:
        # SSE responses can remain idle between heartbeats.
        return httpx.Timeout(self._timeout, read=None)

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
        if response.is_error:
            raise self._build_api_error(path=path, response=response)
        return response

    def _headers(self, *, accept: str) -> dict[str, str] | None:
        headers: dict[str, str] = {"Accept": accept}
        token = (self._api_key or "").strip()
        if token:
            headers["Authorization"] = f"Bearer {token}"
        return headers

    def _build_api_error(self, *, path: str, response: httpx.Response) -> DashboardAPIError:
        detail: str | None = None
        hint: str | None = None
        try:
            payload = response.json()
        except ValueError:
            payload = None

        if isinstance(payload, dict):
            detail = _detail_to_text(payload.get("detail"))
            hint = _optional_nonempty(payload.get("hint"))
        elif payload is not None:
            detail = _detail_to_text(payload)

        if detail is None:
            detail = _optional_nonempty(response.text) or response.reason_phrase or "request failed"

        return DashboardAPIError(
            status_code=response.status_code,
            detail=detail,
            hint=hint,
            path=path,
        )


def _optional_nonempty(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized or None


def _detail_to_text(detail: Any) -> str | None:
    if isinstance(detail, str):
        normalized = detail.strip()
        return normalized or None
    if isinstance(detail, (int, float, bool)):
        return str(detail)
    if detail is None:
        return None
    try:
        return json.dumps(detail, separators=(",", ":"), sort_keys=True)
    except TypeError:
        return str(detail)
