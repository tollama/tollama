"""Minimal A2A client utilities for discovering and calling external agents."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

import httpx


class A2AClientError(RuntimeError):
    """Base A2A client failure."""


class A2ARpcError(A2AClientError):
    """Raised when one A2A JSON-RPC response includes an error object."""

    def __init__(self, *, code: int, message: str, data: Any | None = None) -> None:
        super().__init__(f"A2A RPC error {code}: {message}")
        self.code = code
        self.message = message
        self.data = data


@dataclass(frozen=True, slots=True)
class A2AClient:
    """Small sync client for A2A discovery and JSON-RPC task calls."""

    timeout: float = 10.0
    api_key: str | None = None
    transport: httpx.BaseTransport | None = None

    def discover(self, *, base_url: str) -> dict[str, Any]:
        url = f"{base_url.rstrip('/')}/.well-known/agent-card.json"
        with httpx.Client(
            timeout=self.timeout,
            headers=self._headers(),
            transport=self.transport,
        ) as client:
            response = client.get(url)
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise A2AClientError(f"agent card discovery failed: {exc}") from exc
        payload = response.json()
        if not isinstance(payload, dict):
            raise A2AClientError("agent card discovery returned non-object JSON")
        return payload

    def send_message(
        self,
        *,
        base_url: str,
        message: dict[str, Any],
        configuration: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        request_id: str | int | None = None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {"message": message}
        if configuration is not None:
            params["configuration"] = configuration
        if metadata is not None:
            params["metadata"] = metadata
        return self._rpc(
            base_url=base_url,
            method="message/send",
            params=params,
            request_id=request_id,
        )

    def get_task(
        self,
        *,
        base_url: str,
        task_id: str,
        history_length: int | None = None,
        request_id: str | int | None = None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {"id": task_id}
        if history_length is not None:
            params["historyLength"] = history_length
        return self._rpc(
            base_url=base_url,
            method="tasks/get",
            params=params,
            request_id=request_id,
        )

    def list_tasks(
        self,
        *,
        base_url: str,
        context_id: str | None = None,
        status: str | None = None,
        page_size: int | None = None,
        page_token: str | None = None,
        request_id: str | int | None = None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {}
        if context_id is not None:
            params["contextId"] = context_id
        if status is not None:
            params["status"] = status
        if page_size is not None:
            params["pageSize"] = page_size
        if page_token is not None:
            params["pageToken"] = page_token
        return self._rpc(
            base_url=base_url,
            method="tasks/query",
            params=params,
            request_id=request_id,
        )

    def cancel_task(
        self,
        *,
        base_url: str,
        task_id: str,
        request_id: str | int | None = None,
    ) -> dict[str, Any]:
        return self._rpc(
            base_url=base_url,
            method="tasks/cancel",
            params={"id": task_id},
            request_id=request_id,
        )

    def poll_task(
        self,
        *,
        base_url: str,
        task_id: str,
        timeout_seconds: float = 30.0,
        interval_seconds: float = 0.2,
    ) -> dict[str, Any]:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than zero")
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be greater than zero")

        deadline = time.monotonic() + timeout_seconds
        while True:
            task = self.get_task(base_url=base_url, task_id=task_id)
            status = task.get("status")
            state = status.get("state") if isinstance(status, dict) else None
            if state in {"completed", "failed", "canceled", "rejected"}:
                return task

            if time.monotonic() >= deadline:
                raise A2AClientError(
                    f"task {task_id!r} did not complete within {timeout_seconds:.2f} seconds",
                )
            time.sleep(interval_seconds)

    def _rpc(
        self,
        *,
        base_url: str,
        method: str,
        params: dict[str, Any],
        request_id: str | int | None,
    ) -> dict[str, Any]:
        payload = {
            "jsonrpc": "2.0",
            "id": request_id if request_id is not None else uuid4().hex,
            "method": method,
            "params": params,
        }

        url = f"{base_url.rstrip('/')}/a2a"
        with httpx.Client(
            timeout=self.timeout,
            headers=self._headers(),
            transport=self.transport,
        ) as client:
            response = client.post(url, json=payload)
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise A2AClientError(f"A2A RPC transport failed: {exc}") from exc

        raw = response.json()
        if not isinstance(raw, dict):
            raise A2AClientError("A2A RPC returned non-object JSON")

        error_payload = raw.get("error")
        if isinstance(error_payload, dict):
            code = int(error_payload.get("code", -32603))
            message = str(error_payload.get("message", "unknown A2A error"))
            raise A2ARpcError(code=code, message=message, data=error_payload.get("data"))

        result = raw.get("result")
        if not isinstance(result, dict):
            raise A2AClientError("A2A RPC response is missing object result")
        return result

    def _headers(self) -> dict[str, str]:
        if not self.api_key:
            return {}
        return {"Authorization": f"Bearer {self.api_key}"}
