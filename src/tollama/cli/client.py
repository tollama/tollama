"""HTTP client helpers for talking to tollamad."""

from __future__ import annotations

import json
from typing import Any

import httpx

DEFAULT_DAEMON_HOST = "127.0.0.1"
DEFAULT_DAEMON_PORT = 11435
DEFAULT_BASE_URL = f"http://localhost:{DEFAULT_DAEMON_PORT}"
DEFAULT_TIMEOUT_SECONDS = 10.0


class DaemonHTTPError(RuntimeError):
    """HTTP status error returned by the daemon."""

    def __init__(self, *, action: str, status_code: int, detail: str) -> None:
        super().__init__(f"{action} failed with HTTP {status_code}: {detail}")
        self.action = action
        self.status_code = status_code
        self.detail = detail


class TollamaClient:
    """Minimal client for tollamad HTTP endpoints."""

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

    def forecast(
        self,
        payload: dict[str, Any],
        *,
        stream: bool = True,
    ) -> dict[str, Any] | list[dict[str, Any]]:
        """Submit an Ollama-compatible forecast request."""
        request_payload = dict(payload)
        request_payload["stream"] = stream
        action = f"forecast with model {request_payload.get('model')!r}"
        if stream:
            return self._request_ndjson(
                "POST",
                "/api/forecast",
                json_payload=request_payload,
                action=action,
            )
        return self._request_json(
            "POST",
            "/api/forecast",
            json_payload=request_payload,
            action=action,
        )

    def list_tags(self) -> dict[str, Any]:
        """Fetch installed model tags from the daemon."""
        return self._request_json("GET", "/api/tags", action="list model tags")

    def list_running(self) -> dict[str, Any]:
        """Fetch loaded model process state from the daemon."""
        return self._request_json("GET", "/api/ps", action="list loaded models")

    def pull_model(
        self,
        name: str,
        *,
        stream: bool = True,
        insecure: bool | None = None,
        offline: bool | None = None,
        local_files_only: bool | None = None,
        http_proxy: str | None = None,
        https_proxy: str | None = None,
        no_proxy: str | None = None,
        hf_home: str | None = None,
        max_workers: int | None = None,
        token: str | None = None,
        include_null_fields: set[str] | None = None,
    ) -> dict[str, Any] | list[dict[str, Any]]:
        """Install a model through the Ollama-compatible pull endpoint."""
        null_fields = include_null_fields or set()
        payload: dict[str, Any] = {
            "model": name,
            "stream": stream,
        }
        if insecure is not None or "insecure" in null_fields:
            payload["insecure"] = insecure
        if offline is not None or "offline" in null_fields:
            payload["offline"] = offline
        if local_files_only is not None or "local_files_only" in null_fields:
            payload["local_files_only"] = local_files_only
        if http_proxy is not None or "http_proxy" in null_fields:
            payload["http_proxy"] = http_proxy
        if https_proxy is not None or "https_proxy" in null_fields:
            payload["https_proxy"] = https_proxy
        if no_proxy is not None or "no_proxy" in null_fields:
            payload["no_proxy"] = no_proxy
        if hf_home is not None or "hf_home" in null_fields:
            payload["hf_home"] = hf_home
        if max_workers is not None or "max_workers" in null_fields:
            payload["max_workers"] = max_workers
        if token is not None or "token" in null_fields:
            payload["token"] = token
        action = f"pull model {name!r}"
        if stream:
            return self._request_ndjson("POST", "/api/pull", json_payload=payload, action=action)
        return self._request_json("POST", "/api/pull", json_payload=payload, action=action)

    def show_model(self, name: str) -> dict[str, Any]:
        """Fetch model details through the Ollama-compatible show endpoint."""
        payload = {"model": name}
        return self._request_json(
            "POST",
            "/api/show",
            json_payload=payload,
            action=f"show model {name!r}",
        )

    def remove_model(self, name: str) -> dict[str, Any]:
        """Delete a model through the Ollama-compatible delete endpoint."""
        payload = {"model": name}
        return self._request_json(
            "DELETE",
            "/api/delete",
            json_payload=payload,
            action=f"remove model {name!r}",
        )

    def _request_json(
        self,
        method: str,
        path: str,
        *,
        json_payload: dict[str, Any] | None = None,
        action: str,
    ) -> dict[str, Any]:
        response = self._send_request(method, path, json_payload=json_payload, action=action)
        try:
            data = response.json()
        except ValueError as exc:
            raise RuntimeError("daemon returned non-JSON response") from exc

        if not isinstance(data, dict):
            raise RuntimeError("daemon returned unexpected JSON payload")
        return data

    def _request_ndjson(
        self,
        method: str,
        path: str,
        *,
        json_payload: dict[str, Any] | None = None,
        action: str,
    ) -> list[dict[str, Any]]:
        response = self._send_request(method, path, json_payload=json_payload, action=action)
        entries: list[dict[str, Any]] = []
        for raw_line in response.text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError("daemon returned non-NDJSON response") from exc
            if not isinstance(payload, dict):
                raise RuntimeError("daemon returned unexpected NDJSON payload")
            entries.append(payload)
        return entries

    def _send_request(
        self,
        method: str,
        path: str,
        *,
        json_payload: dict[str, Any] | None = None,
        action: str,
    ) -> httpx.Response:
        try:
            with httpx.Client(base_url=self._base_url, timeout=self._timeout) as client:
                response = client.request(method, path, json=json_payload)
        except httpx.HTTPError as exc:
            raise RuntimeError(f"{action} failed: {exc}") from exc

        if response.is_error:
            raise DaemonHTTPError(
                action=action,
                status_code=response.status_code,
                detail=response.text,
            )
        return response
