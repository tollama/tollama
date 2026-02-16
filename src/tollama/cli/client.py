"""HTTP client helpers for talking to tollamad."""

from __future__ import annotations

from typing import Any

import httpx

DEFAULT_DAEMON_HOST = "127.0.0.1"
DEFAULT_DAEMON_PORT = 11435
DEFAULT_BASE_URL = f"http://{DEFAULT_DAEMON_HOST}:{DEFAULT_DAEMON_PORT}"
DEFAULT_TIMEOUT_SECONDS = 10.0


class TollamaClient:
    """Minimal client for tollamad HTTP endpoints."""

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

    def forecast(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Submit a forecast request payload and return response JSON."""
        with httpx.Client(base_url=self._base_url, timeout=self._timeout) as client:
            response = client.post("/v1/forecast", json=payload)

        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text
            raise RuntimeError(
                f"forecast request failed with HTTP {exc.response.status_code}: {detail}",
            ) from exc
        except httpx.HTTPError as exc:
            raise RuntimeError(f"forecast request failed: {exc}") from exc

        try:
            data = response.json()
        except ValueError as exc:
            raise RuntimeError("daemon returned non-JSON response") from exc

        if not isinstance(data, dict):
            raise RuntimeError("daemon returned unexpected JSON payload")
        return data
