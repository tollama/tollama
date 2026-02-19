"""HTTP client helpers for talking to tollamad."""

from __future__ import annotations

import json
from typing import Any

import httpx

from tollama.core.schemas import ForecastRequest, ForecastResponse

from .exceptions import (
    DaemonHTTPError,
    DaemonUnreachableError,
    ForecastTimeoutError,
    InvalidRequestError,
    LicenseRequiredError,
    ModelMissingError,
    PermissionDeniedError,
    TollamaClientError,
)

DEFAULT_DAEMON_HOST = "127.0.0.1"
DEFAULT_DAEMON_PORT = 11435
DEFAULT_BASE_URL = f"http://localhost:{DEFAULT_DAEMON_PORT}"
DEFAULT_TIMEOUT_SECONDS = 10.0


class TollamaClient:
    """Minimal client for tollamad HTTP endpoints."""

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        *,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._transport = transport

    def health(self) -> dict[str, Any]:
        """Fetch daemon health and version details."""
        health_payload = self._request_json("GET", "/v1/health", action="daemon health")
        version_payload = self._request_json("GET", "/api/version", action="daemon version")
        return {
            "health": health_payload,
            "version": version_payload,
        }

    def models(self, mode: str = "installed") -> list[dict[str, Any]]:
        """Return model items for installed/loaded/available views."""
        normalized_mode = mode.strip().lower()
        if normalized_mode == "installed":
            payload = self.list_tags()
            models = payload.get("models")
            return _coerce_dict_list(models)
        if normalized_mode == "loaded":
            payload = self.list_running()
            models = payload.get("models")
            return _coerce_dict_list(models)
        if normalized_mode == "available":
            payload = self.info()
            models_payload = payload.get("models") if isinstance(payload, dict) else None
            available = (
                models_payload.get("available")
                if isinstance(models_payload, dict)
                else None
            )
            return _coerce_dict_list(available)
        raise InvalidRequestError(action="list models", detail=f"unsupported mode: {mode!r}")

    def info(self) -> dict[str, Any]:
        """Fetch daemon diagnostics payload."""
        return self._request_json("GET", "/api/info", action="daemon info")

    def forecast(
        self,
        payload: dict[str, Any] | ForecastRequest,
        *,
        stream: bool = True,
    ) -> dict[str, Any] | list[dict[str, Any]]:
        """Submit an Ollama-compatible forecast request."""
        request_payload = self._coerce_request_payload(payload)
        request_payload["stream"] = stream
        action = f"forecast with model {request_payload.get('model')!r}"
        if stream:
            try:
                return self._request_ndjson(
                    "POST",
                    "/api/forecast",
                    json_payload=request_payload,
                    action=action,
                )
            except ModelMissingError:
                # For streaming, keep legacy fallback semantics only for endpoint misses.
                return self._request_ndjson(
                    "POST",
                    "/v1/forecast",
                    json_payload=request_payload,
                    action=action,
                )

        try:
            return self._request_json(
                "POST",
                "/api/forecast",
                json_payload=request_payload,
                action=action,
            )
        except ModelMissingError:
            return self._request_json(
                "POST",
                "/v1/forecast",
                json_payload=request_payload,
                action=action,
            )

    def forecast_response(self, request: ForecastRequest) -> ForecastResponse:
        """Submit a non-streaming forecast and validate response schema."""
        response_payload = self.forecast(request, stream=False)
        if not isinstance(response_payload, dict):
            raise TollamaClientError(
                action=f"forecast with model {request.model!r}",
                detail="daemon returned unexpected non-object response",
            )
        try:
            return ForecastResponse.model_validate(response_payload)
        except Exception as exc:  # noqa: BLE001
            raise InvalidRequestError(
                action="validate forecast response",
                detail=str(exc),
            ) from exc

    def validate_request(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Validate a forecast request without executing model inference."""
        return self._request_json(
            "POST",
            "/api/validate",
            json_payload=payload,
            action="validate forecast request",
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
        accept_license: bool = False,
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
            "accept_license": accept_license,
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

    def pull(
        self,
        model: str,
        *,
        accept_license: bool = False,
    ) -> dict[str, Any]:
        """Pull a model in non-stream mode."""
        response = self.pull_model(model, stream=False, accept_license=accept_license)
        if not isinstance(response, dict):
            raise TollamaClientError(
                action=f"pull model {model!r}",
                detail="invalid response payload",
            )
        return response

    def show_model(self, name: str) -> dict[str, Any]:
        """Fetch model details through the Ollama-compatible show endpoint."""
        payload = {"model": name}
        return self._request_json(
            "POST",
            "/api/show",
            json_payload=payload,
            action=f"show model {name!r}",
        )

    def show(self, model: str) -> dict[str, Any]:
        """Alias of show_model."""
        return self.show_model(model)

    def remove_model(self, name: str) -> dict[str, Any]:
        """Delete a model through the Ollama-compatible delete endpoint."""
        payload = {"model": name}
        return self._request_json(
            "DELETE",
            "/api/delete",
            json_payload=payload,
            action=f"remove model {name!r}",
        )

    def rm(self, model: str) -> dict[str, Any]:
        """Alias of remove_model."""
        return self.remove_model(model)

    def _coerce_request_payload(self, payload: dict[str, Any] | ForecastRequest) -> dict[str, Any]:
        if isinstance(payload, ForecastRequest):
            return payload.model_dump(mode="json", exclude_none=True)
        return dict(payload)

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
            raise TollamaClientError(
                action=action,
                detail="daemon returned non-JSON response",
            ) from exc

        if not isinstance(data, dict):
            raise TollamaClientError(
                action=action,
                detail="daemon returned unexpected JSON payload",
            )
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
                raise TollamaClientError(
                    action=action,
                    detail="daemon returned non-NDJSON response",
                ) from exc
            if not isinstance(payload, dict):
                raise TollamaClientError(
                    action=action,
                    detail="daemon returned unexpected NDJSON payload",
                )
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
            with httpx.Client(
                base_url=self._base_url,
                timeout=self._timeout,
                transport=self._transport,
            ) as client:
                response = client.request(method, path, json=json_payload)
        except httpx.TimeoutException as exc:
            raise ForecastTimeoutError(action=action, detail=str(exc)) from exc
        except httpx.ConnectError as exc:
            raise DaemonUnreachableError(action=action, detail=str(exc)) from exc
        except httpx.RequestError as exc:
            raise DaemonUnreachableError(action=action, detail=str(exc)) from exc
        except httpx.HTTPError as exc:
            raise TollamaClientError(action=action, detail=str(exc)) from exc

        if response.is_error:
            detail = _extract_error_detail(response)
            raise _map_http_error(action=action, status_code=response.status_code, detail=detail)
        return response


def _extract_error_detail(response: httpx.Response) -> str:
    text = response.text.strip()
    if not text:
        return f"HTTP {response.status_code}"

    try:
        payload = response.json()
    except ValueError:
        return text

    if isinstance(payload, dict):
        detail = payload.get("detail")
        if isinstance(detail, str) and detail:
            return detail
    return text


def _map_http_error(*, action: str, status_code: int, detail: str) -> DaemonHTTPError:
    lower_detail = detail.lower()

    if status_code == 400:
        return InvalidRequestError(action=action, status_code=status_code, detail=detail)

    if status_code in {408, 504}:
        return ForecastTimeoutError(action=action, status_code=status_code, detail=detail)

    if status_code == 404:
        return ModelMissingError(action=action, status_code=status_code, detail=detail)

    if status_code in {401, 403}:
        if (
            "license" in lower_detail
            or "accept_license" in lower_detail
            or "accept-license" in lower_detail
        ):
            return LicenseRequiredError(action=action, status_code=status_code, detail=detail)
        return PermissionDeniedError(action=action, status_code=status_code, detail=detail)

    if status_code == 409:
        if (
            "license" in lower_detail
            or "accept_license" in lower_detail
            or "accept-license" in lower_detail
        ):
            return LicenseRequiredError(action=action, status_code=status_code, detail=detail)
        return PermissionDeniedError(action=action, status_code=status_code, detail=detail)

    return DaemonHTTPError(action=action, status_code=status_code, detail=detail)


def _coerce_dict_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    items: list[dict[str, Any]] = []
    for item in value:
        if isinstance(item, dict):
            items.append(item)
    return items
