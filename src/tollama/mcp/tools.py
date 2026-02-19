"""MCP tool handlers backed by the shared tollama HTTP client."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from tollama.client import (
    DEFAULT_BASE_URL,
    DEFAULT_TIMEOUT_SECONDS,
    ModelMissingError,
    TollamaClient,
    TollamaClientError,
)
from tollama.core.schemas import ForecastRequest

from .schemas import (
    ForecastToolInput,
    HealthToolInput,
    ModelsToolInput,
    PullToolInput,
    ShowToolInput,
)


@dataclass(slots=True)
class MCPToolError(RuntimeError):
    """Error surfaced by MCP tool handlers with exit-code-aligned metadata."""

    category: str
    exit_code: int
    message: str

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.message


def _make_client(*, base_url: str | None, timeout: float | None) -> TollamaClient:
    return TollamaClient(
        base_url=base_url or DEFAULT_BASE_URL,
        timeout=timeout if timeout is not None else DEFAULT_TIMEOUT_SECONDS,
    )


def _raise_from_client_error(exc: TollamaClientError) -> None:
    raise MCPToolError(
        category=exc.category,
        exit_code=exc.exit_code,
        message=str(exc),
    ) from exc


def _raise_invalid_request(detail: str) -> None:
    raise MCPToolError(category="INVALID_REQUEST", exit_code=2, message=detail)


def tollama_health(*, base_url: str | None = None, timeout: float | None = None) -> dict[str, Any]:
    """Return daemon health payload for MCP consumers."""
    try:
        args = HealthToolInput(base_url=base_url, timeout=timeout)
    except ValidationError as exc:
        _raise_invalid_request(str(exc))

    client = _make_client(base_url=args.base_url, timeout=args.timeout)
    try:
        payload = client.health()
    except TollamaClientError as exc:
        _raise_from_client_error(exc)

    return {
        "healthy": True,
        "health": payload.get("health", {}),
        "version": payload.get("version", {}),
    }


def tollama_models(
    *,
    mode: str = "installed",
    base_url: str | None = None,
    timeout: float | None = None,
) -> dict[str, Any]:
    """Return model list payload for installed/loaded/available modes."""
    try:
        args = ModelsToolInput(mode=mode, base_url=base_url, timeout=timeout)
    except ValidationError as exc:
        _raise_invalid_request(str(exc))

    client = _make_client(base_url=args.base_url, timeout=args.timeout)
    try:
        items = client.models(mode=args.mode)
    except TollamaClientError as exc:
        _raise_from_client_error(exc)

    return {
        "mode": args.mode,
        "items": items,
    }


def tollama_forecast(
    *,
    request: dict[str, Any],
    base_url: str | None = None,
    timeout: float | None = None,
) -> dict[str, Any]:
    """Run a non-streaming forecast and return canonical response payload."""
    try:
        args = ForecastToolInput(request=request, base_url=base_url, timeout=timeout)
    except ValidationError as exc:
        _raise_invalid_request(str(exc))

    try:
        forecast_request = ForecastRequest.model_validate(args.request)
    except ValidationError as exc:
        _raise_invalid_request(str(exc))

    client = _make_client(base_url=args.base_url, timeout=args.timeout)
    try:
        response = client.forecast_response(forecast_request)
    except TollamaClientError as exc:
        _raise_from_client_error(exc)

    return response.model_dump(mode="json", exclude_none=True)


def tollama_pull(
    *,
    model: str,
    accept_license: bool = False,
    base_url: str | None = None,
    timeout: float | None = None,
) -> dict[str, Any]:
    """Pull a model in non-stream mode."""
    try:
        args = PullToolInput(
            model=model,
            accept_license=accept_license,
            base_url=base_url,
            timeout=timeout,
        )
    except ValidationError as exc:
        _raise_invalid_request(str(exc))

    client = _make_client(base_url=args.base_url, timeout=args.timeout)
    try:
        return client.pull(args.model, accept_license=args.accept_license)
    except TollamaClientError as exc:
        _raise_from_client_error(exc)


def tollama_show(
    *,
    model: str,
    base_url: str | None = None,
    timeout: float | None = None,
) -> dict[str, Any]:
    """Fetch model detail payload."""
    try:
        args = ShowToolInput(model=model, base_url=base_url, timeout=timeout)
    except ValidationError as exc:
        _raise_invalid_request(str(exc))

    client = _make_client(base_url=args.base_url, timeout=args.timeout)
    try:
        return client.show(args.model)
    except ModelMissingError as exc:
        _raise_from_client_error(exc)
    except TollamaClientError as exc:
        _raise_from_client_error(exc)
