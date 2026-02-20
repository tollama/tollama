"""Sundial runner process using newline-delimited JSON over stdio."""

from __future__ import annotations

import sys
import time
from collections.abc import Mapping
from typing import Any, TextIO

from pydantic import ValidationError

from tollama.core.protocol import (
    ProtocolDecodeError,
    ProtocolErrorMessage,
    ProtocolRequest,
    ProtocolResponse,
    decode_line,
    encode_line,
    validate_request,
)
from tollama.core.schemas import ForecastRequest
from tollama.runners.runtime_telemetry import enrich_forecast_response

from .adapter import SundialAdapter
from .errors import AdapterInputError, DependencyMissingError, UnsupportedModelError

RUNNER_NAME = "tollama-sundial"
RUNNER_VERSION = "0.1.0"
UNKNOWN_REQUEST_ID = "unknown"
CAPABILITIES = ("hello", "forecast", "unload")
_FORECAST_REQUEST_FIELDS = frozenset(
    {"model", "horizon", "quantiles", "series", "options", "parameters"},
)


def _extract_request_id(payload: Mapping[str, Any] | None) -> str:
    if payload is not None:
        request_id = payload.get("id")
        if isinstance(request_id, str) and request_id:
            return request_id
    return UNKNOWN_REQUEST_ID


def _error_response(
    request_id: str,
    code: int | str,
    message: str,
    data: Any | None = None,
) -> ProtocolResponse:
    return ProtocolResponse(
        id=request_id,
        error=ProtocolErrorMessage(code=code, message=message, data=data),
    )


def _optional_nonempty_str(params: Mapping[str, Any], key: str) -> str | None:
    value = params.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string when provided")
    return value.strip()


def _optional_string_key_mapping(params: Mapping[str, Any], key: str) -> dict[str, Any] | None:
    value = params.get(key)
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError(f"{key} must be an object when provided")

    normalized: dict[str, Any] = {}
    for raw_key, raw_value in value.items():
        if not isinstance(raw_key, str):
            raise ValueError(f"{key} keys must be strings")
        normalized[raw_key] = raw_value
    return normalized


def _handle_hello(request: ProtocolRequest) -> ProtocolResponse:
    return ProtocolResponse(
        id=request.id,
        result={
            "name": RUNNER_NAME,
            "version": RUNNER_VERSION,
            "capabilities": list(CAPABILITIES),
            "supported_families": ["sundial"],
        },
    )


def _handle_unload(request: ProtocolRequest, adapter: SundialAdapter) -> ProtocolResponse:
    model_name = request.params.get("model")
    if model_name is not None and (not isinstance(model_name, str) or not model_name):
        return _error_response(
            request.id,
            code=-32602,
            message="invalid params",
            data={"details": "model must be a non-empty string when provided"},
        )
    adapter.unload(model_name if isinstance(model_name, str) else None)
    return ProtocolResponse(
        id=request.id,
        result={"unloaded": True, "model": model_name},
    )


def _handle_forecast(request: ProtocolRequest, adapter: SundialAdapter) -> ProtocolResponse:
    canonical_params = {
        key: value for key, value in request.params.items() if key in _FORECAST_REQUEST_FIELDS
    }
    try:
        model_local_dir = _optional_nonempty_str(request.params, "model_local_dir")
        model_source = _optional_string_key_mapping(request.params, "model_source")
        model_metadata = _optional_string_key_mapping(request.params, "model_metadata")
    except ValueError as exc:
        return _error_response(
            request.id,
            code="BAD_REQUEST",
            message=str(exc),
        )

    try:
        forecast_request = ForecastRequest.model_validate(canonical_params)
    except ValidationError as exc:
        return _error_response(
            request.id,
            code=-32602,
            message="invalid params",
            data={"details": str(exc)},
        )

    try:
        started_at = time.perf_counter()
        response = adapter.forecast(
            forecast_request,
            model_local_dir=model_local_dir,
            model_source=model_source,
            model_metadata=model_metadata,
        )
        inference_ms = (time.perf_counter() - started_at) * 1000.0
    except DependencyMissingError as exc:
        return _error_response(
            request.id,
            code="DEPENDENCY_MISSING",
            message=str(exc),
        )
    except UnsupportedModelError as exc:
        return _error_response(
            request.id,
            code="MODEL_UNSUPPORTED",
            message=str(exc),
        )
    except AdapterInputError as exc:
        return _error_response(
            request.id,
            code="BAD_REQUEST",
            message=str(exc),
        )
    except ValueError as exc:
        return _error_response(
            request.id,
            code="FORECAST_ERROR",
            message=str(exc),
        )
    except Exception as exc:  # noqa: BLE001
        return _error_response(
            request.id,
            code="FORECAST_ERROR",
            message=f"{type(exc).__name__}: {exc}",
        )

    response = enrich_forecast_response(
        response=response,
        runner_name=RUNNER_NAME,
        inference_ms=inference_ms,
    )
    return ProtocolResponse(
        id=request.id,
        result=response.model_dump(mode="json", exclude_none=True),
    )


def handle_request_line(line: str | bytes, adapter: SundialAdapter) -> ProtocolResponse:
    payload: Mapping[str, Any] | None = None
    request_id = UNKNOWN_REQUEST_ID

    try:
        payload = decode_line(line)
        request_id = _extract_request_id(payload)
        request = validate_request(payload)
    except ProtocolDecodeError as exc:
        return _error_response(
            request_id,
            code=-32600,
            message="invalid request",
            data={"details": str(exc)},
        )

    if request.method == "hello":
        return _handle_hello(request)
    if request.method == "forecast":
        return _handle_forecast(request, adapter)
    if request.method == "unload":
        return _handle_unload(request, adapter)

    return _error_response(
        request.id,
        code=-32601,
        message="method not found",
        data={"method": request.method},
    )


def serve(stdin: TextIO = sys.stdin, stdout: TextIO = sys.stdout) -> int:
    adapter = SundialAdapter()
    for line in stdin:
        if not line.strip():
            continue
        response = handle_request_line(line, adapter)
        stdout.write(encode_line(response))
        stdout.flush()
    return 0


def main() -> int:
    return serve()


if __name__ == "__main__":
    raise SystemExit(main())
