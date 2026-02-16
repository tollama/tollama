"""Torch runner process using newline-delimited JSON over stdio."""

from __future__ import annotations

import sys
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

from .chronos_adapter import ChronosAdapter, DependencyMissingError, UnsupportedModelError

RUNNER_NAME = "tollama-torch"
RUNNER_VERSION = "0.1.0"
UNKNOWN_REQUEST_ID = "unknown"
CAPABILITIES = ("hello", "load", "unload", "forecast")
_FORECAST_REQUEST_FIELDS = frozenset({"model", "horizon", "quantiles", "series", "options"})


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


def _require_model_name(params: Mapping[str, Any]) -> str:
    model_name = params.get("model")
    if not isinstance(model_name, str) or not model_name:
        raise ValueError("model must be a non-empty string")
    return model_name


def _handle_hello(request: ProtocolRequest) -> ProtocolResponse:
    return ProtocolResponse(
        id=request.id,
        result={
            "name": RUNNER_NAME,
            "version": RUNNER_VERSION,
            "capabilities": list(CAPABILITIES),
            "supported_families": ["torch"],
        },
    )


def _handle_load(request: ProtocolRequest, adapter: ChronosAdapter) -> ProtocolResponse:
    try:
        model_name = _require_model_name(request.params)
        adapter.load(model_name)
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
    except ValueError as exc:
        return _error_response(
            request.id,
            code=-32602,
            message="invalid params",
            data={"details": str(exc)},
        )

    return ProtocolResponse(
        id=request.id,
        result={"loaded": True, "model": model_name},
    )


def _handle_unload(request: ProtocolRequest, adapter: ChronosAdapter) -> ProtocolResponse:
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


def _handle_forecast(request: ProtocolRequest, adapter: ChronosAdapter) -> ProtocolResponse:
    canonical_params = {
        key: value for key, value in request.params.items() if key in _FORECAST_REQUEST_FIELDS
    }
    model_local_dir = request.params.get("model_local_dir")
    if model_local_dir is not None and (
        not isinstance(model_local_dir, str) or not model_local_dir
    ):
        return _error_response(
            request.id,
            code=-32602,
            message="invalid params",
            data={"details": "model_local_dir must be a non-empty string when provided"},
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
        response = adapter.forecast(
            forecast_request,
            model_local_dir=model_local_dir if isinstance(model_local_dir, str) else None,
        )
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
    except ValueError as exc:
        return _error_response(
            request.id,
            code="FORECAST_ERROR",
            message=str(exc),
        )

    return ProtocolResponse(
        id=request.id,
        result=response.model_dump(mode="json", exclude_none=True),
    )


def handle_request_line(line: str | bytes, adapter: ChronosAdapter) -> ProtocolResponse:
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
    if request.method == "load":
        return _handle_load(request, adapter)
    if request.method == "unload":
        return _handle_unload(request, adapter)
    if request.method == "forecast":
        return _handle_forecast(request, adapter)

    return _error_response(
        request.id,
        code=-32601,
        message="method not found",
        data={"method": request.method},
    )


def serve(stdin: TextIO = sys.stdin, stdout: TextIO = sys.stdout) -> int:
    adapter = ChronosAdapter()
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
