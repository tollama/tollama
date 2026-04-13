"""Shared helpers for runner NDJSON entrypoints."""

from __future__ import annotations

import sys
from collections.abc import Callable, Mapping
from typing import Any, TextIO

from tollama.core.protocol import (
    ProtocolDecodeError,
    ProtocolErrorMessage,
    ProtocolRequest,
    ProtocolResponse,
    decode_line,
    encode_line,
    validate_request,
)

UNKNOWN_REQUEST_ID = "unknown"
_MISSING = object()


def extract_request_id(
    payload: Mapping[str, Any] | None, *, default: str = UNKNOWN_REQUEST_ID
) -> str:
    """Extract a stable request id from one decoded request payload."""
    if payload is not None:
        request_id = payload.get("id")
        if isinstance(request_id, str) and request_id:
            return request_id
    return default


def error_response(
    request_id: str,
    code: int | str,
    message: str,
    data: Any | None = None,
) -> ProtocolResponse:
    """Build a structured protocol error response."""
    return ProtocolResponse(
        id=request_id,
        error=ProtocolErrorMessage(code=code, message=message, data=data),
    )


def optional_nonempty_str(params: Mapping[str, Any], key: str) -> str | None:
    """Return one optional non-empty string param."""
    value = params.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string when provided")
    return value.strip()


def optional_string_key_mapping(params: Mapping[str, Any], key: str) -> dict[str, Any] | None:
    """Return one optional mapping with string-only keys."""
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


def dispatch_request_line(
    line: str | bytes,
    handlers: Mapping[str, Callable[[ProtocolRequest], ProtocolResponse]],
) -> ProtocolResponse:
    """Decode and validate one request line before dispatching by method."""
    request_id = UNKNOWN_REQUEST_ID

    try:
        payload = decode_line(line)
        request_id = extract_request_id(payload)
        request = validate_request(payload)
    except ProtocolDecodeError as exc:
        return error_response(
            request_id,
            code=-32600,
            message="invalid request",
            data={"details": str(exc)},
        )

    handler = handlers.get(request.method)
    if handler is None:
        return error_response(
            request.id,
            code=-32601,
            message="method not found",
            data={"method": request.method},
        )
    return handler(request)


def serve_runner(
    handle_request_line: Callable[..., ProtocolResponse],
    *,
    stdin: TextIO = sys.stdin,
    stdout: TextIO = sys.stdout,
    handler_arg: Any = _MISSING,
) -> int:
    """Run a simple line-oriented request/response loop."""
    for line in stdin:
        if not line.strip():
            continue
        if handler_arg is _MISSING:
            response = handle_request_line(line)
        else:
            response = handle_request_line(line, handler_arg)
        stdout.write(encode_line(response))
        stdout.flush()
    return 0


__all__ = [
    "UNKNOWN_REQUEST_ID",
    "dispatch_request_line",
    "error_response",
    "extract_request_id",
    "optional_nonempty_str",
    "optional_string_key_mapping",
    "serve_runner",
]
