"""PatchTST runner process over stdio JSON-RPC (Phase-1 placeholder)."""

from __future__ import annotations

import importlib
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

RUNNER_NAME = "tollama-patchtst"
RUNNER_VERSION = "0.1.0"
UNKNOWN_REQUEST_ID = "unknown"
CAPABILITIES = ("hello", "forecast")
_FORECAST_REQUEST_FIELDS = frozenset(
    {"model", "horizon", "quantiles", "series", "options", "parameters"},
)
_REQUIRED_DEPENDENCIES = ("transformers",)
_INSTALL_HINT = 'pip install -e ".[dev,runner_patchtst]"'


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


def _missing_dependencies() -> list[str]:
    missing: list[str] = []
    for module_name in _REQUIRED_DEPENDENCIES:
        try:
            importlib.import_module(module_name)
        except Exception:  # noqa: BLE001
            missing.append(module_name)
    return missing


def _handle_hello(request: ProtocolRequest) -> ProtocolResponse:
    return ProtocolResponse(
        id=request.id,
        result={
            "name": RUNNER_NAME,
            "version": RUNNER_VERSION,
            "capabilities": list(CAPABILITIES),
            "supported_families": ["patchtst"],
            "status": "phase1_placeholder",
        },
    )


def _handle_forecast(request: ProtocolRequest) -> ProtocolResponse:
    canonical_params = {
        key: value for key, value in request.params.items() if key in _FORECAST_REQUEST_FIELDS
    }
    try:
        ForecastRequest.model_validate(canonical_params)
    except ValidationError as exc:
        return _error_response(
            request.id,
            code=-32602,
            message="invalid params",
            data={"details": str(exc)},
        )

    missing = _missing_dependencies()
    if missing:
        joined = ", ".join(sorted(missing))
        return _error_response(
            request.id,
            code="DEPENDENCY_MISSING",
            message=(
                "missing optional patchtst runner dependencies "
                f"({joined}); install them with `{_INSTALL_HINT}`"
            ),
            data={"missing": missing, "install": _INSTALL_HINT},
        )

    return _error_response(
        request.id,
        code="NOT_IMPLEMENTED",
        message=(
            "PatchTST Phase-1 baseline is registered and routable, "
            "but full inference is not implemented yet."
        ),
        data={
            "status": "phase1_placeholder",
            "next_step": "Implement a PatchTST adapter and wire forecast execution.",
        },
    )


def handle_request_line(line: str | bytes) -> ProtocolResponse:
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
        return _handle_forecast(request)

    return _error_response(
        request.id,
        code=-32601,
        message="method not found",
        data={"method": request.method},
    )


def serve(stdin: TextIO = sys.stdin, stdout: TextIO = sys.stdout) -> int:
    for line in stdin:
        if not line.strip():
            continue
        response = handle_request_line(line)
        stdout.write(encode_line(response))
        stdout.flush()
    return 0


def main() -> int:
    return serve()


if __name__ == "__main__":
    raise SystemExit(main())
