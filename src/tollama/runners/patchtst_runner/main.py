"""PatchTST runner process over stdio JSON-RPC."""

from __future__ import annotations

import sys
import time
from typing import TextIO

from pydantic import ValidationError

from tollama.core.protocol import ProtocolRequest, ProtocolResponse
from tollama.core.schemas import ForecastRequest
from tollama.runners.common_protocol import (
    dispatch_request_line,
    error_response,
    optional_nonempty_str,
    optional_string_key_mapping,
    serve_runner,
)
from tollama.runners.runtime_telemetry import enrich_forecast_response

from .adapter import PatchTSTAdapter
from .errors import AdapterInputError, DependencyMissingError, UnsupportedModelError

RUNNER_NAME = "tollama-patchtst"
RUNNER_VERSION = "0.2.1"
CAPABILITIES = ("hello", "forecast", "unload")
_FORECAST_REQUEST_FIELDS = frozenset(
    {"model", "horizon", "quantiles", "series", "options", "parameters"},
)


def _handle_hello(request: ProtocolRequest) -> ProtocolResponse:
    return ProtocolResponse(
        id=request.id,
        result={
            "name": RUNNER_NAME,
            "version": RUNNER_VERSION,
            "capabilities": list(CAPABILITIES),
            "supported_families": ["patchtst"],
            "status": "phase2_inference",
        },
    )


def _handle_unload(request: ProtocolRequest, adapter: PatchTSTAdapter) -> ProtocolResponse:
    model_name = request.params.get("model")
    if model_name is not None and (not isinstance(model_name, str) or not model_name):
        return error_response(
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


def _handle_forecast(request: ProtocolRequest, adapter: PatchTSTAdapter) -> ProtocolResponse:
    canonical_params = {
        key: value for key, value in request.params.items() if key in _FORECAST_REQUEST_FIELDS
    }
    try:
        model_local_dir = optional_nonempty_str(request.params, "model_local_dir")
        model_source = optional_string_key_mapping(request.params, "model_source")
        model_metadata = optional_string_key_mapping(request.params, "model_metadata")
    except ValueError as exc:
        return error_response(request.id, code="BAD_REQUEST", message=str(exc))

    try:
        forecast_request = ForecastRequest.model_validate(canonical_params)
    except ValidationError as exc:
        return error_response(
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
        return error_response(request.id, code="DEPENDENCY_MISSING", message=str(exc))
    except UnsupportedModelError as exc:
        return error_response(request.id, code="MODEL_UNSUPPORTED", message=str(exc))
    except AdapterInputError as exc:
        return error_response(request.id, code="BAD_REQUEST", message=str(exc))
    except ValueError as exc:
        return error_response(request.id, code="FORECAST_ERROR", message=str(exc))

    response = enrich_forecast_response(
        response=response,
        runner_name=RUNNER_NAME,
        inference_ms=inference_ms,
    )
    return ProtocolResponse(
        id=request.id,
        result=response.model_dump(mode="json", exclude_none=True),
    )


def handle_request_line(line: str | bytes, adapter: PatchTSTAdapter) -> ProtocolResponse:
    return dispatch_request_line(
        line,
        {
            "hello": _handle_hello,
            "forecast": lambda request: _handle_forecast(request, adapter),
            "unload": lambda request: _handle_unload(request, adapter),
        },
    )


def serve(stdin: TextIO = sys.stdin, stdout: TextIO = sys.stdout) -> int:
    return serve_runner(
        handle_request_line,
        stdin=stdin,
        stdout=stdout,
        handler_arg=PatchTSTAdapter(),
    )


def main() -> int:
    return serve()


if __name__ == "__main__":
    raise SystemExit(main())
