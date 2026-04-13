"""Torch runner process using newline-delimited JSON over stdio."""

from __future__ import annotations

import sys
import time
from collections.abc import Mapping
from typing import Any, TextIO

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

from .adapter_router import TorchAdapterRouter
from .errors import AdapterInputError, DependencyMissingError, UnsupportedModelError

RUNNER_NAME = "tollama-torch"
RUNNER_VERSION = "0.1.0"
CAPABILITIES = ("hello", "load", "unload", "forecast")
_FORECAST_REQUEST_FIELDS = frozenset(
    {"model", "horizon", "quantiles", "series", "options", "parameters"},
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


def _handle_load(request: ProtocolRequest, adapter_router: TorchAdapterRouter) -> ProtocolResponse:
    try:
        model_name = _require_model_name(request.params)
        model_local_dir = optional_nonempty_str(request.params, "model_local_dir")
        model_source = optional_string_key_mapping(request.params, "model_source")
        model_metadata = optional_string_key_mapping(request.params, "model_metadata")
        adapter_router.load(
            model_name,
            model_local_dir=model_local_dir,
            model_source=model_source,
            model_metadata=model_metadata,
        )
    except DependencyMissingError as exc:
        return error_response(
            request.id,
            code="DEPENDENCY_MISSING",
            message=str(exc),
        )
    except UnsupportedModelError as exc:
        return error_response(
            request.id,
            code="MODEL_UNSUPPORTED",
            message=str(exc),
        )
    except (AdapterInputError, ValueError) as exc:
        return error_response(
            request.id,
            code="BAD_REQUEST",
            message=str(exc),
        )

    return ProtocolResponse(
        id=request.id,
        result={"loaded": True, "model": model_name},
    )


def _handle_unload(
    request: ProtocolRequest,
    adapter_router: TorchAdapterRouter,
) -> ProtocolResponse:
    model_name = request.params.get("model")
    if model_name is not None and (not isinstance(model_name, str) or not model_name):
        return error_response(
            request.id,
            code=-32602,
            message="invalid params",
            data={"details": "model must be a non-empty string when provided"},
        )
    adapter_router.unload(model_name if isinstance(model_name, str) else None)
    return ProtocolResponse(
        id=request.id,
        result={"unloaded": True, "model": model_name},
    )


def _handle_forecast(
    request: ProtocolRequest,
    adapter_router: TorchAdapterRouter,
) -> ProtocolResponse:
    canonical_params = {
        key: value for key, value in request.params.items() if key in _FORECAST_REQUEST_FIELDS
    }
    try:
        model_local_dir = optional_nonempty_str(request.params, "model_local_dir")
        model_source = optional_string_key_mapping(request.params, "model_source")
        model_metadata = optional_string_key_mapping(request.params, "model_metadata")
    except ValueError as exc:
        return error_response(
            request.id,
            code="BAD_REQUEST",
            message=str(exc),
        )

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
        response = adapter_router.forecast(
            forecast_request,
            model_local_dir=model_local_dir,
            model_source=model_source,
            model_metadata=model_metadata,
        )
        inference_ms = (time.perf_counter() - started_at) * 1000.0
    except DependencyMissingError as exc:
        return error_response(
            request.id,
            code="DEPENDENCY_MISSING",
            message=str(exc),
        )
    except UnsupportedModelError as exc:
        return error_response(
            request.id,
            code="MODEL_UNSUPPORTED",
            message=str(exc),
        )
    except AdapterInputError as exc:
        return error_response(
            request.id,
            code="BAD_REQUEST",
            message=str(exc),
        )
    except ValueError as exc:
        return error_response(
            request.id,
            code="FORECAST_ERROR",
            message=str(exc),
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


def handle_request_line(line: str | bytes, adapter_router: TorchAdapterRouter) -> ProtocolResponse:
    return dispatch_request_line(
        line,
        {
            "hello": _handle_hello,
            "load": lambda request: _handle_load(request, adapter_router),
            "unload": lambda request: _handle_unload(request, adapter_router),
            "forecast": lambda request: _handle_forecast(request, adapter_router),
        },
    )


def serve(stdin: TextIO = sys.stdin, stdout: TextIO = sys.stdout) -> int:
    return serve_runner(
        handle_request_line,
        stdin=stdin,
        stdout=stdout,
        handler_arg=TorchAdapterRouter(),
    )


def main() -> int:
    return serve()


if __name__ == "__main__":
    raise SystemExit(main())
