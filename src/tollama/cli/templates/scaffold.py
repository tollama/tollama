"""String templates used by ``tollama dev scaffold``."""

from __future__ import annotations

RUNNER_INIT_TEMPLATE = '''"""Runner package for {family} family."""
'''

RUNNER_ADAPTER_TEMPLATE = '''"""Adapter for the {family} runner family."""

from __future__ import annotations

from tollama.core.schemas import ForecastRequest, ForecastResponse


class {class_name}Adapter:
    """Implement this adapter to serve {family} forecast requests."""

    def forecast(self, request: ForecastRequest) -> ForecastResponse:
        raise NotImplementedError("{class_name}Adapter.forecast is not implemented")
'''

RUNNER_MAIN_TEMPLATE = '''"""Runner process for the {family} family over stdio JSON-RPC."""

from __future__ import annotations

import sys
from collections.abc import Mapping
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
from tollama.core.schemas import ForecastRequest

from .adapter import {class_name}Adapter

RUNNER_NAME = "tollama-{family}"
RUNNER_VERSION = "0.1.0"
UNKNOWN_REQUEST_ID = "unknown"
CAPABILITIES = ("hello", "forecast")
_FORECAST_REQUEST_FIELDS = frozenset(
    {{"model", "horizon", "quantiles", "series", "options", "parameters"}},
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


def _handle_hello(request: ProtocolRequest) -> ProtocolResponse:
    return ProtocolResponse(
        id=request.id,
        result={{
            "name": RUNNER_NAME,
            "version": RUNNER_VERSION,
            "capabilities": list(CAPABILITIES),
            "supported_families": ["{family}"],
        }},
    )


def _handle_forecast(request: ProtocolRequest, adapter: {class_name}Adapter) -> ProtocolResponse:
    canonical_params = {{
        key: value for key, value in request.params.items() if key in _FORECAST_REQUEST_FIELDS
    }}
    try:
        forecast_request = ForecastRequest.model_validate(canonical_params)
    except Exception as exc:  # noqa: BLE001
        return _error_response(
            request.id,
            code=-32602,
            message="invalid params",
            data={{"details": str(exc)}},
        )

    try:
        response = adapter.forecast(forecast_request)
    except ValueError as exc:
        return _error_response(
            request.id,
            code="BAD_REQUEST",
            message=str(exc),
        )
    except NotImplementedError as exc:
        return _error_response(
            request.id,
            code="NOT_IMPLEMENTED",
            message=str(exc),
        )

    return ProtocolResponse(
        id=request.id,
        result=response.model_dump(mode="json", exclude_none=True),
    )


def handle_request_line(line: str | bytes, adapter: {class_name}Adapter) -> ProtocolResponse:
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
            data={{"details": str(exc)}},
        )

    if request.method == "hello":
        return _handle_hello(request)
    if request.method == "forecast":
        return _handle_forecast(request, adapter)

    return _error_response(
        request.id,
        code=-32601,
        message="method not found",
        data={{"method": request.method}},
    )


def serve(stdin: TextIO = sys.stdin, stdout: TextIO = sys.stdout) -> int:
    adapter = {class_name}Adapter()
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
'''

RUNNER_TEST_TEMPLATE = '''"""Protocol smoke tests for the {family} runner skeleton."""

from __future__ import annotations

import json

from tollama.runners.{family}_runner.main import handle_request_line


class _StubAdapter:
    def forecast(self, request):  # noqa: ANN001
        del request
        raise NotImplementedError("implement adapter.forecast first")


def test_{family}_runner_hello_reports_supported_family() -> None:
    response = handle_request_line(
        json.dumps({{"id": "req-1", "method": "hello", "params": {{}}}}),
        _StubAdapter(),
    )
    payload = response.model_dump(mode="json", exclude_none=True)
    assert payload["result"]["supported_families"] == ["{family}"]


def test_{family}_runner_forecast_returns_not_implemented_by_default() -> None:
    response = handle_request_line(
        json.dumps(
            {{
                "id": "req-2",
                "method": "forecast",
                "params": {{
                    "model": "{family}-example",
                    "horizon": 2,
                    "series": [
                        {{
                            "id": "s1",
                            "freq": "D",
                            "timestamps": ["2025-01-01", "2025-01-02"],
                            "target": [1.0, 2.0],
                        }}
                    ],
                    "quantiles": [],
                    "options": {{}},
                }},
            }},
        ),
        _StubAdapter(),
    )
    payload = response.model_dump(mode="json", exclude_none=True)
    assert payload["error"]["code"] == "NOT_IMPLEMENTED"
'''

REGISTRY_ENTRY_TEMPLATE = '''
  - name: {model_name}
    family: {family}
    source:
      type: local
      repo_id: tollama/{family}-runner
      revision: main
    license:
      type: mit
      needs_acceptance: false
    metadata:
      implementation: {family}
      max_context: 512
      max_horizon: 64
    capabilities:
      past_covariates_numeric: false
      past_covariates_categorical: false
      future_covariates_numeric: false
      future_covariates_categorical: false
      static_covariates: false
'''

