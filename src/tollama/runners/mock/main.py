"""Mock runner process using newline-delimited JSON over stdio."""

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
from tollama.core.schemas import ForecastRequest, ForecastResponse, SeriesForecast

RUNNER_NAME = "tollama-mock"
RUNNER_VERSION = "0.1.0"
UNKNOWN_REQUEST_ID = "unknown"
CAPABILITIES = ("hello", "forecast")
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
    code: int,
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
        result={
            "name": RUNNER_NAME,
            "version": RUNNER_VERSION,
            "capabilities": list(CAPABILITIES),
        },
    )


def _series_forecast_payload(request: ForecastRequest) -> list[dict[str, Any]]:
    series_forecasts: list[dict[str, Any]] = []

    for series in request.series:
        last_value = series.target[-1]
        mean = [last_value] * request.horizon

        quantiles_payload: dict[str, list[float | int]] | None = None
        if request.quantiles:
            quantiles_payload = {
                format(quantile, "g"): [last_value] * request.horizon
                for quantile in request.quantiles
            }

        forecast = SeriesForecast(
            id=series.id,
            freq=series.freq,
            start_timestamp=series.timestamps[-1],
            mean=mean,
            quantiles=quantiles_payload,
        )
        series_forecasts.append(forecast.model_dump(mode="json", exclude_none=True))

    return series_forecasts


def _handle_forecast(request: ProtocolRequest) -> ProtocolResponse:
    canonical_params = {
        key: value for key, value in request.params.items() if key in _FORECAST_REQUEST_FIELDS
    }
    try:
        forecast_request = ForecastRequest.model_validate(canonical_params)
    except ValidationError as exc:
        return _error_response(
            request.id,
            code=-32602,
            message="invalid params",
            data={"details": str(exc)},
        )

    response = ForecastResponse(
        model=forecast_request.model,
        forecasts=[
            SeriesForecast.model_validate(item)
            for item in _series_forecast_payload(forecast_request)
        ],
        usage={
            "runner": RUNNER_NAME,
            "series_count": len(forecast_request.series),
            "horizon": forecast_request.horizon,
        },
    )
    return ProtocolResponse(
        id=request.id,
        result=response.model_dump(mode="json", exclude_none=True),
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
