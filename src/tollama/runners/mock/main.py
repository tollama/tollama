"""Mock runner process using newline-delimited JSON over stdio."""

from __future__ import annotations

import sys
from typing import Any, TextIO

from pydantic import ValidationError

from tollama.core.protocol import ProtocolRequest, ProtocolResponse
from tollama.core.schemas import ForecastRequest, ForecastResponse, SeriesForecast
from tollama.runners.common_protocol import dispatch_request_line, error_response, serve_runner

RUNNER_NAME = "tollama-mock"
RUNNER_VERSION = "0.1.0"
CAPABILITIES = ("hello", "forecast")
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
        return error_response(
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
    return dispatch_request_line(
        line,
        {
            "hello": _handle_hello,
            "forecast": _handle_forecast,
        },
    )


def serve(stdin: TextIO = sys.stdin, stdout: TextIO = sys.stdout) -> int:
    return serve_runner(handle_request_line, stdin=stdin, stdout=stdout)


def main() -> int:
    return serve()


if __name__ == "__main__":
    raise SystemExit(main())
