"""Unit tests for TiDE runner process wiring."""

from __future__ import annotations

import json

from tollama.core.schemas import ForecastResponse, SeriesForecast
from tollama.runners.tide_runner import main as tide_main


def _valid_forecast_params() -> dict[str, object]:
    return {
        "model": "tide",
        "horizon": 2,
        "quantiles": [0.1, 0.9],
        "series": [
            {
                "id": "s1",
                "freq": "D",
                "timestamps": ["2025-01-01", "2025-01-02"],
                "target": [1.0, 2.0],
            }
        ],
        "options": {},
    }


class _OkAdapter:
    def forecast(self, request, **kwargs):
        del request, kwargs
        return ForecastResponse(
            model="tide",
            forecasts=[
                SeriesForecast(
                    id="s1",
                    freq="D",
                    start_timestamp="2025-01-03T00:00:00Z",
                    mean=[1.0, 1.0],
                    quantiles={"0.1": [0.9, 0.9], "0.9": [1.1, 1.1]},
                ),
            ],
        )

    def unload(self, model_name=None):
        del model_name


def test_tide_runner_hello_reports_supported_family_and_status() -> None:
    response = tide_main.handle_request_line(
        json.dumps({"id": "req-1", "method": "hello", "params": {}}),
        adapter=_OkAdapter(),
    )
    payload = response.model_dump(mode="json", exclude_none=True)
    assert payload["result"]["supported_families"] == ["tide"]
    assert payload["result"]["status"] == "phase3_probabilistic"


def test_tide_runner_forecast_happy_path_returns_result_payload() -> None:
    response = tide_main.handle_request_line(
        json.dumps({"id": "req-2", "method": "forecast", "params": _valid_forecast_params()}),
        adapter=_OkAdapter(),
    )
    payload = response.model_dump(mode="json", exclude_none=True)
    assert payload["result"]["model"] == "tide"
    assert payload["result"]["forecasts"][0]["quantiles"]["0.1"] == [0.9, 0.9]
