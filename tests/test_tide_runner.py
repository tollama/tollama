"""Unit tests for TiDE runner protocol behavior."""

from __future__ import annotations

import json
from typing import Any

from tollama.core.schemas import ForecastResponse, SeriesForecast
from tollama.runners.tide_runner.errors import DependencyMissingError
from tollama.runners.tide_runner.main import handle_request_line


class _CapturingAdapter:
    def __init__(self) -> None:
        self.forecast_calls: list[dict[str, Any]] = []

    def unload(self, model_name: str | None = None) -> None:
        del model_name

    def forecast(
        self,
        request,
        *,
        model_local_dir: str | None = None,
        model_source: dict[str, object] | None = None,
        model_metadata: dict[str, object] | None = None,
    ) -> ForecastResponse:
        self.forecast_calls.append(
            {
                "request": request,
                "model_local_dir": model_local_dir,
                "model_source": model_source,
                "model_metadata": model_metadata,
            },
        )
        return ForecastResponse(
            model=request.model,
            forecasts=[
                SeriesForecast(
                    id=request.series[0].id,
                    freq=request.series[0].freq,
                    start_timestamp="2025-01-03T00:00:00Z",
                    mean=[2.0, 2.0],
                ),
            ],
            usage={"runner": "tollama-tide"},
        )


class _MissingDependencyAdapter(_CapturingAdapter):
    def forecast(
        self,
        request,
        *,
        model_local_dir: str | None = None,
        model_source: dict[str, object] | None = None,
        model_metadata: dict[str, object] | None = None,
    ) -> ForecastResponse:
        del request, model_local_dir, model_source, model_metadata
        raise DependencyMissingError(
            "missing optional TiDE runner dependencies (darts); "
            "install them with `pip install -e \".[dev,runner_tide]\"`",
        )


def _valid_forecast_params() -> dict[str, Any]:
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


def test_tide_runner_hello_reports_supported_family_and_status() -> None:
    response = handle_request_line(
        json.dumps({"id": "req-1", "method": "hello", "params": {}}),
        _CapturingAdapter(),
    )
    payload = response.model_dump(mode="json", exclude_none=True)
    assert payload["result"]["supported_families"] == ["tide"]
    assert payload["result"]["status"] == "phase2_inference"


def test_tide_runner_forecast_returns_dependency_missing_when_dependencies_absent() -> None:
    response = handle_request_line(
        json.dumps({"id": "req-2", "method": "forecast", "params": _valid_forecast_params()}),
        _MissingDependencyAdapter(),
    )
    payload = response.model_dump(mode="json", exclude_none=True)
    assert payload["error"]["code"] == "DEPENDENCY_MISSING"
    assert "runner_tide" in payload["error"]["message"]


def test_tide_runner_forecast_smoke_wires_request_and_response() -> None:
    adapter = _CapturingAdapter()
    params = _valid_forecast_params()
    params["model_local_dir"] = " /tmp/tide "
    params["model_source"] = {"repo_id": "unit8co/tide", "revision": "main"}
    params["model_metadata"] = {"implementation": "tide"}

    response = handle_request_line(
        json.dumps({"id": "req-3", "method": "forecast", "params": params}),
        adapter,
    )
    payload = response.model_dump(mode="json", exclude_none=True)
    assert payload["result"]["model"] == "tide"
    assert payload["result"]["forecasts"][0]["mean"] == [2.0, 2.0]

    assert len(adapter.forecast_calls) == 1
    call = adapter.forecast_calls[0]
    assert call["model_local_dir"] == "/tmp/tide"
