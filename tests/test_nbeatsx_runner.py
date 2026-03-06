"""Unit tests for N-BEATSx runner process wiring."""

from __future__ import annotations

import json
from typing import Any

from tollama.core.schemas import ForecastResponse, SeriesForecast
from tollama.runners.nbeatsx_runner.errors import DependencyMissingError
from tollama.runners.nbeatsx_runner.main import handle_request_line


class _CapturingAdapter:
    def __init__(self) -> None:
        self.unloaded_models: list[str | None] = []
        self.forecast_calls: list[dict[str, Any]] = []

    def unload(self, model_name: str | None = None) -> None:
        self.unloaded_models.append(model_name)

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
                    quantiles=None,
                ),
            ],
            usage={"runner": "tollama-nbeatsx"},
            warnings=["point forecasts only"],
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
            "missing optional nbeatsx runner dependencies (neuralforecast); "
            "install them with `pip install -e \".[dev,runner_nbeatsx]\"`",
        )


def _valid_forecast_params() -> dict[str, Any]:
    return {
        "model": "nbeatsx",
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


def test_nbeatsx_runner_hello_reports_supported_family_and_status() -> None:
    response = handle_request_line(
        json.dumps({"id": "req-1", "method": "hello", "params": {}}),
        _CapturingAdapter(),
    )
    payload = response.model_dump(mode="json", exclude_none=True)
    assert payload["result"]["supported_families"] == ["nbeatsx"]
    assert payload["result"]["status"] == "phase2_inference"


def test_nbeatsx_runner_forecast_returns_dependency_missing_error() -> None:
    response = handle_request_line(
        json.dumps(
            {
                "id": "req-2",
                "method": "forecast",
                "params": _valid_forecast_params(),
            },
        ),
        _MissingDependencyAdapter(),
    )
    payload = response.model_dump(mode="json", exclude_none=True)
    assert payload["id"] == "req-2"
    assert payload["error"]["code"] == "DEPENDENCY_MISSING"
    assert "runner_nbeatsx" in payload["error"]["message"]


def test_nbeatsx_runner_forecast_smoke_wires_request_and_response() -> None:
    adapter = _CapturingAdapter()
    params = _valid_forecast_params()
    params["model_local_dir"] = " /tmp/nbeatsx "
    params["model_source"] = {"repo_id": "cchallu/nbeatsx-air-passengers", "revision": "main"}
    params["model_metadata"] = {"implementation": "nbeatsx"}

    response = handle_request_line(
        json.dumps({"id": "req-3", "method": "forecast", "params": params}),
        adapter,
    )
    payload = response.model_dump(mode="json", exclude_none=True)

    assert payload["id"] == "req-3"
    assert payload["result"]["model"] == "nbeatsx"
    assert payload["result"]["forecasts"][0]["id"] == "s1"
    assert payload["result"]["forecasts"][0]["mean"] == [2.0, 2.0]

    assert len(adapter.forecast_calls) == 1
    call = adapter.forecast_calls[0]
    assert call["request"].model == "nbeatsx"
    assert call["request"].horizon == 2
    assert call["model_local_dir"] == "/tmp/nbeatsx"
    assert call["model_source"] == {"repo_id": "cchallu/nbeatsx-air-passengers", "revision": "main"}
    assert call["model_metadata"] == {"implementation": "nbeatsx"}


def test_nbeatsx_runner_unload_calls_adapter() -> None:
    adapter = _CapturingAdapter()
    response = handle_request_line(
        json.dumps({"id": "req-4", "method": "unload", "params": {"model": "nbeatsx"}}),
        adapter,
    )
    payload = response.model_dump(mode="json", exclude_none=True)
    assert payload["result"] == {"unloaded": True, "model": "nbeatsx"}
    assert adapter.unloaded_models == ["nbeatsx"]
