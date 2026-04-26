"""Unit tests for lag-llama runner protocol behavior."""

from __future__ import annotations

import json
from typing import Any

from tollama.core.schemas import ForecastResponse, SeriesForecast
from tollama.runners.lag_llama_runner.errors import DependencyMissingError
from tollama.runners.lag_llama_runner.main import handle_request_line


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
                    quantiles={"0.1": [1.0, 1.0], "0.9": [3.0, 3.0]},
                ),
            ],
            usage={"runner": "tollama-lag-llama"},
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
            "missing optional lag-llama runner dependencies (lag_llama); "
            'install them with `pip install -e ".[dev,runner_lag_llama]"`',
        )


class _RuntimeErrorAdapter(_CapturingAdapter):
    def forecast(
        self,
        request,
        *,
        model_local_dir: str | None = None,
        model_source: dict[str, object] | None = None,
        model_metadata: dict[str, object] | None = None,
    ) -> ForecastResponse:
        del request, model_local_dir, model_source, model_metadata
        raise RuntimeError("backend exploded")


def _valid_forecast_params() -> dict[str, Any]:
    return {
        "model": "lag-llama",
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


def test_lag_llama_runner_hello_reports_supported_family_and_capabilities() -> None:
    response = handle_request_line(
        json.dumps({"id": "req-1", "method": "hello", "params": {}}),
        _CapturingAdapter(),
    )
    payload = response.model_dump(mode="json", exclude_none=True)
    assert payload["id"] == "req-1"
    assert payload["result"]["name"] == "tollama-lag-llama"
    assert payload["result"]["supported_families"] == ["lag_llama"]
    assert payload["result"]["capabilities"] == ["hello", "forecast", "unload"]


def test_lag_llama_runner_rejects_unsupported_load_method() -> None:
    response = handle_request_line(
        json.dumps({"id": "req-load", "method": "load", "params": {"model": "lag-llama"}}),
        _CapturingAdapter(),
    )
    payload = response.model_dump(mode="json", exclude_none=True)
    assert payload["id"] == "req-load"
    assert payload["error"]["code"] == -32601
    assert payload["error"]["message"] == "method not found"


def test_lag_llama_runner_forecast_returns_dependency_missing_error() -> None:
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
    assert "runner_lag_llama" in payload["error"]["message"]


def test_lag_llama_runner_forecast_returns_runtime_error_without_crashing() -> None:
    response = handle_request_line(
        json.dumps(
            {
                "id": "req-runtime",
                "method": "forecast",
                "params": _valid_forecast_params(),
            },
        ),
        _RuntimeErrorAdapter(),
    )
    payload = response.model_dump(mode="json", exclude_none=True)
    assert payload["id"] == "req-runtime"
    assert payload["error"]["code"] == "FORECAST_ERROR"
    assert payload["error"]["message"] == "RuntimeError: backend exploded"


def test_lag_llama_runner_validates_runner_specific_optional_params() -> None:
    params = _valid_forecast_params()
    params["model_local_dir"] = 123

    response = handle_request_line(
        json.dumps({"id": "req-3", "method": "forecast", "params": params}),
        _CapturingAdapter(),
    )
    payload = response.model_dump(mode="json", exclude_none=True)
    assert payload["id"] == "req-3"
    assert payload["error"]["code"] == "BAD_REQUEST"
    assert payload["error"]["message"] == "model_local_dir must be a non-empty string when provided"


def test_lag_llama_runner_rejects_non_object_model_source_before_adapter_invocation() -> None:
    params = _valid_forecast_params()
    params["model_source"] = "hf://time-series-foundation-models/Lag-Llama"

    response = handle_request_line(
        json.dumps({"id": "req-3b", "method": "forecast", "params": params}),
        _CapturingAdapter(),
    )
    payload = response.model_dump(mode="json", exclude_none=True)
    assert payload["id"] == "req-3b"
    assert payload["error"]["code"] == "BAD_REQUEST"
    assert payload["error"]["message"] == "model_source must be an object when provided"


def test_lag_llama_runner_performs_payload_validation_before_dependency_gating() -> None:
    params = _valid_forecast_params()
    params["model_local_dir"] = 123

    response = handle_request_line(
        json.dumps({"id": "req-3c", "method": "forecast", "params": params}),
        _MissingDependencyAdapter(),
    )
    payload = response.model_dump(mode="json", exclude_none=True)
    assert payload["id"] == "req-3c"
    assert payload["error"]["code"] == "BAD_REQUEST"
    assert "model_local_dir must be a non-empty string" in payload["error"]["message"]


def test_lag_llama_runner_returns_invalid_params_for_schema_validation_errors() -> None:
    params = _valid_forecast_params()
    params["horizon"] = "2"

    response = handle_request_line(
        json.dumps({"id": "req-4", "method": "forecast", "params": params}),
        _CapturingAdapter(),
    )
    payload = response.model_dump(mode="json", exclude_none=True)
    assert payload["id"] == "req-4"
    assert payload["error"]["code"] == -32602
    assert payload["error"]["message"] == "invalid params"


def test_lag_llama_runner_forecast_smoke_wires_request_and_response() -> None:
    adapter = _CapturingAdapter()
    params = _valid_forecast_params()
    params["model_local_dir"] = " /tmp/lag-llama "
    params["model_source"] = {
        "repo_id": "time-series-foundation-models/Lag-Llama",
        "revision": "main",
    }
    params["model_metadata"] = {"implementation": "lag_llama"}
    params["ignored"] = {"should": "not_break"}

    response = handle_request_line(
        json.dumps({"id": "req-5", "method": "forecast", "params": params}),
        adapter,
    )
    payload = response.model_dump(mode="json", exclude_none=True)

    assert payload["id"] == "req-5"
    assert payload["result"]["model"] == "lag-llama"
    assert payload["result"]["forecasts"][0]["id"] == "s1"
    assert payload["result"]["forecasts"][0]["mean"] == [2.0, 2.0]

    assert len(adapter.forecast_calls) == 1
    call = adapter.forecast_calls[0]
    assert call["request"].model == "lag-llama"
    assert call["request"].horizon == 2
    assert call["model_local_dir"] == "/tmp/lag-llama"
    assert call["model_source"] == {
        "repo_id": "time-series-foundation-models/Lag-Llama",
        "revision": "main",
    }
    assert call["model_metadata"] == {"implementation": "lag_llama"}
