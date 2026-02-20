"""Unit tests for torch runner protocol behavior."""

from __future__ import annotations

import json

from tollama.core.schemas import ForecastResponse
from tollama.runners.torch_runner.errors import DependencyMissingError
from tollama.runners.torch_runner.main import handle_request_line


class _NoopAdapter:
    def load(
        self,
        model_name: str,
        *,
        model_local_dir: str | None = None,
        model_source: dict[str, object] | None = None,
        model_metadata: dict[str, object] | None = None,
    ) -> None:
        del model_name, model_local_dir, model_source, model_metadata
        return None

    def unload(self, model_name: str | None = None) -> None:
        del model_name
        return None

    def forecast(
        self,
        request,
        *,
        model_local_dir: str | None = None,
        model_source: dict[str, object] | None = None,
        model_metadata: dict[str, object] | None = None,
    ):  # pragma: no cover
        del request, model_local_dir, model_source, model_metadata
        raise AssertionError("unexpected call")


class _MissingDependencyAdapter(_NoopAdapter):
    def forecast(
        self,
        request,
        *,
        model_local_dir: str | None = None,
        model_source: dict[str, object] | None = None,
        model_metadata: dict[str, object] | None = None,
    ):
        del request, model_local_dir, model_source, model_metadata
        raise DependencyMissingError("install with pip install -e \".[dev,runner_torch]\"")


class _SuccessAdapter(_NoopAdapter):
    def forecast(
        self,
        request,
        *,
        model_local_dir: str | None = None,
        model_source: dict[str, object] | None = None,
        model_metadata: dict[str, object] | None = None,
    ) -> ForecastResponse:
        del request, model_local_dir, model_source, model_metadata
        return ForecastResponse.model_validate(
            {
                "model": "chronos2",
                "forecasts": [
                    {
                        "id": "s1",
                        "freq": "D",
                        "start_timestamp": "2025-01-02",
                        "mean": [1.0],
                    }
                ],
            },
        )


def test_torch_runner_hello_reports_supported_family() -> None:
    response = handle_request_line(
        json.dumps({"id": "req-1", "method": "hello", "params": {}}),
        _NoopAdapter(),
    )
    payload = response.model_dump(mode="json", exclude_none=True)
    assert payload["id"] == "req-1"
    assert payload["result"]["name"] == "tollama-torch"
    assert payload["result"]["supported_families"] == ["torch"]


def test_torch_runner_forecast_returns_dependency_missing_error() -> None:
    response = handle_request_line(
        json.dumps(
            {
                "id": "req-2",
                "method": "forecast",
                "params": {
                    "model": "chronos2",
                    "horizon": 1,
                    "series": [
                        {
                            "id": "s1",
                            "freq": "D",
                            "timestamps": ["2025-01-01"],
                            "target": [1.0],
                        }
                    ],
                    "quantiles": [],
                    "options": {},
                },
            },
        ),
        _MissingDependencyAdapter(),
    )
    payload = response.model_dump(mode="json", exclude_none=True)
    assert payload["id"] == "req-2"
    assert payload["error"]["code"] == "DEPENDENCY_MISSING"
    assert "runner_torch" in payload["error"]["message"]


def test_torch_runner_forecast_enriches_usage_and_timing() -> None:
    response = handle_request_line(
        json.dumps(
            {
                "id": "req-3",
                "method": "forecast",
                "params": {
                    "model": "chronos2",
                    "horizon": 1,
                    "series": [
                        {
                            "id": "s1",
                            "freq": "D",
                            "timestamps": ["2025-01-01"],
                            "target": [1.0],
                        }
                    ],
                    "quantiles": [],
                    "options": {},
                },
            },
        ),
        _SuccessAdapter(),
    )
    payload = response.model_dump(mode="json", exclude_none=True)
    assert payload["id"] == "req-3"
    result = payload["result"]
    assert result["usage"]["runner"] == "tollama-torch"
    assert result["usage"]["device"] == "unknown"
    assert result["usage"]["peak_memory_mb"] >= 0.0
    assert result["timing"]["model_load_ms"] == 0.0
    assert result["timing"]["inference_ms"] >= 0.0
