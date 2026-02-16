"""Unit tests for sundial runner protocol behavior."""

from __future__ import annotations

import json

from tollama.runners.sundial_runner.errors import DependencyMissingError
from tollama.runners.sundial_runner.main import handle_request_line


class _NoopAdapter:
    def __init__(self) -> None:
        self.unloaded_models: list[str | None] = []

    def unload(self, model_name: str | None = None) -> None:
        self.unloaded_models.append(model_name)

    def forecast(
        self,
        request,
        *,
        model_local_dir: str | None = None,
        model_source: dict[str, object] | None = None,
        model_metadata: dict[str, object] | None = None,
    ):
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
        raise DependencyMissingError(
            "install with pip install -e \".[dev,runner_sundial]\"",
        )


class _CrashAdapter(_NoopAdapter):
    def forecast(
        self,
        request,
        *,
        model_local_dir: str | None = None,
        model_source: dict[str, object] | None = None,
        model_metadata: dict[str, object] | None = None,
    ):
        del request, model_local_dir, model_source, model_metadata
        raise RuntimeError("unexpected crash")


def test_sundial_runner_hello_reports_supported_family() -> None:
    response = handle_request_line(
        json.dumps({"id": "req-1", "method": "hello", "params": {}}),
        _NoopAdapter(),
    )
    payload = response.model_dump(mode="json", exclude_none=True)
    assert payload["id"] == "req-1"
    assert payload["result"]["name"] == "tollama-sundial"
    assert payload["result"]["supported_families"] == ["sundial"]


def test_sundial_runner_invalid_request_maps_protocol_error() -> None:
    response = handle_request_line(
        json.dumps({"id": "req-2", "params": {}}),
        _NoopAdapter(),
    )
    payload = response.model_dump(mode="json", exclude_none=True)
    assert payload["id"] == "req-2"
    assert payload["error"]["code"] == -32600
    assert payload["error"]["message"] == "invalid request"


def test_sundial_runner_forecast_returns_dependency_missing_error() -> None:
    response = handle_request_line(
        json.dumps(
            {
                "id": "req-3",
                "method": "forecast",
                "params": {
                    "model": "sundial-base-128m",
                    "horizon": 2,
                    "series": [
                        {
                            "id": "s1",
                            "freq": "D",
                            "timestamps": ["2025-01-01", "2025-01-02"],
                            "target": [1.0, 2.0],
                        }
                    ],
                    "quantiles": [0.1, 0.9],
                    "options": {},
                },
            },
        ),
        _MissingDependencyAdapter(),
    )
    payload = response.model_dump(mode="json", exclude_none=True)
    assert payload["id"] == "req-3"
    assert payload["error"]["code"] == "DEPENDENCY_MISSING"
    assert "runner_sundial" in payload["error"]["message"]


def test_sundial_runner_forecast_maps_unexpected_exceptions() -> None:
    response = handle_request_line(
        json.dumps(
            {
                "id": "req-3b",
                "method": "forecast",
                "params": {
                    "model": "sundial-base-128m",
                    "horizon": 2,
                    "series": [
                        {
                            "id": "s1",
                            "freq": "D",
                            "timestamps": ["2025-01-01", "2025-01-02"],
                            "target": [1.0, 2.0],
                        }
                    ],
                    "quantiles": [0.1, 0.9],
                    "options": {},
                },
            },
        ),
        _CrashAdapter(),
    )
    payload = response.model_dump(mode="json", exclude_none=True)
    assert payload["id"] == "req-3b"
    assert payload["error"]["code"] == "FORECAST_ERROR"
    assert payload["error"]["message"] == "RuntimeError: unexpected crash"


def test_sundial_runner_unload_calls_adapter() -> None:
    adapter = _NoopAdapter()
    response = handle_request_line(
        json.dumps(
            {
                "id": "req-4",
                "method": "unload",
                "params": {"model": "sundial-base-128m"},
            },
        ),
        adapter,
    )
    payload = response.model_dump(mode="json", exclude_none=True)
    assert payload["id"] == "req-4"
    assert payload["result"] == {"unloaded": True, "model": "sundial-base-128m"}
    assert adapter.unloaded_models == ["sundial-base-128m"]
