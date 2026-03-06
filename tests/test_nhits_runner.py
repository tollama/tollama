"""Unit tests for N-HiTS runner placeholder behavior."""

from __future__ import annotations

import json

from tollama.runners.nhits_runner.main import handle_request_line


class _NoopAdapter:
    def __init__(self) -> None:
        self.unloaded_models: list[str | None] = []

    def unload(self, model_name: str | None = None) -> None:
        self.unloaded_models.append(model_name)

    def forecast(self, request, **kwargs):
        del request, kwargs
        raise AssertionError("unexpected call")


class _DependencyMissingAdapter(_NoopAdapter):
    def forecast(self, request, **kwargs):
        del request, kwargs
        from tollama.runners.nhits_runner.errors import DependencyMissingError

        raise DependencyMissingError(
            "install with python -m pip install -e \".[dev,runner_nhits]\"",
        )


class _NotImplementedAdapter(_NoopAdapter):
    def forecast(self, request, **kwargs):
        del request, kwargs
        from tollama.runners.nhits_runner.errors import NotImplementedRunnerError

        raise NotImplementedRunnerError("phase-1 placeholder only")


def _forecast_request() -> dict[str, object]:
    return {
        "id": "req-2",
        "method": "forecast",
        "params": {
            "model": "nhits",
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
    }


def test_nhits_runner_hello_reports_supported_family_and_status() -> None:
    response = handle_request_line(
        json.dumps({"id": "req-1", "method": "hello", "params": {}}),
        _NoopAdapter(),
    )
    payload = response.model_dump(mode="json", exclude_none=True)
    assert payload["result"]["supported_families"] == ["nhits"]
    assert payload["result"]["status"] == "phase1_placeholder"


def test_nhits_runner_forecast_returns_dependency_missing_error() -> None:
    response = handle_request_line(
        json.dumps(_forecast_request()),
        _DependencyMissingAdapter(),
    )
    payload = response.model_dump(mode="json", exclude_none=True)
    assert payload["error"]["code"] == "DEPENDENCY_MISSING"
    assert "runner_nhits" in payload["error"]["message"]


def test_nhits_runner_forecast_returns_not_implemented_error() -> None:
    response = handle_request_line(
        json.dumps(_forecast_request()),
        _NotImplementedAdapter(),
    )
    payload = response.model_dump(mode="json", exclude_none=True)
    assert payload["error"]["code"] == "NOT_IMPLEMENTED"
    assert "placeholder" in payload["error"]["message"]


def test_nhits_runner_unload_calls_adapter() -> None:
    adapter = _NoopAdapter()
    response = handle_request_line(
        json.dumps({"id": "req-4", "method": "unload", "params": {"model": "nhits"}}),
        adapter,
    )
    payload = response.model_dump(mode="json", exclude_none=True)
    assert payload["result"] == {"unloaded": True, "model": "nhits"}
    assert adapter.unloaded_models == ["nhits"]
