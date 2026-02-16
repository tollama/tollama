"""Unit tests for uni2ts runner protocol behavior."""

from __future__ import annotations

import json

from tollama.runners.uni2ts_runner.errors import DependencyMissingError
from tollama.runners.uni2ts_runner.main import handle_request_line


class _NoopAdapter:
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
            "install with pip install -e \".[dev,runner_uni2ts]\"",
        )


def test_uni2ts_runner_hello_reports_supported_family() -> None:
    response = handle_request_line(
        json.dumps({"id": "req-1", "method": "hello", "params": {}}),
        _NoopAdapter(),
    )
    payload = response.model_dump(mode="json", exclude_none=True)
    assert payload["id"] == "req-1"
    assert payload["result"]["name"] == "tollama-uni2ts"
    assert payload["result"]["supported_families"] == ["uni2ts"]


def test_uni2ts_runner_forecast_returns_dependency_missing_error() -> None:
    response = handle_request_line(
        json.dumps(
            {
                "id": "req-2",
                "method": "forecast",
                "params": {
                    "model": "moirai-2.0-R-small",
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
    assert payload["id"] == "req-2"
    assert payload["error"]["code"] == "DEPENDENCY_MISSING"
    assert "runner_uni2ts" in payload["error"]["message"]
