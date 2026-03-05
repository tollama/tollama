"""Unit tests for TiDE phase-1 placeholder runner."""

from __future__ import annotations

import json

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


def test_tide_runner_hello_reports_supported_family_and_status() -> None:
    response = tide_main.handle_request_line(
        json.dumps({"id": "req-1", "method": "hello", "params": {}}),
    )
    payload = response.model_dump(mode="json", exclude_none=True)
    assert payload["result"]["supported_families"] == ["tide"]
    assert payload["result"]["status"] == "phase1_placeholder"


def test_tide_runner_forecast_returns_dependency_missing_when_darts_absent(monkeypatch) -> None:
    monkeypatch.setattr(tide_main, "_has_tide_dependencies", lambda: False)
    response = tide_main.handle_request_line(
        json.dumps({"id": "req-2", "method": "forecast", "params": _valid_forecast_params()}),
    )
    payload = response.model_dump(mode="json", exclude_none=True)
    assert payload["error"]["code"] == "DEPENDENCY_MISSING"
    assert "runner_tide" in payload["error"]["message"]


def test_tide_runner_forecast_returns_not_implemented_when_darts_present(monkeypatch) -> None:
    monkeypatch.setattr(tide_main, "_has_tide_dependencies", lambda: True)
    response = tide_main.handle_request_line(
        json.dumps({"id": "req-3", "method": "forecast", "params": _valid_forecast_params()}),
    )
    payload = response.model_dump(mode="json", exclude_none=True)
    assert payload["error"]["code"] == "NOT_IMPLEMENTED"
    assert "phase-1 placeholder" in payload["error"]["message"]
