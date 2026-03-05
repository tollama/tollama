"""Protocol tests for PatchTST placeholder runner behavior."""

from __future__ import annotations

import json

from tollama.runners.patchtst_runner.main import handle_request_line


def _forecast_request() -> str:
    return json.dumps(
        {
            "id": "req-forecast",
            "method": "forecast",
            "params": {
                "model": "patchtst",
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
    )


def test_patchtst_runner_hello_reports_phase1_status() -> None:
    response = handle_request_line(json.dumps({"id": "req-1", "method": "hello", "params": {}}))
    payload = response.model_dump(mode="json", exclude_none=True)

    assert payload["result"]["supported_families"] == ["patchtst"]
    assert payload["result"]["status"] == "phase1_placeholder"


def test_patchtst_runner_forecast_returns_dependency_missing_when_dependency_absent(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "tollama.runners.patchtst_runner.main._missing_dependencies",
        lambda: ["transformers"],
    )

    response = handle_request_line(_forecast_request())
    payload = response.model_dump(mode="json", exclude_none=True)

    assert payload["error"]["code"] == "DEPENDENCY_MISSING"
    assert "runner_patchtst" in payload["error"]["message"]


def test_patchtst_runner_forecast_returns_not_implemented_when_dependencies_present(
    monkeypatch,
) -> None:
    monkeypatch.setattr("tollama.runners.patchtst_runner.main._missing_dependencies", lambda: [])

    response = handle_request_line(_forecast_request())
    payload = response.model_dump(mode="json", exclude_none=True)

    assert payload["error"]["code"] == "NOT_IMPLEMENTED"
    assert "full inference is not implemented yet" in payload["error"]["message"].lower()
