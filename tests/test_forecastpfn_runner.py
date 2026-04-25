"""Unit tests for ForecastPFN runner protocol behavior."""

from __future__ import annotations

import json
from typing import Any

from tollama.runners.forecastpfn_runner.adapter import ForecastPFNAdapter
from tollama.runners.forecastpfn_runner.main import handle_request_line


def _valid_forecast_params() -> dict[str, Any]:
    return {
        "model": "forecastpfn",
        "horizon": 2,
        "quantiles": [],
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


def test_forecastpfn_runner_reports_manifest_only_model_unsupported() -> None:
    response = handle_request_line(
        json.dumps(
            {
                "id": "req-forecastpfn",
                "method": "forecast",
                "params": _valid_forecast_params(),
            },
        ),
        ForecastPFNAdapter(),
    )

    payload = response.model_dump(mode="json", exclude_none=True)
    assert payload["id"] == "req-forecastpfn"
    assert payload["error"]["code"] == "MODEL_UNSUPPORTED"
    assert "manifest-only" in payload["error"]["message"]
    assert "ForecastPFN" in payload["error"]["message"]
