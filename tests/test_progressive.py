"""Progressive forecast planning and API streaming tests."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from tollama.core.progressive import build_progressive_stages
from tollama.core.schemas import ForecastResponse, SeriesInput
from tollama.core.storage import TollamaPaths, install_from_registry
from tollama.daemon.app import create_app

daemon_app_module = importlib.import_module("tollama.daemon.app")


def _series_payload() -> list[dict[str, Any]]:
    return [
        {
            "id": "s1",
            "freq": "D",
            "timestamps": ["2025-01-01", "2025-01-02", "2025-01-03"],
            "target": [1.0, 2.0, 3.0],
        }
    ]


def _parse_sse_events(payload: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    event_name = "message"
    data_lines: list[str] = []
    for line in payload.splitlines():
        if not line:
            if data_lines:
                events.append({"event": event_name, "data": json.loads("\n".join(data_lines))})
            event_name = "message"
            data_lines = []
            continue
        if line.startswith(":"):
            continue
        if line.startswith("event:"):
            event_name = line.split(":", 1)[1].strip()
            continue
        if line.startswith("data:"):
            data_lines.append(line.split(":", 1)[1].strip())
    if data_lines:
        events.append({"event": event_name, "data": json.loads("\n".join(data_lines))})
    return events


def test_build_progressive_stages_returns_explicit_stage_for_preferred_model() -> None:
    series = [SeriesInput.model_validate(_series_payload()[0])]
    stages = build_progressive_stages(
        series=series,
        horizon=2,
        include_models=["mock", "chronos2"],
        preferred_model="mock",
        family_by_model={"mock": "mock", "chronos2": "torch"},
    )

    assert len(stages) == 1
    assert stages[0].strategy == "explicit"
    assert stages[0].model == "mock"
    assert stages[0].family == "mock"


def test_api_forecast_progressive_streams_selected_and_completed_events(
    monkeypatch,
    tmp_path: Path,
) -> None:
    paths = TollamaPaths(base_dir=tmp_path / ".tollama")
    monkeypatch.setenv("TOLLAMA_HOME", str(paths.base_dir))
    install_from_registry("mock", accept_license=True, paths=paths)
    install_from_registry("chronos2", accept_license=True, paths=paths)

    called_models: list[str] = []

    def _fake_execute_forecast(
        _app,
        *,
        payload,
        request=None,
        extra_exclude=None,
    ) -> ForecastResponse:
        called_models.append(payload.model)
        return ForecastResponse.model_validate(
            {
                "model": payload.model,
                "forecasts": [
                    {
                        "id": payload.series[0].id,
                        "freq": payload.series[0].freq,
                        "start_timestamp": payload.series[0].timestamps[-1],
                        "mean": [float(len(called_models))] * payload.horizon,
                    }
                ],
            },
        )

    monkeypatch.setattr(daemon_app_module, "_execute_forecast", _fake_execute_forecast)

    with TestClient(create_app()) as client:
        response = client.post(
            "/api/forecast/progressive",
            json={
                "horizon": 2,
                "series": _series_payload(),
                "options": {},
            },
        )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")

    events = _parse_sse_events(response.text)
    names = [event["event"] for event in events]
    assert "model.selected" in names
    assert "forecast.progress" in names
    assert "forecast.complete" in names

    selected_models = [
        event["data"]["model"]
        for event in events
        if event["event"] == "model.selected"
    ]
    assert selected_models
    assert called_models == selected_models

    completed = [event["data"] for event in events if event["event"] == "forecast.complete"]
    assert completed
    assert completed[-1]["status"] == "completed"
    assert completed[-1]["final"] is True
