"""SSE event stream endpoint and emission tests."""

from __future__ import annotations

import importlib
import json
import time
from pathlib import Path
from threading import Thread
from typing import Any

from fastapi.testclient import TestClient

from tollama.core.schemas import AnalyzeResponse
from tollama.core.storage import TollamaPaths, install_from_registry
from tollama.daemon.app import create_app

daemon_app_module = importlib.import_module("tollama.daemon.app")


def _sample_forecast_payload() -> dict[str, Any]:
    return {
        "model": "mock",
        "horizon": 2,
        "series": [
            {
                "id": "s1",
                "freq": "D",
                "timestamps": ["2025-01-01", "2025-01-02", "2025-01-03"],
                "target": [1.0, 2.0, 3.0],
            }
        ],
        "options": {},
    }


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
        if line.startswith(":") or line.startswith("retry:"):
            continue
        if line.startswith("event:"):
            event_name = line.split(":", 1)[1].strip()
            continue
        if line.startswith("data:"):
            data_lines.append(line.split(":", 1)[1].strip())
    if data_lines:
        events.append({"event": event_name, "data": json.loads("\n".join(data_lines))})
    return events


def test_api_events_stream_receives_published_events() -> None:
    app = create_app()
    captured: dict[str, Any] = {}

    def _consume_stream() -> None:
        with TestClient(app) as client:
            captured["response"] = client.get("/api/events?max_events=2&heartbeat=0.05")

    worker = Thread(target=_consume_stream, daemon=True)
    worker.start()
    time.sleep(0.1)

    app.state.event_stream.publish(
        key_id="anonymous",
        event="forecast.progress",
        data={"status": "running"},
    )
    app.state.event_stream.publish(
        key_id="anonymous",
        event="forecast.complete",
        data={"status": "done"},
    )

    worker.join(timeout=2.0)
    assert not worker.is_alive()

    response = captured["response"]
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")

    events = _parse_sse_events(response.text)
    names = [item["event"] for item in events]
    assert "forecast.progress" in names
    assert "forecast.complete" in names


def test_forecast_and_analyze_paths_emit_sse_events(monkeypatch, tmp_path: Path) -> None:
    paths = TollamaPaths(base_dir=tmp_path / ".tollama")
    monkeypatch.setenv("TOLLAMA_HOME", str(paths.base_dir))
    install_from_registry("mock", accept_license=True, paths=paths)

    app = create_app()
    events: list[tuple[str, dict[str, Any]]] = []
    original_publish = app.state.event_stream.publish

    def _capture_publish(*, key_id: str | None, event: str, data: dict[str, Any]):
        events.append((event, dict(data)))
        return original_publish(key_id=key_id, event=event, data=data)

    def _fake_analyze_response(_payload) -> AnalyzeResponse:
        return AnalyzeResponse.model_validate(
            {
                "results": [
                    {
                        "id": "s1",
                        "detected_frequency": "D",
                        "seasonality_periods": [],
                        "trend": {"direction": "flat", "slope": 0.0, "r2": 1.0},
                        "anomaly_indices": [1],
                        "stationarity_flag": True,
                        "data_quality_score": 1.0,
                    }
                ]
            },
        )

    monkeypatch.setattr(app.state.event_stream, "publish", _capture_publish)
    monkeypatch.setattr(daemon_app_module, "analyze_series_request", _fake_analyze_response)

    with TestClient(app) as client:
        forecast_response = client.post("/v1/forecast", json=_sample_forecast_payload())
        analyze_response = client.post(
            "/api/analyze",
            json={
                "series": [
                    {
                        "id": "s1",
                        "freq": "D",
                        "timestamps": ["2025-01-01", "2025-01-02", "2025-01-03"],
                        "target": [1.0, 2.0, 3.0],
                    }
                ]
            },
        )

    assert forecast_response.status_code == 200
    assert analyze_response.status_code == 200

    emitted_names = [event for event, _payload in events]
    assert "model.loaded" in emitted_names
    assert "forecast.progress" in emitted_names
    assert "forecast.complete" in emitted_names
    assert "analysis.complete" in emitted_names
    assert "anomaly.detected" in emitted_names
