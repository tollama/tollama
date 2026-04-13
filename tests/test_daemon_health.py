"""Tests for daemon liveness, readiness, and request correlation."""

from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient

from tollama.core.storage import install_from_registry
from tollama.daemon.app import create_app


def _sample_forecast_payload() -> dict[str, Any]:
    return {
        "model": "mock",
        "horizon": 2,
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


def test_health_live_alias_returns_ok() -> None:
    with TestClient(create_app()) as client:
        response = client.get("/health/live")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_health_ready_returns_structured_summary(tmp_tollama_home) -> None:
    with TestClient(create_app()) as client:
        response = client.get("/health/ready")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["healthy"] is True
    assert payload["runner_manager"]["healthy"] is True
    assert payload["disk"]["healthy"] is True
    assert payload["connectors"] is None


def test_health_ready_includes_live_connector_summary_when_enabled(
    monkeypatch,
    tmp_tollama_home,
) -> None:
    monkeypatch.setenv("TOLLAMA_USE_LIVE_CONNECTORS", "true")

    class _HealthyConnector:
        connector_name = "news-agent"
        domain = "news"

        def health_check(self) -> dict[str, Any]:
            return {"status": "available", "latency_ms": 12.5}

    class _Registry:
        connectors = [_HealthyConnector()]

    monkeypatch.setattr(
        "tollama.xai.connectors.helpers.build_default_connector_registry",
        lambda: _Registry(),
    )

    with TestClient(create_app()) as client:
        response = client.get("/health/ready")

    assert response.status_code == 200
    payload = response.json()
    assert payload["healthy"] is True
    assert payload["connectors"]["enabled"] is True
    assert payload["connectors"]["checked"] == 1
    assert payload["connectors"]["available"] == 1


def test_forecast_request_id_propagates_to_runner_manager(
    monkeypatch,
    tmp_tollama_home,
) -> None:
    install_from_registry("mock", accept_license=True, paths=tmp_tollama_home)
    captured: dict[str, Any] = {}

    def _fake_call(*, family, method, params, timeout, request_id=None):  # noqa: ANN001, ANN202
        captured.update(
            {
                "family": family,
                "method": method,
                "request_id": request_id,
            }
        )
        return {
            "model": "mock",
            "forecasts": [
                {
                    "id": "s1",
                    "freq": "D",
                    "start_timestamp": "2025-01-02T00:00:00Z",
                    "mean": [2.0, 2.0],
                }
            ],
            "usage": {"runner": "tollama-mock", "device": "cpu", "peak_memory_mb": 1.0},
            "timing": {"model_load_ms": 0.0, "inference_ms": 1.0, "total_ms": 1.0},
        }

    with TestClient(create_app()) as client:
        monkeypatch.setattr(client.app.state.runner_manager, "call", _fake_call)
        response = client.post(
            "/v1/forecast",
            json=_sample_forecast_payload(),
            headers={"X-Request-ID": "req-health-123"},
        )

    assert response.status_code == 200
    assert captured["family"] == "mock"
    assert captured["method"] == "forecast"
    assert captured["request_id"] == "req-health-123"
