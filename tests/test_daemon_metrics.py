"""Tests for daemon Prometheus metrics instrumentation."""

from __future__ import annotations

import importlib
import re
from typing import Any

import pytest
from fastapi.testclient import TestClient

from tollama.core.storage import TollamaPaths, install_from_registry
from tollama.daemon import metrics as daemon_metrics
from tollama.daemon.app import create_app

daemon_app_module = importlib.import_module("tollama.daemon.app")


@pytest.fixture(autouse=True)
def _clear_environment(monkeypatch) -> None:
    for key in (
        "TOLLAMA_HOME",
        "TOLLAMA_FORECAST_TIMEOUT_SECONDS",
    ):
        monkeypatch.delenv(key, raising=False)


def _install_mock_model(monkeypatch, tmp_path) -> None:
    paths = TollamaPaths(base_dir=tmp_path / ".tollama")
    monkeypatch.setenv("TOLLAMA_HOME", str(paths.base_dir))
    install_from_registry("mock", accept_license=True, paths=paths)


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


def _metric_value(metrics_text: str, *, metric: str, labels: str | None = None) -> float:
    pattern = rf"^{re.escape(metric)}"
    if labels is not None:
        pattern += rf"\{{{labels}\}}"
    pattern += r"\s+([0-9eE+.-]+)$"
    match = re.search(pattern, metrics_text, flags=re.MULTILINE)
    assert match is not None, f"metric sample not found: {metric} labels={labels}"
    return float(match.group(1))


def test_metrics_endpoint_returns_503_when_prometheus_is_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(daemon_app_module, "create_prometheus_metrics", lambda **_: None)

    with TestClient(create_app()) as client:
        response = client.get("/metrics")

    assert response.status_code == 503
    assert "tollama[metrics]" in response.json()["detail"]


@pytest.mark.skipif(
    not daemon_metrics.metrics_available(),
    reason="prometheus-client is not installed",
)
def test_metrics_endpoint_exposes_prometheus_payload(monkeypatch, tmp_path) -> None:
    _install_mock_model(monkeypatch, tmp_path)

    with TestClient(create_app()) as client:
        response = client.get("/metrics")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    payload = response.text
    assert "tollama_forecast_requests_total" in payload
    assert "tollama_forecast_latency_seconds" in payload
    assert "tollama_models_loaded" in payload
    assert "tollama_runner_restarts_total" in payload


@pytest.mark.skipif(
    not daemon_metrics.metrics_available(),
    reason="prometheus-client is not installed",
)
def test_metrics_track_successful_forecast_and_loaded_models(monkeypatch, tmp_path) -> None:
    _install_mock_model(monkeypatch, tmp_path)
    payload = _sample_forecast_payload()
    payload["keep_alive"] = -1

    with TestClient(create_app()) as client:
        forecast_response = client.post("/v1/forecast", json=payload)
        metrics_response = client.get("/metrics")

    assert forecast_response.status_code == 200
    assert metrics_response.status_code == 200
    metrics_text = metrics_response.text

    success_count = _metric_value(
        metrics_text,
        metric="tollama_forecast_requests_total",
        labels='endpoint="v1_forecast",status_class="2xx"',
    )
    latency_count = _metric_value(
        metrics_text,
        metric="tollama_forecast_latency_seconds_count",
        labels='endpoint="v1_forecast",status_class="2xx"',
    )
    models_loaded = _metric_value(metrics_text, metric="tollama_models_loaded")

    assert success_count >= 1.0
    assert latency_count >= 1.0
    assert models_loaded >= 1.0


@pytest.mark.skipif(
    not daemon_metrics.metrics_available(),
    reason="prometheus-client is not installed",
)
def test_metrics_track_forecast_failures_and_runner_restart_totals(monkeypatch, tmp_path) -> None:
    paths = TollamaPaths(base_dir=tmp_path / ".tollama")
    monkeypatch.setenv("TOLLAMA_HOME", str(paths.base_dir))

    restart_total = {"value": 5}

    with TestClient(create_app()) as client:
        monkeypatch.setattr(
            client.app.state.runner_manager,
            "get_all_statuses",
            lambda: [{"restarts": restart_total["value"]}],
        )

        failed_forecast = client.post("/v1/forecast", json=_sample_forecast_payload())
        first_metrics = client.get("/metrics")
        restart_total["value"] = 7
        second_metrics = client.get("/metrics")

    assert failed_forecast.status_code == 404
    assert first_metrics.status_code == 200
    assert second_metrics.status_code == 200

    failure_count = _metric_value(
        second_metrics.text,
        metric="tollama_forecast_requests_total",
        labels='endpoint="v1_forecast",status_class="4xx"',
    )
    first_restarts = _metric_value(
        first_metrics.text,
        metric="tollama_runner_restarts_total",
    )
    second_restarts = _metric_value(
        second_metrics.text,
        metric="tollama_runner_restarts_total",
    )

    assert failure_count >= 1.0
    assert first_restarts == 5.0
    assert second_restarts == 7.0
