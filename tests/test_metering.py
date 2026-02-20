"""Tests for daemon usage metering and optional rate limiting."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from tollama.core.config import TollamaConfig, save_config
from tollama.core.storage import TollamaPaths, install_from_registry
from tollama.daemon.app import create_app
from tollama.daemon.auth import derive_key_id


@pytest.fixture(autouse=True)
def _clear_environment(monkeypatch) -> None:
    for key in (
        "TOLLAMA_HOME",
        "TOLLAMA_RATE_LIMIT_PER_MINUTE",
        "TOLLAMA_RATE_LIMIT_BURST",
    ):
        monkeypatch.delenv(key, raising=False)


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


def _setup_home(monkeypatch, tmp_path: Path) -> TollamaPaths:
    paths = TollamaPaths(base_dir=tmp_path / ".tollama")
    monkeypatch.setenv("TOLLAMA_HOME", str(paths.base_dir))
    return paths


def _install_mock(paths: TollamaPaths) -> None:
    install_from_registry("mock", accept_license=True, paths=paths)


def test_usage_endpoint_reports_anonymous_usage(monkeypatch, tmp_path: Path) -> None:
    paths = _setup_home(monkeypatch, tmp_path)
    _install_mock(paths)

    with TestClient(create_app()) as client:
        forecast_response = client.post("/v1/forecast", json=_sample_forecast_payload())
        usage_response = client.get("/api/usage")

    assert forecast_response.status_code == 200
    assert usage_response.status_code == 200
    payload = usage_response.json()

    assert payload["summary"]["keys"] == 1
    assert payload["summary"]["request_count"] == 1
    assert payload["summary"]["series_processed"] == 1
    assert payload["summary"]["total_inference_ms"] >= 0.0
    assert payload["items"] == [
        {
            "key_id": "anonymous",
            "request_count": 1,
            "total_inference_ms": payload["items"][0]["total_inference_ms"],
            "series_processed": 1,
            "updated_at": payload["items"][0]["updated_at"],
        }
    ]


def test_usage_endpoint_is_scoped_to_authenticated_key(monkeypatch, tmp_path: Path) -> None:
    paths = _setup_home(monkeypatch, tmp_path)
    _install_mock(paths)
    save_config(
        paths,
        TollamaConfig.model_validate(
            {
                "auth": {
                    "api_keys": ["alpha-key", "beta-key"],
                },
            },
        ),
    )

    with TestClient(create_app()) as client:
        alpha_headers = {"Authorization": "Bearer alpha-key"}
        beta_headers = {"Authorization": "Bearer beta-key"}

        first = client.post("/v1/forecast", json=_sample_forecast_payload(), headers=alpha_headers)
        second = client.post("/v1/forecast", json=_sample_forecast_payload(), headers=alpha_headers)
        third = client.post("/v1/forecast", json=_sample_forecast_payload(), headers=beta_headers)
        alpha_usage = client.get("/api/usage", headers=alpha_headers)
        beta_usage = client.get("/api/usage", headers=beta_headers)

    assert first.status_code == 200
    assert second.status_code == 200
    assert third.status_code == 200

    alpha_payload = alpha_usage.json()
    beta_payload = beta_usage.json()
    assert alpha_payload["summary"]["request_count"] == 2
    assert alpha_payload["summary"]["series_processed"] == 2
    assert alpha_payload["summary"]["keys"] == 1
    assert alpha_payload["items"][0]["key_id"] == derive_key_id("alpha-key")

    assert beta_payload["summary"]["request_count"] == 1
    assert beta_payload["summary"]["series_processed"] == 1
    assert beta_payload["summary"]["keys"] == 1
    assert beta_payload["items"][0]["key_id"] == derive_key_id("beta-key")


def test_rate_limiter_rejects_excess_requests(monkeypatch, tmp_path: Path) -> None:
    paths = _setup_home(monkeypatch, tmp_path)
    _install_mock(paths)
    monkeypatch.setenv("TOLLAMA_RATE_LIMIT_PER_MINUTE", "1")
    monkeypatch.setenv("TOLLAMA_RATE_LIMIT_BURST", "1")

    with TestClient(create_app()) as client:
        first = client.post("/v1/forecast", json=_sample_forecast_payload())
        second = client.post("/v1/forecast", json=_sample_forecast_payload())

    assert first.status_code == 200
    assert second.status_code == 429
    assert second.json()["detail"] == "rate limit exceeded"
