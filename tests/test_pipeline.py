"""Tests for pipeline analysis helpers and /api/pipeline endpoint."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from tollama.core.pipeline import run_pipeline_analysis
from tollama.core.schemas import PipelineRequest
from tollama.core.storage import TollamaPaths, install_from_registry
from tollama.daemon.app import create_app


def _pipeline_payload() -> dict[str, Any]:
    return {
        "horizon": 2,
        "strategy": "auto",
        "series": [
            {
                "id": "s1",
                "freq": "D",
                "timestamps": ["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-04"],
                "target": [1.0, 2.0, 1.5, 2.5],
            }
        ],
        "options": {},
        "pull_if_missing": True,
    }


def _setup_home(monkeypatch, tmp_path: Path) -> TollamaPaths:
    paths = TollamaPaths(base_dir=tmp_path / ".tollama")
    monkeypatch.setenv("TOLLAMA_HOME", str(paths.base_dir))
    return paths


def test_run_pipeline_analysis_returns_analysis_and_recommendation() -> None:
    request = PipelineRequest.model_validate(_pipeline_payload())
    insights = run_pipeline_analysis(request)

    assert insights.analysis.results[0].id == "s1"
    assert insights.recommendation["request"]["horizon"] == 2
    assert isinstance(insights.preferred_model, str)
    assert insights.preferred_model


def test_run_pipeline_analysis_prefers_explicit_model() -> None:
    payload = _pipeline_payload()
    payload["model"] = "mock"
    request = PipelineRequest.model_validate(payload)
    insights = run_pipeline_analysis(request)

    assert insights.preferred_model == "mock"


def test_pipeline_endpoint_returns_full_response(monkeypatch, tmp_path: Path) -> None:
    paths = _setup_home(monkeypatch, tmp_path)
    install_from_registry("mock", accept_license=True, paths=paths)

    with TestClient(create_app()) as client:
        response = client.post("/api/pipeline", json=_pipeline_payload())

    assert response.status_code == 200
    payload = response.json()
    assert payload["analysis"]["results"][0]["id"] == "s1"
    assert payload["recommendation"]["request"]["horizon"] == 2
    assert payload["auto_forecast"]["strategy"] == "auto"
    assert payload["auto_forecast"]["response"]["model"] == "mock"


def test_pipeline_endpoint_pulls_explicit_missing_model(monkeypatch, tmp_path: Path) -> None:
    _setup_home(monkeypatch, tmp_path)
    payload = _pipeline_payload()
    payload["model"] = "mock"
    payload["pull_if_missing"] = True

    with TestClient(create_app()) as client:
        response = client.post("/api/pipeline", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["pulled_model"] == "mock"
    assert body["auto_forecast"]["response"]["model"] == "mock"


def test_pipeline_endpoint_returns_404_when_no_models_and_pull_disabled(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _setup_home(monkeypatch, tmp_path)
    payload = _pipeline_payload()
    payload["pull_if_missing"] = False

    with TestClient(create_app()) as client:
        response = client.post("/api/pipeline", json=payload)

    assert response.status_code == 404
