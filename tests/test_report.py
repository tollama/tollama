"""Tests for report helpers and /api/report endpoint."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from tollama.core.report import run_report_analysis
from tollama.core.schemas import ReportRequest
from tollama.core.storage import TollamaPaths, install_from_registry
from tollama.daemon.app import create_app


def _report_payload() -> dict[str, Any]:
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
        "include_baseline": True,
    }


def _setup_home(monkeypatch, tmp_path: Path) -> TollamaPaths:
    paths = TollamaPaths(base_dir=tmp_path / ".tollama")
    monkeypatch.setenv("TOLLAMA_HOME", str(paths.base_dir))
    return paths


def test_run_report_analysis_returns_analysis_and_recommendation() -> None:
    request = ReportRequest.model_validate(_report_payload())
    insights = run_report_analysis(request)

    assert insights.analysis.results[0].id == "s1"
    assert insights.recommendation["request"]["horizon"] == 2


def test_report_endpoint_returns_composite_payload(monkeypatch, tmp_path: Path) -> None:
    paths = _setup_home(monkeypatch, tmp_path)
    install_from_registry("mock", accept_license=True, paths=paths)

    with TestClient(create_app()) as client:
        response = client.post("/api/report", json=_report_payload())

    assert response.status_code == 200
    body = response.json()
    assert body["analysis"]["results"][0]["id"] == "s1"
    assert body["forecast"]["selection"]["chosen_model"] == "mock"
    assert body["baseline"]["model"] == "mock"


def test_report_endpoint_returns_narrative_when_requested(monkeypatch, tmp_path: Path) -> None:
    paths = _setup_home(monkeypatch, tmp_path)
    install_from_registry("mock", accept_license=True, paths=paths)
    payload = _report_payload()
    payload["response_options"] = {"narrative": True}

    with TestClient(create_app()) as client:
        response = client.post("/api/report", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert isinstance(body.get("narrative"), dict)
    assert body["narrative"]["chosen_model"] == "mock"
