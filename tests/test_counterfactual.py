"""Tests for counterfactual generation helpers and endpoint."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from tollama.core.counterfactual import generate_counterfactual
from tollama.core.schemas import CounterfactualRequest, ForecastResponse
from tollama.core.storage import TollamaPaths, install_from_registry
from tollama.daemon.app import create_app


def _counterfactual_payload() -> dict[str, object]:
    return {
        "model": "mock",
        "intervention_index": 3,
        "series": [
            {
                "id": "s1",
                "freq": "D",
                "timestamps": [
                    "2025-01-01",
                    "2025-01-02",
                    "2025-01-03",
                    "2025-01-04",
                    "2025-01-05",
                ],
                "target": [10.0, 11.0, 12.0, 20.0, 22.0],
            }
        ],
        "options": {},
    }


def _setup_home(monkeypatch, tmp_path: Path) -> TollamaPaths:
    paths = TollamaPaths(base_dir=tmp_path / ".tollama")
    monkeypatch.setenv("TOLLAMA_HOME", str(paths.base_dir))
    return paths


def test_generate_counterfactual_core_with_stub_forecast() -> None:
    request = CounterfactualRequest.model_validate(_counterfactual_payload())

    def _stub_forecast(_request) -> ForecastResponse:  # type: ignore[no-untyped-def]
        return ForecastResponse.model_validate(
            {
                "model": "mock",
                "forecasts": [
                    {
                        "id": "s1",
                        "freq": "D",
                        "start_timestamp": "2025-01-04",
                        "mean": [12.0, 13.0],
                    }
                ],
            }
        )

    response = generate_counterfactual(payload=request, forecast_executor=_stub_forecast)

    assert response.horizon == 2
    assert response.results[0].actual == [20.0, 22.0]
    assert response.results[0].counterfactual == [12.0, 13.0]
    assert response.results[0].direction == "above_counterfactual"


def test_counterfactual_endpoint_returns_payload(monkeypatch, tmp_path: Path) -> None:
    paths = _setup_home(monkeypatch, tmp_path)
    install_from_registry("mock", accept_license=True, paths=paths)

    with TestClient(create_app()) as client:
        response = client.post("/api/counterfactual", json=_counterfactual_payload())

    assert response.status_code == 200
    body = response.json()
    assert body["model"] == "mock"
    assert body["horizon"] == 2
    assert body["results"][0]["id"] == "s1"
    assert len(body["results"][0]["counterfactual"]) == 2
