"""Tests for scenario-tree generation helpers and endpoint."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from tollama.core.scenario_tree import build_scenario_tree
from tollama.core.schemas import ForecastResponse, ScenarioTreeRequest
from tollama.core.storage import TollamaPaths, install_from_registry
from tollama.daemon.app import create_app


def _scenario_tree_payload() -> dict[str, object]:
    return {
        "model": "mock",
        "horizon": 4,
        "depth": 2,
        "branch_quantiles": [0.1, 0.9],
        "series": [
            {
                "id": "s1",
                "freq": "D",
                "timestamps": ["2025-01-01", "2025-01-02", "2025-01-03"],
                "target": [10.0, 11.0, 12.0],
            }
        ],
        "options": {},
    }


def _setup_home(monkeypatch, tmp_path: Path) -> TollamaPaths:
    paths = TollamaPaths(base_dir=tmp_path / ".tollama")
    monkeypatch.setenv("TOLLAMA_HOME", str(paths.base_dir))
    return paths


def test_build_scenario_tree_with_stub_forecast() -> None:
    request = ScenarioTreeRequest.model_validate(_scenario_tree_payload())

    def _stub_forecast(_request) -> ForecastResponse:  # type: ignore[no-untyped-def]
        return ForecastResponse.model_validate(
            {
                "model": "mock",
                "forecasts": [
                    {
                        "id": "s1",
                        "freq": "D",
                        "start_timestamp": "2025-01-04",
                        "mean": [12.0],
                        "quantiles": {
                            "0.1": [11.0],
                            "0.9": [13.0],
                        },
                    }
                ],
            }
        )

    response = build_scenario_tree(payload=request, forecast_executor=_stub_forecast)

    assert response.depth == 2
    assert response.nodes[0].branch == "root"
    assert len(response.nodes) == 7  # root + 2 first-level + 4 second-level


def test_scenario_tree_endpoint_returns_nodes(monkeypatch, tmp_path: Path) -> None:
    paths = _setup_home(monkeypatch, tmp_path)
    install_from_registry("mock", accept_license=True, paths=paths)

    with TestClient(create_app()) as client:
        response = client.post("/api/scenario-tree", json=_scenario_tree_payload())

    assert response.status_code == 200
    body = response.json()
    assert body["model"] == "mock"
    assert body["depth"] == 2
    assert body["nodes"][0]["branch"] == "root"
    assert len(body["nodes"]) >= 3
