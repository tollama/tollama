"""Tests for family-based runner management and daemon routing."""

from __future__ import annotations

from fastapi.testclient import TestClient

from tollama.core.storage import TollamaPaths, install_from_registry
from tollama.daemon.app import create_app
from tollama.daemon.runner_manager import RunnerManager


def _chronos_payload() -> dict[str, object]:
    return {
        "model": "chronos2",
        "horizon": 2,
        "quantiles": [0.1, 0.9],
        "series": [
            {
                "id": "s1",
                "freq": "D",
                "timestamps": ["2025-01-01", "2025-01-02"],
                "target": [10.0, 12.0],
            }
        ],
        "options": {},
    }


def test_daemon_routes_torch_family_to_torch_runner_command_override(monkeypatch, tmp_path) -> None:
    paths = TollamaPaths(base_dir=tmp_path / ".tollama")
    monkeypatch.setenv("TOLLAMA_HOME", str(paths.base_dir))
    install_from_registry("chronos2", accept_license=True, paths=paths)

    manager = RunnerManager(
        runner_commands={"torch": ("tollama-runner-mock",)},
    )
    with TestClient(create_app(runner_manager=manager)) as client:
        response = client.post("/v1/forecast", json=_chronos_payload())

    assert response.status_code == 200
    body = response.json()
    assert body["model"] == "chronos2"
    assert body["forecasts"][0]["id"] == "s1"
    assert body["forecasts"][0]["mean"] == [12.0, 12.0]


def test_missing_torch_runner_command_returns_install_hint(monkeypatch, tmp_path) -> None:
    paths = TollamaPaths(base_dir=tmp_path / ".tollama")
    monkeypatch.setenv("TOLLAMA_HOME", str(paths.base_dir))
    install_from_registry("chronos2", accept_license=True, paths=paths)

    manager = RunnerManager(
        runner_commands={"torch": ("tollama-runner-does-not-exist",)},
    )
    with TestClient(create_app(runner_manager=manager)) as client:
        response = client.post("/v1/forecast", json=_chronos_payload())

    assert response.status_code == 503
    detail = response.json()["detail"]
    assert "runner_torch" in detail
    assert "pip install -e" in detail
