"""Tests for family-based runner management and daemon routing."""

from __future__ import annotations

import sys

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


def _timesfm_payload() -> dict[str, object]:
    return {
        "model": "timesfm-2.5-200m",
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


def _uni2ts_payload() -> dict[str, object]:
    return {
        "model": "moirai-2.0-R-small",
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


def test_daemon_routes_timesfm_family_to_runner_command_override(monkeypatch, tmp_path) -> None:
    paths = TollamaPaths(base_dir=tmp_path / ".tollama")
    monkeypatch.setenv("TOLLAMA_HOME", str(paths.base_dir))
    install_from_registry("timesfm-2.5-200m", accept_license=False, paths=paths)

    manager = RunnerManager(
        runner_commands={"timesfm": ("tollama-runner-mock",)},
    )
    with TestClient(create_app(runner_manager=manager)) as client:
        response = client.post("/v1/forecast", json=_timesfm_payload())

    assert response.status_code == 200
    body = response.json()
    assert body["model"] == "timesfm-2.5-200m"
    assert body["forecasts"][0]["id"] == "s1"
    assert body["forecasts"][0]["mean"] == [12.0, 12.0]


def test_daemon_routes_uni2ts_family_to_runner_command_override(monkeypatch, tmp_path) -> None:
    paths = TollamaPaths(base_dir=tmp_path / ".tollama")
    monkeypatch.setenv("TOLLAMA_HOME", str(paths.base_dir))
    install_from_registry("moirai-2.0-R-small", accept_license=True, paths=paths)

    manager = RunnerManager(
        runner_commands={"uni2ts": ("tollama-runner-mock",)},
    )
    with TestClient(create_app(runner_manager=manager)) as client:
        response = client.post("/v1/forecast", json=_uni2ts_payload())

    assert response.status_code == 200
    body = response.json()
    assert body["model"] == "moirai-2.0-R-small"
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


def test_missing_timesfm_runner_command_returns_install_hint(monkeypatch, tmp_path) -> None:
    paths = TollamaPaths(base_dir=tmp_path / ".tollama")
    monkeypatch.setenv("TOLLAMA_HOME", str(paths.base_dir))
    install_from_registry("timesfm-2.5-200m", accept_license=False, paths=paths)

    manager = RunnerManager(
        runner_commands={"timesfm": ("tollama-runner-timesfm-missing",)},
    )
    with TestClient(create_app(runner_manager=manager)) as client:
        response = client.post("/v1/forecast", json=_timesfm_payload())

    assert response.status_code == 503
    detail = response.json()["detail"]
    assert "runner_timesfm" in detail
    assert "pip install -e" in detail


def test_missing_uni2ts_runner_command_returns_install_hint(monkeypatch, tmp_path) -> None:
    paths = TollamaPaths(base_dir=tmp_path / ".tollama")
    monkeypatch.setenv("TOLLAMA_HOME", str(paths.base_dir))
    install_from_registry("moirai-2.0-R-small", accept_license=True, paths=paths)

    manager = RunnerManager(
        runner_commands={"uni2ts": ("tollama-runner-uni2ts-missing",)},
    )
    with TestClient(create_app(runner_manager=manager)) as client:
        response = client.post("/v1/forecast", json=_uni2ts_payload())

    assert response.status_code == 503
    detail = response.json()["detail"]
    assert "runner_uni2ts" in detail
    assert "pip install -e" in detail


def test_runner_manager_list_families_includes_expected_defaults() -> None:
    manager = RunnerManager()
    assert manager.list_families() == ["mock", "torch", "timesfm", "uni2ts"]


def test_runner_manager_get_all_statuses_does_not_start_missing_supervisors(monkeypatch) -> None:
    class _FakeSupervisor:
        def __init__(self) -> None:
            self.calls = 0

        def get_status(self, *, family: str) -> dict[str, object]:
            self.calls += 1
            return {
                "family": family,
                "command": ["tollama-runner-mock"],
                "installed": True,
                "running": True,
                "pid": 1234,
                "started_at": "2026-02-16T00:00:00Z",
                "last_used_at": "2026-02-16T00:00:01Z",
                "restarts": 0,
                "last_error": None,
            }

        def stop(self) -> None:
            return

    fake = _FakeSupervisor()
    manager = RunnerManager(supervisors={"mock": fake})
    monkeypatch.setattr(
        "tollama.daemon.runner_manager.shutil.which",
        lambda command: "/usr/bin/python3" if command == sys.executable else None,
    )

    statuses = manager.get_all_statuses()
    by_family = {item["family"]: item for item in statuses}

    assert fake.calls == 1
    assert set(by_family) == {"mock", "torch", "timesfm", "uni2ts"}
    assert by_family["mock"]["running"] is True
    assert by_family["torch"]["command"] == [
        sys.executable,
        "-m",
        "tollama.runners.torch_runner.main",
    ]
    assert by_family["torch"]["installed"] is True
    assert by_family["timesfm"]["running"] is False
    assert by_family["uni2ts"]["running"] is False
