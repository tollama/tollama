"""Tests for auto-forecast selection and daemon endpoint behavior."""

from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient

from tollama.core.auto_select import select_auto_models
from tollama.core.schemas import SeriesInput
from tollama.core.storage import TollamaPaths, install_from_registry
from tollama.daemon.app import create_app
from tollama.daemon.runner_manager import RunnerManager


def _series_payload() -> list[dict[str, Any]]:
    return [
        {
            "id": "s1",
            "freq": "D",
            "timestamps": ["2025-01-01", "2025-01-02", "2025-01-03"],
            "target": [1.0, 2.0, 3.0],
        }
    ]


def _auto_payload() -> dict[str, Any]:
    return {
        "horizon": 2,
        "series": _series_payload(),
        "options": {},
    }


def test_select_auto_models_ranks_candidates_for_fastest_strategy() -> None:
    series = [SeriesInput.model_validate(_series_payload()[0])]
    selection = select_auto_models(
        series=series,
        horizon=2,
        strategy="fastest",
        include_models=["mock", "chronos2"],
    )

    assert selection.ranked_candidates
    assert selection.ranked_candidates[0].model == "mock"
    assert selection.selected_models == ("mock",)


def test_daemon_auto_forecast_selects_installed_model(monkeypatch, tmp_path) -> None:
    paths = TollamaPaths(base_dir=tmp_path / ".tollama")
    monkeypatch.setenv("TOLLAMA_HOME", str(paths.base_dir))
    install_from_registry("mock", accept_license=True, paths=paths)

    with TestClient(create_app()) as client:
        response = client.post("/api/auto-forecast", json=_auto_payload())

    assert response.status_code == 200
    body = response.json()
    assert body["selection"]["chosen_model"] == "mock"
    assert body["selection"]["fallback_used"] is False
    assert body["response"]["model"] == "mock"


def test_daemon_auto_forecast_returns_404_for_missing_explicit_model(monkeypatch, tmp_path) -> None:
    paths = TollamaPaths(base_dir=tmp_path / ".tollama")
    monkeypatch.setenv("TOLLAMA_HOME", str(paths.base_dir))
    install_from_registry("mock", accept_license=True, paths=paths)

    payload = _auto_payload()
    payload["model"] = "chronos2"

    with TestClient(create_app()) as client:
        response = client.post("/api/auto-forecast", json=payload)

    assert response.status_code == 404
    assert "not installed" in response.json()["detail"]


def test_daemon_auto_forecast_falls_back_for_missing_explicit_model(monkeypatch, tmp_path) -> None:
    paths = TollamaPaths(base_dir=tmp_path / ".tollama")
    monkeypatch.setenv("TOLLAMA_HOME", str(paths.base_dir))
    install_from_registry("mock", accept_license=True, paths=paths)

    payload = _auto_payload()
    payload["model"] = "chronos2"
    payload["allow_fallback"] = True

    with TestClient(create_app()) as client:
        response = client.post("/api/auto-forecast", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["selection"]["fallback_used"] is True
    assert body["selection"]["chosen_model"] == "mock"
    assert body["response"]["model"] == "mock"


def test_daemon_auto_forecast_falls_back_when_explicit_model_runner_fails(
    monkeypatch,
    tmp_path,
) -> None:
    paths = TollamaPaths(base_dir=tmp_path / ".tollama")
    monkeypatch.setenv("TOLLAMA_HOME", str(paths.base_dir))
    install_from_registry("mock", accept_license=True, paths=paths)
    install_from_registry("chronos2", accept_license=True, paths=paths)

    manager = RunnerManager(
        runner_commands={
            "mock": ("tollama-runner-mock",),
            "torch": ("tollama-runner-does-not-exist",),
        },
    )

    payload = _auto_payload()
    payload["model"] = "chronos2"
    payload["allow_fallback"] = True

    with TestClient(create_app(runner_manager=manager)) as client:
        response = client.post("/api/auto-forecast", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["selection"]["fallback_used"] is True
    assert body["selection"]["chosen_model"] == "mock"
    assert body["response"]["model"] == "mock"


def test_daemon_auto_forecast_ensemble_returns_single_when_only_one_model_succeeds(
    monkeypatch,
    tmp_path,
) -> None:
    paths = TollamaPaths(base_dir=tmp_path / ".tollama")
    monkeypatch.setenv("TOLLAMA_HOME", str(paths.base_dir))
    install_from_registry("mock", accept_license=True, paths=paths)
    install_from_registry("chronos2", accept_license=True, paths=paths)

    payload = _auto_payload()
    payload["strategy"] = "ensemble"
    payload["ensemble_top_k"] = 2
    payload["ensemble_method"] = "median"
    manager = RunnerManager(
        runner_commands={
            "mock": ("tollama-runner-mock",),
            "torch": ("tollama-runner-does-not-exist",),
        },
    )

    with TestClient(create_app(runner_manager=manager)) as client:
        response = client.post("/api/auto-forecast", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["selection"]["strategy"] == "ensemble"
    assert body["response"]["model"] == "mock"
    warnings = body["response"].get("warnings") or []
    assert any("only one model succeeded" in warning for warning in warnings)
