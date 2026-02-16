"""HTTP API tests for the tollama daemon."""

from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient

from tollama.core.storage import TollamaPaths, install_from_registry
from tollama.daemon.app import create_app
from tollama.daemon.supervisor import RunnerCallError, RunnerUnavailableError


def _sample_forecast_payload() -> dict[str, Any]:
    return {
        "model": "mock-naive",
        "horizon": 2,
        "quantiles": [0.1, 0.9],
        "series": [
            {
                "id": "s1",
                "freq": "D",
                "timestamps": ["2025-01-01", "2025-01-02"],
                "target": [1.0, 3.0],
            },
            {
                "id": "s2",
                "freq": "D",
                "timestamps": ["2025-01-01", "2025-01-02"],
                "target": [2.0, 4.0],
            },
        ],
        "options": {},
    }


def test_health_endpoint_returns_ok() -> None:
    with TestClient(create_app()) as client:
        response = client.get("/v1/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_forecast_routes_end_to_end_to_mock_runner() -> None:
    with TestClient(create_app()) as client:
        response = client.post("/v1/forecast", json=_sample_forecast_payload())

    assert response.status_code == 200
    body = response.json()
    assert body["model"] == "mock-naive"
    assert len(body["forecasts"]) == 2

    first = body["forecasts"][0]
    assert first["id"] == "s1"
    assert first["mean"] == [3.0, 3.0]
    assert first["quantiles"] == {"0.1": [3.0, 3.0], "0.9": [3.0, 3.0]}
    assert first["start_timestamp"] == "2025-01-02"

    second = body["forecasts"][1]
    assert second["id"] == "s2"
    assert second["mean"] == [4.0, 4.0]
    assert second["quantiles"] == {"0.1": [4.0, 4.0], "0.9": [4.0, 4.0]}
    assert second["start_timestamp"] == "2025-01-02"


def test_forecast_invalid_payload_returns_400() -> None:
    payload = _sample_forecast_payload()
    payload["horizon"] = "2"

    with TestClient(create_app()) as client:
        response = client.post("/v1/forecast", json=payload)

    assert response.status_code == 400
    assert "detail" in response.json()


def test_models_endpoint_returns_available_and_installed(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("TOLLAMA_HOME", str(tmp_path / ".tollama"))
    install_from_registry(
        "mock",
        accept_license=False,
        paths=TollamaPaths(base_dir=tmp_path / ".tollama"),
    )

    with TestClient(create_app()) as client:
        response = client.get("/v1/models")

    assert response.status_code == 200
    body = response.json()
    assert any(spec["name"] == "mock" for spec in body["available"])
    assert [item["name"] for item in body["installed"]] == ["mock"]


class _UnavailableSupervisor:
    def call(self, method: str, params: dict[str, Any], timeout: float) -> dict[str, Any]:
        raise RunnerUnavailableError(f"runner unavailable for {method}")

    def stop(self) -> None:
        return None


class _BadGatewaySupervisor:
    def call(self, method: str, params: dict[str, Any], timeout: float) -> dict[str, Any]:
        raise RunnerCallError(code=-32602, message="invalid params", data={"method": method})

    def stop(self) -> None:
        return None


def test_forecast_returns_503_when_runner_unavailable() -> None:
    app = create_app(supervisor=_UnavailableSupervisor())  # type: ignore[arg-type]
    with TestClient(app) as client:
        response = client.post("/v1/forecast", json=_sample_forecast_payload())

    assert response.status_code == 503
    assert "runner unavailable" in response.json()["detail"]


def test_forecast_returns_502_when_runner_returns_error() -> None:
    app = create_app(supervisor=_BadGatewaySupervisor())  # type: ignore[arg-type]
    with TestClient(app) as client:
        response = client.post("/v1/forecast", json=_sample_forecast_payload())

    assert response.status_code == 502
    assert response.json()["detail"] == {
        "code": -32602,
        "message": "invalid params",
        "data": {"method": "forecast"},
    }
