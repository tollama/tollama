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


def test_version_endpoint_returns_string_version() -> None:
    with TestClient(create_app()) as client:
        response = client.get("/api/version")
    assert response.status_code == 200
    body = response.json()
    assert isinstance(body.get("version"), str)


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
    mock_available = next(spec for spec in body["available"] if spec["name"] == "mock")
    assert mock_available["family"] == "mock"
    assert mock_available["installed"] is True
    assert mock_available["license"] == {
        "type": "mit",
        "needs_acceptance": False,
        "accepted": True,
    }

    chronos = next(spec for spec in body["available"] if spec["name"] == "chronos2")
    assert chronos["installed"] is False
    assert chronos["license"]["needs_acceptance"] is True

    assert body["installed"] == [
        {
            "name": "mock",
            "family": "mock",
            "installed": True,
            "license": {
                "type": "mit",
                "needs_acceptance": False,
                "accepted": True,
            },
        }
    ]


def test_models_pull_and_delete_flow(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("TOLLAMA_HOME", str(tmp_path / ".tollama"))

    with TestClient(create_app()) as client:
        pull_response = client.post(
            "/v1/models/pull",
            json={"name": "mock", "accept_license": False},
        )
        assert pull_response.status_code == 200
        pulled = pull_response.json()
        assert pulled["name"] == "mock"

        listed = client.get("/v1/models")
        assert listed.status_code == 200
        assert [item["name"] for item in listed.json()["installed"]] == ["mock"]

        delete_response = client.delete("/v1/models/mock")
        assert delete_response.status_code == 200
        assert delete_response.json() == {"removed": True, "name": "mock"}

        delete_again = client.delete("/v1/models/mock")
        assert delete_again.status_code == 404


def test_models_pull_error_mapping(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("TOLLAMA_HOME", str(tmp_path / ".tollama"))

    with TestClient(create_app()) as client:
        missing = client.post(
            "/v1/models/pull",
            json={"name": "does-not-exist", "accept_license": True},
        )
        assert missing.status_code == 404

        conflict = client.post(
            "/v1/models/pull",
            json={"name": "chronos2", "accept_license": False},
        )
        assert conflict.status_code == 409

        bad_payload = client.post(
            "/v1/models/pull",
            json={"name": "mock", "accept_license": "false"},
        )
        assert bad_payload.status_code == 400


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
