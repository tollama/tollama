"""HTTP API tests for the tollama daemon."""

from __future__ import annotations

import json
from typing import Any

from fastapi.testclient import TestClient

from tollama.core.storage import TollamaPaths, install_from_registry
from tollama.daemon.app import create_app
from tollama.daemon.supervisor import RunnerCallError, RunnerUnavailableError


def _sample_forecast_payload() -> dict[str, Any]:
    return {
        "model": "mock",
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


def _install_model(monkeypatch, tmp_path, name: str = "mock") -> None:
    paths = TollamaPaths(base_dir=tmp_path / ".tollama")
    monkeypatch.setenv("TOLLAMA_HOME", str(paths.base_dir))
    install_from_registry(name, accept_license=True, paths=paths)


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


def test_ollama_tags_and_show_endpoints_from_installed_manifest(monkeypatch, tmp_path) -> None:
    paths = TollamaPaths(base_dir=tmp_path / ".tollama")
    monkeypatch.setenv("TOLLAMA_HOME", str(paths.base_dir))

    manifest = {
        "name": "mock",
        "family": "mock",
        "source": {
            "repo_id": "tollama/mock-runner",
            "revision": "main",
            "entrypoint": "tollama-runner-mock",
        },
        "installed_at": "2026-02-16T00:00:00Z",
        "license": {"type": "mit", "needs_acceptance": False, "accepted": True},
        "size": 1234,
        "digest": "sha256:abc123",
    }
    manifest_path = paths.manifest_path("mock")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with TestClient(create_app()) as client:
        tags_response = client.get("/api/tags")
        show_response = client.post("/api/show", json={"model": "mock"})

    assert tags_response.status_code == 200
    assert tags_response.json() == {
        "models": [
            {
                "name": "mock",
                "model": "mock",
                "modified_at": "2026-02-16T00:00:00Z",
                "size": 1234,
                "digest": "sha256:abc123",
                "details": {
                    "format": "tollama",
                    "family": "mock",
                    "families": ["mock"],
                    "parameter_size": "",
                    "quantization_level": "",
                },
            }
        ]
    }

    assert show_response.status_code == 200
    assert show_response.json() == {
        "name": "mock",
        "model": "mock",
        "family": "mock",
        "source": {
            "repo_id": "tollama/mock-runner",
            "revision": "main",
            "entrypoint": "tollama-runner-mock",
        },
        "license": {"type": "mit", "needs_acceptance": False, "accepted": True},
        "modelfile": "",
        "parameters": "",
    }


def test_ollama_show_returns_404_for_missing_model(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("TOLLAMA_HOME", str(tmp_path / ".tollama"))

    with TestClient(create_app()) as client:
        response = client.post("/api/show", json={"model": "not-installed"})

    assert response.status_code == 404
    assert "is not installed" in response.json()["detail"]


def test_ollama_pull_non_stream_and_delete_flow(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("TOLLAMA_HOME", str(tmp_path / ".tollama"))

    with TestClient(create_app()) as client:
        pulled = client.post("/api/pull", json={"model": "mock", "stream": False})
        assert pulled.status_code == 200
        assert pulled.json()["name"] == "mock"

        tags = client.get("/api/tags")
        assert tags.status_code == 200
        assert [item["name"] for item in tags.json()["models"]] == ["mock"]

        deleted = client.request("DELETE", "/api/delete", json={"model": "mock"})
        assert deleted.status_code == 200
        assert deleted.json() == {"deleted": True, "model": "mock"}

        deleted_again = client.request("DELETE", "/api/delete", json={"model": "mock"})
        assert deleted_again.status_code == 404


def test_ollama_pull_stream_returns_ndjson(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("TOLLAMA_HOME", str(tmp_path / ".tollama"))

    with TestClient(create_app()) as client:
        response = client.post("/api/pull", json={"model": "mock"})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/x-ndjson")
    lines = [line for line in response.text.splitlines() if line.strip()]
    assert len(lines) >= 2
    payloads = [json.loads(line) for line in lines]
    assert payloads[0].get("status") == "pulling model manifest"
    assert payloads[-1].get("done") is True
    assert payloads[-1].get("model") == "mock"


def test_api_ps_omits_model_after_forecast_keep_alive_zero(monkeypatch, tmp_path) -> None:
    _install_model(monkeypatch, tmp_path, "mock")
    payload = _sample_forecast_payload()
    payload["keep_alive"] = 0

    with TestClient(create_app()) as client:
        forecast_response = client.post("/v1/forecast", json=payload)
        ps_response = client.get("/api/ps")

    assert forecast_response.status_code == 200
    assert ps_response.status_code == 200
    assert ps_response.json() == {"models": []}


def test_api_ps_tracks_model_after_forecast_keep_alive_negative(monkeypatch, tmp_path) -> None:
    _install_model(monkeypatch, tmp_path, "mock")
    payload = _sample_forecast_payload()
    payload["keep_alive"] = -1

    with TestClient(create_app()) as client:
        forecast_response = client.post("/v1/forecast", json=payload)
        ps_response = client.get("/api/ps")

    assert forecast_response.status_code == 200
    assert ps_response.status_code == 200
    models = ps_response.json().get("models")
    assert isinstance(models, list)
    assert len(models) == 1

    loaded = models[0]
    assert loaded["name"] == "mock"
    assert loaded["model"] == "mock"
    assert loaded["expires_at"] is None
    assert loaded["size"] == 0
    assert loaded["size_vram"] == 0
    assert loaded["context_length"] == 0
    assert loaded["details"] == {"family": "mock"}


def test_api_forecast_stream_false_returns_forecast_response(monkeypatch, tmp_path) -> None:
    _install_model(monkeypatch, tmp_path, "mock")
    payload = _sample_forecast_payload()
    payload["stream"] = False

    with TestClient(create_app()) as client:
        response = client.post("/api/forecast", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["model"] == "mock"
    assert isinstance(body["forecasts"], list)
    assert len(body["forecasts"]) == 2


def test_api_forecast_stream_true_returns_ndjson_with_done(monkeypatch, tmp_path) -> None:
    _install_model(monkeypatch, tmp_path, "mock")
    with TestClient(create_app()) as client:
        response = client.post("/api/forecast", json=_sample_forecast_payload())

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/x-ndjson")

    lines = [line for line in response.text.splitlines() if line.strip()]
    assert len(lines) >= 2

    payloads = [json.loads(line) for line in lines]
    assert payloads[0] == {"status": "loading model"}
    assert payloads[1] == {"status": "running forecast"}
    assert payloads[-1].get("done") is True
    assert payloads[-1]["response"]["model"] == "mock"


def test_api_forecast_keep_alive_updates_loaded_model_tracker(monkeypatch, tmp_path) -> None:
    _install_model(monkeypatch, tmp_path, "mock")
    payload = _sample_forecast_payload()
    payload["stream"] = False
    payload["keep_alive"] = -1

    with TestClient(create_app()) as client:
        forecast_response = client.post("/api/forecast", json=payload)
        ps_response = client.get("/api/ps")

    assert forecast_response.status_code == 200
    assert ps_response.status_code == 200
    models = ps_response.json().get("models")
    assert isinstance(models, list)
    assert len(models) == 1
    assert models[0]["model"] == "mock"
    assert models[0]["expires_at"] is None


def test_forecast_routes_end_to_end_to_mock_runner(monkeypatch, tmp_path) -> None:
    _install_model(monkeypatch, tmp_path, "mock")
    with TestClient(create_app()) as client:
        response = client.post("/v1/forecast", json=_sample_forecast_payload())

    assert response.status_code == 200
    body = response.json()
    assert body["model"] == "mock"
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


def test_forecast_invalid_payload_returns_400(monkeypatch, tmp_path) -> None:
    _install_model(monkeypatch, tmp_path, "mock")
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
    assert chronos["license"]["needs_acceptance"] is False

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
            json={"name": "timesfm2p5", "accept_license": False},
        )
        assert conflict.status_code == 409

        bad_payload = client.post(
            "/v1/models/pull",
            json={"name": "mock", "accept_license": "false"},
        )
        assert bad_payload.status_code == 400


class _UnavailableRunnerManager:
    def call(
        self,
        family: str,
        method: str,
        params: dict[str, Any],
        timeout: float,
    ) -> dict[str, Any]:
        raise RunnerUnavailableError(f"runner unavailable for {family}:{method}")

    def stop(self, family: str | None = None) -> None:
        return None

    def unload(self, family: str, *, model: str | None = None, timeout: float) -> None:
        return None


class _BadGatewayRunnerManager:
    def call(
        self,
        family: str,
        method: str,
        params: dict[str, Any],
        timeout: float,
    ) -> dict[str, Any]:
        raise RunnerCallError(
            code=-32602,
            message="invalid params",
            data={"family": family, "method": method},
        )

    def stop(self, family: str | None = None) -> None:
        return None

    def unload(self, family: str, *, model: str | None = None, timeout: float) -> None:
        return None


def test_forecast_returns_503_when_runner_unavailable(monkeypatch, tmp_path) -> None:
    _install_model(monkeypatch, tmp_path, "mock")
    app = create_app(runner_manager=_UnavailableRunnerManager())  # type: ignore[arg-type]
    with TestClient(app) as client:
        response = client.post("/v1/forecast", json=_sample_forecast_payload())

    assert response.status_code == 503
    assert "runner unavailable" in response.json()["detail"]


def test_forecast_returns_502_when_runner_returns_error(monkeypatch, tmp_path) -> None:
    _install_model(monkeypatch, tmp_path, "mock")
    app = create_app(runner_manager=_BadGatewayRunnerManager())  # type: ignore[arg-type]
    with TestClient(app) as client:
        response = client.post("/v1/forecast", json=_sample_forecast_payload())

    assert response.status_code == 502
    assert response.json()["detail"] == {
        "code": -32602,
        "message": "invalid params",
        "data": {"family": "mock", "method": "forecast"},
    }
