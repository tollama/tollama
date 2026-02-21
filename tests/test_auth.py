"""Tests for daemon API-key authentication behavior."""

from __future__ import annotations

from pathlib import Path

from fastapi import Request
from fastapi.testclient import TestClient

from tollama.core.config import TollamaConfig, save_config
from tollama.core.storage import TollamaPaths
from tollama.daemon.app import create_app
from tollama.daemon.auth import current_key_id, derive_key_id


def _paths(monkeypatch, tmp_path: Path) -> TollamaPaths:
    paths = TollamaPaths(base_dir=tmp_path / ".tollama")
    monkeypatch.setenv("TOLLAMA_HOME", str(paths.base_dir))
    return paths


def _save_auth_config(paths: TollamaPaths, api_keys: list[str]) -> None:
    save_config(
        paths,
        TollamaConfig.model_validate({"auth": {"api_keys": api_keys}}),
    )


def test_auth_is_disabled_by_default(monkeypatch, tmp_path: Path) -> None:
    _paths(monkeypatch, tmp_path)

    with TestClient(create_app()) as client:
        response = client.get("/v1/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_auth_requires_bearer_token_when_api_keys_are_configured(
    monkeypatch,
    tmp_path: Path,
) -> None:
    paths = _paths(monkeypatch, tmp_path)
    _save_auth_config(paths, ["secret-key"])

    with TestClient(create_app()) as client:
        response = client.get("/v1/health")

    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"
    assert response.json()["detail"] == "missing bearer token"


def test_auth_rejects_invalid_scheme_and_invalid_api_key(monkeypatch, tmp_path: Path) -> None:
    paths = _paths(monkeypatch, tmp_path)
    _save_auth_config(paths, ["secret-key"])

    with TestClient(create_app()) as client:
        wrong_scheme = client.get(
            "/v1/health",
            headers={"Authorization": "Basic secret-key"},
        )
        wrong_key = client.get(
            "/v1/health",
            headers={"Authorization": "Bearer not-secret"},
        )

    assert wrong_scheme.status_code == 401
    assert wrong_scheme.json()["detail"] == "authorization scheme must be Bearer"
    assert wrong_key.status_code == 401
    assert wrong_key.json()["detail"] == "invalid api key"


def test_auth_accepts_valid_api_key_and_exposes_key_id(monkeypatch, tmp_path: Path) -> None:
    paths = _paths(monkeypatch, tmp_path)
    _save_auth_config(paths, ["secret-key"])
    app = create_app()

    @app.get("/__auth_probe")
    def _auth_probe(request: Request) -> dict[str, str]:
        return {"key_id": current_key_id(request)}

    with TestClient(app) as client:
        response = client.get("/__auth_probe", headers={"Authorization": "Bearer secret-key"})

    assert response.status_code == 200
    assert response.json() == {"key_id": derive_key_id("secret-key")}


def test_auth_ignores_blank_configured_keys(monkeypatch, tmp_path: Path) -> None:
    paths = _paths(monkeypatch, tmp_path)
    _save_auth_config(paths, ["   "])

    with TestClient(create_app()) as client:
        response = client.get("/v1/health")

    assert response.status_code == 200


def test_docs_require_auth_when_api_keys_are_configured(monkeypatch, tmp_path: Path) -> None:
    paths = _paths(monkeypatch, tmp_path)
    _save_auth_config(paths, ["secret-key"])

    with TestClient(create_app()) as client:
        docs = client.get("/docs")
        redoc = client.get("/redoc")
        openapi = client.get("/openapi.json")

    assert docs.status_code == 401
    assert redoc.status_code == 401
    assert openapi.status_code == 401


def test_docs_allow_bearer_auth_when_api_keys_are_configured(monkeypatch, tmp_path: Path) -> None:
    paths = _paths(monkeypatch, tmp_path)
    _save_auth_config(paths, ["secret-key"])
    headers = {"Authorization": "Bearer secret-key"}

    with TestClient(create_app()) as client:
        docs = client.get("/docs", headers=headers)
        redoc = client.get("/redoc", headers=headers)
        openapi = client.get("/openapi.json", headers=headers)

    assert docs.status_code == 200
    assert redoc.status_code == 200
    assert openapi.status_code == 200
    assert "openapi" in openapi.json()


def test_docs_can_be_public_with_env_override(monkeypatch, tmp_path: Path) -> None:
    paths = _paths(monkeypatch, tmp_path)
    _save_auth_config(paths, ["secret-key"])
    monkeypatch.setenv("TOLLAMA_DOCS_PUBLIC", "1")

    with TestClient(create_app()) as client:
        docs = client.get("/docs")
        openapi = client.get("/openapi.json")
        health = client.get("/v1/health")

    assert docs.status_code == 200
    assert openapi.status_code == 200
    assert health.status_code == 401
