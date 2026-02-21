"""Tests for dashboard routes, CORS behavior, and aggregate state API."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from tollama.core.config import TollamaConfig, save_config
from tollama.core.storage import TollamaPaths
from tollama.daemon.app import create_app


def _paths(monkeypatch, tmp_path: Path) -> TollamaPaths:
    paths = TollamaPaths(base_dir=tmp_path / ".tollama")
    monkeypatch.setenv("TOLLAMA_HOME", str(paths.base_dir))
    return paths


def _save_auth_config(paths: TollamaPaths, api_keys: list[str]) -> None:
    save_config(
        paths,
        TollamaConfig.model_validate({"auth": {"api_keys": api_keys}}),
    )


def test_dashboard_state_aggregates_info_ps_and_usage(monkeypatch, tmp_path: Path) -> None:
    _paths(monkeypatch, tmp_path)

    with TestClient(create_app()) as client:
        response = client.get("/api/dashboard/state")

    assert response.status_code == 200
    payload = response.json()
    assert isinstance(payload.get("info"), dict)
    assert isinstance(payload.get("ps"), dict)
    assert isinstance(payload.get("usage"), dict)
    assert payload.get("warnings") == []


def test_dashboard_state_returns_partial_payload_when_usage_fails(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _paths(monkeypatch, tmp_path)
    app = create_app()
    app.state.usage_meter = None

    with TestClient(app) as client:
        response = client.get("/api/dashboard/state")

    assert response.status_code == 200
    payload = response.json()
    assert isinstance(payload.get("info"), dict)
    assert isinstance(payload.get("ps"), dict)
    assert payload.get("usage") is None
    warnings = payload.get("warnings")
    assert isinstance(warnings, list)
    assert warnings
    assert warnings[0]["source"] == "usage"
    assert warnings[0]["status_code"] == 503


def test_dashboard_routes_are_public_by_default(monkeypatch, tmp_path: Path) -> None:
    _paths(monkeypatch, tmp_path)

    with TestClient(create_app()) as client:
        root = client.get("/dashboard")
        deep_link = client.get("/dashboard/models")
        static_index = client.get("/dashboard/static/index.html")
        partial = client.get("/dashboard/partials/models-table")

    assert root.status_code == 200
    assert "Tollama Dashboard" in root.text
    assert deep_link.status_code == 200
    assert "Tollama Dashboard" in deep_link.text
    assert static_index.status_code == 200
    assert partial.status_code == 200
    assert "Installed Models" in partial.text


def test_dashboard_can_be_disabled_via_env(monkeypatch, tmp_path: Path) -> None:
    _paths(monkeypatch, tmp_path)
    monkeypatch.setenv("TOLLAMA_DASHBOARD", "0")

    with TestClient(create_app()) as client:
        root = client.get("/dashboard")
        static_index = client.get("/dashboard/static/index.html")

    assert root.status_code == 404
    assert static_index.status_code == 404


def test_dashboard_static_can_require_auth_via_env(monkeypatch, tmp_path: Path) -> None:
    paths = _paths(monkeypatch, tmp_path)
    _save_auth_config(paths, ["secret-key"])
    monkeypatch.setenv("TOLLAMA_DASHBOARD_REQUIRE_AUTH", "1")

    with TestClient(create_app()) as client:
        unauthorized_root = client.get("/dashboard")
        unauthorized_static = client.get("/dashboard/static/index.html")
        authorized_root = client.get("/dashboard", headers={"Authorization": "Bearer secret-key"})
        authorized_static = client.get(
            "/dashboard/static/index.html",
            headers={"Authorization": "Bearer secret-key"},
        )

    assert unauthorized_root.status_code == 401
    assert unauthorized_static.status_code == 401
    assert authorized_root.status_code == 200
    assert authorized_static.status_code == 200


def test_dashboard_state_endpoint_requires_api_key_when_configured(
    monkeypatch,
    tmp_path: Path,
) -> None:
    paths = _paths(monkeypatch, tmp_path)
    _save_auth_config(paths, ["secret-key"])

    with TestClient(create_app()) as client:
        unauthorized = client.get("/api/dashboard/state")
        authorized = client.get(
            "/api/dashboard/state",
            headers={"Authorization": "Bearer secret-key"},
        )

    assert unauthorized.status_code == 401
    assert authorized.status_code == 200


def test_cors_is_not_enabled_by_default(monkeypatch, tmp_path: Path) -> None:
    _paths(monkeypatch, tmp_path)

    with TestClient(create_app()) as client:
        response = client.get(
            "/api/version",
            headers={"Origin": "http://localhost:3000"},
        )

    assert response.status_code == 200
    assert "access-control-allow-origin" not in response.headers


def test_cors_allows_explicit_origins(monkeypatch, tmp_path: Path) -> None:
    _paths(monkeypatch, tmp_path)
    monkeypatch.setenv("TOLLAMA_CORS_ORIGINS", "http://localhost:3000, https://app.example")

    with TestClient(create_app()) as client:
        response = client.options(
            "/api/version",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET",
            },
        )

    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == "http://localhost:3000"


def test_cors_does_not_allow_unlisted_origin(monkeypatch, tmp_path: Path) -> None:
    _paths(monkeypatch, tmp_path)
    monkeypatch.setenv("TOLLAMA_CORS_ORIGINS", "https://app.example")

    with TestClient(create_app()) as client:
        response = client.get(
            "/api/version",
            headers={"Origin": "http://localhost:3000"},
        )

    assert response.status_code == 200
    assert "access-control-allow-origin" not in response.headers
