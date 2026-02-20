"""Tests for A2A Agent Card discovery endpoint."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from tollama.core.config import TollamaConfig, save_config
from tollama.core.storage import TollamaPaths, install_from_registry
from tollama.daemon.app import create_app


def _paths(monkeypatch, tmp_path: Path) -> TollamaPaths:
    paths = TollamaPaths(base_dir=tmp_path / ".tollama")
    monkeypatch.setenv("TOLLAMA_HOME", str(paths.base_dir))
    return paths


def test_agent_card_route_returns_latest_a2a_shape(monkeypatch, tmp_path: Path) -> None:
    paths = _paths(monkeypatch, tmp_path)
    install_from_registry("mock", accept_license=True, paths=paths)

    with TestClient(create_app()) as client:
        response = client.get("/.well-known/agent-card.json")

    assert response.status_code == 200
    card = response.json()
    assert card["name"] == "tollama"
    assert isinstance(card["supportedInterfaces"], list)
    assert card["supportedInterfaces"][0]["protocolBinding"] == "JSONRPC"
    assert card["supportedInterfaces"][0]["protocolVersion"] == "1.0"
    assert card["defaultInputModes"] == ["application/json"]
    assert card["defaultOutputModes"] == ["application/json"]
    assert card["capabilities"]["streaming"] is True

    skill_ids = {item["id"] for item in card["skills"]}
    assert {
        "analyze",
        "recommend",
        "generate",
        "forecast",
        "auto_forecast",
        "pipeline",
    } <= skill_ids


def test_agent_card_route_requires_auth_when_api_keys_are_configured(
    monkeypatch,
    tmp_path: Path,
) -> None:
    paths = _paths(monkeypatch, tmp_path)
    save_config(
        paths,
        TollamaConfig.model_validate({"auth": {"api_keys": ["secret-key"]}}),
    )

    with TestClient(create_app()) as client:
        unauthorized = client.get("/.well-known/agent-card.json")
        authorized = client.get(
            "/.well-known/agent-card.json",
            headers={"Authorization": "Bearer secret-key"},
        )

    assert unauthorized.status_code == 401
    assert authorized.status_code == 200

    card = authorized.json()
    assert card["securitySchemes"]["bearerAuth"]["type"] == "http"
    assert card["securityRequirements"] == [{"bearerAuth": []}]


def test_agent_card_skills_trim_when_no_models_installed(monkeypatch, tmp_path: Path) -> None:
    _paths(monkeypatch, tmp_path)

    with TestClient(create_app()) as client:
        response = client.get("/.well-known/agent-card.json")

    assert response.status_code == 200
    card = response.json()
    skill_ids = {item["id"] for item in card["skills"]}
    assert skill_ids == {"analyze", "recommend", "generate"}
