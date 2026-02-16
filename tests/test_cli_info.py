"""Tests for tollama info collection and CLI output."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
from typer.testing import CliRunner

from tollama.cli.info import collect_info
from tollama.cli.main import app
from tollama.core.config import TollamaConfig, save_config
from tollama.core.storage import TollamaPaths


def _write_manifest(paths: TollamaPaths, *, name: str) -> None:
    manifest = {
        "name": name,
        "family": "mock",
        "resolved": {"commit_sha": "local", "snapshot_path": None},
        "size_bytes": 7,
        "pulled_at": "2026-02-16T00:00:00Z",
        "installed_at": "2026-02-16T00:00:00Z",
    }
    manifest_path = paths.manifest_path(name)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


def test_collect_info_with_daemon_unreachable_falls_back_to_local(
    monkeypatch,
    tmp_path: Path,
) -> None:
    paths = TollamaPaths(base_dir=tmp_path / ".tollama")
    monkeypatch.setenv("TOLLAMA_HOME", str(paths.base_dir))
    save_config(
        paths,
        TollamaConfig.model_validate(
            {
                "pull": {
                    "offline": True,
                    "https_proxy": "http://proxy.internal:3128",
                }
            },
        ),
    )
    _write_manifest(paths, name="mock")

    info = collect_info(base_url="http://127.0.0.1:1", paths=paths, timeout_s=0.1)

    assert info["daemon"]["reachable"] is False
    assert info["filesystem"]["config"]["pull"]["offline"] is True
    assert info["models"]["installed_source"] == "local"
    assert [item["name"] for item in info["models"]["installed"]] == ["mock"]
    assert info["models"]["loaded"] == []


def test_collect_info_with_daemon_reachable_uses_daemon_models(monkeypatch, tmp_path: Path) -> None:
    paths = TollamaPaths(base_dir=tmp_path / ".tollama")
    monkeypatch.setenv("TOLLAMA_HOME", str(paths.base_dir))

    def _handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/version":
            return httpx.Response(200, json={"version": "0.0.1"})
        if request.url.path == "/api/tags":
            return httpx.Response(
                200,
                json={
                    "models": [
                        {
                            "name": "chronos2",
                            "model": "chronos2",
                            "digest": "fake-digest",
                            "size": 128,
                            "modified_at": "2026-02-16T00:00:00Z",
                            "details": {"family": "torch"},
                        }
                    ]
                },
            )
        if request.url.path == "/api/ps":
            return httpx.Response(
                200,
                json={
                    "models": [
                        {
                            "name": "chronos2",
                            "model": "chronos2",
                            "expires_at": None,
                            "details": {"family": "torch"},
                        }
                    ]
                },
            )
        return httpx.Response(404, json={"detail": "not found"})

    transport = httpx.MockTransport(_handler)

    def _mock_client_factory(*, base_url: str, timeout_s: float) -> httpx.Client:
        return httpx.Client(base_url=base_url, timeout=timeout_s, transport=transport)

    monkeypatch.setattr("tollama.cli.info._make_http_client", _mock_client_factory)

    info = collect_info(base_url="http://localhost:11435", paths=paths, timeout_s=0.2)

    assert info["daemon"]["reachable"] is True
    assert info["daemon"]["version"] == "0.0.1"
    assert info["models"]["installed_source"] == "daemon"
    assert [item["name"] for item in info["models"]["installed"]] == ["chronos2"]
    assert [item["model"] for item in info["models"]["loaded"]] == ["chronos2"]


def test_collect_info_never_exposes_token_value(monkeypatch, tmp_path: Path) -> None:
    paths = TollamaPaths(base_dir=tmp_path / ".tollama")
    monkeypatch.setenv("TOLLAMA_HOME", str(paths.base_dir))
    monkeypatch.setenv("TOLLAMA_HF_TOKEN", "secret-token")
    config_path = paths.config_path
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        '{"version":1,"pull":{"offline":true,"token":"secret-from-config"}}',
        encoding="utf-8",
    )

    info = collect_info(base_url="http://127.0.0.1:1", paths=paths, timeout_s=0.1)
    payload = json.dumps(info, sort_keys=True)

    assert info["client"]["env"]["TOLLAMA_HF_TOKEN_present"] is True
    assert "secret-token" not in payload
    assert "secret-from-config" not in payload


def test_info_command_json_output(monkeypatch) -> None:
    snapshot: dict[str, Any] = {
        "client": {
            "base_url": "http://localhost:11435",
            "api_base_url": "http://localhost:11435/api",
            "env": {"TOLLAMA_HF_TOKEN_present": False},
        },
        "filesystem": {
            "tollama_home": "/tmp/tollama",
            "config_path": "/tmp/tollama/config.json",
            "config_exists": False,
            "config": {"version": 1, "pull": {}, "daemon": {}},
            "config_error": None,
        },
        "daemon": {"reachable": False, "version": None, "error": "connection refused"},
        "pull_defaults": {"offline": {"value": False, "source": "default"}},
        "models": {"installed_source": "local", "installed": [], "loaded": []},
    }
    monkeypatch.setattr("tollama.cli.main.collect_info", lambda **_: snapshot)

    runner = CliRunner()
    result = runner.invoke(app, ["info", "--json"])

    assert result.exit_code == 0
    parsed = json.loads(result.stdout)
    assert parsed["client"]["base_url"] == "http://localhost:11435"
    assert parsed["daemon"]["reachable"] is False
