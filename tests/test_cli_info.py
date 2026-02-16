"""Tests for tollama info collection and CLI output."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest
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


def test_collect_info_auto_falls_back_to_local_when_api_info_unreachable(
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

    info = collect_info(base_url="http://127.0.0.1:1", paths=paths, timeout_s=0.1, mode="auto")

    assert info["daemon"]["reachable"] is False
    assert info["config"]["pull"]["offline"] is True
    assert [item["name"] for item in info["models"]["installed"]] == ["mock"]
    assert info["models"]["loaded"] == []
    assert info["client"]["source"] == "local"


def test_collect_info_remote_mode_uses_api_info_payload(monkeypatch, tmp_path: Path) -> None:
    paths = TollamaPaths(base_dir=tmp_path / ".tollama")
    monkeypatch.setenv("TOLLAMA_HOME", str(paths.base_dir))

    remote_payload = {
        "daemon": {
            "version": "0.0.1",
            "started_at": "2026-02-16T00:00:00Z",
            "uptime_seconds": 20,
            "host_binding": "127.0.0.1:11435",
        },
        "paths": {
            "tollama_home": str(paths.base_dir),
            "config_path": str(paths.config_path),
            "config_exists": False,
        },
        "config": None,
        "env": {"TOLLAMA_HF_TOKEN_present": False},
        "pull_defaults": {"offline": {"value": False, "source": "default"}},
        "models": {"installed": [{"name": "chronos2"}], "loaded": [{"model": "chronos2"}]},
        "runners": [],
    }

    def _handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/info":
            return httpx.Response(200, json=remote_payload)
        return httpx.Response(404, json={"detail": "not found"})

    transport = httpx.MockTransport(_handler)

    def _mock_client_factory(*, base_url: str, timeout_s: float) -> httpx.Client:
        return httpx.Client(base_url=base_url, timeout=timeout_s, transport=transport)

    monkeypatch.setattr("tollama.cli.info._make_http_client", _mock_client_factory)

    info = collect_info(
        base_url="http://localhost:11435",
        paths=paths,
        timeout_s=0.2,
        mode="remote",
    )

    assert info["daemon"]["reachable"] is True
    assert info["daemon"]["version"] == "0.0.1"
    assert [item["name"] for item in info["models"]["installed"]] == ["chronos2"]
    assert [item["model"] for item in info["models"]["loaded"]] == ["chronos2"]
    assert info["client"]["source"] == "remote"


def test_collect_info_remote_mode_raises_when_unreachable(tmp_path: Path) -> None:
    paths = TollamaPaths(base_dir=tmp_path / ".tollama")
    with pytest.raises(RuntimeError):
        collect_info(base_url="http://127.0.0.1:1", paths=paths, timeout_s=0.1, mode="remote")


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

    info = collect_info(base_url="http://127.0.0.1:1", paths=paths, timeout_s=0.1, mode="local")
    payload = json.dumps(info, sort_keys=True)

    assert info["env"]["TOLLAMA_HF_TOKEN_present"] is True
    assert "secret-token" not in payload
    assert "secret-from-config" not in payload


def test_info_command_json_output_contains_collected_payload(monkeypatch) -> None:
    snapshot: dict[str, Any] = {
        "client": {
            "base_url": "http://localhost:11435",
            "api_base_url": "http://localhost:11435/api",
            "source": "remote",
            "version": "0.1.0",
        },
        "daemon": {
            "version": "0.1.0",
            "started_at": "2026-02-16T00:00:00Z",
            "uptime_seconds": 10,
            "host_binding": "127.0.0.1:11435",
            "reachable": True,
            "error": None,
        },
        "paths": {
            "tollama_home": "/tmp/tollama",
            "config_path": "/tmp/tollama/config.json",
            "config_exists": False,
        },
        "config": None,
        "env": {"TOLLAMA_HF_TOKEN_present": False},
        "pull_defaults": {"offline": {"value": False, "source": "default"}},
        "models": {"installed": [], "loaded": []},
        "runners": [],
    }
    monkeypatch.setattr("tollama.cli.main.collect_info", lambda **_: snapshot)

    runner = CliRunner()
    result = runner.invoke(app, ["info", "--json"])

    assert result.exit_code == 0
    parsed = json.loads(result.stdout)
    assert parsed["daemon"]["version"] == "0.1.0"
    assert parsed["paths"]["tollama_home"] == "/tmp/tollama"


def test_info_command_remote_flag_errors_when_daemon_unreachable(monkeypatch) -> None:
    def _raise_error(**kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("connection refused")

    monkeypatch.setattr("tollama.cli.main.collect_info", _raise_error)

    runner = CliRunner()
    result = runner.invoke(app, ["info", "--remote"])

    assert result.exit_code == 1
    assert "connection refused" in result.stdout


def test_info_command_local_flag_forces_local_mode(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    snapshot: dict[str, Any] = {
        "client": {"base_url": "http://localhost:11435", "api_base_url": "", "source": "local"},
        "daemon": {"reachable": False, "error": None},
        "paths": {
            "tollama_home": "/tmp",
            "config_path": "/tmp/config.json",
            "config_exists": False,
        },
        "config": None,
        "env": {},
        "pull_defaults": {},
        "models": {"installed": [], "loaded": []},
        "runners": [],
    }

    def _fake_collect_info(*, mode: str, **kwargs: Any) -> dict[str, Any]:
        captured["mode"] = mode
        return snapshot

    monkeypatch.setattr("tollama.cli.main.collect_info", _fake_collect_info)

    runner = CliRunner()
    result = runner.invoke(app, ["info", "--local", "--json"])

    assert result.exit_code == 0
    assert captured["mode"] == "local"
