"""Tests for tollama config CLI commands."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from tollama.cli.main import app


def test_config_set_get_list_and_unset(monkeypatch, tmp_path: Path) -> None:
    home = tmp_path / "state-home"
    monkeypatch.setenv("TOLLAMA_HOME", str(home))
    runner = CliRunner()

    listed_default = runner.invoke(app, ["config", "list"])
    assert listed_default.exit_code == 0
    default_payload = json.loads(listed_default.stdout)
    assert default_payload["version"] == 1
    assert default_payload["pull"]["offline"] is None
    assert not (home / "config.json").exists()

    set_offline = runner.invoke(app, ["config", "set", "pull.offline", "true"])
    assert set_offline.exit_code == 0
    assert (home / "config.json").exists()

    get_offline = runner.invoke(app, ["config", "get", "pull.offline"])
    assert get_offline.exit_code == 0
    assert get_offline.stdout.strip() == "true"

    set_proxy = runner.invoke(app, ["config", "set", "pull.https_proxy", "http://proxy:3128"])
    assert set_proxy.exit_code == 0

    listed = runner.invoke(app, ["config", "list", "--json"])
    assert listed.exit_code == 0
    payload = json.loads(listed.stdout)
    assert payload["pull"]["offline"] is True
    assert payload["pull"]["https_proxy"] == "http://proxy:3128"

    unset_offline = runner.invoke(app, ["config", "unset", "pull.offline"])
    assert unset_offline.exit_code == 0
    get_unset = runner.invoke(app, ["config", "get", "pull.offline"])
    assert get_unset.exit_code == 0
    assert get_unset.stdout.strip() == "null"


def test_config_set_rejects_unknown_key(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("TOLLAMA_HOME", str(tmp_path / "state-home"))
    runner = CliRunner()

    result = runner.invoke(app, ["config", "set", "pull.unknown", "true"])

    assert result.exit_code != 0
    assert "unsupported key" in result.stdout


def test_pull_no_config_sends_neutral_defaults(monkeypatch) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setenv("TOLLAMA_HF_TOKEN", "env-token")

    class _FakeClient:
        def __init__(self, base_url: str, timeout: float) -> None:
            captured["base_url"] = base_url
            captured["timeout"] = timeout

        def pull_model(
            self,
            name: str,
            *,
            stream: bool,
            insecure: bool | None,
            offline: bool | None,
            local_files_only: bool | None,
            http_proxy: str | None,
            https_proxy: str | None,
            no_proxy: str | None,
            hf_home: str | None,
            max_workers: int | None = None,
            token: str | None = None,
            include_null_fields: set[str] | None = None,
        ) -> dict[str, object]:
            captured["pull"] = {
                "name": name,
                "stream": stream,
                "insecure": insecure,
                "offline": offline,
                "local_files_only": local_files_only,
                "http_proxy": http_proxy,
                "https_proxy": https_proxy,
                "no_proxy": no_proxy,
                "hf_home": hf_home,
                "max_workers": max_workers,
                "token": token,
                "include_null_fields": include_null_fields,
            }
            return {"status": "success", "model": name}

    monkeypatch.setattr("tollama.cli.main.TollamaClient", _FakeClient)
    runner = CliRunner()

    result = runner.invoke(app, ["pull", "mock", "--no-config", "--no-stream"])

    assert result.exit_code == 0
    assert captured["pull"] == {
        "name": "mock",
        "stream": False,
        "insecure": False,
        "offline": False,
        "local_files_only": False,
        "http_proxy": None,
        "https_proxy": None,
        "no_proxy": None,
        "hf_home": None,
        "max_workers": None,
        "token": "env-token",
        "include_null_fields": {"http_proxy", "https_proxy", "no_proxy", "hf_home"},
    }
