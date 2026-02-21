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
    assert "unknown key" in result.stdout


def test_config_set_unknown_key_suggests_closest_match(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("TOLLAMA_HOME", str(tmp_path / "state-home"))
    runner = CliRunner()

    result = runner.invoke(app, ["config", "set", "offline", "true"])

    assert result.exit_code != 0
    assert "Did you mean 'pull.offline'" in result.stdout


def test_config_keys_lists_descriptions_and_values(monkeypatch, tmp_path: Path) -> None:
    home = tmp_path / "state-home"
    monkeypatch.setenv("TOLLAMA_HOME", str(home))
    runner = CliRunner()

    set_result = runner.invoke(app, ["config", "set", "pull.offline", "true"])
    assert set_result.exit_code == 0

    keys_result = runner.invoke(app, ["config", "keys"])
    assert keys_result.exit_code == 0
    assert "pull.offline" in keys_result.stdout
    assert "Force pulls to run in offline mode by default." in keys_result.stdout
    assert "true" in keys_result.stdout

    keys_json = runner.invoke(app, ["config", "keys", "--json"])
    assert keys_json.exit_code == 0
    payload = json.loads(keys_json.stdout)
    offline_entry = next(item for item in payload if item["key"] == "pull.offline")
    assert offline_entry["value"] is True
    assert offline_entry["description"] == "Force pulls to run in offline mode by default."


def test_config_init_writes_default_json_and_respects_force(monkeypatch, tmp_path: Path) -> None:
    home = tmp_path / "state-home"
    monkeypatch.setenv("TOLLAMA_HOME", str(home))
    runner = CliRunner()

    init_result = runner.invoke(app, ["config", "init"])
    assert init_result.exit_code == 0
    config_path = home / "config.json"
    assert config_path.exists()
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    assert payload["version"] == 1

    blocked = runner.invoke(app, ["config", "init"])
    assert blocked.exit_code != 0
    assert "use --force to overwrite" in blocked.stdout

    config_path.write_text(
        '{"version":1,"pull":{"offline":true},"daemon":{"auto_bootstrap":true},"auth":{"api_keys":[]}}',
        encoding="utf-8",
    )
    forced = runner.invoke(app, ["config", "init", "--force"])
    assert forced.exit_code == 0
    forced_payload = json.loads(config_path.read_text(encoding="utf-8"))
    assert forced_payload["pull"]["offline"] is None


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
            accept_license: bool = False,
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
                "accept_license": accept_license,
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
        "accept_license": False,
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
