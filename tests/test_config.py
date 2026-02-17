"""Tests for persistent tollama config helpers."""

from __future__ import annotations

import json

import pytest

from tollama.core.config import (
    ConfigFileError,
    TollamaConfig,
    get_config_path,
    load_config,
    save_config,
    update_config,
)
from tollama.core.storage import TollamaPaths


def _paths_from_env(monkeypatch, tmp_path) -> TollamaPaths:
    home = tmp_path / "custom-home"
    monkeypatch.setenv("TOLLAMA_HOME", str(home))
    return TollamaPaths.default()


def test_load_config_returns_defaults_when_file_missing(monkeypatch, tmp_path) -> None:
    paths = _paths_from_env(monkeypatch, tmp_path)

    config = load_config(paths)

    assert config == TollamaConfig()
    assert get_config_path(paths) == paths.base_dir / "config.json"
    assert not get_config_path(paths).exists()


def test_save_config_writes_file(monkeypatch, tmp_path) -> None:
    paths = _paths_from_env(monkeypatch, tmp_path)
    config = TollamaConfig.model_validate(
        {
            "pull": {
                "offline": True,
                "https_proxy": "http://proxy.internal:3128",
                "max_workers": 4,
            }
        },
    )

    save_config(paths, config)

    on_disk = json.loads(get_config_path(paths).read_text(encoding="utf-8"))
    assert on_disk["pull"]["offline"] is True
    assert on_disk["pull"]["https_proxy"] == "http://proxy.internal:3128"
    assert on_disk["pull"]["max_workers"] == 4


def test_update_config_updates_fields_and_persists(monkeypatch, tmp_path) -> None:
    paths = _paths_from_env(monkeypatch, tmp_path)

    updated = update_config(
        paths,
        {"pull": {"offline": True, "hf_home": "/tmp/hf-cache", "local_files_only": True}},
    )

    assert updated.pull.offline is True
    assert updated.pull.hf_home == "/tmp/hf-cache"
    assert updated.pull.local_files_only is True

    reloaded = load_config(paths)
    assert reloaded.pull.offline is True
    assert reloaded.pull.hf_home == "/tmp/hf-cache"
    assert reloaded.pull.local_files_only is True


def test_save_config_recovers_from_stale_tmp_file(monkeypatch, tmp_path) -> None:
    paths = _paths_from_env(monkeypatch, tmp_path)
    save_config(paths, TollamaConfig())
    config_path = get_config_path(paths)
    temp_path = config_path.with_name(f"{config_path.name}.tmp")
    temp_path.write_text('{"version": 1, "pull": {"offline": tru', encoding="utf-8")

    save_config(paths, TollamaConfig.model_validate({"pull": {"offline": True}}))

    payload = json.loads(config_path.read_text(encoding="utf-8"))
    assert payload["pull"]["offline"] is True
    assert not temp_path.exists()


def test_load_config_invalid_json_reports_file_path(monkeypatch, tmp_path) -> None:
    paths = _paths_from_env(monkeypatch, tmp_path)
    config_path = get_config_path(paths)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text("{not-json}", encoding="utf-8")

    with pytest.raises(ConfigFileError) as exc_info:
        load_config(paths)

    assert str(config_path) in str(exc_info.value)


def test_daemon_defaults_auto_bootstrap_defaults_to_true(monkeypatch, tmp_path) -> None:
    paths = _paths_from_env(monkeypatch, tmp_path)
    config = load_config(paths)
    assert config.daemon.auto_bootstrap is True


def test_daemon_defaults_runner_commands_defaults_to_none(monkeypatch, tmp_path) -> None:
    paths = _paths_from_env(monkeypatch, tmp_path)
    config = load_config(paths)
    assert config.daemon.runner_commands is None


def test_daemon_defaults_runner_commands_round_trip(monkeypatch, tmp_path) -> None:
    paths = _paths_from_env(monkeypatch, tmp_path)
    updated = update_config(
        paths,
        {
            "daemon": {
                "runner_commands": {
                    "torch": ["/opt/venv/bin/python", "-m", "tollama.runners.torch_runner.main"],
                },
            },
        },
    )
    assert updated.daemon.runner_commands is not None
    assert "torch" in updated.daemon.runner_commands

    reloaded = load_config(paths)
    assert reloaded.daemon.runner_commands == updated.daemon.runner_commands


def test_daemon_defaults_auto_bootstrap_can_be_disabled(monkeypatch, tmp_path) -> None:
    paths = _paths_from_env(monkeypatch, tmp_path)
    updated = update_config(paths, {"daemon": {"auto_bootstrap": False}})
    assert updated.daemon.auto_bootstrap is False

    reloaded = load_config(paths)
    assert reloaded.daemon.auto_bootstrap is False
