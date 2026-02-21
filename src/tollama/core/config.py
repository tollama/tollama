"""Persistent tollama configuration defaults."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, StrictStr, ValidationError

from .storage import TollamaPaths


class PullDefaults(BaseModel):
    """Default pull settings applied when request fields are omitted."""

    model_config = ConfigDict(extra="forbid", strict=True)

    hf_home: StrictStr | None = None
    offline: StrictBool | None = None
    local_files_only: StrictBool | None = None
    insecure: StrictBool | None = None
    http_proxy: StrictStr | None = None
    https_proxy: StrictStr | None = None
    no_proxy: StrictStr | None = None
    max_workers: StrictInt | None = Field(default=None, gt=0)


class DaemonDefaults(BaseModel):
    """Optional daemon defaults."""

    model_config = ConfigDict(extra="forbid", strict=True)

    host: StrictStr | None = None
    auto_bootstrap: StrictBool = True
    runner_commands: dict[StrictStr, list[StrictStr]] | None = None


class AuthConfig(BaseModel):
    """Optional API key auth configuration."""

    model_config = ConfigDict(extra="forbid", strict=True)

    api_keys: list[StrictStr] = Field(default_factory=list)


class TollamaConfig(BaseModel):
    """Top-level persisted tollama config."""

    model_config = ConfigDict(extra="forbid", strict=True)

    version: StrictInt = 1
    pull: PullDefaults = Field(default_factory=PullDefaults)
    daemon: DaemonDefaults = Field(default_factory=DaemonDefaults)
    auth: AuthConfig = Field(default_factory=AuthConfig)


CONFIG_KEY_DESCRIPTIONS: dict[str, str] = {
    "pull.hf_home": "Default HF_HOME path used during model pulls.",
    "pull.http_proxy": "Default HTTP proxy URL used during model pulls.",
    "pull.https_proxy": "Default HTTPS proxy URL used during model pulls.",
    "pull.no_proxy": "Default comma-separated no-proxy host patterns.",
    "pull.offline": "Force pulls to run in offline mode by default.",
    "pull.local_files_only": "Use cached artifacts only without forcing full offline mode.",
    "pull.insecure": "Disable TLS verification during pulls (debugging only).",
    "pull.max_workers": "Maximum download worker threads for model pulls.",
}


class ConfigFileError(RuntimeError):
    """Raised when the persisted config cannot be parsed or validated."""


def get_config_path(paths: TollamaPaths) -> Path:
    """Resolve config file path for one tollama home."""
    return paths.config_path


def load_config(paths: TollamaPaths) -> TollamaConfig:
    """Load config file or return defaults when missing."""
    config_path = get_config_path(paths)
    if not config_path.exists():
        return TollamaConfig()

    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigFileError(f"invalid JSON in {config_path}: {exc.msg}") from exc
    except OSError as exc:
        raise ConfigFileError(f"unable to read {config_path}: {exc}") from exc

    try:
        return TollamaConfig.model_validate(payload)
    except ValidationError as exc:
        raise ConfigFileError(f"invalid config in {config_path}: {exc}") from exc


def save_config(paths: TollamaPaths, config: TollamaConfig) -> None:
    """Atomically persist config using a .tmp file then rename."""
    config_path = get_config_path(paths)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = config_path.with_name(f"{config_path.name}.tmp")
    payload = config.model_dump(mode="json")
    temp_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temp_path.replace(config_path)


def update_config(paths: TollamaPaths, updates: dict[str, Any]) -> TollamaConfig:
    """Apply partial updates and persist the resulting config."""
    current = load_config(paths)
    merged = current.model_dump(mode="json")
    _deep_merge_dict(merged, updates)
    updated = TollamaConfig.model_validate(merged)
    save_config(paths, updated)
    return updated


def _deep_merge_dict(target: dict[str, Any], updates: dict[str, Any]) -> None:
    for key, value in updates.items():
        existing = target.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            _deep_merge_dict(existing, value)
            continue
        target[key] = value
