"""Helpers for resolving effective pull defaults with source metadata."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .config import PullDefaults, TollamaConfig

_TOKEN_ENV_VARS = (
    "TOLLAMA_HF_TOKEN",
    "HF_TOKEN",
    "HUGGINGFACE_HUB_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    "HF_HUB_TOKEN",
)


def resolve_effective_pull_defaults(
    *,
    env: Mapping[str, Any] | None = None,
    config: TollamaConfig | PullDefaults | Mapping[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    """Resolve /api/pull defaults from env, config, and hard defaults."""
    env_mapping = env or {}
    config_values = _coerce_pull_mapping(config)

    resolved: dict[str, dict[str, Any]] = {}

    offline_value, offline_source = _resolve_bool_default(
        env_value=_optional_bool_str(env_mapping.get("HF_HUB_OFFLINE")),
        config_value=config_values.get("offline"),
        hard_default=False,
    )
    resolved["offline"] = {"value": offline_value, "source": offline_source}

    local_files_only_value, local_files_only_source = _resolve_bool_default(
        env_value=None,
        config_value=config_values.get("local_files_only"),
        hard_default=False,
    )
    if offline_value and not local_files_only_value:
        local_files_only_value = True
        local_files_only_source = f"{offline_source} (offline)"
    resolved["local_files_only"] = {
        "value": local_files_only_value,
        "source": local_files_only_source,
    }

    insecure_value, insecure_source = _resolve_bool_default(
        env_value=None,
        config_value=config_values.get("insecure"),
        hard_default=False,
    )
    resolved["insecure"] = {"value": insecure_value, "source": insecure_source}

    hf_home_value, hf_home_source = _resolve_str_default(
        env_value=_optional_nonempty_str(env_mapping.get("HF_HOME")),
        config_value=config_values.get("hf_home"),
    )
    resolved["hf_home"] = {"value": hf_home_value, "source": hf_home_source}

    http_proxy_value, http_proxy_source = _resolve_str_default(
        env_value=_optional_nonempty_str(env_mapping.get("HTTP_PROXY")),
        config_value=config_values.get("http_proxy"),
    )
    resolved["http_proxy"] = {"value": http_proxy_value, "source": http_proxy_source}

    https_proxy_value, https_proxy_source = _resolve_str_default(
        env_value=_optional_nonempty_str(env_mapping.get("HTTPS_PROXY")),
        config_value=config_values.get("https_proxy"),
    )
    resolved["https_proxy"] = {"value": https_proxy_value, "source": https_proxy_source}

    no_proxy_value, no_proxy_source = _resolve_str_default(
        env_value=_optional_nonempty_str(env_mapping.get("NO_PROXY")),
        config_value=config_values.get("no_proxy"),
    )
    resolved["no_proxy"] = {"value": no_proxy_value, "source": no_proxy_source}

    max_workers_value, max_workers_source = _resolve_int_default(
        config_value=config_values.get("max_workers"),
        hard_default=8,
    )
    resolved["max_workers"] = {"value": max_workers_value, "source": max_workers_source}

    token_present = any(_optional_nonempty_str(env_mapping.get(key)) for key in _TOKEN_ENV_VARS)
    resolved["token_present"] = {
        "value": token_present,
        "source": "env" if token_present else "default",
    }
    return resolved


def _coerce_pull_mapping(
    config: TollamaConfig | PullDefaults | Mapping[str, Any] | None,
) -> Mapping[str, Any]:
    if isinstance(config, TollamaConfig):
        return config.pull.model_dump(mode="json")
    if isinstance(config, PullDefaults):
        return config.model_dump(mode="json")
    if isinstance(config, Mapping):
        nested_pull = config.get("pull")
        if isinstance(nested_pull, Mapping):
            return nested_pull
        return config
    return {}


def _resolve_bool_default(
    *,
    env_value: bool | None,
    config_value: Any,
    hard_default: bool,
) -> tuple[bool, str]:
    if env_value is not None:
        return env_value, "env"
    if isinstance(config_value, bool):
        return config_value, "config"
    return hard_default, "default"


def _resolve_str_default(
    *,
    env_value: str | None,
    config_value: Any,
) -> tuple[str | None, str]:
    if env_value is not None:
        return env_value, "env"
    normalized = _optional_nonempty_str(config_value)
    if normalized is not None:
        return normalized, "config"
    return None, "default"


def _resolve_int_default(
    *,
    config_value: Any,
    hard_default: int,
) -> tuple[int, str]:
    if isinstance(config_value, int) and config_value > 0:
        return config_value, "config"
    return hard_default, "default"


def _optional_bool_str(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if not isinstance(value, str):
        return None

    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return None


def _optional_nonempty_str(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    if not normalized:
        return None
    return normalized

