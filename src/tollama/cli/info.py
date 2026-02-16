"""Diagnostics collectors for the tollama CLI info command."""

from __future__ import annotations

import os
from typing import Any

import httpx

from tollama.core.config import ConfigFileError, load_config
from tollama.core.storage import TollamaPaths, list_installed

_REDACTED_VALUE = "<redacted>"
_SENSITIVE_KEY_TOKENS = ("token", "secret", "password", "authorization", "api_key", "apikey")


def collect_info(base_url: str, paths: TollamaPaths, timeout_s: float = 2.0) -> dict[str, Any]:
    """Collect one diagnostics snapshot for `tollama info`."""
    resolved_base_url = base_url.rstrip("/")
    if not resolved_base_url:
        resolved_base_url = base_url

    client_env = _collect_env()
    filesystem = _collect_filesystem(paths)
    daemon, daemon_tags, daemon_loaded = _collect_daemon_state(
        base_url=resolved_base_url,
        timeout_s=timeout_s,
    )

    installed_source = "daemon"
    installed_models = daemon_tags
    if not daemon["reachable"] or daemon_tags is None:
        installed_source = "local"
        installed_models = _local_installed_models(paths)

    loaded_models = daemon_loaded if daemon["reachable"] else []
    pull_defaults = _effective_pull_defaults(
        env=client_env,
        config=filesystem.get("config"),
    )

    return {
        "client": {
            "base_url": resolved_base_url,
            "api_base_url": f"{resolved_base_url}/api",
            "env": client_env,
        },
        "filesystem": filesystem,
        "daemon": daemon,
        "pull_defaults": pull_defaults,
        "models": {
            "installed_source": installed_source,
            "installed": installed_models,
            "loaded": loaded_models,
        },
    }


def _collect_env() -> dict[str, Any]:
    return {
        "TOLLAMA_HOST": _env_or_none("TOLLAMA_HOST"),
        "TOLLAMA_HOME": _env_or_none("TOLLAMA_HOME"),
        "HF_HOME": _env_or_none("HF_HOME"),
        "HF_HUB_OFFLINE": _env_or_none("HF_HUB_OFFLINE"),
        "HTTP_PROXY": _env_or_none("HTTP_PROXY"),
        "HTTPS_PROXY": _env_or_none("HTTPS_PROXY"),
        "NO_PROXY": _env_or_none("NO_PROXY"),
        "TOLLAMA_HF_TOKEN_present": _env_or_none("TOLLAMA_HF_TOKEN") is not None,
    }


def _collect_filesystem(paths: TollamaPaths) -> dict[str, Any]:
    config_path = paths.config_path
    config_exists = config_path.exists()
    filesystem: dict[str, Any] = {
        "tollama_home": str(paths.base_dir),
        "config_path": str(config_path),
        "config_exists": config_exists,
        "config": None,
        "config_error": None,
    }

    try:
        config = load_config(paths)
        filesystem["config"] = _redact_sensitive(config.model_dump(mode="json"))
        return filesystem
    except ConfigFileError:
        filesystem["config_error"] = f"invalid config file: {config_path}"
    except OSError as exc:
        filesystem["config_error"] = f"unable to read config file {config_path}: {exc}"

    return filesystem


def _collect_daemon_state(
    *,
    base_url: str,
    timeout_s: float,
) -> tuple[dict[str, Any], list[dict[str, Any]] | None, list[dict[str, Any]]]:
    daemon_info: dict[str, Any] = {
        "reachable": False,
        "version": None,
        "error": None,
    }
    installed_models: list[dict[str, Any]] | None = None
    loaded_models: list[dict[str, Any]] = []

    try:
        with _make_http_client(base_url=base_url, timeout_s=timeout_s) as client:
            version_payload = _get_json(client, "/api/version")
            version_value = version_payload.get("version")
            daemon_info["version"] = str(version_value) if version_value is not None else None
            daemon_info["reachable"] = True

            try:
                tags_payload = _get_json(client, "/api/tags")
                installed_models = _models_from_daemon_tags(tags_payload)
            except Exception as exc:  # noqa: BLE001
                daemon_info["error"] = f"failed to read /api/tags: {exc}"

            try:
                ps_payload = _get_json(client, "/api/ps")
                loaded_models = _models_from_daemon_ps(ps_payload)
            except Exception as exc:  # noqa: BLE001
                error_message = f"failed to read /api/ps: {exc}"
                daemon_info["error"] = (
                    f"{daemon_info['error']}; {error_message}"
                    if daemon_info["error"]
                    else error_message
                )
    except Exception as exc:  # noqa: BLE001
        daemon_info["error"] = str(exc)

    return daemon_info, installed_models, loaded_models


def _local_installed_models(paths: TollamaPaths) -> list[dict[str, Any]]:
    try:
        manifests = list_installed(paths=paths)
    except Exception:  # noqa: BLE001
        return []

    models: list[dict[str, Any]] = []
    for manifest in manifests:
        if not isinstance(manifest, dict):
            continue
        models.append(_model_from_manifest(manifest))
    return models


def _models_from_daemon_tags(payload: dict[str, Any]) -> list[dict[str, Any]]:
    models = payload.get("models")
    if not isinstance(models, list):
        return []

    entries: list[dict[str, Any]] = []
    for item in models:
        if not isinstance(item, dict):
            continue
        details = item.get("details")
        family = details.get("family") if isinstance(details, dict) else None
        entries.append(
            {
                "name": _str_or_none(item.get("name")),
                "family": _str_or_none(family),
                "digest": _str_or_none(item.get("digest")),
                "size": _int_or_none(item.get("size")),
                "modified_at": _str_or_none(item.get("modified_at")),
                "raw": _redact_sensitive(item),
            },
        )
    return entries


def _models_from_daemon_ps(payload: dict[str, Any]) -> list[dict[str, Any]]:
    models = payload.get("models")
    if not isinstance(models, list):
        return []

    entries: list[dict[str, Any]] = []
    for item in models:
        if not isinstance(item, dict):
            continue
        entries.append(
            {
                "name": _str_or_none(item.get("name")),
                "model": _str_or_none(item.get("model")),
                "family": _str_or_none(_extract_family(item)),
                "expires_at": _str_or_none(item.get("expires_at")),
                "raw": _redact_sensitive(item),
            },
        )
    return entries


def _extract_family(payload: dict[str, Any]) -> str | None:
    details = payload.get("details")
    if isinstance(details, dict):
        return _str_or_none(details.get("family"))
    return None


def _model_from_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    resolved = manifest.get("resolved")
    resolved_map = resolved if isinstance(resolved, dict) else {}
    digest = _str_or_none(resolved_map.get("commit_sha")) or _str_or_none(manifest.get("digest"))
    size = _int_or_none(manifest.get("size_bytes"))
    if size is None:
        size = _int_or_none(manifest.get("size"))

    modified_at = _str_or_none(manifest.get("pulled_at")) or _str_or_none(
        manifest.get("installed_at"),
    )
    return {
        "name": _str_or_none(manifest.get("name")),
        "family": _str_or_none(manifest.get("family")),
        "digest": digest,
        "size": size,
        "modified_at": modified_at,
        "raw": _redact_sensitive(manifest),
    }


def _effective_pull_defaults(
    *,
    env: dict[str, Any],
    config: dict[str, Any] | None,
) -> dict[str, dict[str, Any]]:
    config_pull = config.get("pull") if isinstance(config, dict) else None
    pull_defaults = config_pull if isinstance(config_pull, dict) else {}

    resolved: dict[str, dict[str, Any]] = {}
    offline_value, offline_source = _resolve_pull_default_bool(
        env_var=_optional_bool_str(env.get("HF_HUB_OFFLINE")),
        config_value=pull_defaults.get("offline"),
        hard_default=False,
    )
    resolved["offline"] = {"value": offline_value, "source": offline_source}

    local_files_only_value, local_files_only_source = _resolve_pull_default_bool(
        env_var=None,
        config_value=pull_defaults.get("local_files_only"),
        hard_default=False,
    )
    if offline_value and not local_files_only_value:
        local_files_only_value = True
        local_files_only_source = f"{offline_source} (offline)"
    resolved["local_files_only"] = {
        "value": local_files_only_value,
        "source": local_files_only_source,
    }

    insecure_value, insecure_source = _resolve_pull_default_bool(
        env_var=None,
        config_value=pull_defaults.get("insecure"),
        hard_default=False,
    )
    resolved["insecure"] = {"value": insecure_value, "source": insecure_source}

    hf_home_value, hf_home_source = _resolve_pull_default_str(
        env_var=_optional_nonempty_str(env.get("HF_HOME")),
        config_value=pull_defaults.get("hf_home"),
    )
    resolved["hf_home"] = {"value": hf_home_value, "source": hf_home_source}

    http_proxy_value, http_proxy_source = _resolve_pull_default_str(
        env_var=_optional_nonempty_str(env.get("HTTP_PROXY")),
        config_value=pull_defaults.get("http_proxy"),
    )
    resolved["http_proxy"] = {"value": http_proxy_value, "source": http_proxy_source}

    https_proxy_value, https_proxy_source = _resolve_pull_default_str(
        env_var=_optional_nonempty_str(env.get("HTTPS_PROXY")),
        config_value=pull_defaults.get("https_proxy"),
    )
    resolved["https_proxy"] = {"value": https_proxy_value, "source": https_proxy_source}

    no_proxy_value, no_proxy_source = _resolve_pull_default_str(
        env_var=_optional_nonempty_str(env.get("NO_PROXY")),
        config_value=pull_defaults.get("no_proxy"),
    )
    resolved["no_proxy"] = {"value": no_proxy_value, "source": no_proxy_source}

    max_workers_value, max_workers_source = _resolve_pull_default_int(
        config_value=pull_defaults.get("max_workers"),
        hard_default=8,
    )
    resolved["max_workers"] = {"value": max_workers_value, "source": max_workers_source}
    return resolved


def _resolve_pull_default_bool(
    *,
    env_var: bool | None,
    config_value: Any,
    hard_default: bool,
) -> tuple[bool, str]:
    if env_var is not None:
        return env_var, "env"
    if isinstance(config_value, bool):
        return config_value, "config"
    return hard_default, "default"


def _resolve_pull_default_str(
    *,
    env_var: str | None,
    config_value: Any,
) -> tuple[str | None, str]:
    if env_var is not None:
        return env_var, "env"
    config_value_normalized = _optional_nonempty_str(config_value)
    if config_value_normalized is not None:
        return config_value_normalized, "config"
    return None, "default"


def _resolve_pull_default_int(*, config_value: Any, hard_default: int) -> tuple[int, str]:
    if isinstance(config_value, int) and config_value > 0:
        return config_value, "config"
    return hard_default, "default"


def _get_json(client: httpx.Client, path: str) -> dict[str, Any]:
    response = client.get(path)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError(f"unexpected response payload for {path}")
    return payload


def _make_http_client(*, base_url: str, timeout_s: float) -> httpx.Client:
    return httpx.Client(base_url=base_url, timeout=timeout_s)


def _redact_sensitive(value: Any) -> Any:
    if isinstance(value, dict):
        redacted: dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key)
            if _looks_sensitive_key(key_text):
                redacted[key_text] = _REDACTED_VALUE
                continue
            redacted[key_text] = _redact_sensitive(item)
        return redacted
    if isinstance(value, list):
        return [_redact_sensitive(item) for item in value]
    return value


def _looks_sensitive_key(key: str) -> bool:
    lowered = key.lower()
    return any(token in lowered for token in _SENSITIVE_KEY_TOKENS)


def _env_or_none(name: str) -> str | None:
    return _optional_nonempty_str(os.environ.get(name))


def _optional_nonempty_str(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    if not normalized:
        return None
    return normalized


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


def _str_or_none(value: Any) -> str | None:
    if isinstance(value, str):
        normalized = value.strip()
        return normalized if normalized else None
    return None


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None
