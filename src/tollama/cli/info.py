"""Diagnostics collectors for the tollama CLI info command."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from typing import Any, Literal

import httpx

from tollama import __version__ as CLI_VERSION
from tollama.core.config import ConfigFileError, TollamaConfig, load_config
from tollama.core.pull_defaults import resolve_effective_pull_defaults
from tollama.core.redact import redact_config_dict, redact_env_dict, redact_proxy_url
from tollama.core.registry import get_model_spec, list_registry_models
from tollama.core.storage import TollamaPaths, list_installed

InfoMode = Literal["auto", "local", "remote"]

_INFO_ENV_KEYS = (
    "TOLLAMA_HOME",
    "TOLLAMA_HOST",
    "HF_HOME",
    "HF_HUB_OFFLINE",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "NO_PROXY",
)
_TOKEN_ENV_KEYS = (
    "TOLLAMA_HF_TOKEN",
    "HF_TOKEN",
    "HUGGINGFACE_HUB_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    "HF_HUB_TOKEN",
)


def collect_info(
    base_url: str,
    paths: TollamaPaths,
    timeout_s: float = 2.0,
    *,
    mode: InfoMode = "auto",
    api_key: str | None = None,
) -> dict[str, Any]:
    """Collect one diagnostics snapshot for `tollama info`."""
    resolved_base_url = base_url.rstrip("/") or base_url
    if mode == "local":
        return _collect_local_snapshot(
            base_url=resolved_base_url,
            paths=paths,
            daemon_error=None,
        )

    try:
        remote_payload = _collect_remote_snapshot(
            base_url=resolved_base_url,
            timeout_s=timeout_s,
            api_key=api_key,
        )
    except Exception as exc:  # noqa: BLE001
        if mode == "remote":
            raise RuntimeError(str(exc)) from exc
        return _collect_local_snapshot(
            base_url=resolved_base_url,
            paths=paths,
            daemon_error=str(exc),
        )

    return _normalize_remote_snapshot(base_url=resolved_base_url, payload=remote_payload)


def _collect_remote_snapshot(
    *,
    base_url: str,
    timeout_s: float,
    api_key: str | None,
) -> dict[str, Any]:
    with _make_http_client(base_url=base_url, timeout_s=timeout_s, api_key=api_key) as client:
        return _get_json(client, "/api/info")


def _normalize_remote_snapshot(*, base_url: str, payload: dict[str, Any]) -> dict[str, Any]:
    snapshot = dict(payload)
    daemon = snapshot.get("daemon")
    daemon_payload = dict(daemon) if isinstance(daemon, dict) else {}
    daemon_payload["reachable"] = True
    daemon_payload["error"] = None
    snapshot["daemon"] = daemon_payload

    env_payload = snapshot.get("env")
    if isinstance(env_payload, dict):
        snapshot["env"] = redact_env_dict(env_payload)
    else:
        snapshot["env"] = {}

    config_payload = snapshot.get("config")
    if isinstance(config_payload, dict) and "error" not in config_payload:
        snapshot["config"] = redact_config_dict(config_payload)

    pull_defaults = snapshot.get("pull_defaults")
    snapshot["pull_defaults"] = _redact_pull_defaults(pull_defaults)

    snapshot["client"] = {
        "base_url": base_url,
        "api_base_url": f"{base_url}/api",
        "version": CLI_VERSION,
        "source": "remote",
    }
    return snapshot


def _collect_local_snapshot(
    *,
    base_url: str,
    paths: TollamaPaths,
    daemon_error: str | None,
) -> dict[str, Any]:
    config_payload = _collect_redacted_config_payload(paths)
    config_for_defaults = _load_config_or_default(paths)
    return {
        "daemon": {
            "version": None,
            "started_at": None,
            "uptime_seconds": None,
            "host_binding": None,
            "reachable": False,
            "error": daemon_error,
        },
        "paths": {
            "tollama_home": str(paths.base_dir),
            "config_path": str(paths.config_path),
            "config_exists": paths.config_path.exists(),
        },
        "config": config_payload,
        "env": _collect_env_payload(),
        "pull_defaults": _collect_pull_defaults(config_for_defaults),
        "models": {
            "installed": _local_installed_models(paths),
            "loaded": [],
            "available": _local_available_models(),
        },
        "runners": [],
        "client": {
            "base_url": base_url,
            "api_base_url": f"{base_url}/api",
            "version": CLI_VERSION,
            "source": "local",
        },
    }


def _collect_redacted_config_payload(paths: TollamaPaths) -> dict[str, Any] | None:
    config_path = paths.config_path
    if not config_path.exists():
        return None

    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {"error": f"invalid JSON in {config_path}: {exc.msg}"}
    except OSError as exc:
        return {"error": f"unable to read {config_path}: {exc}"}

    if not isinstance(payload, dict):
        return {"error": f"invalid config in {config_path}: top-level JSON must be an object"}
    return redact_config_dict(payload)


def _load_config_or_default(paths: TollamaPaths) -> TollamaConfig:
    try:
        return load_config(paths)
    except ConfigFileError:
        return TollamaConfig()


def _collect_env_payload() -> dict[str, Any]:
    payload = {key: _env_or_none(key) for key in _INFO_ENV_KEYS}
    for key in _TOKEN_ENV_KEYS:
        payload[f"{key}_present"] = _env_or_none(key) is not None
    return redact_env_dict(payload)


def _collect_pull_defaults(config: TollamaConfig) -> dict[str, dict[str, Any]]:
    defaults = resolve_effective_pull_defaults(env=os.environ, config=config)
    return _redact_pull_defaults(defaults)


def _redact_pull_defaults(payload: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(payload, Mapping):
        return {}

    redacted: dict[str, dict[str, Any]] = {}
    for raw_key, detail in payload.items():
        key = str(raw_key)
        if isinstance(detail, Mapping):
            value = detail.get("value")
            if isinstance(value, str) and "proxy" in key:
                value = redact_proxy_url(value)
            redacted[key] = {"value": value, "source": detail.get("source")}
            continue
        redacted[key] = {"value": None, "source": "unknown"}
    return redacted


def _local_installed_models(paths: TollamaPaths) -> list[dict[str, Any]]:
    try:
        manifests = list_installed(paths=paths)
    except Exception:  # noqa: BLE001
        return []

    entries: list[dict[str, Any]] = []
    for manifest in manifests:
        if isinstance(manifest, dict):
            entries.append(_model_from_manifest(manifest))
    return entries


def _model_from_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "name": _str_or_none(manifest.get("name")),
        "family": _str_or_none(manifest.get("family")),
        "digest": _manifest_digest(manifest),
        "size": _manifest_size(manifest),
        "modified_at": _str_or_none(manifest.get("pulled_at") or manifest.get("installed_at")),
    }
    capabilities = manifest.get("capabilities")
    if isinstance(capabilities, dict):
        payload["capabilities"] = capabilities
    else:
        model_name = _str_or_none(manifest.get("name"))
        if model_name is not None:
            try:
                spec = get_model_spec(model_name)
            except KeyError:
                spec = None
            if spec is not None and spec.capabilities is not None:
                payload["capabilities"] = spec.capabilities.model_dump(mode="json")
    return payload


def _local_available_models() -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for spec in list_registry_models():
        entry: dict[str, Any] = {"name": spec.name, "family": spec.family}
        if spec.capabilities is not None:
            entry["capabilities"] = spec.capabilities.model_dump(mode="json")
        entries.append(entry)
    return entries


def _manifest_digest(manifest: dict[str, Any]) -> str | None:
    resolved = manifest.get("resolved")
    if isinstance(resolved, dict):
        digest = _str_or_none(resolved.get("commit_sha"))
        if digest is not None:
            return digest
    return _str_or_none(manifest.get("digest"))


def _manifest_size(manifest: dict[str, Any]) -> int | None:
    size_bytes = manifest.get("size_bytes")
    if isinstance(size_bytes, int) and not isinstance(size_bytes, bool):
        return size_bytes
    size = manifest.get("size")
    if isinstance(size, int) and not isinstance(size, bool):
        return size
    return None


def _get_json(client: httpx.Client, path: str) -> dict[str, Any]:
    response = client.get(path)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError(f"unexpected response payload for {path}")
    return payload


def _make_http_client(*, base_url: str, timeout_s: float, api_key: str | None) -> httpx.Client:
    return httpx.Client(
        base_url=base_url,
        timeout=timeout_s,
        headers=_auth_header(api_key),
    )


def _auth_header(api_key: str | None) -> dict[str, str] | None:
    if api_key is None:
        return None
    token = api_key.strip()
    if not token:
        return None
    return {"Authorization": f"Bearer {token}"}


def _env_or_none(name: str) -> str | None:
    return _str_or_none(os.environ.get(name))


def _str_or_none(value: Any) -> str | None:
    if isinstance(value, str):
        normalized = value.strip()
        return normalized if normalized else None
    return None
