"""Optional API key authentication helpers for daemon routes."""

from __future__ import annotations

import hashlib
import hmac
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import HTTPException, Request

from tollama.core.config import ConfigFileError, TollamaConfig
from tollama.core.storage import TollamaPaths

_AUTH_HEADER = "Authorization"
_AUTH_SCHEME = "Bearer"
_WWW_AUTHENTICATE = "Bearer"


@dataclass(frozen=True)
class AuthPrincipal:
    """Authenticated caller context derived from an API key."""

    key_id: str


def require_api_key(request: Request) -> AuthPrincipal | None:
    """Authenticate request if API keys are configured."""
    config = _load_config(request)
    configured_keys = _configured_api_keys(config)
    if not configured_keys:
        request.state.auth_principal = None
        return None

    token = _extract_bearer_token(request.headers.get(_AUTH_HEADER))
    if token is None:
        raise _unauthorized("missing bearer token")

    if not _matches_any_key(token, configured_keys):
        raise _unauthorized("invalid api key")

    principal = AuthPrincipal(key_id=derive_key_id(token))
    request.state.auth_principal = principal
    return principal


def derive_key_id(api_key: str) -> str:
    """Derive a stable non-secret identifier for one API key."""
    digest = hashlib.sha256(api_key.encode("utf-8")).hexdigest()
    return f"k_{digest[:16]}"


def current_key_id(request: Request, *, default: str = "anonymous") -> str:
    """Return the authenticated key id stored on request state."""
    principal = getattr(request.state, "auth_principal", None)
    if isinstance(principal, AuthPrincipal):
        return principal.key_id
    return default


def _load_config(request: Request) -> TollamaConfig:
    provider = getattr(request.app.state, "config_provider", None)
    if provider is None or not hasattr(provider, "get"):
        return TollamaConfig()
    try:
        config = provider.get()
    except ConfigFileError:
        recovered_keys = _recover_auth_keys_from_raw_config()
        if recovered_keys:
            return TollamaConfig.model_validate({"auth": {"api_keys": recovered_keys}})
        return TollamaConfig()
    if isinstance(config, TollamaConfig):
        return config
    try:
        return TollamaConfig.model_validate(config)
    except Exception:
        recovered_keys = _recover_auth_keys_from_raw_config()
        if recovered_keys:
            return TollamaConfig.model_validate({"auth": {"api_keys": recovered_keys}})
        return TollamaConfig()


def _configured_api_keys(config: TollamaConfig) -> tuple[str, ...]:
    keys: list[str] = []
    for raw in config.auth.api_keys:
        normalized = raw.strip()
        if normalized:
            keys.append(normalized)
    return tuple(keys)


def _extract_bearer_token(header: str | None) -> str | None:
    if header is None:
        return None
    normalized = header.strip()
    if not normalized:
        return None
    scheme, _, token = normalized.partition(" ")
    if scheme != _AUTH_SCHEME:
        raise _unauthorized("authorization scheme must be Bearer")
    token = token.strip()
    if not token:
        return None
    return token


def _matches_any_key(token: str, configured_keys: tuple[str, ...]) -> bool:
    token_bytes = token.encode("utf-8")
    for key in configured_keys:
        if hmac.compare_digest(token_bytes, key.encode("utf-8")):
            return True
    return False


def _unauthorized(detail: str) -> HTTPException:
    return HTTPException(
        status_code=401,
        detail=detail,
        headers={"WWW-Authenticate": _WWW_AUTHENTICATE},
    )


def _recover_auth_keys_from_raw_config() -> list[str]:
    config_path = TollamaPaths.default().config_path
    payload = _load_json_object(config_path)
    if payload is None:
        return []

    auth_raw = payload.get("auth")
    if not isinstance(auth_raw, dict):
        return []

    api_keys = auth_raw.get("api_keys")
    if not isinstance(api_keys, list):
        return []

    recovered: list[str] = []
    for value in api_keys:
        if not isinstance(value, str):
            continue
        normalized = value.strip()
        if normalized:
            recovered.append(normalized)
    return recovered


def _load_json_object(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    return payload
