"""Helpers for redacting sensitive diagnostics values."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from urllib.parse import urlsplit, urlunsplit

_SENSITIVE_KEY_TOKENS = (
    "token",
    "secret",
    "password",
    "authorization",
    "api_key",
    "apikey",
)
_KNOWN_TOKEN_ENV_VARS = frozenset(
    {
        "TOLLAMA_HF_TOKEN",
        "HF_TOKEN",
        "HUGGINGFACE_HUB_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
        "HF_HUB_TOKEN",
    },
)


def redact_proxy_url(url: str) -> str:
    """Redact credentials from a proxy URL."""
    normalized = url.strip()
    if not normalized:
        return normalized

    parsed = urlsplit(normalized)
    if parsed.netloc and "@" in parsed.netloc:
        userinfo, hostinfo = parsed.netloc.rsplit("@", 1)
        if not userinfo:
            return normalized
        redacted_userinfo = "***:***" if ":" in userinfo else "***"
        return urlunsplit(
            (
                parsed.scheme,
                f"{redacted_userinfo}@{hostinfo}",
                parsed.path,
                parsed.query,
                parsed.fragment,
            ),
        )

    if "://" not in normalized and "@" in normalized:
        userinfo, hostinfo = normalized.rsplit("@", 1)
        if userinfo:
            redacted_userinfo = "***:***" if ":" in userinfo else "***"
            return f"{redacted_userinfo}@{hostinfo}"

    return normalized


def redact_config_dict(payload: dict[str, Any]) -> dict[str, Any]:
    """Redact sensitive values from config-like mappings."""
    return _redact_mapping(payload, drop_known_token_env_vars=False)


def redact_env_dict(payload: dict[str, Any]) -> dict[str, Any]:
    """Redact sensitive values from env-like mappings and drop known token vars."""
    return _redact_mapping(payload, drop_known_token_env_vars=True)


def _redact_mapping(
    payload: Mapping[str, Any],
    *,
    drop_known_token_env_vars: bool,
) -> dict[str, Any]:
    redacted: dict[str, Any] = {}
    for raw_key, value in payload.items():
        key = str(raw_key)
        if drop_known_token_env_vars and key in _KNOWN_TOKEN_ENV_VARS:
            continue
        if _looks_sensitive_key(key) and not key.lower().endswith("_present"):
            redacted[key] = "***"
            continue
        redacted[key] = _redact_value(
            key=key,
            value=value,
            drop_known_token_env_vars=drop_known_token_env_vars,
        )
    return redacted


def _redact_value(
    *,
    key: str,
    value: Any,
    drop_known_token_env_vars: bool,
) -> Any:
    if isinstance(value, Mapping):
        return _redact_mapping(value, drop_known_token_env_vars=drop_known_token_env_vars)
    if isinstance(value, list):
        return [
            _redact_value(
                key=key,
                value=item,
                drop_known_token_env_vars=drop_known_token_env_vars,
            )
            for item in value
        ]
    if isinstance(value, str) and "proxy" in key.lower():
        return redact_proxy_url(value)
    return value


def _looks_sensitive_key(key: str) -> bool:
    lowered = key.lower()
    return any(token in lowered for token in _SENSITIVE_KEY_TOKENS)

