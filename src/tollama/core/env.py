"""Central registry and typed accessors for tollama-owned environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})


@dataclass(frozen=True)
class EnvVar:
    """One registered environment variable."""

    name: str
    owner: str
    description: str


REGISTERED_ENV_VARS: tuple[EnvVar, ...] = (
    EnvVar("TOLLAMA_ALLOW_REMOTE_DATA_URL", "daemon", "Allow remote data_url sources."),
    EnvVar("TOLLAMA_API_KEYS", "daemon", "Inline API keys for auth backend."),
    EnvVar("TOLLAMA_AWS_SECRET_NAME", "daemon", "AWS secret name for API key lookup."),
    EnvVar("TOLLAMA_CACHE_TTL_FINANCIAL", "xai", "Financial connector cache TTL."),
    EnvVar("TOLLAMA_CACHE_TTL_NEWS", "xai", "News connector cache TTL."),
    EnvVar("TOLLAMA_CONNECTOR_MAX_RETRIES", "xai", "Connector retry count."),
    EnvVar("TOLLAMA_CONNECTOR_RETRY_BASE_DELAY", "xai", "Connector retry base delay."),
    EnvVar("TOLLAMA_CONNECTOR_TIMEOUT", "xai", "Connector timeout seconds."),
    EnvVar("TOLLAMA_CORS_ORIGINS", "daemon", "Allowed CORS origins."),
    EnvVar("TOLLAMA_DASHBOARD", "daemon", "Enable dashboard endpoints."),
    EnvVar(
        "TOLLAMA_DASHBOARD_REQUIRE_AUTH",
        "daemon",
        "Require auth for dashboard assets and views.",
    ),
    EnvVar("TOLLAMA_DOCS_PUBLIC", "daemon", "Expose docs without auth."),
    EnvVar("TOLLAMA_EFFECTIVE_HOST_BINDING", "daemon", "Effective host:port binding."),
    EnvVar("TOLLAMA_FORECAST_TIMEOUT_SECONDS", "daemon", "Default forecast timeout."),
    EnvVar("TOLLAMA_HF_TOKEN", "core", "Hugging Face access token."),
    EnvVar("TOLLAMA_HOME", "core", "State directory override."),
    EnvVar("TOLLAMA_HOST", "daemon", "Daemon host binding."),
    EnvVar("TOLLAMA_LOG_LEVEL", "daemon", "Daemon log level."),
    EnvVar("TOLLAMA_MAX_REQUEST_BODY_MB", "daemon", "Max HTTP request body size."),
    EnvVar("TOLLAMA_PORT", "daemon", "Daemon port binding."),
    EnvVar("TOLLAMA_RATE_LIMIT_BACKEND", "daemon", "Rate limiter backend."),
    EnvVar("TOLLAMA_RATE_LIMIT_BURST", "daemon", "Burst allowance."),
    EnvVar("TOLLAMA_RATE_LIMIT_PER_MINUTE", "daemon", "Requests per minute."),
    EnvVar("TOLLAMA_RBAC_POLICY", "daemon", "RBAC policy file."),
    EnvVar("TOLLAMA_REDIS_URL", "daemon", "Redis URL for distributed rate limit."),
    EnvVar("TOLLAMA_ROUTING_MANIFEST", "core", "Routing manifest override."),
    EnvVar("TOLLAMA_SECRETS_BACKEND", "daemon", "Secrets backend selection."),
    EnvVar("TOLLAMA_SECRETS_CACHE_TTL", "daemon", "Secrets cache TTL."),
    EnvVar("TOLLAMA_SECRETS_FILE", "daemon", "Secrets file path."),
    EnvVar("TOLLAMA_USE_LIVE_CONNECTORS", "xai", "Enable live XAI connectors."),
)

REGISTERED_ENV_VAR_NAMES = frozenset(item.name for item in REGISTERED_ENV_VARS)


def env_or_none(name: str) -> str | None:
    """Return one env var value, normalized to ``None`` when empty."""
    raw = os.environ.get(name)
    if raw is None:
        return None
    value = raw.strip()
    return value or None


def env_flag(name: str, *, default: bool) -> bool:
    """Parse a truthy/falsy flag from the environment."""
    value = env_or_none(name)
    if value is None:
        return default
    return value.lower() in _TRUE_VALUES


def env_float(name: str, *, default: float, minimum: float | None = None) -> float:
    """Parse a float from the environment with optional floor."""
    value = env_or_none(name)
    if value is None:
        return default
    try:
        resolved = float(value)
    except ValueError:
        return default
    if minimum is not None and resolved < minimum:
        return minimum
    return resolved


def env_int(name: str, *, default: int, minimum: int | None = None) -> int:
    """Parse an integer from the environment with optional floor."""
    value = env_or_none(name)
    if value is None:
        return default
    try:
        resolved = int(value)
    except ValueError:
        return default
    if minimum is not None and resolved < minimum:
        return minimum
    return resolved


def registered_env_map() -> dict[str, dict[str, Any]]:
    """Return the registry as a JSON-friendly mapping for diagnostics/tests."""
    return {
        item.name: {"owner": item.owner, "description": item.description}
        for item in REGISTERED_ENV_VARS
    }


__all__ = [
    "REGISTERED_ENV_VARS",
    "REGISTERED_ENV_VAR_NAMES",
    "EnvVar",
    "env_flag",
    "env_float",
    "env_int",
    "env_or_none",
    "registered_env_map",
]
