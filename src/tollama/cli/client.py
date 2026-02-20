"""Backward-compatible CLI client exports.

The canonical implementation now lives under :mod:`tollama.client` so MCP and
other adapters can share a single HTTP client.
"""

from tollama.client import (
    AsyncTollamaClient,
    DEFAULT_BASE_URL,
    DEFAULT_DAEMON_HOST,
    DEFAULT_DAEMON_PORT,
    DEFAULT_TIMEOUT_SECONDS,
    DaemonHTTPError,
    DaemonUnreachableError,
    ForecastTimeoutError,
    InvalidRequestError,
    LicenseRequiredError,
    ModelMissingError,
    PermissionDeniedError,
    TollamaClient,
    TollamaClientError,
)

__all__ = [
    "DEFAULT_BASE_URL",
    "DEFAULT_DAEMON_HOST",
    "DEFAULT_DAEMON_PORT",
    "DEFAULT_TIMEOUT_SECONDS",
    "AsyncTollamaClient",
    "DaemonHTTPError",
    "DaemonUnreachableError",
    "ForecastTimeoutError",
    "InvalidRequestError",
    "LicenseRequiredError",
    "ModelMissingError",
    "PermissionDeniedError",
    "TollamaClient",
    "TollamaClientError",
]
