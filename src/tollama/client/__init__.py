"""Shared HTTP client API for CLI, MCP, and external tool adapters."""

from .exceptions import (
    DaemonHTTPError,
    DaemonUnreachableError,
    ForecastTimeoutError,
    InvalidRequestError,
    LicenseRequiredError,
    ModelMissingError,
    PermissionDeniedError,
    TollamaClientError,
)
from .http import (
    DEFAULT_BASE_URL,
    DEFAULT_DAEMON_HOST,
    DEFAULT_DAEMON_PORT,
    DEFAULT_TIMEOUT_SECONDS,
    AsyncTollamaClient,
    TollamaClient,
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
