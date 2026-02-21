"""Typed client-side exception hierarchy for tollama HTTP calls."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class ErrorMetadata:
    """Structured metadata for mapping errors across interfaces."""

    category: str
    exit_code: int


class TollamaClientError(RuntimeError):
    """Base error for tollama client operations."""

    metadata = ErrorMetadata(category="INTERNAL_ERROR", exit_code=10)

    def __init__(
        self,
        *,
        action: str,
        detail: str,
        status_code: int | None = None,
        hint: str | None = None,
    ) -> None:
        message = f"{action} failed"
        if status_code is not None:
            message = f"{message} with HTTP {status_code}"
        if detail:
            message = f"{message}: {detail}"
        super().__init__(message)
        self.action = action
        self.detail = detail
        self.status_code = status_code
        self.hint = hint

    @property
    def exit_code(self) -> int:
        return self.metadata.exit_code

    @property
    def category(self) -> str:
        return self.metadata.category


class DaemonHTTPError(TollamaClientError):
    """Generic daemon HTTP status error."""


class InvalidRequestError(DaemonHTTPError):
    """Invalid client request payload or parameters."""

    metadata = ErrorMetadata(category="INVALID_REQUEST", exit_code=2)


class DaemonUnreachableError(TollamaClientError):
    """Daemon endpoint could not be reached."""

    metadata = ErrorMetadata(category="DAEMON_UNREACHABLE", exit_code=3)


class ModelMissingError(DaemonHTTPError):
    """Requested model is not installed or not found."""

    metadata = ErrorMetadata(category="MODEL_MISSING", exit_code=4)


class PermissionDeniedError(DaemonHTTPError):
    """Permission denied or unauthorized action."""

    metadata = ErrorMetadata(category="PERMISSION_DENIED", exit_code=5)


class LicenseRequiredError(PermissionDeniedError):
    """Model action requires explicit license acceptance."""

    metadata = ErrorMetadata(category="LICENSE_REQUIRED", exit_code=5)


class ForecastTimeoutError(TollamaClientError):
    """Forecast or daemon request timed out."""

    metadata = ErrorMetadata(category="TIMEOUT", exit_code=6)
