"""
tollama.xai.connectors.protocol — Core types for external data feed connectors.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


class ConnectorResult(BaseModel):
    """Result returned by a DataConnector.fetch() call."""

    domain: str = Field(min_length=1)
    payload: dict[str, Any]
    source_id: str = Field(min_length=1)
    source_type: str = Field(default="unknown")
    fetched_at: str = Field(default_factory=_utc_now_iso)
    freshness_seconds: float | None = Field(default=None, ge=0.0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ConnectorError(BaseModel):
    """Structured error from a failed connector fetch."""

    domain: str = Field(min_length=1)
    source_id: str = Field(default="")
    error_type: str = Field(default="internal")
    message: str = Field(default="")
    retryable: bool = Field(default=False)
    detail: dict[str, Any] = Field(default_factory=dict)


class ConnectorFetchError(Exception):
    """Raised when a DataConnector.fetch() call fails."""

    def __init__(self, error: ConnectorError) -> None:
        self.error = error
        super().__init__(error.message)


@runtime_checkable
class DataConnector(Protocol):
    """Protocol for external data feed connectors."""

    connector_name: str
    domain: str

    def supports(self, identifier: str, context: dict[str, Any]) -> bool: ...

    def fetch(self, identifier: str, context: dict[str, Any]) -> ConnectorResult: ...


@runtime_checkable
class AsyncDataConnector(Protocol):
    """Async protocol for external data feed connectors."""

    connector_name: str
    domain: str

    def supports(self, identifier: str, context: dict[str, Any]) -> bool: ...

    async def fetch(self, identifier: str, context: dict[str, Any]) -> ConnectorResult: ...


__all__ = [
    "AsyncDataConnector",
    "ConnectorError",
    "ConnectorFetchError",
    "ConnectorResult",
    "DataConnector",
]
