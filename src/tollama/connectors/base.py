"""Abstract base for tollama data connectors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ConnectorConfig:
    """Configuration for a data connector."""

    backend: str
    connection_string: str | None = None
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SeriesChunk:
    """A chunk of time series data returned by a connector.

    Designed to be compatible with ``tollama.core.schemas.SeriesInput``.
    """

    id: str
    timestamps: list[str]
    values: list[float]
    freq: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class DataConnector(ABC):
    """Base class for all tollama data connectors."""

    name: str = "base"

    @abstractmethod
    def connect(self, config: ConnectorConfig) -> None:
        """Establish connection to the data source."""

    @abstractmethod
    def query_series(
        self,
        *,
        series_id: str | None = None,
        start: str | None = None,
        end: str | None = None,
        limit: int | None = None,
        **kwargs: Any,
    ) -> list[SeriesChunk]:
        """Query time series data from the source."""

    def stream_series(
        self,
        *,
        series_id: str | None = None,
        batch_size: int = 1000,
        **kwargs: Any,
    ) -> Iterator[SeriesChunk]:
        """Stream time series data in batches.  Default: single query."""
        yield from self.query_series(series_id=series_id, limit=batch_size, **kwargs)

    @abstractmethod
    def list_series(self) -> list[str]:
        """List available series IDs in the source."""

    def close(self) -> None:
        """Clean up resources.  Override if needed."""

    def __enter__(self) -> DataConnector:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def health_check(self) -> bool:
        """Return True if the connection is healthy."""
        try:
            self.list_series()
            return True
        except Exception:
            return False
