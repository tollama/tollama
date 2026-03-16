"""Data connectors for ingesting time series from external sources.

Supported backends:
- PostgreSQL / TimescaleDB
- InfluxDB 2.x
- S3 (CSV/Parquet objects)
- Kafka (streaming ingestion)
"""

from __future__ import annotations

from .base import ConnectorConfig, DataConnector, SeriesChunk
from .registry import get_connector, list_connectors, register_connector

__all__ = [
    "ConnectorConfig",
    "DataConnector",
    "SeriesChunk",
    "get_connector",
    "list_connectors",
    "register_connector",
]
