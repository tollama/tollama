"""Tests for data connector infrastructure."""

from __future__ import annotations

import pytest

from tollama.connectors.base import ConnectorConfig, SeriesChunk
from tollama.connectors.registry import list_connectors


def test_series_chunk_creation() -> None:
    chunk = SeriesChunk(
        id="test_series",
        timestamps=["2025-01-01", "2025-01-02"],
        values=[1.0, 2.0],
        freq="D",
    )
    assert chunk.id == "test_series"
    assert len(chunk.timestamps) == 2
    assert len(chunk.values) == 2
    assert chunk.freq == "D"


def test_connector_config() -> None:
    config = ConnectorConfig(
        backend="postgresql",
        connection_string="postgresql://localhost/test",
        params={"table": "metrics"},
    )
    assert config.backend == "postgresql"
    assert config.params["table"] == "metrics"


def test_list_connectors_returns_list() -> None:
    # Should not raise even without optional deps installed
    result = list_connectors()
    assert isinstance(result, list)


def test_get_connector_unknown_backend() -> None:
    from tollama.connectors.registry import get_connector

    with pytest.raises(ValueError, match="unknown connector backend"):
        get_connector(ConnectorConfig(backend="nonexistent_db"))
