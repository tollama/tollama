"""PostgreSQL / TimescaleDB data connector for tollama."""

from __future__ import annotations

import logging
from typing import Any

from .base import ConnectorConfig, DataConnector, SeriesChunk
from .registry import register_connector

logger = logging.getLogger(__name__)


class PostgreSQLConnector(DataConnector):
    """Connector for PostgreSQL and TimescaleDB time series tables."""

    name = "postgresql"

    def __init__(self) -> None:
        self._conn: Any = None
        self._table: str = "timeseries"
        self._ts_col: str = "timestamp"
        self._val_col: str = "value"
        self._id_col: str = "series_id"

    def connect(self, config: ConnectorConfig) -> None:
        try:
            import psycopg2
        except ImportError as exc:
            raise ImportError(
                "psycopg2 is required for PostgreSQL connector. "
                "Install with: pip install psycopg2-binary"
            ) from exc

        self._conn = psycopg2.connect(config.connection_string)
        self._table = config.params.get("table", self._table)
        self._ts_col = config.params.get("timestamp_column", self._ts_col)
        self._val_col = config.params.get("value_column", self._val_col)
        self._id_col = config.params.get("id_column", self._id_col)
        logger.info("connected to PostgreSQL")

    def query_series(
        self,
        *,
        series_id: str | None = None,
        start: str | None = None,
        end: str | None = None,
        limit: int | None = None,
        **kwargs: Any,
    ) -> list[SeriesChunk]:
        if self._conn is None:
            raise RuntimeError("not connected")

        conditions: list[str] = []
        params: list[Any] = []

        if series_id is not None:
            conditions.append(f"{self._id_col} = %s")
            params.append(series_id)
        if start is not None:
            conditions.append(f"{self._ts_col} >= %s")
            params.append(start)
        if end is not None:
            conditions.append(f"{self._ts_col} <= %s")
            params.append(end)

        where = " AND ".join(conditions) if conditions else "TRUE"
        query = (
            f"SELECT {self._id_col}, {self._ts_col}, {self._val_col} "
            f"FROM {self._table} WHERE {where} "
            f"ORDER BY {self._id_col}, {self._ts_col}"
        )
        if limit is not None:
            query += f" LIMIT {int(limit)}"

        with self._conn.cursor() as cur:
            cur.execute(query, params)
            rows = cur.fetchall()

        return _rows_to_chunks(rows)

    def list_series(self) -> list[str]:
        if self._conn is None:
            raise RuntimeError("not connected")
        with self._conn.cursor() as cur:
            cur.execute(f"SELECT DISTINCT {self._id_col} FROM {self._table} ORDER BY 1")
            return [row[0] for row in cur.fetchall()]

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None


def _rows_to_chunks(rows: list[tuple[Any, ...]]) -> list[SeriesChunk]:
    """Group (id, timestamp, value) rows into SeriesChunk objects."""
    by_id: dict[str, tuple[list[str], list[float]]] = {}
    for row in rows:
        sid, ts, val = str(row[0]), str(row[1]), float(row[2])
        if sid not in by_id:
            by_id[sid] = ([], [])
        by_id[sid][0].append(ts)
        by_id[sid][1].append(val)

    return [
        SeriesChunk(id=sid, timestamps=timestamps, values=values)
        for sid, (timestamps, values) in by_id.items()
    ]


register_connector("postgresql", PostgreSQLConnector)
register_connector("timescaledb", PostgreSQLConnector)
