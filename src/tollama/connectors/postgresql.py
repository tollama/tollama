"""PostgreSQL / TimescaleDB data connector for tollama."""

from __future__ import annotations

import logging
import re
from contextlib import contextmanager
from typing import Any

from .base import ConnectorConfig, DataConnector, SeriesChunk
from .registry import register_connector

logger = logging.getLogger(__name__)
_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_DEFAULT_POOL_SIZE = 4


class PostgreSQLConnector(DataConnector):
    """Connector for PostgreSQL and TimescaleDB time series tables."""

    name = "postgresql"

    def __init__(self) -> None:
        self._pool: Any = None
        self._table: str = "timeseries"
        self._ts_col: str = "timestamp"
        self._val_col: str = "value"
        self._id_col: str = "series_id"
        self._pool_size: int = _DEFAULT_POOL_SIZE

    def connect(self, config: ConnectorConfig) -> None:
        try:
            from psycopg2 import sql
            from psycopg2.pool import ThreadedConnectionPool
        except ImportError as exc:
            raise ImportError(
                "psycopg2 is required for PostgreSQL connector. "
                "Install with: pip install psycopg2-binary"
            ) from exc

        table = str(config.params.get("table", self._table))
        ts_col = str(config.params.get("timestamp_column", self._ts_col))
        val_col = str(config.params.get("value_column", self._val_col))
        id_col = str(config.params.get("id_column", self._id_col))
        pool_size = _resolve_pool_size(config.params.get("pool_size", _DEFAULT_POOL_SIZE))

        _validate_column_identifier(ts_col, field_name="timestamp_column")
        _validate_column_identifier(val_col, field_name="value_column")
        _validate_column_identifier(id_col, field_name="id_column")
        _validate_table_identifier(table, field_name="table")

        self._pool = ThreadedConnectionPool(1, pool_size, config.connection_string)
        self._table = table
        self._ts_col = ts_col
        self._val_col = val_col
        self._id_col = id_col
        self._pool_size = pool_size
        self._sql = sql
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
        del kwargs
        if self._pool is None:
            raise RuntimeError("not connected")

        conditions: list[Any] = []
        params: list[Any] = []

        if series_id is not None:
            conditions.append(self._sql.SQL("{} = %s").format(_column_sql(self._sql, self._id_col)))
            params.append(series_id)
        if start is not None:
            conditions.append(
                self._sql.SQL("{} >= %s").format(_column_sql(self._sql, self._ts_col))
            )
            params.append(start)
        if end is not None:
            conditions.append(
                self._sql.SQL("{} <= %s").format(_column_sql(self._sql, self._ts_col))
            )
            params.append(end)

        where = self._sql.SQL(" AND ").join(conditions) if conditions else self._sql.SQL("TRUE")
        query = self._sql.SQL(
            "SELECT {id_col}, {ts_col}, {val_col} "
            "FROM {table} WHERE {where} "
            "ORDER BY {id_col}, {ts_col}"
        ).format(
            id_col=_column_sql(self._sql, self._id_col),
            ts_col=_column_sql(self._sql, self._ts_col),
            val_col=_column_sql(self._sql, self._val_col),
            table=_table_sql(self._sql, self._table),
            where=where,
        )
        if limit is not None:
            query += self._sql.SQL(" LIMIT {}").format(self._sql.Literal(int(limit)))

        with self._cursor() as cur:
            cur.execute(query, params)
            rows = cur.fetchall()

        return _rows_to_chunks(rows)

    def list_series(self) -> list[str]:
        if self._pool is None:
            raise RuntimeError("not connected")
        query = self._sql.SQL("SELECT DISTINCT {id_col} FROM {table} ORDER BY 1").format(
            id_col=_column_sql(self._sql, self._id_col),
            table=_table_sql(self._sql, self._table),
        )
        with self._cursor() as cur:
            cur.execute(query)
            return [row[0] for row in cur.fetchall()]

    def close(self) -> None:
        if self._pool is not None:
            self._pool.closeall()
            self._pool = None

    @contextmanager
    def _cursor(self):
        if self._pool is None:
            raise RuntimeError("not connected")
        conn = self._pool.getconn()
        try:
            with conn.cursor() as cur:
                yield cur
        finally:
            self._pool.putconn(conn)


def _resolve_pool_size(value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError("pool_size must be a positive integer")
    try:
        resolved = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("pool_size must be a positive integer") from exc
    if resolved < 1:
        raise ValueError("pool_size must be a positive integer")
    return resolved


def _validate_column_identifier(name: str, *, field_name: str) -> None:
    if not _IDENTIFIER_PATTERN.fullmatch(name):
        raise ValueError(f"{field_name} must be an unquoted single-part SQL identifier: {name!r}")


def _validate_table_identifier(name: str, *, field_name: str) -> None:
    parts = name.split(".")
    if not 1 <= len(parts) <= 2:
        raise ValueError(f"{field_name} must be a single-part or schema.table identifier: {name!r}")
    for part in parts:
        if not _IDENTIFIER_PATTERN.fullmatch(part):
            raise ValueError(
                f"{field_name} must be a single-part or schema.table identifier: {name!r}"
            )


def _column_sql(sql_module: Any, name: str) -> Any:
    return sql_module.Identifier(name)


def _table_sql(sql_module: Any, name: str) -> Any:
    return sql_module.SQL(".").join(sql_module.Identifier(part) for part in name.split("."))


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
