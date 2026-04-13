"""Tests for PostgreSQL connector identifier safety."""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest

from tollama.connectors.base import ConnectorConfig
from tollama.connectors.postgresql import PostgreSQLConnector


class _FakeComposable:
    def __init__(self, text: str) -> None:
        self.text = text

    def format(self, *args: Any, **kwargs: Any) -> _FakeComposable:
        if args:
            return _FakeComposable(self.text.format(*(str(value) for value in args)))
        return _FakeComposable(
            self.text.format(**{key: str(value) for key, value in kwargs.items()})
        )

    def join(self, items) -> _FakeComposable:  # noqa: ANN001
        return _FakeComposable(self.text.join(str(item) for item in items))

    def __add__(self, other: Any) -> _FakeComposable:
        return _FakeComposable(self.text + str(other))

    def __str__(self) -> str:
        return self.text


class _FakeIdentifier(_FakeComposable):
    def __init__(self, name: str) -> None:
        super().__init__(f'"{name}"')


class _FakeLiteral(_FakeComposable):
    def __init__(self, value: Any) -> None:
        super().__init__(repr(value))


class _FakeSqlModule:
    @staticmethod
    def SQL(text: str) -> _FakeComposable:
        return _FakeComposable(text)

    Identifier = _FakeIdentifier
    Literal = _FakeLiteral


class _FakeCursor:
    def __init__(self, rows: list[tuple[Any, ...]]) -> None:
        self._rows = rows
        self.executed_query: str | None = None
        self.executed_params: list[Any] | None = None

    def __enter__(self) -> _FakeCursor:
        return self

    def __exit__(self, *_args: Any) -> None:
        return None

    def execute(self, query: Any, params: list[Any] | None = None) -> None:
        self.executed_query = str(query)
        self.executed_params = params

    def fetchall(self) -> list[tuple[Any, ...]]:
        return list(self._rows)


class _FakeConnection:
    def __init__(self, cursor: _FakeCursor) -> None:
        self._cursor = cursor

    def cursor(self) -> _FakeCursor:
        return self._cursor


class _FakePool:
    def __init__(self, cursor: _FakeCursor) -> None:
        self._cursor = cursor
        self.closed = False

    def getconn(self) -> _FakeConnection:
        return _FakeConnection(self._cursor)

    def putconn(self, _connection: _FakeConnection) -> None:
        return None

    def closeall(self) -> None:
        self.closed = True


class _CapturingThreadedConnectionPool:
    last_init: tuple[int, int, str] | None = None

    def __init__(self, minconn: int, maxconn: int, dsn: str) -> None:
        type(self).last_init = (minconn, maxconn, dsn)


def _install_fake_psycopg2(monkeypatch) -> None:  # noqa: ANN001
    psycopg2_module = types.ModuleType("psycopg2")
    psycopg2_module.sql = _FakeSqlModule
    pool_module = types.ModuleType("psycopg2.pool")
    pool_module.ThreadedConnectionPool = _CapturingThreadedConnectionPool

    monkeypatch.setitem(sys.modules, "psycopg2", psycopg2_module)
    monkeypatch.setitem(sys.modules, "psycopg2.pool", pool_module)


def test_postgresql_connector_rejects_malicious_identifiers(monkeypatch) -> None:
    _install_fake_psycopg2(monkeypatch)
    connector = PostgreSQLConnector()

    with pytest.raises(ValueError, match="table must be a single-part or schema.table identifier"):
        connector.connect(
            ConnectorConfig(
                backend="postgresql",
                connection_string="postgresql://localhost/tollama",
                params={"table": "metrics; DROP TABLE users; --"},
            )
        )

    with pytest.raises(
        ValueError, match="id_column must be an unquoted single-part SQL identifier"
    ):
        connector.connect(
            ConnectorConfig(
                backend="postgresql",
                connection_string="postgresql://localhost/tollama",
                params={"id_column": 'series_id" OR 1=1 --'},
            )
        )


def test_postgresql_connector_accepts_schema_qualified_table_and_pool_size(monkeypatch) -> None:
    _install_fake_psycopg2(monkeypatch)
    connector = PostgreSQLConnector()

    connector.connect(
        ConnectorConfig(
            backend="postgresql",
            connection_string="postgresql://localhost/tollama",
            params={
                "table": "analytics.timeseries",
                "id_column": "series_id",
                "timestamp_column": "timestamp",
                "value_column": "value",
                "pool_size": 6,
            },
        )
    )

    assert _CapturingThreadedConnectionPool.last_init == (
        1,
        6,
        "postgresql://localhost/tollama",
    )
    assert connector._table == "analytics.timeseries"  # noqa: SLF001
    assert connector._pool_size == 6  # noqa: SLF001


def test_postgresql_query_series_uses_parameterized_values() -> None:
    cursor = _FakeCursor(
        rows=[
            ("series-1", "2025-01-01T00:00:00Z", 1.0),
            ("series-1", "2025-01-02T00:00:00Z", 2.0),
        ]
    )
    connector = PostgreSQLConnector()
    connector._pool = _FakePool(cursor)  # noqa: SLF001
    connector._sql = _FakeSqlModule  # noqa: SLF001
    connector._table = "analytics.timeseries"  # noqa: SLF001
    connector._id_col = "series_id"  # noqa: SLF001
    connector._ts_col = "timestamp"  # noqa: SLF001
    connector._val_col = "value"  # noqa: SLF001

    chunks = connector.query_series(
        series_id="series-1'; DROP TABLE users; --",
        start="2025-01-01T00:00:00Z",
        end="2025-01-31T00:00:00Z",
        limit=5,
    )

    assert len(chunks) == 1
    assert cursor.executed_params == [
        "series-1'; DROP TABLE users; --",
        "2025-01-01T00:00:00Z",
        "2025-01-31T00:00:00Z",
    ]
    assert cursor.executed_query is not None
    assert "%s" in cursor.executed_query
    assert '"analytics"."timeseries"' in cursor.executed_query
    assert "DROP TABLE" not in cursor.executed_query
