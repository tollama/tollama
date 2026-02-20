"""SQLite-backed usage metering for daemon requests."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock

_USAGE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS usage_by_key (
    key_id TEXT PRIMARY KEY,
    request_count INTEGER NOT NULL DEFAULT 0,
    total_inference_ms REAL NOT NULL DEFAULT 0.0,
    series_processed INTEGER NOT NULL DEFAULT 0,
    updated_at TEXT NOT NULL
)
"""

_USAGE_UPSERT_SQL = """
INSERT INTO usage_by_key (
    key_id,
    request_count,
    total_inference_ms,
    series_processed,
    updated_at
) VALUES (?, 1, ?, ?, ?)
ON CONFLICT(key_id) DO UPDATE SET
    request_count = usage_by_key.request_count + 1,
    total_inference_ms = usage_by_key.total_inference_ms + excluded.total_inference_ms,
    series_processed = usage_by_key.series_processed + excluded.series_processed,
    updated_at = excluded.updated_at
"""

_USAGE_SELECT_ALL_SQL = """
SELECT key_id, request_count, total_inference_ms, series_processed, updated_at
FROM usage_by_key
ORDER BY key_id
"""

_USAGE_SELECT_KEY_SQL = """
SELECT key_id, request_count, total_inference_ms, series_processed, updated_at
FROM usage_by_key
WHERE key_id = ?
"""

_USAGE_UNAVAILABLE_HINT = "usage metering is unavailable"


@dataclass(frozen=True, slots=True)
class UsageRecord:
    """Normalized usage row for one API key identifier."""

    key_id: str
    request_count: int
    total_inference_ms: float
    series_processed: int
    updated_at: str

    def to_json(self) -> dict[str, int | float | str]:
        """Render a JSON-serializable row payload."""
        return {
            "key_id": self.key_id,
            "request_count": self.request_count,
            "total_inference_ms": self.total_inference_ms,
            "series_processed": self.series_processed,
            "updated_at": self.updated_at,
        }


class UsageMeter:
    """Thread-safe SQLite usage aggregator keyed by authenticated principal id."""

    def __init__(self, *, db_path: Path) -> None:
        self._db_path = db_path
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        self._init_schema()

    def record_usage(self, *, key_id: str, inference_ms: float, series_processed: int) -> None:
        """Record one successful forecast invocation."""
        normalized_key = key_id.strip() or "anonymous"
        inference_value = max(float(inference_ms), 0.0)
        series_value = max(int(series_processed), 0)
        updated_at = _utc_now_iso()

        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    _USAGE_UPSERT_SQL,
                    (
                        normalized_key,
                        inference_value,
                        series_value,
                        updated_at,
                    ),
                )

    def snapshot(self, *, key_id: str | None = None) -> dict[str, object]:
        """Return usage rows and aggregate totals."""
        records = self._read_records(key_id=key_id)
        summary = {
            "keys": len(records),
            "request_count": sum(item.request_count for item in records),
            "total_inference_ms": round(
                sum(item.total_inference_ms for item in records),
                6,
            ),
            "series_processed": sum(item.series_processed for item in records),
        }
        return {
            "items": [item.to_json() for item in records],
            "summary": summary,
        }

    def _read_records(self, *, key_id: str | None) -> list[UsageRecord]:
        with self._lock:
            with self._connect() as conn:
                if key_id is None:
                    rows = conn.execute(_USAGE_SELECT_ALL_SQL).fetchall()
                else:
                    rows = conn.execute(_USAGE_SELECT_KEY_SQL, (key_id,)).fetchall()

        records: list[UsageRecord] = []
        for row in rows:
            record = _coerce_row(row)
            if record is not None:
                records.append(record)
        return records

    def _init_schema(self) -> None:
        with self._lock:
            with self._connect() as conn:
                conn.execute(_USAGE_TABLE_SQL)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._db_path)
        connection.row_factory = sqlite3.Row
        return connection


def create_usage_meter(*, db_path: Path) -> UsageMeter:
    """Build a usage-meter instance."""
    return UsageMeter(db_path=db_path)


def usage_unavailable_hint() -> str:
    """Return a stable error hint for unavailable usage storage."""
    return _USAGE_UNAVAILABLE_HINT


def _coerce_row(row: sqlite3.Row) -> UsageRecord | None:
    key_id = row["key_id"]
    request_count = row["request_count"]
    total_inference_ms = row["total_inference_ms"]
    series_processed = row["series_processed"]
    updated_at = row["updated_at"]

    if not isinstance(key_id, str) or not key_id:
        return None
    if not isinstance(request_count, int):
        return None
    if not isinstance(series_processed, int):
        return None
    if not isinstance(total_inference_ms, (int, float)):
        return None
    if not isinstance(updated_at, str) or not updated_at:
        return None

    return UsageRecord(
        key_id=key_id,
        request_count=max(request_count, 0),
        total_inference_ms=max(float(total_inference_ms), 0.0),
        series_processed=max(series_processed, 0),
        updated_at=updated_at,
    )


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")
