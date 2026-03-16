"""Structured JSON logging and request correlation for the tollama daemon."""

from __future__ import annotations

import contextvars
import json
import logging
import time
import uuid
from typing import Any

# ---------------------------------------------------------------------------
# Request context — propagated via contextvars
# ---------------------------------------------------------------------------

_request_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "tollama_request_id", default=None,
)


def get_request_id() -> str | None:
    """Return the current request's correlation ID."""
    return _request_id_var.get()


def set_request_id(request_id: str | None = None) -> str:
    """Set (or generate) a request ID for the current context."""
    rid = request_id or uuid.uuid4().hex[:16]
    _request_id_var.set(rid)
    return rid


def clear_request_id() -> None:
    """Clear the request ID from the current context."""
    _request_id_var.set(None)


# ---------------------------------------------------------------------------
# JSON log formatter
# ---------------------------------------------------------------------------


class StructuredJsonFormatter(logging.Formatter):
    """Emit log records as single-line JSON objects.

    Each record includes: timestamp, level, logger, message, request_id
    (when available), and any ``extra`` fields attached to the record.
    """

    def format(self, record: logging.LogRecord) -> str:
        entry: dict[str, Any] = {
            "ts": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }

        request_id = get_request_id()
        if request_id is not None:
            entry["request_id"] = request_id

        # Propagate structured extras (family, model, duration_ms, etc.)
        for key in ("family", "model", "duration_ms", "method", "status_code", "path"):
            value = getattr(record, key, None)
            if value is not None:
                entry[key] = value

        if record.exc_info and record.exc_info[1] is not None:
            entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(entry, default=str, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------


def configure_structured_logging(*, level: str = "INFO", json_output: bool = True) -> None:
    """Configure the root logger for structured JSON output.

    Args:
        level: Root log level (e.g. "INFO", "DEBUG").
        json_output: If False, use a plain text format (useful for development).
    """
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Remove existing handlers to avoid double-logging
    for handler in root.handlers[:]:
        root.removeHandler(handler)

    handler = logging.StreamHandler()
    if json_output:
        handler.setFormatter(StructuredJsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")
        )
    root.addHandler(handler)


# ---------------------------------------------------------------------------
# Request timing context manager
# ---------------------------------------------------------------------------


class RequestTimer:
    """Simple timing context that records elapsed milliseconds."""

    def __init__(self) -> None:
        self._start: float = 0.0
        self.duration_ms: float = 0.0

    def __enter__(self) -> RequestTimer:
        self._start = time.monotonic()
        return self

    def __exit__(self, *args: object) -> None:
        self.duration_ms = (time.monotonic() - self._start) * 1000
