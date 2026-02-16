"""Loaded model tracking and keep-alive parsing for daemon endpoints."""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

_KEEP_ALIVE_PATTERN = re.compile(r"^(-?\d+(?:\.\d+)?)([smhd]?)$", flags=re.IGNORECASE)
_KEEP_ALIVE_MULTIPLIER_SECONDS = {
    "": 1.0,
    "s": 1.0,
    "m": 60.0,
    "h": 3600.0,
    "d": 86400.0,
}


@dataclass(frozen=True)
class LoadedModel:
    """Loaded model metadata tracked by runner identity."""

    name: str
    model: str
    family: str
    runner: str
    expires_at: datetime | None
    device: dict[str, Any] | None = None


@dataclass(frozen=True)
class KeepAlivePolicy:
    """Derived keep-alive decision for a request."""

    unload_immediately: bool
    expires_at: datetime | None


class LoadedModelTracker:
    """In-memory loaded model tracker keyed by runner and model name."""

    def __init__(self) -> None:
        self._entries: dict[tuple[str, str], LoadedModel] = {}
        self._lock = threading.Lock()

    def upsert(
        self,
        *,
        name: str,
        model: str,
        family: str,
        runner: str,
        expires_at: datetime | None,
        device: dict[str, Any] | None = None,
    ) -> None:
        with self._lock:
            self._entries[(runner, model)] = LoadedModel(
                name=name,
                model=model,
                family=family,
                runner=runner,
                expires_at=expires_at,
                device=device,
            )

    def unload_runner(self, runner: str) -> None:
        with self._lock:
            self._entries = {
                key: value for key, value in self._entries.items() if value.runner != runner
            }

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()

    def list_models(self) -> list[LoadedModel]:
        with self._lock:
            models = list(self._entries.values())
        return sorted(models, key=lambda item: item.model)

    def expired_runners(self, now: datetime) -> set[str]:
        with self._lock:
            return {
                entry.runner
                for entry in self._entries.values()
                if entry.expires_at is not None and entry.expires_at <= now
            }


def parse_keep_alive(value: str | int | float | None, *, now: datetime) -> KeepAlivePolicy:
    """Parse keep_alive values from API payloads."""
    if value is None:
        return KeepAlivePolicy(unload_immediately=False, expires_at=None)

    if isinstance(value, (int, float)):
        seconds = float(value)
        return _seconds_to_policy(seconds=seconds, now=now)

    raw = value.strip().lower()
    if not raw:
        raise ValueError("invalid keep_alive: value is empty")

    matched = _KEEP_ALIVE_PATTERN.fullmatch(raw)
    if matched is None:
        raise ValueError(f"invalid keep_alive: {value!r}")

    amount = float(matched.group(1))
    unit = matched.group(2).lower()
    seconds = amount * _KEEP_ALIVE_MULTIPLIER_SECONDS[unit]
    return _seconds_to_policy(seconds=seconds, now=now)


def to_utc_iso(value: datetime | None) -> str | None:
    """Convert a datetime to RFC3339-like UTC timestamp with Z suffix."""
    if value is None:
        return None
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _seconds_to_policy(*, seconds: float, now: datetime) -> KeepAlivePolicy:
    if seconds < 0:
        return KeepAlivePolicy(unload_immediately=False, expires_at=None)
    if seconds == 0:
        return KeepAlivePolicy(unload_immediately=True, expires_at=now)
    return KeepAlivePolicy(unload_immediately=False, expires_at=now + timedelta(seconds=seconds))
