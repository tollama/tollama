"""SSE event bus helpers for daemon real-time streams."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from queue import Empty, Full, Queue
from threading import Lock
from typing import Any
from uuid import uuid4

_CLOSE_SENTINEL = object()


@dataclass(frozen=True, slots=True)
class EventRecord:
    """One normalized event emitted through the in-process stream bus."""

    event_id: str
    sequence: int
    event: str
    timestamp: str
    data: dict[str, Any]


@dataclass(slots=True)
class EventSubscription:
    """Per-client queue subscription to the daemon event bus."""

    token: int
    key_id: str
    event_types: frozenset[str] | None
    queue: Queue[Any]


class EventStream:
    """Thread-safe pub/sub stream for SSE event delivery."""

    def __init__(self, *, max_queue_size: int = 256) -> None:
        if max_queue_size <= 0:
            raise ValueError("max_queue_size must be greater than zero")
        self._max_queue_size = max_queue_size
        self._lock = Lock()
        self._sequence = 0
        self._next_token = 1
        self._subscriptions: dict[int, EventSubscription] = {}

    def subscribe(
        self,
        *,
        key_id: str,
        event_types: set[str] | None = None,
    ) -> EventSubscription:
        normalized_key = key_id.strip() if key_id.strip() else "anonymous"
        normalized_events: frozenset[str] | None = None
        if event_types is not None:
            normalized_events = frozenset(
                item.strip()
                for item in event_types
                if isinstance(item, str) and item.strip()
            )
            if not normalized_events:
                normalized_events = None

        with self._lock:
            token = self._next_token
            self._next_token += 1
            subscription = EventSubscription(
                token=token,
                key_id=normalized_key,
                event_types=normalized_events,
                queue=Queue(maxsize=self._max_queue_size),
            )
            self._subscriptions[token] = subscription
            return subscription

    def unsubscribe(self, subscription: EventSubscription) -> None:
        with self._lock:
            existing = self._subscriptions.pop(subscription.token, None)
        if existing is None:
            return
        try:
            existing.queue.put_nowait(_CLOSE_SENTINEL)
        except Full:
            self._drop_one(existing.queue)
            try:
                existing.queue.put_nowait(_CLOSE_SENTINEL)
            except Full:
                return

    def publish(
        self,
        *,
        key_id: str | None,
        event: str,
        data: dict[str, Any],
    ) -> EventRecord:
        if not event.strip():
            raise ValueError("event must be a non-empty string")

        with self._lock:
            self._sequence += 1
            sequence = self._sequence
            subscribers = list(self._subscriptions.values())

        record = EventRecord(
            event_id=uuid4().hex,
            sequence=sequence,
            event=event.strip(),
            timestamp=_utc_now_iso(),
            data=dict(data),
        )

        for subscription in subscribers:
            if key_id is not None and subscription.key_id != key_id:
                continue
            if (
                subscription.event_types is not None
                and record.event not in subscription.event_types
            ):
                continue
            self._enqueue(subscription.queue, record)

        return record

    def iter_sse_lines(
        self,
        *,
        subscription: EventSubscription,
        heartbeat_seconds: float = 15.0,
        max_events: int | None = None,
    ):
        if heartbeat_seconds <= 0:
            raise ValueError("heartbeat_seconds must be greater than zero")
        if max_events is not None and max_events <= 0:
            raise ValueError("max_events must be greater than zero when provided")

        yield "retry: 3000\n\n"

        emitted = 0
        while True:
            try:
                item = subscription.queue.get(timeout=heartbeat_seconds)
            except Empty:
                yield format_sse_comment("keepalive")
                continue

            if item is _CLOSE_SENTINEL:
                break
            if not isinstance(item, EventRecord):
                continue

            yield format_sse_event(
                event=item.event,
                data=event_payload(item),
                event_id=item.event_id,
            )
            emitted += 1
            if max_events is not None and emitted >= max_events:
                break

    def _enqueue(self, queue: Queue[Any], record: EventRecord) -> None:
        try:
            queue.put_nowait(record)
        except Full:
            self._drop_one(queue)
            try:
                queue.put_nowait(record)
            except Full:
                return

    @staticmethod
    def _drop_one(queue: Queue[Any]) -> None:
        try:
            queue.get_nowait()
        except Empty:
            return


def event_payload(record: EventRecord) -> dict[str, Any]:
    return {
        "event_id": record.event_id,
        "sequence": record.sequence,
        "timestamp": record.timestamp,
        "event": record.event,
        "data": record.data,
    }


def format_sse_event(*, event: str, data: dict[str, Any], event_id: str | None = None) -> str:
    lines: list[str] = []
    if event_id is not None and event_id.strip():
        lines.append(f"id: {event_id.strip()}")
    lines.append(f"event: {event}")
    serialized = json.dumps(data, separators=(",", ":"), sort_keys=True)
    lines.append(f"data: {serialized}")
    return "\n".join(lines) + "\n\n"


def format_sse_comment(text: str = "keepalive") -> str:
    return f": {text}\n\n"


def _utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
