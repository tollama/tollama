"""In-memory task store for A2A task lifecycle management."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from datetime import UTC, datetime
from threading import Lock
from typing import Any
from uuid import uuid4

TERMINAL_TASK_STATES = frozenset({"completed", "failed", "canceled", "rejected"})
INTERRUPTED_TASK_STATES = frozenset({"input-required", "auth-required"})


class TaskStoreError(RuntimeError):
    """Base task store error."""


class TaskNotFoundError(TaskStoreError):
    """Raised when one task does not exist."""


class TaskNotCancelableError(TaskStoreError):
    """Raised when one task is already terminal and cannot be canceled."""


@dataclass(slots=True)
class _TaskRecord:
    id: str
    context_id: str
    state: str
    status_message: dict[str, Any] | None
    status_timestamp: str
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    history: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] | None = None
    cancel_requested: bool = False
    running_family: str | None = None


class A2ATaskStore:
    """Thread-safe in-memory task persistence for A2A operations."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._tasks: dict[str, _TaskRecord] = {}

    def create_task(
        self,
        *,
        context_id: str | None = None,
        history_message: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        status_message: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            task_id = uuid4().hex
            resolved_context_id = context_id or uuid4().hex
            now = _utc_iso_now()
            history: list[dict[str, Any]] = []
            if history_message is not None:
                history.append(deepcopy(history_message))

            record = _TaskRecord(
                id=task_id,
                context_id=resolved_context_id,
                state="submitted",
                status_message=deepcopy(status_message),
                status_timestamp=now,
                history=history,
                metadata=deepcopy(metadata),
            )
            self._tasks[task_id] = record
            return _task_payload(record, include_artifacts=True, history_length=None)

    def get_task(
        self,
        task_id: str,
        *,
        history_length: int | None = None,
        include_artifacts: bool = True,
    ) -> dict[str, Any]:
        with self._lock:
            record = self._tasks.get(task_id)
            if record is None:
                raise TaskNotFoundError(f"task {task_id!r} was not found")
            return _task_payload(
                record,
                include_artifacts=include_artifacts,
                history_length=history_length,
            )

    def list_tasks(
        self,
        *,
        context_id: str | None = None,
        state: str | None = None,
        page_size: int = 50,
        page_token: str | None = None,
        history_length: int | None = None,
        include_artifacts: bool = False,
        status_timestamp_after: str | None = None,
    ) -> dict[str, Any]:
        if page_size <= 0:
            raise ValueError("page_size must be greater than zero")
        if page_size > 100:
            raise ValueError("page_size must not exceed 100")

        normalized_state = normalize_state(state) if state is not None else None
        min_timestamp = (
            _parse_iso8601(status_timestamp_after)
            if status_timestamp_after is not None
            else None
        )
        offset = _parse_page_token(page_token)

        with self._lock:
            records = list(self._tasks.values())

        records.sort(key=lambda item: (item.status_timestamp, item.id), reverse=True)

        filtered: list[_TaskRecord] = []
        for record in records:
            if context_id is not None and record.context_id != context_id:
                continue
            if normalized_state is not None and record.state != normalized_state:
                continue
            if min_timestamp is not None:
                timestamp = _parse_iso8601(record.status_timestamp)
                if timestamp < min_timestamp:
                    continue
            filtered.append(record)

        total_size = len(filtered)
        start = min(offset, total_size)
        end = min(start + page_size, total_size)

        tasks_payload = [
            _task_payload(
                record,
                include_artifacts=include_artifacts,
                history_length=history_length,
            )
            for record in filtered[start:end]
        ]

        next_page_token = str(end) if end < total_size else ""
        return {
            "tasks": tasks_payload,
            "nextPageToken": next_page_token,
            "pageSize": page_size,
            "totalSize": total_size,
        }

    def mark_working(self, task_id: str, *, status_message: dict[str, Any] | None = None) -> None:
        self._transition(task_id, state="working", status_message=status_message)

    def mark_completed(
        self,
        task_id: str,
        *,
        status_message: dict[str, Any] | None = None,
        artifacts: list[dict[str, Any]] | None = None,
        history_message: dict[str, Any] | None = None,
    ) -> None:
        with self._lock:
            record = self._require_record(task_id)
            if record.state in TERMINAL_TASK_STATES:
                return
            record.state = "completed"
            record.status_timestamp = _utc_iso_now()
            record.status_message = deepcopy(status_message)
            if artifacts is not None:
                record.artifacts = deepcopy(artifacts)
            if history_message is not None:
                record.history.append(deepcopy(history_message))

    def mark_failed(
        self,
        task_id: str,
        *,
        status_message: dict[str, Any] | None = None,
        history_message: dict[str, Any] | None = None,
    ) -> None:
        with self._lock:
            record = self._require_record(task_id)
            if record.state in TERMINAL_TASK_STATES:
                return
            record.state = "failed"
            record.status_timestamp = _utc_iso_now()
            record.status_message = deepcopy(status_message)
            if history_message is not None:
                record.history.append(deepcopy(history_message))

    def mark_rejected(
        self,
        task_id: str,
        *,
        status_message: dict[str, Any] | None = None,
        history_message: dict[str, Any] | None = None,
    ) -> None:
        with self._lock:
            record = self._require_record(task_id)
            if record.state in TERMINAL_TASK_STATES:
                return
            record.state = "rejected"
            record.status_timestamp = _utc_iso_now()
            record.status_message = deepcopy(status_message)
            if history_message is not None:
                record.history.append(deepcopy(history_message))

    def mark_canceled(
        self,
        task_id: str,
        *,
        status_message: dict[str, Any] | None = None,
        history_message: dict[str, Any] | None = None,
    ) -> None:
        with self._lock:
            record = self._require_record(task_id)
            if record.state in TERMINAL_TASK_STATES:
                return
            record.state = "canceled"
            record.status_timestamp = _utc_iso_now()
            record.status_message = deepcopy(status_message)
            if history_message is not None:
                record.history.append(deepcopy(history_message))

    def request_cancel(self, task_id: str) -> dict[str, Any]:
        with self._lock:
            record = self._require_record(task_id)
            if record.state in TERMINAL_TASK_STATES:
                raise TaskNotCancelableError(
                    "task "
                    f"{task_id!r} is in terminal state {record.state!r} "
                    "and cannot be canceled",
                )
            record.cancel_requested = True
            return _task_payload(record, include_artifacts=True, history_length=None)

    def is_cancel_requested(self, task_id: str) -> bool:
        with self._lock:
            record = self._require_record(task_id)
            return record.cancel_requested

    def set_running_family(self, task_id: str, *, family: str | None) -> None:
        if family is None:
            return
        normalized = family.strip()
        if not normalized:
            return
        with self._lock:
            record = self._require_record(task_id)
            record.running_family = normalized

    def running_family(self, task_id: str) -> str | None:
        with self._lock:
            record = self._require_record(task_id)
            return record.running_family

    def state(self, task_id: str) -> str:
        with self._lock:
            record = self._require_record(task_id)
            return record.state

    def context_id(self, task_id: str) -> str:
        with self._lock:
            record = self._require_record(task_id)
            return record.context_id

    def _transition(
        self,
        task_id: str,
        *,
        state: str,
        status_message: dict[str, Any] | None,
    ) -> None:
        normalized_state = normalize_state(state)
        with self._lock:
            record = self._require_record(task_id)
            if record.state in TERMINAL_TASK_STATES:
                return
            record.state = normalized_state
            record.status_timestamp = _utc_iso_now()
            record.status_message = deepcopy(status_message)

    def _require_record(self, task_id: str) -> _TaskRecord:
        record = self._tasks.get(task_id)
        if record is None:
            raise TaskNotFoundError(f"task {task_id!r} was not found")
        return record


def _task_payload(
    record: _TaskRecord,
    *,
    include_artifacts: bool,
    history_length: int | None,
) -> dict[str, Any]:
    if history_length is not None and history_length < 0:
        raise ValueError("history_length must be >= 0")

    if history_length is None:
        history = deepcopy(record.history)
    elif history_length == 0:
        history = []
    else:
        history = deepcopy(record.history[-history_length:])

    status: dict[str, Any] = {
        "state": record.state,
        "timestamp": record.status_timestamp,
    }
    if record.status_message is not None:
        status["message"] = deepcopy(record.status_message)

    payload: dict[str, Any] = {
        "id": record.id,
        "contextId": record.context_id,
        "status": status,
        "history": history,
    }
    payload["artifacts"] = deepcopy(record.artifacts) if include_artifacts else []
    if record.metadata is not None:
        payload["metadata"] = deepcopy(record.metadata)
    return payload


def normalize_state(value: str) -> str:
    """Normalize state names from user/API forms to canonical A2A values."""
    normalized = value.strip().lower().replace("_", "-")
    if not normalized:
        raise ValueError("task state must not be empty")

    mapping = {
        "task-state-submitted": "submitted",
        "task-state-working": "working",
        "task-state-completed": "completed",
        "task-state-failed": "failed",
        "task-state-canceled": "canceled",
        "task-state-input-required": "input-required",
        "task-state-rejected": "rejected",
        "task-state-auth-required": "auth-required",
        "task-state-unspecified": "submitted",
        "task-state-unspecfied": "submitted",
        "task-state-unknown": "submitted",
        "task-state": "submitted",
        "task-state-": "submitted",
        "task-state-unspecified-state": "submitted",
        "task-state-unspecified-status": "submitted",
    }
    if normalized in mapping:
        return mapping[normalized]

    if normalized.startswith("task-state-"):
        normalized = normalized.removeprefix("task-state-")

    valid = {
        "submitted",
        "working",
        "completed",
        "failed",
        "canceled",
        "input-required",
        "rejected",
        "auth-required",
    }
    if normalized not in valid:
        raise ValueError(f"unsupported task state filter: {value!r}")
    return normalized


def _utc_iso_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _parse_iso8601(value: str) -> datetime:
    normalized = value.strip()
    if not normalized:
        raise ValueError("timestamp filter must not be empty")
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ValueError(f"invalid ISO 8601 timestamp: {value!r}") from exc
    if parsed.tzinfo is None:
        raise ValueError("timestamp must include timezone offset")
    return parsed.astimezone(UTC)


def _parse_page_token(page_token: str | None) -> int:
    if page_token is None:
        return 0
    token = page_token.strip()
    if not token:
        return 0
    if not token.isdigit():
        raise ValueError("page_token must be an integer offset")
    return int(token)
