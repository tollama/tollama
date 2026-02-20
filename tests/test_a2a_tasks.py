"""Unit tests for A2A in-memory task store behavior."""

from __future__ import annotations

import pytest

from tollama.a2a.tasks import A2ATaskStore, TaskNotCancelableError


def _sample_message() -> dict[str, object]:
    return {
        "messageId": "m1",
        "role": "user",
        "parts": [{"text": "hello", "mediaType": "text/plain"}],
    }


def test_task_store_create_and_complete_roundtrip() -> None:
    store = A2ATaskStore()
    created = store.create_task(history_message=_sample_message())
    task_id = created["id"]

    assert created["status"]["state"] == "submitted"
    assert created["history"][0]["messageId"] == "m1"

    store.mark_working(task_id)
    store.mark_completed(
        task_id,
        artifacts=[
            {
                "artifactId": "a1",
                "parts": [{"mediaType": "application/json", "data": {"ok": True}}],
            }
        ],
    )

    completed = store.get_task(task_id)
    assert completed["status"]["state"] == "completed"
    assert completed["artifacts"][0]["artifactId"] == "a1"


def test_task_store_query_filters_and_paging() -> None:
    store = A2ATaskStore()
    first = store.create_task(context_id="ctx-a")
    second = store.create_task(context_id="ctx-a")
    third = store.create_task(context_id="ctx-b")

    store.mark_completed(first["id"])
    store.mark_working(second["id"])
    store.mark_failed(third["id"])

    filtered = store.list_tasks(context_id="ctx-a", state="working", page_size=10)
    assert filtered["totalSize"] == 1
    assert filtered["tasks"][0]["id"] == second["id"]

    paged = store.list_tasks(page_size=2)
    assert paged["pageSize"] == 2
    assert len(paged["tasks"]) == 2
    assert paged["nextPageToken"]

    paged_next = store.list_tasks(page_size=2, page_token=paged["nextPageToken"])
    assert len(paged_next["tasks"]) == 1


def test_task_store_cancel_terminal_task_rejected() -> None:
    store = A2ATaskStore()
    created = store.create_task()
    task_id = created["id"]
    store.mark_completed(task_id)

    with pytest.raises(TaskNotCancelableError):
        store.request_cancel(task_id)
