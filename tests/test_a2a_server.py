"""Integration tests for A2A JSON-RPC endpoint behavior."""

from __future__ import annotations

import json
from pathlib import Path
from threading import Event
from typing import Any

from fastapi.testclient import TestClient

from tollama.core.storage import TollamaPaths, install_from_registry
from tollama.daemon.app import create_app


def _paths(monkeypatch, tmp_path: Path) -> TollamaPaths:
    paths = TollamaPaths(base_dir=tmp_path / ".tollama")
    monkeypatch.setenv("TOLLAMA_HOME", str(paths.base_dir))
    return paths


def _forecast_request_payload() -> dict[str, Any]:
    return {
        "model": "mock",
        "horizon": 2,
        "series": [
            {
                "id": "s1",
                "freq": "D",
                "timestamps": ["2025-01-01", "2025-01-02", "2025-01-03"],
                "target": [1.0, 2.0, 3.0],
            }
        ],
        "options": {},
    }


def _generate_request_payload() -> dict[str, Any]:
    return {
        "series": [
            {
                "id": "s1",
                "freq": "D",
                "timestamps": [
                    "2025-01-01",
                    "2025-01-02",
                    "2025-01-03",
                    "2025-01-04",
                    "2025-01-05",
                ],
                "target": [1.0, 2.0, 3.0, 2.5, 3.5],
            }
        ],
        "count": 1,
        "length": 5,
        "seed": 7,
        "method": "statistical",
    }


def _send_message_rpc(
    *,
    request_id: str,
    blocking: bool,
    skill: str = "forecast",
    request_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    resolved_payload = request_payload or _forecast_request_payload()
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "method": "message/send",
        "params": {
            "message": {
                "messageId": "m-user-1",
                "role": "user",
                "parts": [
                    {
                        "mediaType": "application/json",
                        "data": {
                            "skill": skill,
                            "request": resolved_payload,
                        },
                    }
                ],
            },
            "configuration": {"blocking": blocking},
        },
    }


def _parse_sse_events(payload: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    event_name = "message"
    data_lines: list[str] = []
    for line in payload.splitlines():
        if not line:
            if data_lines:
                events.append({"event": event_name, "data": json.loads("\n".join(data_lines))})
            event_name = "message"
            data_lines = []
            continue
        if line.startswith(":") or line.startswith("retry:"):
            continue
        if line.startswith("event:"):
            event_name = line.split(":", 1)[1].strip()
            continue
        if line.startswith("data:"):
            data_lines.append(line.split(":", 1)[1].strip())
    if data_lines:
        events.append({"event": event_name, "data": json.loads("\n".join(data_lines))})
    return events


def test_a2a_message_send_blocking_and_task_get(monkeypatch, tmp_path: Path) -> None:
    paths = _paths(monkeypatch, tmp_path)
    install_from_registry("mock", accept_license=True, paths=paths)

    with TestClient(create_app()) as client:
        send_response = client.post("/a2a", json=_send_message_rpc(request_id="1", blocking=True))

        assert send_response.status_code == 200
        send_body = send_response.json()
        assert send_body["jsonrpc"] == "2.0"
        assert send_body["id"] == "1"

        task = send_body["result"]["task"]
        assert task["status"]["state"] == "completed"
        task_id = task["id"]

        get_response = client.post(
            "/a2a",
            json={
                "jsonrpc": "2.0",
                "id": "2",
                "method": "tasks/get",
                "params": {"id": task_id},
            },
        )

    assert get_response.status_code == 200
    get_body = get_response.json()
    assert get_body["result"]["id"] == task_id
    assert get_body["result"]["status"]["state"] == "completed"


def test_a2a_message_send_supports_generate_skill() -> None:
    with TestClient(create_app()) as client:
        send_response = client.post(
            "/a2a",
            json=_send_message_rpc(
                request_id="gen-1",
                blocking=True,
                skill="generate",
                request_payload=_generate_request_payload(),
            ),
        )

    assert send_response.status_code == 200
    body = send_response.json()
    task = body["result"]["task"]
    assert task["status"]["state"] == "completed"
    artifacts = task.get("artifacts") or []
    assert artifacts
    payload = artifacts[0]["parts"][0]["data"]
    assert payload["method"] == "statistical"
    assert payload["generated"][0]["source_id"] == "s1"


def test_a2a_tasks_query_and_method_errors(monkeypatch, tmp_path: Path) -> None:
    paths = _paths(monkeypatch, tmp_path)
    install_from_registry("mock", accept_license=True, paths=paths)

    with TestClient(create_app()) as client:
        _ = client.post("/a2a", json=_send_message_rpc(request_id="10", blocking=True))

        query_response = client.post(
            "/a2a",
            json={
                "jsonrpc": "2.0",
                "id": "11",
                "method": "tasks/query",
                "params": {"status": "completed", "pageSize": 10},
            },
        )

        missing_method_response = client.post(
            "/a2a",
            json={
                "jsonrpc": "2.0",
                "id": "12",
                "method": "tasks/does-not-exist",
                "params": {},
            },
        )

    assert query_response.status_code == 200
    query_body = query_response.json()
    assert query_body["result"]["totalSize"] >= 1
    assert all(item["status"]["state"] == "completed" for item in query_body["result"]["tasks"])

    assert missing_method_response.status_code == 200
    error_body = missing_method_response.json()
    assert error_body["error"]["code"] == -32601


def test_a2a_cancel_requests_runner_stop_when_family_known(monkeypatch, tmp_path: Path) -> None:
    paths = _paths(monkeypatch, tmp_path)
    install_from_registry("mock", accept_license=True, paths=paths)

    app = create_app()
    started = Event()
    release = Event()
    stopped_families: list[str] = []

    def _slow_dispatch(routed, request):
        started.set()
        release.wait(timeout=2.0)
        return {"ok": True, "operation": routed.operation}

    def _capture_stop(*, family: str | None = None) -> None:
        if family is not None:
            stopped_families.append(family)

    monkeypatch.setattr(app.state.a2a_server._router, "dispatch", _slow_dispatch)
    monkeypatch.setattr(app.state.runner_manager, "stop", _capture_stop)

    with TestClient(app) as client:
        send_response = client.post("/a2a", json=_send_message_rpc(request_id="20", blocking=False))
        assert send_response.status_code == 200
        task_id = send_response.json()["result"]["task"]["id"]

        assert started.wait(timeout=1.0)

        cancel_response = client.post(
            "/a2a",
            json={
                "jsonrpc": "2.0",
                "id": "21",
                "method": "tasks/cancel",
                "params": {"id": task_id},
            },
        )
        assert cancel_response.status_code == 200
        assert cancel_response.json()["result"]["status"]["state"] == "canceled"

        release.set()

        get_response = client.post(
            "/a2a",
            json={
                "jsonrpc": "2.0",
                "id": "22",
                "method": "tasks/get",
                "params": {"id": task_id},
            },
        )

    assert get_response.status_code == 200
    assert get_response.json()["result"]["status"]["state"] == "canceled"
    assert "mock" in stopped_families


def test_a2a_message_stream_returns_status_and_artifact_events(monkeypatch, tmp_path: Path) -> None:
    paths = _paths(monkeypatch, tmp_path)
    install_from_registry("mock", accept_license=True, paths=paths)

    rpc_payload = _send_message_rpc(request_id="stream-1", blocking=False)
    rpc_payload["method"] = "message/stream"
    rpc_payload["params"]["configuration"] = {
        "pollIntervalMs": 25,
        "heartbeatSeconds": 0.1,
    }

    with TestClient(create_app()) as client:
        response = client.post("/a2a", json=rpc_payload)

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")

    events = _parse_sse_events(response.text)
    names = [event["event"] for event in events]
    assert "TaskStatusUpdateEvent" in names
    assert "TaskArtifactUpdateEvent" in names

    status_payloads = [
        event["data"]
        for event in events
        if event["event"] == "TaskStatusUpdateEvent"
    ]
    assert status_payloads
    assert status_payloads[-1]["status"]["state"] == "completed"
