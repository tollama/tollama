"""Unit tests for outbound A2A client helpers."""

from __future__ import annotations

import httpx
import pytest

from tollama.a2a.client import A2AClient, A2ARpcError


def test_a2a_client_discover_and_send_message_roundtrip() -> None:
    def _handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET" and request.url.path == "/.well-known/agent-card.json":
            return httpx.Response(
                status_code=200,
                json={
                    "name": "agent",
                    "supportedInterfaces": [
                        {
                            "url": "http://agent.test/a2a",
                            "protocolBinding": "JSONRPC",
                            "protocolVersion": "1.0",
                        }
                    ],
                },
            )

        if request.method == "POST" and request.url.path == "/a2a":
            payload = request.read().decode("utf-8")
            assert "message/send" in payload
            return httpx.Response(
                status_code=200,
                json={
                    "jsonrpc": "2.0",
                    "id": "1",
                    "result": {"task": {"id": "task-1"}},
                },
            )

        return httpx.Response(status_code=404, json={"error": "not found"})

    client = A2AClient(transport=httpx.MockTransport(_handler))
    card = client.discover(base_url="http://agent.test")
    assert card["name"] == "agent"

    result = client.send_message(
        base_url="http://agent.test",
        message={"messageId": "m1", "role": "user", "parts": [{"text": "hi"}]},
        request_id="1",
    )
    assert result["task"]["id"] == "task-1"


def test_a2a_client_poll_task_until_completed() -> None:
    call_count = {"tasks_get": 0}

    def _handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path == "/a2a":
            payload = request.read().decode("utf-8")
            if "tasks/get" in payload:
                call_count["tasks_get"] += 1
                state = "working" if call_count["tasks_get"] == 1 else "completed"
                return httpx.Response(
                    status_code=200,
                    json={
                        "jsonrpc": "2.0",
                        "id": "x",
                        "result": {
                            "id": "task-1",
                            "status": {"state": state},
                        },
                    },
                )

        return httpx.Response(status_code=404, json={"error": "not found"})

    client = A2AClient(transport=httpx.MockTransport(_handler))
    task = client.poll_task(base_url="http://agent.test", task_id="task-1", timeout_seconds=2.0)
    assert task["status"]["state"] == "completed"
    assert call_count["tasks_get"] >= 2


def test_a2a_client_raises_rpc_error() -> None:
    def _handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code=200,
            json={
                "jsonrpc": "2.0",
                "id": "1",
                "error": {"code": -32001, "message": "task not found"},
            },
        )

    client = A2AClient(transport=httpx.MockTransport(_handler))
    with pytest.raises(A2ARpcError) as exc_info:
        client.get_task(base_url="http://agent.test", task_id="missing")

    assert exc_info.value.code == -32001
