"""A2A JSON-RPC endpoint support for tollama daemon."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from threading import Thread
from typing import Any
from uuid import uuid4

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import ValidationError

from tollama.core.registry import get_model_spec, list_registry_models
from tollama.core.storage import list_installed, read_manifest

from .agent_card import AgentCardContext, build_agent_card
from .message_router import A2AMessageRouter, MessageRoutingError, RoutedMessage
from .tasks import (
    TERMINAL_TASK_STATES,
    A2ATaskStore,
    TaskNotCancelableError,
    TaskNotFoundError,
    normalize_state,
)

JSONRPC_VERSION = "2.0"

JSONRPC_PARSE_ERROR = -32700
JSONRPC_INVALID_REQUEST = -32600
JSONRPC_METHOD_NOT_FOUND = -32601
JSONRPC_INVALID_PARAMS = -32602
JSONRPC_INTERNAL_ERROR = -32603

A2A_TASK_NOT_FOUND = -32001
A2A_TASK_NOT_CANCELABLE = -32002
A2A_UNSUPPORTED_OPERATION = -32004

Handler = Callable[[dict[str, Any], Any], dict[str, Any]]


@dataclass(frozen=True, slots=True)
class A2AOperationHandlers:
    """Operation handlers used by message router dispatch."""

    forecast: Handler
    auto_forecast: Handler
    analyze: Handler
    generate: Handler
    compare: Handler
    what_if: Handler
    pipeline: Handler
    recommend: Handler

    def as_mapping(self) -> dict[str, Handler]:
        return {
            "forecast": self.forecast,
            "auto_forecast": self.auto_forecast,
            "analyze": self.analyze,
            "generate": self.generate,
            "compare": self.compare,
            "what_if": self.what_if,
            "pipeline": self.pipeline,
            "recommend": self.recommend,
        }


class A2AServer:
    """JSON-RPC dispatcher + task manager for A2A methods."""

    def __init__(
        self,
        *,
        app: Any,
        package_version: str,
        handlers: A2AOperationHandlers,
    ) -> None:
        self._app = app
        self._package_version = package_version
        self._task_store = A2ATaskStore()
        self._router = A2AMessageRouter(handlers=handlers.as_mapping())

    def agent_card(self, *, request: Request) -> dict[str, object]:
        interface_url = f"{str(request.base_url).rstrip('/')}/a2a"
        return build_agent_card(
            AgentCardContext(
                interface_url=interface_url,
                version=self._package_version,
                require_authentication=self._auth_required(),
                installed_models=self._installed_models(),
                available_models=self._available_models(),
            ),
        )

    def handle_jsonrpc(self, *, payload: Any, request: Request) -> Response:
        if not isinstance(payload, dict):
            return _jsonrpc_error_response(
                request_id=None,
                code=JSONRPC_INVALID_REQUEST,
                message="request body must be a JSON object",
            )

        request_id = payload.get("id")
        if payload.get("jsonrpc") != JSONRPC_VERSION:
            return _jsonrpc_error_response(
                request_id=request_id,
                code=JSONRPC_INVALID_REQUEST,
                message="jsonrpc must be '2.0'",
            )

        method = payload.get("method")
        if not isinstance(method, str) or not method.strip():
            return _jsonrpc_error_response(
                request_id=request_id,
                code=JSONRPC_INVALID_REQUEST,
                message="method must be a non-empty string",
            )

        params = payload.get("params", {})
        if not isinstance(params, dict):
            return _jsonrpc_error_response(
                request_id=request_id,
                code=JSONRPC_INVALID_PARAMS,
                message="params must be an object",
            )

        normalized_method = method.strip()
        try:
            if normalized_method == "message/send":
                result = self._handle_message_send(params=params, request=request)
            elif normalized_method == "tasks/get":
                result = self._handle_tasks_get(params=params)
            elif normalized_method == "tasks/query":
                result = self._handle_tasks_query(params=params)
            elif normalized_method == "tasks/cancel":
                result = self._handle_tasks_cancel(params=params)
            elif normalized_method == "message/stream":
                return self._handle_message_stream(
                    params=params,
                    request=request,
                    request_id=request_id,
                )
            else:
                return _jsonrpc_error_response(
                    request_id=request_id,
                    code=JSONRPC_METHOD_NOT_FOUND,
                    message=f"unsupported method: {normalized_method!r}",
                )
        except TaskNotFoundError as exc:
            return _jsonrpc_error_response(
                request_id=request_id,
                code=A2A_TASK_NOT_FOUND,
                message=str(exc),
            )
        except TaskNotCancelableError as exc:
            return _jsonrpc_error_response(
                request_id=request_id,
                code=A2A_TASK_NOT_CANCELABLE,
                message=str(exc),
            )
        except (MessageRoutingError, ValidationError, ValueError) as exc:
            return _jsonrpc_error_response(
                request_id=request_id,
                code=JSONRPC_INVALID_PARAMS,
                message=str(exc),
            )
        except Exception as exc:  # noqa: BLE001
            return _jsonrpc_error_response(
                request_id=request_id,
                code=JSONRPC_INTERNAL_ERROR,
                message=str(exc),
            )

        return JSONResponse(_jsonrpc_result_payload(request_id=request_id, result=result))

    def _handle_message_send(self, *, params: dict[str, Any], request: Request) -> dict[str, Any]:
        message = params.get("message")
        if not isinstance(message, dict):
            raise ValueError("message/send requires params.message as an object")
        if not isinstance(message.get("parts"), list) or not message["parts"]:
            raise ValueError("message.parts must contain at least one part")

        normalized_message = dict(message)
        normalized_message.setdefault("messageId", uuid4().hex)
        normalized_message.setdefault("role", "user")

        context_id_raw = normalized_message.get("contextId")
        context_id = context_id_raw.strip() if isinstance(context_id_raw, str) else None

        metadata = params.get("metadata")
        metadata_payload = dict(metadata) if isinstance(metadata, dict) else None

        created = self._task_store.create_task(
            context_id=context_id,
            history_message=normalized_message,
            metadata=metadata_payload,
        )
        task_id = created["id"]
        resolved_context_id = created["contextId"]

        configuration = params.get("configuration")
        blocking = bool(configuration.get("blocking")) if isinstance(configuration, dict) else False

        auth_request = _request_proxy(request)
        if blocking:
            self._run_task(
                task_id=task_id,
                context_id=resolved_context_id,
                message=normalized_message,
                auth_request=auth_request,
            )
        else:
            worker = Thread(
                target=self._run_task,
                kwargs={
                    "task_id": task_id,
                    "context_id": resolved_context_id,
                    "message": normalized_message,
                    "auth_request": auth_request,
                },
                daemon=True,
            )
            worker.start()

        task = self._task_store.get_task(task_id, include_artifacts=True, history_length=None)
        return {"task": task}

    def _handle_message_stream(
        self,
        *,
        params: dict[str, Any],
        request: Request,
        request_id: Any,
    ) -> Response:
        configuration_raw = params.get("configuration")
        configuration = dict(configuration_raw) if isinstance(configuration_raw, dict) else {}
        configuration["blocking"] = False

        send_params = dict(params)
        send_params["configuration"] = configuration
        send_result = self._handle_message_send(params=send_params, request=request)
        task = send_result["task"]
        task_id = str(task["id"])
        context_id = str(task["contextId"])

        poll_interval_ms_raw = configuration.get("pollIntervalMs")
        poll_interval_ms = 150
        if poll_interval_ms_raw is not None:
            poll_interval_ms = _optional_non_negative_int(
                poll_interval_ms_raw,
                key="configuration.pollIntervalMs",
            ) or 0
            if poll_interval_ms <= 0:
                raise ValueError("configuration.pollIntervalMs must be a positive integer")
        poll_interval_seconds = max(poll_interval_ms / 1000.0, 0.01)

        heartbeat_raw = configuration.get("heartbeatSeconds")
        heartbeat_seconds = 10.0
        if heartbeat_raw is not None:
            if isinstance(heartbeat_raw, bool) or not isinstance(heartbeat_raw, (int, float)):
                raise ValueError("configuration.heartbeatSeconds must be a positive number")
            if heartbeat_raw <= 0:
                raise ValueError("configuration.heartbeatSeconds must be a positive number")
            heartbeat_seconds = float(heartbeat_raw)

        max_events_raw = configuration.get("maxEvents")
        max_events: int | None = None
        if max_events_raw is not None:
            max_events = _optional_non_negative_int(max_events_raw, key="configuration.maxEvents")
            if max_events is not None and max_events <= 0:
                raise ValueError("configuration.maxEvents must be a positive integer")

        stream_headers = {
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }

        def _lines():
            yield "retry: 3000\n\n"
            emitted = 0
            seen_artifacts = len(task.get("artifacts") or [])
            seen_state = None
            status = task.get("status")
            if isinstance(status, dict):
                seen_state = status.get("state")
                yield _format_sse_event(
                    event="TaskStatusUpdateEvent",
                    data={
                        "requestId": request_id,
                        "taskId": task_id,
                        "contextId": context_id,
                        "status": status,
                    },
                )
                emitted += 1
                if max_events is not None and emitted >= max_events:
                    return

            yield from self._stream_task_updates(
                task_id=task_id,
                context_id=context_id,
                request_id=request_id,
                poll_interval_seconds=poll_interval_seconds,
                heartbeat_seconds=heartbeat_seconds,
                initial_state=seen_state,
                initial_artifact_count=seen_artifacts,
                max_events=None if max_events is None else (max_events - emitted),
            )

        return StreamingResponse(
            _lines(),
            media_type="text/event-stream",
            headers=stream_headers,
        )

    def _handle_tasks_get(self, *, params: dict[str, Any]) -> dict[str, Any]:
        task_id = _required_nonempty_str(params, "id")
        history_length = _optional_non_negative_int(
            params.get("historyLength"),
            key="historyLength",
        )
        return self._task_store.get_task(
            task_id,
            history_length=history_length,
            include_artifacts=True,
        )

    def _handle_tasks_query(self, *, params: dict[str, Any]) -> dict[str, Any]:
        context_id = _optional_nonempty_str(params.get("contextId"))

        status_raw = params.get("status")
        status_value: str | None = None
        if isinstance(status_raw, dict):
            status_value = _optional_nonempty_str(status_raw.get("state"))
        elif isinstance(status_raw, str):
            status_value = status_raw
        elif status_raw is not None:
            raise ValueError("status filter must be a string or object with state")

        if status_value is not None:
            status_value = normalize_state(status_value)

        page_size = _optional_non_negative_int(params.get("pageSize"), key="pageSize")
        if page_size is None:
            page_size = 50

        page_token = _optional_nonempty_str(params.get("pageToken"))
        history_length = _optional_non_negative_int(
            params.get("historyLength"),
            key="historyLength",
        )

        include_artifacts_raw = params.get("includeArtifacts")
        include_artifacts = (
            bool(include_artifacts_raw) if include_artifacts_raw is not None else False
        )

        timestamp_after = _optional_nonempty_str(params.get("statusTimestampAfter"))

        return self._task_store.list_tasks(
            context_id=context_id,
            state=status_value,
            page_size=page_size,
            page_token=page_token,
            history_length=history_length,
            include_artifacts=include_artifacts,
            status_timestamp_after=timestamp_after,
        )

    def _handle_tasks_cancel(self, *, params: dict[str, Any]) -> dict[str, Any]:
        task_id = _required_nonempty_str(params, "id")
        self._task_store.request_cancel(task_id)

        running_family = self._task_store.running_family(task_id)
        if running_family is not None:
            self._stop_runner_family(running_family)

        context_id = self._task_store.context_id(task_id)
        self._task_store.mark_canceled(
            task_id,
            status_message=_status_message(
                context_id=context_id,
                task_id=task_id,
                text="Task canceled",
            ),
        )
        return self._task_store.get_task(task_id, include_artifacts=True, history_length=None)

    def _stream_task_updates(
        self,
        *,
        task_id: str,
        context_id: str,
        request_id: Any,
        poll_interval_seconds: float,
        heartbeat_seconds: float,
        initial_state: str | None,
        initial_artifact_count: int,
        max_events: int | None,
    ):
        last_state = initial_state
        artifact_count = max(initial_artifact_count, 0)
        emitted = 0
        last_emit_at = time.monotonic()

        while True:
            task = self._task_store.get_task(task_id, include_artifacts=True, history_length=None)
            status = task.get("status")
            state = status.get("state") if isinstance(status, dict) else None
            produced_event = False

            if isinstance(status, dict) and state != last_state:
                yield _format_sse_event(
                    event="TaskStatusUpdateEvent",
                    data={
                        "requestId": request_id,
                        "taskId": task_id,
                        "contextId": context_id,
                        "status": status,
                    },
                )
                last_state = state
                produced_event = True
                emitted += 1
                if max_events is not None and emitted >= max_events:
                    return

            artifacts = task.get("artifacts")
            artifact_list = artifacts if isinstance(artifacts, list) else []
            if len(artifact_list) > artifact_count:
                for artifact in artifact_list[artifact_count:]:
                    yield _format_sse_event(
                        event="TaskArtifactUpdateEvent",
                        data={
                            "requestId": request_id,
                            "taskId": task_id,
                            "contextId": context_id,
                            "artifact": artifact,
                        },
                    )
                    produced_event = True
                    emitted += 1
                    if max_events is not None and emitted >= max_events:
                        return
                artifact_count = len(artifact_list)

            now = time.monotonic()
            if produced_event:
                last_emit_at = now

            if isinstance(state, str) and state in TERMINAL_TASK_STATES:
                return

            if now - last_emit_at >= heartbeat_seconds:
                yield _format_sse_comment("keepalive")
                last_emit_at = now

            time.sleep(poll_interval_seconds)

    def _run_task(
        self,
        *,
        task_id: str,
        context_id: str,
        message: dict[str, Any],
        auth_request: Any,
    ) -> None:
        self._task_store.mark_working(
            task_id,
            status_message=_status_message(
                context_id=context_id,
                task_id=task_id,
                text="Task is running",
            ),
        )

        if self._task_store.is_cancel_requested(task_id):
            self._task_store.mark_canceled(
                task_id,
                status_message=_status_message(
                    context_id=context_id,
                    task_id=task_id,
                    text="Task canceled",
                ),
            )
            return

        try:
            routed: RoutedMessage = self._router.resolve_message(message)
            family = _resolve_model_family(routed.model_name)
            self._task_store.set_running_family(task_id, family=family)

            if self._task_store.is_cancel_requested(task_id):
                if family is not None:
                    self._stop_runner_family(family)
                self._task_store.mark_canceled(
                    task_id,
                    status_message=_status_message(
                        context_id=context_id,
                        task_id=task_id,
                        text="Task canceled",
                    ),
                )
                return

            response_payload = self._router.dispatch(routed, request=auth_request)

            if self._task_store.is_cancel_requested(task_id):
                if family is not None:
                    self._stop_runner_family(family)
                self._task_store.mark_canceled(
                    task_id,
                    status_message=_status_message(
                        context_id=context_id,
                        task_id=task_id,
                        text="Task canceled",
                    ),
                )
                return

            self._task_store.mark_completed(
                task_id,
                status_message=_status_message(
                    context_id=context_id,
                    task_id=task_id,
                    text="Task completed",
                ),
                artifacts=[
                    {
                        "artifactId": uuid4().hex,
                        "name": f"{routed.operation}_result",
                        "parts": [
                            {
                                "mediaType": "application/json",
                                "data": response_payload,
                            }
                        ],
                    }
                ],
                history_message=_agent_data_message(
                    context_id=context_id,
                    task_id=task_id,
                    payload=response_payload,
                ),
            )
        except (MessageRoutingError, ValidationError, ValueError) as exc:
            self._task_store.mark_rejected(
                task_id,
                status_message=_status_message(
                    context_id=context_id,
                    task_id=task_id,
                    text=f"Task rejected: {exc}",
                ),
                history_message=_status_message(
                    context_id=context_id,
                    task_id=task_id,
                    text=f"Task rejected: {exc}",
                ),
            )
        except HTTPException as exc:
            detail = str(exc.detail)
            if self._task_store.is_cancel_requested(task_id):
                self._task_store.mark_canceled(
                    task_id,
                    status_message=_status_message(
                        context_id=context_id,
                        task_id=task_id,
                        text="Task canceled",
                    ),
                )
                return

            if exc.status_code == 400:
                self._task_store.mark_rejected(
                    task_id,
                    status_message=_status_message(
                        context_id=context_id,
                        task_id=task_id,
                        text=f"Task rejected: {detail}",
                    ),
                )
                return

            self._task_store.mark_failed(
                task_id,
                status_message=_status_message(
                    context_id=context_id,
                    task_id=task_id,
                    text=f"Task failed: {detail}",
                ),
            )
        except Exception as exc:  # noqa: BLE001
            if self._task_store.is_cancel_requested(task_id):
                self._task_store.mark_canceled(
                    task_id,
                    status_message=_status_message(
                        context_id=context_id,
                        task_id=task_id,
                        text="Task canceled",
                    ),
                )
                return
            self._task_store.mark_failed(
                task_id,
                status_message=_status_message(
                    context_id=context_id,
                    task_id=task_id,
                    text=f"Task failed: {exc}",
                ),
            )

    def _auth_required(self) -> bool:
        provider = getattr(self._app.state, "config_provider", None)
        if provider is None or not hasattr(provider, "get"):
            return False

        try:
            config = provider.get()
        except Exception:  # noqa: BLE001
            return False

        auth = getattr(config, "auth", None)
        api_keys = getattr(auth, "api_keys", None)
        if not isinstance(api_keys, list):
            return False
        return any(isinstance(item, str) and item.strip() for item in api_keys)

    def _installed_models(self) -> tuple[str, ...]:
        names: list[str] = []
        for manifest in list_installed():
            name = manifest.get("name") if isinstance(manifest, dict) else None
            if isinstance(name, str) and name:
                names.append(name)
        return tuple(sorted(set(names)))

    def _available_models(self) -> tuple[str, ...]:
        return tuple(spec.name for spec in list_registry_models())

    def _stop_runner_family(self, family: str) -> None:
        runner_manager = getattr(self._app.state, "runner_manager", None)
        if runner_manager is None or not hasattr(runner_manager, "stop"):
            return
        try:
            runner_manager.stop(family=family)
        except Exception:  # noqa: BLE001
            return


def _format_sse_event(*, event: str, data: dict[str, Any]) -> str:
    return (
        f"event: {event}\n"
        f"data: {json.dumps(data, separators=(',', ':'), sort_keys=True)}\n\n"
    )


def _format_sse_comment(text: str) -> str:
    return f": {text}\n\n"


def _jsonrpc_result_payload(*, request_id: Any, result: dict[str, Any]) -> dict[str, Any]:
    return {
        "jsonrpc": JSONRPC_VERSION,
        "id": request_id,
        "result": result,
    }


def _jsonrpc_error_response(
    *,
    request_id: Any,
    code: int,
    message: str,
    data: Any | None = None,
) -> JSONResponse:
    error_payload: dict[str, Any] = {
        "code": code,
        "message": message,
    }
    if data is not None:
        error_payload["data"] = data

    return JSONResponse(
        {
            "jsonrpc": JSONRPC_VERSION,
            "id": request_id,
            "error": error_payload,
        }
    )


def _required_nonempty_str(payload: dict[str, Any], key: str) -> str:
    raw = payload.get(key)
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"{key} is required and must be a non-empty string")
    return raw.strip()


def _optional_nonempty_str(raw: Any) -> str | None:
    if not isinstance(raw, str):
        return None
    normalized = raw.strip()
    if not normalized:
        return None
    return normalized


def _optional_non_negative_int(raw: Any, *, key: str) -> int | None:
    if raw is None:
        return None
    if not isinstance(raw, int) or isinstance(raw, bool) or raw < 0:
        raise ValueError(f"{key} must be a non-negative integer")
    return raw


def _resolve_model_family(model_name: str | None) -> str | None:
    if model_name is None:
        return None

    try:
        manifest = read_manifest(model_name)
    except ValueError:
        manifest = None

    if isinstance(manifest, dict):
        family = manifest.get("family")
        if isinstance(family, str) and family.strip():
            return family.strip()

    try:
        spec = get_model_spec(model_name)
    except KeyError:
        return None
    return spec.family


def _status_message(*, context_id: str, task_id: str, text: str) -> dict[str, Any]:
    return {
        "messageId": uuid4().hex,
        "contextId": context_id,
        "taskId": task_id,
        "role": "agent",
        "parts": [
            {
                "mediaType": "text/plain",
                "text": text,
            }
        ],
    }


def _agent_data_message(
    *,
    context_id: str,
    task_id: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    return {
        "messageId": uuid4().hex,
        "contextId": context_id,
        "taskId": task_id,
        "role": "agent",
        "parts": [
            {
                "mediaType": "application/json",
                "data": payload,
            }
        ],
    }


class _RequestStateProxy:
    def __init__(self, *, auth_principal: Any) -> None:
        self.auth_principal = auth_principal


class _RequestProxy:
    def __init__(self, *, auth_principal: Any) -> None:
        self.state = _RequestStateProxy(auth_principal=auth_principal)


def _request_proxy(request: Request) -> Any:
    auth_principal = getattr(request.state, "auth_principal", None)
    if auth_principal is None:
        return None
    return _RequestProxy(auth_principal=auth_principal)
