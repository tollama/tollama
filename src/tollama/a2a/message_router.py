"""Route inbound A2A messages to tollama forecasting operations."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

A2A_OPERATION_NAMES = frozenset(
    {
        "forecast",
        "auto_forecast",
        "analyze",
        "generate",
        "compare",
        "what_if",
        "pipeline",
        "recommend",
    }
)


class MessageRoutingError(ValueError):
    """Raised when one inbound A2A message cannot be mapped to a tollama operation."""


@dataclass(frozen=True, slots=True)
class RoutedMessage:
    """Resolved operation + payload extracted from one A2A message."""

    operation: str
    request_payload: dict[str, Any]
    model_name: str | None


Handler = Callable[[dict[str, Any], Any], dict[str, Any]]


class A2AMessageRouter:
    """Resolve and dispatch one A2A message against configured handlers."""

    def __init__(self, *, handlers: dict[str, Handler]) -> None:
        missing = sorted(A2A_OPERATION_NAMES - set(handlers))
        if missing:
            raise ValueError(f"missing handlers for operations: {', '.join(missing)}")
        self._handlers = dict(handlers)

    def resolve_message(self, message: dict[str, Any]) -> RoutedMessage:
        payload = _extract_payload_from_message(message)
        operation = _resolve_operation(payload)

        request_payload_raw = payload.get("request", payload)
        if not isinstance(request_payload_raw, dict):
            raise MessageRoutingError("message payload 'request' must be an object")
        request_payload = dict(request_payload_raw)

        for key in ("skill", "operation", "tool", "name"):
            request_payload.pop(key, None)

        model_name = _resolve_model_name(operation=operation, payload=request_payload)
        return RoutedMessage(
            operation=operation,
            request_payload=request_payload,
            model_name=model_name,
        )

    def dispatch(self, routed: RoutedMessage, *, request: Any) -> dict[str, Any]:
        handler = self._handlers.get(routed.operation)
        if handler is None:
            raise MessageRoutingError(f"unsupported skill/operation: {routed.operation!r}")
        return handler(routed.request_payload, request)


def _extract_payload_from_message(message: dict[str, Any]) -> dict[str, Any]:
    parts = message.get("parts")
    if not isinstance(parts, list) or not parts:
        raise MessageRoutingError("message.parts must contain at least one part")

    for part in parts:
        if not isinstance(part, dict):
            continue

        raw_data = part.get("data")
        if isinstance(raw_data, dict):
            return dict(raw_data)

        raw_text = part.get("text")
        if isinstance(raw_text, str):
            parsed = _parse_text_part(raw_text)
            if parsed is not None:
                return parsed

    raise MessageRoutingError(
        "message parts must include JSON object data (part.data or JSON part.text)",
    )


def _parse_text_part(value: str) -> dict[str, Any] | None:
    text = value.strip()
    if not text:
        return None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None
    if isinstance(parsed, dict):
        return dict(parsed)
    return None


def _resolve_operation(payload: dict[str, Any]) -> str:
    for key in ("skill", "operation", "tool", "name"):
        raw_value = payload.get(key)
        if isinstance(raw_value, str):
            resolved = _normalize_operation(raw_value)
            if resolved is not None:
                return resolved

    inferred = _infer_operation_from_payload(payload)
    if inferred is not None:
        return inferred

    raise MessageRoutingError(
        "unable to infer operation; include one of skill/operation/tool with a supported value",
    )


def _normalize_operation(value: str) -> str | None:
    normalized = value.strip().lower().replace("-", "_")
    if not normalized:
        return None

    aliases = {
        "tollama_forecast": "forecast",
        "forecast": "forecast",
        "tollama_auto_forecast": "auto_forecast",
        "auto_forecast": "auto_forecast",
        "autoforecast": "auto_forecast",
        "tollama_analyze": "analyze",
        "analyze": "analyze",
        "tollama_generate": "generate",
        "generate": "generate",
        "tollama_compare": "compare",
        "compare": "compare",
        "tollama_what_if": "what_if",
        "what_if": "what_if",
        "whatif": "what_if",
        "tollama_pipeline": "pipeline",
        "pipeline": "pipeline",
        "tollama_recommend": "recommend",
        "recommend": "recommend",
    }
    return aliases.get(normalized)


def _infer_operation_from_payload(payload: dict[str, Any]) -> str | None:
    request_payload = payload.get("request")
    if isinstance(request_payload, dict):
        candidate = dict(request_payload)
    else:
        candidate = dict(payload)

    if "scenarios" in candidate:
        return "what_if"
    if "models" in candidate:
        return "compare"
    if "count" in candidate or "variation" in candidate or candidate.get("method") == "statistical":
        return "generate"
    if "pull_if_missing" in candidate or "recommend_top_k" in candidate:
        return "pipeline"
    if "allow_fallback" in candidate:
        return "auto_forecast"
    if "model" in candidate and "horizon" in candidate and "series" in candidate:
        return "forecast"
    if "horizon" in candidate and "has_future_covariates" in candidate:
        return "recommend"
    if "series" in candidate:
        return "analyze"
    return None


def _resolve_model_name(*, operation: str, payload: dict[str, Any]) -> str | None:
    if operation in {"forecast", "auto_forecast", "what_if", "pipeline"}:
        model = payload.get("model")
        if isinstance(model, str) and model.strip():
            return model.strip()
    if operation == "compare":
        models = payload.get("models")
        if isinstance(models, list):
            for item in models:
                if isinstance(item, str) and item.strip():
                    return item.strip()
    return None
