"""AutoGen tool wrappers backed by tollama shared tool handlers."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from tollama.client import DEFAULT_BASE_URL, DEFAULT_TIMEOUT_SECONDS

from .framework_common import get_agent_tool_specs

_AUTOGEN_IMPORT_HINT = (
    'AutoGen dependency is not installed. Install with: pip install "pyautogen"'
)


def get_autogen_tool_specs(
    *,
    base_url: str = DEFAULT_BASE_URL,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> list[dict[str, Any]]:
    """Return OpenAI-style function tool specs for AutoGen agents."""
    specs = get_agent_tool_specs(base_url=base_url, timeout=timeout)
    return [
        {
            "type": "function",
            "function": {
                "name": spec.name,
                "description": spec.description,
                "parameters": spec.input_schema,
            },
        }
        for spec in specs
    ]


def get_autogen_function_map(
    *,
    base_url: str = DEFAULT_BASE_URL,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> dict[str, Callable[..., dict[str, Any]]]:
    """Return a function map usable by AutoGen executors."""
    specs = get_agent_tool_specs(base_url=base_url, timeout=timeout)
    return {spec.name: spec.handler for spec in specs}


def register_autogen_tools(
    *,
    caller: Any,
    executor: Any,
    base_url: str = DEFAULT_BASE_URL,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Register tollama tool handlers onto AutoGen caller/executor agents."""
    try:
        import autogen
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(_AUTOGEN_IMPORT_HINT) from exc

    register_function = getattr(autogen, "register_function", None)
    if not callable(register_function):  # pragma: no cover - version-dependent
        raise RuntimeError("AutoGen register_function API is unavailable in installed version")

    specs = get_agent_tool_specs(base_url=base_url, timeout=timeout)
    for spec in specs:
        register_function(
            spec.handler,
            caller=caller,
            executor=executor,
            name=spec.name,
            description=spec.description,
        )

    return {
        "tools": get_autogen_tool_specs(base_url=base_url, timeout=timeout),
        "function_map": get_autogen_function_map(base_url=base_url, timeout=timeout),
        "count": len(specs),
    }
