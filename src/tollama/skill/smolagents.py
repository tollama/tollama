"""smolagents tool wrappers backed by tollama shared tool handlers."""

from __future__ import annotations

from typing import Any

from tollama.client import DEFAULT_BASE_URL, DEFAULT_TIMEOUT_SECONDS

from .framework_common import AgentToolSpec, get_agent_tool_specs

_SMOLAGENTS_IMPORT_HINT = (
    'smolagents dependency is not installed. Install with: pip install "smolagents"'
)


def get_smolagents_tools(
    *,
    base_url: str = DEFAULT_BASE_URL,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> list[Any]:
    """Return smolagents ``Tool`` wrappers for all tollama agent tools."""
    try:
        from smolagents import Tool
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(_SMOLAGENTS_IMPORT_HINT) from exc

    specs = get_agent_tool_specs(base_url=base_url, timeout=timeout)
    return [_build_smolagents_tool(Tool=Tool, spec=spec)() for spec in specs]


def _build_smolagents_tool(*, Tool: type[Any], spec: AgentToolSpec) -> type[Any]:
    input_schema = _smolagents_inputs(spec.input_schema)

    class _SmolagentsTool(Tool):
        name = spec.name
        description = spec.description
        inputs: dict[str, dict[str, Any]]
        output_type = "object"

        def forward(self, **kwargs: Any) -> dict[str, Any]:
            return spec.handler(**kwargs)

    _SmolagentsTool.inputs = input_schema
    _SmolagentsTool.__name__ = "".join(part.capitalize() for part in spec.name.split("_")) + "Tool"
    return _SmolagentsTool


def _smolagents_inputs(schema: dict[str, Any]) -> dict[str, dict[str, Any]]:
    properties = schema.get("properties", {})
    required = set(schema.get("required", []))

    inputs: dict[str, dict[str, Any]] = {}
    for name, raw_property in properties.items():
        property_schema = raw_property if isinstance(raw_property, dict) else {}
        item: dict[str, Any] = {
            "type": property_schema.get("type", "any"),
            "required": name in required,
            "description": property_schema.get("description", ""),
        }
        enum_values = property_schema.get("enum")
        if isinstance(enum_values, list) and enum_values:
            item["enum"] = enum_values
        inputs[name] = item

    return inputs
