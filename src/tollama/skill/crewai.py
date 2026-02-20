"""CrewAI tool wrappers backed by tollama shared tool handlers."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, create_model

from tollama.client import DEFAULT_BASE_URL, DEFAULT_TIMEOUT_SECONDS

from .framework_common import AgentToolSpec, get_agent_tool_specs

_CREWAI_IMPORT_HINT = (
    'CrewAI dependency is not installed. Install with: pip install "crewai"'
)


def get_crewai_tools(
    *,
    base_url: str = DEFAULT_BASE_URL,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> list[Any]:
    """Return CrewAI ``BaseTool`` wrappers for all tollama agent tools."""
    try:
        from crewai.tools import BaseTool
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(_CREWAI_IMPORT_HINT) from exc

    specs = get_agent_tool_specs(base_url=base_url, timeout=timeout)
    return [_build_crewai_tool(BaseTool=BaseTool, spec=spec)() for spec in specs]


def _build_crewai_tool(*, BaseTool: type[Any], spec: AgentToolSpec) -> type[Any]:
    args_model = _build_args_schema(spec)

    class _CrewAITool(BaseTool):
        name: str = spec.name
        description: str = spec.description
        args_schema: type[BaseModel]

        def _run(self, **kwargs: Any) -> dict[str, Any]:
            return spec.handler(**kwargs)

    _CrewAITool.args_schema = args_model
    _CrewAITool.__name__ = "".join(part.capitalize() for part in spec.name.split("_")) + "Tool"
    return _CrewAITool


def _build_args_schema(spec: AgentToolSpec) -> type[BaseModel]:
    properties = spec.input_schema.get("properties", {})
    required = set(spec.input_schema.get("required", []))

    field_defs: dict[str, tuple[type[Any], Any]] = {}
    for name, schema in properties.items():
        schema_dict = schema if isinstance(schema, dict) else {}
        annotation = _annotation_from_schema(schema_dict)
        default = ... if name in required and "default" not in schema_dict else schema_dict.get(
            "default",
            None,
        )
        field_defs[name] = (annotation, default)

    args_model = create_model(
        f"{_spec_name_prefix(spec)}CrewAIArgs",
        __config__=ConfigDict(extra="forbid", strict=True),
        **field_defs,
    )
    return args_model


def _spec_name_prefix(spec: AgentToolSpec) -> str:
    return "".join(part.capitalize() for part in spec.name.split("_"))


def _annotation_from_schema(schema: dict[str, Any]) -> type[Any]:
    schema_type = schema.get("type")
    if isinstance(schema.get("enum"), list) and schema.get("enum"):
        enum_values = schema["enum"]
        if all(isinstance(item, str) for item in enum_values):
            return str
        if all(isinstance(item, bool) for item in enum_values):
            return bool
        if all(isinstance(item, int) for item in enum_values):
            return int
        if all(isinstance(item, (int, float)) for item in enum_values):
            return float

    if schema_type == "boolean":
        return bool
    if schema_type == "integer":
        return int
    if schema_type == "number":
        return float
    if schema_type == "string":
        return str
    if schema_type == "array":
        return list[Any]
    if schema_type == "object":
        return dict[str, Any]
    return Any
