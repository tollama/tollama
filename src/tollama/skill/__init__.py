"""Skill integrations for external agent frameworks."""

from __future__ import annotations

_LANGCHAIN_EXPORTS = {
    "TollamaAutoForecastTool",
    "TollamaAnalyzeTool",
    "TollamaCompareTool",
    "TollamaForecastTool",
    "TollamaHealthTool",
    "TollamaModelsTool",
    "TollamaPipelineTool",
    "TollamaWhatIfTool",
    "TollamaRecommendTool",
    "get_tollama_tools",
}
_CREWAI_EXPORTS = {"get_crewai_tools"}
_AUTOGEN_EXPORTS = {"get_autogen_function_map", "get_autogen_tool_specs", "register_autogen_tools"}
_SMOLAGENTS_EXPORTS = {"get_smolagents_tools"}

__all__ = sorted(_LANGCHAIN_EXPORTS | _CREWAI_EXPORTS | _AUTOGEN_EXPORTS | _SMOLAGENTS_EXPORTS)


def __getattr__(name: str):
    if name in _LANGCHAIN_EXPORTS:
        from .langchain import (
            TollamaAnalyzeTool,
            TollamaAutoForecastTool,
            TollamaCompareTool,
            TollamaForecastTool,
            TollamaHealthTool,
            TollamaModelsTool,
            TollamaPipelineTool,
            TollamaRecommendTool,
            TollamaWhatIfTool,
            get_tollama_tools,
        )

        exports = {
            "TollamaAutoForecastTool": TollamaAutoForecastTool,
            "TollamaAnalyzeTool": TollamaAnalyzeTool,
            "TollamaCompareTool": TollamaCompareTool,
            "TollamaForecastTool": TollamaForecastTool,
            "TollamaHealthTool": TollamaHealthTool,
            "TollamaModelsTool": TollamaModelsTool,
            "TollamaPipelineTool": TollamaPipelineTool,
            "TollamaWhatIfTool": TollamaWhatIfTool,
            "TollamaRecommendTool": TollamaRecommendTool,
            "get_tollama_tools": get_tollama_tools,
        }
        return exports[name]

    if name in _CREWAI_EXPORTS:
        from .crewai import get_crewai_tools

        return {"get_crewai_tools": get_crewai_tools}[name]

    if name in _AUTOGEN_EXPORTS:
        from .autogen import (
            get_autogen_function_map,
            get_autogen_tool_specs,
            register_autogen_tools,
        )

        return {
            "get_autogen_function_map": get_autogen_function_map,
            "get_autogen_tool_specs": get_autogen_tool_specs,
            "register_autogen_tools": register_autogen_tools,
        }[name]

    if name in _SMOLAGENTS_EXPORTS:
        from .smolagents import get_smolagents_tools

        return {"get_smolagents_tools": get_smolagents_tools}[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
