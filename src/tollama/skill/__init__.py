"""Skill integrations for external agent frameworks."""

from __future__ import annotations

__all__ = [
    "TollamaCompareTool",
    "TollamaForecastTool",
    "TollamaHealthTool",
    "TollamaModelsTool",
    "TollamaRecommendTool",
    "get_tollama_tools",
]


def __getattr__(name: str):
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from .langchain import (
        TollamaCompareTool,
        TollamaForecastTool,
        TollamaHealthTool,
        TollamaModelsTool,
        TollamaRecommendTool,
        get_tollama_tools,
    )

    exports = {
        "TollamaCompareTool": TollamaCompareTool,
        "TollamaForecastTool": TollamaForecastTool,
        "TollamaHealthTool": TollamaHealthTool,
        "TollamaModelsTool": TollamaModelsTool,
        "TollamaRecommendTool": TollamaRecommendTool,
        "get_tollama_tools": get_tollama_tools,
    }
    return exports[name]
