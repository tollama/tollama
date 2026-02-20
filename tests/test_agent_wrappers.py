"""Tests for additional agent framework wrappers."""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest

from tollama.skill import (
    get_autogen_function_map,
    get_autogen_tool_specs,
    get_crewai_tools,
    get_smolagents_tools,
    register_autogen_tools,
)


def test_autogen_wrappers_expose_tool_specs_and_function_map() -> None:
    specs = get_autogen_tool_specs()
    function_map = get_autogen_function_map()

    names = [item["function"]["name"] for item in specs]
    assert names == [
        "tollama_health",
        "tollama_models",
        "tollama_forecast",
        "tollama_auto_forecast",
        "tollama_analyze",
        "tollama_generate",
        "tollama_counterfactual",
        "tollama_scenario_tree",
        "tollama_report",
        "tollama_compare",
        "tollama_recommend",
    ]

    payload = function_map["tollama_recommend"](
        horizon=24,
        has_future_covariates=True,
        covariates_type="numeric",
        top_k=2,
    )
    assert payload["request"]["horizon"] == 24
    assert len(payload["recommendations"]) <= 2


def test_autogen_register_uses_register_function(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []
    fake_module = types.ModuleType("autogen")

    def _register_function(
        function: Any,
        *,
        caller: Any,
        executor: Any,
        name: str,
        description: str,
    ) -> None:
        calls.append(
            {
                "function": function,
                "caller": caller,
                "executor": executor,
                "name": name,
                "description": description,
            },
        )

    fake_module.register_function = _register_function  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "autogen", fake_module)

    caller = object()
    executor = object()
    result = register_autogen_tools(caller=caller, executor=executor)

    assert result["count"] == 11
    assert len(calls) == 11
    assert calls[0]["caller"] is caller
    assert calls[0]["executor"] is executor
    assert calls[0]["name"] == "tollama_health"


def test_crewai_wrapper_builds_tools_from_specs(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_crewai_module = types.ModuleType("crewai")
    fake_tools_module = types.ModuleType("crewai.tools")

    class _FakeBaseTool:
        name: str = ""
        description: str = ""
        args_schema: Any = None

        def run(self, **kwargs: Any) -> dict[str, Any]:
            return self._run(**kwargs)

    fake_tools_module.BaseTool = _FakeBaseTool  # type: ignore[attr-defined]
    fake_crewai_module.tools = fake_tools_module  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "crewai", fake_crewai_module)
    monkeypatch.setitem(sys.modules, "crewai.tools", fake_tools_module)

    tools = get_crewai_tools()
    tools_by_name = {tool.name: tool for tool in tools}

    recommend_payload = tools_by_name["tollama_recommend"].run(
        horizon=12,
        has_future_covariates=True,
        covariates_type="numeric",
        top_k=3,
    )
    assert recommend_payload["request"]["horizon"] == 12
    assert tools_by_name["tollama_recommend"].args_schema is not None


def test_smolagents_wrapper_builds_tools_from_specs(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_smolagents_module = types.ModuleType("smolagents")

    class _FakeTool:
        name: str = ""
        description: str = ""
        inputs: dict[str, Any] = {}
        output_type: str = "object"

    fake_smolagents_module.Tool = _FakeTool  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "smolagents", fake_smolagents_module)

    tools = get_smolagents_tools()
    tools_by_name = {tool.name: tool for tool in tools}

    recommend_tool = tools_by_name["tollama_recommend"]
    payload = recommend_tool.forward(
        horizon=6,
        has_future_covariates=True,
        covariates_type="numeric",
        top_k=1,
    )

    assert payload["request"]["horizon"] == 6
    assert recommend_tool.inputs["horizon"]["required"] is True
