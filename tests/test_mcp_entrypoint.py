"""Tests for tollama MCP entrypoint behavior."""

from __future__ import annotations

from typing import Any

import pytest

from tollama.mcp import __main__ as mcp_main
from tollama.mcp import server as mcp_server


def test_main_runs_server_when_available(monkeypatch) -> None:
    called = {"value": False}

    def _fake_run_server() -> None:
        called["value"] = True

    monkeypatch.setattr(mcp_main, "_run_server", _fake_run_server)

    mcp_main.main()

    assert called["value"] is True


def test_main_exits_with_install_hint_on_runtime_error(monkeypatch, capsys) -> None:
    def _fake_run_server() -> None:
        raise RuntimeError(
            'MCP dependency is not installed. Install with: pip install "tollama[mcp]"'
        )

    monkeypatch.setattr(mcp_main, "_run_server", _fake_run_server)

    with pytest.raises(SystemExit) as exc_info:
        mcp_main.main()

    captured = capsys.readouterr()
    assert exc_info.value.code == 1
    assert "pip install \"tollama[mcp]\"" in captured.err


def test_create_server_registers_descriptions_for_all_tools(monkeypatch) -> None:
    registered: dict[str, dict[str, Any]] = {}

    class _FakeFastMCP:
        def __init__(self, name: str) -> None:
            self.name = name

        def tool(self, *, name: str, description: str):  # noqa: ANN202
            def _decorate(func):  # type: ignore[no-untyped-def]
                registered[name] = {"description": description, "func": func}
                return func

            return _decorate

    monkeypatch.setattr(mcp_server, "_load_fastmcp", lambda: _FakeFastMCP)

    server = mcp_server.create_server()

    assert isinstance(server, _FakeFastMCP)
    assert set(registered) == {
        "tollama_health",
        "tollama_models",
        "tollama_forecast",
        "tollama_auto_forecast",
        "tollama_analyze",
        "tollama_generate",
        "tollama_counterfactual",
        "tollama_scenario_tree",
        "tollama_report",
        "tollama_what_if",
        "tollama_pipeline",
        "tollama_compare",
        "tollama_pull",
        "tollama_show",
        "tollama_recommend",
    }
    for payload in registered.values():
        description = payload["description"]
        assert "Example:" in description
        assert "model" in description
    assert "horizon" in registered["tollama_forecast"]["description"]
