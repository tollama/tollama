"""Tests for tollama MCP entrypoint behavior."""

from __future__ import annotations

import pytest

from tollama.mcp import __main__ as mcp_main


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
