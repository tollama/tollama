"""MCP server entrypoints for tollama tools."""

from __future__ import annotations

import json
from typing import Any

from .tools import (
    MCPToolError,
)
from .tools import (
    tollama_forecast as _tollama_forecast,
)
from .tools import (
    tollama_health as _tollama_health,
)
from .tools import (
    tollama_models as _tollama_models,
)
from .tools import (
    tollama_pull as _tollama_pull,
)
from .tools import (
    tollama_show as _tollama_show,
)

_MCP_IMPORT_HINT = 'MCP dependency is not installed. Install with: pip install "tollama[mcp]"'


def _load_fastmcp() -> type[Any]:
    try:
        from mcp.server.fastmcp import FastMCP  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise RuntimeError(_MCP_IMPORT_HINT) from exc
    return FastMCP


def _raise_mcp_tool_error(exc: MCPToolError) -> None:
    payload = {
        "error": {
            "category": exc.category,
            "exit_code": exc.exit_code,
            "message": exc.message,
        }
    }
    raise RuntimeError(json.dumps(payload, separators=(",", ":"), sort_keys=True)) from exc


def create_server() -> Any:
    """Create a FastMCP server and register tollama tools."""
    FastMCP = _load_fastmcp()
    server = FastMCP("tollama")

    @server.tool(name="tollama_health")
    def tollama_health(base_url: str | None = None, timeout: float | None = None) -> dict[str, Any]:
        try:
            return _tollama_health(base_url=base_url, timeout=timeout)
        except MCPToolError as exc:
            _raise_mcp_tool_error(exc)

    @server.tool(name="tollama_models")
    def tollama_models(
        mode: str = "installed",
        base_url: str | None = None,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        try:
            return _tollama_models(mode=mode, base_url=base_url, timeout=timeout)
        except MCPToolError as exc:
            _raise_mcp_tool_error(exc)

    @server.tool(name="tollama_forecast")
    def tollama_forecast(
        request: dict[str, Any],
        base_url: str | None = None,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        try:
            return _tollama_forecast(request=request, base_url=base_url, timeout=timeout)
        except MCPToolError as exc:
            _raise_mcp_tool_error(exc)

    @server.tool(name="tollama_pull")
    def tollama_pull(
        model: str,
        accept_license: bool = False,
        base_url: str | None = None,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        try:
            return _tollama_pull(
                model=model,
                accept_license=accept_license,
                base_url=base_url,
                timeout=timeout,
            )
        except MCPToolError as exc:
            _raise_mcp_tool_error(exc)

    @server.tool(name="tollama_show")
    def tollama_show(
        model: str,
        base_url: str | None = None,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        try:
            return _tollama_show(model=model, base_url=base_url, timeout=timeout)
        except MCPToolError as exc:
            _raise_mcp_tool_error(exc)

    return server


def run_server() -> None:
    """Run the MCP server over stdio."""
    server = create_server()
    if hasattr(server, "run"):
        server.run()
        return
    if hasattr(server, "run_stdio"):
        server.run_stdio()
        return
    raise RuntimeError("Unsupported MCP server runtime: missing run()/run_stdio()")
