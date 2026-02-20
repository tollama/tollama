"""MCP server entrypoints for tollama tools."""

from __future__ import annotations

import json
from typing import Any

from .tools import (
    MCPToolError,
)
from .tools import (
    tollama_compare as _tollama_compare,
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
    tollama_recommend as _tollama_recommend,
)
from .tools import (
    tollama_show as _tollama_show,
)

_MCP_IMPORT_HINT = 'MCP dependency is not installed. Install with: pip install "tollama[mcp]"'
_MODEL_NAME_EXAMPLES = (
    "mock, chronos2, granite-ttm-r2, timesfm-2.5-200m, "
    "moirai-2.0-R-small, sundial-base-128m, toto-open-base-1.0"
)


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

    @server.tool(
        name="tollama_health",
        description=(
            "Check daemon reachability and version information. "
            "No required inputs. Optional base_url and timeout overrides are supported. "
            "Available models can be queried with tollama_models "
            f"(examples include: {_MODEL_NAME_EXAMPLES}). "
            'Example: tollama_health({"base_url":"http://127.0.0.1:11435","timeout":5.0}).'
        ),
    )
    def tollama_health(base_url: str | None = None, timeout: float | None = None) -> dict[str, Any]:
        try:
            return _tollama_health(base_url=base_url, timeout=timeout)
        except MCPToolError as exc:
            _raise_mcp_tool_error(exc)

    @server.tool(
        name="tollama_models",
        description=(
            "List models from tollama by mode. "
            "Required inputs: none. Optional mode values are installed, loaded, or available. "
            "Available model names include "
            f"{_MODEL_NAME_EXAMPLES}. "
            'Example: tollama_models({"mode":"available"}).'
        ),
    )
    def tollama_models(
        mode: str = "installed",
        base_url: str | None = None,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        try:
            return _tollama_models(mode=mode, base_url=base_url, timeout=timeout)
        except MCPToolError as exc:
            _raise_mcp_tool_error(exc)

    @server.tool(
        name="tollama_forecast",
        description=(
            "Run a non-streaming forecast request and return canonical forecast JSON. "
            "Required inputs: request.model, request.horizon, request.series[]. "
            "Series entries must include id, timestamps, and target. "
            "Supported model names include "
            f"{_MODEL_NAME_EXAMPLES}. "
            'Example: tollama_forecast({"request":{"model":"chronos2","horizon":3,'
            '"series":[{"id":"s1","freq":"D","timestamps":["2025-01-01","2025-01-02"],'
            '"target":[10,11]}],"options":{}}}).'
        ),
    )
    def tollama_forecast(
        request: dict[str, Any],
        base_url: str | None = None,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        try:
            return _tollama_forecast(request=request, base_url=base_url, timeout=timeout)
        except MCPToolError as exc:
            _raise_mcp_tool_error(exc)

    @server.tool(
        name="tollama_compare",
        description=(
            "Run the same forecast request across multiple models and return side-by-side results. "
            "Required input: request.models, request.horizon, request.series[]. "
            "Optional request fields mirror forecast (quantiles, options, timeout, parameters). "
            "Each model result is returned with ok=true/false and response/error payload. "
            "Supported model names include "
            f"{_MODEL_NAME_EXAMPLES}. "
            'Example: tollama_compare({"request":{"models":["chronos2","timesfm-2.5-200m"],'
            '"horizon":3,"series":[{"id":"s1","freq":"D","timestamps":["2025-01-01","2025-01-02"],'
            '"target":[10,11]}],"options":{}}}).'
        ),
    )
    def tollama_compare(
        request: dict[str, Any],
        base_url: str | None = None,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        try:
            return _tollama_compare(request=request, base_url=base_url, timeout=timeout)
        except MCPToolError as exc:
            _raise_mcp_tool_error(exc)

    @server.tool(
        name="tollama_pull",
        description=(
            "Pull an installed model manifest and snapshot into local storage. "
            "Required input: model. Optional accept_license, base_url, and timeout. "
            "Model name choices include "
            f"{_MODEL_NAME_EXAMPLES}. "
            'Example: tollama_pull({"model":"mock","accept_license":false}).'
        ),
    )
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

    @server.tool(
        name="tollama_show",
        description=(
            "Fetch model metadata and capabilities from installed manifests. "
            "Required input: model. Optional base_url and timeout. "
            "Model names include "
            f"{_MODEL_NAME_EXAMPLES}. "
            'Example: tollama_show({"model":"timesfm-2.5-200m"}).'
        ),
    )
    def tollama_show(
        model: str,
        base_url: str | None = None,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        try:
            return _tollama_show(model=model, base_url=base_url, timeout=timeout)
        except MCPToolError as exc:
            _raise_mcp_tool_error(exc)

    @server.tool(
        name="tollama_recommend",
        description=(
            "Recommend forecast models from registry metadata and capability flags. "
            "Required input: horizon. Optional inputs: freq, has_past_covariates, "
            "has_future_covariates, has_static_covariates, covariates_type, "
            "allow_restricted_license, and top_k. "
            "Uses model metadata and covariate compatibility to return ranked suggestions. "
            "Model pool includes "
            f"{_MODEL_NAME_EXAMPLES}. "
            'Example: tollama_recommend({"horizon":48,"freq":"D","has_future_covariates":true,'
            '"covariates_type":"numeric","top_k":3}).'
        ),
    )
    def tollama_recommend(
        horizon: int,
        freq: str | None = None,
        has_past_covariates: bool = False,
        has_future_covariates: bool = False,
        has_static_covariates: bool = False,
        covariates_type: str = "numeric",
        allow_restricted_license: bool = False,
        top_k: int = 3,
    ) -> dict[str, Any]:
        try:
            return _tollama_recommend(
                horizon=horizon,
                freq=freq,
                has_past_covariates=has_past_covariates,
                has_future_covariates=has_future_covariates,
                has_static_covariates=has_static_covariates,
                covariates_type=covariates_type,
                allow_restricted_license=allow_restricted_license,
                top_k=top_k,
            )
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
