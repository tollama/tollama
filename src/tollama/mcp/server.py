"""MCP server entrypoints for tollama tools."""

from __future__ import annotations

import json
from typing import Any

from .tools import (
    MCPToolError,
)
from .tools import (
    tollama_analyze as _tollama_analyze,
)
from .tools import (
    tollama_auto_forecast as _tollama_auto_forecast,
)
from .tools import (
    tollama_compare as _tollama_compare,
)
from .tools import (
    tollama_counterfactual as _tollama_counterfactual,
)
from .tools import (
    tollama_forecast as _tollama_forecast,
)
from .tools import (
    tollama_generate as _tollama_generate,
)
from .tools import (
    tollama_health as _tollama_health,
)
from .tools import (
    tollama_models as _tollama_models,
)
from .tools import (
    tollama_pipeline as _tollama_pipeline,
)
from .tools import (
    tollama_pull as _tollama_pull,
)
from .tools import (
    tollama_recommend as _tollama_recommend,
)
from .tools import (
    tollama_report as _tollama_report,
)
from .tools import (
    tollama_scenario_tree as _tollama_scenario_tree,
)
from .tools import (
    tollama_show as _tollama_show,
)
from .tools import (
    tollama_what_if as _tollama_what_if,
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
        name="tollama_auto_forecast",
        description=(
            "Run zero-config auto-forecast (model optional) and return selection metadata plus "
            "forecast payload. Required input: request.horizon and request.series[]. "
            "Optional: request.strategy (auto|fastest|best_accuracy|ensemble), "
            "request.model override, request.allow_fallback, and request.ensemble_top_k. "
            'Example: tollama_auto_forecast({"request":{"horizon":3,"strategy":"auto",'
            '"series":[{"id":"s1","freq":"D","timestamps":["2025-01-01","2025-01-02"],'
            '"target":[10,11]}],"options":{}}}).'
        ),
    )
    def tollama_auto_forecast(
        request: dict[str, Any],
        base_url: str | None = None,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        try:
            return _tollama_auto_forecast(request=request, base_url=base_url, timeout=timeout)
        except MCPToolError as exc:
            _raise_mcp_tool_error(exc)

    @server.tool(
        name="tollama_analyze",
        description=(
            "Analyze one or more time series and return frequency, seasonality, trend, "
            "anomalies, stationarity, and data quality diagnostics. "
            "This is a model-free diagnostic endpoint. "
            "Required input: request.series[]. Optional request.parameters tuning knobs include "
            "max_points, max_lag, top_k_seasonality, and anomaly_iqr_k. "
            'Example: tollama_analyze({"request":{"series":[{"id":"s1","freq":"D",'
            '"timestamps":["2025-01-01","2025-01-02","2025-01-03"],"target":[10,11,12]}]}}).'
        ),
    )
    def tollama_analyze(
        request: dict[str, Any],
        base_url: str | None = None,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        try:
            return _tollama_analyze(request=request, base_url=base_url, timeout=timeout)
        except MCPToolError as exc:
            _raise_mcp_tool_error(exc)

    @server.tool(
        name="tollama_generate",
        description=(
            "Generate model-free synthetic time series from historical statistical profiles. "
            "Required input: request.series[]. Optional: request.count, request.length, "
            "request.seed, request.variation, request.method='statistical'. "
            'Example: tollama_generate({"request":{"series":[{"id":"s1","freq":"D",'
            '"timestamps":["2025-01-01","2025-01-02","2025-01-03"],'
            '"target":[10,11,12]}],"count":3,"length":7,"seed":42}}).'
        ),
    )
    def tollama_generate(
        request: dict[str, Any],
        base_url: str | None = None,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        try:
            return _tollama_generate(request=request, base_url=base_url, timeout=timeout)
        except MCPToolError as exc:
            _raise_mcp_tool_error(exc)

    @server.tool(
        name="tollama_counterfactual",
        description=(
            "Generate intervention counterfactual trajectories by forecasting from "
            "pre-intervention history and comparing against observed post-intervention values. "
            "Required input: request.model, request.series[], request.intervention_index. "
            "Optional: request.intervention_label, request.quantiles, request.options, "
            "request.parameters. "
            'Example: tollama_counterfactual({"request":{"model":"mock","intervention_index":3,'
            '"series":[{"id":"s1","freq":"D","timestamps":["2025-01-01","2025-01-02",'
            '"2025-01-03","2025-01-04","2025-01-05"],"target":[10,11,12,30,31]}],'
            '"options":{}}}).'
        ),
    )
    def tollama_counterfactual(
        request: dict[str, Any],
        base_url: str | None = None,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        try:
            return _tollama_counterfactual(request=request, base_url=base_url, timeout=timeout)
        except MCPToolError as exc:
            _raise_mcp_tool_error(exc)

    @server.tool(
        name="tollama_scenario_tree",
        description=(
            "Build a probabilistic scenario tree using recursive one-step quantile forecasts. "
            "Required input: request.model, request.horizon, request.series[]. "
            "Optional: request.depth, request.branch_quantiles, request.options, "
            "request.parameters. "
            'Example: tollama_scenario_tree({"request":{"model":"mock","horizon":4,"depth":2,'
            '"branch_quantiles":[0.1,0.5,0.9],'
            '"series":[{"id":"s1","freq":"D","timestamps":["2025-01-01","2025-01-02"],'
            '"target":[10,11]}],"options":{}}}).'
        ),
    )
    def tollama_scenario_tree(
        request: dict[str, Any],
        base_url: str | None = None,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        try:
            return _tollama_scenario_tree(request=request, base_url=base_url, timeout=timeout)
        except MCPToolError as exc:
            _raise_mcp_tool_error(exc)

    @server.tool(
        name="tollama_report",
        description=(
            "Run one-call composite report: analyze -> recommend -> optional pull -> "
            "auto-forecast with optional narrative and baseline inclusion. "
            "Required input: request.horizon and request.series[]. "
            "Optional: request.strategy, request.model, request.recommend_top_k, "
            "request.include_baseline, request.response_options.narrative. "
            'Example: tollama_report({"request":{"horizon":3,"strategy":"auto",'
            '"series":[{"id":"s1","freq":"D","timestamps":["2025-01-01","2025-01-02"],'
            '"target":[10,11]}],"options":{},"include_baseline":true}}).'
        ),
    )
    def tollama_report(
        request: dict[str, Any],
        base_url: str | None = None,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        try:
            return _tollama_report(request=request, base_url=base_url, timeout=timeout)
        except MCPToolError as exc:
            _raise_mcp_tool_error(exc)

    @server.tool(
        name="tollama_what_if",
        description=(
            "Run scenario analysis by applying named transforms to a base forecast request "
            "and returning baseline + per-scenario outputs. "
            "Required input: request.model, request.horizon, request.series[], "
            "request.scenarios[]. "
            "Each scenario includes name and transforms with operation "
            "(multiply|add|replace), field (target|past_covariates|future_covariates), "
            "optional key for covariates, optional series_id, and value. "
            'Example: tollama_what_if({"request":{"model":"mock","horizon":2,'
            '"series":[{"id":"s1","freq":"D","timestamps":["2025-01-01","2025-01-02"],'
            '"target":[10,11]}],"scenarios":[{"name":"high_demand","transforms":'
            '[{"operation":"multiply","field":"target","value":1.2}]}],"options":{}}}).'
        ),
    )
    def tollama_what_if(
        request: dict[str, Any],
        base_url: str | None = None,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        try:
            return _tollama_what_if(request=request, base_url=base_url, timeout=timeout)
        except MCPToolError as exc:
            _raise_mcp_tool_error(exc)

    @server.tool(
        name="tollama_pipeline",
        description=(
            "Run the full autonomous forecasting workflow in one call: "
            "analyze -> recommend -> optional pull -> auto-forecast. "
            "Required input: request.horizon and request.series[]. "
            "Optional: request.strategy, request.model override, request.recommend_top_k, "
            "request.pull_if_missing, request.accept_license, and "
            "request.allow_restricted_license. "
            'Example: tollama_pipeline({"request":{"horizon":3,"strategy":"auto",'
            '"series":[{"id":"s1","freq":"D","timestamps":["2025-01-01","2025-01-02"],'
            '"target":[10,11]}],"options":{},"pull_if_missing":true}}).'
        ),
    )
    def tollama_pipeline(
        request: dict[str, Any],
        base_url: str | None = None,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        try:
            return _tollama_pipeline(request=request, base_url=base_url, timeout=timeout)
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
