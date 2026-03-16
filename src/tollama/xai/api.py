"""
tollama.xai.api — XAI API Endpoints for Tollama Core

v3.8 Phase 2b: Trust-gated explanation endpoints.
Trust Intelligence Pipeline (when installed) is wired into decisioning:
trust score, constraint violations, and risk category gate auto-execution.

Endpoints:
  POST /api/explain-decision     — End-to-end decision explanation (trust-aware)
  POST /api/forecast             — (extended) ?explain=true parameter
  POST /api/model-card           — Generate EU AI Act model card
  POST /api/trust-breakdown      — Trust Score breakdown
  POST /api/forecast-decompose   — Trend/seasonal/residual decomposition
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field, ValidationError

router = APIRouter(prefix="/api/xai", tags=["xai"])


def _get_trust_router(request: Request | None = None):
    """Return shared trust router from app.state, or build a fresh one."""
    if request is not None:
        tr = getattr(getattr(request, "app", None), "state", None)
        if tr is not None:
            shared = getattr(tr, "trust_router", None)
            if shared is not None:
                return shared
    from tollama.xai.trust_router import build_default_trust_router

    return build_default_trust_router()


# ──────────────────────────────────────────────────────────────
# Request / Response Models
# ──────────────────────────────────────────────────────────────

class ExplainDecisionRequest(BaseModel):
    """Request body for /api/explain-decision"""
    forecast_result: dict[str, Any] = Field(
        ..., description="Output from tollama forecast endpoint"
    )
    eval_result: dict[str, Any] | None = Field(
        None, description="Output from tollama-eval"
    )
    calibration_result: dict[str, Any] | None = Field(
        None, description="Output from Market Calibration Agent"
    )
    trust_result: dict[str, Any] | None = Field(
        None, description="Normalized output from a domain trust agent"
    )
    policy_config: dict[str, Any] | None = Field(
        None, description="Decision policy configuration"
    )
    time_series_data: list[float] | None = Field(
        None, description="Raw time series for decomposition"
    )
    explain_options: dict[str, Any] | None = Field(
        default_factory=lambda: {"decompose": True, "attribution": False},
        description="Control explanation depth",
    )
    trust_features: dict[str, float] | None = Field(
        None, description="Features for SHAP attribution in trust pipeline"
    )
    trust_context: dict[str, Any] | None = Field(
        None,
        description=(
            "Context for trust agent routing. Keys: 'domain' (prediction_market|"
            "financial_market|supply_chain|news|geopolitical|regulatory), "
            "'source_type' (optional refinement), "
            "'mode' ('single' default or 'multi' for multi-agent aggregation)."
        ),
    )
    trust_payload: dict[str, Any] | None = Field(
        None,
        description=(
            "Payload routed through the default trust-agent registry. "
            "Schema depends on domain: financial_market requires 'instrument_id', "
            "news requires 'story_id', supply_chain requires 'network_id', "
            "geopolitical requires 'region_id', regulatory requires 'jurisdiction_id'."
        ),
    )


class ExplainDecisionResponse(BaseModel):
    """Response from /api/explain-decision"""
    explanation_id: str = Field(..., description="Unique identifier for this explanation")
    timestamp: str = Field(..., description="ISO timestamp when explanation was generated")
    version: str = Field(..., description="Explanation schema version")
    input_explanation: dict[str, Any] = Field(
        ..., description="Input-stage explanation: signals used and trust rationale"
    )
    plan_explanation: dict[str, Any] = Field(
        ..., description="Plan-stage explanation: model selection and forecast decomposition"
    )
    decision_policy_explanation: dict[str, Any] = Field(
        ..., description="Decision policy explanation: auto-execution and escalation reasoning"
    )
    trust_intelligence_explanation: dict[str, Any] | None = Field(
        None, description="Trust Intelligence Pipeline evidence (optional)"
    )
    metadata: dict[str, Any] = Field(
        ..., description="Additional metadata about the explanation"
    )


class TrustBreakdownRequest(BaseModel):
    """Request body for /api/trust-breakdown"""
    trust_score: float = Field(..., description="Overall trust score")
    metrics: dict[str, float] = Field(
        ..., description="Calibration metrics (brier_score, log_loss, ece, ...)"
    )
    source: str = Field(default="polymarket", description="Signal source name")
    signals: list[dict[str, Any]] | None = Field(
        None, description="Multiple signals"
    )


class ForecastDecomposeRequest(BaseModel):
    """Request body for /api/forecast-decompose"""
    data: list[float] = Field(..., description="Time series values")
    period: int | None = Field(None, description="Seasonal period (auto-detect if None)")
    method: str = Field(default="stl", description="Decomposition method")


class ModelCardRequest(BaseModel):
    """Request body for /api/model-card"""
    model_info: dict[str, Any] = Field(
        ..., description="Model identity information"
    )
    eval_result: dict[str, Any] | None = Field(
        None, description="Evaluation results to include in card"
    )
    explanation_result: dict[str, Any] | None = Field(
        None, description="Explanation output to include in card"
    )
    governance_info: dict[str, Any] | None = Field(
        None, description="Governance and compliance information"
    )
    format: str = Field(default="json", description="Output format: json or markdown")


class DecisionReportRequest(BaseModel):
    """Request body for /api/decision-report"""
    explanation: dict[str, Any] = Field(
        ..., description="Output from /api/explain-decision"
    )
    forecast_result: dict[str, Any] | None = Field(
        None, description="Forecast result to include in report"
    )
    report_config: dict[str, Any] | None = Field(
        default_factory=dict,
        description="Report customization (title, audience, format)",
    )
    report_type: str = Field(
        default="decision",
        description="Report type: 'decision' or 'explanation'",
    )
    format: str = Field(default="json", description="Output format: json, markdown")


class DashboardTrustRequest(BaseModel):
    """Request body for /api/dashboard/trust"""
    domains: list[str] | None = Field(
        None,
        description="Filter to specific domains. Omit for all.",
    )
    include_calibration: bool = Field(
        default=True,
        description="Include calibration stats in response.",
    )


class RecordOutcomeRequest(BaseModel):
    """Request body for /api/xai/record-outcome"""
    agent_name: str = Field(
        ..., min_length=1,
        description="Trust agent name that produced the prediction.",
    )
    domain: str = Field(
        ..., min_length=1,
        description="Domain of the prediction.",
    )
    predicted_score: float = Field(
        ..., ge=0.0, le=1.0,
        description="Trust score predicted by the agent.",
    )
    actual_outcome: float = Field(
        ..., ge=0.0, le=1.0,
        description="Actual outcome observed (0=bad, 1=perfect).",
    )
    component_scores: dict[str, float] | None = Field(
        default_factory=dict,
        description="Per-component scores for calibration learning.",
    )


class TrustHistoryRequest(BaseModel):
    """Request body for /api/xai/dashboard/history"""
    domains: list[str] | None = Field(
        None,
        description="Filter to specific domains. Omit for all.",
    )
    limit: int = Field(
        default=50,
        ge=1,
        le=500,
        description="Maximum number of history records per domain.",
    )
    include_stats: bool = Field(
        default=True,
        description="Include aggregated stats and trend.",
    )


class ConnectorHealthRequest(BaseModel):
    """Request body for /api/xai/connectors/health"""
    domains: list[str] | None = Field(
        None,
        description="Filter to specific domains. Omit for all.",
    )


# ──────────────────────────────────────────────────────────────
# Response Models (must be defined before endpoint decorators)
# ──────────────────────────────────────────────────────────────


class GateStatus(BaseModel):
    """A single gate evaluation result."""
    gate: str = Field(..., description="Gate name.")
    status: str = Field(..., description="PASS, WARN, or BLOCK.")
    detail: str = Field(..., description="Human-readable detail.")


class GateDecisionResponse(BaseModel):
    """Response from /api/xai/gate-decision"""
    allowed: bool = Field(..., description="Whether auto-execution is allowed.")
    gates: list[GateStatus] = Field(..., description="Per-gate evaluation results.")
    risk_category: str | None = Field(None, description="Trust result risk category.")
    trust_score: float | None = Field(None, description="Trust result score.")


class TrustAttributionItem(BaseModel):
    """A single component attribution."""
    component: str = Field(..., description="Component name.")
    weight: float = Field(..., description="Component weight.")
    score: float = Field(..., description="Component score.")
    contribution: float = Field(..., description="Weighted contribution.")
    impact_pct: float = Field(..., description="Percentage of total score.")


class TrustAttributionResponse(BaseModel):
    """Response from /api/xai/trust-attribution"""
    attributions: list[TrustAttributionItem] = Field(
        ..., description="Component attributions ordered by contribution.",
    )
    baseline: float = Field(..., description="Baseline value.")
    total_score: float = Field(..., description="Total trust score.")
    top_driver: str | None = Field(None, description="Top contributing component.")


class BatchResultItem(BaseModel):
    """A single batch analysis result."""
    status: str = Field(..., description="'ok' or 'no_agent'.")
    result: dict[str, Any] | None = Field(None, description="Trust result summary.")


class BatchAnalyzeResponse(BaseModel):
    """Response from /api/xai/batch-analyze"""
    results: list[BatchResultItem] = Field(..., description="Per-item results.")
    count: int = Field(..., description="Total number of results.")


class AlertItem(BaseModel):
    """A triggered alert."""
    domain: str = Field(..., description="Domain that triggered the alert.")
    trust_score: float = Field(..., description="Current trust score.")
    risk_category: str = Field(..., description="Current risk category.")
    trend: str | None = Field(None, description="Current trust trend.")
    reasons: list[str] = Field(..., description="Alert trigger reasons.")
    webhook_url: str | None = Field(None, description="Webhook URL if configured.")


class AlertCheckResponse(BaseModel):
    """Response from /api/xai/alerts/check"""
    alerts: list[AlertItem] = Field(..., description="Triggered alerts.")
    alert_count: int = Field(..., description="Number of triggered alerts.")
    trust_result: dict[str, Any] | None = Field(
        None, description="Trust analysis result summary.",
    )


class AlertConfigResponse(BaseModel):
    """Response from /api/xai/alerts/configure and /api/xai/alerts/config"""
    status: str | None = Field(None, description="Configuration status.")
    threshold_count: int = Field(..., description="Number of configured thresholds.")
    thresholds: list[dict[str, Any]] = Field(
        ..., description="Configured alert thresholds.",
    )


class CacheStatsResponse(BaseModel):
    """Response from /api/xai/cache/stats"""
    hits: int = Field(..., description="Cache hit count.")
    misses: int = Field(..., description="Cache miss count.")
    total: int = Field(..., description="Total cache lookups.")
    hit_rate: float = Field(..., description="Cache hit rate (0.0-1.0).")
    cached_entries: int = Field(..., description="Current number of cached entries.")
    ttl: float = Field(..., description="Cache TTL in seconds.")


class CacheTTLResponse(BaseModel):
    """Response from /api/xai/cache/ttl"""
    previous_ttl: float = Field(..., description="Previous TTL value.")
    new_ttl: float = Field(..., description="New TTL value.")


class CacheInvalidateResponse(BaseModel):
    """Response from /api/xai/cache/invalidate"""
    cleared: int = Field(..., description="Number of cache entries cleared.")


# ──────────────────────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────────────────────

@router.post(
    "/explain-decision",
    response_model=ExplainDecisionResponse,
    summary="Generate end-to-end decision explanation",
    description=(
        "Assembles evidence from eval, calibration, policy, and trust-intelligence "
        "layers into a unified Decision Explanation. "
        "Phase 2b: Trust-gated decisioning with L1-L5 pipeline integration. "
        "Phase 4: Full explanation API."
    ),
)
async def explain_decision(request: ExplainDecisionRequest, raw_request: Request):
    """
    POST /api/explain-decision

    v3.8 Target — 핵심 XAI 엔드포인트
    Input → Plan → Decision Policy 전 단계 evidence-backed explanation.
    """
    from tollama.xai.decision_policy import DecisionPolicyExplainer
    from tollama.xai.engine import ExplanationEngine
    from tollama.xai.feature_attribution import TemporalFeatureAttribution
    from tollama.xai.forecast_decompose import ForecastDecomposer
    from tollama.xai.model_selection import ModelSelectionExplainer
    from tollama.xai.scenario_rationale import ScenarioRationale
    from tollama.xai.trust_breakdown import TrustBreakdown

    trust_pipeline = None
    try:
        from trust_intelligence.pipeline.trust_pipeline import TrustIntelligencePipeline
        trust_pipeline = TrustIntelligencePipeline()
    except ImportError:
        pass

    engine = ExplanationEngine(
        model_selection_explainer=ModelSelectionExplainer(),
        forecast_decomposer=ForecastDecomposer(),
        feature_attribution=TemporalFeatureAttribution(),
        scenario_rationale=ScenarioRationale(),
        trust_breakdown=TrustBreakdown(),
        decision_policy_explainer=DecisionPolicyExplainer(),
        trust_intelligence_pipeline=trust_pipeline,
        trust_router=_get_trust_router(raw_request),
    )

    explain_options = request.explain_options or {}
    if request.trust_features:
        explain_options["trust_intelligence_features"] = request.trust_features
    if request.trust_context:
        explain_options["trust_intelligence_context"] = request.trust_context

    import asyncio

    def _run_explain():
        return engine.explain_decision(
            forecast_result=request.forecast_result,
            eval_result=request.eval_result,
            calibration_result=request.calibration_result,
            trust_result=request.trust_result,
            policy_config=request.policy_config,
            time_series_data=request.time_series_data,
            explain_options=explain_options,
            trust_context=request.trust_context,
            trust_payload=request.trust_payload,
        )

    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, _run_explain)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors()) from exc

    return result.to_dict()


@router.post(
    "/trust-breakdown",
    summary="Generate Trust Score breakdown",
    description="Decomposes Trust Score into component-level explanations.",
)
async def trust_breakdown(request: TrustBreakdownRequest):
    """POST /api/trust-breakdown"""
    from tollama.xai.trust_breakdown import TrustBreakdown

    tb = TrustBreakdown()
    result = tb.explain({
        "trust_score": request.trust_score,
        "metrics": request.metrics,
        "source": request.source,
        "signals": request.signals or [{
            "name": request.source,
            "trust_score": request.trust_score,
            "metrics": request.metrics,
        }],
    })
    return result


@router.post(
    "/forecast-decompose",
    summary="Decompose time series into trend/seasonal/residual",
    description="Provides forecast decomposition for explainability.",
)
async def forecast_decompose(request: ForecastDecomposeRequest):
    """POST /api/forecast-decompose"""
    from tollama.xai.forecast_decompose import ForecastDecomposer

    decomposer = ForecastDecomposer(method=request.method, period=request.period)
    result = decomposer.decompose(request.data, period=request.period)
    return result


@router.post(
    "/model-card",
    summary="Generate EU AI Act model card",
    description="Auto-generates regulatory-compliant model documentation.",
)
async def generate_model_card(request: ModelCardRequest):
    """POST /api/model-card"""
    from tollama.xai.model_card import ModelCardGenerator

    generator = ModelCardGenerator()
    card = generator.generate(
        model_info=request.model_info,
        eval_result=request.eval_result,
        explanation_result=request.explanation_result,
        governance_info=request.governance_info,
    )

    if request.format == "markdown":
        return {"format": "markdown", "content": generator.to_markdown(card)}
    return card


@router.post(
    "/decision-report",
    summary="Generate decision or explanation report",
    description="Builds structured reports from explanation output.",
)
async def generate_decision_report(request: DecisionReportRequest):
    """POST /api/decision-report"""
    from tollama.xai.report_generator import DecisionReportBuilder

    builder = DecisionReportBuilder()

    if request.report_type == "explanation":
        report = builder.build_explanation_report(request.explanation)
    else:
        report = builder.build_decision_report(
            explanation=request.explanation,
            forecast_result=request.forecast_result,
            report_config=request.report_config,
        )

    if request.format == "markdown":
        return {"format": "markdown", "content": builder.to_markdown(report)}
    return report


# ──────────────────────────────────────────────────────────────
# Dashboard Endpoints
# ──────────────────────────────────────────────────────────────


@router.get(
    "/dashboard/agents",
    summary="List registered trust agents",
    description="Returns all trust agents with their domain and priority.",
)
async def dashboard_agents(raw_request: Request):
    """GET /api/xai/dashboard/agents"""
    router_instance = _get_trust_router(raw_request)
    agents = []
    for agent in router_instance.registry.agents:
        agents.append({
            "agent_name": agent.agent_name,
            "domain": agent.domain,
            "priority": getattr(agent, "priority", 100),
        })
    return {"agents": agents}


@router.post(
    "/dashboard/trust",
    summary="Trust dashboard summary",
    description=(
        "Returns trust agent overview with optional calibration stats. "
        "Suitable for dashboard rendering."
    ),
)
async def dashboard_trust(request: DashboardTrustRequest, raw_request: Request):
    """POST /api/xai/dashboard/trust"""
    router_instance = _get_trust_router(raw_request)

    all_domains = {
        "prediction_market", "financial_market", "supply_chain",
        "news", "geopolitical", "regulatory",
    }
    domains = set(request.domains) if request.domains else all_domains

    agent_summaries = []
    for agent in router_instance.registry.agents:
        if agent.domain not in domains:
            continue
        summary: dict[str, Any] = {
            "agent_name": agent.agent_name,
            "domain": agent.domain,
            "priority": getattr(agent, "priority", 100),
        }
        agent_summaries.append(summary)

    calibration_stats = []
    if request.include_calibration and router_instance.calibration_tracker:
        tracker = router_instance.calibration_tracker
        for name in tracker.agents:
            stats = tracker.get_calibration_stats(name)
            calibration_stats.append(stats.model_dump(mode="json"))

    return {
        "agents": agent_summaries,
        "calibration": calibration_stats,
        "domains": sorted(domains),
    }


# ──────────────────────────────────────────────────────────────
# Calibration Feedback
# ──────────────────────────────────────────────────────────────


@router.post(
    "/record-outcome",
    summary="Record a prediction-outcome pair for calibration",
    description=(
        "Feeds an actual outcome back into the CalibrationTracker "
        "so trust agents can learn and self-correct over time."
    ),
)
async def record_outcome(request: RecordOutcomeRequest, raw_request: Request):
    """POST /api/xai/record-outcome"""
    router_instance = _get_trust_router(raw_request)
    router_instance.record_outcome(
        agent_name=request.agent_name,
        domain=request.domain,
        predicted_score=request.predicted_score,
        actual_outcome=request.actual_outcome,
        component_scores=request.component_scores or {},
    )
    router_instance.persist_calibration()

    stats = None
    if router_instance.calibration_tracker:
        s = router_instance.calibration_tracker.get_calibration_stats(
            request.agent_name,
        )
        stats = s.model_dump(mode="json")

    return {
        "status": "recorded",
        "agent_name": request.agent_name,
        "calibration_stats": stats,
    }


# ──────────────────────────────────────────────────────────────
# Trust History & Trends
# ──────────────────────────────────────────────────────────────


@router.post(
    "/dashboard/history",
    summary="Trust score history and trends",
    description=(
        "Returns trust score history per domain with trend analysis. "
        "Suitable for dashboard time-series charts."
    ),
)
async def dashboard_history(request: TrustHistoryRequest):
    """POST /api/xai/dashboard/history"""
    from tollama.xai.trust_history import TrustHistoryTracker, default_history_path

    path = default_history_path()
    tracker = TrustHistoryTracker.load(path)

    all_domains = tracker.domains or [
        "prediction_market", "financial_market", "supply_chain",
        "news", "geopolitical", "regulatory",
    ]
    domains = request.domains or all_domains

    result: dict[str, Any] = {"domains": {}}
    for domain in domains:
        entry: dict[str, Any] = {
            "history": [
                r.model_dump(mode="json")
                for r in tracker.get_history(domain, limit=request.limit)
            ],
        }
        if request.include_stats:
            entry["stats"] = tracker.get_stats(domain).model_dump(mode="json")
        result["domains"][domain] = entry

    return result


# ──────────────────────────────────────────────────────────────
# Connector Health
# ──────────────────────────────────────────────────────────────


@router.post(
    "/connectors/health",
    summary="Check connector availability",
    description="Returns health status for registered data connectors.",
)
async def connectors_health(request: ConnectorHealthRequest):
    """POST /api/xai/connectors/health"""
    from tollama.xai.connectors.helpers import (
        build_default_connector_registry,
    )

    registry = build_default_connector_registry()
    all_domains = {
        "financial_market", "news", "supply_chain",
        "geopolitical", "regulatory",
    }
    domains = set(request.domains) if request.domains else all_domains

    results = []
    for connector in registry.connectors:
        if connector.domain not in domains:
            continue
        status = "available"
        try:
            connector.supports("__health_check__", {})
        except Exception:  # noqa: BLE001
            status = "error"
        results.append({
            "connector_name": connector.connector_name,
            "domain": connector.domain,
            "status": status,
        })

    return {"connectors": results}


# ──────────────────────────────────────────────────────────────
# Cache Stats
# ──────────────────────────────────────────────────────────────


@router.get(
    "/cache/stats",
    response_model=CacheStatsResponse,
    summary="Trust cache metrics",
    description="Returns trust result cache hit/miss statistics and TTL configuration.",
)
async def cache_stats(raw_request: Request):
    """GET /api/xai/cache/stats"""
    router_instance = _get_trust_router(raw_request)
    return router_instance.cache_stats()


class CacheTTLRequest(BaseModel):
    """Request body for /api/xai/cache/ttl"""
    ttl: float = Field(
        ..., ge=0.0,
        description="New cache TTL in seconds. 0 disables caching.",
    )


@router.put(
    "/cache/ttl",
    response_model=CacheTTLResponse,
    summary="Configure cache TTL",
    description="Update the trust result cache TTL. Set to 0 to disable caching.",
)
async def set_cache_ttl(request: CacheTTLRequest, raw_request: Request):
    """PUT /api/xai/cache/ttl"""
    router_instance = _get_trust_router(raw_request)
    old_ttl = router_instance.set_cache_ttl(request.ttl)
    return {
        "previous_ttl": old_ttl,
        "new_ttl": request.ttl,
    }


@router.delete(
    "/cache/invalidate",
    response_model=CacheInvalidateResponse,
    summary="Clear trust cache",
    description="Invalidate all cached trust results. Returns number of entries cleared.",
)
async def invalidate_cache(raw_request: Request):
    """DELETE /api/xai/cache/invalidate"""
    router_instance = _get_trust_router(raw_request)
    cleared = router_instance.clear_cache()
    return {"cleared": cleared}


# ──────────────────────────────────────────────────────────────
# Trust Gate (auto-execution gating)
# ──────────────────────────────────────────────────────────────


class GateDecisionRequest(BaseModel):
    """Request body for /api/xai/gate-decision"""
    context: dict[str, Any] = Field(
        ..., description="Trust agent routing context (must include 'domain').",
    )
    payload: dict[str, Any] = Field(
        ..., description="Payload for trust analysis.",
    )
    trust_threshold: float = Field(
        default=0.5, ge=0.0, le=1.0,
        description="Minimum trust score for auto-execution.",
    )


@router.post(
    "/gate-decision",
    response_model=GateDecisionResponse,
    summary="Trust-gated auto-execution check",
    description=(
        "Runs trust analysis and evaluates three safety gates "
        "(trust score, constraint violations, risk category) to determine "
        "whether auto-execution should proceed."
    ),
)
async def gate_decision(request: GateDecisionRequest, raw_request: Request):
    """POST /api/xai/gate-decision"""
    router_instance = _get_trust_router(raw_request)
    result = router_instance.analyze(
        context=request.context,
        payload=request.payload,
    )
    if result is None:
        return {
            "allowed": False,
            "gates": [{"gate": "no_agent", "status": "BLOCK", "detail": "no matching agent"}],
            "risk_category": None,
            "trust_score": None,
        }
    return router_instance.gate_decision(
        result, trust_threshold=request.trust_threshold,
    )


# ──────────────────────────────────────────────────────────────
# Trust Feature Attribution
# ──────────────────────────────────────────────────────────────


class TrustAttributionRequest(BaseModel):
    """Request body for /api/xai/trust-attribution"""
    context: dict[str, Any] = Field(
        ..., description="Trust agent routing context (must include 'domain').",
    )
    payload: dict[str, Any] = Field(
        ..., description="Payload for trust analysis.",
    )


@router.post(
    "/trust-attribution",
    response_model=TrustAttributionResponse,
    summary="SHAP-like trust component attribution",
    description=(
        "Runs trust analysis and computes feature attribution from "
        "trust component weights, showing each component's contribution "
        "to the overall trust score."
    ),
)
async def trust_attribution(request: TrustAttributionRequest, raw_request: Request):
    """POST /api/xai/trust-attribution"""
    router_instance = _get_trust_router(raw_request)
    result = router_instance.analyze(
        context=request.context,
        payload=request.payload,
    )
    if result is None:
        raise HTTPException(status_code=404, detail="No matching trust agent")
    return router_instance.trust_feature_attribution(result)


# ──────────────────────────────────────────────────────────────
# Batch Trust Analysis
# ──────────────────────────────────────────────────────────────


class BatchAnalyzeItem(BaseModel):
    """Single item in a batch trust analysis request."""
    context: dict[str, Any] = Field(
        ..., description="Trust agent routing context.",
    )
    payload: dict[str, Any] = Field(
        ..., description="Payload for trust analysis.",
    )


class BatchAnalyzeRequest(BaseModel):
    """Request body for /api/xai/batch-analyze"""
    items: list[BatchAnalyzeItem] = Field(
        ..., min_length=1, max_length=100,
        description="List of trust analysis requests (max 100).",
    )


@router.post(
    "/batch-analyze",
    response_model=BatchAnalyzeResponse,
    summary="Batch trust analysis",
    description="Evaluate multiple trust contexts in a single request using concurrent execution.",
)
async def batch_analyze(request: BatchAnalyzeRequest, raw_request: Request):
    """POST /api/xai/batch-analyze"""
    import asyncio

    router_instance = _get_trust_router(raw_request)

    async def _analyze_one(item: BatchAnalyzeItem) -> dict[str, Any]:
        result = await router_instance.analyze_async(
            context=item.context, payload=item.payload,
        )
        if result is None:
            return {"status": "no_agent", "result": None}
        return {"status": "ok", "result": result.to_summary()}

    results = await asyncio.gather(*[_analyze_one(item) for item in request.items])
    return {"results": list(results), "count": len(results)}


# ──────────────────────────────────────────────────────────────
# Trust Alert / Threshold Webhook
# ──────────────────────────────────────────────────────────────


class AlertThreshold(BaseModel):
    """A single alert threshold configuration."""
    domain: str = Field(..., min_length=1, description="Domain to monitor.")
    min_trust_score: float = Field(
        default=0.5, ge=0.0, le=1.0,
        description="Alert when trust score drops below this.",
    )
    risk_categories: list[str] = Field(
        default_factory=lambda: ["RED"],
        description="Alert on these risk categories.",
    )
    alert_on_trend: list[str] = Field(
        default_factory=list,
        description="Alert on these trends: 'declining', 'stable', 'improving'.",
    )
    webhook_url: str | None = Field(
        None, description="URL to POST alert payload. Omit for log-only.",
    )


class AlertConfigRequest(BaseModel):
    """Request body for /api/xai/alerts/configure"""
    thresholds: list[AlertThreshold] = Field(
        ..., min_length=1, max_length=20,
        description="Alert threshold configurations.",
    )


class AlertCheckRequest(BaseModel):
    """Request body for /api/xai/alerts/check"""
    context: dict[str, Any] = Field(
        ..., description="Trust agent routing context.",
    )
    payload: dict[str, Any] = Field(
        ..., description="Payload for trust analysis.",
    )


_WEBHOOK_MAX_RETRIES = 3
_WEBHOOK_BASE_DELAY = 0.5  # seconds


def _fire_webhook(
    url: str,
    payload: dict[str, Any],
    logger: Any,
) -> bool:
    """Fire a webhook with exponential backoff retry.

    Returns True if delivered successfully, False otherwise.
    """
    import time

    import httpx

    for attempt in range(_WEBHOOK_MAX_RETRIES):
        try:
            resp = httpx.post(url, json=payload, timeout=5.0)
            if resp.status_code < 500:
                return True
            logger.warning(
                "Webhook to %s returned %d (attempt %d/%d)",
                url, resp.status_code, attempt + 1, _WEBHOOK_MAX_RETRIES,
            )
        except Exception:  # noqa: BLE001
            logger.warning(
                "Webhook to %s failed (attempt %d/%d)",
                url, attempt + 1, _WEBHOOK_MAX_RETRIES,
                exc_info=True,
            )
        if attempt < _WEBHOOK_MAX_RETRIES - 1:
            time.sleep(_WEBHOOK_BASE_DELAY * (2 ** attempt))
    return False


# Module-level alert config (shared within the app via singleton router pattern)
_alert_thresholds: list[AlertThreshold] = []


@router.post(
    "/alerts/configure",
    response_model=AlertConfigResponse,
    summary="Configure trust alert thresholds",
    description="Set up alert thresholds for trust score drops and risk escalations.",
)
async def configure_alerts(request: AlertConfigRequest):
    """POST /api/xai/alerts/configure"""
    global _alert_thresholds  # noqa: PLW0603
    _alert_thresholds = list(request.thresholds)
    return {
        "status": "configured",
        "threshold_count": len(_alert_thresholds),
        "thresholds": [t.model_dump() for t in _alert_thresholds],
    }


@router.get(
    "/alerts/config",
    response_model=AlertConfigResponse,
    summary="Get current alert configuration",
    description="Returns the currently configured alert thresholds.",
)
async def get_alert_config():
    """GET /api/xai/alerts/config"""
    return {
        "threshold_count": len(_alert_thresholds),
        "thresholds": [t.model_dump() for t in _alert_thresholds],
    }


@router.post(
    "/alerts/check",
    response_model=AlertCheckResponse,
    summary="Check trust against alert thresholds",
    description=(
        "Runs trust analysis and checks the result against configured "
        "alert thresholds. Returns any triggered alerts."
    ),
)
async def check_alerts(request: AlertCheckRequest, raw_request: Request):
    """POST /api/xai/alerts/check"""
    import logging

    _log = logging.getLogger(__name__)

    router_instance = _get_trust_router(raw_request)
    result = router_instance.analyze(
        context=request.context,
        payload=request.payload,
    )
    if result is None:
        return {"alerts": [], "alert_count": 0, "trust_result": None}

    # Get trend from history tracker if available
    current_trend: str | None = None
    if router_instance.history_tracker is not None:
        stats = router_instance.history_tracker.get_stats(result.domain)
        current_trend = stats.trend

    triggered: list[dict[str, Any]] = []
    for threshold in _alert_thresholds:
        if threshold.domain != result.domain:
            continue

        reasons: list[str] = []
        if result.trust_score < threshold.min_trust_score:
            reasons.append(
                f"trust_score {result.trust_score:.2f} "
                f"< threshold {threshold.min_trust_score:.2f}"
            )
        if result.risk_category in threshold.risk_categories:
            reasons.append(f"risk_category {result.risk_category}")
        if current_trend and threshold.alert_on_trend and current_trend in threshold.alert_on_trend:
            reasons.append(f"trend {current_trend}")

        if reasons:
            alert = {
                "domain": threshold.domain,
                "trust_score": result.trust_score,
                "risk_category": result.risk_category,
                "trend": current_trend,
                "reasons": reasons,
                "webhook_url": threshold.webhook_url,
            }
            triggered.append(alert)

            # Fire webhook if configured (with exponential backoff retry)
            if threshold.webhook_url:
                _fire_webhook(threshold.webhook_url, alert, _log)

    return {
        "alerts": triggered,
        "alert_count": len(triggered),
        "trust_result": {
            "agent_name": result.agent_name,
            "domain": result.domain,
            "trust_score": result.trust_score,
            "risk_category": result.risk_category,
            "trend": current_trend,
        },
    }




