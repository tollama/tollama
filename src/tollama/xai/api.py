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

from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field, ValidationError

router = APIRouter(prefix="/api/xai", tags=["xai"])


# ──────────────────────────────────────────────────────────────
# Request / Response Models
# ──────────────────────────────────────────────────────────────

class ExplainDecisionRequest(BaseModel):
    """Request body for /api/explain-decision"""
    forecast_result: dict[str, Any] = Field(
        ..., description="Output from tollama forecast endpoint"
    )
    eval_result: Optional[dict[str, Any]] = Field(
        None, description="Output from tollama-eval"
    )
    calibration_result: Optional[dict[str, Any]] = Field(
        None, description="Output from Market Calibration Agent"
    )
    trust_result: Optional[dict[str, Any]] = Field(
        None, description="Normalized output from a domain trust agent"
    )
    policy_config: Optional[dict[str, Any]] = Field(
        None, description="Decision policy configuration"
    )
    time_series_data: Optional[list[float]] = Field(
        None, description="Raw time series for decomposition"
    )
    explain_options: Optional[dict[str, Any]] = Field(
        default_factory=lambda: {"decompose": True, "attribution": False},
        description="Control explanation depth",
    )
    trust_features: Optional[dict[str, float]] = Field(
        None, description="Features for SHAP attribution in trust pipeline"
    )
    trust_context: Optional[dict[str, Any]] = Field(
        None,
        description=(
            "Context for trust agent routing. Keys: 'domain' (prediction_market|"
            "financial_market|supply_chain|news|geopolitical|regulatory), "
            "'source_type' (optional refinement), "
            "'mode' ('single' default or 'multi' for multi-agent aggregation)."
        ),
    )
    trust_payload: Optional[dict[str, Any]] = Field(
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
    explanation_id: str
    timestamp: str
    version: str
    input_explanation: dict[str, Any]
    plan_explanation: dict[str, Any]
    decision_policy_explanation: dict[str, Any]
    trust_intelligence_explanation: Optional[dict[str, Any]] = None
    metadata: dict[str, Any]


class TrustBreakdownRequest(BaseModel):
    """Request body for /api/trust-breakdown"""
    trust_score: float = Field(..., description="Overall trust score")
    metrics: dict[str, float] = Field(
        ..., description="Calibration metrics (brier_score, log_loss, ece, ...)"
    )
    source: str = Field(default="polymarket", description="Signal source name")
    signals: Optional[list[dict[str, Any]]] = Field(
        None, description="Multiple signals"
    )


class ForecastDecomposeRequest(BaseModel):
    """Request body for /api/forecast-decompose"""
    data: list[float] = Field(..., description="Time series values")
    period: Optional[int] = Field(None, description="Seasonal period (auto-detect if None)")
    method: str = Field(default="stl", description="Decomposition method")


class ModelCardRequest(BaseModel):
    """Request body for /api/model-card"""
    model_info: dict[str, Any] = Field(
        ..., description="Model identity information"
    )
    eval_result: Optional[dict[str, Any]] = None
    explanation_result: Optional[dict[str, Any]] = None
    governance_info: Optional[dict[str, Any]] = None
    format: str = Field(default="json", description="Output format: json or markdown")


class DecisionReportRequest(BaseModel):
    """Request body for /api/decision-report"""
    explanation: dict[str, Any] = Field(
        ..., description="Output from /api/explain-decision"
    )
    forecast_result: Optional[dict[str, Any]] = None
    report_config: Optional[dict[str, Any]] = Field(
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
    domains: Optional[list[str]] = Field(
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
    component_scores: Optional[dict[str, float]] = Field(
        default_factory=dict,
        description="Per-component scores for calibration learning.",
    )


class TrustHistoryRequest(BaseModel):
    """Request body for /api/xai/dashboard/history"""
    domains: Optional[list[str]] = Field(
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
    domains: Optional[list[str]] = Field(
        None,
        description="Filter to specific domains. Omit for all.",
    )


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
async def explain_decision(request: ExplainDecisionRequest):
    """
    POST /api/explain-decision

    v3.8 Target — 핵심 XAI 엔드포인트
    Input → Plan → Decision Policy 전 단계 evidence-backed explanation.
    """
    from tollama.xai.engine import ExplanationEngine
    from tollama.xai.model_selection import ModelSelectionExplainer
    from tollama.xai.forecast_decompose import ForecastDecomposer
    from tollama.xai.feature_attribution import TemporalFeatureAttribution
    from tollama.xai.scenario_rationale import ScenarioRationale
    from tollama.xai.trust_breakdown import TrustBreakdown
    from tollama.xai.decision_policy import DecisionPolicyExplainer
    from tollama.xai.trust_router import build_default_trust_router

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
        trust_router=build_default_trust_router(),
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
async def dashboard_agents():
    """GET /api/xai/dashboard/agents"""
    from tollama.xai.trust_router import build_default_trust_router

    router_instance = build_default_trust_router(enable_calibration=False)
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
async def dashboard_trust(request: DashboardTrustRequest):
    """POST /api/xai/dashboard/trust"""
    from tollama.xai.trust_router import build_default_trust_router

    router_instance = build_default_trust_router(
        enable_calibration=request.include_calibration,
    )

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
async def record_outcome(request: RecordOutcomeRequest):
    """POST /api/xai/record-outcome"""
    from tollama.xai.trust_router import build_default_trust_router

    router_instance = build_default_trust_router(enable_calibration=True)
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
