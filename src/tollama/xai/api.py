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

from fastapi import APIRouter, Query
from pydantic import BaseModel, Field

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
        None, description="Context for constraint verification in trust pipeline"
    )
    trust_payload: Optional[dict[str, Any]] = Field(
        None, description="Payload routed through the default trust-agent registry"
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

    result = engine.explain_decision(
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
