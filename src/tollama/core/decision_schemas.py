"""Structured decision-explanation schemas for the v3.8 trust layer."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, StrictBool, StrictFloat, StrictInt

from tollama.core.schemas import (
    AutoSelectionInfo,
    CanonicalModel,
    ConfidenceLevel,
    ForecastRequest,
    ForecastResponse,
    JsonValue,
    NonEmptyStr,
    TrendDirection,
)

EvidenceKind = Literal[
    "input",
    "signal",
    "selection",
    "forecast",
    "policy",
    "metric",
    "warning",
    "diagnostic",
]
ImportanceDirection = Literal["positive", "negative", "mixed", "neutral"]


class ExplanationEvidence(CanonicalModel):
    kind: EvidenceKind = Field(
        description="Category of evidence (input, signal, policy, etc.)",
    )
    label: NonEmptyStr = Field(
        description="Human-readable label for this evidence item",
    )
    detail: NonEmptyStr | None = Field(
        default=None, description="Extended detail text",
    )
    value: JsonValue | None = Field(
        default=None, description="Structured value for this evidence",
    )
    source: NonEmptyStr | None = Field(
        default=None, description="Origin source of the evidence",
    )


class TemporalImportancePoint(CanonicalModel):
    lag: StrictInt = Field(ge=1, description="Time lag in periods")
    timestamp: NonEmptyStr | None = Field(
        default=None, description="ISO timestamp for this point",
    )
    value: StrictFloat = Field(description="Observed value at this lag")
    importance: StrictFloat = Field(
        ge=0.0, le=1.0, description="Importance weight (0-1)",
    )
    direction: ImportanceDirection = Field(
        default="mixed", description="Direction of influence",
    )
    reason: NonEmptyStr = Field(description="Why this lag matters")


class ForecastDecomposition(CanonicalModel):
    trend: StrictFloat = Field(
        ge=0.0, le=1.0, description="Trend component weight (0-1)",
    )
    seasonal: StrictFloat = Field(
        ge=0.0, le=1.0, description="Seasonal component weight (0-1)",
    )
    residual: StrictFloat = Field(
        ge=0.0, le=1.0, description="Residual component weight (0-1)",
    )
    dominant_driver: NonEmptyStr = Field(
        description="Primary driver of forecast movement",
    )


class SignalTrustInput(CanonicalModel):
    name: NonEmptyStr = Field(description="Signal name identifier")
    trust_score: StrictFloat = Field(
        ge=0.0, le=1.0, description="Trust score (0-1)",
    )
    metrics: dict[NonEmptyStr, StrictFloat] = Field(
        default_factory=dict, description="Calibration metrics",
    )
    rationale: list[NonEmptyStr] = Field(
        default_factory=list, description="Trust assessment reasons",
    )
    source: NonEmptyStr | None = Field(
        default=None, description="Data source for the signal",
    )


class SignalTrustExplanation(CanonicalModel):
    name: NonEmptyStr = Field(description="Signal name identifier")
    trust_score: StrictFloat = Field(
        ge=0.0, le=1.0, description="Trust score (0-1)",
    )
    confidence_level: ConfidenceLevel = Field(
        description="Confidence level classification",
    )
    why_trusted: NonEmptyStr = Field(
        description="Human-readable trust rationale",
    )
    evidence: list[ExplanationEvidence] = Field(
        default_factory=list, description="Supporting evidence",
    )


class InputExplanation(CanonicalModel):
    signals_used: list[NonEmptyStr] = Field(
        default_factory=list, description="Signals consumed",
    )
    signal_explanations: list[SignalTrustExplanation] = Field(
        default_factory=list, description="Per-signal explanations",
    )
    why_this_input: list[NonEmptyStr] = Field(
        default_factory=list, description="Input selection reasons",
    )


class ModelCandidateExplanation(CanonicalModel):
    model: NonEmptyStr = Field(description="Model identifier")
    rank: StrictInt = Field(ge=1, description="Selection rank (1 = best)")
    score: StrictFloat = Field(description="Model selection score")
    reasons: list[NonEmptyStr] = Field(
        default_factory=list, description="Reasons for ranking",
    )


class PlanSeriesExplanation(CanonicalModel):
    id: NonEmptyStr = Field(description="Series identifier")
    trend_direction: TrendDirection = Field(
        description="Detected trend direction",
    )
    confidence_level: ConfidenceLevel = Field(
        description="Confidence level classification",
    )
    temporal_importance: list[TemporalImportancePoint] = Field(
        default_factory=list, description="Temporal importance breakdown",
    )
    forecast_decomposition: ForecastDecomposition = Field(
        description="Trend/seasonal/residual decomposition",
    )
    evidence: list[ExplanationEvidence] = Field(
        default_factory=list, description="Supporting evidence",
    )


class PlanExplanation(CanonicalModel):
    model_selected: NonEmptyStr = Field(
        description="Name of the selected model",
    )
    why_this_model: NonEmptyStr = Field(
        description="Rationale for model selection",
    )
    selection_rationale: list[NonEmptyStr] = Field(
        default_factory=list, description="Selection reasons",
    )
    candidates: list[ModelCandidateExplanation] = Field(
        default_factory=list, description="Candidate models considered",
    )
    series: list[PlanSeriesExplanation] = Field(
        default_factory=list, description="Per-series plan explanations",
    )


class DecisionPolicyInput(CanonicalModel):
    confidence: StrictFloat | None = Field(
        default=None, ge=0.0, le=1.0,
        description="Current confidence level (0-1)",
    )
    threshold: StrictFloat | None = Field(
        default=None, ge=0.0, le=1.0,
        description="Auto-execution threshold (0-1)",
    )
    auto_execute: StrictBool = Field(
        default=False, description="Whether auto-execution is enabled",
    )
    human_override: StrictBool = Field(
        default=True, description="Whether human override is allowed",
    )
    override_available: StrictBool = Field(
        default=True, description="Whether override is available",
    )
    rationale: list[NonEmptyStr] = Field(
        default_factory=list, description="Policy decision rationale",
    )


class DecisionPolicyExplanation(CanonicalModel):
    auto_executed: StrictBool = Field(
        description="Whether the decision was auto-executed",
    )
    confidence: StrictFloat | None = Field(
        default=None, ge=0.0, le=1.0,
        description="Confidence at decision time (0-1)",
    )
    threshold: StrictFloat | None = Field(
        default=None, ge=0.0, le=1.0,
        description="Auto-execution threshold (0-1)",
    )
    reason: NonEmptyStr = Field(
        description="Human-readable decision reason",
    )
    human_override: StrictBool = Field(
        description="Whether human override was available",
    )
    evidence: list[ExplanationEvidence] = Field(
        default_factory=list, description="Supporting evidence",
    )


class DecisionExplanationRequest(CanonicalModel):
    request: ForecastRequest = Field(
        description="Original forecast request",
    )
    response: ForecastResponse = Field(
        description="Forecast response to explain",
    )
    selection: AutoSelectionInfo | None = Field(
        default=None, description="Auto-selection metadata",
    )
    signal_trust: list[SignalTrustInput] = Field(
        default_factory=list, description="Signal trust inputs",
    )
    policy: DecisionPolicyInput | None = Field(
        default=None, description="Decision policy configuration",
    )


class TrustIntelligenceEvidence(CanonicalModel):
    """v3.0 Trust Intelligence Pipeline evidence (optional)."""

    trust_score: StrictFloat = Field(
        ge=0.0, le=1.0, description="Overall trust score (0-1)",
    )
    calibration_status: NonEmptyStr = Field(
        description="Calibration status label",
    )
    weights: dict[str, float] = Field(
        default_factory=dict, description="Component weight map",
    )
    components: dict[str, float] = Field(
        default_factory=dict, description="Component score map",
    )
    risk_category: NonEmptyStr = Field(
        default="GREEN", description="Risk category (GREEN/YELLOW/RED)",
    )
    constraint_satisfied: StrictBool = Field(
        default=True, description="Whether constraints are satisfied",
    )
    shap_top_features: list[dict[str, JsonValue]] = Field(
        default_factory=list, description="Top SHAP attributions",
    )
    violations: list[dict[str, JsonValue]] = Field(
        default_factory=list, description="Trust constraint violations",
    )
    ece: StrictFloat = Field(
        default=0.0, description="Expected calibration error",
    )
    ocr: StrictFloat = Field(
        default=0.0, description="Overconfidence ratio",
    )
    pipeline_version: NonEmptyStr = Field(
        default="3.0", description="Pipeline version",
    )


class DecisionExplanationResponse(CanonicalModel):
    input_explanation: InputExplanation = Field(
        description="Input-stage explanation",
    )
    plan_explanation: PlanExplanation = Field(
        description="Plan-stage explanation",
    )
    decision_policy: DecisionPolicyExplanation = Field(
        description="Decision policy explanation",
    )
    trust_intelligence: TrustIntelligenceEvidence | None = Field(
        default=None, description="Trust Intelligence Pipeline evidence",
    )
