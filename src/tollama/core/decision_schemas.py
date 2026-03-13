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
    kind: EvidenceKind
    label: NonEmptyStr
    detail: NonEmptyStr | None = None
    value: JsonValue | None = None
    source: NonEmptyStr | None = None


class TemporalImportancePoint(CanonicalModel):
    lag: StrictInt = Field(ge=1)
    timestamp: NonEmptyStr | None = None
    value: StrictFloat
    importance: StrictFloat = Field(ge=0.0, le=1.0)
    direction: ImportanceDirection = "mixed"
    reason: NonEmptyStr


class ForecastDecomposition(CanonicalModel):
    trend: StrictFloat = Field(ge=0.0, le=1.0)
    seasonal: StrictFloat = Field(ge=0.0, le=1.0)
    residual: StrictFloat = Field(ge=0.0, le=1.0)
    dominant_driver: NonEmptyStr


class SignalTrustInput(CanonicalModel):
    name: NonEmptyStr
    trust_score: StrictFloat = Field(ge=0.0, le=1.0)
    metrics: dict[NonEmptyStr, StrictFloat] = Field(default_factory=dict)
    rationale: list[NonEmptyStr] = Field(default_factory=list)
    source: NonEmptyStr | None = None


class SignalTrustExplanation(CanonicalModel):
    name: NonEmptyStr
    trust_score: StrictFloat = Field(ge=0.0, le=1.0)
    confidence_level: ConfidenceLevel
    why_trusted: NonEmptyStr
    evidence: list[ExplanationEvidence] = Field(default_factory=list)


class InputExplanation(CanonicalModel):
    signals_used: list[NonEmptyStr] = Field(default_factory=list)
    signal_explanations: list[SignalTrustExplanation] = Field(default_factory=list)
    why_this_input: list[NonEmptyStr] = Field(default_factory=list)


class ModelCandidateExplanation(CanonicalModel):
    model: NonEmptyStr
    rank: StrictInt = Field(ge=1)
    score: StrictFloat
    reasons: list[NonEmptyStr] = Field(default_factory=list)


class PlanSeriesExplanation(CanonicalModel):
    id: NonEmptyStr
    trend_direction: TrendDirection
    confidence_level: ConfidenceLevel
    temporal_importance: list[TemporalImportancePoint] = Field(default_factory=list)
    forecast_decomposition: ForecastDecomposition
    evidence: list[ExplanationEvidence] = Field(default_factory=list)


class PlanExplanation(CanonicalModel):
    model_selected: NonEmptyStr
    why_this_model: NonEmptyStr
    selection_rationale: list[NonEmptyStr] = Field(default_factory=list)
    candidates: list[ModelCandidateExplanation] = Field(default_factory=list)
    series: list[PlanSeriesExplanation] = Field(default_factory=list)


class DecisionPolicyInput(CanonicalModel):
    confidence: StrictFloat | None = Field(default=None, ge=0.0, le=1.0)
    threshold: StrictFloat | None = Field(default=None, ge=0.0, le=1.0)
    auto_execute: StrictBool = False
    human_override: StrictBool = True
    override_available: StrictBool = True
    rationale: list[NonEmptyStr] = Field(default_factory=list)


class DecisionPolicyExplanation(CanonicalModel):
    auto_executed: StrictBool
    confidence: StrictFloat | None = Field(default=None, ge=0.0, le=1.0)
    threshold: StrictFloat | None = Field(default=None, ge=0.0, le=1.0)
    reason: NonEmptyStr
    human_override: StrictBool
    evidence: list[ExplanationEvidence] = Field(default_factory=list)


class DecisionExplanationRequest(CanonicalModel):
    request: ForecastRequest
    response: ForecastResponse
    selection: AutoSelectionInfo | None = None
    signal_trust: list[SignalTrustInput] = Field(default_factory=list)
    policy: DecisionPolicyInput | None = None


class DecisionExplanationResponse(CanonicalModel):
    input_explanation: InputExplanation
    plan_explanation: PlanExplanation
    decision_policy: DecisionPolicyExplanation
