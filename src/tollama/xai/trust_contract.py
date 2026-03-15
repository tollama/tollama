"""
tollama.xai.trust_contract — Normalized contract for domain trust agents.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class TrustComponent(BaseModel):
    """Normalized component-level trust contribution."""

    model_config = ConfigDict(extra="allow")

    score: float = Field(ge=0.0, le=1.0)
    weight: float = Field(default=0.0, ge=0.0)
    value: Any = None
    assessment: str | None = None
    rationale: str | None = None


class TrustViolation(BaseModel):
    """Normalized policy violation or warning."""

    model_config = ConfigDict(extra="allow")

    name: str = Field(min_length=1)
    severity: str = Field(default="warning")
    type: str | None = None
    detail: str | None = None


class TrustEvidence(BaseModel):
    """Evidence metadata for audit and provenance."""

    model_config = ConfigDict(extra="allow")

    source_type: str = Field(default="unknown")
    source_ids: list[str] = Field(default_factory=list)
    freshness_seconds: float | None = Field(default=None, ge=0.0)
    payload_schema: str | None = None
    attributes: dict[str, Any] = Field(default_factory=dict)


class TrustAudit(BaseModel):
    """Audit metadata for generated trust results."""

    model_config = ConfigDict(extra="allow")

    formula_version: str = Field(default="v1")
    generated_at: str = Field(default_factory=_utc_now_iso)
    agent_version: str = Field(default="0.1.0")


class NormalizedTrustResult(BaseModel):
    """Common output schema for all domain trust agents."""

    model_config = ConfigDict(extra="allow")

    agent_name: str = Field(min_length=1)
    domain: str = Field(min_length=1)
    trust_score: float = Field(ge=0.0, le=1.0)
    risk_category: str = Field(default="YELLOW")
    calibration_status: str | None = None
    component_breakdown: dict[str, TrustComponent] = Field(default_factory=dict)
    violations: list[TrustViolation] = Field(default_factory=list)
    why_trusted: str = Field(default="")
    evidence: TrustEvidence = Field(default_factory=TrustEvidence)
    audit: TrustAudit = Field(default_factory=TrustAudit)


@runtime_checkable
class TrustAgent(Protocol):
    """Protocol for in-repo domain trust agents."""

    agent_name: str
    domain: str
    priority: int

    def supports(self, context: dict[str, Any]) -> bool: ...

    def analyze(self, payload: dict[str, Any]) -> NormalizedTrustResult | dict[str, Any]: ...


def coerce_normalized_trust_result(
    value: NormalizedTrustResult | dict[str, Any],
) -> NormalizedTrustResult:
    """Validate and normalize arbitrary trust result payloads."""
    if isinstance(value, NormalizedTrustResult):
        return value
    return NormalizedTrustResult.model_validate(value)


def normalized_result_to_legacy_metadata(
    value: NormalizedTrustResult | dict[str, Any],
) -> dict[str, Any]:
    """Convert normalized trust output into legacy trust metadata."""
    result = coerce_normalized_trust_result(value)
    weights = {
        name: component.weight for name, component in result.component_breakdown.items()
    }
    components = {
        name: component.score for name, component in result.component_breakdown.items()
    }
    components["risk_category"] = result.risk_category
    components["constraint_satisfied"] = not any(
        violation.severity == "critical" for violation in result.violations
    )
    return {
        "trust_intelligence": {
            "agent_name": result.agent_name,
            "domain": result.domain,
            "version": result.audit.agent_version,
            "trust_score": result.trust_score,
            "calibration_status": result.calibration_status or _derive_calibration_status(
                result.trust_score
            ),
            "weights": weights,
            "components": components,
            "shap_top_features": [],
            "violations": [violation.model_dump() for violation in result.violations],
            "meta_metrics": {},
            "why_trusted": result.why_trusted,
            "evidence": result.evidence.model_dump(),
            "audit": result.audit.model_dump(),
        }
    }


def normalized_result_to_breakdown(
    value: NormalizedTrustResult | dict[str, Any],
) -> dict[str, Any]:
    """Format normalized trust output into the existing trust_breakdown shape."""
    result = coerce_normalized_trust_result(value)
    return {
        "trust_scores": {result.agent_name: result.trust_score},
        "breakdowns": {
            result.agent_name: {
                name: component.model_dump()
                for name, component in result.component_breakdown.items()
            }
        },
        "why_trusted": {result.agent_name: result.why_trusted},
        "recommendations": [],
        "trust_level_thresholds": {
            "high": 0.75,
            "medium": 0.50,
            "low": 0.25,
        },
    }


def _derive_calibration_status(score: float) -> str:
    if score >= 0.75:
        return "well_calibrated"
    if score >= 0.50:
        return "moderately_calibrated"
    return "poorly_calibrated"


__all__ = [
    "NormalizedTrustResult",
    "TrustAgent",
    "TrustAudit",
    "TrustComponent",
    "TrustEvidence",
    "TrustViolation",
    "coerce_normalized_trust_result",
    "normalized_result_to_breakdown",
    "normalized_result_to_legacy_metadata",
]
