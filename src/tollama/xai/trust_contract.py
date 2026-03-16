"""
tollama.xai.trust_contract — Normalized contract for domain trust agents.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Protocol, runtime_checkable

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


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

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict suitable for JSON responses."""
        return {
            "agent_name": self.agent_name,
            "domain": self.domain,
            "trust_score": self.trust_score,
            "risk_category": self.risk_category,
            "calibration_status": self.calibration_status,
            "why_trusted": self.why_trusted,
            "violations": [
                {"name": v.name, "severity": v.severity, "detail": v.detail}
                for v in self.violations
            ],
            "component_breakdown": {
                name: {"score": c.score, "weight": c.weight}
                for name, c in self.component_breakdown.items()
            },
            "evidence": self.evidence.model_dump(mode="json"),
            "audit": self.audit.model_dump(mode="json"),
        }

    def to_summary(self) -> dict[str, Any]:
        """Compact summary with just the key fields."""
        return {
            "agent_name": self.agent_name,
            "domain": self.domain,
            "trust_score": self.trust_score,
            "risk_category": self.risk_category,
            "calibration_status": self.calibration_status,
            "why_trusted": self.why_trusted,
            "violations": [
                {"name": v.name, "severity": v.severity, "detail": v.detail}
                for v in self.violations
            ],
        }


class FinancialTrustPayload(BaseModel):
    """Normalized payload for financial market trust analysis."""

    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
        str_strip_whitespace=True,
    )

    instrument_id: str = Field(
        min_length=1,
        validation_alias=AliasChoices("instrument_id", "symbol"),
    )
    liquidity_depth: float = Field(default=0.5)
    bid_ask_spread_bps: float = Field(
        default=50.0,
        ge=0.0,
        validation_alias=AliasChoices("bid_ask_spread_bps", "spread_bps"),
    )
    realized_volatility: float = Field(
        default=0.3,
        ge=0.0,
        validation_alias=AliasChoices("realized_volatility", "volatility_regime"),
    )
    execution_risk: float = Field(
        default=0.5,
        validation_alias=AliasChoices("execution_risk", "slippage_risk"),
    )
    data_freshness: float = Field(
        default=0.5,
        validation_alias=AliasChoices("data_freshness", "freshness_score"),
    )
    news_signal_ref: str | None = None

    @field_validator("liquidity_depth", "execution_risk", "data_freshness", mode="before")
    @classmethod
    def _clip_unit_fields(cls, value: Any) -> float:
        return _clip_unit(value, default=0.5)


class NewsTrustPayload(BaseModel):
    """Normalized payload for news trust analysis."""

    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
        str_strip_whitespace=True,
    )

    story_id: str = Field(
        min_length=1,
        validation_alias=AliasChoices("story_id", "headline"),
    )
    source_credibility: float = Field(default=0.5)
    corroboration: float = Field(
        default=0.5,
        validation_alias=AliasChoices("corroboration", "source_agreement"),
    )
    contradiction_score: float = Field(default=0.5)
    propagation_delay_seconds: float = Field(default=300.0, ge=0.0)
    freshness_score: float = Field(default=0.5)
    novelty: float | None = None

    @field_validator(
        "source_credibility",
        "corroboration",
        "contradiction_score",
        "freshness_score",
        mode="before",
    )
    @classmethod
    def _clip_news_unit_fields(cls, value: Any) -> float:
        return _clip_unit(value, default=0.5)

    @field_validator("novelty", mode="before")
    @classmethod
    def _clip_optional_novelty(cls, value: Any) -> float | None:
        if value is None:
            return None
        return _clip_unit(value, default=0.5)


class SupplyChainTrustPayload(BaseModel):
    """Normalized payload for supply chain trust analysis."""

    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
        str_strip_whitespace=True,
    )

    network_id: str = Field(
        min_length=1,
        validation_alias=AliasChoices("network_id", "shipment_id"),
    )
    lead_time_reliability: float = Field(default=0.5)
    inventory_visibility: float = Field(default=0.5)
    disruption_risk: float = Field(default=0.5)
    sensor_quality: float = Field(default=0.5)
    data_freshness: float = Field(
        default=0.5,
        validation_alias=AliasChoices("data_freshness", "freshness_score", "data_latency"),
    )
    data_latency_seconds: float | None = Field(default=None, ge=0.0)

    @field_validator(
        "lead_time_reliability",
        "inventory_visibility",
        "disruption_risk",
        "sensor_quality",
        "data_freshness",
        mode="before",
    )
    @classmethod
    def _clip_unit_fields(cls, value: Any) -> float:
        return _clip_unit(value, default=0.5)

    def model_post_init(self, __context: Any) -> None:
        """Convert data_latency_seconds to data_freshness if provided."""
        if self.data_latency_seconds is not None:
            ceiling = 3600.0
            converted = max(0.0, min(1.0, 1.0 - (self.data_latency_seconds / ceiling)))
            object.__setattr__(self, "data_freshness", converted)


class GeopoliticalTrustPayload(BaseModel):
    """Normalized payload for geopolitical trust analysis."""

    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
        str_strip_whitespace=True,
    )

    region_id: str = Field(
        min_length=1,
        validation_alias=AliasChoices("region_id", "country_code", "geo_zone"),
    )
    political_stability: float = Field(default=0.5)
    sanctions_exposure: float = Field(default=0.5)
    conflict_proximity: float = Field(default=0.5)
    regulatory_alignment: float = Field(default=0.5)
    data_freshness: float = Field(
        default=0.5,
        validation_alias=AliasChoices("data_freshness", "freshness_score"),
    )

    @field_validator(
        "political_stability",
        "sanctions_exposure",
        "conflict_proximity",
        "regulatory_alignment",
        "data_freshness",
        mode="before",
    )
    @classmethod
    def _clip_unit_fields(cls, value: Any) -> float:
        return _clip_unit(value, default=0.5)


class RegulatoryTrustPayload(BaseModel):
    """Normalized payload for regulatory trust analysis."""

    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
        str_strip_whitespace=True,
    )

    jurisdiction_id: str = Field(
        min_length=1,
        validation_alias=AliasChoices("jurisdiction_id", "regulation_id", "compliance_scope"),
    )
    compliance_score: float = Field(default=0.5)
    enforcement_risk: float = Field(default=0.5)
    reporting_quality: float = Field(default=0.5)
    audit_recency: float = Field(default=0.5)
    data_freshness: float = Field(
        default=0.5,
        validation_alias=AliasChoices("data_freshness", "freshness_score"),
    )

    @field_validator(
        "compliance_score",
        "enforcement_risk",
        "reporting_quality",
        "audit_recency",
        "data_freshness",
        mode="before",
    )
    @classmethod
    def _clip_unit_fields(cls, value: Any) -> float:
        return _clip_unit(value, default=0.5)


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


def coerce_financial_payload(
    value: FinancialTrustPayload | dict[str, Any],
) -> FinancialTrustPayload:
    """Validate and normalize a financial trust payload."""
    if isinstance(value, FinancialTrustPayload):
        return value
    return FinancialTrustPayload.model_validate(value)


def coerce_news_payload(
    value: NewsTrustPayload | dict[str, Any],
) -> NewsTrustPayload:
    """Validate and normalize a news trust payload."""
    if isinstance(value, NewsTrustPayload):
        return value
    return NewsTrustPayload.model_validate(value)


def coerce_supply_chain_payload(
    value: SupplyChainTrustPayload | dict[str, Any],
) -> SupplyChainTrustPayload:
    """Validate and normalize a supply chain trust payload."""
    if isinstance(value, SupplyChainTrustPayload):
        return value
    return SupplyChainTrustPayload.model_validate(value)


def coerce_geopolitical_payload(
    value: GeopoliticalTrustPayload | dict[str, Any],
) -> GeopoliticalTrustPayload:
    """Validate and normalize a geopolitical trust payload."""
    if isinstance(value, GeopoliticalTrustPayload):
        return value
    return GeopoliticalTrustPayload.model_validate(value)


def coerce_regulatory_payload(
    value: RegulatoryTrustPayload | dict[str, Any],
) -> RegulatoryTrustPayload:
    """Validate and normalize a regulatory trust payload."""
    if isinstance(value, RegulatoryTrustPayload):
        return value
    return RegulatoryTrustPayload.model_validate(value)


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


def _clip_unit(value: Any, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if number <= 0.0:
        return 0.0
    if number >= 1.0:
        return 1.0
    return number


__all__ = [
    "FinancialTrustPayload",
    "GeopoliticalTrustPayload",
    "NewsTrustPayload",
    "NormalizedTrustResult",
    "RegulatoryTrustPayload",
    "SupplyChainTrustPayload",
    "TrustAgent",
    "TrustAudit",
    "TrustComponent",
    "TrustEvidence",
    "TrustViolation",
    "coerce_financial_payload",
    "coerce_geopolitical_payload",
    "coerce_news_payload",
    "coerce_normalized_trust_result",
    "coerce_regulatory_payload",
    "coerce_supply_chain_payload",
    "normalized_result_to_breakdown",
    "normalized_result_to_legacy_metadata",
]
