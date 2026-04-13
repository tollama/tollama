"""
tollama.xai.trust_agents.heuristic — Schema-aware deterministic trust agents.
"""

from __future__ import annotations

from typing import Any

from tollama.xai.trust_contract import (
    FinancialTrustPayload,
    GeopoliticalTrustPayload,
    NewsTrustPayload,
    NormalizedTrustResult,
    RegulatoryTrustPayload,
    SupplyChainTrustPayload,
    TrustAudit,
    TrustComponent,
    TrustEvidence,
    TrustViolation,
    coerce_financial_payload,
    coerce_geopolitical_payload,
    coerce_news_payload,
    coerce_regulatory_payload,
    coerce_supply_chain_payload,
)


def _clip_unit(value: Any, default: float = 0.5) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if number <= 0.0:
        return 0.0
    if number >= 1.0:
        return 1.0
    return number


def _invert_ratio(value: Any, ceiling: float, default: float = 0.5) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if ceiling <= 0:
        return default
    return max(0.0, min(1.0, 1.0 - (number / ceiling)))


def _risk_category(score: float) -> str:
    if score >= 0.75:
        return "GREEN"
    if score >= 0.50:
        return "YELLOW"
    return "RED"


def _derive_risk_category(
    trust_score: float,
    violations: list[TrustViolation],
) -> str:
    if any(violation.severity == "critical" for violation in violations):
        return "RED"
    if trust_score < 0.50:
        return "RED"
    if violations:
        return "YELLOW"
    return _risk_category(trust_score)


def _derive_calibration_status(
    trust_score: float,
    violations: list[TrustViolation],
) -> str:
    if any(violation.severity == "critical" for violation in violations):
        return "blocked"
    if trust_score >= 0.75 and not violations:
        return "high_trust"
    if trust_score >= 0.50:
        return "moderate_trust"
    return "low_trust"


def _build_result(
    *,
    agent_name: str,
    domain: str,
    component_breakdown: dict[str, TrustComponent],
    violations: list[TrustViolation],
    why_trusted: str,
    source_type: str,
    source_id: str,
    evidence_attributes: dict[str, Any] | None = None,
    calibration_tracker: Any | None = None,
) -> NormalizedTrustResult:
    if calibration_tracker is not None:
        adjustments = calibration_tracker.get_weight_adjustments(agent_name)
        if adjustments:
            component_breakdown = {
                name: TrustComponent(
                    score=comp.score,
                    weight=comp.weight * adjustments.get(name, 1.0),
                    value=comp.value,
                    assessment=comp.assessment,
                    rationale=comp.rationale,
                )
                for name, comp in component_breakdown.items()
            }
    total_weight = sum(component.weight for component in component_breakdown.values()) or 1.0
    trust_score = (
        sum(component.score * component.weight for component in component_breakdown.values())
        / total_weight
    )
    return NormalizedTrustResult(
        agent_name=agent_name,
        domain=domain,
        trust_score=trust_score,
        risk_category=_derive_risk_category(trust_score, violations),
        calibration_status=_derive_calibration_status(trust_score, violations),
        component_breakdown=component_breakdown,
        violations=violations,
        why_trusted=why_trusted,
        evidence=TrustEvidence(
            source_type=source_type,
            source_ids=[source_id],
            payload_schema=f"{agent_name}_trust_payload_v1",
            attributes=evidence_attributes or {},
        ),
        audit=TrustAudit(formula_version="baseline-v1", agent_version="0.1.0"),
    )


def _financial_component_breakdown(payload: FinancialTrustPayload) -> dict[str, TrustComponent]:
    return {
        "liquidity_depth": TrustComponent(
            score=payload.liquidity_depth,
            weight=0.25,
            value=payload.liquidity_depth,
            assessment="higher_is_better",
            rationale="Direct liquidity depth score after normalization.",
        ),
        "spread_slippage": TrustComponent(
            score=_invert_ratio(payload.bid_ask_spread_bps, 100.0),
            weight=0.25,
            value=payload.bid_ask_spread_bps,
            assessment="lower_is_better",
            rationale="Tighter spreads indicate lower execution friction.",
        ),
        "regime_stability": TrustComponent(
            score=_invert_ratio(payload.realized_volatility, 1.0),
            weight=0.25,
            value=payload.realized_volatility,
            assessment="lower_is_better",
            rationale="Lower realized volatility indicates a more stable market regime.",
        ),
        "execution_risk": TrustComponent(
            score=1.0 - payload.execution_risk,
            weight=0.15,
            value=payload.execution_risk,
            assessment="lower_is_better",
            rationale="Lower execution risk improves trust in actionable signals.",
        ),
        "data_freshness": TrustComponent(
            score=payload.data_freshness,
            weight=0.10,
            value=payload.data_freshness,
            assessment="higher_is_better",
            rationale="Fresh market data reduces stale-signal risk.",
        ),
    }


def _financial_violations(payload: FinancialTrustPayload) -> list[TrustViolation]:
    violations: list[TrustViolation] = []
    if payload.bid_ask_spread_bps >= 100.0:
        violations.append(
            TrustViolation(
                name="spread_slippage_extreme",
                severity="critical",
                type="liquidity",
                detail="Bid/ask spread is too wide for high-confidence execution.",
            )
        )
    elif payload.bid_ask_spread_bps >= 25.0:
        violations.append(
            TrustViolation(
                name="spread_slippage_elevated",
                severity="warning",
                type="liquidity",
                detail="Bid/ask spread is elevated and may degrade execution quality.",
            )
        )

    if payload.execution_risk >= 0.8:
        violations.append(
            TrustViolation(
                name="execution_risk_high",
                severity="critical",
                type="execution",
                detail="Execution risk is too high for automated decisioning.",
            )
        )
    elif payload.execution_risk >= 0.6:
        violations.append(
            TrustViolation(
                name="execution_risk_elevated",
                severity="warning",
                type="execution",
                detail="Execution risk is elevated and should be reviewed.",
            )
        )

    if payload.data_freshness <= 0.10:
        violations.append(
            TrustViolation(
                name="market_data_stale",
                severity="critical",
                type="freshness",
                detail="Market data is stale for high-confidence trust scoring.",
            )
        )
    elif payload.data_freshness <= 0.25:
        violations.append(
            TrustViolation(
                name="market_data_aging",
                severity="warning",
                type="freshness",
                detail="Market data freshness is deteriorating.",
            )
        )
    return violations


def _financial_why_trusted(
    payload: FinancialTrustPayload,
    violations: list[TrustViolation],
) -> str:
    base = (
        f"Liquidity={payload.liquidity_depth:.2f}, spread={payload.bid_ask_spread_bps:.1f}bps, "
        f"volatility={payload.realized_volatility:.2f}, "
        f"execution_risk={payload.execution_risk:.2f}, "
        f"freshness={payload.data_freshness:.2f}."
    )
    if not violations:
        return f"Financial market signal is decision-ready. {base}"
    return f"Financial market signal requires caution. {base}"


def _news_component_breakdown(payload: NewsTrustPayload) -> dict[str, TrustComponent]:
    return {
        "source_credibility": TrustComponent(
            score=payload.source_credibility,
            weight=0.35,
            value=payload.source_credibility,
            assessment="higher_is_better",
            rationale="Higher source credibility improves trust in the story.",
        ),
        "corroboration": TrustComponent(
            score=payload.corroboration,
            weight=0.25,
            value=payload.corroboration,
            assessment="higher_is_better",
            rationale="Independent corroboration increases confidence.",
        ),
        "contradiction_penalty": TrustComponent(
            score=1.0 - payload.contradiction_score,
            weight=0.20,
            value=payload.contradiction_score,
            assessment="lower_is_better",
            rationale="Higher contradiction reduces trust in the narrative.",
        ),
        "propagation_delay": TrustComponent(
            score=_invert_ratio(payload.propagation_delay_seconds, 3600.0),
            weight=0.10,
            value=payload.propagation_delay_seconds,
            assessment="lower_is_better",
            rationale="Lower propagation delay keeps the signal timely.",
        ),
        "freshness": TrustComponent(
            score=payload.freshness_score,
            weight=0.10,
            value=payload.freshness_score,
            assessment="higher_is_better",
            rationale="Fresh news signals retain higher trust.",
        ),
    }


def _news_violations(payload: NewsTrustPayload) -> list[TrustViolation]:
    violations: list[TrustViolation] = []
    if payload.source_credibility < 0.2:
        violations.append(
            TrustViolation(
                name="source_credibility_low",
                severity="critical",
                type="credibility",
                detail="Source credibility is too low for trusted automation.",
            )
        )
    elif payload.source_credibility < 0.4:
        violations.append(
            TrustViolation(
                name="source_credibility_weak",
                severity="warning",
                type="credibility",
                detail="Source credibility is weaker than preferred.",
            )
        )

    if payload.contradiction_score > 0.8:
        violations.append(
            TrustViolation(
                name="contradiction_high",
                severity="critical",
                type="consistency",
                detail="Story contradiction is too high for trusted use.",
            )
        )
    elif payload.contradiction_score > 0.6:
        violations.append(
            TrustViolation(
                name="contradiction_elevated",
                severity="warning",
                type="consistency",
                detail="Story contradiction is elevated and should be reviewed.",
            )
        )

    if payload.propagation_delay_seconds > 86400:
        violations.append(
            TrustViolation(
                name="propagation_delay_excessive",
                severity="critical",
                type="timeliness",
                detail="Propagation delay is too high for timely news trust.",
            )
        )
    elif payload.propagation_delay_seconds > 21600:
        violations.append(
            TrustViolation(
                name="propagation_delay_elevated",
                severity="warning",
                type="timeliness",
                detail="Propagation delay is elevated.",
            )
        )
    return violations


def _news_why_trusted(
    payload: NewsTrustPayload,
    violations: list[TrustViolation],
) -> str:
    base = (
        f"Credibility={payload.source_credibility:.2f}, corroboration={payload.corroboration:.2f}, "
        f"contradiction={payload.contradiction_score:.2f}, "
        f"propagation_delay={payload.propagation_delay_seconds:.0f}s, "
        f"freshness={payload.freshness_score:.2f}."
    )
    if not violations:
        return f"News signal is well-supported and timely. {base}"
    return f"News signal has reliability concerns. {base}"


class FinancialMarketTrustAgent:
    """Schema-aware trust agent for general financial markets."""

    agent_name = "financial_market"
    domain = "financial_market"
    priority = 20

    def __init__(
        self,
        trust_router: Any | None = None,
        calibration_tracker: Any | None = None,
    ):
        self._trust_router = trust_router
        self._calibration_tracker = calibration_tracker

    def supports(self, context: dict[str, Any]) -> bool:
        domain = str(context.get("domain", ""))
        source_type = str(context.get("source_type", ""))
        return domain == self.domain or source_type in {
            "financial_market",
            "equity_market",
            "fx_market",
            "rates_market",
            "commodity_market",
        }

    def analyze(self, payload: dict[str, Any]) -> NormalizedTrustResult:
        normalized = coerce_financial_payload(payload)
        components = _financial_component_breakdown(normalized)
        violations = _financial_violations(normalized)
        evidence_attrs: dict[str, Any] = {
            "news_signal_ref": normalized.news_signal_ref,
        }

        if normalized.news_signal_ref and self._trust_router is not None:
            news_result = self._resolve_news_signal(normalized.news_signal_ref)
            if news_result is not None:
                components["news_signal"] = TrustComponent(
                    score=news_result.trust_score,
                    weight=0.10,
                    value=news_result.trust_score,
                    assessment="higher_is_better",
                    rationale=f"News signal trust from '{normalized.news_signal_ref}'.",
                )
                for v in news_result.violations:
                    if v.severity == "critical":
                        violations.append(
                            TrustViolation(
                                name=f"news_{v.name}",
                                severity="critical",
                                type=f"news_{v.type or 'unknown'}",
                                detail=f"From news signal: {v.detail}",
                            )
                        )
                evidence_attrs["news_trust_score"] = news_result.trust_score
                evidence_attrs["news_risk_category"] = news_result.risk_category
                evidence_attrs["news_violations_count"] = len(news_result.violations)

        return _build_result(
            agent_name=self.agent_name,
            domain=self.domain,
            component_breakdown=components,
            violations=violations,
            why_trusted=_financial_why_trusted(normalized, violations),
            source_type="financial_market",
            source_id=normalized.instrument_id,
            evidence_attributes=evidence_attrs,
            calibration_tracker=self._calibration_tracker,
        )

    def _resolve_news_signal(
        self,
        signal_ref: str,
    ) -> NormalizedTrustResult | None:
        """Run the news agent with the signal reference as story_id."""
        try:
            return self._trust_router.analyze(
                context={"domain": "news", "source_type": "news"},
                payload={"story_id": signal_ref},
            )
        except Exception:
            return None


def _supply_chain_component_breakdown(
    payload: SupplyChainTrustPayload,
) -> dict[str, TrustComponent]:
    return {
        "lead_time_reliability": TrustComponent(
            score=payload.lead_time_reliability,
            weight=0.30,
            value=payload.lead_time_reliability,
            assessment="higher_is_better",
            rationale="Higher lead-time reliability indicates a dependable supply chain.",
        ),
        "inventory_visibility": TrustComponent(
            score=payload.inventory_visibility,
            weight=0.25,
            value=payload.inventory_visibility,
            assessment="higher_is_better",
            rationale="Better inventory visibility reduces uncertainty in stock levels.",
        ),
        "disruption_risk": TrustComponent(
            score=1.0 - payload.disruption_risk,
            weight=0.20,
            value=payload.disruption_risk,
            assessment="lower_is_better",
            rationale="Lower disruption risk improves trust in supply continuity.",
        ),
        "sensor_quality": TrustComponent(
            score=payload.sensor_quality,
            weight=0.15,
            value=payload.sensor_quality,
            assessment="higher_is_better",
            rationale="Higher sensor quality improves data reliability.",
        ),
        "data_freshness": TrustComponent(
            score=payload.data_freshness,
            weight=0.10,
            value=payload.data_freshness,
            assessment="higher_is_better",
            rationale="Fresh supply-chain data reduces stale-signal risk.",
        ),
    }


def _supply_chain_violations(
    payload: SupplyChainTrustPayload,
) -> list[TrustViolation]:
    violations: list[TrustViolation] = []
    if payload.disruption_risk >= 0.8:
        violations.append(
            TrustViolation(
                name="disruption_risk_extreme",
                severity="critical",
                type="disruption",
                detail="Disruption risk is too high for automated decisioning.",
            )
        )
    elif payload.disruption_risk >= 0.6:
        violations.append(
            TrustViolation(
                name="disruption_risk_elevated",
                severity="warning",
                type="disruption",
                detail="Disruption risk is elevated and should be monitored.",
            )
        )

    if payload.lead_time_reliability <= 0.2:
        violations.append(
            TrustViolation(
                name="lead_time_unreliable",
                severity="critical",
                type="reliability",
                detail="Lead-time reliability is too low for trusted planning.",
            )
        )
    elif payload.lead_time_reliability <= 0.4:
        violations.append(
            TrustViolation(
                name="lead_time_degraded",
                severity="warning",
                type="reliability",
                detail="Lead-time reliability is degraded.",
            )
        )

    if payload.sensor_quality <= 0.15:
        violations.append(
            TrustViolation(
                name="sensor_quality_critical",
                severity="critical",
                type="data_quality",
                detail="Sensor quality is critically low.",
            )
        )
    elif payload.sensor_quality <= 0.3:
        violations.append(
            TrustViolation(
                name="sensor_quality_low",
                severity="warning",
                type="data_quality",
                detail="Sensor quality is below acceptable threshold.",
            )
        )

    if payload.data_freshness <= 0.10:
        violations.append(
            TrustViolation(
                name="supply_data_stale",
                severity="critical",
                type="freshness",
                detail="Supply-chain data is stale for trusted scoring.",
            )
        )
    elif payload.data_freshness <= 0.25:
        violations.append(
            TrustViolation(
                name="supply_data_aging",
                severity="warning",
                type="freshness",
                detail="Supply-chain data freshness is deteriorating.",
            )
        )
    return violations


def _supply_chain_why_trusted(
    payload: SupplyChainTrustPayload,
    violations: list[TrustViolation],
) -> str:
    base = (
        f"Lead-time={payload.lead_time_reliability:.2f}, "
        f"visibility={payload.inventory_visibility:.2f}, "
        f"disruption={payload.disruption_risk:.2f}, sensor={payload.sensor_quality:.2f}, "
        f"freshness={payload.data_freshness:.2f}."
    )
    if not violations:
        return f"Supply-chain signal is decision-ready. {base}"
    return f"Supply-chain signal requires caution. {base}"


class SupplyChainTrustAgent:
    """Schema-aware trust agent for supply chain and logistics domains."""

    agent_name = "supply_chain"
    domain = "supply_chain"
    priority = 30

    def __init__(self, calibration_tracker: Any | None = None):
        self._calibration_tracker = calibration_tracker

    def supports(self, context: dict[str, Any]) -> bool:
        domain = str(context.get("domain", ""))
        source_type = str(context.get("source_type", ""))
        return domain == self.domain or source_type in {"supply_chain", "logistics"}

    def analyze(self, payload: dict[str, Any]) -> NormalizedTrustResult:
        normalized = coerce_supply_chain_payload(payload)
        components = _supply_chain_component_breakdown(normalized)
        violations = _supply_chain_violations(normalized)
        return _build_result(
            agent_name=self.agent_name,
            domain=self.domain,
            component_breakdown=components,
            violations=violations,
            why_trusted=_supply_chain_why_trusted(normalized, violations),
            source_type="supply_chain",
            source_id=normalized.network_id,
            calibration_tracker=self._calibration_tracker,
        )


class NewsTrustAgent:
    """Schema-aware trust agent for news and intelligence streams."""

    agent_name = "news"
    domain = "news"
    priority = 40

    def __init__(self, calibration_tracker: Any | None = None):
        self._calibration_tracker = calibration_tracker

    def supports(self, context: dict[str, Any]) -> bool:
        domain = str(context.get("domain", ""))
        source_type = str(context.get("source_type", ""))
        return domain == self.domain or source_type in {"news", "intel", "media"}

    def analyze(self, payload: dict[str, Any]) -> NormalizedTrustResult:
        normalized = coerce_news_payload(payload)
        components = _news_component_breakdown(normalized)
        violations = _news_violations(normalized)
        return _build_result(
            agent_name=self.agent_name,
            domain=self.domain,
            component_breakdown=components,
            violations=violations,
            why_trusted=_news_why_trusted(normalized, violations),
            source_type="news",
            source_id=normalized.story_id,
            evidence_attributes={
                "novelty": normalized.novelty,
            },
            calibration_tracker=self._calibration_tracker,
        )


def _geopolitical_component_breakdown(
    payload: GeopoliticalTrustPayload,
) -> dict[str, TrustComponent]:
    return {
        "political_stability": TrustComponent(
            score=payload.political_stability,
            weight=0.30,
            value=payload.political_stability,
            assessment="higher_is_better",
            rationale="Higher political stability indicates a more predictable environment.",
        ),
        "sanctions_exposure": TrustComponent(
            score=1.0 - payload.sanctions_exposure,
            weight=0.25,
            value=payload.sanctions_exposure,
            assessment="lower_is_better",
            rationale="Lower sanctions exposure reduces compliance and operational risk.",
        ),
        "conflict_proximity": TrustComponent(
            score=1.0 - payload.conflict_proximity,
            weight=0.20,
            value=payload.conflict_proximity,
            assessment="lower_is_better",
            rationale="Distance from conflict zones improves operational trust.",
        ),
        "regulatory_alignment": TrustComponent(
            score=payload.regulatory_alignment,
            weight=0.15,
            value=payload.regulatory_alignment,
            assessment="higher_is_better",
            rationale="Higher regulatory alignment reduces cross-border risk.",
        ),
        "data_freshness": TrustComponent(
            score=payload.data_freshness,
            weight=0.10,
            value=payload.data_freshness,
            assessment="higher_is_better",
            rationale="Fresh geopolitical data reduces stale-signal risk.",
        ),
    }


def _geopolitical_violations(
    payload: GeopoliticalTrustPayload,
) -> list[TrustViolation]:
    violations: list[TrustViolation] = []
    if payload.sanctions_exposure >= 0.8:
        violations.append(
            TrustViolation(
                name="sanctions_exposure_extreme",
                severity="critical",
                type="sanctions",
                detail="Sanctions exposure is too high for automated decisioning.",
            )
        )
    elif payload.sanctions_exposure >= 0.6:
        violations.append(
            TrustViolation(
                name="sanctions_exposure_elevated",
                severity="warning",
                type="sanctions",
                detail="Sanctions exposure is elevated and should be reviewed.",
            )
        )

    if payload.conflict_proximity >= 0.8:
        violations.append(
            TrustViolation(
                name="conflict_proximity_extreme",
                severity="critical",
                type="conflict",
                detail="Conflict proximity is too high for trusted operations.",
            )
        )
    elif payload.conflict_proximity >= 0.6:
        violations.append(
            TrustViolation(
                name="conflict_proximity_elevated",
                severity="warning",
                type="conflict",
                detail="Conflict proximity is elevated.",
            )
        )

    if payload.political_stability <= 0.2:
        violations.append(
            TrustViolation(
                name="political_stability_critical",
                severity="critical",
                type="stability",
                detail="Political stability is critically low.",
            )
        )
    elif payload.political_stability <= 0.4:
        violations.append(
            TrustViolation(
                name="political_stability_low",
                severity="warning",
                type="stability",
                detail="Political stability is below acceptable threshold.",
            )
        )

    if payload.data_freshness <= 0.10:
        violations.append(
            TrustViolation(
                name="geo_data_stale",
                severity="critical",
                type="freshness",
                detail="Geopolitical data is stale for trusted scoring.",
            )
        )
    elif payload.data_freshness <= 0.25:
        violations.append(
            TrustViolation(
                name="geo_data_aging",
                severity="warning",
                type="freshness",
                detail="Geopolitical data freshness is deteriorating.",
            )
        )
    return violations


def _geopolitical_why_trusted(
    payload: GeopoliticalTrustPayload,
    violations: list[TrustViolation],
) -> str:
    base = (
        f"Stability={payload.political_stability:.2f}, sanctions={payload.sanctions_exposure:.2f}, "
        f"conflict={payload.conflict_proximity:.2f}, alignment={payload.regulatory_alignment:.2f}, "
        f"freshness={payload.data_freshness:.2f}."
    )
    if not violations:
        return f"Geopolitical signal is decision-ready. {base}"
    return f"Geopolitical signal requires caution. {base}"


class GeopoliticalTrustAgent:
    """Schema-aware trust agent for geopolitical risk domains."""

    agent_name = "geopolitical"
    domain = "geopolitical"
    priority = 50

    def __init__(self, calibration_tracker: Any | None = None):
        self._calibration_tracker = calibration_tracker

    def supports(self, context: dict[str, Any]) -> bool:
        domain = str(context.get("domain", ""))
        source_type = str(context.get("source_type", ""))
        return domain == self.domain or source_type in {
            "geopolitical",
            "country_risk",
            "sanctions",
        }

    def analyze(self, payload: dict[str, Any]) -> NormalizedTrustResult:
        normalized = coerce_geopolitical_payload(payload)
        components = _geopolitical_component_breakdown(normalized)
        violations = _geopolitical_violations(normalized)
        return _build_result(
            agent_name=self.agent_name,
            domain=self.domain,
            component_breakdown=components,
            violations=violations,
            why_trusted=_geopolitical_why_trusted(normalized, violations),
            source_type="geopolitical",
            source_id=normalized.region_id,
            calibration_tracker=self._calibration_tracker,
        )


def _regulatory_component_breakdown(
    payload: RegulatoryTrustPayload,
) -> dict[str, TrustComponent]:
    return {
        "compliance_score": TrustComponent(
            score=payload.compliance_score,
            weight=0.30,
            value=payload.compliance_score,
            assessment="higher_is_better",
            rationale="Higher compliance score indicates better regulatory adherence.",
        ),
        "enforcement_risk": TrustComponent(
            score=1.0 - payload.enforcement_risk,
            weight=0.25,
            value=payload.enforcement_risk,
            assessment="lower_is_better",
            rationale="Lower enforcement risk reduces regulatory exposure.",
        ),
        "reporting_quality": TrustComponent(
            score=payload.reporting_quality,
            weight=0.20,
            value=payload.reporting_quality,
            assessment="higher_is_better",
            rationale="Higher reporting quality improves transparency trust.",
        ),
        "audit_recency": TrustComponent(
            score=payload.audit_recency,
            weight=0.15,
            value=payload.audit_recency,
            assessment="higher_is_better",
            rationale="Recent audits provide more current assurance.",
        ),
        "data_freshness": TrustComponent(
            score=payload.data_freshness,
            weight=0.10,
            value=payload.data_freshness,
            assessment="higher_is_better",
            rationale="Fresh regulatory data reduces stale-signal risk.",
        ),
    }


def _regulatory_violations(
    payload: RegulatoryTrustPayload,
) -> list[TrustViolation]:
    violations: list[TrustViolation] = []
    if payload.compliance_score <= 0.2:
        violations.append(
            TrustViolation(
                name="compliance_critical",
                severity="critical",
                type="compliance",
                detail="Compliance score is critically low for trusted operations.",
            )
        )
    elif payload.compliance_score <= 0.4:
        violations.append(
            TrustViolation(
                name="compliance_low",
                severity="warning",
                type="compliance",
                detail="Compliance score is below acceptable threshold.",
            )
        )

    if payload.enforcement_risk >= 0.8:
        violations.append(
            TrustViolation(
                name="enforcement_risk_extreme",
                severity="critical",
                type="enforcement",
                detail="Enforcement risk is too high for automated decisioning.",
            )
        )
    elif payload.enforcement_risk >= 0.6:
        violations.append(
            TrustViolation(
                name="enforcement_risk_elevated",
                severity="warning",
                type="enforcement",
                detail="Enforcement risk is elevated and should be reviewed.",
            )
        )

    if payload.reporting_quality <= 0.2:
        violations.append(
            TrustViolation(
                name="reporting_quality_critical",
                severity="critical",
                type="reporting",
                detail="Reporting quality is critically low.",
            )
        )
    elif payload.reporting_quality <= 0.4:
        violations.append(
            TrustViolation(
                name="reporting_quality_low",
                severity="warning",
                type="reporting",
                detail="Reporting quality is below acceptable threshold.",
            )
        )

    if payload.audit_recency <= 0.1:
        violations.append(
            TrustViolation(
                name="audit_recency_stale",
                severity="critical",
                type="audit",
                detail="Audit recency is critically low.",
            )
        )
    elif payload.audit_recency <= 0.25:
        violations.append(
            TrustViolation(
                name="audit_recency_aging",
                severity="warning",
                type="audit",
                detail="Audit recency is deteriorating.",
            )
        )
    return violations


def _regulatory_why_trusted(
    payload: RegulatoryTrustPayload,
    violations: list[TrustViolation],
) -> str:
    base = (
        f"Compliance={payload.compliance_score:.2f}, enforcement={payload.enforcement_risk:.2f}, "
        f"reporting={payload.reporting_quality:.2f}, audit={payload.audit_recency:.2f}, "
        f"freshness={payload.data_freshness:.2f}."
    )
    if not violations:
        return f"Regulatory signal is decision-ready. {base}"
    return f"Regulatory signal requires caution. {base}"


class RegulatoryTrustAgent:
    """Schema-aware trust agent for regulatory and compliance domains."""

    agent_name = "regulatory"
    domain = "regulatory"
    priority = 60

    def __init__(self, calibration_tracker: Any | None = None):
        self._calibration_tracker = calibration_tracker

    def supports(self, context: dict[str, Any]) -> bool:
        domain = str(context.get("domain", ""))
        source_type = str(context.get("source_type", ""))
        return domain == self.domain or source_type in {
            "regulatory",
            "compliance",
            "legal",
        }

    def analyze(self, payload: dict[str, Any]) -> NormalizedTrustResult:
        normalized = coerce_regulatory_payload(payload)
        components = _regulatory_component_breakdown(normalized)
        violations = _regulatory_violations(normalized)
        return _build_result(
            agent_name=self.agent_name,
            domain=self.domain,
            component_breakdown=components,
            violations=violations,
            why_trusted=_regulatory_why_trusted(normalized, violations),
            source_type="regulatory",
            source_id=normalized.jurisdiction_id,
            calibration_tracker=self._calibration_tracker,
        )
