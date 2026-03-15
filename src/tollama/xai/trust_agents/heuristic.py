"""
tollama.xai.trust_agents.heuristic — Schema-aware deterministic trust agents.
"""

from __future__ import annotations

from typing import Any

from tollama.xai.trust_contract import (
    FinancialTrustPayload,
    NewsTrustPayload,
    NormalizedTrustResult,
    TrustAudit,
    TrustComponent,
    TrustEvidence,
    TrustViolation,
    coerce_financial_payload,
    coerce_news_payload,
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
) -> NormalizedTrustResult:
    total_weight = sum(component.weight for component in component_breakdown.values()) or 1.0
    trust_score = sum(
        component.score * component.weight for component in component_breakdown.values()
    ) / total_weight
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
        f"volatility={payload.realized_volatility:.2f}, execution_risk={payload.execution_risk:.2f}, "
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
        f"contradiction={payload.contradiction_score:.2f}, propagation_delay={payload.propagation_delay_seconds:.0f}s, "
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
        return _build_result(
            agent_name=self.agent_name,
            domain=self.domain,
            component_breakdown=components,
            violations=violations,
            why_trusted=_financial_why_trusted(normalized, violations),
            source_type="financial_market",
            source_id=normalized.instrument_id,
            evidence_attributes={
                "news_signal_ref": normalized.news_signal_ref,
            },
        )


class SupplyChainTrustAgent:
    """Baseline trust agent for supply chain and logistics domains."""

    agent_name = "supply_chain"
    domain = "supply_chain"
    priority = 30

    def supports(self, context: dict[str, Any]) -> bool:
        domain = str(context.get("domain", ""))
        source_type = str(context.get("source_type", ""))
        return domain == self.domain or source_type in {"supply_chain", "logistics"}

    def analyze(self, payload: dict[str, Any]) -> NormalizedTrustResult:
        components = {
            "lead_time_reliability": TrustComponent(
                score=_clip_unit(payload.get("lead_time_reliability", 0.5)),
                weight=0.30,
                value=payload.get("lead_time_reliability"),
                assessment="higher_is_better",
            ),
            "inventory_visibility": TrustComponent(
                score=_clip_unit(payload.get("inventory_visibility", 0.5)),
                weight=0.25,
                value=payload.get("inventory_visibility"),
                assessment="higher_is_better",
            ),
            "disruption_risk": TrustComponent(
                score=_invert_ratio(payload.get("disruption_risk", 0.5), 1.0),
                weight=0.20,
                value=payload.get("disruption_risk"),
                assessment="lower_is_better",
            ),
            "sensor_quality": TrustComponent(
                score=_clip_unit(payload.get("sensor_quality", 0.5)),
                weight=0.15,
                value=payload.get("sensor_quality"),
                assessment="higher_is_better",
            ),
            "data_freshness": TrustComponent(
                score=_invert_ratio(payload.get("data_latency_seconds", 300.0), 3600.0),
                weight=0.10,
                value=payload.get("data_latency_seconds"),
                assessment="lower_is_better",
            ),
        }
        return _build_result(
            agent_name=self.agent_name,
            domain=self.domain,
            component_breakdown=components,
            why_trusted=(
                "Supply-chain trust is derived from lead-time reliability, inventory "
                "visibility, disruption risk, sensor quality, and data freshness."
            ),
            source_type="supply_chain",
            source_id=str(payload.get("network_id", payload.get("shipment_id", "supply_chain"))),
        )


class NewsTrustAgent:
    """Schema-aware trust agent for news and intelligence streams."""

    agent_name = "news"
    domain = "news"
    priority = 40

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
        )
