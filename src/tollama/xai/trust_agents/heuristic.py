"""
tollama.xai.trust_agents.heuristic — Baseline heuristic trust agents.
"""

from __future__ import annotations

from typing import Any

from tollama.xai.trust_contract import (
    NormalizedTrustResult,
    TrustAudit,
    TrustComponent,
    TrustEvidence,
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


def _build_result(
    *,
    agent_name: str,
    domain: str,
    component_breakdown: dict[str, TrustComponent],
    why_trusted: str,
    source_type: str,
    source_id: str,
) -> NormalizedTrustResult:
    total_weight = sum(component.weight for component in component_breakdown.values()) or 1.0
    trust_score = sum(
        component.score * component.weight for component in component_breakdown.values()
    ) / total_weight
    return NormalizedTrustResult(
        agent_name=agent_name,
        domain=domain,
        trust_score=trust_score,
        risk_category=_risk_category(trust_score),
        calibration_status="baseline_heuristic",
        component_breakdown=component_breakdown,
        why_trusted=why_trusted,
        evidence=TrustEvidence(
            source_type=source_type,
            source_ids=[source_id],
            payload_schema=f"{agent_name}_heuristic",
        ),
        audit=TrustAudit(formula_version="baseline-v1", agent_version="0.1.0"),
    )


class FinancialMarketTrustAgent:
    """Baseline trust agent for general financial markets."""

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
        components = {
            "liquidity_depth": TrustComponent(
                score=_clip_unit(payload.get("liquidity_depth", payload.get("liquidity"))),
                weight=0.25,
                value=payload.get("liquidity_depth", payload.get("liquidity")),
                assessment="higher_is_better",
            ),
            "spread_slippage": TrustComponent(
                score=_invert_ratio(
                    payload.get("spread_bps", payload.get("bid_ask_spread_bps", 50.0)),
                    100.0,
                ),
                weight=0.25,
                value=payload.get("spread_bps", payload.get("bid_ask_spread_bps")),
                assessment="lower_is_better",
            ),
            "regime_stability": TrustComponent(
                score=_invert_ratio(
                    payload.get("volatility_regime", payload.get("realized_volatility", 0.3)),
                    1.0,
                ),
                weight=0.25,
                value=payload.get("volatility_regime", payload.get("realized_volatility")),
                assessment="lower_is_better",
            ),
            "execution_risk": TrustComponent(
                score=_invert_ratio(
                    payload.get("execution_risk", payload.get("slippage_risk", 0.5)),
                    1.0,
                ),
                weight=0.15,
                value=payload.get("execution_risk", payload.get("slippage_risk")),
                assessment="lower_is_better",
            ),
            "data_freshness": TrustComponent(
                score=_clip_unit(payload.get("data_freshness", payload.get("freshness_score", 0.5))),
                weight=0.10,
                value=payload.get("data_freshness", payload.get("freshness_score")),
                assessment="higher_is_better",
            ),
        }
        return _build_result(
            agent_name=self.agent_name,
            domain=self.domain,
            component_breakdown=components,
            why_trusted=(
                "Financial market trust is derived from liquidity, spread/slippage, "
                "volatility regime, execution risk, and data freshness."
            ),
            source_type="financial_market",
            source_id=str(payload.get("instrument_id", payload.get("symbol", "financial_market"))),
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
    """Baseline trust agent for news and intelligence streams."""

    agent_name = "news"
    domain = "news"
    priority = 40

    def supports(self, context: dict[str, Any]) -> bool:
        domain = str(context.get("domain", ""))
        source_type = str(context.get("source_type", ""))
        return domain == self.domain or source_type in {"news", "intel", "media"}

    def analyze(self, payload: dict[str, Any]) -> NormalizedTrustResult:
        components = {
            "source_credibility": TrustComponent(
                score=_clip_unit(payload.get("source_credibility", 0.5)),
                weight=0.30,
                value=payload.get("source_credibility"),
                assessment="higher_is_better",
            ),
            "corroboration": TrustComponent(
                score=_clip_unit(payload.get("corroboration", payload.get("source_agreement", 0.5))),
                weight=0.25,
                value=payload.get("corroboration", payload.get("source_agreement")),
                assessment="higher_is_better",
            ),
            "novelty": TrustComponent(
                score=_clip_unit(payload.get("novelty", 0.5)),
                weight=0.10,
                value=payload.get("novelty"),
                assessment="contextual",
            ),
            "contradiction_penalty": TrustComponent(
                score=_invert_ratio(payload.get("contradiction_score", 0.5), 1.0),
                weight=0.20,
                value=payload.get("contradiction_score"),
                assessment="lower_is_better",
            ),
            "propagation_delay": TrustComponent(
                score=_invert_ratio(payload.get("propagation_delay_seconds", 300.0), 3600.0),
                weight=0.15,
                value=payload.get("propagation_delay_seconds"),
                assessment="lower_is_better",
            ),
        }
        return _build_result(
            agent_name=self.agent_name,
            domain=self.domain,
            component_breakdown=components,
            why_trusted=(
                "News trust is derived from source credibility, corroboration, novelty, "
                "contradiction penalty, and propagation delay."
            ),
            source_type="news",
            source_id=str(payload.get("story_id", payload.get("headline", "news"))),
        )
