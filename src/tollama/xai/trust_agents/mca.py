"""
tollama.xai.trust_agents.mca — MCA adapter for normalized trust output.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from tollama.xai.trust_breakdown import TrustBreakdown
from tollama.xai.trust_contract import (
    NormalizedTrustResult,
    TrustAudit,
    TrustComponent,
    TrustEvidence,
    TrustViolation,
)


class MarketCalibrationTrustAgent:
    """Adapter for Market Calibration Agent calibration outputs."""

    agent_name = "market_calibration"
    domain = "prediction_market"
    priority = 10

    def __init__(self):
        self._trust_breakdown = TrustBreakdown()

    def supports(self, context: dict[str, Any]) -> bool:
        domain = str(context.get("domain", ""))
        source_type = str(context.get("source_type", ""))
        return domain == "prediction_market" or source_type == "prediction_market"

    def analyze(self, payload: dict[str, Any]) -> NormalizedTrustResult:
        breakdown = self._trust_breakdown.explain(payload)
        source = payload.get("source")
        if not source:
            signals = payload.get("signals", [])
            source = signals[0].get("name", "prediction_market") if signals else "prediction_market"
        score = float(payload.get("trust_score", breakdown["trust_scores"].get(source, 0.0)))
        components = breakdown.get("breakdowns", {}).get(source, {})
        result_components = {
            name: TrustComponent(
                score=float(info.get("normalized_score", 0.0)),
                weight=float(info.get("weight", 0.0)),
                value=info.get("value"),
                assessment=info.get("assessment"),
                rationale=info.get("description"),
            )
            for name, info in components.items()
        }
        violations: list[TrustViolation] = []
        if score < 0.5:
            violations.append(
                TrustViolation(
                    name="low_market_trust",
                    severity="warning",
                    type="calibration",
                    detail="Prediction market trust score is below the recommended threshold.",
                )
            )
        return NormalizedTrustResult(
            agent_name=self.agent_name,
            domain=self.domain,
            trust_score=score,
            risk_category=_risk_category(score),
            calibration_status=_calibration_status(score),
            component_breakdown=result_components,
            violations=violations,
            why_trusted=breakdown.get("why_trusted", {}).get(
                source, f"Trust Score {score:.2f} derived from market calibration."
            ),
            evidence=TrustEvidence(
                source_type="prediction_market",
                source_ids=[str(payload.get("market_id", source))],
                payload_schema="market_calibration",
                attributes={
                    "source": source,
                    "metrics": payload.get("metrics", {}),
                },
            ),
            audit=TrustAudit(
                formula_version=str(payload.get("formula_version", "mca-v1")),
                generated_at=payload.get(
                    "generated_at", datetime.now(UTC).isoformat()
                ),
                agent_version="0.1.0",
            ),
        )


def _risk_category(score: float) -> str:
    if score >= 0.75:
        return "GREEN"
    if score >= 0.50:
        return "YELLOW"
    return "RED"


def _calibration_status(score: float) -> str:
    if score >= 0.75:
        return "well_calibrated"
    if score >= 0.50:
        return "moderately_calibrated"
    return "poorly_calibrated"
