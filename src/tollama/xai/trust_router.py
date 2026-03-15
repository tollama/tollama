"""
tollama.xai.trust_router — Registry and router for domain trust agents.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from tollama.xai.trust_contract import (
    NormalizedTrustResult,
    TrustAgent,
    TrustAudit,
    TrustComponent,
    TrustEvidence,
    TrustViolation,
    coerce_normalized_trust_result,
)

_log = logging.getLogger(__name__)


class TrustAgentRegistry:
    """In-memory registry for trust agents."""

    def __init__(self):
        self._agents: list[TrustAgent] = []

    def register(self, agent: TrustAgent) -> None:
        if not hasattr(agent, "agent_name") or not hasattr(agent, "domain"):
            raise ValueError("Trust agent must define agent_name and domain")
        if not callable(getattr(agent, "supports", None)):
            raise ValueError("Trust agent must implement supports(context)")
        if not callable(getattr(agent, "analyze", None)):
            raise ValueError("Trust agent must implement analyze(payload)")
        self._agents.append(agent)

    def resolve(self, context: dict[str, Any]) -> list[TrustAgent]:
        return [agent for agent in self._agents if agent.supports(context)]

    @property
    def agents(self) -> list[TrustAgent]:
        return list(self._agents)


class TrustRouter:
    """Select and execute the primary trust agent for a request."""

    DEFAULT_PRIMARY_ORDER = {
        "prediction_market": ["market_calibration"],
        "financial_market": ["financial_market"],
        "supply_chain": ["supply_chain"],
        "news": ["news"],
        "geopolitical": ["geopolitical"],
        "regulatory": ["regulatory"],
    }

    def __init__(
        self,
        registry: TrustAgentRegistry,
        primary_order: dict[str, list[str]] | None = None,
        calibration_tracker: Any | None = None,
        calibration_path: Path | None = None,
        auto_persist_every: int = 10,
    ):
        self.registry = registry
        self.primary_order = primary_order or self.DEFAULT_PRIMARY_ORDER
        self.calibration_tracker = calibration_tracker
        self._calibration_path = calibration_path
        self._auto_persist_every = auto_persist_every
        self._analyze_count = 0

    def select_agent(self, context: dict[str, Any]) -> TrustAgent | None:
        matches = self.registry.resolve(context)
        if not matches:
            return None

        domain = context.get("domain")
        preferred = self.primary_order.get(str(domain), [])
        for agent_name in preferred:
            for match in matches:
                if match.agent_name == agent_name:
                    return match

        return sorted(matches, key=lambda agent: getattr(agent, "priority", 100))[0]

    def analyze(
        self,
        *,
        context: dict[str, Any],
        payload: dict[str, Any],
    ) -> NormalizedTrustResult | None:
        agent = self.select_agent(context)
        if agent is None:
            return None
        result = coerce_normalized_trust_result(agent.analyze(payload))
        self._maybe_auto_persist()
        return result

    def analyze_multi(
        self,
        *,
        context: dict[str, Any],
        payload: dict[str, Any],
    ) -> NormalizedTrustResult | None:
        """Run all matching agents and aggregate results."""
        matches = self.registry.resolve(context)
        if not matches:
            return None
        agents = sorted(matches, key=lambda a: getattr(a, "priority", 100))
        results = [coerce_normalized_trust_result(a.analyze(payload)) for a in agents]
        self._maybe_auto_persist()
        if len(results) == 1:
            return results[0]
        priorities = [getattr(a, "priority", 100) for a in agents]
        return _aggregate_trust_results(results, priorities)

    def record_outcome(
        self,
        agent_name: str,
        domain: str,
        predicted_score: float,
        actual_outcome: float,
        component_scores: dict[str, float] | None = None,
    ) -> None:
        """Record a prediction-outcome pair for calibration learning.

        Auto-persists to disk after every ``auto_persist_every`` calls.
        """
        if self.calibration_tracker is None:
            return
        self.calibration_tracker.record(
            agent_name=agent_name,
            domain=domain,
            predicted_score=predicted_score,
            actual_outcome=actual_outcome,
            component_scores=component_scores or {},
        )
        self._maybe_auto_persist()

    def persist_calibration(self) -> None:
        """Manually persist calibration data to disk."""
        if self.calibration_tracker is None or self._calibration_path is None:
            return
        try:
            self.calibration_tracker.save(self._calibration_path)
        except OSError:
            _log.warning("Failed to persist calibration data", exc_info=True)

    def _maybe_auto_persist(self) -> None:
        """Persist calibration data every N analyze calls."""
        if (
            self.calibration_tracker is None
            or self._calibration_path is None
            or self._auto_persist_every <= 0
        ):
            return
        self._analyze_count += 1
        if self._analyze_count % self._auto_persist_every == 0:
            self.persist_calibration()


_RISK_ORDER = {"GREEN": 0, "YELLOW": 1, "RED": 2}
_CALIBRATION_ORDER = {"high_trust": 0, "moderate_trust": 1, "low_trust": 2, "blocked": 3}


def _aggregate_trust_results(
    results: list[NormalizedTrustResult],
    priorities: list[int],
) -> NormalizedTrustResult:
    """Aggregate multiple trust results following conservative policy."""
    # risk_category: most conservative wins
    risk_categories = [r.risk_category for r in results]
    agg_risk = max(risk_categories, key=lambda rc: _RISK_ORDER.get(rc, 1))

    # trust_score: priority-weighted average (lower priority number = higher weight)
    inv_priorities = [1.0 / max(p, 1) for p in priorities]
    total_inv = sum(inv_priorities) or 1.0
    agg_score = sum(
        r.trust_score * w for r, w in zip(results, inv_priorities)
    ) / total_inv

    # violations: merge, deduplicate by (name, severity)
    seen: set[tuple[str, str]] = set()
    agg_violations: list[TrustViolation] = []
    for r in results:
        for v in r.violations:
            key = (v.name, v.severity)
            if key not in seen:
                seen.add(key)
                agg_violations.append(v)

    # component_breakdown: prefix with agent_name to avoid collisions
    agg_components: dict[str, TrustComponent] = {}
    for r in results:
        for name, comp in r.component_breakdown.items():
            agg_components[f"{r.agent_name}/{name}"] = comp

    # calibration_status: worst wins
    cal_statuses = [r.calibration_status or "moderate_trust" for r in results]
    agg_calibration = max(cal_statuses, key=lambda cs: _CALIBRATION_ORDER.get(cs, 1))

    # why_trusted: join
    agg_why = " | ".join(f"[{r.agent_name}] {r.why_trusted}" for r in results)

    # evidence: merge source_ids, combine attributes
    all_source_ids: list[str] = []
    agg_attrs: dict[str, Any] = {}
    for r in results:
        all_source_ids.extend(r.evidence.source_ids)
        agg_attrs[r.agent_name] = r.evidence.attributes

    # human_review if agents disagree on risk_category
    unique_risks = set(risk_categories)
    if len(unique_risks) > 1:
        agg_attrs["human_review_reason"] = "agents_disagree_on_risk"

    return NormalizedTrustResult(
        agent_name="multi_agent_aggregate",
        domain=results[0].domain if len({r.domain for r in results}) == 1 else "multi",
        trust_score=agg_score,
        risk_category=agg_risk,
        calibration_status=agg_calibration,
        component_breakdown=agg_components,
        violations=agg_violations,
        why_trusted=agg_why,
        evidence=TrustEvidence(
            source_type="multi_agent",
            source_ids=all_source_ids,
            payload_schema="multi_agent_aggregate_v1",
            attributes=agg_attrs,
        ),
        audit=TrustAudit(formula_version="aggregate-v1", agent_version="0.1.0"),
    )


def build_default_trust_router(*, enable_calibration: bool = True) -> TrustRouter:
    """Build the default in-repo trust agent router.

    When *enable_calibration* is True (default), loads persisted calibration
    data from disk and auto-persists every 10 analyze calls.
    """
    from tollama.xai.trust_agents import (
        CalibrationTracker,
        FinancialMarketTrustAgent,
        GeopoliticalTrustAgent,
        MarketCalibrationTrustAgent,
        NewsTrustAgent,
        RegulatoryTrustAgent,
        SupplyChainTrustAgent,
    )
    from tollama.xai.trust_agents.calibration import default_calibration_path

    calibration_tracker = None
    calibration_path = None
    if enable_calibration:
        calibration_path = default_calibration_path()
        try:
            calibration_tracker = CalibrationTracker.load(calibration_path)
        except Exception:  # noqa: BLE001
            _log.warning("Failed to load calibration data, starting fresh", exc_info=True)
            calibration_tracker = CalibrationTracker()

    registry = TrustAgentRegistry()
    registry.register(MarketCalibrationTrustAgent())
    financial_agent = FinancialMarketTrustAgent()
    registry.register(financial_agent)
    registry.register(SupplyChainTrustAgent())
    registry.register(NewsTrustAgent())
    registry.register(GeopoliticalTrustAgent())
    registry.register(RegulatoryTrustAgent())
    router = TrustRouter(
        registry,
        calibration_tracker=calibration_tracker,
        calibration_path=calibration_path,
    )
    financial_agent._trust_router = router
    return router


__all__ = [
    "TrustAgentRegistry",
    "TrustRouter",
    "build_default_trust_router",
]
