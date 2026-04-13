"""
tollama.xai.trust_router — Registry and router for domain trust agents.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any

from pydantic import ValidationError

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
        cache_ttl: float = 0.0,
        history_tracker: Any | None = None,
        history_path: Path | None = None,
        connector_registry: Any | None = None,
    ):
        self.registry = registry
        self.primary_order = primary_order or self.DEFAULT_PRIMARY_ORDER
        self.calibration_tracker = calibration_tracker
        self._calibration_path = calibration_path
        self._auto_persist_every = auto_persist_every
        self._analyze_count = 0
        self._cache_ttl = cache_ttl
        self._cache: dict[str, tuple[float, NormalizedTrustResult]] = {}
        self.history_tracker = history_tracker
        self._history_path = history_path
        self._cache_hits = 0
        self._cache_misses = 0
        self.connector_registry = connector_registry

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

    @staticmethod
    def _cache_key(context: dict[str, Any], payload: dict[str, Any]) -> str:
        """Deterministic hash for a context+payload pair."""
        raw = json.dumps({"c": context, "p": payload}, sort_keys=True)
        return hashlib.sha256(raw.encode()).hexdigest()

    def _get_cached(self, key: str) -> NormalizedTrustResult | None:
        """Return cached result if within TTL, else None."""
        if self._cache_ttl <= 0:
            return None
        entry = self._cache.get(key)
        if entry is None:
            return None
        ts, result = entry
        if time.monotonic() - ts > self._cache_ttl:
            del self._cache[key]
            return None
        return result

    def _put_cache(self, key: str, result: NormalizedTrustResult) -> None:
        if self._cache_ttl > 0:
            self._cache[key] = (time.monotonic(), result)

    def clear_cache(self) -> int:
        """Clear all cached trust results. Returns number of entries removed."""
        count = len(self._cache)
        self._cache.clear()
        return count

    def set_cache_ttl(self, ttl: float) -> float:
        """Update cache TTL. Returns previous TTL value."""
        old = self._cache_ttl
        self._cache_ttl = max(ttl, 0.0)
        if self._cache_ttl <= 0:
            self._cache.clear()
        return old

    def cache_stats(self) -> dict[str, Any]:
        """Return cache hit/miss statistics."""
        total = self._cache_hits + self._cache_misses
        return {
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "total": total,
            "hit_rate": self._cache_hits / total if total > 0 else 0.0,
            "cached_entries": len(self._cache),
            "ttl": self._cache_ttl,
        }

    def analyze(
        self,
        *,
        context: dict[str, Any],
        payload: dict[str, Any],
    ) -> NormalizedTrustResult | None:
        cache_key = self._cache_key(context, payload)
        cached = self._get_cached(cache_key)
        if cached is not None:
            self._cache_hits += 1
            return cached
        self._cache_misses += 1
        agent = self.select_agent(context)
        if agent is None:
            return None
        try:
            result = coerce_normalized_trust_result(agent.analyze(payload))
        except ValidationError:
            raise
        except Exception as exc:  # noqa: BLE001
            result = self._handle_agent_failure(agent, context, payload, exc)
            if result is None:
                return None
        self._put_cache(cache_key, result)
        self._record_history(result)
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

    def persist_history(self) -> None:
        """Manually persist trust history data to disk."""
        if self.history_tracker is None or self._history_path is None:
            return
        try:
            self.history_tracker.save(self._history_path)
        except OSError:
            _log.warning("Failed to persist history data", exc_info=True)

    async def analyze_async(
        self,
        *,
        context: dict[str, Any],
        payload: dict[str, Any],
    ) -> NormalizedTrustResult | None:
        """Async wrapper — runs sync agent analysis in a thread executor."""
        import asyncio

        return await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.analyze(context=context, payload=payload),
        )

    async def analyze_multi_async(
        self,
        *,
        context: dict[str, Any],
        payload: dict[str, Any],
    ) -> NormalizedTrustResult | None:
        """Async wrapper — runs sync multi-agent analysis in a thread executor."""
        import asyncio

        return await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.analyze_multi(context=context, payload=payload),
        )

    def trust_feature_attribution(
        self,
        trust_result: NormalizedTrustResult,
    ) -> dict[str, Any]:
        """Compute SHAP-like feature attribution from trust component weights.

        Returns a dict mapping component names to their contribution
        towards the final trust score, ordered by absolute impact.
        """
        components = trust_result.component_breakdown
        if not components:
            return {
                "attributions": [],
                "baseline": 0.0,
                "total_score": trust_result.trust_score,
                "top_driver": None,
            }

        # Compute each component's contribution = weight * score
        attributions: list[dict[str, Any]] = []
        total_weight = sum(c.weight for c in components.values()) or 1.0
        for name, comp in components.items():
            contribution = (comp.weight / total_weight) * comp.score
            attributions.append(
                {
                    "component": name,
                    "weight": comp.weight,
                    "score": comp.score,
                    "contribution": round(contribution, 4),
                    "impact_pct": round(
                        contribution / trust_result.trust_score * 100,
                        1,
                    )
                    if trust_result.trust_score > 0
                    else 0.0,
                }
            )

        # Sort by absolute contribution descending
        attributions.sort(key=lambda a: abs(a["contribution"]), reverse=True)

        return {
            "attributions": attributions,
            "baseline": 0.0,
            "total_score": trust_result.trust_score,
            "top_driver": attributions[0]["component"] if attributions else None,
        }

    def analyze_with_connector(
        self,
        *,
        connector: Any,
        identifier: str,
        connector_context: dict[str, Any] | None = None,
        trust_context: dict[str, Any] | None = None,
    ) -> NormalizedTrustResult | None:
        """Fetch data from a connector and pipe into trust analysis.

        Parameters
        ----------
        connector : DataConnector
            A connector instance to fetch data from.
        identifier : str
            Connector-specific identifier (instrument_id, story_id, etc.).
        connector_context : dict
            Additional context for the connector fetch.
        trust_context : dict
            Context for trust agent routing (must include 'domain').

        On connector fetch failure, attempts heuristic fallback for retryable
        errors, or returns a degraded result for auth/schema errors.
        """
        from tollama.xai.connectors.protocol import ConnectorFetchError

        try:
            result = connector.fetch(identifier, connector_context or {})
        except ConnectorFetchError as exc:
            _log.warning(
                "Connector %s fetch failed for %s: %s",
                connector.connector_name,
                identifier,
                exc,
            )
            domain = (trust_context or {}).get("domain", getattr(connector, "domain", "unknown"))
            context = trust_context or {"domain": domain}

            # For retryable errors, try heuristic fallback with empty payload
            if exc.error.retryable:
                fallback = self._find_heuristic_fallback(
                    context,
                    exclude="",  # no agent to exclude since connector failed
                )
                if fallback is not None:
                    _log.info(
                        "Connector fetch failed, falling back to heuristic agent %s",
                        fallback.agent_name,
                    )
                    try:
                        return coerce_normalized_trust_result(
                            fallback.analyze({"identifier": identifier}),
                        )
                    except Exception:  # noqa: BLE001
                        _log.warning("Heuristic fallback also failed", exc_info=True)

            # Return degraded result
            return self._build_degraded_result(
                agent_name=f"connector:{connector.connector_name}",
                domain=domain,
                error_type=exc.error.error_type,
                message=str(exc),
            )

        context = trust_context or {"domain": result.domain}
        payload = {
            "connector_data": result.payload,
            "source_id": result.source_id,
            "source_type": result.source_type,
            "freshness_seconds": result.freshness_seconds,
            **result.payload,
        }
        return self.analyze(context=context, payload=payload)

    def analyze_with_auto_connector(
        self,
        *,
        domain: str,
        identifier: str,
        connector_context: dict[str, Any] | None = None,
    ) -> NormalizedTrustResult | None:
        """Auto-select a connector by domain and pipe into trust analysis.

        Requires ``connector_registry`` to be set on this router instance.
        Returns None if no connector matches or fetch fails.
        """
        if self.connector_registry is None:
            _log.debug("No connector registry configured")
            return None
        connector = self.connector_registry.get(domain, identifier, connector_context)
        if connector is None:
            _log.debug("No connector found for domain=%s identifier=%s", domain, identifier)
            return None
        return self.analyze_with_connector(
            connector=connector,
            identifier=identifier,
            connector_context=connector_context,
            trust_context={"domain": domain},
        )

    def gate_decision(
        self,
        trust_result: NormalizedTrustResult,
        *,
        trust_threshold: float = 0.5,
    ) -> dict[str, Any]:
        """Evaluate trust gates for auto-execution gating.

        Returns a dict with:
          allowed: bool — whether auto-execution should proceed
          gates: list of gate evaluation dicts
          risk_category: str
          trust_score: float
        """
        gates: list[dict[str, Any]] = []
        allowed = True

        # Gate A: Trust Score
        if trust_result.trust_score < trust_threshold:
            gates.append(
                {
                    "gate": "trust_score",
                    "status": "BLOCK",
                    "detail": (
                        f"trust_score {trust_result.trust_score:.2f} "
                        f"< threshold {trust_threshold:.2f}"
                    ),
                }
            )
            allowed = False
        else:
            gates.append(
                {
                    "gate": "trust_score",
                    "status": "PASS",
                    "detail": (
                        f"trust_score {trust_result.trust_score:.2f} "
                        f">= threshold {trust_threshold:.2f}"
                    ),
                }
            )

        # Gate B: Critical violations
        critical = [v for v in trust_result.violations if v.severity == "critical"]
        if critical:
            names = ", ".join(v.name for v in critical)
            gates.append(
                {
                    "gate": "constraint_violations",
                    "status": "BLOCK",
                    "detail": f"{len(critical)} critical violation(s): {names}",
                }
            )
            allowed = False
        else:
            gates.append(
                {
                    "gate": "constraint_violations",
                    "status": "PASS",
                    "detail": "no critical violations",
                }
            )

        # Gate C: Risk Category
        if trust_result.risk_category == "RED":
            gates.append(
                {
                    "gate": "risk_category",
                    "status": "BLOCK",
                    "detail": "risk_category RED",
                }
            )
            allowed = False
        elif trust_result.risk_category == "YELLOW":
            gates.append(
                {
                    "gate": "risk_category",
                    "status": "WARN",
                    "detail": "risk_category YELLOW",
                }
            )
        else:
            gates.append(
                {
                    "gate": "risk_category",
                    "status": "PASS",
                    "detail": f"risk_category {trust_result.risk_category}",
                }
            )

        return {
            "allowed": allowed,
            "gates": gates,
            "risk_category": trust_result.risk_category,
            "trust_score": trust_result.trust_score,
        }

    def _handle_agent_failure(
        self,
        agent: TrustAgent,
        context: dict[str, Any],
        payload: dict[str, Any],
        exc: Exception,
    ) -> NormalizedTrustResult | None:
        """Handle agent analysis failure with fallback or degraded result.

        Fallback policy:
        - Timeout / 5xx / network errors (retryable) → try heuristic fallback agent
        - Auth errors (401/403, not retryable) → return degraded result, no fallback
        - Schema mismatch (422, not retryable) → return degraded result with error info
        - Other errors → try heuristic fallback
        """
        from tollama.xai.connectors.protocol import ConnectorFetchError

        error_type = "unknown"
        retryable = True

        if isinstance(exc, ConnectorFetchError):
            error_type = exc.error.error_type
            retryable = exc.error.retryable
        elif isinstance(exc, TimeoutError):
            error_type = "network"
            retryable = True

        _log.warning(
            "Agent %s failed (error_type=%s, retryable=%s): %s",
            agent.agent_name,
            error_type,
            retryable,
            exc,
        )

        # Non-retryable auth/schema errors → degraded result, no fallback
        if error_type in ("auth", "schema") or (not retryable and error_type != "not_found"):
            return self._build_degraded_result(
                agent_name=agent.agent_name,
                domain=context.get("domain", agent.domain),
                error_type=error_type,
                message=str(exc),
            )

        # Retryable errors → try heuristic fallback
        fallback = self._find_heuristic_fallback(context, exclude=agent.agent_name)
        if fallback is not None:
            _log.info(
                "Falling back from %s to heuristic agent %s",
                agent.agent_name,
                fallback.agent_name,
            )
            try:
                return coerce_normalized_trust_result(fallback.analyze(payload))
            except Exception:  # noqa: BLE001
                _log.warning(
                    "Heuristic fallback %s also failed",
                    fallback.agent_name,
                    exc_info=True,
                )

        # All fallbacks exhausted → degraded result
        return self._build_degraded_result(
            agent_name=agent.agent_name,
            domain=context.get("domain", agent.domain),
            error_type=error_type,
            message=str(exc),
        )

    def _find_heuristic_fallback(
        self,
        context: dict[str, Any],
        *,
        exclude: str,
    ) -> TrustAgent | None:
        """Find a heuristic (local) fallback agent for the given context.

        Excludes the agent that already failed. Returns None if no fallback
        is available.
        """
        matches = self.registry.resolve(context)
        candidates = [a for a in matches if a.agent_name != exclude]
        if not candidates:
            return None
        return sorted(candidates, key=lambda a: getattr(a, "priority", 100))[0]

    @staticmethod
    def _build_degraded_result(
        *,
        agent_name: str,
        domain: str,
        error_type: str,
        message: str,
    ) -> NormalizedTrustResult:
        """Build a degraded NormalizedTrustResult for when analysis fails."""
        return NormalizedTrustResult(
            agent_name=agent_name,
            domain=domain,
            trust_score=0.0,
            risk_category="RED",
            calibration_status="poorly_calibrated",
            component_breakdown={},
            violations=[
                TrustViolation(
                    name=f"agent_failure:{error_type}",
                    severity="critical",
                    detail=message[:500],
                ),
            ],
            why_trusted=f"Agent {agent_name} failed ({error_type}): {message[:200]}",
            evidence=TrustEvidence(
                source_type="degraded",
                source_ids=[],
                attributes={"error_type": error_type, "degraded": True},
            ),
            audit=TrustAudit(
                formula_version="degraded-v1",
                agent_version="0.1.0",
            ),
        )

    def _record_history(self, result: NormalizedTrustResult) -> None:
        """Record a trust result in the history tracker if available."""
        if self.history_tracker is None:
            return
        try:
            self.history_tracker.record(
                agent_name=result.agent_name,
                domain=result.domain,
                trust_score=result.trust_score,
                risk_category=result.risk_category,
                calibration_status=result.calibration_status,
            )
        except Exception:  # noqa: BLE001
            _log.debug("Failed to record trust history", exc_info=True)

    def _maybe_auto_persist(self) -> None:
        """Persist calibration and history data every N analyze calls."""
        if self._auto_persist_every <= 0:
            return
        self._analyze_count += 1
        if self._analyze_count % self._auto_persist_every == 0:
            self.persist_calibration()
            self.persist_history()


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
    agg_score = sum(r.trust_score * w for r, w in zip(results, inv_priorities)) / total_inv

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


def build_default_trust_router(
    *,
    enable_calibration: bool = True,
    enable_history: bool = True,
) -> TrustRouter:
    """Build the default in-repo trust agent router.

    When *enable_calibration* is True (default), loads persisted calibration
    data from disk and auto-persists every 10 analyze calls.

    When *enable_history* is True (default), loads persisted trust history
    from disk for trend analysis.
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

    history_tracker = None
    history_path = None
    if enable_history:
        from tollama.xai.trust_history import TrustHistoryTracker, default_history_path

        history_path = default_history_path()
        try:
            history_tracker = TrustHistoryTracker.load(history_path)
        except Exception:  # noqa: BLE001
            _log.warning("Failed to load trust history, starting fresh", exc_info=True)
            history_tracker = TrustHistoryTracker()

    registry = TrustAgentRegistry()
    registry.register(MarketCalibrationTrustAgent())
    financial_agent = FinancialMarketTrustAgent(
        calibration_tracker=calibration_tracker,
    )
    registry.register(financial_agent)
    registry.register(SupplyChainTrustAgent(calibration_tracker=calibration_tracker))
    registry.register(NewsTrustAgent(calibration_tracker=calibration_tracker))
    registry.register(GeopoliticalTrustAgent(calibration_tracker=calibration_tracker))
    registry.register(RegulatoryTrustAgent(calibration_tracker=calibration_tracker))
    # Wire connector registry for auto-discovery
    connector_reg = None
    try:
        from tollama.xai.connectors.helpers import build_default_connector_registry

        connector_reg = build_default_connector_registry()
    except Exception:  # noqa: BLE001
        _log.debug("Failed to build connector registry", exc_info=True)

    router = TrustRouter(
        registry,
        calibration_tracker=calibration_tracker,
        calibration_path=calibration_path,
        history_tracker=history_tracker,
        history_path=history_path,
        connector_registry=connector_reg,
    )
    financial_agent._trust_router = router
    return router


__all__ = [
    "TrustAgentRegistry",
    "TrustRouter",
    "build_default_trust_router",
]
