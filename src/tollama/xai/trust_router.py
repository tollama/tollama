"""
tollama.xai.trust_router — Registry and router for domain trust agents.
"""

from __future__ import annotations

from typing import Any

from tollama.xai.trust_contract import (
    NormalizedTrustResult,
    TrustAgent,
    coerce_normalized_trust_result,
)


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
    }

    def __init__(
        self,
        registry: TrustAgentRegistry,
        primary_order: dict[str, list[str]] | None = None,
    ):
        self.registry = registry
        self.primary_order = primary_order or self.DEFAULT_PRIMARY_ORDER

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
        return coerce_normalized_trust_result(agent.analyze(payload))


def build_default_trust_router() -> TrustRouter:
    """Build the default in-repo trust agent router."""
    from tollama.xai.trust_agents import (
        FinancialMarketTrustAgent,
        MarketCalibrationTrustAgent,
        NewsTrustAgent,
        SupplyChainTrustAgent,
    )

    registry = TrustAgentRegistry()
    registry.register(MarketCalibrationTrustAgent())
    registry.register(FinancialMarketTrustAgent())
    registry.register(SupplyChainTrustAgent())
    registry.register(NewsTrustAgent())
    return TrustRouter(registry)


__all__ = [
    "TrustAgentRegistry",
    "TrustRouter",
    "build_default_trust_router",
]
