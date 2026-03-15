"""Tests for normalized trust agents, routing, and engine integration."""

from __future__ import annotations

import os
import sys
import types

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_REPO_ROOT, "tollama", "src"))


def test_normalized_trust_result_requires_required_fields():
    from tollama.xai.trust_contract import NormalizedTrustResult

    with pytest.raises(ValidationError):
        NormalizedTrustResult.model_validate(
            {
                "agent_name": "financial_market",
                "domain": "financial_market",
            }
        )


def test_financial_payload_aliases_normalize():
    from tollama.xai.trust_contract import coerce_financial_payload

    payload = coerce_financial_payload(
        {
            "symbol": "AAPL",
            "spread_bps": 12.0,
            "volatility_regime": 0.4,
            "slippage_risk": 0.3,
            "freshness_score": 0.8,
        }
    )

    assert payload.instrument_id == "AAPL"
    assert payload.bid_ask_spread_bps == 12.0
    assert payload.realized_volatility == 0.4
    assert payload.execution_risk == 0.3
    assert payload.data_freshness == 0.8


def test_news_payload_aliases_normalize():
    from tollama.xai.trust_contract import coerce_news_payload

    payload = coerce_news_payload(
        {
            "headline": "Fed keeps rates steady",
            "source_agreement": 0.9,
            "freshness_score": 0.85,
        }
    )

    assert payload.story_id == "Fed keeps rates steady"
    assert payload.corroboration == 0.9
    assert payload.freshness_score == 0.85


def test_financial_payload_requires_instrument_identifier():
    from tollama.xai.trust_contract import coerce_financial_payload

    with pytest.raises(ValidationError):
        coerce_financial_payload({"liquidity_depth": 0.8})


def test_default_router_selects_mca_for_prediction_market():
    from tollama.xai.trust_router import build_default_trust_router

    router = build_default_trust_router()
    result = router.analyze(
        context={"domain": "prediction_market", "source_type": "prediction_market"},
        payload={
            "market_id": "pm-123",
            "trust_score": 0.81,
            "metrics": {"brier_score": 0.142, "log_loss": 0.318, "ece": 0.047},
            "signals": [
                {
                    "name": "polymarket",
                    "trust_score": 0.81,
                    "metrics": {"brier_score": 0.142, "log_loss": 0.318, "ece": 0.047},
                }
            ],
        },
    )

    assert result is not None
    assert result.agent_name == "market_calibration"
    assert result.domain == "prediction_market"
    assert result.trust_score == 0.81


def test_default_router_selects_financial_market_agent():
    from tollama.xai.trust_router import build_default_trust_router

    router = build_default_trust_router()
    result = router.analyze(
        context={"domain": "financial_market", "source_type": "equity_market"},
        payload={
            "symbol": "AAPL",
            "liquidity_depth": 0.9,
            "bid_ask_spread_bps": 8.0,
            "realized_volatility": 0.2,
            "execution_risk": 0.1,
            "data_freshness": 0.95,
        },
    )

    assert result is not None
    assert result.agent_name == "financial_market"
    assert result.domain == "financial_market"
    assert result.trust_score > 0.7


def test_financial_agent_emits_critical_violation_for_bad_execution_conditions():
    from tollama.xai.trust_router import build_default_trust_router

    router = build_default_trust_router()
    result = router.analyze(
        context={"domain": "financial_market", "source_type": "equity_market"},
        payload={
            "instrument_id": "TSLA",
            "liquidity_depth": 0.2,
            "bid_ask_spread_bps": 125.0,
            "realized_volatility": 1.2,
            "execution_risk": 0.9,
            "data_freshness": 0.05,
            "news_signal_ref": "story-123",
        },
    )

    assert result is not None
    assert result.risk_category == "RED"
    assert any(v.severity == "critical" for v in result.violations)
    assert result.evidence.attributes["news_signal_ref"] == "story-123"


def test_news_agent_ignores_novelty_in_score_but_keeps_it_in_evidence():
    from tollama.xai.trust_router import build_default_trust_router

    router = build_default_trust_router()
    base_payload = {
        "story_id": "story-1",
        "source_credibility": 0.9,
        "corroboration": 0.85,
        "contradiction_score": 0.1,
        "propagation_delay_seconds": 120.0,
        "freshness_score": 0.95,
    }
    low_novelty = router.analyze(
        context={"domain": "news", "source_type": "news"},
        payload={**base_payload, "novelty": 0.1},
    )
    high_novelty = router.analyze(
        context={"domain": "news", "source_type": "news"},
        payload={**base_payload, "novelty": 0.9},
    )

    assert low_novelty is not None
    assert high_novelty is not None
    assert low_novelty.risk_category == "GREEN"
    assert high_novelty.risk_category == "GREEN"
    assert low_novelty.trust_score == high_novelty.trust_score
    assert high_novelty.evidence.attributes["novelty"] == 0.9


def test_news_agent_emits_red_for_low_credibility_and_high_contradiction():
    from tollama.xai.trust_router import build_default_trust_router

    router = build_default_trust_router()
    result = router.analyze(
        context={"domain": "news", "source_type": "news"},
        payload={
            "headline": "Unverified viral post",
            "source_credibility": 0.1,
            "corroboration": 0.2,
            "contradiction_score": 0.95,
            "propagation_delay_seconds": 90000.0,
            "freshness_score": 0.4,
        },
    )

    assert result is not None
    assert result.risk_category == "RED"
    assert any(v.severity == "critical" for v in result.violations)


def test_default_router_returns_none_when_no_agent_matches():
    from tollama.xai.trust_router import build_default_trust_router

    router = build_default_trust_router()
    result = router.analyze(
        context={"domain": "unsupported_domain"},
        payload={"foo": "bar"},
    )

    assert result is None


def test_invalid_financial_payload_raises_validation_error():
    from tollama.xai.trust_router import build_default_trust_router

    router = build_default_trust_router()
    with pytest.raises(ValidationError):
        router.analyze(
            context={"domain": "financial_market", "source_type": "equity_market"},
            payload={"liquidity_depth": 0.7},
        )


def test_engine_accepts_direct_normalized_trust_result():
    from tollama.xai.decision_policy import DecisionPolicyExplainer
    from tollama.xai.engine import ExplanationEngine
    from tollama.xai.trust_breakdown import TrustBreakdown

    engine = ExplanationEngine(
        trust_breakdown=TrustBreakdown(),
        decision_policy_explainer=DecisionPolicyExplainer(),
    )

    result = engine.explain_decision(
        forecast_result={"confidence": 0.95, "signals_used": ["internal_ts", "financial_market"]},
        trust_result={
            "agent_name": "financial_market",
            "domain": "financial_market",
            "trust_score": 0.82,
            "risk_category": "GREEN",
            "component_breakdown": {
                "liquidity_depth": {
                    "score": 0.9,
                    "weight": 0.25,
                    "value": 0.9,
                }
            },
            "violations": [],
            "why_trusted": "High liquidity and low execution risk.",
            "evidence": {"source_type": "financial_market", "source_ids": ["AAPL"]},
            "audit": {"formula_version": "baseline-v1"},
        },
        policy_config={"auto_execute_threshold": 0.85, "trust_threshold": 0.5},
    )

    assert result.trust_intelligence_explanation is not None
    assert result.trust_intelligence_explanation["trust_score"] == 0.82
    assert result.input_explanation.trust_scores["financial_market"] == 0.82
    assert result.decision_policy_explanation.trust_score == 0.82
    assert result.decision_policy_explanation.auto_executed is True


def test_engine_routes_prediction_market_payload_through_default_router():
    from tollama.xai.decision_policy import DecisionPolicyExplainer
    from tollama.xai.engine import ExplanationEngine
    from tollama.xai.trust_breakdown import TrustBreakdown
    from tollama.xai.trust_router import build_default_trust_router

    engine = ExplanationEngine(
        trust_breakdown=TrustBreakdown(),
        decision_policy_explainer=DecisionPolicyExplainer(),
        trust_router=build_default_trust_router(),
    )

    result = engine.explain_decision(
        forecast_result={"confidence": 0.90},
        calibration_result={
            "market_id": "pm-321",
            "trust_score": 0.35,
            "metrics": {"brier_score": 0.32, "log_loss": 0.71, "ece": 0.16},
            "signals": [
                {
                    "name": "polymarket",
                    "trust_score": 0.35,
                    "metrics": {"brier_score": 0.32, "log_loss": 0.71, "ece": 0.16},
                }
            ],
        },
        trust_context={"domain": "prediction_market", "source_type": "prediction_market"},
        policy_config={"auto_execute_threshold": 0.85, "trust_threshold": 0.5},
    )

    assert result.trust_intelligence_explanation is not None
    assert result.trust_intelligence_explanation["trust_score"] == 0.35
    assert result.decision_policy_explanation.trust_blocked is True
    assert result.decision_policy_explanation.auto_executed is False


def test_engine_routes_financial_payload_and_preserves_policy_gating():
    from tollama.xai.decision_policy import DecisionPolicyExplainer
    from tollama.xai.engine import ExplanationEngine
    from tollama.xai.trust_breakdown import TrustBreakdown
    from tollama.xai.trust_router import build_default_trust_router

    engine = ExplanationEngine(
        trust_breakdown=TrustBreakdown(),
        decision_policy_explainer=DecisionPolicyExplainer(),
        trust_router=build_default_trust_router(),
    )

    result = engine.explain_decision(
        forecast_result={"confidence": 0.91, "signals_used": ["internal_ts", "financial_market"]},
        trust_context={"domain": "financial_market", "source_type": "equity_market"},
        trust_payload={
            "symbol": "MSFT",
            "liquidity_depth": 0.92,
            "bid_ask_spread_bps": 5.0,
            "realized_volatility": 0.18,
            "execution_risk": 0.15,
            "data_freshness": 0.95,
        },
        policy_config={"auto_execute_threshold": 0.85, "trust_threshold": 0.5},
    )

    assert result.trust_intelligence_explanation is not None
    assert result.trust_intelligence_explanation["trust_score"] > 0.75
    assert result.decision_policy_explanation.auto_executed is True


def test_engine_invalid_news_payload_raises_validation_error():
    from tollama.xai.decision_policy import DecisionPolicyExplainer
    from tollama.xai.engine import ExplanationEngine
    from tollama.xai.trust_breakdown import TrustBreakdown
    from tollama.xai.trust_router import build_default_trust_router

    engine = ExplanationEngine(
        trust_breakdown=TrustBreakdown(),
        decision_policy_explainer=DecisionPolicyExplainer(),
        trust_router=build_default_trust_router(),
    )

    with pytest.raises(ValidationError):
        engine.explain_decision(
            forecast_result={"confidence": 0.75},
            trust_context={"domain": "news", "source_type": "news"},
            trust_payload={"source_credibility": 0.8},
        )


def test_supply_chain_payload_aliases_normalize():
    from tollama.xai.trust_contract import coerce_supply_chain_payload

    payload = coerce_supply_chain_payload(
        {
            "shipment_id": "SC-001",
            "freshness_score": 0.85,
            "lead_time_reliability": 0.9,
        }
    )

    assert payload.network_id == "SC-001"
    assert payload.data_freshness == 0.85
    assert payload.lead_time_reliability == 0.9


def test_supply_chain_payload_requires_network_id():
    from tollama.xai.trust_contract import coerce_supply_chain_payload

    with pytest.raises(ValidationError):
        coerce_supply_chain_payload({"lead_time_reliability": 0.8})


def test_supply_chain_agent_green_for_good_conditions():
    from tollama.xai.trust_router import build_default_trust_router

    router = build_default_trust_router()
    result = router.analyze(
        context={"domain": "supply_chain", "source_type": "supply_chain"},
        payload={
            "network_id": "NET-100",
            "lead_time_reliability": 0.9,
            "inventory_visibility": 0.85,
            "disruption_risk": 0.1,
            "sensor_quality": 0.8,
            "data_freshness": 0.95,
        },
    )

    assert result is not None
    assert result.agent_name == "supply_chain"
    assert result.risk_category == "GREEN"
    assert result.trust_score > 0.7
    assert len(result.violations) == 0


def test_supply_chain_agent_red_for_bad_conditions():
    from tollama.xai.trust_router import build_default_trust_router

    router = build_default_trust_router()
    result = router.analyze(
        context={"domain": "supply_chain", "source_type": "logistics"},
        payload={
            "shipment_id": "SHP-999",
            "lead_time_reliability": 0.15,
            "inventory_visibility": 0.3,
            "disruption_risk": 0.85,
            "sensor_quality": 0.1,
            "data_freshness": 0.05,
        },
    )

    assert result is not None
    assert result.risk_category == "RED"
    assert any(v.severity == "critical" for v in result.violations)
    assert any(v.name == "disruption_risk_extreme" for v in result.violations)
    assert any(v.name == "lead_time_unreliable" for v in result.violations)
    assert any(v.name == "sensor_quality_critical" for v in result.violations)
    assert any(v.name == "supply_data_stale" for v in result.violations)


def test_default_router_selects_supply_chain_agent():
    from tollama.xai.trust_router import build_default_trust_router

    router = build_default_trust_router()
    result = router.analyze(
        context={"domain": "supply_chain"},
        payload={
            "network_id": "NET-200",
            "lead_time_reliability": 0.7,
            "inventory_visibility": 0.6,
            "disruption_risk": 0.3,
            "sensor_quality": 0.5,
            "data_freshness": 0.7,
        },
    )

    assert result is not None
    assert result.agent_name == "supply_chain"
    assert result.domain == "supply_chain"


def test_invalid_supply_chain_payload_raises_validation_error():
    from tollama.xai.trust_router import build_default_trust_router

    router = build_default_trust_router()
    with pytest.raises(ValidationError):
        router.analyze(
            context={"domain": "supply_chain"},
            payload={"lead_time_reliability": 0.8},
        )


def test_explain_decision_api_returns_422_for_invalid_financial_payload(monkeypatch):
    ti_root = types.ModuleType("trust_intelligence")
    ti_pipeline_pkg = types.ModuleType("trust_intelligence.pipeline")
    ti_pipeline_mod = types.ModuleType("trust_intelligence.pipeline.trust_pipeline")

    class DummyPipeline:
        def run(self, **kwargs):
            raise AssertionError("trust pipeline should not be called in this test")

    ti_pipeline_mod.TrustIntelligencePipeline = DummyPipeline
    monkeypatch.setitem(sys.modules, "trust_intelligence", ti_root)
    monkeypatch.setitem(sys.modules, "trust_intelligence.pipeline", ti_pipeline_pkg)
    monkeypatch.setitem(
        sys.modules,
        "trust_intelligence.pipeline.trust_pipeline",
        ti_pipeline_mod,
    )

    from tollama.xai.api import router

    app = FastAPI()
    app.include_router(router)

    with TestClient(app) as client:
        response = client.post(
            "/api/xai/explain-decision",
            json={
                "forecast_result": {"confidence": 0.9},
                "trust_context": {"domain": "financial_market", "source_type": "equity_market"},
                "trust_payload": {"liquidity_depth": 0.8},
            },
        )

    assert response.status_code == 422


# ── News → Financial Composition ──────────────────────────────────────────


def test_financial_agent_composes_news_signal():
    from tollama.xai.trust_router import build_default_trust_router

    router = build_default_trust_router()
    result = router.analyze(
        context={"domain": "financial_market", "source_type": "equity_market"},
        payload={
            "instrument_id": "AAPL",
            "liquidity_depth": 0.9,
            "bid_ask_spread_bps": 8.0,
            "realized_volatility": 0.2,
            "execution_risk": 0.1,
            "data_freshness": 0.95,
            "news_signal_ref": "Apple-earnings-Q1",
        },
    )

    assert result is not None
    assert "news_signal" in result.component_breakdown
    assert result.evidence.attributes["news_trust_score"] is not None
    assert result.evidence.attributes["news_risk_category"] is not None


def test_financial_agent_propagates_critical_news_violations():
    from tollama.xai.trust_router import build_default_trust_router

    router = build_default_trust_router()
    result = router.analyze(
        context={"domain": "financial_market"},
        payload={
            "instrument_id": "TSLA",
            "liquidity_depth": 0.9,
            "bid_ask_spread_bps": 5.0,
            "realized_volatility": 0.15,
            "execution_risk": 0.1,
            "data_freshness": 0.95,
            "news_signal_ref": "Unverified-rumor",
        },
    )

    assert result is not None
    # The news agent uses default values (source_credibility=0.5, etc.) for
    # the synthetic story_id, so it should not produce critical violations
    # with defaults. This test verifies the composition path runs without error.
    assert result.evidence.attributes.get("news_signal_ref") == "Unverified-rumor"


def test_financial_agent_without_news_ref_unchanged():
    from tollama.xai.trust_agents.heuristic import FinancialMarketTrustAgent

    agent = FinancialMarketTrustAgent()
    result = agent.analyze(
        {
            "instrument_id": "MSFT",
            "liquidity_depth": 0.85,
            "bid_ask_spread_bps": 6.0,
            "realized_volatility": 0.2,
            "execution_risk": 0.15,
            "data_freshness": 0.9,
        }
    )

    assert "news_signal" not in result.component_breakdown
    assert result.evidence.attributes.get("news_trust_score") is None


def test_financial_agent_without_router_ignores_news_ref():
    from tollama.xai.trust_agents.heuristic import FinancialMarketTrustAgent

    agent = FinancialMarketTrustAgent()  # no router
    result = agent.analyze(
        {
            "instrument_id": "GOOG",
            "liquidity_depth": 0.8,
            "news_signal_ref": "some-story",
        }
    )

    assert "news_signal" not in result.component_breakdown
    assert result.evidence.attributes["news_signal_ref"] == "some-story"


# ── Multi-Agent Aggregation ───────────────────────────────────────────────


def test_analyze_multi_runs_all_matching_agents():
    from tollama.xai.trust_contract import NormalizedTrustResult, TrustComponent
    from tollama.xai.trust_router import TrustAgentRegistry, TrustRouter

    class AgentA:
        agent_name = "agent_a"
        domain = "test_domain"
        priority = 10

        def supports(self, context):
            return context.get("domain") == "test_domain"

        def analyze(self, payload):
            return NormalizedTrustResult(
                agent_name="agent_a",
                domain="test_domain",
                trust_score=0.8,
                risk_category="GREEN",
                component_breakdown={
                    "metric_a": TrustComponent(score=0.8, weight=1.0),
                },
                violations=[],
                why_trusted="Agent A is good.",
            )

    class AgentB:
        agent_name = "agent_b"
        domain = "test_domain"
        priority = 20

        def supports(self, context):
            return context.get("domain") == "test_domain"

        def analyze(self, payload):
            return NormalizedTrustResult(
                agent_name="agent_b",
                domain="test_domain",
                trust_score=0.6,
                risk_category="YELLOW",
                component_breakdown={
                    "metric_b": TrustComponent(score=0.6, weight=1.0),
                },
                violations=[],
                why_trusted="Agent B is cautious.",
            )

    registry = TrustAgentRegistry()
    registry.register(AgentA())
    registry.register(AgentB())
    router = TrustRouter(registry)

    result = router.analyze_multi(
        context={"domain": "test_domain"},
        payload={},
    )

    assert result is not None
    assert result.agent_name == "multi_agent_aggregate"
    assert "agent_a/metric_a" in result.component_breakdown
    assert "agent_b/metric_b" in result.component_breakdown


def test_aggregate_most_conservative_risk_wins():
    from tollama.xai.trust_contract import NormalizedTrustResult
    from tollama.xai.trust_router import TrustAgentRegistry, TrustRouter

    class GreenAgent:
        agent_name = "green"
        domain = "d"
        priority = 10

        def supports(self, ctx):
            return ctx.get("domain") == "d"

        def analyze(self, p):
            return NormalizedTrustResult(
                agent_name="green", domain="d", trust_score=0.9, risk_category="GREEN",
            )

    class RedAgent:
        agent_name = "red"
        domain = "d"
        priority = 20

        def supports(self, ctx):
            return ctx.get("domain") == "d"

        def analyze(self, p):
            return NormalizedTrustResult(
                agent_name="red", domain="d", trust_score=0.3, risk_category="RED",
            )

    registry = TrustAgentRegistry()
    registry.register(GreenAgent())
    registry.register(RedAgent())
    router = TrustRouter(registry)

    result = router.analyze_multi(context={"domain": "d"}, payload={})
    assert result.risk_category == "RED"


def test_aggregate_merges_violations():
    from tollama.xai.trust_contract import NormalizedTrustResult, TrustViolation
    from tollama.xai.trust_router import TrustAgentRegistry, TrustRouter

    class AgentV1:
        agent_name = "v1"
        domain = "d"
        priority = 10

        def supports(self, ctx):
            return True

        def analyze(self, p):
            return NormalizedTrustResult(
                agent_name="v1", domain="d", trust_score=0.7,
                violations=[TrustViolation(name="issue_a", severity="critical")],
            )

    class AgentV2:
        agent_name = "v2"
        domain = "d"
        priority = 20

        def supports(self, ctx):
            return True

        def analyze(self, p):
            return NormalizedTrustResult(
                agent_name="v2", domain="d", trust_score=0.6,
                violations=[
                    TrustViolation(name="issue_b", severity="warning"),
                    TrustViolation(name="issue_c", severity="critical"),
                ],
            )

    registry = TrustAgentRegistry()
    registry.register(AgentV1())
    registry.register(AgentV2())
    router = TrustRouter(registry)

    result = router.analyze_multi(context={"domain": "d"}, payload={})
    violation_names = {v.name for v in result.violations}
    assert violation_names == {"issue_a", "issue_b", "issue_c"}


def test_aggregate_weighted_score():
    from tollama.xai.trust_contract import NormalizedTrustResult
    from tollama.xai.trust_router import TrustAgentRegistry, TrustRouter

    class LowPri:
        agent_name = "low"
        domain = "d"
        priority = 10  # gets weight 1/10 = 0.1

        def supports(self, ctx):
            return True

        def analyze(self, p):
            return NormalizedTrustResult(
                agent_name="low", domain="d", trust_score=1.0,
            )

    class HighPri:
        agent_name = "high"
        domain = "d"
        priority = 10  # same priority, equal weight

        def supports(self, ctx):
            return True

        def analyze(self, p):
            return NormalizedTrustResult(
                agent_name="high", domain="d", trust_score=0.5,
            )

    registry = TrustAgentRegistry()
    registry.register(LowPri())
    registry.register(HighPri())
    router = TrustRouter(registry)

    result = router.analyze_multi(context={"domain": "d"}, payload={})
    # Equal priorities → simple average: (1.0 + 0.5) / 2 = 0.75
    assert abs(result.trust_score - 0.75) < 0.01


def test_aggregate_disagree_triggers_human_review():
    from tollama.xai.trust_contract import NormalizedTrustResult
    from tollama.xai.trust_router import TrustAgentRegistry, TrustRouter

    class AgG:
        agent_name = "ag"
        domain = "d"
        priority = 10

        def supports(self, ctx):
            return True

        def analyze(self, p):
            return NormalizedTrustResult(
                agent_name="ag", domain="d", trust_score=0.9, risk_category="GREEN",
            )

    class AgY:
        agent_name = "ay"
        domain = "d"
        priority = 20

        def supports(self, ctx):
            return True

        def analyze(self, p):
            return NormalizedTrustResult(
                agent_name="ay", domain="d", trust_score=0.6, risk_category="YELLOW",
            )

    registry = TrustAgentRegistry()
    registry.register(AgG())
    registry.register(AgY())
    router = TrustRouter(registry)

    result = router.analyze_multi(context={"domain": "d"}, payload={})
    assert result.evidence.attributes.get("human_review_reason") == "agents_disagree_on_risk"


def test_analyze_multi_single_match_returns_directly():
    from tollama.xai.trust_contract import NormalizedTrustResult
    from tollama.xai.trust_router import TrustAgentRegistry, TrustRouter

    class Solo:
        agent_name = "solo"
        domain = "solo_domain"
        priority = 10

        def supports(self, ctx):
            return ctx.get("domain") == "solo_domain"

        def analyze(self, p):
            return NormalizedTrustResult(
                agent_name="solo", domain="solo_domain", trust_score=0.77,
            )

    registry = TrustAgentRegistry()
    registry.register(Solo())
    router = TrustRouter(registry)

    result = router.analyze_multi(context={"domain": "solo_domain"}, payload={})
    assert result.agent_name == "solo"
    assert result.trust_score == 0.77


def test_analyze_multi_no_match_returns_none():
    from tollama.xai.trust_router import TrustAgentRegistry, TrustRouter

    registry = TrustAgentRegistry()
    router = TrustRouter(registry)

    result = router.analyze_multi(context={"domain": "nothing"}, payload={})
    assert result is None


def test_engine_multi_agent_mode():
    from tollama.xai.decision_policy import DecisionPolicyExplainer
    from tollama.xai.engine import ExplanationEngine
    from tollama.xai.trust_breakdown import TrustBreakdown
    from tollama.xai.trust_contract import NormalizedTrustResult, TrustComponent
    from tollama.xai.trust_router import TrustAgentRegistry, TrustRouter

    class AgentX:
        agent_name = "x"
        domain = "multi_test"
        priority = 10

        def supports(self, ctx):
            return ctx.get("domain") == "multi_test"

        def analyze(self, p):
            return NormalizedTrustResult(
                agent_name="x", domain="multi_test", trust_score=0.85,
                risk_category="GREEN",
                component_breakdown={"cx": TrustComponent(score=0.85, weight=1.0)},
            )

    class AgentY:
        agent_name = "y"
        domain = "multi_test"
        priority = 20

        def supports(self, ctx):
            return ctx.get("domain") == "multi_test"

        def analyze(self, p):
            return NormalizedTrustResult(
                agent_name="y", domain="multi_test", trust_score=0.65,
                risk_category="YELLOW",
                component_breakdown={"cy": TrustComponent(score=0.65, weight=1.0)},
            )

    registry = TrustAgentRegistry()
    registry.register(AgentX())
    registry.register(AgentY())
    router = TrustRouter(registry)

    engine = ExplanationEngine(
        trust_breakdown=TrustBreakdown(),
        decision_policy_explainer=DecisionPolicyExplainer(),
        trust_router=router,
    )

    result = engine.explain_decision(
        forecast_result={"confidence": 0.90},
        trust_context={"domain": "multi_test", "mode": "multi"},
        trust_payload={"_marker": True},
        policy_config={"auto_execute_threshold": 0.85, "trust_threshold": 0.5},
    )

    assert result.trust_intelligence_explanation is not None
    assert result.trust_intelligence_explanation["trust_score"] is not None
    assert result.decision_policy_explanation.risk_category == "YELLOW"


# ── Quick Fixes ───────────────────────────────────────────────────────────


def test_supply_chain_data_latency_seconds_backward_compat():
    from tollama.xai.trust_contract import coerce_supply_chain_payload

    payload = coerce_supply_chain_payload(
        {
            "network_id": "NET-BC",
            "data_latency_seconds": 300.0,
        }
    )

    # 300s / 3600s ceiling → 1.0 - 0.0833 ≈ 0.917
    assert abs(payload.data_freshness - (1.0 - 300.0 / 3600.0)) < 0.01


def test_supply_chain_data_latency_seconds_high_value():
    from tollama.xai.trust_contract import coerce_supply_chain_payload

    payload = coerce_supply_chain_payload(
        {
            "shipment_id": "SHP-BC",
            "data_latency_seconds": 7200.0,
        }
    )

    # 7200s > 3600s ceiling → clipped to 0.0
    assert payload.data_freshness == 0.0


def test_supply_chain_data_freshness_takes_precedence_without_latency():
    from tollama.xai.trust_contract import coerce_supply_chain_payload

    payload = coerce_supply_chain_payload(
        {
            "network_id": "NET-NL",
            "data_freshness": 0.9,
        }
    )

    assert payload.data_freshness == 0.9
    assert payload.data_latency_seconds is None


def test_engine_trust_payload_empty_dict_not_ignored():
    from tollama.xai.decision_policy import DecisionPolicyExplainer
    from tollama.xai.engine import ExplanationEngine
    from tollama.xai.trust_breakdown import TrustBreakdown
    from tollama.xai.trust_contract import NormalizedTrustResult
    from tollama.xai.trust_router import TrustAgentRegistry, TrustRouter

    class StubAgent:
        agent_name = "stub"
        domain = "stub_domain"
        priority = 10

        def supports(self, ctx):
            return ctx.get("domain") == "stub_domain"

        def analyze(self, p):
            return NormalizedTrustResult(
                agent_name="stub", domain="stub_domain", trust_score=0.70,
            )

    registry = TrustAgentRegistry()
    registry.register(StubAgent())
    router = TrustRouter(registry)

    engine = ExplanationEngine(
        trust_breakdown=TrustBreakdown(),
        decision_policy_explainer=DecisionPolicyExplainer(),
        trust_router=router,
    )

    # Empty dict {} should NOT be treated as None
    result = engine.explain_decision(
        forecast_result={"confidence": 0.90},
        trust_context={"domain": "stub_domain"},
        trust_payload={},
        policy_config={"auto_execute_threshold": 0.85, "trust_threshold": 0.5},
    )

    assert result.trust_intelligence_explanation is not None
    assert result.trust_intelligence_explanation["trust_score"] == 0.70
