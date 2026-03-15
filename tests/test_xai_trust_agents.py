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
