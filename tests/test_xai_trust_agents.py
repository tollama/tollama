"""Tests for normalized trust agents, routing, and engine integration."""

from __future__ import annotations

import os
import sys

import pytest
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


def test_default_router_returns_none_when_no_agent_matches():
    from tollama.xai.trust_router import build_default_trust_router

    router = build_default_trust_router()
    result = router.analyze(
        context={"domain": "unsupported_domain"},
        payload={"foo": "bar"},
    )

    assert result is None


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
