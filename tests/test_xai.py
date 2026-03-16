"""
tests/test_xai.py — Comprehensive tests for Tollama XAI modules

Tests all v3.8 XAI components across all 5 repositories.
"""

import os
import sys

# Add source paths — resolve to TollamaAI-Github root
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_REPO_ROOT, "tollama", "src"))

import numpy as np


def test_explanation_engine():
    """Test 1: ExplanationEngine — end-to-end decision explanation"""
    print("\n" + "=" * 70)
    print("TEST 1: ExplanationEngine — End-to-End Decision Explanation")
    print("=" * 70)

    from tollama.xai.decision_policy import DecisionPolicyExplainer
    from tollama.xai.engine import ExplanationEngine
    from tollama.xai.feature_attribution import TemporalFeatureAttribution
    from tollama.xai.forecast_decompose import ForecastDecomposer
    from tollama.xai.model_selection import ModelSelectionExplainer
    from tollama.xai.scenario_rationale import ScenarioRationale
    from tollama.xai.trust_breakdown import TrustBreakdown

    engine = ExplanationEngine(
        model_selection_explainer=ModelSelectionExplainer(primary_metric="mse"),
        forecast_decomposer=ForecastDecomposer(method="moving_average"),
        feature_attribution=TemporalFeatureAttribution(method="lag_correlation"),
        scenario_rationale=ScenarioRationale(),
        trust_breakdown=TrustBreakdown(),
        decision_policy_explainer=DecisionPolicyExplainer(),
    )

    # Simulate inputs
    forecast_result = {
        "confidence": 0.87,
        "forecast_value": 42.5,
        "signals_used": ["internal_ts", "polymarket"],
        "scenarios": {
            "branches": {
                "base": {"probability": 0.62, "value": 42.5, "conditions": ["stable macro"]},
                "upside": {"probability": 0.24, "value": 48.0, "conditions": ["growth acceleration"]},
                "downside": {"probability": 0.14, "value": 35.0, "conditions": ["recession"]},
            }
        },
    }

    eval_result = {
        "best_model": "chronos_large",
        "model_results": [
            {"model_name": "chronos_large", "metrics": {"mse": 0.12, "mae": 0.25, "mape": 8.5}},
            {"model_name": "timesfm", "metrics": {"mse": 0.15, "mae": 0.28, "mape": 9.2}},
            {"model_name": "moirai", "metrics": {"mse": 0.18, "mae": 0.32, "mape": 10.1}},
        ],
        "cv_config": {"strategy": "expanding-window", "n_splits": 5, "seed": 42},
        "version": "0.8.0",
        "dataset_info": {"n_rows": 365, "n_cols": 1, "frequency": "daily"},
    }

    calibration_result = {
        "trust_score": 0.81,
        "metrics": {"brier_score": 0.142, "log_loss": 0.318, "ece": 0.047},
        "signals": [{
            "name": "polymarket",
            "trust_score": 0.81,
            "metrics": {"brier_score": 0.142, "log_loss": 0.318, "ece": 0.047},
        }],
    }

    policy_config = {
        "auto_execute_threshold": 0.85,
        "audit_required": True,
    }

    # Generate time series data
    np.random.seed(42)
    ts_data = np.sin(np.linspace(0, 4 * np.pi, 100)) * 10 + np.random.randn(100) * 2

    result = engine.explain_decision(
        forecast_result=forecast_result,
        eval_result=eval_result,
        calibration_result=calibration_result,
        policy_config=policy_config,
        time_series_data=ts_data.tolist(),
        explain_options={"decompose": True, "attribution": False},
    )

    output = result.to_dict()
    print(f"\n  explanation_id: {output['explanation_id']}")
    print(f"  timestamp: {output['timestamp']}")

    # Input Explanation
    ie = output["input_explanation"]
    print(f"\n  [INPUT] Signals used: {ie['signals_used']}")
    print(f"  [INPUT] Trust scores: {ie['trust_scores']}")
    print(f"  [INPUT] Why trusted: {ie['why_trusted']}")

    # Plan Explanation
    pe = output["plan_explanation"]
    print(f"\n  [PLAN] Model selected: {pe['model_selected']}")
    print(f"  [PLAN] Why: {pe['why_this_model']}")
    print(f"  [PLAN] Ranking: {[r['model_name'] for r in pe['model_ranking']]}")
    print(f"  [PLAN] Decomposition: {pe['forecast_decomposition'].get('summary', 'N/A')[:80]}")

    # Decision Policy
    dpe = output["decision_policy_explanation"]
    print(f"\n  [POLICY] Auto-executed: {dpe['auto_executed']}")
    print(f"  [POLICY] Confidence: {dpe['confidence']}")
    print(f"  [POLICY] Reason: {dpe['reason']}")
    print(f"  [POLICY] Rules: {dpe['policy_rules_applied']}")

    print(f"\n  Metadata: {output['metadata']}")

    assert output["explanation_id"]
    assert ie["signals_used"] == ["internal_ts", "polymarket"]
    assert pe["model_selected"] == "chronos_large"
    assert dpe["auto_executed"] is True  # 0.87 >= 0.85
    print("\n  ✓ ExplanationEngine test PASSED")


def test_model_selection_explainer():
    """Test 2: ModelSelectionExplainer"""
    print("\n" + "=" * 70)
    print("TEST 2: ModelSelectionExplainer")
    print("=" * 70)

    from tollama.xai.model_selection import ModelSelectionExplainer

    explainer = ModelSelectionExplainer(primary_metric="brier_score")
    result = explainer.explain({
        "model_results": [
            {"model_name": "chronos", "metrics": {"brier_score": 0.12, "mae": 0.25}},
            {"model_name": "timesfm", "metrics": {"brier_score": 0.15, "mae": 0.22}},
            {"model_name": "moirai", "metrics": {"brier_score": 0.18, "mae": 0.30}},
        ],
        "cv_config": {"strategy": "expanding-window", "n_splits": 5},
        "best_model": "chronos",
    })

    print(f"\n  Selected: {result['model_selected']}")
    print(f"  Why: {result['why_this_model']}")
    print(f"  Rankings: {len(result['model_ranking'])} models")
    for r in result["model_ranking"]:
        print(f"    #{r['rank']} {r['model_name']}: strengths={r['strengths']}")

    assert result["model_selected"] == "chronos"
    assert "lowest" in result["why_this_model"].lower()
    print("\n  ✓ ModelSelectionExplainer test PASSED")


def test_forecast_decomposer():
    """Test 3: ForecastDecomposer"""
    print("\n" + "=" * 70)
    print("TEST 3: ForecastDecomposer")
    print("=" * 70)

    from tollama.xai.forecast_decompose import ForecastDecomposer

    decomposer = ForecastDecomposer(method="moving_average")

    # Generate seasonal data
    np.random.seed(42)
    t = np.arange(200)
    trend = 0.05 * t
    seasonal = 5 * np.sin(2 * np.pi * t / 12)
    noise = np.random.randn(200) * 0.5
    data = trend + seasonal + noise

    result = decomposer.decompose(data.tolist(), period=12)

    print(f"\n  Method: {result['method']}")
    print(f"  Period: {result['period']}")
    print(f"  Trend strength: {result['trend_strength']:.4f}")
    print(f"  Seasonal strength: {result['seasonal_strength']:.4f}")
    print(f"  Residual ratio: {result['residual_ratio']:.4f}")
    print(f"  Summary: {result['summary']}")

    assert result["method"] == "moving_average"
    assert result["trend_strength"] > 0
    print("\n  ✓ ForecastDecomposer test PASSED")


def test_trust_breakdown():
    """Test 4: TrustBreakdown"""
    print("\n" + "=" * 70)
    print("TEST 4: TrustBreakdown")
    print("=" * 70)

    from tollama.xai.trust_breakdown import TrustBreakdown

    tb = TrustBreakdown()
    result = tb.explain({
        "trust_score": 0.81,
        "metrics": {"brier_score": 0.142, "log_loss": 0.318, "ece": 0.047},
        "signals": [{
            "name": "polymarket",
            "trust_score": 0.81,
            "metrics": {"brier_score": 0.142, "log_loss": 0.318, "ece": 0.047},
        }],
    })

    print(f"\n  Trust scores: {result['trust_scores']}")
    print(f"  Why trusted: {result['why_trusted']}")
    print("  Breakdowns:")
    for comp, info in result["breakdowns"].get("polymarket", {}).items():
        if info.get("value") is not None:
            print(f"    {comp}: {info['value']} ({info.get('assessment', 'N/A')})")

    assert "polymarket" in result["trust_scores"]
    assert result["trust_scores"]["polymarket"] == 0.81
    print("\n  ✓ TrustBreakdown test PASSED")


def test_temporal_feature_attribution():
    """Test 5: TemporalFeatureAttribution"""
    print("\n" + "=" * 70)
    print("TEST 5: TemporalFeatureAttribution")
    print("=" * 70)

    from tollama.xai.feature_attribution import TemporalFeatureAttribution

    tfa = TemporalFeatureAttribution(method="lag_correlation")
    np.random.seed(42)
    data = np.sin(np.linspace(0, 6 * np.pi, 100)) + np.random.randn(100) * 0.3

    result = tfa.compute(data.tolist(), forecast_horizon=5)

    print(f"\n  Method: {result[0]['method']}")
    print(f"  Top contributing periods: {result[0]['top_contributing_periods'][:3]}")
    print(f"  Summary: {result[0]['summary']}")

    assert len(result) == 1
    assert result[0]["method"] == "lag_correlation"
    print("\n  ✓ TemporalFeatureAttribution test PASSED")


def test_decision_policy():
    """Test 6: DecisionPolicyExplainer"""
    print("\n" + "=" * 70)
    print("TEST 6: DecisionPolicyExplainer")
    print("=" * 70)

    from tollama.xai.decision_policy import DecisionPolicyExplainer

    dpe = DecisionPolicyExplainer()

    # Case 1: Auto-execute
    result1 = dpe.explain(
        {"confidence": 0.90},
        {"auto_execute_threshold": 0.85, "audit_required": True},
    )
    print("\n  Case 1 (confidence=0.90):")
    print(f"    Auto: {result1['auto_executed']}, Reason: {result1['reason']}")

    # Case 2: Needs approval
    result2 = dpe.explain(
        {"confidence": 0.70},
        {"auto_execute_threshold": 0.85, "audit_required": True},
    )
    print("  Case 2 (confidence=0.70):")
    print(f"    Auto: {result2['auto_executed']}, Escalation: {result2['escalation_triggered']}")

    assert result1["auto_executed"] is True
    assert result2["auto_executed"] is False
    assert result2["escalation_triggered"] is True
    print("\n  ✓ DecisionPolicyExplainer test PASSED")


def test_model_card_generator():
    """Test 7: ModelCardGenerator"""
    print("\n" + "=" * 70)
    print("TEST 7: ModelCardGenerator (EU AI Act)")
    print("=" * 70)

    from tollama.xai.model_card import ModelCardGenerator

    gen = ModelCardGenerator()
    card = gen.generate(
        model_info={
            "name": "Tollama Chronos Large",
            "version": "1.0.0",
            "type": "time_series_forecasting",
            "description": "Large-scale time series foundation model",
        },
        eval_result={
            "best_model": "chronos_large",
            "model_results": [
                {"model_name": "chronos_large", "metrics": {"mse": 0.12, "mape": 8.5}},
            ],
            "cv_config": {"strategy": "expanding-window", "n_splits": 5, "seed": 42},
            "version": "0.8.0",
            "dataset_info": {"n_rows": 365, "frequency": "daily"},
        },
    )

    print(f"\n  Card version: {card['model_card_version']}")
    print(f"  Model: {card['model_details']['name']}")
    classification = card["intended_use"]["eu_ai_act_classification"]
    print(f"  EU AI Act: {classification}")
    assert "risk" in classification.lower() or "AI Act" in classification
    print(f"  Explainability status: {card['explainability']['status']}")

    md = gen.to_markdown(card)
    print(f"\n  Markdown length: {len(md)} chars")
    print(f"  First 200 chars:\n{md[:200]}")

    assert card["model_details"]["name"] == "Tollama Chronos Large"
    print("\n  ✓ ModelCardGenerator test PASSED")


def test_report_builder():
    """Test 8: DecisionReportBuilder"""
    print("\n" + "=" * 70)
    print("TEST 8: DecisionReportBuilder")
    print("=" * 70)

    from tollama.xai.report_generator import DecisionReportBuilder

    builder = DecisionReportBuilder()

    explanation = {
        "explanation_id": "test-123",
        "version": "0.1.0",
        "input_explanation": {
            "signals_used": ["internal_ts", "polymarket"],
            "trust_scores": {"polymarket": 0.81},
            "why_trusted": {"polymarket": "High trust (0.81). Supported by: Brier Score 0.142."},
        },
        "plan_explanation": {
            "model_selected": "chronos_large",
            "why_this_model": "lowest MSE (0.12) across 3 models",
            "model_ranking": [
                {"model_name": "chronos_large", "rank": 1, "metrics": {"mse": 0.12}},
            ],
        },
        "decision_policy_explanation": {
            "auto_executed": True,
            "confidence": 0.87,
            "threshold": 0.85,
            "reason": "confidence 0.87 >= threshold 0.85",
            "policy_rules_applied": ["PASS: confidence 0.87 >= threshold 0.85"],
            "human_override": True,
        },
        "metadata": {"phase": "2a"},
    }

    report = builder.build_decision_report(explanation)
    md = builder.to_markdown(report)

    print(f"\n  Report type: {report['report_type']}")
    print(f"  Executive summary: {report['executive_summary'][:100]}...")
    print(f"  Markdown length: {len(md)} chars")

    assert report["report_type"] == "decision_report"
    assert "chronos_large" in report["model_selection"]["selected_model"]
    print("\n  ✓ DecisionReportBuilder test PASSED")


def test_tollama_eval_xai():
    """Test 9: tollama-eval XAI integration"""
    print("\n" + "=" * 70)
    print("TEST 9: tollama-eval XAI Integration")
    print("=" * 70)

    eval_path = os.path.join(_REPO_ROOT, "tollama-eval", "src", "ts_autopilot", "reporting")
    sys.path.insert(0, eval_path)
    if "xai_integration" in sys.modules:
        del sys.modules["xai_integration"]
    from xai_integration import EvalExplanationExtender

    extender = EvalExplanationExtender()

    eval_result = {
        "best_model": "chronos_large",
        "model_results": [
            {"model_name": "chronos_large", "metrics": {"mse": 0.12, "mae": 0.25}},
            {"model_name": "timesfm", "metrics": {"mse": 0.15, "mae": 0.22}},
        ],
        "cv_config": {"strategy": "expanding-window", "n_splits": 5, "seed": 42},
        "version": "0.8.0",
    }

    extended = extender.extend_eval_result(eval_result, primary_metric="mse")

    xai = extended["xai_explanation"]
    print(f"\n  XAI version: {xai['version']}")
    print(f"  Model selection: {xai['model_selection']['rationale']}")
    print(f"  Narrative: {xai['narrative'][:100]}...")
    print(f"  Reproducibility hash: {xai['reproducibility']['config_hash']}")

    assert "chronos_large" in xai["model_selection"]["selected_model"]
    assert xai["reproducibility"]["config_hash"]
    print("\n  ✓ tollama-eval XAI Integration test PASSED")


def test_market_calibration_xai():
    """Test 10: Market Calibration Agent XAI"""
    print("\n" + "=" * 70)
    print("TEST 10: Market Calibration Agent XAI")
    print("=" * 70)

    mc_path = os.path.join(_REPO_ROOT, "Market-Calibration-Agent", "calibration")
    sys.path.insert(0, mc_path)
    if "xai_integration" in sys.modules:
        del sys.modules["xai_integration"]
    import xai_integration as mca_xai

    explainer = mca_xai.TrustScoreExplainer()

    result = explainer.explain_trust_score({
        "market_id": "0x1a2b",
        "current_probability": 0.74,
        "trust_score": 0.81,
        "metrics": {"brier_score": 0.142, "log_loss": 0.318, "ece": 0.047},
        "tsfm_signal": {"forecast": [0.72, 0.75, 0.71]},
    })

    print(f"\n  Trust score: {result['trust_score']}")
    print(f"  Trust level: {result['trust_level']}")
    print(f"  Explanation: {result['explanation'][:100]}...")
    print(f"  Recommendation: {result['recommendation']}")

    # Test drift detection
    drift = explainer.explain_calibration_drift([
        {"trust_score": 0.85, "timestamp": "2026-01-01"},
        {"trust_score": 0.83, "timestamp": "2026-01-15"},
        {"trust_score": 0.80, "timestamp": "2026-02-01"},
        {"trust_score": 0.75, "timestamp": "2026-02-15"},
        {"trust_score": 0.70, "timestamp": "2026-03-01"},
    ])
    print(f"\n  Drift detected: {drift['drift_detected']}")
    print(f"  Drift magnitude: {drift['drift_magnitude']}")
    print(f"  Direction: {drift['direction']}")

    assert result["trust_level"] == "high"
    print("\n  ✓ Market Calibration Agent XAI test PASSED")


def test_spline_lstm_xai():
    """Test 11: Spline-LSTM XAI"""
    print("\n" + "=" * 70)
    print("TEST 11: Spline-LSTM XAI")
    print("=" * 70)

    sl_path = os.path.join(_REPO_ROOT, "spline-lstm", "backend", "app")
    sys.path.insert(0, sl_path)
    if "xai_integration" in sys.modules:
        del sys.modules["xai_integration"]
    import xai_integration as sl_xai

    explainer = sl_xai.SplineLSTMExplainer()

    np.random.seed(42)
    raw = np.sin(np.linspace(0, 4 * np.pi, 100)) * 10 + np.random.randn(100) * 3
    smooth = np.sin(np.linspace(0, 4 * np.pi, 100)) * 10  # Idealized smooth

    result = explainer.explain_forecast(
        raw_data=raw.tolist(),
        spline_output=smooth.tolist(),
        n_spline_knots=10,
    )

    print(f"\n  Model type: {result['model_type']}")
    print(f"  Deployment: {result['deployment']}")
    print(f"  Spline noise reduction: {result['spline_analysis'].get('noise_reduction_pct')}%")
    print(f"  Top lag: {result['temporal_importance'].get('most_important_lag')}")
    print(f"  Dominant: {result['forecast_decomposition'].get('dominant_component')}")
    print(f"  Summary: {result['summary']}")

    assert result["model_type"] == "spline_lstm"
    print("\n  ✓ Spline-LSTM XAI test PASSED")


def test_coding_agent_xai():
    """Test 12: Coding Agent Decision Tracer"""
    print("\n" + "=" * 70)
    print("TEST 12: Coding Agent Decision Tracer")
    print("=" * 70)

    ca_path = os.path.join(_REPO_ROOT, "coding-agent", "autodev")
    sys.path.insert(0, ca_path)
    if "xai_integration" in sys.modules:
        del sys.modules["xai_integration"]
    import xai_integration as ca_xai

    tracer = ca_xai.AgentDecisionTracer()

    session_id = tracer.start_session("Build a REST API for user management")
    print(f"\n  Session: {session_id}")

    tracer.record_parsing_decision(
        prd_sections=["Overview", "Requirements", "API Spec"],
        requirements_extracted=12,
    )

    tracer.record_planning_decision(
        architecture="FastAPI + PostgreSQL",
        components=["api", "models", "auth", "tests"],
        tech_stack={"backend": "FastAPI", "db": "PostgreSQL", "auth": "JWT"},
        alternatives_rejected=[
            {"option": "Django", "reason": "Heavier than needed for API-only service"},
        ],
    )

    tracer.record_code_generation_decision(
        file_path="src/api/routes.py",
        language="python",
        lines_generated=150,
        patterns_used=["RESTful", "dependency_injection"],
    )

    tracer.record_validation_decision(
        validator="ruff",
        passed=True,
        issues_found=0,
    )

    tracer.record_validation_decision(
        validator="pytest",
        passed=False,
        issues_found=3,
        severity_summary={"error": 2, "warning": 1},
    )

    tracer.record_autofix_decision(
        issue="Missing return type annotation",
        fix_applied="Added -> dict return type to 3 endpoints",
        fix_attempt=1,
    )

    trace = tracer.get_trace()
    print(f"  Total decisions: {trace['n_decisions']}")
    print(f"  Summary: {trace['summary']}")

    audit = tracer.get_audit_report()
    print("\n  Audit report:")
    print(f"    Human review needed: {audit['audit_report']['human_review_needed']}")
    print(f"    Low confidence: {len(audit['audit_report']['low_confidence_decisions'])}")

    assert trace["n_decisions"] == 7  # Including session_start
    print("\n  ✓ Coding Agent Decision Tracer test PASSED")


def main():
    print("=" * 70)
    print("  TOLLAMA XAI v3.8 — COMPREHENSIVE TEST SUITE")
    print("  Testing all XAI modules across 5 repositories")
    print("=" * 70)

    tests = [
        test_explanation_engine,
        test_model_selection_explainer,
        test_forecast_decomposer,
        test_trust_breakdown,
        test_temporal_feature_attribution,
        test_decision_policy,
        test_model_card_generator,
        test_report_builder,
        test_tollama_eval_xai,
        test_market_calibration_xai,
        test_spline_lstm_xai,
        test_coding_agent_xai,
    ]

    passed = 0
    failed = 0
    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"\n  ✗ FAILED: {test_fn.__name__}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 70)
    print(f"  RESULTS: {passed}/{len(tests)} passed, {failed} failed")
    print("=" * 70)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    exit(main())
