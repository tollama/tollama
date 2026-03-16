"""
tests/test_xai_trust_integration.py — Integration tests for trust-intelligence
wiring into the XAI decisioning layer.

Validates that:
- Trust scores block low-trust decisions
- Constraint violations prevent auto-execution
- Risk category RED triggers escalation
- SHAP receives real predict_fn when provided
- The API endpoint surfaces trust-driven explanations
- Backward compatibility is preserved when trust is absent
"""

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

# Add source paths
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_REPO_ROOT, "tollama", "src"))


# ──────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────

@pytest.fixture
def high_confidence_forecast():
    return {"confidence": 0.95, "model": "test_model", "forecast_value": 100}


@pytest.fixture
def default_policy():
    return {"auto_execute_threshold": 0.85, "audit_required": True}


@pytest.fixture
def trust_result_low_trust():
    """Trust result with low trust score — should block."""
    return {
        "trust_intelligence": {
            "trust_score": 0.3,
            "calibration_status": "overconfident",
            "components": {
                "uncertainty": 0.8,
                "coverage_tightness": 0.5,
                "shap_stability": 0.6,
                "risk_category": "YELLOW",
                "constraint_satisfied": True,
            },
            "violations": [],
            "meta_metrics": {"ece": 0.15, "ocr": 0.3},
        }
    }


@pytest.fixture
def trust_result_high_trust():
    """Trust result with high trust score — should pass."""
    return {
        "trust_intelligence": {
            "trust_score": 0.85,
            "calibration_status": "well_calibrated",
            "components": {
                "uncertainty": 0.2,
                "coverage_tightness": 0.9,
                "shap_stability": 0.95,
                "risk_category": "GREEN",
                "constraint_satisfied": True,
            },
            "violations": [],
            "meta_metrics": {"ece": 0.02, "ocr": 0.05},
        }
    }


@pytest.fixture
def trust_result_critical_violations():
    """Trust result with critical constraint violations."""
    return {
        "trust_intelligence": {
            "trust_score": 0.8,
            "calibration_status": "well_calibrated",
            "components": {
                "uncertainty": 0.2,
                "coverage_tightness": 0.9,
                "shap_stability": 0.95,
                "risk_category": "RED",
                "constraint_satisfied": False,
            },
            "violations": [
                {
                    "name": "risk_threshold",
                    "type": "Risk Threshold",
                    "severity": "critical",
                },
                {
                    "name": "liquidity_minimum",
                    "type": "Liquidity Min",
                    "severity": "warning",
                },
            ],
            "meta_metrics": {"ece": 0.02, "ocr": 0.05},
        }
    }


@pytest.fixture
def trust_result_red_risk():
    """Trust result with RED risk category but passing trust score."""
    return {
        "trust_intelligence": {
            "trust_score": 0.75,
            "calibration_status": "well_calibrated",
            "components": {
                "uncertainty": 0.3,
                "coverage_tightness": 0.8,
                "shap_stability": 0.9,
                "risk_category": "RED",
                "constraint_satisfied": False,
            },
            "violations": [
                {
                    "name": "time_horizon",
                    "type": "Time Horizon",
                    "severity": "critical",
                },
            ],
            "meta_metrics": {"ece": 0.05, "ocr": 0.1},
        }
    }


# ──────────────────────────────────────────────────────────────
# Decision Policy Trust Gating
# ──────────────────────────────────────────────────────────────

class TestDecisionPolicyTrustGating:
    """Tests for trust gates in DecisionPolicyExplainer."""

    def test_trust_score_blocks_low_trust_decision(
        self, high_confidence_forecast, default_policy, trust_result_low_trust
    ):
        """Gate A: High confidence + low trust = blocked."""
        from tollama.xai.decision_policy import DecisionPolicyExplainer

        explainer = DecisionPolicyExplainer()
        result = explainer.explain(
            high_confidence_forecast, default_policy, trust_result=trust_result_low_trust
        )

        assert result["auto_executed"] is False
        assert result["escalation_triggered"] is True
        assert result["trust_blocked"] is True
        assert result["trust_score"] == 0.3
        assert any("BLOCK" in r and "trust_score" in r for r in result["policy_rules_applied"])

    def test_constraint_violations_block_execution(
        self, high_confidence_forecast, default_policy, trust_result_critical_violations
    ):
        """Gate B: Critical constraint violations block auto-execution."""
        from tollama.xai.decision_policy import DecisionPolicyExplainer

        explainer = DecisionPolicyExplainer()
        result = explainer.explain(
            high_confidence_forecast, default_policy,
            trust_result=trust_result_critical_violations,
        )

        assert result["auto_executed"] is False
        assert result["trust_blocked"] is True
        assert result["constraint_violations_count"] == 1  # only critical ones
        assert any("constraint violation" in r for r in result["policy_rules_applied"])

    def test_risk_category_red_escalates(
        self, high_confidence_forecast, default_policy, trust_result_red_risk
    ):
        """Gate C: Risk category RED triggers escalation."""
        from tollama.xai.decision_policy import DecisionPolicyExplainer

        explainer = DecisionPolicyExplainer()
        result = explainer.explain(
            high_confidence_forecast, default_policy,
            trust_result=trust_result_red_risk,
        )

        assert result["auto_executed"] is False
        assert result["escalation_triggered"] is True
        assert result["risk_category"] == "RED"
        assert any("BLOCK: risk_category RED" in r for r in result["policy_rules_applied"])

    def test_high_trust_allows_execution(
        self, high_confidence_forecast, default_policy, trust_result_high_trust
    ):
        """High confidence + high trust = allowed."""
        from tollama.xai.decision_policy import DecisionPolicyExplainer

        explainer = DecisionPolicyExplainer()
        result = explainer.explain(
            high_confidence_forecast, default_policy,
            trust_result=trust_result_high_trust,
        )

        assert result["auto_executed"] is True
        assert result["trust_blocked"] is False
        assert result["trust_score"] == 0.85
        assert any("PASS" in r and "trust_score" in r for r in result["policy_rules_applied"])

    def test_no_trust_result_backward_compatible(
        self, high_confidence_forecast, default_policy
    ):
        """Without trust_result, behavior is identical to pre-trust code."""
        from tollama.xai.decision_policy import DecisionPolicyExplainer

        explainer = DecisionPolicyExplainer()
        result = explainer.explain(high_confidence_forecast, default_policy)

        # confidence 0.95 >= 0.85 threshold → auto-executed
        assert result["auto_executed"] is True
        assert result["trust_blocked"] is False
        assert result["trust_score"] is None
        assert result["risk_category"] is None
        assert result["constraint_violations_count"] == 0

    def test_yellow_risk_warns_but_does_not_block(
        self, high_confidence_forecast, default_policy
    ):
        """YELLOW risk category warns but does not block by itself."""
        from tollama.xai.decision_policy import DecisionPolicyExplainer

        trust_result = {
            "trust_intelligence": {
                "trust_score": 0.8,
                "components": {"risk_category": "YELLOW", "constraint_satisfied": True},
                "violations": [],
            }
        }

        explainer = DecisionPolicyExplainer()
        result = explainer.explain(
            high_confidence_forecast, default_policy, trust_result=trust_result,
        )

        assert result["auto_executed"] is True
        assert any("WARN: risk_category YELLOW" in r for r in result["policy_rules_applied"])


# ──────────────────────────────────────────────────────────────
# SHAP predict_fn Threading
# ──────────────────────────────────────────────────────────────

class TestSHAPPredictFn:
    """Tests for predict_fn threading through the bridge."""

    def test_trust_bridge_passes_predict_fn(self):
        """run_trust_pipeline forwards predict_fn to pipeline.run()."""
        from tollama.xai.trust_intelligence_bridge import run_trust_pipeline

        mock_pipeline = MagicMock()
        mock_pipeline.run.return_value = MagicMock(
            pipeline_version="3.0",
            trust=MagicMock(
                trust_score=0.8, calibration_status="well_calibrated",
                weights={}, ece=0.0, ocr=0.0,
            ),
            uncertainty=MagicMock(normalized_uncertainty=0.2),
            conformal=MagicMock(coverage_tightness=0.9, coverage_validity=True),
            shap=MagicMock(
                shap_stability=0.95, feature_contributions=[],
            ),
            constraints=MagicMock(
                risk_category=MagicMock(value="GREEN"),
                constraint_satisfied=True, violations=[],
            ),
        )

        def my_predict_fn(x):
            return x

        with patch(
            "tollama.xai.trust_intelligence_bridge.HAS_TRUST_INTELLIGENCE", True
        ):
            run_trust_pipeline(
                mock_pipeline,
                prediction_probability=0.8,
                features={"volume_change_24h": 0.5},
                predict_fn=my_predict_fn,
            )

        mock_pipeline.run.assert_called_once()
        call_kwargs = mock_pipeline.run.call_args[1]
        assert call_kwargs["predict_fn"] is my_predict_fn

    def test_trust_bridge_without_predict_fn(self):
        """run_trust_pipeline works without predict_fn (backward-compatible)."""
        from tollama.xai.trust_intelligence_bridge import run_trust_pipeline

        mock_pipeline = MagicMock()
        mock_pipeline.run.return_value = MagicMock(
            pipeline_version="3.0",
            trust=MagicMock(
                trust_score=0.8, calibration_status="well_calibrated",
                weights={}, ece=0.0, ocr=0.0,
            ),
            uncertainty=MagicMock(normalized_uncertainty=0.2),
            conformal=MagicMock(coverage_tightness=0.9, coverage_validity=True),
            shap=MagicMock(
                shap_stability=0.95, feature_contributions=[],
            ),
            constraints=MagicMock(
                risk_category=MagicMock(value="GREEN"),
                constraint_satisfied=True, violations=[],
            ),
        )

        with patch(
            "tollama.xai.trust_intelligence_bridge.HAS_TRUST_INTELLIGENCE", True
        ):
            result = run_trust_pipeline(
                mock_pipeline,
                prediction_probability=0.8,
            )

        assert result is not None
        call_kwargs = mock_pipeline.run.call_args[1]
        assert call_kwargs["predict_fn"] is None


# ──────────────────────────────────────────────────────────────
# Engine Integration
# ──────────────────────────────────────────────────────────────

class TestEngineIntegration:
    """Tests for ExplanationEngine with trust pipeline integration."""

    def test_engine_without_trust_pipeline_no_trust_explanation(self):
        """Engine without trust pipeline → trust_intelligence_explanation is None."""
        from tollama.xai.engine import ExplanationEngine

        engine = ExplanationEngine()
        result = engine.explain_decision(
            forecast_result={"confidence": 0.9},
        )

        assert result.trust_intelligence_explanation is None
        assert "trust_intelligence" not in result.metadata

    def test_engine_with_trust_pipeline_produces_trust_explanation(self):
        """Engine with trust pipeline → trust_intelligence_explanation populated."""
        from tollama.xai.decision_policy import DecisionPolicyExplainer
        from tollama.xai.engine import ExplanationEngine

        mock_pipeline = MagicMock()
        mock_run_result = MagicMock(
            pipeline_version="3.0",
            trust=MagicMock(
                trust_score=0.82, calibration_status="well_calibrated",
                weights={"w1": 0.25}, ece=0.02, ocr=0.05,
            ),
            uncertainty=MagicMock(normalized_uncertainty=0.3),
            conformal=MagicMock(coverage_tightness=0.85, coverage_validity=True),
            shap=MagicMock(
                shap_stability=0.92,
                feature_contributions=[
                    MagicMock(feature_name="volume", shap_value=0.15, rank=1, direction="positive"),
                ],
            ),
            constraints=MagicMock(
                risk_category=MagicMock(value="GREEN"),
                constraint_satisfied=True, violations=[],
            ),
        )
        mock_pipeline.run.return_value = mock_run_result

        engine = ExplanationEngine(
            decision_policy_explainer=DecisionPolicyExplainer(),
            trust_intelligence_pipeline=mock_pipeline,
        )

        with patch(
            "tollama.xai.trust_intelligence_bridge.HAS_TRUST_INTELLIGENCE", True
        ):
            result = engine.explain_decision(
                forecast_result={"confidence": 0.9},
                policy_config={"auto_execute_threshold": 0.85},
                explain_options={
                    "trust_intelligence_features": {"volume_change_24h": 0.5},
                },
            )

        assert result.trust_intelligence_explanation is not None
        assert result.trust_intelligence_explanation["trust_score"] == 0.82
        assert result.trust_intelligence_explanation["risk_category"] == "GREEN"
        assert result.trust_intelligence_explanation["constraint_satisfied"] is True
        assert len(result.trust_intelligence_explanation["top_features"]) == 1

        # Also in metadata for backward compat
        assert "trust_intelligence" in result.metadata

        # Decision policy should have trust fields
        dpe = result.decision_policy_explanation
        assert dpe.trust_score == 0.82
        assert dpe.trust_blocked is False

    def test_engine_trust_blocks_auto_execution(self):
        """Engine with low-trust pipeline result blocks auto-execution."""
        from tollama.xai.decision_policy import DecisionPolicyExplainer
        from tollama.xai.engine import ExplanationEngine

        mock_pipeline = MagicMock()
        mock_run_result = MagicMock(
            pipeline_version="3.0",
            trust=MagicMock(
                trust_score=0.3, calibration_status="overconfident",
                weights={}, ece=0.2, ocr=0.4,
            ),
            uncertainty=MagicMock(normalized_uncertainty=0.8),
            conformal=MagicMock(coverage_tightness=0.4, coverage_validity=False),
            shap=MagicMock(shap_stability=0.5, feature_contributions=[]),
            constraints=MagicMock(
                risk_category=MagicMock(value="YELLOW"),
                constraint_satisfied=True, violations=[],
            ),
        )
        mock_pipeline.run.return_value = mock_run_result

        engine = ExplanationEngine(
            decision_policy_explainer=DecisionPolicyExplainer(),
            trust_intelligence_pipeline=mock_pipeline,
        )

        with patch(
            "tollama.xai.trust_intelligence_bridge.HAS_TRUST_INTELLIGENCE", True
        ):
            result = engine.explain_decision(
                forecast_result={"confidence": 0.95},
                policy_config={"auto_execute_threshold": 0.85},
            )

        dpe = result.decision_policy_explanation
        assert dpe.auto_executed is False
        assert dpe.trust_blocked is True
        assert dpe.trust_score == 0.3

    def test_engine_predict_fn_forwarded(self):
        """Engine forwards predict_fn to trust pipeline."""
        from tollama.xai.engine import ExplanationEngine

        mock_pipeline = MagicMock()
        mock_pipeline.run.return_value = MagicMock(
            pipeline_version="3.0",
            trust=MagicMock(
                trust_score=0.8, calibration_status="well_calibrated",
                weights={}, ece=0.0, ocr=0.0,
            ),
            uncertainty=MagicMock(normalized_uncertainty=0.2),
            conformal=MagicMock(coverage_tightness=0.9, coverage_validity=True),
            shap=MagicMock(shap_stability=0.95, feature_contributions=[]),
            constraints=MagicMock(
                risk_category=MagicMock(value="GREEN"),
                constraint_satisfied=True, violations=[],
            ),
        )

        def my_model(x):
            return x * 2

        engine = ExplanationEngine(trust_intelligence_pipeline=mock_pipeline)

        with patch(
            "tollama.xai.trust_intelligence_bridge.HAS_TRUST_INTELLIGENCE", True
        ):
            engine.explain_decision(
                forecast_result={"confidence": 0.9},
                predict_fn=my_model,
            )

        call_kwargs = mock_pipeline.run.call_args[1]
        assert call_kwargs["predict_fn"] is my_model
