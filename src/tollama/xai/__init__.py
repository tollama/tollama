"""
tollama.xai — Explainable AI module for Tollama Core Runtime

v3.8 Roadmap: Phase 2a Explanation Facade
기존 eval·calibration·policy output을 decision-ready evidence package로 조립하는
explanation interface layer.

Components:
  - explain_decision:   End-to-end decision explanation (Phase 4 target, facade now)
  - model_selection:    Model Selection Explanation from eval CV results
  - forecast_decompose: Trend/Seasonal/Residual decomposition
  - feature_attribution: Temporal Feature Importance
  - scenario_rationale:  Scenario tree + counterfactual rationale
  - trust_breakdown:     Trust Score breakdown packaging
  - decision_policy:     Decision policy explanation (confidence/threshold)
  - model_card:          EU AI Act Model Card auto-generation
  - report_generator:    Decision Report + Explanation Report builder
"""

__version__ = "0.1.0"
__all__ = [
    "ExplanationEngine",
    "ModelSelectionExplainer",
    "ForecastDecomposer",
    "TemporalFeatureAttribution",
    "ScenarioRationale",
    "TrustBreakdown",
    "DecisionPolicyExplainer",
    "FinancialTrustPayload",
    "NewsTrustPayload",
    "NormalizedTrustResult",
    "TrustAgentRegistry",
    "TrustRouter",
    "ModelCardGenerator",
    "DecisionReportBuilder",
]

try:
    from tollama.xai.decision_policy import DecisionPolicyExplainer
    from tollama.xai.engine import ExplanationEngine
    from tollama.xai.feature_attribution import TemporalFeatureAttribution
    from tollama.xai.forecast_decompose import ForecastDecomposer
    from tollama.xai.model_card import ModelCardGenerator
    from tollama.xai.model_selection import ModelSelectionExplainer
    from tollama.xai.report_generator import DecisionReportBuilder
    from tollama.xai.scenario_rationale import ScenarioRationale
    from tollama.xai.trust_breakdown import TrustBreakdown
    from tollama.xai.trust_contract import (
        FinancialTrustPayload,
        NewsTrustPayload,
        NormalizedTrustResult,
    )
    from tollama.xai.trust_router import TrustAgentRegistry, TrustRouter
except ImportError:
    from xai.decision_policy import DecisionPolicyExplainer
    from xai.engine import ExplanationEngine
    from xai.feature_attribution import TemporalFeatureAttribution
    from xai.forecast_decompose import ForecastDecomposer
    from xai.model_card import ModelCardGenerator
    from xai.model_selection import ModelSelectionExplainer
    from xai.report_generator import DecisionReportBuilder
    from xai.scenario_rationale import ScenarioRationale
    from xai.trust_breakdown import TrustBreakdown
    from xai.trust_contract import (
        FinancialTrustPayload,
        NewsTrustPayload,
        NormalizedTrustResult,
    )
    from xai.trust_router import TrustAgentRegistry, TrustRouter
