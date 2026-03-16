"""
tollama.xai.engine — Central Explanation Engine

v3.8: Trust-integrated XAI engine.

Phase 2b (현재): Trust-Gated Decisioning
  - Trust Intelligence Pipeline (L1-L5) runs before decisioning
  - Trust score, constraint violations, and risk category gate auto-execution
  - SHAP attributions use real model predict_fn when available
  - Trust results surfaced as first-class explanation section

Phase 3+: Feature Attribution Layer 완성, richer decomposition
Phase 4: /api/explain-decision 정식 출시
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any

from tollama.xai.trust_contract import (
    coerce_normalized_trust_result,
    normalized_result_to_legacy_metadata,
)

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────

@dataclass
class InputExplanation:
    """Input 단계 설명: 왜 이 신호를 반영했는가"""
    signals_used: list[str] = field(default_factory=list)
    trust_scores: dict[str, float] = field(default_factory=dict)
    trust_breakdowns: dict[str, dict[str, Any]] = field(default_factory=dict)
    why_trusted: dict[str, str] = field(default_factory=dict)
    data_quality: dict[str, Any] = field(default_factory=dict)


@dataclass
class PlanExplanation:
    """Plan 단계 설명: 왜 이 모델과 시나리오를 추천했는가"""
    model_selected: str = ""
    why_this_model: str = ""
    model_ranking: list[dict[str, Any]] = field(default_factory=list)
    eval_evidence: dict[str, Any] = field(default_factory=dict)
    forecast_decomposition: dict[str, float] = field(default_factory=dict)
    scenarios: dict[str, Any] = field(default_factory=dict)
    feature_importance: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class DecisionPolicyExplanation:
    """Decision Policy 단계 설명: 왜 이 승인/자동화 정책이 나왔는가"""
    auto_executed: bool = False
    confidence: float = 0.0
    threshold: float = 0.0
    reason: str = ""
    human_override: bool = True
    policy_rules_applied: list[str] = field(default_factory=list)
    escalation_triggered: bool = False
    escalation_reason: str = ""
    trust_score: float | None = None
    risk_category: str | None = None
    trust_blocked: bool = False
    constraint_violations_count: int = 0


@dataclass
class DecisionExplanation:
    """End-to-end Decision Explanation — /api/explain-decision 응답 구조"""
    explanation_id: str = ""
    timestamp: str = ""
    version: str = "0.1.0"
    input_explanation: InputExplanation = field(default_factory=InputExplanation)
    plan_explanation: PlanExplanation = field(default_factory=PlanExplanation)
    decision_policy_explanation: DecisionPolicyExplanation = field(
        default_factory=DecisionPolicyExplanation
    )
    trust_intelligence_explanation: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ──────────────────────────────────────────────────────────────
# Explanation Engine
# ──────────────────────────────────────────────────────────────

class ExplanationEngine:
    """
    Central orchestrator for Tollama XAI.

    Phase 2b (현재): Trust-Gated Decisioning
    - Trust Intelligence Pipeline (L1-L5) 통합
    - Trust score, constraint violations, risk category가
      auto-execution 결정을 제어
    - SHAP attributions에 실제 모델 predict_fn 전달 가능
    - Existing eval/calibration/policy evidence를
      trust-aware decision-ready format으로 조립

    Phase 3+: Feature Attribution Layer
    - Temporal Feature Importance
    - Forecast Decomposition (trend/seasonal/residual)
    - /api/forecast?explain=true

    Phase 4: /api/explain-decision 정식 출시
    - 외부 에이전트 프레임워크(LangChain, CrewAI, MCP)에서 호출 가능
    """

    def __init__(
        self,
        model_selection_explainer=None,
        forecast_decomposer=None,
        feature_attribution=None,
        scenario_rationale=None,
        trust_breakdown=None,
        decision_policy_explainer=None,
        trust_intelligence_pipeline=None,
        trust_router=None,
    ):
        self.model_selection = model_selection_explainer
        self.forecast_decomposer = forecast_decomposer
        self.feature_attribution = feature_attribution
        self.scenario_rationale = scenario_rationale
        self.trust_breakdown = trust_breakdown
        self.decision_policy = decision_policy_explainer
        self.trust_intelligence = trust_intelligence_pipeline
        self.trust_router = trust_router

    def explain_decision(
        self,
        forecast_result: dict[str, Any],
        eval_result: dict[str, Any] | None = None,
        calibration_result: dict[str, Any] | None = None,
        trust_result: dict[str, Any] | None = None,
        policy_config: dict[str, Any] | None = None,
        time_series_data: Any | None = None,
        explain_options: dict[str, Any] | None = None,
        trust_context: dict[str, Any] | None = None,
        trust_payload: dict[str, Any] | None = None,
        predict_fn: Callable | None = None,
    ) -> DecisionExplanation:
        """
        End-to-end decision explanation.

        Assembles evidence from eval, calibration, and policy layers
        into a unified explanation.

        Parameters
        ----------
        forecast_result : dict
            Output from tollama forecast endpoint (e.g., /api/forecast, /api/auto-forecast)
        eval_result : dict, optional
            Output from tollama-eval (CV results, model rankings, metrics)
        calibration_result : dict, optional
            Output from Market Calibration Agent (trust scores, calibration metrics)
        trust_result : dict, optional
            Normalized trust result from a domain trust agent.
        policy_config : dict, optional
            Decision policy configuration (thresholds, rules, approval config)
        time_series_data : array-like, optional
            Raw time series for decomposition and feature attribution
        explain_options : dict, optional
            Control explanation depth: {"decompose": True, "attribution": True, ...}
        predict_fn : callable, optional
            Model prediction function for faithful SHAP attribution.
            Signature: (features_array: np.ndarray) -> np.ndarray.
            When provided, SHAP computes attributions using the actual model
            instead of a surrogate.

        Returns
        -------
        DecisionExplanation
            Structured explanation covering Input → Plan → Decision Policy
        """
        import uuid

        options = explain_options or {}
        resolved_trust_result = self._resolve_trust_result(
            calibration_result=calibration_result,
            trust_result=trust_result,
            trust_context=trust_context,
            trust_payload=trust_payload,
            explain_options=options,
        )
        explanation = DecisionExplanation(
            explanation_id=str(uuid.uuid4()),
            timestamp=datetime.now(UTC).isoformat(),
        )

        # ── Phase 1: Input Explanation (Signal Trust) ──
        explanation.input_explanation = self._explain_input(
            forecast_result=forecast_result,
            calibration_result=calibration_result,
            trust_result=resolved_trust_result.model_dump() if resolved_trust_result else None,
        )

        # ── Phase 2: Plan Explanation (Model Selection + Decomposition) ──
        explanation.plan_explanation = self._explain_plan(
            forecast_result=forecast_result,
            eval_result=eval_result,
            time_series_data=time_series_data,
            include_decomposition=options.get("decompose", True),
            include_attribution=options.get("attribution", False),
        )

        # ── Trust Intelligence Pipeline (before decisioning) ──
        ti_metadata = (
            normalized_result_to_legacy_metadata(resolved_trust_result)
            if resolved_trust_result is not None
            else None
        )
        if ti_metadata is None and self.trust_intelligence is not None:
            from tollama.xai.trust_intelligence_bridge import run_trust_pipeline

            prediction_prob = forecast_result.get("confidence", 0.5)
            ti_features = options.get("trust_intelligence_features")
            ti_context = options.get("trust_intelligence_context")

            ti_metadata = run_trust_pipeline(
                self.trust_intelligence,
                prediction_probability=prediction_prob,
                features=ti_features,
                context=ti_context,
                predict_fn=predict_fn,
            )

        # ── Decision Policy Explanation (trust-aware) ──
        explanation.decision_policy_explanation = self._explain_decision_policy(
            forecast_result=forecast_result,
            policy_config=policy_config,
            trust_metadata=ti_metadata,
        )

        # ── Trust Intelligence Explanation (first-class section) ──
        if ti_metadata and "trust_intelligence" in ti_metadata:
            ti = ti_metadata["trust_intelligence"]
            explanation.trust_intelligence_explanation = {
                "trust_score": ti.get("trust_score"),
                "calibration_status": ti.get("calibration_status"),
                "risk_category": ti.get("components", {}).get("risk_category"),
                "constraint_satisfied": ti.get("components", {}).get("constraint_satisfied"),
                "top_features": ti.get("shap_top_features", []),
                "violations": ti.get("violations", []),
                "meta_metrics": ti.get("meta_metrics", {}),
            }

        has_trust = ti_metadata is not None and "trust_intelligence" in (ti_metadata or {})
        explanation.metadata = {
            "engine_version": "0.2.0",
            "phase": "2b" if has_trust else "2a",
            "explanation_type": "trust_gated" if has_trust else "facade",
            "note": (
                "Phase 2b: trust-gated decisioning with L1-L5 pipeline integration. "
                "Trust score, constraint violations, and risk category control "
                "auto-execution decisions."
            ) if has_trust else (
                "Phase 2a explanation is evidence repackaging, "
                "not deep interpretability. Install trust-intelligence "
                "for trust-gated decisioning."
            ),
        }

        # Backward-compatible: trust results also in metadata
        if ti_metadata:
            explanation.metadata.update(ti_metadata)

        logger.info(
            "Generated decision explanation %s", explanation.explanation_id
        )
        return explanation

    def _explain_input(
        self,
        forecast_result: dict[str, Any],
        calibration_result: dict[str, Any] | None,
        trust_result: dict[str, Any] | None = None,
    ) -> InputExplanation:
        """Assemble input-stage explanation: why these signals were used."""
        ie = InputExplanation()

        # Identify signals used
        ie.signals_used = forecast_result.get("signals_used", ["internal_ts"])

        # Trust Score breakdown from calibration
        if trust_result and self.trust_breakdown:
            breakdown = self.trust_breakdown.explain(trust_result)
            ie.trust_scores = breakdown.get("trust_scores", {})
            ie.trust_breakdowns = breakdown.get("breakdowns", {})
            ie.why_trusted = breakdown.get("why_trusted", {})
        elif calibration_result and self.trust_breakdown:
            breakdown = self.trust_breakdown.explain(calibration_result)
            ie.trust_scores = breakdown.get("trust_scores", {})
            ie.trust_breakdowns = breakdown.get("breakdowns", {})
            ie.why_trusted = breakdown.get("why_trusted", {})
        elif calibration_result:
            # Fallback: direct passthrough
            ie.trust_scores = calibration_result.get("trust_scores", {})
            for signal, score in ie.trust_scores.items():
                ie.why_trusted[signal] = (
                    f"Trust Score {score:.2f} based on "
                    f"calibration metrics"
                )

        # Data quality signals
        ie.data_quality = forecast_result.get("data_quality", {})

        return ie

    def _resolve_trust_result(
        self,
        *,
        calibration_result: dict[str, Any] | None,
        trust_result: dict[str, Any] | None,
        trust_context: dict[str, Any] | None,
        trust_payload: dict[str, Any] | None,
        explain_options: dict[str, Any],
    ):
        if trust_result is not None:
            return coerce_normalized_trust_result(trust_result)

        if self.trust_router is None:
            return None

        context = dict(explain_options.get("trust_context", {}))
        if trust_context:
            context.update(trust_context)

        payload = trust_payload if trust_payload is not None else explain_options.get("trust_payload")
        if payload is None and calibration_result is not None:
            payload = calibration_result
            if "domain" not in context:
                context["domain"] = "prediction_market"
            if "source_type" not in context:
                context["source_type"] = "prediction_market"

        if payload is None:
            return None

        mode = context.get("mode", "single")
        if mode == "multi":
            return self.trust_router.analyze_multi(context=context, payload=payload)
        return self.trust_router.analyze(context=context, payload=payload)

    def _explain_plan(
        self,
        forecast_result: dict[str, Any],
        eval_result: dict[str, Any] | None,
        time_series_data: Any | None,
        include_decomposition: bool = True,
        include_attribution: bool = False,
    ) -> PlanExplanation:
        """Assemble plan-stage explanation: why this model and scenario."""
        pe = PlanExplanation()

        # Model Selection Explanation
        if eval_result and self.model_selection:
            ms = self.model_selection.explain(eval_result)
            pe.model_selected = ms.get("model_selected", "")
            pe.why_this_model = ms.get("why_this_model", "")
            pe.model_ranking = ms.get("model_ranking", [])
            pe.eval_evidence = ms.get("eval_evidence", {})
        elif eval_result:
            pe.model_selected = eval_result.get("best_model", "")
            pe.why_this_model = (
                "Selected based on lowest error metric in "
                "expanding-window cross-validation"
            )

        # Forecast Decomposition
        if include_decomposition and time_series_data is not None:
            if self.forecast_decomposer:
                decomp = self.forecast_decomposer.decompose(time_series_data)
                pe.forecast_decomposition = decomp
            else:
                pe.forecast_decomposition = {
                    "note": "Decomposition available in Phase 3+"
                }

        # Feature Attribution (Phase 3+)
        if include_attribution and time_series_data is not None:
            if self.feature_attribution:
                pe.feature_importance = self.feature_attribution.compute(
                    time_series_data
                )

        # Scenario Rationale
        if self.scenario_rationale and "scenarios" in forecast_result:
            pe.scenarios = self.scenario_rationale.explain(
                forecast_result["scenarios"]
            )
        elif "scenarios" in forecast_result:
            pe.scenarios = forecast_result["scenarios"]

        return pe

    def _explain_decision_policy(
        self,
        forecast_result: dict[str, Any],
        policy_config: dict[str, Any] | None,
        trust_metadata: dict[str, Any] | None = None,
    ) -> DecisionPolicyExplanation:
        """Assemble decision-policy explanation: why this action was recommended."""
        dpe = DecisionPolicyExplanation()

        if policy_config and self.decision_policy:
            dp = self.decision_policy.explain(
                forecast_result, policy_config, trust_result=trust_metadata
            )
            dpe.auto_executed = dp.get("auto_executed", False)
            dpe.confidence = dp.get("confidence", 0.0)
            dpe.threshold = dp.get("threshold", 0.0)
            dpe.reason = dp.get("reason", "")
            dpe.human_override = dp.get("human_override", True)
            dpe.policy_rules_applied = dp.get("policy_rules_applied", [])
            dpe.escalation_triggered = dp.get("escalation_triggered", False)
            dpe.escalation_reason = dp.get("escalation_reason", "")
            dpe.trust_score = dp.get("trust_score")
            dpe.risk_category = dp.get("risk_category")
            dpe.trust_blocked = dp.get("trust_blocked", False)
            dpe.constraint_violations_count = dp.get("constraint_violations_count", 0)
        elif policy_config:
            # Simple threshold-based explanation
            confidence = forecast_result.get("confidence", 0.0)
            threshold = policy_config.get("auto_execute_threshold", 0.85)
            dpe.confidence = confidence
            dpe.threshold = threshold
            dpe.auto_executed = confidence >= threshold
            dpe.human_override = True
            if dpe.auto_executed:
                dpe.reason = (
                    f"confidence {confidence:.2f} >= threshold {threshold:.2f} "
                    f"→ auto-execution recommended"
                )
            else:
                dpe.reason = (
                    f"confidence {confidence:.2f} < threshold {threshold:.2f} "
                    f"→ human approval required"
                )
                dpe.escalation_triggered = True
                dpe.escalation_reason = (
                    "Confidence below auto-execution threshold. "
                    "Evidence package attached for reviewer."
                )

        return dpe
