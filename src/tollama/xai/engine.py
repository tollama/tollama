"""
tollama.xai.engine — Central Explanation Engine

v3.8: "XAI는 처음부터 별도의 마법 같은 엔진이 아니라,
기존 증거를 구조화하여 의사결정에 연결하는 인터페이스 레이어로 시작합니다."

Phase 2a: Explanation Facade — 기존 eval/calibration/policy output을
decision-ready format으로 조립
Phase 3+: Feature Attribution Layer 완성, richer decomposition
Phase 4: /api/explain-decision 정식 출시
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────

@dataclass
class InputExplanation:
    """Input 단계 설명: 왜 이 신호를 반영했는가"""
    signals_used: list[str] = field(default_factory=list)
    trust_scores: dict[str, float] = field(default_factory=dict)
    trust_breakdowns: dict[str, dict[str, float]] = field(default_factory=dict)
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
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ──────────────────────────────────────────────────────────────
# Explanation Engine
# ──────────────────────────────────────────────────────────────

class ExplanationEngine:
    """
    Central orchestrator for Tollama XAI.

    Phase 2a (현재): Explanation Facade
    - 기존 eval 결과, Trust Score 분해, counterfactual을
      "Model Selection Explanation", "Trust Score Breakdown",
      "Scenario Rationale"로 제품화
    - 새로운 ML 추론이 아니라, existing evidence를
      decision-ready format으로 조립

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
    ):
        self.model_selection = model_selection_explainer
        self.forecast_decomposer = forecast_decomposer
        self.feature_attribution = feature_attribution
        self.scenario_rationale = scenario_rationale
        self.trust_breakdown = trust_breakdown
        self.decision_policy = decision_policy_explainer

    def explain_decision(
        self,
        forecast_result: dict[str, Any],
        eval_result: Optional[dict[str, Any]] = None,
        calibration_result: Optional[dict[str, Any]] = None,
        policy_config: Optional[dict[str, Any]] = None,
        time_series_data: Optional[Any] = None,
        explain_options: Optional[dict[str, Any]] = None,
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
        policy_config : dict, optional
            Decision policy configuration (thresholds, rules, approval config)
        time_series_data : array-like, optional
            Raw time series for decomposition and feature attribution
        explain_options : dict, optional
            Control explanation depth: {"decompose": True, "attribution": True, ...}

        Returns
        -------
        DecisionExplanation
            Structured explanation covering Input → Plan → Decision Policy
        """
        import uuid

        options = explain_options or {}
        explanation = DecisionExplanation(
            explanation_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

        # ── Phase 1: Input Explanation (Signal Trust) ──
        explanation.input_explanation = self._explain_input(
            forecast_result=forecast_result,
            calibration_result=calibration_result,
        )

        # ── Phase 2: Plan Explanation (Model Selection + Decomposition) ──
        explanation.plan_explanation = self._explain_plan(
            forecast_result=forecast_result,
            eval_result=eval_result,
            time_series_data=time_series_data,
            include_decomposition=options.get("decompose", True),
            include_attribution=options.get("attribution", False),
        )

        # ── Phase 3: Decision Policy Explanation ──
        explanation.decision_policy_explanation = self._explain_decision_policy(
            forecast_result=forecast_result,
            policy_config=policy_config,
        )

        explanation.metadata = {
            "engine_version": "0.1.0",
            "phase": "2a",
            "explanation_type": "facade",
            "note": (
                "Phase 2a explanation is evidence repackaging, "
                "not deep interpretability. Phase 3+ adds richer "
                "feature attribution and decomposition."
            ),
        }

        logger.info(
            "Generated decision explanation %s", explanation.explanation_id
        )
        return explanation

    def _explain_input(
        self,
        forecast_result: dict[str, Any],
        calibration_result: Optional[dict[str, Any]],
    ) -> InputExplanation:
        """Assemble input-stage explanation: why these signals were used."""
        ie = InputExplanation()

        # Identify signals used
        ie.signals_used = forecast_result.get("signals_used", ["internal_ts"])

        # Trust Score breakdown from calibration
        if calibration_result and self.trust_breakdown:
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

    def _explain_plan(
        self,
        forecast_result: dict[str, Any],
        eval_result: Optional[dict[str, Any]],
        time_series_data: Optional[Any],
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
                f"Selected based on lowest error metric in "
                f"expanding-window cross-validation"
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
        policy_config: Optional[dict[str, Any]],
    ) -> DecisionPolicyExplanation:
        """Assemble decision-policy explanation: why this action was recommended."""
        dpe = DecisionPolicyExplanation()

        if policy_config and self.decision_policy:
            dp = self.decision_policy.explain(forecast_result, policy_config)
            dpe.auto_executed = dp.get("auto_executed", False)
            dpe.confidence = dp.get("confidence", 0.0)
            dpe.threshold = dp.get("threshold", 0.0)
            dpe.reason = dp.get("reason", "")
            dpe.human_override = dp.get("human_override", True)
            dpe.policy_rules_applied = dp.get("policy_rules_applied", [])
            dpe.escalation_triggered = dp.get("escalation_triggered", False)
            dpe.escalation_reason = dp.get("escalation_reason", "")
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
                    f"Confidence below auto-execution threshold. "
                    f"Evidence package attached for reviewer."
                )

        return dpe
