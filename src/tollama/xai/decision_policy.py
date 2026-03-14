"""
tollama.xai.decision_policy — Decision Policy Explanation

v3.8: "왜 이 승인/자동화 정책이 권고되었는가"
Confidence threshold, policy rule, override 가능 여부를 보여주고,
최종 액션이 아니라 정책 판단의 이유를 설명.
"""

from __future__ import annotations

from typing import Any, Optional


class DecisionPolicyExplainer:
    """
    Explains decision policy outcomes: why was auto-execution recommended,
    or why was human approval requested?

    v3.8 principle: "Human-in-the-Loop First"
    초기 상용화의 hero story는 full autonomy가 아니라
    추천 → 설명 → 승인 → 실행 구조.
    """

    def __init__(
        self,
        default_threshold: float = 0.85,
        escalation_rules: Optional[list[dict[str, Any]]] = None,
    ):
        self.default_threshold = default_threshold
        self.escalation_rules = escalation_rules or []

    def explain(
        self,
        forecast_result: dict[str, Any],
        policy_config: dict[str, Any],
        trust_result: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """
        Generate decision policy explanation.

        Parameters
        ----------
        forecast_result : dict
            Contains confidence, model info, forecast values
        policy_config : dict
            Contains:
              - auto_execute_threshold: float
              - require_approval_above: float (value threshold)
              - escalation_rules: list of rule dicts
              - allowed_actions: list of str
              - audit_required: bool
              - trust_threshold: float (default 0.5)
        trust_result : dict, optional
            Trust intelligence metadata from the trust pipeline bridge.
            Expected structure: {"trust_intelligence": {"trust_score": ..., ...}}

        Returns
        -------
        dict with policy explanation
        """
        confidence = forecast_result.get("confidence", 0.0)
        threshold = policy_config.get(
            "auto_execute_threshold", self.default_threshold
        )
        audit_required = policy_config.get("audit_required", True)

        # Evaluate policy
        auto_executed = confidence >= threshold
        human_override = True  # Always true in Phase 2a

        # Check escalation rules
        escalation_triggered = False
        escalation_reason = ""
        policy_rules_applied = []

        # Rule: confidence threshold
        if auto_executed:
            policy_rules_applied.append(
                f"PASS: confidence {confidence:.2f} >= threshold {threshold:.2f}"
            )
        else:
            policy_rules_applied.append(
                f"FAIL: confidence {confidence:.2f} < threshold {threshold:.2f}"
            )
            escalation_triggered = True
            escalation_reason = (
                f"Confidence below auto-execution threshold. "
                f"Human review required with evidence package."
            )

        # ── Trust Intelligence Gates ──
        trust_score = None
        risk_category = None
        trust_blocked = False
        constraint_violations_count = 0

        if trust_result is not None:
            ti = trust_result.get("trust_intelligence", {})
            trust_score = ti.get("trust_score")
            components = ti.get("components", {})
            risk_category = components.get("risk_category")
            violations = ti.get("violations", [])

            # Gate A: Trust Score Hard Gate
            if trust_score is not None:
                trust_threshold = policy_config.get("trust_threshold", 0.5)
                if trust_score < trust_threshold:
                    policy_rules_applied.append(
                        f"BLOCK: trust_score {trust_score:.2f} "
                        f"< trust_threshold {trust_threshold:.2f}"
                    )
                    auto_executed = False
                    escalation_triggered = True
                    trust_blocked = True
                    escalation_reason = (
                        f"Trust score {trust_score:.2f} below threshold "
                        f"{trust_threshold:.2f}. "
                    )
                else:
                    policy_rules_applied.append(
                        f"PASS: trust_score {trust_score:.2f} "
                        f">= trust_threshold {trust_threshold:.2f}"
                    )

            # Gate B: Constraint Violations Block
            critical_violations = [
                v for v in violations if v.get("severity") == "critical"
            ]
            constraint_violations_count = len(critical_violations)
            if critical_violations:
                names = ", ".join(v.get("name", "unknown") for v in critical_violations)
                policy_rules_applied.append(
                    f"BLOCK: {len(critical_violations)} critical "
                    f"constraint violation(s): {names}"
                )
                auto_executed = False
                escalation_triggered = True
                trust_blocked = True
                escalation_reason += (
                    f"{len(critical_violations)} critical constraint "
                    f"violation(s). "
                )

            # Gate C: Risk Category Escalation
            if risk_category == "RED":
                policy_rules_applied.append("BLOCK: risk_category RED")
                auto_executed = False
                escalation_triggered = True
                trust_blocked = True
                escalation_reason += "Risk category RED. "
            elif risk_category == "YELLOW":
                policy_rules_applied.append("WARN: risk_category YELLOW")

        # Rule: value threshold
        value_threshold = policy_config.get("require_approval_above")
        forecast_value = forecast_result.get("forecast_value")
        if value_threshold is not None and forecast_value is not None:
            if abs(forecast_value) > value_threshold:
                policy_rules_applied.append(
                    f"ESCALATE: forecast value |{forecast_value}| > "
                    f"approval threshold {value_threshold}"
                )
                escalation_triggered = True
                auto_executed = False
                escalation_reason += (
                    f" Forecast value exceeds approval threshold."
                )

        # Rule: custom escalation rules
        for rule in policy_config.get("escalation_rules", self.escalation_rules):
            rule_result = self._evaluate_rule(rule, forecast_result)
            if rule_result["triggered"]:
                policy_rules_applied.append(
                    f"RULE [{rule.get('name', 'custom')}]: {rule_result['message']}"
                )
                if rule.get("blocks_auto_execute", False):
                    escalation_triggered = True
                    auto_executed = False
                    escalation_reason += f" {rule_result['message']}"

        # Rule: audit requirement
        if audit_required:
            policy_rules_applied.append(
                "AUDIT: Decision audit trail will be recorded"
            )

        # Generate reason
        if auto_executed:
            reason = (
                f"confidence {confidence:.2f} >= threshold {threshold:.2f} "
                f"→ auto-execution recommended"
            )
        else:
            reason = (
                f"confidence {confidence:.2f} < threshold {threshold:.2f} "
                f"→ human approval required"
            )

        return {
            "auto_executed": auto_executed,
            "confidence": confidence,
            "threshold": threshold,
            "reason": reason,
            "human_override": human_override,
            "policy_rules_applied": policy_rules_applied,
            "escalation_triggered": escalation_triggered,
            "escalation_reason": escalation_reason.strip(),
            "audit_required": audit_required,
            "trust_score": trust_score,
            "risk_category": risk_category,
            "trust_blocked": trust_blocked,
            "constraint_violations_count": constraint_violations_count,
            "governance_note": (
                "Phase 2a: All decisions support human override. "
                "Full approval workflow in Phase 5."
            ),
        }

    def _evaluate_rule(
        self,
        rule: dict[str, Any],
        forecast_result: dict[str, Any],
    ) -> dict[str, Any]:
        """Evaluate a single escalation rule."""
        rule_type = rule.get("type", "threshold")
        field = rule.get("field", "confidence")
        operator = rule.get("operator", "lt")
        value = rule.get("value", 0)

        actual = forecast_result.get(field)
        if actual is None:
            return {"triggered": False, "message": f"Field '{field}' not found"}

        triggered = False
        if operator == "lt":
            triggered = actual < value
        elif operator == "gt":
            triggered = actual > value
        elif operator == "eq":
            triggered = actual == value
        elif operator == "lte":
            triggered = actual <= value
        elif operator == "gte":
            triggered = actual >= value

        message = (
            f"{field} ({actual}) {operator} {value} = "
            f"{'triggered' if triggered else 'not triggered'}"
        )

        return {"triggered": triggered, "message": message}
