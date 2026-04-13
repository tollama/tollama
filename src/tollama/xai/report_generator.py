"""
tollama.xai.report_generator — Decision Report & Explanation Report Builder

v3.8: "구조화된 Decision Report + Explanation Report 자동 출력"
Evidence-backed decision report를 JSON, Markdown, HTML 형식으로 생성.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any


class DecisionReportBuilder:
    """
    Builds decision reports from explanation engine output.

    Report types:
    - Decision Report: Executive summary of forecast + recommendation
    - Explanation Report: Detailed evidence for audit/compliance
    - Comparison Report: Multi-scenario analysis summary
    """

    def build_decision_report(
        self,
        explanation: dict[str, Any],
        forecast_result: dict[str, Any] | None = None,
        report_config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Build a decision report for C-suite/stakeholders.

        Parameters
        ----------
        explanation : dict
            Output from ExplanationEngine.explain_decision().to_dict()
        forecast_result : dict, optional
            Raw forecast output for additional context
        report_config : dict, optional
            Report customization (title, audience, format)

        Returns
        -------
        dict: Structured decision report
        """
        config = report_config or {}
        ie = explanation.get("input_explanation", {})
        pe = explanation.get("plan_explanation", {})
        dpe = explanation.get("decision_policy_explanation", {})

        report = {
            "report_type": "decision_report",
            "title": config.get("title", "Forecast Decision Report"),
            "generated_at": datetime.now(UTC).isoformat(),
            "explanation_id": explanation.get("explanation_id", ""),
            "executive_summary": self._executive_summary(ie, pe, dpe, forecast_result),
            "signal_assessment": {
                "signals_used": ie.get("signals_used", []),
                "trust_scores": ie.get("trust_scores", {}),
                "trust_assessment": ie.get("why_trusted", {}),
            },
            "model_selection": {
                "selected_model": pe.get("model_selected", ""),
                "rationale": pe.get("why_this_model", ""),
                "ranking_summary": [
                    {
                        "model": r.get("model_name"),
                        "rank": r.get("rank"),
                        "primary_metric": list(r.get("metrics", {}).values())[0]
                        if r.get("metrics")
                        else None,
                    }
                    for r in pe.get("model_ranking", [])[:5]
                ],
            },
            "forecast_analysis": {
                "decomposition": pe.get("forecast_decomposition", {}),
                "scenarios": pe.get("scenarios", {}),
            },
            "decision_recommendation": {
                "auto_execute_recommended": dpe.get("auto_executed", False),
                "confidence": dpe.get("confidence", 0.0),
                "threshold": dpe.get("threshold", 0.0),
                "action_required": (
                    "No action required — auto-execution recommended"
                    if dpe.get("auto_executed")
                    else "Human approval required before execution"
                ),
                "reason": dpe.get("reason", ""),
                "escalation": dpe.get("escalation_reason", ""),
            },
            "audit_trail": {
                "rules_applied": dpe.get("policy_rules_applied", []),
                "human_override_available": dpe.get("human_override", True),
                "explanation_version": explanation.get("version", ""),
            },
        }

        return report

    def build_explanation_report(
        self,
        explanation: dict[str, Any],
        model_card: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Build a detailed explanation report for audit/compliance.

        More detailed than decision report — includes all evidence.
        """
        ie = explanation.get("input_explanation", {})
        pe = explanation.get("plan_explanation", {})
        dpe = explanation.get("decision_policy_explanation", {})

        report = {
            "report_type": "explanation_report",
            "title": "Detailed Explanation & Audit Report",
            "generated_at": datetime.now(UTC).isoformat(),
            "explanation_id": explanation.get("explanation_id", ""),
            "compliance_note": (
                "This report provides structured explanation of the "
                "decision-making process for regulatory compliance "
                "(EU AI Act, MiFID II)."
            ),
            "input_evidence": {
                "signals": ie.get("signals_used", []),
                "trust_scores": ie.get("trust_scores", {}),
                "trust_breakdowns": ie.get("trust_breakdowns", {}),
                "trust_explanations": ie.get("why_trusted", {}),
                "data_quality": ie.get("data_quality", {}),
            },
            "model_evidence": {
                "selected": pe.get("model_selected", ""),
                "selection_rationale": pe.get("why_this_model", ""),
                "full_ranking": pe.get("model_ranking", []),
                "evaluation_evidence": pe.get("eval_evidence", {}),
            },
            "forecast_evidence": {
                "decomposition": pe.get("forecast_decomposition", {}),
                "feature_importance": pe.get("feature_importance", []),
                "scenarios": pe.get("scenarios", {}),
            },
            "policy_evidence": {
                "confidence": dpe.get("confidence", 0.0),
                "threshold": dpe.get("threshold", 0.0),
                "auto_executed": dpe.get("auto_executed", False),
                "reason": dpe.get("reason", ""),
                "rules_applied": dpe.get("policy_rules_applied", []),
                "escalation_triggered": dpe.get("escalation_triggered", False),
                "escalation_reason": dpe.get("escalation_reason", ""),
                "human_override": dpe.get("human_override", True),
            },
            "model_card": model_card or {"status": "Generate with ModelCardGenerator"},
            "metadata": explanation.get("metadata", {}),
        }

        return report

    def to_markdown(self, report: dict[str, Any]) -> str:
        """Convert report to Markdown."""
        lines = []
        report_type = report.get("report_type", "report")
        title = report.get("title", "Report")

        lines.append(f"# {title}")
        lines.append(f"\n*Generated: {report.get('generated_at', '')}*")
        lines.append(f"*ID: {report.get('explanation_id', '')}*\n")

        if report_type == "decision_report":
            lines.extend(self._decision_report_markdown(report))
        else:
            lines.extend(self._explanation_report_markdown(report))

        return "\n".join(lines)

    def to_json(self, report: dict[str, Any]) -> str:
        """Convert report to JSON."""
        return json.dumps(report, indent=2, ensure_ascii=False, default=str)

    # ── Private helpers ──

    def _executive_summary(
        self,
        ie: dict,
        pe: dict,
        dpe: dict,
        forecast_result: dict | None,
    ) -> str:
        model = pe.get("model_selected", "selected model")
        confidence = dpe.get("confidence", 0)
        auto = dpe.get("auto_executed", False)
        n_signals = len(ie.get("signals_used", []))

        summary = (
            f"Based on {n_signals} input signal(s) and evaluation of "
            f"the {model} model, the system recommends "
        )
        if auto:
            summary += (
                f"auto-execution (confidence: {confidence:.2f}). All evidence is attached below."
            )
        else:
            summary += (
                f"human review before execution "
                f"(confidence: {confidence:.2f}, below threshold). "
                f"Evidence package is attached for the approver."
            )

        return summary

    def _decision_report_markdown(self, report: dict) -> list[str]:
        lines = []

        # Executive Summary
        lines.append("## Executive Summary")
        lines.append(f"\n{report.get('executive_summary', '')}\n")

        # Signal Assessment
        sa = report.get("signal_assessment", {})
        lines.append("## Signal Assessment")
        for signal, score in sa.get("trust_scores", {}).items():
            assessment = sa.get("trust_assessment", {}).get(signal, "")
            lines.append(f"- **{signal}**: Trust Score {score:.2f} — {assessment}")

        # Model Selection
        ms = report.get("model_selection", {})
        lines.append("\n## Model Selection")
        lines.append(f"\n**Selected**: {ms.get('selected_model', '')}")
        lines.append(f"**Rationale**: {ms.get('rationale', '')}")

        # Decision Recommendation
        dr = report.get("decision_recommendation", {})
        lines.append("\n## Decision Recommendation")
        lines.append(f"\n**Action**: {dr.get('action_required', '')}")
        lines.append(f"**Confidence**: {dr.get('confidence', 0):.2f}")
        lines.append(f"**Reason**: {dr.get('reason', '')}")
        if dr.get("escalation"):
            lines.append(f"**Escalation**: {dr['escalation']}")

        return lines

    def _explanation_report_markdown(self, report: dict) -> list[str]:
        lines = []
        lines.append("## Compliance Note")
        lines.append(f"\n{report.get('compliance_note', '')}\n")

        for section_name, section_data in report.items():
            if section_name in (
                "report_type",
                "title",
                "generated_at",
                "explanation_id",
                "compliance_note",
                "metadata",
            ):
                continue
            if isinstance(section_data, dict):
                lines.append(f"\n## {section_name.replace('_', ' ').title()}")
                for k, v in section_data.items():
                    if isinstance(v, (list, dict)):
                        lines.append(f"- **{k}**: {json.dumps(v, default=str)[:200]}")
                    else:
                        lines.append(f"- **{k}**: {v}")

        return lines
