"""
tollama.xai.model_card — EU AI Act Model Card Generator

v3.8: EU AI Act Model Card 자동 생성.
Phase 2b: Trust-intelligence 통합 시 trust score, risk category, constraint status 포함.
Phase 5에서 완전 자동화.

References:
  - EU AI Act (Regulation 2024/1689)
  - MiFID II model documentation requirements
  - Google Model Cards for Model Reporting (Mitchell et al. 2019)
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any


class ModelCardGenerator:
    """
    Generates EU AI Act compliant model cards for forecast models.

    A Model Card documents:
    - Model identity and version
    - Intended use and limitations
    - Training data description
    - Evaluation results and methodology
    - Fairness and bias considerations
    - Explanation capabilities
    - Governance and approval information
    """

    TEMPLATE_VERSION = "1.0.0"

    def generate(
        self,
        model_info: dict[str, Any],
        eval_result: dict[str, Any] | None = None,
        explanation_result: dict[str, Any] | None = None,
        governance_info: dict[str, Any] | None = None,
        custom_sections: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Generate a model card.

        Parameters
        ----------
        model_info : dict
            Model identity: name, version, type, description, author
        eval_result : dict, optional
            Evaluation results from tollama-eval
        explanation_result : dict, optional
            XAI explanation output
        governance_info : dict, optional
            Approval, audit, compliance information
        custom_sections : dict, optional
            Additional sections

        Returns
        -------
        dict: Structured model card
        """
        card = {
            "model_card_version": self.TEMPLATE_VERSION,
            "generated_at": datetime.now(UTC).isoformat(),
            "generator": "tollama-xai",
        }

        # Section 1: Model Details
        card["model_details"] = self._model_details(model_info)

        # Section 2: Intended Use
        card["intended_use"] = self._intended_use(model_info)

        # Section 3: Training Data
        card["training_data"] = self._training_data(model_info, eval_result)

        # Section 4: Evaluation
        card["evaluation"] = self._evaluation(eval_result)

        # Section 5: Explanation Capabilities
        card["explainability"] = self._explainability(explanation_result)

        # Section 6: Ethical & Fairness Considerations
        card["ethical_considerations"] = self._ethical_considerations(model_info)

        # Section 7: Governance & Compliance
        card["governance"] = self._governance(governance_info)

        # Section 8: Limitations & Risks
        card["limitations"] = self._limitations(model_info, eval_result)

        # Custom sections
        if custom_sections:
            card["additional_sections"] = custom_sections

        return card

    def to_markdown(self, card: dict[str, Any]) -> str:
        """Convert model card to Markdown format for reporting."""
        lines = []
        lines.append(f"# Model Card: {card.get('model_details', {}).get('name', 'Unknown Model')}")
        lines.append(f"\n*Generated: {card.get('generated_at', '')}*")
        lines.append(f"*Template: v{card.get('model_card_version', '')}*\n")

        # Model Details
        md = card.get("model_details", {})
        lines.append("## 1. Model Details")
        for key, value in md.items():
            if value:
                lines.append(f"- **{key.replace('_', ' ').title()}**: {value}")

        # Intended Use
        iu = card.get("intended_use", {})
        lines.append("\n## 2. Intended Use")
        for key, value in iu.items():
            if isinstance(value, list):
                lines.append(f"- **{key.replace('_', ' ').title()}**: {', '.join(str(v) for v in value)}")
            elif value:
                lines.append(f"- **{key.replace('_', ' ').title()}**: {value}")

        # Evaluation
        ev = card.get("evaluation", {})
        lines.append("\n## 3. Evaluation")
        if ev.get("metrics"):
            lines.append("\n| Metric | Value |")
            lines.append("| --- | --- |")
            for metric, value in ev["metrics"].items():
                lines.append(f"| {metric} | {value} |")
        if ev.get("methodology"):
            lines.append(f"\n**Methodology**: {ev['methodology']}")

        # Explainability
        ex = card.get("explainability", {})
        lines.append("\n## 4. Explainability")
        for key, value in ex.items():
            if value:
                lines.append(f"- **{key.replace('_', ' ').title()}**: {value}")

        # Governance
        gov = card.get("governance", {})
        lines.append("\n## 5. Governance & Compliance")
        for key, value in gov.items():
            if value:
                lines.append(f"- **{key.replace('_', ' ').title()}**: {value}")

        # Limitations
        lim = card.get("limitations", {})
        lines.append("\n## 6. Limitations & Risks")
        for key, value in lim.items():
            if isinstance(value, list):
                for item in value:
                    lines.append(f"- {item}")
            elif value:
                lines.append(f"- **{key.replace('_', ' ').title()}**: {value}")

        return "\n".join(lines)

    def to_json(self, card: dict[str, Any]) -> str:
        """Convert model card to JSON format."""
        return json.dumps(card, indent=2, ensure_ascii=False, default=str)

    # ── Section builders ──

    def _model_details(self, model_info: dict[str, Any]) -> dict[str, Any]:
        return {
            "name": model_info.get("name", ""),
            "version": model_info.get("version", ""),
            "type": model_info.get("type", "time_series_forecasting"),
            "description": model_info.get("description", ""),
            "author": model_info.get("author", "Tollama AI"),
            "license": model_info.get("license", ""),
            "framework": model_info.get("framework", "tollama"),
            "date_created": model_info.get("date_created", ""),
            "contact": model_info.get("contact", ""),
        }

    def _intended_use(self, model_info: dict[str, Any]) -> dict[str, Any]:
        return {
            "primary_use": model_info.get(
                "primary_use", "Time series forecasting for enterprise decision support"
            ),
            "primary_users": model_info.get(
                "primary_users",
                ["Quantitative analysts", "Risk managers", "Platform engineers"],
            ),
            "out_of_scope": model_info.get(
                "out_of_scope",
                [
                    "Real-time trading execution without human oversight",
                    "Medical or clinical decision-making",
                    "Decisions affecting individual rights without human review",
                ],
            ),
            "risk_category": model_info.get("risk_category", "limited_risk"),
            "eu_ai_act_classification": model_info.get(
                "eu_ai_act_classification",
                "Limited risk AI system — transparency obligations apply",
            ),
        }

    def _training_data(
        self, model_info: dict[str, Any], eval_result: dict[str, Any] | None
    ) -> dict[str, Any]:
        data_info = model_info.get("training_data", {})
        if eval_result:
            dataset_info = eval_result.get("dataset_info", {})
            data_info.update({
                "n_samples": dataset_info.get("n_rows"),
                "n_features": dataset_info.get("n_cols"),
                "frequency": dataset_info.get("frequency"),
                "date_range": dataset_info.get("date_range"),
            })
        return data_info

    def _evaluation(self, eval_result: dict[str, Any] | None) -> dict[str, Any]:
        if not eval_result:
            return {"status": "Evaluation pending", "metrics": {}}

        model_results = eval_result.get("model_results", [])
        best = None
        for r in model_results:
            if r.get("model_name") == eval_result.get("best_model"):
                best = r
                break

        cv_config = eval_result.get("cv_config", {})

        return {
            "methodology": (
                f"Expanding-window cross-validation with "
                f"{cv_config.get('n_splits', 'N')} splits"
            ),
            "metrics": best.get("metrics", {}) if best else {},
            "n_models_compared": len(model_results),
            "best_model": eval_result.get("best_model", ""),
            "reproducibility_seed": cv_config.get("seed"),
            "tollama_eval_version": eval_result.get("version", ""),
        }

    def _explainability(
        self, explanation_result: dict[str, Any] | None
    ) -> dict[str, Any]:
        if not explanation_result:
            return {
                "status": "Phase 2a: Explanation facade available",
                "methods": [
                    "Model Selection Explanation",
                    "Trust Score Breakdown",
                    "Scenario Rationale",
                ],
                "future_methods": [
                    "Feature Attribution (Phase 3)",
                    "Forecast Decomposition (Phase 3)",
                    "Decision Chain Tracing (Phase 5)",
                ],
            }

        metadata = explanation_result.get("metadata", {})
        has_trust_intelligence = "trust_intelligence" in metadata

        methods_used = [
            "Model Selection Explanation",
            "Trust Score Breakdown",
            "Forecast Decomposition",
            "Scenario Rationale",
            "Decision Policy Explanation",
        ]
        if has_trust_intelligence:
            methods_used.extend([
                "Trust Intelligence Pipeline (L1-L5)",
                "SHAP Feature Attribution",
                "Z3 Constraint Verification",
                "Trust-Gated Decisioning",
            ])

        result = {
            "status": (
                "Trust-intelligence integrated explanation available"
                if has_trust_intelligence
                else "Explanation available"
            ),
            "explanation_type": metadata.get("explanation_type", "facade"),
            "methods_used": methods_used,
            "explanation_coverage": "100% of forecast outputs",
        }
        if has_trust_intelligence:
            ti = metadata["trust_intelligence"]
            result["trust_intelligence_status"] = {
                "trust_score": ti.get("trust_score"),
                "risk_category": ti.get("components", {}).get("risk_category"),
                "constraint_satisfied": ti.get("components", {}).get("constraint_satisfied"),
            }
        return result

    def _ethical_considerations(self, model_info: dict[str, Any]) -> dict[str, Any]:
        return {
            "fairness": model_info.get(
                "fairness_note",
                "Time series forecasting models may reflect historical biases in data. "
                "Users should validate predictions against domain expertise.",
            ),
            "human_oversight": (
                "All automated decisions support human override. "
                "Confidence thresholds control auto-execution boundaries."
            ),
            "transparency": (
                "Tollama provides structured explanations for model selection, "
                "signal calibration, and decision policy at each prediction."
            ),
        }

    def _governance(
        self, governance_info: dict[str, Any] | None
    ) -> dict[str, Any]:
        if not governance_info:
            return {
                "status": "Phase 2a governance framework",
                "audit_trail": (
                    "Chain-of-Trust audit trail available via trust-intelligence pipeline; "
                    "decision-level audit trail integration in progress"
                ),
                "approval_workflow": "Human-in-the-loop approval supported",
                "compliance_frameworks": [
                    "EU AI Act (transparency obligations)",
                    "MiFID II (model documentation)",
                ],
            }

        return {
            "approved_by": governance_info.get("approved_by", ""),
            "approval_date": governance_info.get("approval_date", ""),
            "review_schedule": governance_info.get("review_schedule", "Quarterly"),
            "audit_trail": governance_info.get("audit_trail", "Enabled"),
            "compliance_frameworks": governance_info.get(
                "compliance_frameworks",
                ["EU AI Act", "MiFID II"],
            ),
        }

    def _limitations(
        self, model_info: dict[str, Any], eval_result: dict[str, Any] | None
    ) -> dict[str, Any]:
        limitations = [
            "Forecasts are probabilistic estimates, not guarantees",
            "Performance may degrade on data distributions significantly different from training",
            "External signal calibration depends on market data availability and quality",
        ]

        risks = [
            "Regime changes may invalidate historical patterns",
            "Over-reliance on automated recommendations without domain expertise",
            "Trust Score reflects historical calibration, not future reliability",
        ]

        if eval_result:
            model_results = eval_result.get("model_results", [])
            if model_results:
                metrics = model_results[0].get("metrics", {})
                if metrics.get("mape", 0) > 20:
                    limitations.append(
                        f"High MAPE ({metrics['mape']:.1f}%) indicates significant prediction uncertainty"
                    )

        return {
            "known_limitations": limitations,
            "known_risks": risks,
            "mitigation": (
                "Use with human oversight. Validate against domain expertise. "
                "Monitor model drift with continuous evaluation."
            ),
        }
