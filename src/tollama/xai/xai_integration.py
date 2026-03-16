"""
tollama_eval.xai — XAI Integration for tollama-eval

v3.8 Phase 2a: tollama-eval의 CV 결과를 explanation-ready format으로 확장.
- Eval result에 model selection rationale 자동 첨부
- CV 결과를 evidence package로 구조화
- Private Eval Harness의 핵심: reproducible evaluation + selection rationale

기존 tollama-eval의 출력 형식을 유지하면서,
xai_explanation 필드를 추가하여 ExplanationEngine에서 바로 소비 가능하게 함.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from typing import Any


class EvalExplanationExtender:
    """
    Extends tollama-eval results with XAI explanation metadata.

    This module wraps around existing eval output and adds:
    - Model selection rationale (natural language)
    - Metric comparison evidence
    - CV reproducibility proof
    - Audit-ready evaluation report
    """

    def extend_eval_result(
        self,
        eval_result: dict[str, Any],
        primary_metric: str = "mse",
        include_narrative: bool = True,
    ) -> dict[str, Any]:
        """
        Add XAI explanation fields to eval result.

        Parameters
        ----------
        eval_result : dict
            Standard tollama-eval output
        primary_metric : str
            Primary metric for model selection rationale
        include_narrative : bool
            Include natural language narrative

        Returns
        -------
        dict: Extended eval result with xai_explanation field
        """
        extended = dict(eval_result)

        model_results = eval_result.get("model_results", [])
        cv_config = eval_result.get("cv_config", {})
        best_model = eval_result.get("best_model", "")

        # Build explanation
        xai = {
            "version": "0.1.0",
            "generated_at": datetime.now(UTC).isoformat(),
            "explanation_type": "eval_evidence",
        }

        # Model selection rationale
        xai["model_selection"] = self._build_model_selection(
            model_results, best_model, primary_metric, cv_config
        )

        # Metric comparison matrix
        xai["metric_comparison"] = self._build_metric_comparison(
            model_results, primary_metric
        )

        # Reproducibility proof
        xai["reproducibility"] = self._build_reproducibility_proof(
            eval_result, cv_config
        )

        # Narrative (natural language)
        if include_narrative:
            xai["narrative"] = self._generate_narrative(
                model_results, best_model, primary_metric, cv_config
            )

        extended["xai_explanation"] = xai
        return extended

    def generate_eval_report(
        self,
        eval_result: dict[str, Any],
        format: str = "json",
    ) -> str | dict:
        """
        Generate a standalone evaluation explanation report.

        Parameters
        ----------
        eval_result : dict
            Extended eval result (with xai_explanation)
        format : str
            "json" or "markdown"
        """
        xai = eval_result.get("xai_explanation", {})

        if format == "markdown":
            return self._to_markdown(eval_result, xai)
        return xai

    # ── Private builders ──

    def _build_model_selection(
        self,
        model_results: list[dict],
        best_model: str,
        primary_metric: str,
        cv_config: dict,
    ) -> dict[str, Any]:
        """Build model selection evidence."""
        if not model_results:
            return {"rationale": "No models evaluated", "evidence": []}

        # Sort by primary metric
        sorted_models = sorted(
            model_results,
            key=lambda m: m.get("metrics", {}).get(primary_metric, float("inf")),
        )

        evidence = []
        for i, model in enumerate(sorted_models):
            metrics = model.get("metrics", {})
            evidence.append({
                "rank": i + 1,
                "model_name": model.get("model_name", ""),
                "primary_metric_value": metrics.get(primary_metric),
                "all_metrics": metrics,
                "is_selected": model.get("model_name") == best_model,
            })

        winner = sorted_models[0] if sorted_models else {}
        winner_name = winner.get("model_name", best_model)
        winner_val = winner.get("metrics", {}).get(primary_metric, 0)
        n_models = len(sorted_models)
        strategy = cv_config.get("strategy", "expanding-window")

        rationale = (
            f"{winner_name} selected with {primary_metric}={winner_val:.4f}, "
            f"best among {n_models} models evaluated via {strategy} CV"
        )

        if len(sorted_models) > 1:
            runner = sorted_models[1]
            runner_val = runner.get("metrics", {}).get(primary_metric, 0)
            margin = abs(winner_val - runner_val)
            rationale += (
                f" (margin over {runner.get('model_name', '')}: {margin:.4f})"
            )

        return {
            "selected_model": winner_name,
            "rationale": rationale,
            "primary_metric": primary_metric,
            "n_candidates": n_models,
            "evidence": evidence,
        }

    def _build_metric_comparison(
        self,
        model_results: list[dict],
        primary_metric: str,
    ) -> dict[str, Any]:
        """Build cross-model metric comparison."""
        if not model_results:
            return {}

        # Collect all metric names
        all_metrics = set()
        for m in model_results:
            all_metrics.update(m.get("metrics", {}).keys())

        comparison = {
            "metrics": sorted(all_metrics),
            "models": [],
        }
        for model in model_results:
            comparison["models"].append({
                "name": model.get("model_name", ""),
                "values": {
                    metric: model.get("metrics", {}).get(metric)
                    for metric in sorted(all_metrics)
                },
            })

        return comparison

    def _build_reproducibility_proof(
        self,
        eval_result: dict[str, Any],
        cv_config: dict[str, Any],
    ) -> dict[str, Any]:
        """Build reproducibility evidence."""
        # Hash the eval config for reproducibility tracking
        config_str = json.dumps(cv_config, sort_keys=True, default=str)
        config_hash = hashlib.sha256(config_str.encode()).hexdigest()[:12]

        return {
            "cv_strategy": cv_config.get("strategy", "expanding-window"),
            "n_splits": cv_config.get("n_splits"),
            "seed": cv_config.get("seed"),
            "config_hash": config_hash,
            "tollama_eval_version": eval_result.get("version", ""),
            "timestamp": eval_result.get("timestamp", ""),
            "note": (
                "This evaluation is reproducible given the same data, "
                "CV configuration, and random seed."
            ),
        }

    def _generate_narrative(
        self,
        model_results: list[dict],
        best_model: str,
        primary_metric: str,
        cv_config: dict,
    ) -> str:
        """Generate natural language evaluation narrative."""
        n = len(model_results)
        strategy = cv_config.get("strategy", "expanding-window")
        n_splits = cv_config.get("n_splits", "N")

        if n == 0:
            return "No evaluation performed."

        narrative = (
            f"Evaluation Summary: {n} models were evaluated using "
            f"{strategy} cross-validation with {n_splits} splits. "
        )

        sorted_models = sorted(
            model_results,
            key=lambda m: m.get("metrics", {}).get(primary_metric, float("inf")),
        )

        winner = sorted_models[0]
        winner_name = winner.get("model_name", best_model)
        winner_metrics = winner.get("metrics", {})

        narrative += (
            f"{winner_name} achieved the best performance with "
            f"{primary_metric}={winner_metrics.get(primary_metric, 'N/A')}"
        )

        # Add key insight
        if len(sorted_models) > 1:
            runner = sorted_models[1]
            narrative += (
                f", outperforming {runner.get('model_name', '')} "
                f"({primary_metric}="
                f"{runner.get('metrics', {}).get(primary_metric, 'N/A')})"
            )

        narrative += ". "

        # Add metric breadth insight
        metric_keys = list(winner_metrics.keys())
        if len(metric_keys) > 1:
            other_metrics = [m for m in metric_keys if m != primary_metric][:2]
            for m in other_metrics:
                val = winner_metrics.get(m)
                if val is not None:
                    narrative += f"{m}={val:.4f} "

        return narrative.strip()

    def _to_markdown(
        self, eval_result: dict[str, Any], xai: dict[str, Any]
    ) -> str:
        """Convert to Markdown report."""
        lines = []
        lines.append("# Evaluation Explanation Report")
        lines.append(f"\n*Generated: {xai.get('generated_at', '')}*\n")

        # Model Selection
        ms = xai.get("model_selection", {})
        lines.append("## Model Selection")
        lines.append(f"\n**Selected**: {ms.get('selected_model', '')}")
        lines.append(f"**Rationale**: {ms.get('rationale', '')}\n")

        # Rankings
        lines.append("### Model Rankings\n")
        lines.append("| Rank | Model | Primary Metric |")
        lines.append("| --- | --- | --- |")
        for e in ms.get("evidence", []):
            selected = " ✓" if e.get("is_selected") else ""
            lines.append(
                f"| {e.get('rank')} | {e.get('model_name')}{selected} | "
                f"{e.get('primary_metric_value', 'N/A')} |"
            )

        # Reproducibility
        repro = xai.get("reproducibility", {})
        lines.append("\n## Reproducibility")
        lines.append(f"\n- **Strategy**: {repro.get('cv_strategy', '')}")
        lines.append(f"- **Splits**: {repro.get('n_splits', '')}")
        lines.append(f"- **Seed**: {repro.get('seed', '')}")
        lines.append(f"- **Config Hash**: {repro.get('config_hash', '')}")

        # Narrative
        if xai.get("narrative"):
            lines.append(f"\n## Narrative\n\n{xai['narrative']}")

        return "\n".join(lines)
