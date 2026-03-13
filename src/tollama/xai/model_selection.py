"""
tollama.xai.model_selection — Model Selection Explanation

v3.8 Phase 2a: "Model Selection Explanation" 제품화
tollama-eval의 expanding-window CV 결과를 구조화된 모델 선택 근거로 변환.

"왜 이 모델이 선택되었는가?"에 대한 evidence-backed rationale 생성.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class ModelRank:
    """Single model's ranking entry with evidence."""
    model_name: str = ""
    rank: int = 0
    metrics: dict[str, float] = field(default_factory=dict)
    strengths: list[str] = field(default_factory=list)
    weaknesses: list[str] = field(default_factory=list)


class ModelSelectionExplainer:
    """
    Converts tollama-eval cross-validation results into structured
    model selection explanations.

    Phase 2a: Repackages existing eval metrics
    Phase 3+: Adds comparative narratives and visualization metadata
    """

    # Metrics we recognize and know how to explain
    METRIC_DESCRIPTIONS = {
        "brier_score": {
            "name": "Brier Score",
            "direction": "lower_is_better",
            "explanation": "Measures the mean squared difference between predicted probabilities and actual outcomes",
        },
        "mse": {
            "name": "Mean Squared Error",
            "direction": "lower_is_better",
            "explanation": "Average squared forecast error across all time steps",
        },
        "mae": {
            "name": "Mean Absolute Error",
            "direction": "lower_is_better",
            "explanation": "Average absolute forecast error — robust to outliers",
        },
        "rmse": {
            "name": "Root Mean Squared Error",
            "direction": "lower_is_better",
            "explanation": "Square root of MSE — same unit as target variable",
        },
        "mape": {
            "name": "Mean Absolute Percentage Error",
            "direction": "lower_is_better",
            "explanation": "Percentage-based error metric for intuitive interpretation",
        },
        "smape": {
            "name": "Symmetric MAPE",
            "direction": "lower_is_better",
            "explanation": "Symmetric percentage error — handles near-zero values better",
        },
        "ece": {
            "name": "Expected Calibration Error",
            "direction": "lower_is_better",
            "explanation": "Measures how well predicted probabilities match actual frequencies",
        },
        "log_loss": {
            "name": "Log Loss",
            "direction": "lower_is_better",
            "explanation": "Logarithmic scoring rule for probabilistic predictions",
        },
        "r2": {
            "name": "R² Score",
            "direction": "higher_is_better",
            "explanation": "Proportion of variance explained by the model",
        },
        "coverage": {
            "name": "Prediction Interval Coverage",
            "direction": "higher_is_better",
            "explanation": "Fraction of actuals falling within prediction intervals",
        },
    }

    def __init__(
        self,
        primary_metric: str = "mse",
        secondary_metrics: Optional[list[str]] = None,
    ):
        self.primary_metric = primary_metric
        self.secondary_metrics = secondary_metrics or ["mae", "rmse", "mape"]

    def explain(self, eval_result: dict[str, Any]) -> dict[str, Any]:
        """
        Generate model selection explanation from eval results.

        Parameters
        ----------
        eval_result : dict
            tollama-eval output containing:
              - model_results: list of {model_name, metrics: {metric: value}}
              - cv_config: {n_splits, strategy, ...}
              - best_model: str
              - dataset_info: {n_rows, n_cols, frequency, ...}

        Returns
        -------
        dict with:
            - model_selected: str
            - why_this_model: str (natural language explanation)
            - model_ranking: list of ModelRank dicts
            - eval_evidence: dict of supporting evidence
        """
        model_results = eval_result.get("model_results", [])
        best_model = eval_result.get("best_model", "")
        cv_config = eval_result.get("cv_config", {})

        if not model_results:
            return {
                "model_selected": best_model or "unknown",
                "why_this_model": "No evaluation results available",
                "model_ranking": [],
                "eval_evidence": {},
            }

        # ── Rank models ──
        ranking = self._rank_models(model_results)

        # ── Generate natural language rationale ──
        winner = ranking[0] if ranking else None
        why = self._generate_rationale(winner, ranking, cv_config)

        # ── Build evidence summary ──
        evidence = self._build_evidence(ranking, cv_config, eval_result)

        return {
            "model_selected": winner.model_name if winner else best_model,
            "why_this_model": why,
            "model_ranking": [
                {
                    "model_name": r.model_name,
                    "rank": r.rank,
                    "metrics": r.metrics,
                    "strengths": r.strengths,
                    "weaknesses": r.weaknesses,
                }
                for r in ranking
            ],
            "eval_evidence": evidence,
        }

    def _rank_models(
        self, model_results: list[dict[str, Any]]
    ) -> list[ModelRank]:
        """Rank models by primary metric, annotate strengths/weaknesses."""
        # Sort by primary metric
        metric_info = self.METRIC_DESCRIPTIONS.get(self.primary_metric, {})
        ascending = metric_info.get("direction", "lower_is_better") == "lower_is_better"

        sorted_results = sorted(
            model_results,
            key=lambda m: m.get("metrics", {}).get(self.primary_metric, float("inf")),
            reverse=not ascending,
        )

        ranking = []
        for i, result in enumerate(sorted_results):
            metrics = result.get("metrics", {})
            mr = ModelRank(
                model_name=result.get("model_name", f"model_{i}"),
                rank=i + 1,
                metrics=metrics,
            )

            # Identify strengths/weaknesses across secondary metrics
            mr.strengths = self._identify_strengths(
                metrics, model_results, result.get("model_name", "")
            )
            mr.weaknesses = self._identify_weaknesses(
                metrics, model_results, result.get("model_name", "")
            )

            ranking.append(mr)

        return ranking

    def _identify_strengths(
        self,
        metrics: dict[str, float],
        all_results: list[dict[str, Any]],
        model_name: str,
    ) -> list[str]:
        """Identify metrics where this model excels relative to peers."""
        strengths = []
        all_metrics = self.secondary_metrics + [self.primary_metric]

        for metric in all_metrics:
            if metric not in metrics:
                continue
            values = [
                r.get("metrics", {}).get(metric)
                for r in all_results
                if r.get("metrics", {}).get(metric) is not None
            ]
            if not values:
                continue

            info = self.METRIC_DESCRIPTIONS.get(metric, {})
            is_lower_better = info.get("direction", "lower_is_better") == "lower_is_better"
            best_val = min(values) if is_lower_better else max(values)

            if metrics[metric] == best_val:
                metric_name = info.get("name", metric)
                strengths.append(f"Best {metric_name} ({metrics[metric]:.4f})")

        return strengths

    def _identify_weaknesses(
        self,
        metrics: dict[str, float],
        all_results: list[dict[str, Any]],
        model_name: str,
    ) -> list[str]:
        """Identify metrics where this model underperforms."""
        weaknesses = []
        for metric in self.secondary_metrics:
            if metric not in metrics:
                continue
            values = [
                r.get("metrics", {}).get(metric)
                for r in all_results
                if r.get("metrics", {}).get(metric) is not None
            ]
            if not values:
                continue

            info = self.METRIC_DESCRIPTIONS.get(metric, {})
            is_lower_better = info.get("direction", "lower_is_better") == "lower_is_better"
            worst_val = max(values) if is_lower_better else min(values)

            if metrics[metric] == worst_val and len(values) > 1:
                metric_name = info.get("name", metric)
                weaknesses.append(
                    f"Worst {metric_name} ({metrics[metric]:.4f})"
                )

        return weaknesses

    def _generate_rationale(
        self,
        winner: Optional[ModelRank],
        ranking: list[ModelRank],
        cv_config: dict[str, Any],
    ) -> str:
        """Generate human-readable model selection rationale."""
        if not winner:
            return "No models evaluated"

        metric_info = self.METRIC_DESCRIPTIONS.get(self.primary_metric, {})
        metric_name = metric_info.get("name", self.primary_metric)
        primary_val = winner.metrics.get(self.primary_metric, 0)
        n_models = len(ranking)
        n_splits = cv_config.get("n_splits", "unknown")
        strategy = cv_config.get("strategy", "expanding-window")

        rationale = (
            f"{winner.model_name} selected: lowest {metric_name} "
            f"({primary_val:.4f}) across {n_models} models "
            f"evaluated with {strategy} cross-validation"
        )
        if n_splits != "unknown":
            rationale += f" ({n_splits} splits)"

        # Add runner-up comparison
        if len(ranking) > 1:
            runner = ranking[1]
            runner_val = runner.metrics.get(self.primary_metric, 0)
            margin = abs(primary_val - runner_val)
            rationale += (
                f". Runner-up: {runner.model_name} "
                f"({runner_val:.4f}, margin: {margin:.4f})"
            )

        return rationale

    def _build_evidence(
        self,
        ranking: list[ModelRank],
        cv_config: dict[str, Any],
        eval_result: dict[str, Any],
    ) -> dict[str, Any]:
        """Build structured evidence summary."""
        return {
            "cv_strategy": cv_config.get("strategy", "expanding-window"),
            "n_splits": cv_config.get("n_splits", None),
            "n_models_evaluated": len(ranking),
            "primary_metric": self.primary_metric,
            "primary_metric_description": self.METRIC_DESCRIPTIONS.get(
                self.primary_metric, {}
            ).get("explanation", ""),
            "dataset_info": eval_result.get("dataset_info", {}),
            "reproducibility": {
                "cv_seed": cv_config.get("seed", None),
                "tollama_eval_version": eval_result.get("version", "unknown"),
                "timestamp": eval_result.get("timestamp", ""),
            },
        }
