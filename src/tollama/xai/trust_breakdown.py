"""
tollama.xai.trust_breakdown — Trust Score Breakdown

v3.8 Phase 2a: Trust Score Breakdown 제품화
Market Calibration Agent의 Trust Score 구성요소를 분해하여
"왜 이 외부 신호를 반영해야 하는가"에 답하는 설명 생성.
"""

from __future__ import annotations

from typing import Any, Optional


class TrustBreakdown:
    """
    Converts Trust Score metrics from Market Calibration Agent
    into structured, human-readable explanations.

    Trust Score = f(Brier Score, Log Loss, ECE, volume, recency, ...)
    This module explains each component's contribution.
    """

    # Trust thresholds
    HIGH_TRUST = 0.75
    MEDIUM_TRUST = 0.50
    LOW_TRUST = 0.25

    # Component descriptions
    COMPONENT_INFO = {
        "brier_score": {
            "name": "Brier Score",
            "description": "Mean squared prediction error for probability forecasts",
            "good_threshold": 0.2,
            "direction": "lower_is_better",
        },
        "log_loss": {
            "name": "Log Loss",
            "description": "Logarithmic scoring rule penalizing confident wrong predictions",
            "good_threshold": 0.5,
            "direction": "lower_is_better",
        },
        "ece": {
            "name": "Expected Calibration Error",
            "description": "Gap between predicted probabilities and actual outcome frequencies",
            "good_threshold": 0.05,
            "direction": "lower_is_better",
        },
        "volume": {
            "name": "Market Volume",
            "description": "Trading volume as proxy for information quality",
            "good_threshold": 10000,
            "direction": "higher_is_better",
        },
        "recency": {
            "name": "Recency",
            "description": "How recently the signal was updated",
            "good_threshold": 0.8,
            "direction": "higher_is_better",
        },
        "resolution_clarity": {
            "name": "Resolution Clarity",
            "description": "How clearly the market question resolves to yes/no",
            "good_threshold": 0.9,
            "direction": "higher_is_better",
        },
    }

    def __init__(
        self,
        weight_config: Optional[dict[str, float]] = None,
    ):
        """
        Parameters
        ----------
        weight_config : dict, optional
            Custom weights for trust score components.
            Default: equal weighting.
        """
        self.weights = weight_config or {
            "brier_score": 0.30,
            "log_loss": 0.20,
            "ece": 0.25,
            "volume": 0.10,
            "recency": 0.10,
            "resolution_clarity": 0.05,
        }

    def explain(self, calibration_result: dict[str, Any]) -> dict[str, Any]:
        """
        Generate trust breakdown explanation from calibration results.

        Parameters
        ----------
        calibration_result : dict
            Market Calibration Agent output containing:
              - trust_score: float (overall)
              - metrics: {brier_score, log_loss, ece, ...}
              - signals: list of signal sources
              - market_id: str (optional)

        Returns
        -------
        dict with:
            - trust_scores: {signal_name: score}
            - breakdowns: {signal_name: {component: {value, weight, contribution, assessment}}}
            - why_trusted: {signal_name: natural language explanation}
            - recommendations: list of improvement suggestions
        """
        trust_scores = {}
        breakdowns = {}
        why_trusted = {}
        recommendations = []

        # Handle single signal or multiple signals
        signals = calibration_result.get("signals", [])
        if not signals:
            # Single signal mode
            signal_name = calibration_result.get("source", "polymarket")
            signals = [{
                "name": signal_name,
                "trust_score": calibration_result.get("trust_score", 0.0),
                "metrics": calibration_result.get("metrics", {}),
            }]

        for signal in signals:
            name = signal.get("name", "unknown")
            score = signal.get("trust_score", 0.0)
            metrics = signal.get("metrics", {})

            trust_scores[name] = score

            # Component-level breakdown
            breakdown = self._decompose_trust_score(metrics)
            breakdowns[name] = breakdown

            # Natural language explanation
            why_trusted[name] = self._generate_trust_explanation(
                name, score, metrics, breakdown
            )

            # Recommendations
            recs = self._generate_recommendations(name, score, metrics, breakdown)
            recommendations.extend(recs)

        return {
            "trust_scores": trust_scores,
            "breakdowns": breakdowns,
            "why_trusted": why_trusted,
            "recommendations": recommendations,
            "trust_level_thresholds": {
                "high": self.HIGH_TRUST,
                "medium": self.MEDIUM_TRUST,
                "low": self.LOW_TRUST,
            },
        }

    def _decompose_trust_score(
        self, metrics: dict[str, float]
    ) -> dict[str, dict[str, Any]]:
        """Break down trust score into component contributions."""
        breakdown = {}

        for component, weight in self.weights.items():
            info = self.COMPONENT_INFO.get(component, {})
            value = metrics.get(component)

            if value is None:
                breakdown[component] = {
                    "value": None,
                    "weight": weight,
                    "contribution": 0.0,
                    "assessment": "not_available",
                    "description": info.get("description", ""),
                }
                continue

            # Assess quality
            good_threshold = info.get("good_threshold", 0.5)
            direction = info.get("direction", "lower_is_better")

            if direction == "lower_is_better":
                normalized = max(0, 1 - (value / (good_threshold * 3)))
            else:
                normalized = min(1, value / good_threshold)

            contribution = normalized * weight

            if normalized >= 0.8:
                assessment = "excellent"
            elif normalized >= 0.6:
                assessment = "good"
            elif normalized >= 0.4:
                assessment = "fair"
            else:
                assessment = "poor"

            breakdown[component] = {
                "value": round(value, 4),
                "weight": weight,
                "contribution": round(contribution, 4),
                "normalized_score": round(normalized, 4),
                "assessment": assessment,
                "description": info.get("description", ""),
                "name": info.get("name", component),
            }

        return breakdown

    def _generate_trust_explanation(
        self,
        signal_name: str,
        score: float,
        metrics: dict[str, float],
        breakdown: dict[str, dict[str, Any]],
    ) -> str:
        """Generate natural language trust explanation."""

        # Trust level
        if score >= self.HIGH_TRUST:
            level = "High trust"
        elif score >= self.MEDIUM_TRUST:
            level = "Moderate trust"
        elif score >= self.LOW_TRUST:
            level = "Low trust"
        else:
            level = "Very low trust"

        explanation = f"{level} ({score:.2f}). "

        # Highlight key metrics
        key_metrics = []
        for comp, info in breakdown.items():
            if info.get("value") is not None:
                assessment = info.get("assessment", "")
                name = info.get("name", comp)
                val = info.get("value", 0)
                if assessment in ("excellent", "good"):
                    key_metrics.append(f"{name} {val}")

        if key_metrics:
            explanation += f"Supported by: {', '.join(key_metrics[:3])}. "

        # Warnings
        warnings = []
        for comp, info in breakdown.items():
            if info.get("assessment") in ("poor", "fair"):
                warnings.append(info.get("name", comp))

        if warnings:
            explanation += f"Caution on: {', '.join(warnings)}."

        return explanation

    def _generate_recommendations(
        self,
        signal_name: str,
        score: float,
        metrics: dict[str, float],
        breakdown: dict[str, dict[str, Any]],
    ) -> list[dict[str, str]]:
        """Generate improvement recommendations."""
        recs = []

        for comp, info in breakdown.items():
            if info.get("assessment") == "poor":
                recs.append({
                    "signal": signal_name,
                    "component": info.get("name", comp),
                    "issue": f"{info.get('name', comp)} is poor ({info.get('value')})",
                    "suggestion": (
                        f"Consider reducing weight of {signal_name} signal "
                        f"or supplementing with additional sources"
                    ),
                })

        if score < self.LOW_TRUST:
            recs.append({
                "signal": signal_name,
                "component": "overall",
                "issue": f"Overall trust score very low ({score:.2f})",
                "suggestion": (
                    f"Signal {signal_name} should not be used as primary "
                    f"input without additional corroboration"
                ),
            })

        return recs
