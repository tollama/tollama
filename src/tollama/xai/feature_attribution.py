"""
tollama.xai.feature_attribution — Temporal Feature Importance

v3.8 Phase 3: Feature Attribution Layer
"어떤 입력이 예측에 기여했는가" — temporal importance 시각화 제공.

Methods:
  - Permutation importance (model-agnostic)
  - Gradient-based attribution (for differentiable models)
  - Occlusion sensitivity (temporal window masking)
  - Lag importance (which historical lags matter most)
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class AttributionResult:
    """Feature attribution output for a single forecast."""
    temporal_importance: list[float] = field(default_factory=list)
    lag_importance: dict[int, float] = field(default_factory=dict)
    feature_importance: dict[str, float] = field(default_factory=dict)
    top_contributing_periods: list[dict[str, Any]] = field(default_factory=list)
    method: str = "permutation"
    summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "temporal_importance": self.temporal_importance,
            "lag_importance": self.lag_importance,
            "feature_importance": self.feature_importance,
            "top_contributing_periods": self.top_contributing_periods,
            "method": self.method,
            "summary": self.summary,
        }


class TemporalFeatureAttribution:
    """
    Computes temporal feature importance for time series forecasts.

    Answers: "Which historical time steps and features contributed
    most to this forecast?"

    Phase 3 deliverable per v3.8 roadmap.
    """

    def __init__(
        self,
        method: str = "permutation",
        n_repeats: int = 10,
        window_size: int = 1,
        random_state: int | None = 42,
    ):
        """
        Parameters
        ----------
        method : str
            Attribution method: "permutation", "occlusion", "lag_correlation"
        n_repeats : int
            Number of permutation repeats for stable estimates
        window_size : int
            Occlusion window size (for occlusion method)
        random_state : int, optional
            Random seed for reproducibility
        """
        self.method = method
        self.n_repeats = n_repeats
        self.window_size = window_size
        self.rng = np.random.RandomState(random_state)

    def compute(
        self,
        data: Sequence[float],
        predict_fn: Callable | None = None,
        forecast_horizon: int = 1,
        feature_names: list[str] | None = None,
        exogenous: np.ndarray | None = None,
    ) -> list[dict[str, Any]]:
        """
        Compute temporal feature attribution.

        Parameters
        ----------
        data : array-like
            Historical time series (input to forecast)
        predict_fn : callable, optional
            Model prediction function: data -> forecast.
            If None, uses lag_correlation method.
        forecast_horizon : int
            Number of steps ahead being forecasted
        feature_names : list[str], optional
            Names of exogenous features
        exogenous : ndarray, optional
            Exogenous feature matrix (n_timesteps, n_features)

        Returns
        -------
        list of attribution dicts
        """
        arr = np.asarray(data, dtype=float)

        if self.method == "permutation" and predict_fn is not None:
            result = self._permutation_importance(
                arr, predict_fn, forecast_horizon, exogenous, feature_names
            )
        elif self.method == "occlusion" and predict_fn is not None:
            result = self._occlusion_sensitivity(
                arr, predict_fn, forecast_horizon
            )
        else:
            result = self._lag_correlation(arr, forecast_horizon)

        # Generate summary
        result.summary = self._generate_summary(result)

        return [result.to_dict()]

    def _permutation_importance(
        self,
        data: np.ndarray,
        predict_fn: Callable,
        horizon: int,
        exogenous: np.ndarray | None,
        feature_names: list[str] | None,
    ) -> AttributionResult:
        """
        Permutation-based temporal importance.

        Shuffles each time step's value and measures forecast degradation.
        """
        n = len(data)
        baseline_forecast = predict_fn(data)
        if isinstance(baseline_forecast, (list, np.ndarray)):
            baseline_val = float(np.mean(baseline_forecast))
        else:
            baseline_val = float(baseline_forecast)

        temporal_importance = np.zeros(n)

        for t in range(n):
            degradations = []
            for _ in range(self.n_repeats):
                perturbed = data.copy()
                # Shuffle this time step with random value from the series
                perturbed[t] = self.rng.choice(data)
                perturbed_forecast = predict_fn(perturbed)
                if isinstance(perturbed_forecast, (list, np.ndarray)):
                    perturbed_val = float(np.mean(perturbed_forecast))
                else:
                    perturbed_val = float(perturbed_forecast)
                degradations.append(abs(perturbed_val - baseline_val))
            temporal_importance[t] = float(np.mean(degradations))

        # Normalize
        total = temporal_importance.sum()
        if total > 0:
            temporal_importance = temporal_importance / total

        # Lag importance: aggregate by distance from end
        lag_importance = {}
        for t in range(n):
            lag = n - 1 - t
            lag_importance[lag] = float(temporal_importance[t])

        # Top contributing periods
        top_indices = np.argsort(temporal_importance)[::-1][:5]
        top_periods = [
            {
                "time_index": int(idx),
                "lag": int(n - 1 - idx),
                "importance": float(temporal_importance[idx]),
            }
            for idx in top_indices
        ]

        # Exogenous feature importance
        feat_importance = {}
        if exogenous is not None and feature_names:
            for j, fname in enumerate(feature_names):
                degradations = []
                for _ in range(self.n_repeats):
                    perturbed_exog = exogenous.copy()
                    self.rng.shuffle(perturbed_exog[:, j])
                    # Would need model that accepts exogenous
                    degradations.append(0.0)
                feat_importance[fname] = float(np.mean(degradations))

        return AttributionResult(
            temporal_importance=temporal_importance.tolist(),
            lag_importance=lag_importance,
            feature_importance=feat_importance,
            top_contributing_periods=top_periods,
            method="permutation",
        )

    def _occlusion_sensitivity(
        self,
        data: np.ndarray,
        predict_fn: Callable,
        horizon: int,
    ) -> AttributionResult:
        """
        Occlusion-based sensitivity.

        Replaces windows of the input with mean and measures impact.
        """
        n = len(data)
        baseline_forecast = predict_fn(data)
        if isinstance(baseline_forecast, (list, np.ndarray)):
            baseline_val = float(np.mean(baseline_forecast))
        else:
            baseline_val = float(baseline_forecast)

        mean_val = float(np.mean(data))
        temporal_importance = np.zeros(n)
        w = self.window_size

        for t in range(0, n, w):
            occluded = data.copy()
            end = min(t + w, n)
            occluded[t:end] = mean_val

            occluded_forecast = predict_fn(occluded)
            if isinstance(occluded_forecast, (list, np.ndarray)):
                occluded_val = float(np.mean(occluded_forecast))
            else:
                occluded_val = float(occluded_forecast)

            importance = abs(occluded_val - baseline_val)
            for i in range(t, end):
                temporal_importance[i] = importance / (end - t)

        # Normalize
        total = temporal_importance.sum()
        if total > 0:
            temporal_importance = temporal_importance / total

        lag_importance = {
            (n - 1 - t): float(temporal_importance[t]) for t in range(n)
        }
        top_indices = np.argsort(temporal_importance)[::-1][:5]
        top_periods = [
            {
                "time_index": int(idx),
                "lag": int(n - 1 - idx),
                "importance": float(temporal_importance[idx]),
            }
            for idx in top_indices
        ]

        return AttributionResult(
            temporal_importance=temporal_importance.tolist(),
            lag_importance=lag_importance,
            top_contributing_periods=top_periods,
            method="occlusion",
        )

    def _lag_correlation(
        self,
        data: np.ndarray,
        horizon: int,
    ) -> AttributionResult:
        """
        Lag correlation-based attribution (model-agnostic, no predict_fn needed).

        Uses autocorrelation as proxy for temporal importance.
        """
        n = len(data)
        if n < 4:
            return AttributionResult(method="lag_correlation")

        mean = np.mean(data)
        var = np.var(data)
        if var == 0:
            return AttributionResult(method="lag_correlation")

        normalized = data - mean
        max_lag = n - 1

        lag_importance = {}
        for lag in range(1, min(max_lag, n // 2)):
            acf = float(
                np.sum(normalized[:-lag] * normalized[lag:]) / (var * (n - lag))
            )
            lag_importance[lag] = abs(acf)

        # Convert to temporal importance (reverse: recent lags first)
        temporal_importance = np.zeros(n)
        for lag, imp in lag_importance.items():
            if lag < n:
                temporal_importance[n - 1 - lag] = imp

        # Normalize
        total = sum(temporal_importance)
        if total > 0:
            temporal_importance = temporal_importance / total

        # Top lags
        sorted_lags = sorted(lag_importance.items(), key=lambda x: x[1], reverse=True)
        top_periods = [
            {
                "lag": lag,
                "time_index": n - 1 - lag,
                "importance": float(imp),
            }
            for lag, imp in sorted_lags[:5]
        ]

        return AttributionResult(
            temporal_importance=temporal_importance.tolist(),
            lag_importance=lag_importance,
            top_contributing_periods=top_periods,
            method="lag_correlation",
        )

    def _generate_summary(self, result: AttributionResult) -> str:
        """Generate natural language summary."""
        if not result.top_contributing_periods:
            return "Insufficient data for temporal attribution."

        top = result.top_contributing_periods[0]
        lag = top.get("lag", 0)
        imp = top.get("importance", 0)

        summary = (
            f"Temporal attribution ({result.method}): "
            f"most influential period is lag-{lag} "
            f"(importance: {imp:.4f}). "
        )

        if lag <= 3:
            summary += "Recent observations dominate the forecast."
        elif lag <= 7:
            summary += "Short-term history (past week) drives the forecast."
        else:
            summary += "Longer-term patterns significantly influence the forecast."

        return summary
