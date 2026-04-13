"""
tollama.xai.forecast_decompose — Forecast Decomposition

v3.8 Phase 3: Forecast Decomposition (trend/seasonal/residual)
분해 결과를 통해 "예측이 왜 이렇게 나왔는가"를 구성요소별로 설명.

Phase 2a에서 기존 기능 재포장으로 시작, Phase 3에서 완성.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DecompositionResult:
    """Forecast decomposition output."""

    trend: list[float] = field(default_factory=list)
    seasonal: list[float] = field(default_factory=list)
    residual: list[float] = field(default_factory=list)
    trend_strength: float = 0.0
    seasonal_strength: float = 0.0
    residual_ratio: float = 0.0
    period: int = 0
    method: str = "stl"
    summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "trend": self.trend,
            "seasonal": self.seasonal,
            "residual": self.residual,
            "trend_strength": round(self.trend_strength, 4),
            "seasonal_strength": round(self.seasonal_strength, 4),
            "residual_ratio": round(self.residual_ratio, 4),
            "period": self.period,
            "method": self.method,
            "summary": self.summary,
        }

    def to_proportions(self) -> dict[str, float]:
        """Return proportion of variance explained by each component."""
        total = self.trend_strength + self.seasonal_strength + self.residual_ratio
        if total == 0:
            return {"trend": 0.0, "seasonal": 0.0, "residual": 0.0}
        return {
            "trend": round(self.trend_strength / total, 4),
            "seasonal": round(self.seasonal_strength / total, 4),
            "residual": round(self.residual_ratio / total, 4),
        }


class ForecastDecomposer:
    """
    Decomposes time series into trend, seasonal, and residual components
    for explainability.

    Methods:
    - STL (Seasonal and Trend decomposition using Loess)
    - Classical additive/multiplicative decomposition
    - Moving average trend extraction

    The decomposition results feed into the Explanation Engine to answer
    "what proportion of the forecast is driven by trend vs seasonality vs noise?"
    """

    def __init__(
        self,
        method: str = "stl",
        period: int | None = None,
        robust: bool = True,
    ):
        """
        Parameters
        ----------
        method : str
            Decomposition method: "stl", "classical_additive",
            "classical_multiplicative", "moving_average"
        period : int, optional
            Seasonal period. Auto-detected if None.
        robust : bool
            Use robust fitting (less sensitive to outliers)
        """
        self.method = method
        self.period = period
        self.robust = robust

    def decompose(
        self,
        data: Sequence[float],
        period: int | None = None,
    ) -> dict[str, Any]:
        """
        Decompose time series and return explanation-ready result.

        Parameters
        ----------
        data : array-like
            Time series values
        period : int, optional
            Override seasonal period

        Returns
        -------
        dict with decomposition components and proportions
        """
        arr = np.asarray(data, dtype=float)
        if len(arr) < 4:
            return {"error": "Series too short for decomposition", "min_length": 4}

        p = period or self.period or self._detect_period(arr)

        if self.method == "stl":
            result = self._stl_decompose(arr, p)
        elif self.method == "classical_additive":
            result = self._classical_decompose(arr, p, model="additive")
        elif self.method == "classical_multiplicative":
            result = self._classical_decompose(arr, p, model="multiplicative")
        else:
            result = self._moving_average_decompose(arr, p)

        # Generate summary
        proportions = result.to_proportions()
        dominant = max(proportions, key=proportions.get)
        result.summary = self._generate_summary(result, proportions, dominant)

        return result.to_dict()

    def _detect_period(self, data: np.ndarray) -> int:
        """Auto-detect seasonal period using autocorrelation."""
        n = len(data)
        if n < 10:
            return 1

        # Compute autocorrelation
        mean = np.mean(data)
        var = np.var(data)
        if var == 0:
            return 1

        normalized = data - mean
        max_lag = min(n // 2, 365)
        acf = np.correlate(normalized, normalized, mode="full")
        acf = acf[n - 1 : n - 1 + max_lag] / (var * n)

        # Find first significant peak after lag 1
        if len(acf) < 3:
            return 1

        peaks = []
        for i in range(2, len(acf) - 1):
            if acf[i] > acf[i - 1] and acf[i] > acf[i + 1] and acf[i] > 0.1:
                peaks.append((i, acf[i]))

        if peaks:
            return peaks[0][0]  # First significant peak

        # Common defaults
        if n >= 365:
            return 7  # Weekly for daily data
        elif n >= 52:
            return 12  # Monthly for weekly data
        return 1

    def _stl_decompose(self, data: np.ndarray, period: int) -> DecompositionResult:
        """STL decomposition."""
        try:
            from statsmodels.tsa.seasonal import STL

            stl = STL(data, period=max(period, 2), robust=self.robust)
            res = stl.fit()
            trend = res.trend
            seasonal = res.seasonal
            residual = res.resid
        except ImportError:
            logger.warning("statsmodels not available, falling back to moving average")
            return self._moving_average_decompose(data, period)

        return self._build_result(data, trend, seasonal, residual, period, "stl")

    def _classical_decompose(
        self, data: np.ndarray, period: int, model: str = "additive"
    ) -> DecompositionResult:
        """Classical decomposition."""
        try:
            from statsmodels.tsa.seasonal import seasonal_decompose

            res = seasonal_decompose(data, model=model, period=max(period, 2))
            trend = np.nan_to_num(res.trend, nan=0.0)
            seasonal = np.nan_to_num(res.seasonal, nan=0.0)
            residual = np.nan_to_num(res.resid, nan=0.0)
        except ImportError:
            return self._moving_average_decompose(data, period)

        method = f"classical_{model}"
        return self._build_result(data, trend, seasonal, residual, period, method)

    def _moving_average_decompose(self, data: np.ndarray, period: int) -> DecompositionResult:
        """Simple moving average decomposition (no external deps)."""
        n = len(data)
        p = max(period, 2)

        # Trend via moving average
        if n >= p:
            kernel = np.ones(p) / p
            trend = np.convolve(data, kernel, mode="same")
        else:
            trend = np.full(n, np.mean(data))

        # Detrended
        detrended = data - trend

        # Seasonal: average of detrended values at each position in cycle
        seasonal = np.zeros(n)
        if p > 1:
            for i in range(p):
                indices = range(i, n, p)
                seasonal_mean = np.mean(detrended[list(indices)])
                for idx in indices:
                    seasonal[idx] = seasonal_mean

        # Residual
        residual = data - trend - seasonal

        return self._build_result(data, trend, seasonal, residual, period, "moving_average")

    def _build_result(
        self,
        data: np.ndarray,
        trend: np.ndarray,
        seasonal: np.ndarray,
        residual: np.ndarray,
        period: int,
        method: str,
    ) -> DecompositionResult:
        """Build DecompositionResult with strength metrics."""
        total_var = np.var(data)
        if total_var == 0:
            total_var = 1.0

        trend_var = np.var(trend)
        seasonal_var = np.var(seasonal)
        residual_var = np.var(residual)

        return DecompositionResult(
            trend=trend.tolist(),
            seasonal=seasonal.tolist(),
            residual=residual.tolist(),
            trend_strength=float(trend_var / total_var),
            seasonal_strength=float(seasonal_var / total_var),
            residual_ratio=float(residual_var / total_var),
            period=period,
            method=method,
        )

    def _generate_summary(
        self,
        result: DecompositionResult,
        proportions: dict[str, float],
        dominant: str,
    ) -> str:
        """Generate natural language summary of decomposition."""
        trend_pct = proportions["trend"] * 100
        seasonal_pct = proportions["seasonal"] * 100
        residual_pct = proportions["residual"] * 100

        summary = (
            f"Forecast decomposition ({result.method}): "
            f"trend explains {trend_pct:.1f}% of variance, "
            f"seasonality {seasonal_pct:.1f}%, "
            f"residual noise {residual_pct:.1f}%. "
        )

        if dominant == "trend":
            summary += "The series is primarily trend-driven."
        elif dominant == "seasonal":
            summary += f"Strong seasonal pattern detected (period={result.period})."
        else:
            summary += "High residual variance suggests noisy or regime-changing data."

        return summary
