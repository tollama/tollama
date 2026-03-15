"""
tollama.xai.trust_agents.calibration — Learned calibration tracker for trust agents.
"""

from __future__ import annotations

import math
from collections import deque
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class CalibrationRecord(BaseModel):
    """A single observation pairing a trust prediction with its actual outcome."""

    agent_name: str = Field(min_length=1)
    domain: str = Field(min_length=1)
    predicted_score: float = Field(ge=0.0, le=1.0)
    actual_outcome: float = Field(ge=0.0, le=1.0)
    component_scores: dict[str, float] = Field(default_factory=dict)
    recorded_at: str = Field(default_factory=_utc_now_iso)


class CalibrationStats(BaseModel):
    """Summary statistics for an agent's calibration history."""

    agent_name: str = Field(min_length=1)
    record_count: int = Field(ge=0)
    mean_bias: float = 0.0
    ece: float = 0.0
    adjustment_factors: dict[str, float] = Field(default_factory=dict)


class CalibrationTracker:
    """In-memory tracker that learns weight adjustments from outcome feedback.

    Stores a sliding window of CalibrationRecord per agent and computes
    component-level weight adjustments based on correlation with residuals.
    """

    MIN_RECORDS = 5

    def __init__(self, window_size: int = 100) -> None:
        self._window_size = window_size
        self._records: dict[str, deque[CalibrationRecord]] = {}

    def record(
        self,
        agent_name: str,
        domain: str,
        predicted_score: float,
        actual_outcome: float,
        component_scores: dict[str, float],
    ) -> None:
        """Add an observation for a given agent."""
        rec = CalibrationRecord(
            agent_name=agent_name,
            domain=domain,
            predicted_score=predicted_score,
            actual_outcome=actual_outcome,
            component_scores=component_scores,
        )
        if agent_name not in self._records:
            self._records[agent_name] = deque(maxlen=self._window_size)
        self._records[agent_name].append(rec)

    def get_weight_adjustments(self, agent_name: str) -> dict[str, float]:
        """Return per-component weight multipliers based on calibration history.

        Returns an empty dict if fewer than MIN_RECORDS observations exist.
        """
        records = list(self._records.get(agent_name, []))
        if len(records) < self.MIN_RECORDS:
            return {}
        return self._compute_adjustments(records)

    def get_calibration_stats(self, agent_name: str) -> CalibrationStats:
        """Compute summary calibration statistics for an agent."""
        records = list(self._records.get(agent_name, []))
        if not records:
            return CalibrationStats(agent_name=agent_name, record_count=0)

        residuals = [r.predicted_score - r.actual_outcome for r in records]
        mean_bias = sum(residuals) / len(residuals)
        ece = self._compute_ece(records)
        adjustments = (
            self._compute_adjustments(records) if len(records) >= self.MIN_RECORDS else {}
        )

        return CalibrationStats(
            agent_name=agent_name,
            record_count=len(records),
            mean_bias=mean_bias,
            ece=ece,
            adjustment_factors=adjustments,
        )

    @property
    def agents(self) -> list[str]:
        """Return agent names with recorded data."""
        return list(self._records.keys())

    @staticmethod
    def _compute_adjustments(records: list[CalibrationRecord]) -> dict[str, float]:
        """Compute per-component weight adjustment factors.

        For each component, correlate its score with the residual
        (actual_outcome - predicted_score). Positive correlation means the
        component predicts under-estimation, so boost its weight.
        """
        residuals = [r.actual_outcome - r.predicted_score for r in records]
        component_names: set[str] = set()
        for r in records:
            component_names.update(r.component_scores.keys())

        adjustments: dict[str, float] = {}
        for name in sorted(component_names):
            scores = [r.component_scores.get(name, 0.0) for r in records]
            corr = _pearson_correlation(scores, residuals)
            adjustments[name] = max(0.5, min(1.5, 1.0 + corr))
        return adjustments

    @staticmethod
    def _compute_ece(records: list[CalibrationRecord], n_bins: int = 10) -> float:
        """Compute Expected Calibration Error (ECE).

        Bins predictions into n_bins buckets and computes the weighted average
        of |mean_predicted - mean_actual| per bin.
        """
        if not records:
            return 0.0

        bins: list[list[CalibrationRecord]] = [[] for _ in range(n_bins)]
        for r in records:
            idx = min(int(r.predicted_score * n_bins), n_bins - 1)
            bins[idx].append(r)

        total = len(records)
        ece = 0.0
        for bin_records in bins:
            if not bin_records:
                continue
            avg_pred = sum(r.predicted_score for r in bin_records) / len(bin_records)
            avg_actual = sum(r.actual_outcome for r in bin_records) / len(bin_records)
            ece += (len(bin_records) / total) * abs(avg_pred - avg_actual)
        return ece


def _pearson_correlation(x: list[float], y: list[float]) -> float:
    """Compute Pearson correlation coefficient between two lists."""
    n = len(x)
    if n < 2:
        return 0.0
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    cov = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    var_x = sum((xi - mean_x) ** 2 for xi in x)
    var_y = sum((yi - mean_y) ** 2 for yi in y)
    denom = math.sqrt(var_x * var_y)
    if denom == 0.0:
        return 0.0
    return cov / denom


__all__ = [
    "CalibrationRecord",
    "CalibrationStats",
    "CalibrationTracker",
]
