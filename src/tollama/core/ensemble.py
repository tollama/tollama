"""Helpers for combining multiple forecast responses into one ensemble output."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Literal

from .schemas import ForecastResponse, SeriesForecast

EnsembleMethod = Literal["mean", "median", "trimmed_mean", "winsorized_mean"]

logger = logging.getLogger(__name__)


class EnsembleError(ValueError):
    """Raised when ensemble merge input is inconsistent."""


def merge_forecast_responses(
    responses: Sequence[ForecastResponse],
    *,
    weights: Mapping[str, float] | None = None,
    method: EnsembleMethod = "mean",
    model_name: str = "ensemble",
    partial_ok: bool = False,
    min_models: int = 1,
) -> ForecastResponse:
    """Combine multiple forecast responses using weighted mean/median by step.

    Args:
        responses: Forecast responses from individual models.
        weights: Optional per-model weight mapping.
        method: Aggregation method.
        model_name: Name for the merged response.
        partial_ok: If True, skip models with mismatched series/horizon instead
            of raising. Requires at least ``min_models`` to succeed.
        min_models: Minimum number of models that must contribute to each series
            when ``partial_ok`` is True.
    """
    if not responses:
        raise EnsembleError("cannot merge empty ensemble responses")

    base = responses[0]
    merged_forecasts: list[SeriesForecast] = []
    component_warnings: list[str] = []

    for response in responses:
        if response.warnings:
            component_warnings.extend(
                f"{response.model}: {warning}" for warning in response.warnings
            )

    for series_index, base_series in enumerate(base.forecasts):
        horizon = len(base_series.mean)
        values_by_step: list[list[float]] = [[] for _ in range(horizon)]
        weights_by_step: list[list[float]] = [[] for _ in range(horizon)]

        for response in responses:
            # --- series count mismatch ---
            if series_index >= len(response.forecasts):
                if partial_ok:
                    component_warnings.append(
                        f"{response.model}: missing series index {series_index}, skipped"
                    )
                    continue
                raise EnsembleError(
                    "ensemble merge failed: model "
                    f"{response.model!r} returned mismatched series count",
                )

            candidate_series = response.forecasts[series_index]

            # --- series id mismatch ---
            if candidate_series.id != base_series.id:
                if partial_ok:
                    component_warnings.append(
                        f"{response.model}: series id mismatch "
                        f"({candidate_series.id!r} vs {base_series.id!r}), skipped"
                    )
                    continue
                raise EnsembleError(
                    "ensemble merge failed: model "
                    f"{response.model!r} returned series id {candidate_series.id!r} "
                    f"but expected {base_series.id!r}",
                )

            # --- horizon mismatch ---
            candidate_horizon = len(candidate_series.mean)
            if candidate_horizon != horizon:
                if partial_ok:
                    component_warnings.append(
                        f"{response.model}: horizon mismatch "
                        f"({candidate_horizon} vs {horizon}), skipped"
                    )
                    continue
                raise EnsembleError(
                    "ensemble merge failed: model "
                    f"{response.model!r} returned horizon {candidate_horizon} "
                    f"but expected {horizon}",
                )

            raw_weight = 1.0
            if weights is not None:
                raw_weight = float(weights.get(response.model, 1.0))
            weight = raw_weight if raw_weight > 0.0 else 1.0

            for step_index, value in enumerate(candidate_series.mean):
                values_by_step[step_index].append(float(value))
                weights_by_step[step_index].append(weight)

        # Check min_models constraint
        contributor_count = len(values_by_step[0]) if values_by_step and values_by_step[0] else 0
        if contributor_count < min_models:
            raise EnsembleError(
                f"ensemble merge failed: only {contributor_count} model(s) contributed "
                f"to series {base_series.id!r} but min_models={min_models}"
            )

        merged_mean: list[float] = []
        for step_values, step_weights in zip(values_by_step, weights_by_step, strict=True):
            merged_mean.append(
                _merge_step(
                    values=step_values,
                    weights=step_weights,
                    method=method,
                )
            )

        merged_forecasts.append(
            SeriesForecast(
                id=base_series.id,
                freq=base_series.freq,
                start_timestamp=base_series.start_timestamp,
                mean=[round(value, 8) for value in merged_mean],
                quantiles=None,
            ),
        )

    return ForecastResponse(
        model=model_name,
        forecasts=merged_forecasts,
        warnings=component_warnings or None,
    )


def _merge_step(*, values: list[float], weights: list[float], method: EnsembleMethod) -> float:
    if not values:
        raise EnsembleError("ensemble merge failed: no values available for step")

    if method == "mean":
        return _weighted_mean(values=values, weights=weights)

    if method == "median":
        return _weighted_median(values=values, weights=weights)

    if method == "trimmed_mean":
        return _trimmed_mean(values=values, weights=weights, trim_fraction=0.1)

    if method == "winsorized_mean":
        return _winsorized_mean(values=values, weights=weights, limit_fraction=0.1)

    # Unreachable for valid EnsembleMethod values but keeps mypy happy.
    return _weighted_mean(values=values, weights=weights)


def _weighted_mean(*, values: list[float], weights: list[float]) -> float:
    total_weight = sum(weights)
    if total_weight <= 0.0:
        raise EnsembleError("ensemble merge failed due to invalid weights")
    weighted_sum = sum(value * weight for value, weight in zip(values, weights, strict=True))
    return weighted_sum / total_weight


def _trimmed_mean(
    *,
    values: list[float],
    weights: list[float],
    trim_fraction: float = 0.1,
) -> float:
    """Compute weighted mean after dropping the top/bottom ``trim_fraction`` of values."""
    if len(values) < 3:
        # Not enough values to trim — fall back to weighted mean.
        return _weighted_mean(values=values, weights=weights)

    ordered = sorted(zip(values, weights, strict=True), key=lambda item: item[0])
    n = len(ordered)
    trim_count = max(1, int(n * trim_fraction))
    trimmed = ordered[trim_count : n - trim_count]

    if not trimmed:
        # Edge case: everything trimmed — fall back to median
        return _weighted_median(values=values, weights=weights)

    total_weight = sum(w for _, w in trimmed)
    if total_weight <= 0.0:
        raise EnsembleError("ensemble merge failed due to invalid weights after trimming")
    return sum(v * w for v, w in trimmed) / total_weight


def _winsorized_mean(
    *,
    values: list[float],
    weights: list[float],
    limit_fraction: float = 0.1,
) -> float:
    """Compute weighted mean after clipping outlier values to boundary percentiles."""
    if len(values) < 3:
        return _weighted_mean(values=values, weights=weights)

    sorted_values = sorted(values)
    n = len(sorted_values)
    k = max(1, int(n * limit_fraction))
    low = sorted_values[k]
    high = sorted_values[n - k - 1]

    clipped = [max(low, min(high, v)) for v in values]
    return _weighted_mean(values=clipped, weights=weights)


def _weighted_median(*, values: list[float], weights: list[float]) -> float:
    ordered = sorted(zip(values, weights, strict=True), key=lambda item: item[0])
    total_weight = sum(weight for _, weight in ordered)
    if total_weight <= 0.0:
        raise EnsembleError("ensemble merge failed due to invalid weights")

    threshold = total_weight / 2.0
    cumulative = 0.0
    for value, weight in ordered:
        cumulative += weight
        if cumulative >= threshold:
            return value

    return ordered[-1][0]
