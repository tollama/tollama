"""Helpers for combining multiple forecast responses into one ensemble output."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal

from .schemas import ForecastResponse, SeriesForecast

EnsembleMethod = Literal["mean", "median"]


class EnsembleError(ValueError):
    """Raised when ensemble merge input is inconsistent."""


def merge_forecast_responses(
    responses: Sequence[ForecastResponse],
    *,
    weights: Mapping[str, float] | None = None,
    method: EnsembleMethod = "mean",
    model_name: str = "ensemble",
) -> ForecastResponse:
    """Combine multiple forecast responses using weighted mean/median by step."""
    if not responses:
        raise EnsembleError("cannot merge empty ensemble responses")

    base = responses[0]
    merged_forecasts: list[SeriesForecast] = []
    component_warnings: list[str] = []

    for response in responses:
        if response.warnings:
            component_warnings.extend(
                f"{response.model}: {warning}"
                for warning in response.warnings
            )

    for series_index, base_series in enumerate(base.forecasts):
        horizon = len(base_series.mean)
        values_by_step: list[list[float]] = [[] for _ in range(horizon)]
        weights_by_step: list[list[float]] = [[] for _ in range(horizon)]

        for response in responses:
            candidate_series = _series_at_index(response=response, index=series_index)
            if candidate_series.id != base_series.id:
                raise EnsembleError(
                    "ensemble merge failed: model "
                    f"{response.model!r} returned series id {candidate_series.id!r} "
                    f"but expected {base_series.id!r}",
                )

            candidate_horizon = len(candidate_series.mean)
            if candidate_horizon != horizon:
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


def _series_at_index(*, response: ForecastResponse, index: int) -> SeriesForecast:
    if index >= len(response.forecasts):
        raise EnsembleError(
            "ensemble merge failed: model "
            f"{response.model!r} returned mismatched series count",
        )
    return response.forecasts[index]


def _merge_step(*, values: list[float], weights: list[float], method: EnsembleMethod) -> float:
    if not values:
        raise EnsembleError("ensemble merge failed: no values available for step")

    if method == "mean":
        total_weight = sum(weights)
        if total_weight <= 0.0:
            raise EnsembleError("ensemble merge failed due to invalid weights")
        weighted_sum = sum(value * weight for value, weight in zip(values, weights, strict=True))
        return weighted_sum / total_weight

    return _weighted_median(values=values, weights=weights)


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
