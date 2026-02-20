"""Deterministic counterfactual generation for intervention analysis."""

from __future__ import annotations

from collections.abc import Callable
from statistics import fmean

from tollama.core.schemas import (
    CounterfactualDirection,
    CounterfactualRequest,
    CounterfactualResponse,
    CounterfactualSeriesResult,
    ForecastRequest,
    ForecastResponse,
    SeriesInput,
)

ForecastExecutor = Callable[[ForecastRequest], ForecastResponse]


def generate_counterfactual(
    *,
    payload: CounterfactualRequest,
    forecast_executor: ForecastExecutor,
) -> CounterfactualResponse:
    """Generate intervention counterfactuals and compare against observed values."""
    horizon = _post_horizon(payload=payload)
    if horizon <= 0:
        raise ValueError("counterfactual horizon must be positive")

    baseline_series: list[SeriesInput] = []
    actual_by_id: dict[str, list[float]] = {}
    warnings: list[str] = []
    for series in payload.series:
        prepared, actual_values, series_warnings = _prepare_series(
            series=series,
            intervention_index=payload.intervention_index,
            horizon=horizon,
        )
        baseline_series.append(prepared)
        actual_by_id[series.id] = actual_values
        warnings.extend(series_warnings)

    baseline_request = ForecastRequest(
        model=payload.model,
        horizon=horizon,
        quantiles=payload.quantiles,
        series=baseline_series,
        options=payload.options,
        timeout=payload.timeout,
        parameters=payload.parameters.model_copy(update={"metrics": None}),
        response_options=payload.response_options,
    )
    baseline_response = forecast_executor(baseline_request)

    forecasts_by_id = {item.id: item for item in baseline_response.forecasts}
    results: list[CounterfactualSeriesResult] = []
    for series in payload.series:
        forecast = forecasts_by_id.get(series.id)
        if forecast is None:
            warnings.append(f"missing baseline forecast for series {series.id!r}; skipped")
            continue

        actual = actual_by_id[series.id]
        counterfactual = [float(value) for value in forecast.mean]
        actual, counterfactual, truncate_warning = _aligned_series(
            series_id=series.id,
            actual=actual,
            counterfactual=counterfactual,
        )
        if truncate_warning is not None:
            warnings.append(truncate_warning)
        if not actual:
            warnings.append(f"no post-intervention points available for series {series.id!r}")
            continue

        delta = [
            observed - expected
            for observed, expected in zip(actual, counterfactual, strict=True)
        ]
        absolute_delta = [abs(value) for value in delta]
        total_delta = float(sum(delta))
        mean_absolute_delta = float(sum(absolute_delta) / len(absolute_delta))
        mean_actual = float(fmean(actual))
        mean_delta = float(fmean(delta))
        if abs(mean_actual) <= 1e-9:
            average_delta_pct = None
        else:
            average_delta_pct = float((mean_delta / abs(mean_actual)) * 100.0)

        results.append(
            CounterfactualSeriesResult(
                id=series.id,
                actual=actual,
                counterfactual=counterfactual,
                delta=delta,
                absolute_delta=absolute_delta,
                mean_absolute_delta=mean_absolute_delta,
                total_delta=total_delta,
                average_delta_pct=average_delta_pct,
                direction=_counterfactual_direction(total_delta=total_delta),
            )
        )

    if not results:
        raise ValueError("counterfactual execution produced no comparable series")

    return CounterfactualResponse(
        model=payload.model,
        horizon=horizon,
        intervention_index=payload.intervention_index,
        intervention_label=payload.intervention_label,
        baseline=baseline_response,
        results=results,
        warnings=_dedupe(warnings) or None,
    )


def _post_horizon(*, payload: CounterfactualRequest) -> int:
    first = payload.series[0]
    return len(first.target) - payload.intervention_index


def _prepare_series(
    *,
    series: SeriesInput,
    intervention_index: int,
    horizon: int,
) -> tuple[SeriesInput, list[float], list[str]]:
    warnings: list[str] = []
    past_covariates = {
        name: list(values[:intervention_index])
        for name, values in (series.past_covariates or {}).items()
    }
    future_covariates: dict[str, list[int | float | str]] = {}
    for name, values in (series.future_covariates or {}).items():
        if len(values) < horizon:
            warnings.append(
                f"dropped future covariate {name!r} for series {series.id!r}: "
                f"expected at least {horizon} values",
            )
            continue
        future_covariates[name] = list(values[:horizon])

    for name in sorted(set(future_covariates) - set(past_covariates)):
        del future_covariates[name]
        warnings.append(
            f"dropped future covariate {name!r} for series {series.id!r}: "
            "missing matching past_covariates key",
        )

    prepared = series.model_copy(
        update={
            "timestamps": list(series.timestamps[:intervention_index]),
            "target": list(series.target[:intervention_index]),
            "actuals": None,
            "past_covariates": past_covariates or None,
            "future_covariates": future_covariates or None,
        }
    )
    actual_values = [float(value) for value in series.target[intervention_index:]]
    return prepared, actual_values, warnings


def _aligned_series(
    *,
    series_id: str,
    actual: list[float],
    counterfactual: list[float],
) -> tuple[list[float], list[float], str | None]:
    if len(actual) == len(counterfactual):
        return actual, counterfactual, None

    length = min(len(actual), len(counterfactual))
    warning = (
        f"truncated mismatch for series {series_id!r}: "
        f"actual={len(actual)} counterfactual={len(counterfactual)}"
    )
    return actual[:length], counterfactual[:length], warning


def _counterfactual_direction(*, total_delta: float) -> CounterfactualDirection:
    if total_delta > 1e-9:
        return "above_counterfactual"
    if total_delta < -1e-9:
        return "below_counterfactual"
    return "neutral"


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered
