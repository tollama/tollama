"""Forecast accuracy metrics calculated against per-series actuals."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

from tollama.core.schemas import (
    ForecastMetrics,
    ForecastRequest,
    ForecastResponse,
    MetricName,
    MetricsParameters,
    SeriesForecast,
    SeriesInput,
    SeriesMetrics,
)


@dataclass(frozen=True)
class _SeriesMetricContext:
    """Prepared values used by metric calculators for one series."""

    request_series: SeriesInput
    response_series: SeriesForecast
    actuals: list[float]
    predictions: list[float]


_MetricCalculator = Callable[
    [_SeriesMetricContext, MetricsParameters],
    tuple[float | None, str | None],
]


def compute_forecast_metrics(
    *,
    request: ForecastRequest,
    response: ForecastResponse,
) -> tuple[ForecastMetrics | None, list[str]]:
    """Compute requested metrics with best-effort handling for undefined values."""
    metrics_parameters = request.parameters.metrics
    if metrics_parameters is None:
        return None, []

    forecasts_by_id = {forecast.id: forecast for forecast in response.forecasts}
    warnings: list[str] = []
    per_series: list[SeriesMetrics] = []
    aggregate_values: dict[str, list[float]] = {name: [] for name in metrics_parameters.names}

    for series in request.series:
        forecast = forecasts_by_id.get(series.id)
        if forecast is None:
            warnings.append(
                f"metrics skipped for series {series.id!r}: missing matching forecast output",
            )
            continue

        context, context_warnings = _build_series_context(series=series, forecast=forecast)
        warnings.extend(context_warnings)
        if context is None:
            continue

        series_values: dict[str, float] = {}
        for metric_name in metrics_parameters.names:
            calculator = _METRIC_REGISTRY[metric_name]
            value, warning = calculator(context, metrics_parameters)
            if warning:
                warnings.append(warning)
            if value is None:
                continue
            series_values[metric_name] = value
            aggregate_values[metric_name].append(value)

        if series_values:
            per_series.append(
                SeriesMetrics(
                    id=series.id,
                    values=series_values,
                ),
            )

    aggregate = {
        metric_name: sum(values) / len(values)
        for metric_name, values in aggregate_values.items()
        if values
    }

    if not per_series or not aggregate:
        return None, warnings
    return ForecastMetrics(aggregate=aggregate, series=per_series), warnings


def _build_series_context(
    *,
    series: SeriesInput,
    forecast: SeriesForecast,
) -> tuple[_SeriesMetricContext | None, list[str]]:
    warnings: list[str] = []
    if series.actuals is None:
        warnings.append(
            f"metrics skipped for series {series.id!r}: actuals are required for evaluation",
        )
        return None, warnings

    actuals = [float(value) for value in series.actuals]
    predictions = [float(value) for value in forecast.mean]

    if len(predictions) != len(actuals):
        overlap = min(len(predictions), len(actuals))
        if overlap <= 0:
            warnings.append(
                "metrics skipped for series "
                f"{series.id!r}: no overlapping forecast and actual values",
            )
            return None, warnings
        warnings.append(
            f"metrics evaluation for series {series.id!r} uses {overlap} points because "
            f"forecast mean length ({len(predictions)}) != actuals length ({len(actuals)})",
        )
        actuals = actuals[:overlap]
        predictions = predictions[:overlap]

    return (
        _SeriesMetricContext(
            request_series=series,
            response_series=forecast,
            actuals=actuals,
            predictions=predictions,
        ),
        warnings,
    )


def _compute_mape(
    context: _SeriesMetricContext,
    _: MetricsParameters,
) -> tuple[float | None, str | None]:
    errors = [
        abs((actual - prediction) / actual)
        for actual, prediction in zip(context.actuals, context.predictions, strict=True)
        if actual != 0.0
    ]
    if not errors:
        return (
            None,
            f"metrics.mape skipped for series {context.request_series.id!r}: "
            "all actual values are zero",
        )
    value = sum(errors) / len(errors) * 100.0
    if not math.isfinite(value):
        return (
            None,
            f"metrics.mape skipped for series {context.request_series.id!r}: non-finite value",
        )
    return value, None


def _compute_mase(
    context: _SeriesMetricContext,
    parameters: MetricsParameters,
) -> tuple[float | None, str | None]:
    seasonality = parameters.mase_seasonality
    history = [float(value) for value in context.request_series.target]
    if len(history) <= seasonality:
        return (
            None,
            f"metrics.mase skipped for series {context.request_series.id!r}: "
            f"target length must be greater than mase_seasonality ({seasonality})",
        )

    naive_errors = [
        abs(history[index] - history[index - seasonality])
        for index in range(seasonality, len(history))
    ]
    scale = sum(naive_errors) / len(naive_errors)
    if scale == 0.0:
        return (
            None,
            f"metrics.mase skipped for series {context.request_series.id!r}: "
            "seasonal naive denominator is zero",
        )

    absolute_errors = [
        abs(actual - prediction)
        for actual, prediction in zip(context.actuals, context.predictions, strict=True)
    ]
    mae = sum(absolute_errors) / len(absolute_errors)
    value = mae / scale
    if not math.isfinite(value):
        return (
            None,
            f"metrics.mase skipped for series {context.request_series.id!r}: non-finite value",
        )
    return value, None


_METRIC_REGISTRY: dict[MetricName, _MetricCalculator] = {
    "mape": _compute_mape,
    "mase": _compute_mase,
}
