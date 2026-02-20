"""Deterministic forecast explainability helpers."""

from __future__ import annotations

from tollama.core.schemas import (
    ForecastExplanation,
    ForecastRequest,
    ForecastResponse,
    SeriesForecast,
    SeriesForecastExplanation,
    SeriesInput,
    TrendDirection,
)


def generate_explanation(
    *,
    request: ForecastRequest,
    response: ForecastResponse,
) -> ForecastExplanation | None:
    """Generate a lightweight, deterministic explanation for forecast outputs."""
    request_series_by_id = {series.id: series for series in request.series}
    series_payloads: list[SeriesForecastExplanation] = []

    for forecast in response.forecasts:
        request_series = request_series_by_id.get(forecast.id)
        if request_series is None:
            continue
        if not forecast.mean:
            continue

        trend_direction = _trend_direction(forecast.mean)
        confidence_assessment = _confidence_assessment(forecast=forecast)
        historical_comparison = _historical_comparison(
            request_series=request_series,
            forecast=forecast,
        )
        notable_patterns = _notable_patterns(
            request_series=request_series,
            forecast=forecast,
            trend_direction=trend_direction,
        )

        series_payloads.append(
            SeriesForecastExplanation(
                id=forecast.id,
                trend_direction=trend_direction,
                confidence_assessment=confidence_assessment,
                historical_comparison=historical_comparison,
                notable_patterns=notable_patterns,
            ),
        )

    if not series_payloads:
        return None
    return ForecastExplanation(series=series_payloads)


def _trend_direction(mean: list[int | float]) -> TrendDirection:
    first = float(mean[0])
    last = float(mean[-1])
    delta = last - first
    tolerance = max(abs(first) * 0.01, 1e-6)
    if delta > tolerance:
        return "up"
    if delta < -tolerance:
        return "down"
    return "flat"


def _confidence_assessment(*, forecast: SeriesForecast) -> str:
    quantile_vectors = _sorted_quantile_vectors(forecast)
    if quantile_vectors is None:
        return "quantile uncertainty unavailable"
    low = quantile_vectors[0][1]
    high = quantile_vectors[-1][1]
    spreads = [abs(float(hi) - float(lo)) for lo, hi in zip(low, high, strict=True)]
    if not spreads:
        return "quantile uncertainty unavailable"

    baseline = sum(abs(float(value)) for value in forecast.mean) / len(forecast.mean)
    scale = baseline if baseline > 0.0 else 1.0
    spread_ratio = (sum(spreads) / len(spreads)) / scale
    if spread_ratio < 0.2:
        return "high confidence (tight quantile spread)"
    if spread_ratio < 0.5:
        return "moderate confidence (medium quantile spread)"
    return "low confidence (wide quantile spread)"


def _historical_comparison(
    *,
    request_series: SeriesInput,
    forecast: SeriesForecast,
) -> str:
    history = [float(value) for value in request_series.target]
    if not history:
        return "historical baseline unavailable"

    historical_mean = sum(history) / len(history)
    forecast_mean = sum(float(value) for value in forecast.mean) / len(forecast.mean)
    delta = forecast_mean - historical_mean

    if historical_mean == 0.0:
        if delta > 0.0:
            return "forecast mean is above historical mean"
        if delta < 0.0:
            return "forecast mean is below historical mean"
        return "forecast mean is aligned with historical mean"

    delta_pct = delta / abs(historical_mean) * 100.0
    if abs(delta_pct) < 2.0:
        return "forecast mean is close to historical mean"
    if delta_pct > 0.0:
        return f"forecast mean is {abs(delta_pct):.1f}% above historical mean"
    return f"forecast mean is {abs(delta_pct):.1f}% below historical mean"


def _notable_patterns(
    *,
    request_series: SeriesInput,
    forecast: SeriesForecast,
    trend_direction: TrendDirection,
) -> list[str]:
    patterns: list[str] = []

    if trend_direction == "up":
        patterns.append("mean forecast shows an upward trajectory")
    elif trend_direction == "down":
        patterns.append("mean forecast shows a downward trajectory")
    else:
        patterns.append("mean forecast is relatively flat")

    if _is_monotonic_increasing(forecast.mean):
        patterns.append("forecast is monotonic increasing")
    elif _is_monotonic_decreasing(forecast.mean):
        patterns.append("forecast is monotonic decreasing")

    history_non_negative = all(float(value) >= 0.0 for value in request_series.target)
    if history_non_negative and any(float(value) < 0.0 for value in forecast.mean):
        patterns.append("forecast crosses below zero while history is non-negative")

    quantile_vectors = _sorted_quantile_vectors(forecast)
    if quantile_vectors is not None:
        low = quantile_vectors[0][1]
        high = quantile_vectors[-1][1]
        spreads = [abs(float(hi) - float(lo)) for lo, hi in zip(low, high, strict=True)]
        if len(spreads) >= 2:
            first = spreads[0]
            last = spreads[-1]
            if first > 0.0 and last > first * 1.2:
                patterns.append("uncertainty widens over the forecast horizon")
            elif first > 0.0 and last < first * 0.8:
                patterns.append("uncertainty narrows over the forecast horizon")

    return patterns


def _is_monotonic_increasing(values: list[int | float]) -> bool:
    if len(values) < 2:
        return False
    return all(float(values[index]) >= float(values[index - 1]) for index in range(1, len(values)))


def _is_monotonic_decreasing(values: list[int | float]) -> bool:
    if len(values) < 2:
        return False
    return all(float(values[index]) <= float(values[index - 1]) for index in range(1, len(values)))


def _sorted_quantile_vectors(
    forecast: SeriesForecast,
) -> list[tuple[float, list[int | float]]] | None:
    quantiles = forecast.quantiles
    if not quantiles:
        return None

    parsed: list[tuple[float, list[int | float]]] = []
    for level, values in quantiles.items():
        try:
            parsed.append((float(level), list(values)))
        except ValueError:
            return None

    if len(parsed) < 2:
        return None
    parsed.sort(key=lambda item: item[0])
    return parsed
