"""Deterministic narrative builders for LLM-ready structured responses."""

from __future__ import annotations

import math
from statistics import fmean, pstdev

from tollama.core.schemas import (
    AnalysisNarrative,
    AnalysisNarrativeEntry,
    AnalyzeResponse,
    CompareNarrative,
    CompareNarrativeEntry,
    CompareResponse,
    ConfidenceLevel,
    ForecastNarrative,
    ForecastRequest,
    ForecastResponse,
    NarrativeAnomalies,
    NarrativeConfidence,
    NarrativeHistoryComparison,
    NarrativeSeasonality,
    NarrativeTrend,
    PipelineNarrative,
    PipelineResponse,
    SeriesForecast,
    TrendDirection,
)
from tollama.core.series_analysis import detect_anomalies, detect_seasonality

_METRIC_RANK_ORDER = (
    "smape",
    "mape",
    "mae",
    "rmse",
    "mase",
    "wape",
    "rmsse",
    "pinball",
)


def build_forecast_narrative(
    *,
    request: ForecastRequest,
    response: ForecastResponse,
) -> ForecastNarrative | None:
    """Build deterministic per-series narrative summaries for one forecast response."""
    request_series_by_id = {series.id: series for series in request.series}
    narrative_series = []
    for forecast in response.forecasts:
        request_series = request_series_by_id.get(forecast.id)
        if request_series is None:
            continue
        mean_values = [float(item) for item in forecast.mean]
        if not mean_values:
            continue

        history = [float(item) for item in request_series.target]
        trend_direction = _trend_direction(mean_values)
        trend = NarrativeTrend(
            direction=trend_direction,
            strength=_trend_strength(mean_values),
        )
        confidence = _build_confidence(
            forecast=forecast,
            response=response,
            series_id=forecast.id,
        )
        seasonality = _build_seasonality(history)
        anomaly_indices, _warnings = detect_anomalies(history, iqr_k=1.5)
        anomalies = NarrativeAnomalies(
            count=len(anomaly_indices),
            indices=anomaly_indices,
        )
        comparison = _comparison_to_history(history=history, forecast_mean=mean_values)

        mean_delta_pct = comparison.mean_delta_pct
        if mean_delta_pct is None:
            delta_hint = "stable against zero-centered history"
        elif abs(mean_delta_pct) < 2.0:
            delta_hint = "close to historical mean"
        elif mean_delta_pct > 0:
            delta_hint = f"{abs(mean_delta_pct):.1f}% above historical mean"
        else:
            delta_hint = f"{abs(mean_delta_pct):.1f}% below historical mean"

        key_insight = (
            f"Trend is {trend_direction} with {confidence.level} confidence; "
            f"forecast mean is {delta_hint}."
        )
        narrative_series.append(
            {
                "id": forecast.id,
                "summary": (
                    f"{len(mean_values)}-step forecast narrative for series {forecast.id!r} "
                    f"using model {response.model!r}"
                ),
                "trend": trend,
                "confidence": confidence,
                "seasonality": seasonality,
                "anomalies": anomalies,
                "key_insight": key_insight,
                "comparison_to_history": comparison,
            }
        )

    if not narrative_series:
        return None
    return ForecastNarrative.model_validate({"series": narrative_series})


def build_analysis_narrative(*, response: AnalyzeResponse) -> AnalysisNarrative | None:
    """Build deterministic per-series narrative summaries for one analyze response."""
    entries = []
    for result in response.results:
        key_risks: list[str] = []
        if result.anomaly_indices:
            key_risks.append("anomalies_detected")
        if result.stationarity_flag is False:
            key_risks.append("non_stationary")
        if result.data_quality_score < 0.7:
            key_risks.append("low_data_quality")
        for warning in (result.warnings or [])[:2]:
            key_risks.append(warning)

        dominant_period = result.seasonality_periods[0] if result.seasonality_periods else None
        entries.append(
            AnalysisNarrativeEntry(
                id=result.id,
                summary=(
                    f"Series {result.id!r} shows {result.trend.direction} trend, "
                    f"{len(result.anomaly_indices)} anomalies, and data quality "
                    f"{result.data_quality_score:.2f}"
                ),
                trend_direction=result.trend.direction,
                dominant_seasonality_period=dominant_period,
                anomaly_count=len(result.anomaly_indices),
                data_quality_score=result.data_quality_score,
                key_risks=key_risks,
            )
        )

    if not entries:
        return None
    return AnalysisNarrative(series=entries)


def build_comparison_narrative(*, response: CompareResponse) -> CompareNarrative:
    """Build deterministic compare-response narrative using best available ranking signal."""
    successful = [item for item in response.results if item.ok and item.response is not None]
    if not successful:
        return CompareNarrative(
            summary=f"Compared {len(response.results)} models; no successful forecast responses.",
            criterion="availability",
            best_model=None,
            rankings=[],
        )

    metric_name, metric_rankings = _rank_successful_by_metric(successful)
    if metric_name is not None and metric_rankings:
        rankings = [
            CompareNarrativeEntry(model=model, rank=index + 1, score=score)
            for index, (model, score) in enumerate(metric_rankings)
        ]
        best_model = metric_rankings[0][0]
        criterion = f"metrics.aggregate.{metric_name}"
        return CompareNarrative(
            summary=(
                f"Compared {len(response.results)} models; "
                f"best model is {best_model!r} by {criterion}."
            ),
            criterion=criterion,
            best_model=best_model,
            rankings=rankings,
        )

    timed_rankings = _rank_successful_by_timing(successful)
    if timed_rankings:
        rankings = [
            CompareNarrativeEntry(model=model, rank=index + 1, score=score)
            for index, (model, score) in enumerate(timed_rankings)
        ]
        best_model = timed_rankings[0][0]
        return CompareNarrative(
            summary=(
                f"Compared {len(response.results)} models; "
                f"best model is {best_model!r} by timing.total_ms."
            ),
            criterion="timing.total_ms",
            best_model=best_model,
            rankings=rankings,
        )

    rankings = [
        CompareNarrativeEntry(model=item.model, rank=index + 1)
        for index, item in enumerate(successful)
    ]
    best_model = rankings[0].model if rankings else None
    return CompareNarrative(
        summary=f"Compared {len(response.results)} models; selected first successful response.",
        criterion="first_success",
        best_model=best_model,
        rankings=rankings,
    )


def build_pipeline_narrative(*, response: PipelineResponse) -> PipelineNarrative:
    """Build deterministic top-level narrative for pipeline output."""
    chosen_model = response.auto_forecast.selection.chosen_model
    warnings_count = len(response.warnings or [])
    return PipelineNarrative(
        summary=(
            f"Pipeline analyzed {len(response.analysis.results)} series and selected model "
            f"{chosen_model!r}."
        ),
        chosen_model=chosen_model,
        pulled_model=response.pulled_model,
        warnings_count=warnings_count,
    )


def _build_confidence(
    *,
    forecast: SeriesForecast,
    response: ForecastResponse,
    series_id: str,
) -> NarrativeConfidence:
    spread_ratio = _quantile_spread_ratio(forecast)
    if spread_ratio is not None:
        level, reason = _confidence_from_spread(spread_ratio)
        return NarrativeConfidence(level=level, spread_ratio=spread_ratio, reason=reason)

    explanation_level = _confidence_from_explanation(response=response, series_id=series_id)
    return NarrativeConfidence(
        level=explanation_level,
        spread_ratio=None,
        reason="quantile uncertainty unavailable",
    )


def _build_seasonality(history: list[float]) -> NarrativeSeasonality:
    if len(history) < 6:
        return NarrativeSeasonality(detected=False, period=None)

    max_lag = min(365, max(2, len(history) // 2))
    periods, _warnings = detect_seasonality(history, max_lag=max_lag, top_k=1)
    if periods:
        return NarrativeSeasonality(detected=True, period=periods[0])
    return NarrativeSeasonality(detected=False, period=None)


def _comparison_to_history(
    *,
    history: list[float],
    forecast_mean: list[float],
) -> NarrativeHistoryComparison:
    if not history or not forecast_mean:
        return NarrativeHistoryComparison(mean_delta_pct=None, volatility_change="unknown")

    history_mean = fmean(history)
    forecast_avg = fmean(forecast_mean)
    mean_delta_pct = None
    if abs(history_mean) > 1e-9:
        mean_delta_pct = ((forecast_avg - history_mean) / abs(history_mean)) * 100.0

    history_vol = _std(history)
    forecast_vol = _std(forecast_mean)
    if history_vol <= 1e-9:
        volatility_change: str = "stable" if forecast_vol <= 1e-9 else "higher"
    else:
        ratio = forecast_vol / history_vol
        if ratio > 1.15:
            volatility_change = "higher"
        elif ratio < 0.85:
            volatility_change = "lower"
        else:
            volatility_change = "stable"

    return NarrativeHistoryComparison(
        mean_delta_pct=mean_delta_pct,
        volatility_change=volatility_change,
    )


def _confidence_from_spread(spread_ratio: float) -> tuple[ConfidenceLevel, str]:
    if spread_ratio < 0.2:
        return "high", "tight quantile spread"
    if spread_ratio < 0.5:
        return "medium", "moderate quantile spread"
    return "low", "wide quantile spread"


def _confidence_from_explanation(
    *,
    response: ForecastResponse,
    series_id: str,
) -> ConfidenceLevel:
    explanation = response.explanation
    if explanation is None:
        return "unknown"

    for series in explanation.series:
        if series.id != series_id:
            continue
        text = series.confidence_assessment.lower()
        if "high confidence" in text:
            return "high"
        if "moderate confidence" in text or "medium" in text:
            return "medium"
        if "low confidence" in text:
            return "low"
        return "unknown"
    return "unknown"


def _quantile_spread_ratio(forecast: SeriesForecast) -> float | None:
    quantiles = forecast.quantiles
    if not quantiles or len(quantiles) < 2:
        return None

    parsed: list[tuple[float, list[float]]] = []
    for key, values in quantiles.items():
        try:
            level = float(key)
        except ValueError:
            return None
        parsed.append((level, [float(item) for item in values]))
    parsed.sort(key=lambda item: item[0])
    low = parsed[0][1]
    high = parsed[-1][1]
    if not low or len(low) != len(high):
        return None

    spreads = [abs(hi - lo) for lo, hi in zip(low, high, strict=True)]
    if not spreads:
        return None

    mean_values = [float(item) for item in forecast.mean]
    scale = max(fmean(abs(value) for value in mean_values), 1.0)
    return fmean(spreads) / scale


def _trend_direction(values: list[float]) -> TrendDirection:
    if len(values) < 2:
        return "flat"
    first = values[0]
    last = values[-1]
    delta = last - first
    tolerance = max(abs(first) * 0.01, 1e-6)
    if delta > tolerance:
        return "up"
    if delta < -tolerance:
        return "down"
    return "flat"


def _trend_strength(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    first = values[0]
    last = values[-1]
    normalized = abs(last - first) / max(abs(first), 1.0)
    return min(max(normalized, 0.0), 1.0)


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    std = pstdev(values)
    if not math.isfinite(std):
        return 0.0
    return float(std)


def _rank_successful_by_metric(
    successful: list,
) -> tuple[str | None, list[tuple[str, float]]]:
    for metric_name in _METRIC_RANK_ORDER:
        scored: list[tuple[str, float]] = []
        for item in successful:
            metrics = item.response.metrics
            if metrics is None:
                continue
            value = metrics.aggregate.get(metric_name)
            if value is None:
                continue
            scored.append((item.model, float(value)))
        if scored:
            scored.sort(key=lambda pair: pair[1])
            return metric_name, scored
    return None, []


def _rank_successful_by_timing(successful: list) -> list[tuple[str, float]]:
    scored: list[tuple[str, float]] = []
    for item in successful:
        timing = item.response.timing
        if timing is None or timing.total_ms is None:
            continue
        scored.append((item.model, float(timing.total_ms)))
    scored.sort(key=lambda pair: pair[1])
    return scored
