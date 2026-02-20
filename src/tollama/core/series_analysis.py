"""Deterministic helpers for lightweight time-series analysis."""

from __future__ import annotations

import math
import warnings
from collections import Counter

import pandas as pd

from .schemas import (
    AnalyzeParameters,
    AnalyzeRequest,
    AnalyzeResponse,
    AnomalyRecord,
    SeriesAnalysis,
    SeriesInput,
    TrendAnalysis,
)

_MIN_POINTS_FOR_SEASONALITY = 8
_MIN_POINTS_FOR_STATIONARITY = 8
_SEASONALITY_CORRELATION_THRESHOLD = 0.3
_FLAT_SLOPE_RATIO_THRESHOLD = 1e-3


def analyze_series_request(request: AnalyzeRequest) -> AnalyzeResponse:
    """Analyze every input series and return canonical response payload."""
    results = [
        analyze_series(series=series, parameters=request.parameters)
        for series in request.series
    ]

    merged_warnings: list[str] = []
    for result in results:
        if result.warnings:
            merged_warnings.extend(f"{result.id}: {warning}" for warning in result.warnings)

    response_warnings = _dedupe_preserve_order(merged_warnings)
    return AnalyzeResponse(results=results, warnings=response_warnings or None)


def analyze_series(*, series: SeriesInput, parameters: AnalyzeParameters) -> SeriesAnalysis:
    """Analyze one series deterministically with bounded compute."""
    timestamps = [str(item) for item in series.timestamps]
    values = [float(item) for item in series.target]

    warnings: list[str] = []
    sampled_timestamps, sampled_values, sampling_warnings = _downsample_series(
        timestamps=timestamps,
        values=values,
        max_points=parameters.max_points,
    )
    warnings.extend(sampling_warnings)
    (
        ordered_timestamps,
        ordered_values,
        ordering_warnings,
    ) = _order_for_chronological_analysis(
        timestamps=sampled_timestamps,
        values=sampled_values,
    )
    warnings.extend(ordering_warnings)

    detected_frequency, regularity_score, frequency_warnings = detect_frequency(timestamps)
    warnings.extend(frequency_warnings)

    seasonality_periods, seasonality_warnings = detect_seasonality(
        ordered_values,
        max_lag=parameters.max_lag,
        top_k=parameters.top_k_seasonality,
    )
    warnings.extend(seasonality_warnings)

    trend, trend_warnings = detect_trend(ordered_values)
    warnings.extend(trend_warnings)

    anomaly_indices, anomaly_warnings = detect_anomalies(values, iqr_k=parameters.anomaly_iqr_k)
    warnings.extend(anomaly_warnings)
    anomalies = build_anomaly_records(
        values=values,
        anomaly_indices=anomaly_indices,
        iqr_k=parameters.anomaly_iqr_k,
    )

    stationarity_flag, stationarity_warnings = detect_stationarity(ordered_values)
    warnings.extend(stationarity_warnings)

    quality = compute_data_quality(
        timestamps=ordered_timestamps,
        values=sampled_values,
        regularity_score=regularity_score,
    )

    deduped = _dedupe_preserve_order(warnings)
    return SeriesAnalysis(
        id=series.id,
        detected_frequency=detected_frequency,
        seasonality_periods=sorted(seasonality_periods),
        trend=trend,
        anomaly_indices=anomaly_indices,
        anomalies=anomalies,
        stationarity_flag=stationarity_flag,
        data_quality_score=quality,
        warnings=deduped or None,
    )


def detect_frequency(timestamps: list[str]) -> tuple[str, float, list[str]]:
    """Detect likely frequency and return a regularity score in [0, 1]."""
    warnings: list[str] = []
    parsed_index = _parse_timestamps(timestamps)
    if parsed_index.isna().any():
        warnings.append("some timestamps could not be parsed")

    valid = parsed_index.dropna()
    if len(valid) < 3:
        warnings.append("insufficient valid timestamps for frequency detection")
        return "unknown", 0.0, warnings

    index = pd.DatetimeIndex(valid)
    if not index.is_monotonic_increasing:
        warnings.append("timestamps were not monotonic; chronological order was applied")
        index = index.sort_values()
    regularity_score = _regularity_score(index)
    inferred = pd.infer_freq(index)
    if inferred is not None:
        return inferred, regularity_score, warnings

    if regularity_score >= 0.8:
        return _nearest_offset_alias(index), regularity_score, warnings

    warnings.append("timestamps are irregular")
    return "irregular", regularity_score, warnings


def detect_seasonality(
    values: list[float],
    *,
    max_lag: int,
    top_k: int,
) -> tuple[list[int], list[str]]:
    """Detect seasonality lags from autocorrelation peaks."""
    warnings: list[str] = []
    n_obs = len(values)
    if n_obs < _MIN_POINTS_FOR_SEASONALITY:
        warnings.append("insufficient points for seasonality detection")
        return [], warnings

    usable_max_lag = min(max_lag, max(2, n_obs // 2))
    if usable_max_lag < 2:
        warnings.append("insufficient lag window for seasonality detection")
        return [], warnings

    correlations: list[tuple[int, float]] = []
    for lag in range(2, usable_max_lag + 1):
        correlation = _autocorrelation(values, lag)
        correlations.append((lag, correlation))

    peaks: list[tuple[int, float]] = []
    for index, (lag, corr) in enumerate(correlations):
        prev_corr = correlations[index - 1][1] if index > 0 else -1.0
        next_corr = correlations[index + 1][1] if index + 1 < len(correlations) else -1.0
        if corr >= _SEASONALITY_CORRELATION_THRESHOLD and corr >= prev_corr and corr >= next_corr:
            peaks.append((lag, corr))

    if not peaks:
        peaks = [
            (lag, corr)
            for lag, corr in correlations
            if corr >= _SEASONALITY_CORRELATION_THRESHOLD
        ]

    ranked = sorted(peaks, key=lambda item: (-item[1], item[0]))[:top_k]
    return sorted({lag for lag, _ in ranked}), warnings


def detect_trend(values: list[float]) -> tuple[TrendAnalysis, list[str]]:
    """Estimate trend direction/strength using a simple OLS fit."""
    warnings: list[str] = []
    n_obs = len(values)
    if n_obs < 2:
        warnings.append("insufficient points for trend detection")
        return TrendAnalysis(direction="flat", slope=0.0, r2=0.0), warnings

    mean_x = (n_obs - 1) / 2.0
    mean_y = sum(values) / n_obs

    xx = sum((idx - mean_x) ** 2 for idx in range(n_obs))
    if xx <= 0.0:
        warnings.append("unable to compute trend slope")
        return TrendAnalysis(direction="flat", slope=0.0, r2=0.0), warnings

    xy = sum((idx - mean_x) * (value - mean_y) for idx, value in enumerate(values))
    slope = xy / xx
    intercept = mean_y - slope * mean_x

    ss_tot = sum((value - mean_y) ** 2 for value in values)
    if ss_tot <= 0.0:
        r2 = 0.0
    else:
        ss_res = sum(
            (value - (intercept + slope * idx)) ** 2
            for idx, value in enumerate(values)
        )
        r2 = _clamp(1.0 - (ss_res / ss_tot))

    scale = max(abs(mean_y), 1.0)
    normalized_slope = abs(slope) / scale
    if normalized_slope < _FLAT_SLOPE_RATIO_THRESHOLD:
        direction = "flat"
    elif slope > 0:
        direction = "up"
    else:
        direction = "down"

    return TrendAnalysis(direction=direction, slope=float(slope), r2=float(r2)), warnings


def detect_anomalies(values: list[float], *, iqr_k: float) -> tuple[list[int], list[str]]:
    """Detect point anomalies using the IQR rule."""
    warnings: list[str] = []
    if len(values) < 4:
        warnings.append("insufficient points for anomaly detection")
        return [], warnings

    series = pd.Series(values, dtype="float64")
    q1 = float(series.quantile(0.25))
    q3 = float(series.quantile(0.75))
    if not math.isfinite(q1) or not math.isfinite(q3):
        warnings.append("unable to compute anomaly thresholds")
        return [], warnings

    iqr = q3 - q1
    if iqr <= 0.0:
        return [], warnings

    lower_bound = q1 - iqr_k * iqr
    upper_bound = q3 + iqr_k * iqr
    anomaly_indices = [
        index
        for index, value in enumerate(values)
        if math.isfinite(value) and (value < lower_bound or value > upper_bound)
    ]
    return anomaly_indices, warnings


def build_anomaly_records(
    *,
    values: list[float],
    anomaly_indices: list[int],
    iqr_k: float,
) -> list[AnomalyRecord]:
    """Build structured anomaly records from detected anomaly indices."""
    if not anomaly_indices:
        return []

    series = pd.Series(values, dtype="float64")
    q1 = float(series.quantile(0.25))
    q3 = float(series.quantile(0.75))
    iqr = q3 - q1
    if not math.isfinite(q1) or not math.isfinite(q3) or iqr <= 0.0:
        return []

    lower_bound = q1 - iqr_k * iqr
    upper_bound = q3 + iqr_k * iqr
    anomalies = set(anomaly_indices)
    records: list[AnomalyRecord] = []
    for index in sorted(anomaly_indices):
        value = float(values[index])
        anomaly_type = _classify_anomaly_type(
            values=values,
            anomalies=anomalies,
            index=index,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            iqr=iqr,
        )
        if value > upper_bound:
            distance = value - upper_bound
        else:
            distance = lower_bound - value
        severity = _severity_from_distance(distance=distance, iqr=iqr)
        records.append(
            AnomalyRecord(
                index=index,
                type=anomaly_type,
                severity=severity,
                value=value,
                range_start=max(0, index - 1),
                range_end=min(len(values) - 1, index + 1),
                suggested_handling=_suggested_handling(anomaly_type),
            )
        )
    return records


def _classify_anomaly_type(
    *,
    values: list[float],
    anomalies: set[int],
    index: int,
    lower_bound: float,
    upper_bound: float,
    iqr: float,
) -> str:
    if (index - 1 in anomalies) or (index + 1 in anomalies):
        return "shift"

    previous = values[index - 1] if index > 0 else values[index]
    current = values[index]
    following = values[index + 1] if index + 1 < len(values) else values[index]
    step_before = current - previous
    step_after = following - current
    if (
        step_before != 0.0
        and step_after != 0.0
        and step_before * step_after < 0.0
        and (abs(step_before) > iqr or abs(step_after) > iqr)
    ):
        return "trend_break"

    if current > upper_bound:
        return "spike"
    if current < lower_bound:
        return "dip"
    return "trend_break"


def _severity_from_distance(*, distance: float, iqr: float) -> str:
    if iqr <= 0.0:
        return "low"
    ratio = max(distance, 0.0) / iqr
    if ratio >= 2.0:
        return "high"
    if ratio >= 1.0:
        return "medium"
    return "low"


def _suggested_handling(anomaly_type: str) -> str:
    if anomaly_type == "spike":
        return "Validate event drivers and cap/clip if this is an external shock."
    if anomaly_type == "dip":
        return "Check data gaps and business outages before retraining."
    if anomaly_type == "shift":
        return "Evaluate regime change and retrain with a recent window."
    return "Review structural change assumptions and segment the series if needed."


def detect_stationarity(values: list[float]) -> tuple[bool | None, list[str]]:
    """Lightweight stationarity heuristic based on split-window drift."""
    warnings: list[str] = []
    if len(values) < _MIN_POINTS_FOR_STATIONARITY:
        warnings.append("insufficient points for stationarity detection")
        return None, warnings

    midpoint = len(values) // 2
    first = values[:midpoint]
    second = values[midpoint:]
    if not first or not second:
        warnings.append("insufficient split-window points for stationarity detection")
        return None, warnings

    first_mean = _mean(first)
    second_mean = _mean(second)
    overall_scale = max(abs(_mean(values)), _std(values), 1.0)
    normalized_mean_shift = abs(first_mean - second_mean) / overall_scale

    first_var = _variance(first)
    second_var = _variance(second)
    variance_ratio = _variance_ratio(first_var, second_var)

    stationary = normalized_mean_shift <= 0.25 and variance_ratio <= 2.5
    return stationary, warnings


def compute_data_quality(
    *,
    timestamps: list[str],
    values: list[float],
    regularity_score: float,
) -> float:
    """Compute a simple quality score from completeness/regularity/uniqueness."""
    if not timestamps or not values:
        return 0.0

    length = min(len(timestamps), len(values))
    if length == 0:
        return 0.0

    valid_values = sum(1 for value in values[:length] if math.isfinite(value))
    parsed = _parse_timestamps(timestamps[:length])
    valid_timestamps = int((~parsed.isna()).sum())
    completeness = (valid_values + valid_timestamps) / (2.0 * length)

    uniqueness = len(set(timestamps[:length])) / length
    regularity = _clamp(regularity_score)

    score = 0.50 * completeness + 0.35 * regularity + 0.15 * uniqueness
    return round(_clamp(score), 6)


def _downsample_series(
    *,
    timestamps: list[str],
    values: list[float],
    max_points: int,
) -> tuple[list[str], list[float], list[str]]:
    if len(values) <= max_points:
        return timestamps, values, []

    stride = max(1, math.ceil(len(values) / max_points))
    indices = list(range(0, len(values), stride))
    if indices[-1] != len(values) - 1:
        indices.append(len(values) - 1)
    if len(indices) > max_points:
        indices = indices[:-1]
        indices[-1] = len(values) - 1

    sampled_timestamps = [timestamps[index] for index in indices]
    sampled_values = [values[index] for index in indices]
    warnings = [
        f"series downsampled from {len(values)} to {len(sampled_values)} points for analysis",
    ]
    return sampled_timestamps, sampled_values, warnings


def _order_for_chronological_analysis(
    *,
    timestamps: list[str],
    values: list[float],
) -> tuple[list[str], list[float], list[str]]:
    if len(timestamps) != len(values):
        return timestamps, values, ["timestamps/target length mismatch in analysis input"]

    parsed = _parse_timestamps(timestamps)
    if parsed.isna().any():
        return timestamps, values, ["chronological ordering skipped due to unparseable timestamps"]

    sort_index = sorted(
        range(len(values)),
        key=lambda idx: (parsed[idx].value, idx),
    )
    if sort_index == list(range(len(values))):
        return timestamps, values, []

    ordered_timestamps = [timestamps[idx] for idx in sort_index]
    ordered_values = [values[idx] for idx in sort_index]
    return (
        ordered_timestamps,
        ordered_values,
        ["timestamps were not monotonic; chronological order was applied"],
    )


def _regularity_score(index: pd.DatetimeIndex) -> float:
    if len(index) < 2:
        return 0.0

    deltas: list[float] = []
    for previous, current in zip(index[:-1], index[1:], strict=True):
        deltas.append(round((current - previous).total_seconds(), 6))
    if not deltas:
        return 0.0

    positive_deltas = [delta for delta in deltas if delta > 0.0]
    if not positive_deltas:
        return 0.0

    counts = Counter(positive_deltas)
    most_common = counts.most_common(1)[0][1]
    return _clamp(most_common / len(deltas))


def _nearest_offset_alias(index: pd.DatetimeIndex) -> str:
    deltas: list[float] = []
    for previous, current in zip(index[:-1], index[1:], strict=True):
        delta = (current - previous).total_seconds()
        if delta > 0.0:
            deltas.append(delta)

    if not deltas:
        return "irregular"

    median_seconds = _median(deltas)
    known_offsets = [
        ("S", 1.0),
        ("min", 60.0),
        ("H", 3600.0),
        ("D", 86400.0),
        ("W", 604800.0),
        ("MS", 2629800.0),
        ("QS", 7889400.0),
        ("YS", 31557600.0),
    ]
    best_alias, _best_seconds = min(
        known_offsets,
        key=lambda item: abs(math.log(max(median_seconds, 1e-9) / item[1])),
    )
    return best_alias


def _autocorrelation(values: list[float], lag: int) -> float:
    if lag <= 0 or lag >= len(values):
        return 0.0

    first = values[:-lag]
    second = values[lag:]
    if not first or not second:
        return 0.0

    mean_first = _mean(first)
    mean_second = _mean(second)
    numerator = sum(
        (x_value - mean_first) * (y_value - mean_second)
        for x_value, y_value in zip(first, second, strict=True)
    )
    denom_x = sum((x_value - mean_first) ** 2 for x_value in first)
    denom_y = sum((y_value - mean_second) ** 2 for y_value in second)
    denominator = math.sqrt(denom_x * denom_y)
    if denominator <= 0.0:
        return 0.0

    return float(_clamp(numerator / denominator, lower=-1.0, upper=1.0))


def _variance_ratio(first_var: float, second_var: float) -> float:
    epsilon = 1e-12
    if first_var <= epsilon and second_var <= epsilon:
        return 1.0
    if first_var <= epsilon or second_var <= epsilon:
        return float("inf")
    return max(first_var, second_var) / min(first_var, second_var)


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _variance(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    center = _mean(values)
    return sum((value - center) ** 2 for value in values) / len(values)


def _std(values: list[float]) -> float:
    return math.sqrt(_variance(values))


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    midpoint = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return ordered[midpoint]
    return (ordered[midpoint - 1] + ordered[midpoint]) / 2.0


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        deduped.append(item)
    return deduped


def _clamp(value: float, *, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def _parse_timestamps(timestamps: list[str]) -> pd.DatetimeIndex:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        parsed = pd.to_datetime(timestamps, errors="coerce", utc=True)
    return pd.DatetimeIndex(parsed)
