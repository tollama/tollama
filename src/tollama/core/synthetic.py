"""Deterministic synthetic time-series generation helpers."""

from __future__ import annotations

import math
import random
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from statistics import fmean

from .schemas import GeneratedSeries, GenerateRequest, GenerateResponse, GenerateVariation


@dataclass(frozen=True, slots=True)
class _SeriesProfile:
    level: float
    slope: float
    seasonal_template: tuple[float, ...]
    seasonal_period: int
    noise_std: float
    non_negative: bool


def generate_synthetic_series(request: GenerateRequest) -> GenerateResponse:
    """Generate synthetic time series from input statistical profiles."""
    if request.method != "statistical":
        raise ValueError(f"unsupported generate method {request.method!r}")

    rng = random.Random(request.seed if request.seed is not None else 0)
    generated: list[GeneratedSeries] = []
    warnings: list[str] = []

    for source in request.series:
        target_length = request.length or len(source.target)
        profile = _build_profile(values=[float(value) for value in source.target], freq=source.freq)

        timestamps, timestamp_warning = _build_timestamps(
            source_timestamps=list(source.timestamps),
            freq=source.freq,
            length=target_length,
        )
        if timestamp_warning is not None:
            warnings.append(f"{source.id}: {timestamp_warning}")

        for sample_index in range(request.count):
            values = _sample_values(
                profile=profile,
                length=target_length,
                variation=request.variation,
                rng=rng,
            )
            generated.append(
                GeneratedSeries(
                    id=f"{source.id}_synthetic_{sample_index + 1}",
                    source_id=source.id,
                    freq=source.freq,
                    timestamps=timestamps,
                    target=values,
                ),
            )

    return GenerateResponse(
        method=request.method,
        generated=generated,
        warnings=_dedupe(warnings) or None,
    )


def _build_profile(*, values: list[float], freq: str) -> _SeriesProfile:
    n_obs = len(values)
    level = float(fmean(values))
    slope = _ols_slope(values)

    seasonal_period = _seasonal_period(freq=freq, length=n_obs)
    seasonal_template: tuple[float, ...] = tuple()
    if seasonal_period > 1:
        baseline = [level + slope * (index - ((n_obs - 1) / 2.0)) for index in range(n_obs)]
        residuals = [value - base for value, base in zip(values, baseline, strict=True)]
        seasonal_template = _seasonal_template(residuals=residuals, period=seasonal_period)

    if seasonal_template:
        seasonal_component = [
            seasonal_template[index % seasonal_period]
            for index in range(n_obs)
        ]
    else:
        seasonal_component = [0.0 for _ in range(n_obs)]

    centered = [
        value
        - (level + slope * (index - ((n_obs - 1) / 2.0)))
        - seasonal_component[index]
        for index, value in enumerate(values)
    ]
    noise_std = max(_std(centered), _std(values) * 0.05, 1e-6)

    return _SeriesProfile(
        level=level,
        slope=slope,
        seasonal_template=seasonal_template,
        seasonal_period=seasonal_period,
        noise_std=noise_std,
        non_negative=all(value >= 0.0 for value in values),
    )


def _sample_values(
    *,
    profile: _SeriesProfile,
    length: int,
    variation: GenerateVariation,
    rng: random.Random,
) -> list[float]:
    level_factor = 1.0 + rng.uniform(-variation.level_jitter, variation.level_jitter)
    trend_factor = 1.0 + rng.uniform(-variation.trend_jitter, variation.trend_jitter)
    seasonality_factor = 1.0 + rng.uniform(
        -variation.seasonality_jitter,
        variation.seasonality_jitter,
    )

    center = (length - 1) / 2.0
    generated: list[float] = []
    for step in range(length):
        trend_value = (profile.level * level_factor) + (profile.slope * trend_factor) * (
            step - center
        )

        seasonal_value = 0.0
        if profile.seasonal_period > 1 and profile.seasonal_template:
            seasonal_value = (
                profile.seasonal_template[step % profile.seasonal_period] * seasonality_factor
            )

        noise = rng.gauss(0.0, profile.noise_std * variation.noise_scale)
        value = trend_value + seasonal_value + noise

        if variation.respect_non_negative and profile.non_negative:
            value = max(0.0, value)

        generated.append(round(float(value), 6))

    return generated


def _seasonal_period(*, freq: str, length: int) -> int:
    if length < 8:
        return 1

    token = _freq_token(freq)
    candidate_map = {
        "MIN": 60,
        "H": 24,
        "D": 7,
        "W": 52,
        "M": 12,
        "Q": 4,
        "Y": 1,
    }
    candidate = candidate_map.get(token, 7 if length >= 14 else 1)
    max_reasonable = max(length // 2, 2)
    if candidate < 2 or candidate > max_reasonable:
        return 1
    return candidate


def _seasonal_template(*, residuals: list[float], period: int) -> tuple[float, ...]:
    buckets: list[list[float]] = [[] for _ in range(period)]
    for index, value in enumerate(residuals):
        buckets[index % period].append(value)

    means = [
        fmean(bucket) if bucket else 0.0
        for bucket in buckets
    ]
    center = fmean(means) if means else 0.0
    return tuple(value - center for value in means)


def _ols_slope(values: list[float]) -> float:
    n_obs = len(values)
    if n_obs < 2:
        return 0.0

    mean_x = (n_obs - 1) / 2.0
    mean_y = fmean(values)

    xx = sum((index - mean_x) ** 2 for index in range(n_obs))
    if xx <= 0.0:
        return 0.0

    xy = sum((index - mean_x) * (value - mean_y) for index, value in enumerate(values))
    return float(xy / xx)


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = fmean(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return math.sqrt(max(variance, 0.0))


def _build_timestamps(
    *,
    source_timestamps: list[str],
    freq: str,
    length: int,
) -> tuple[list[str], str | None]:
    if length <= len(source_timestamps):
        return source_timestamps[:length], None

    start = _parse_timestamp(source_timestamps[0]) if source_timestamps else None
    cadence = _step_from_freq(freq)

    if cadence is None and len(source_timestamps) >= 2:
        first = _parse_timestamp(source_timestamps[0])
        second = _parse_timestamp(source_timestamps[1])
        if first is not None and second is not None and second > first:
            cadence = second - first
            start = first

    if start is None or cadence is None or cadence.total_seconds() <= 0:
        return [str(index) for index in range(length)], (
            "could not infer timestamp cadence; emitted index-based timestamps"
        )

    generated: list[str] = []
    current = start
    for _ in range(length):
        generated.append(_format_timestamp(current))
        current = current + cadence
    return generated, None


def _parse_timestamp(value: str) -> datetime | None:
    candidate = value.strip()
    if not candidate:
        return None
    try:
        return datetime.fromisoformat(candidate.replace("Z", "+00:00"))
    except ValueError:
        return None


def _format_timestamp(value: datetime) -> str:
    if value.tzinfo is not None and value.utcoffset() == timedelta(0):
        return value.isoformat().replace("+00:00", "Z")
    return value.isoformat()


def _step_from_freq(freq: str) -> timedelta | None:
    match = re.fullmatch(r"\s*(\d+)?\s*([A-Za-z]+)\s*", freq or "")
    if match is None:
        return None

    multiplier = int(match.group(1) or "1")
    token = _freq_token(match.group(2))
    if token == "MIN":
        return timedelta(minutes=multiplier)
    if token == "H":
        return timedelta(hours=multiplier)
    if token == "D":
        return timedelta(days=multiplier)
    if token == "W":
        return timedelta(weeks=multiplier)
    if token == "M":
        return timedelta(days=30 * multiplier)
    if token == "Q":
        return timedelta(days=91 * multiplier)
    if token == "Y":
        return timedelta(days=365 * multiplier)
    return None


def _freq_token(freq: str) -> str:
    raw = freq.strip().upper()
    if not raw or raw == "AUTO":
        return "AUTO"

    raw = re.sub(r"^\d+", "", raw)
    if raw in {"T", "MIN", "MINS", "MINUTE", "MINUTES"}:
        return "MIN"
    if raw.startswith("H"):
        return "H"
    if raw.startswith("D"):
        return "D"
    if raw.startswith("W"):
        return "W"
    if raw in {"M", "MS", "ME", "MONTH", "MONTHS"}:
        return "M"
    if raw.startswith("Q"):
        return "Q"
    if raw.startswith("Y") or raw.startswith("A"):
        return "Y"
    return raw


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered
