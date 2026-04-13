"""Request payload construction helpers for real-data E2E scenarios."""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    _THIS_DIR = Path(__file__).resolve().parent
    if str(_THIS_DIR) not in sys.path:
        sys.path.append(str(_THIS_DIR))
    import model_policy as model_policy  # noqa: PLC0414
else:
    from . import model_policy


def expected_status_for_strict_covariates(model: str) -> int:
    """Return strict covariates expected status by model capability contract."""
    return model_policy.strict_expected_status_for_model(model)


def best_effort_warning_required(model: str) -> bool:
    """Return whether best-effort covariates should emit warnings."""
    return model_policy.best_effort_warning_required(model)


def build_target_only_request(
    *,
    model: str,
    series: dict[str, Any],
    horizon: int,
    timeout_seconds: float,
) -> dict[str, Any]:
    """Build the benchmark target-only request payload."""
    request_series = _base_series_payload(series)
    request_series["actuals"] = list(series.get("actuals", []))

    return {
        "model": model,
        "horizon": horizon,
        "quantiles": [],
        "series": [request_series],
        "options": model_policy.model_options(model),
        "timeout": timeout_seconds,
        "parameters": {
            "covariates_mode": "best_effort",
            "metrics": {"names": list(model_policy.BENCHMARK_METRIC_NAMES)},
        },
    }


def build_covariate_request(
    *,
    model: str,
    series: dict[str, Any],
    horizon: int,
    covariates_mode: str,
    timeout_seconds: float,
) -> dict[str, Any]:
    """Build a contract-validation request with mixed covariate types."""
    past_covariates, future_covariates = build_mixed_calendar_covariates(
        timestamps=list(series["timestamps"]),
        horizon=horizon,
        freq=str(series["freq"]),
    )
    request_series = _base_series_payload(series)
    request_series["past_covariates"] = past_covariates
    request_series["future_covariates"] = future_covariates

    return {
        "model": model,
        "horizon": horizon,
        "quantiles": [],
        "series": [request_series],
        "options": model_policy.model_options(model),
        "timeout": timeout_seconds,
        "parameters": {
            "covariates_mode": covariates_mode,
        },
    }


def filter_covariates_for_model(
    *,
    model: str,
    past_covariates: dict[str, list[Any]],
    future_covariates: dict[str, list[Any]],
) -> tuple[dict[str, list[Any]], dict[str, list[Any]]]:
    """Filter covariates to model-supported subsets."""
    numeric_keys = [name for name, values in past_covariates.items() if _is_numeric_list(values)]

    if model == "chronos2":
        return dict(past_covariates), dict(future_covariates)

    if model in {
        "granite-ttm-r2",
        "timesfm-2.5-200m",
        "moirai-2.0-R-small",
        "nhits",
        "nbeatsx",
    }:
        filtered_past = {name: past_covariates[name] for name in numeric_keys}
        filtered_future = {
            name: values
            for name, values in future_covariates.items()
            if name in filtered_past and _is_numeric_list(values)
        }
        return filtered_past, filtered_future

    if model == "toto-open-base-1.0":
        filtered_past = {name: past_covariates[name] for name in numeric_keys}
        return filtered_past, {}

    if model == "sundial-base-128m":
        return {}, {}

    raise ValueError(f"unsupported model: {model!r}")


def build_mixed_calendar_covariates(
    *,
    timestamps: list[str],
    horizon: int,
    freq: str,
) -> tuple[dict[str, list[Any]], dict[str, list[Any]]]:
    """Build known-future numeric + categorical covariates from timestamps."""
    parsed_history = [_parse_timestamp(value) for value in timestamps]
    if any(item is None for item in parsed_history):
        raise ValueError("unable to parse timestamps for covariate generation")

    history = [item for item in parsed_history if item is not None]
    future = _future_timestamps(history=history, horizon=horizon, freq=freq)
    use_hour_bucket = freq.strip().upper().startswith("H")

    past_numeric = [
        float(_numeric_bucket(point, use_hour_bucket=use_hour_bucket)) for point in history
    ]
    future_numeric = [
        float(_numeric_bucket(point, use_hour_bucket=use_hour_bucket)) for point in future
    ]

    past_categorical = [_calendar_label(point) for point in history]
    future_categorical = [_calendar_label(point) for point in future]

    past_covariates = {
        "calendar_num": past_numeric,
        "calendar_cat": past_categorical,
    }
    future_covariates = {
        "calendar_num": future_numeric,
        "calendar_cat": future_categorical,
    }
    return past_covariates, future_covariates


def _base_series_payload(series: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": str(series["id"]),
        "freq": str(series["freq"]),
        "timestamps": list(series["timestamps"]),
        "target": list(series["target"]),
    }


def _parse_timestamp(raw: str) -> datetime | None:
    normalized = raw.strip().replace("Z", "+00:00")
    if not normalized:
        return None

    if normalized.isdigit():
        try:
            epoch = int(normalized)
            if len(normalized) >= 13:
                epoch = epoch // 1000
            return datetime.fromtimestamp(epoch)
        except (OverflowError, OSError, ValueError):
            pass

    try:
        parsed = datetime.fromisoformat(normalized)
        return parsed.replace(tzinfo=None) if parsed.tzinfo else parsed
    except ValueError:
        pass

    formats = (
        "%Y-%m-%d",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%dT%H:%M:%S",
        "%Y/%m/%d",
        "%Y/%m/%d %H:%M:%S",
        "%Y/%m/%d %H:%M",
        "%m/%d/%Y",
        "%m/%d/%Y %H:%M:%S",
        "%m/%d/%Y %H:%M",
        "%d/%m/%Y",
        "%d/%m/%Y %H:%M:%S",
        "%d/%m/%Y %H:%M",
    )
    for fmt in formats:
        try:
            return datetime.strptime(normalized, fmt)
        except ValueError:
            continue
    return None


def _future_timestamps(*, history: list[datetime], horizon: int, freq: str) -> list[datetime]:
    if not history:
        raise ValueError("history timestamps are required")

    normalized_freq = freq.strip().upper()
    step = timedelta(hours=1) if normalized_freq.startswith("H") else timedelta(days=1)

    anchor = history[-1]
    return [anchor + step * (index + 1) for index in range(horizon)]


def _numeric_bucket(point: datetime, *, use_hour_bucket: bool) -> int:
    if use_hour_bucket:
        return point.hour
    return point.weekday()


def _calendar_label(point: datetime) -> str:
    if point.weekday() in {5, 6}:
        return "weekend"
    return "weekday"


def _is_numeric_list(values: list[Any]) -> bool:
    return all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in values)
