"""Probabilistic scenario-tree construction for branching forecasts."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta

from tollama.core.schemas import (
    ForecastRequest,
    ForecastResponse,
    ScenarioTreeNode,
    ScenarioTreeRequest,
    ScenarioTreeResponse,
    SeriesForecast,
    SeriesInput,
)

ForecastExecutor = Callable[[ForecastRequest], ForecastResponse]
_MAX_SCENARIO_TREE_NODES = 5000


@dataclass(slots=True)
class _PathState:
    series: SeriesInput
    parent_id: str
    probability: float


def build_scenario_tree(
    *,
    payload: ScenarioTreeRequest,
    forecast_executor: ForecastExecutor,
) -> ScenarioTreeResponse:
    """Build a flattened probabilistic scenario tree via recursive one-step forecasts."""
    warnings: list[str] = []
    nodes: list[ScenarioTreeNode] = []
    states: list[_PathState] = []
    node_counter = 0

    def _next_id() -> str:
        nonlocal node_counter
        node_counter += 1
        return f"node_{node_counter}"

    for series in payload.series:
        base_series = _sanitize_series(series=series, warnings=warnings)
        root_id = _next_id()
        nodes.append(
            ScenarioTreeNode(
                node_id=root_id,
                parent_id=None,
                series_id=base_series.id,
                depth=0,
                step=0,
                branch="root",
                quantile=None,
                value=float(base_series.target[-1]),
                probability=1.0,
            )
        )
        states.append(_PathState(series=base_series, parent_id=root_id, probability=1.0))

    for depth in range(1, payload.depth + 1):
        next_states: list[_PathState] = []
        for state in states:
            forecast_request = ForecastRequest(
                model=payload.model,
                horizon=1,
                quantiles=list(payload.branch_quantiles),
                series=[state.series],
                options=payload.options,
                timeout=payload.timeout,
                parameters=payload.parameters.model_copy(update={"metrics": None}),
                response_options=payload.response_options,
            )
            response = forecast_executor(forecast_request)
            series_forecast = _forecast_for_series(
                response=response,
                series_id=state.series.id,
            )
            if series_forecast is None:
                warnings.append(
                    f"missing forecast for series {state.series.id!r} at depth {depth}; skipped",
                )
                continue

            branch_values, branch_warnings = _branch_values(
                forecast=series_forecast,
                branch_quantiles=payload.branch_quantiles,
            )
            warnings.extend(branch_warnings)
            if not branch_values:
                continue

            branch_probability = state.probability / len(branch_values)
            for quantile, value in branch_values:
                if len(nodes) >= _MAX_SCENARIO_TREE_NODES:
                    warnings.append(
                        f"scenario tree node cap reached ({_MAX_SCENARIO_TREE_NODES}); "
                        "truncated remaining branches",
                    )
                    break

                node_id = _next_id()
                nodes.append(
                    ScenarioTreeNode(
                        node_id=node_id,
                        parent_id=state.parent_id,
                        series_id=state.series.id,
                        depth=depth,
                        step=depth,
                        branch=f"q{quantile:g}",
                        quantile=quantile,
                        value=value,
                        probability=branch_probability,
                    )
                )
                next_states.append(
                    _PathState(
                        series=_append_target_value(state.series, value=value, step=depth),
                        parent_id=node_id,
                        probability=branch_probability,
                    )
                )
            if len(nodes) >= _MAX_SCENARIO_TREE_NODES:
                break
        states = next_states
        if not states or len(nodes) >= _MAX_SCENARIO_TREE_NODES:
            break

    return ScenarioTreeResponse(
        model=payload.model,
        depth=payload.depth,
        branch_quantiles=list(payload.branch_quantiles),
        nodes=nodes,
        warnings=_dedupe(warnings) or None,
    )


def _sanitize_series(*, series: SeriesInput, warnings: list[str]) -> SeriesInput:
    if (
        series.past_covariates is not None
        or series.future_covariates is not None
        or series.static_covariates is not None
    ):
        warnings.append(
            f"scenario-tree dropped covariates for series {series.id!r} "
            "to keep recursive branching deterministic",
        )

    return series.model_copy(
        update={
            "actuals": None,
            "past_covariates": None,
            "future_covariates": None,
            "static_covariates": None,
        }
    )


def _forecast_for_series(*, response: ForecastResponse, series_id: str) -> SeriesForecast | None:
    for item in response.forecasts:
        if item.id == series_id:
            return item
    return None


def _branch_values(
    *,
    forecast: SeriesForecast,
    branch_quantiles: list[float],
) -> tuple[list[tuple[float, float]], list[str]]:
    warnings: list[str] = []
    if not forecast.mean:
        return [], [f"forecast mean is empty for series {forecast.id!r}"]

    mean_value = float(forecast.mean[0])
    quantile_map: dict[float, float] = {}
    for key, values in (forecast.quantiles or {}).items():
        if not values:
            continue
        try:
            quantile = float(key)
        except ValueError:
            continue
        quantile_map[quantile] = float(values[0])

    resolved: list[tuple[float, float]] = []
    for quantile in branch_quantiles:
        value = _lookup_quantile_value(
            quantile=quantile,
            quantile_map=quantile_map,
            fallback=mean_value,
        )
        if quantile_map and value == mean_value and quantile not in quantile_map:
            warnings.append(
                f"quantile {quantile:g} missing for series {forecast.id!r}; used mean fallback",
            )
        resolved.append((float(quantile), value))
    return resolved, warnings


def _lookup_quantile_value(
    *,
    quantile: float,
    quantile_map: dict[float, float],
    fallback: float,
) -> float:
    for key, value in quantile_map.items():
        if abs(key - quantile) <= 1e-9:
            return value
    if not quantile_map:
        return fallback
    nearest = min(quantile_map, key=lambda value: abs(value - quantile))
    return quantile_map.get(nearest, fallback)


def _append_target_value(series: SeriesInput, *, value: float, step: int) -> SeriesInput:
    timestamps = list(series.timestamps)
    timestamps.append(_next_timestamp(timestamps=timestamps, freq=series.freq, step=step))
    target = list(series.target)
    target.append(value)
    return series.model_copy(
        update={
            "timestamps": timestamps,
            "target": target,
            "actuals": None,
            "past_covariates": None,
            "future_covariates": None,
            "static_covariates": None,
        }
    )


def _next_timestamp(*, timestamps: list[str], freq: str, step: int) -> str:
    if len(timestamps) < 1:
        return str(step)
    if len(timestamps) >= 2:
        first = _parse_timestamp(timestamps[-2])
        second = _parse_timestamp(timestamps[-1])
        if first is not None and second is not None and second > first:
            return _format_timestamp(second + (second - first))

    last = _parse_timestamp(timestamps[-1])
    cadence = _step_from_freq(freq)
    if last is not None and cadence is not None:
        return _format_timestamp(last + cadence)
    return str(len(timestamps))


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
