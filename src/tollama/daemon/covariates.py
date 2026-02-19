"""Daemon-level covariate normalization and compatibility filtering."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from tollama.core.registry import ModelCapabilities
from tollama.core.schemas import CovariateValues, SeriesInput


@dataclass(frozen=True)
class SeriesCovariateProfile:
    """Typed covariate names grouped by role for one normalized input series."""

    past_only_numeric: frozenset[str]
    past_only_categorical: frozenset[str]
    known_future_numeric: frozenset[str]
    known_future_categorical: frozenset[str]
    static_covariates: frozenset[str]


def normalize_covariates(
    inputs: Sequence[SeriesInput],
    prediction_length: int,
) -> tuple[list[SeriesInput], list[str]]:
    """Normalize per-series covariates and enforce unified request contract rules."""
    normalized: list[SeriesInput] = []
    warnings: list[str] = []

    for series in inputs:
        resolved_freq = series.freq
        if resolved_freq == "auto":
            inferred_freq = _infer_freq_from_timestamps(series.timestamps)
            if inferred_freq is None:
                raise ValueError(
                    f"series {series.id!r}: freq='auto' but could not infer frequency from "
                    f"{len(series.timestamps)} timestamps. Provide an explicit freq value.",
                )
            resolved_freq = inferred_freq

        expected_history = len(series.target)
        past_covariates = {
            name: list(values)
            for name, values in sorted((series.past_covariates or {}).items())
        }
        future_covariates = {
            name: list(values)
            for name, values in sorted((series.future_covariates or {}).items())
        }

        past_kinds: dict[str, str] = {}
        for name, values in past_covariates.items():
            if len(values) != expected_history:
                raise ValueError(
                    f"series {series.id!r} past_covariates[{name!r}] length must equal "
                    f"target length ({expected_history})",
                )
            past_kinds[name] = _covariate_kind(values)

        future_kinds: dict[str, str] = {}
        for name, values in future_covariates.items():
            if len(values) != prediction_length:
                raise ValueError(
                    f"series {series.id!r} future_covariates[{name!r}] length must equal "
                    f"horizon ({prediction_length})",
                )
            future_kinds[name] = _covariate_kind(values)

        future_only = set(future_covariates) - set(past_covariates)
        if future_only:
            first = sorted(future_only)[0]
            raise ValueError(
                f"series {series.id!r} future_covariates[{first!r}] is missing in "
                "past_covariates; known-future covariates must be present in both",
            )

        known_future = set(past_covariates).intersection(future_covariates)
        for name in known_future:
            if past_kinds[name] != future_kinds[name]:
                raise ValueError(
                    f"series {series.id!r} covariate {name!r} type differs between "
                    f"past ({past_kinds[name]}) and future ({future_kinds[name]})",
                )

        normalized.append(
            series.model_copy(
                update={
                    "freq": resolved_freq,
                    "past_covariates": past_covariates or None,
                    "future_covariates": (
                        {name: future_covariates[name] for name in sorted(known_future)} or None
                    ),
                },
            ),
        )

    return normalized, warnings


def build_covariate_profile(series: SeriesInput) -> SeriesCovariateProfile:
    """Classify normalized covariate names into past-only and known-future groups."""
    past_covariates = series.past_covariates or {}
    future_covariates = series.future_covariates or {}
    known_future = set(past_covariates).intersection(future_covariates)
    past_only = set(past_covariates) - known_future

    past_only_numeric: set[str] = set()
    past_only_categorical: set[str] = set()
    known_future_numeric: set[str] = set()
    known_future_categorical: set[str] = set()

    for name in known_future:
        kind = _covariate_kind(past_covariates[name])
        if kind == "numeric":
            known_future_numeric.add(name)
        else:
            known_future_categorical.add(name)

    for name in past_only:
        kind = _covariate_kind(past_covariates[name])
        if kind == "numeric":
            past_only_numeric.add(name)
        else:
            past_only_categorical.add(name)

    static_covariates = set((series.static_covariates or {}).keys())
    return SeriesCovariateProfile(
        past_only_numeric=frozenset(past_only_numeric),
        past_only_categorical=frozenset(past_only_categorical),
        known_future_numeric=frozenset(known_future_numeric),
        known_future_categorical=frozenset(known_future_categorical),
        static_covariates=frozenset(static_covariates),
    )


def apply_covariate_capabilities(
    *,
    model_name: str,
    model_family: str,
    inputs: Sequence[SeriesInput],
    capabilities: ModelCapabilities | None,
    covariates_mode: str,
) -> tuple[list[SeriesInput], list[str]]:
    """Drop unsupported covariates in best-effort mode or raise in strict mode."""
    resolved_capabilities = capabilities or ModelCapabilities()
    strict_mode = covariates_mode == "strict"

    filtered: list[SeriesInput] = []
    warnings: list[str] = []
    errors: list[str] = []
    warning_seen: set[str] = set()

    for series in inputs:
        profile = build_covariate_profile(series)
        past_covariates = dict(series.past_covariates or {})
        future_covariates = dict(series.future_covariates or {})
        static_covariates = dict(series.static_covariates or {})

        drop_past: set[str] = set()
        drop_future: set[str] = set()
        drop_static = False

        for name in sorted(profile.past_only_numeric):
            if not resolved_capabilities.past_covariates_numeric:
                drop_past.add(name)
                _append_covariate_issue(
                    issues=errors if strict_mode else warnings,
                    seen=warning_seen,
                    strict_mode=strict_mode,
                    model_name=model_name,
                    model_family=model_family,
                    series_id=series.id,
                    role="past-only",
                    kind="numeric",
                    name=name,
                )

        for name in sorted(profile.past_only_categorical):
            if not resolved_capabilities.past_covariates_categorical:
                drop_past.add(name)
                _append_covariate_issue(
                    issues=errors if strict_mode else warnings,
                    seen=warning_seen,
                    strict_mode=strict_mode,
                    model_name=model_name,
                    model_family=model_family,
                    series_id=series.id,
                    role="past-only",
                    kind="categorical",
                    name=name,
                )

        for name in sorted(profile.known_future_numeric):
            if not resolved_capabilities.future_covariates_numeric:
                drop_past.add(name)
                drop_future.add(name)
                _append_covariate_issue(
                    issues=errors if strict_mode else warnings,
                    seen=warning_seen,
                    strict_mode=strict_mode,
                    model_name=model_name,
                    model_family=model_family,
                    series_id=series.id,
                    role="known-future",
                    kind="numeric",
                    name=name,
                )

        for name in sorted(profile.known_future_categorical):
            if not resolved_capabilities.future_covariates_categorical:
                drop_past.add(name)
                drop_future.add(name)
                _append_covariate_issue(
                    issues=errors if strict_mode else warnings,
                    seen=warning_seen,
                    strict_mode=strict_mode,
                    model_name=model_name,
                    model_family=model_family,
                    series_id=series.id,
                    role="known-future",
                    kind="categorical",
                    name=name,
                )

        if profile.static_covariates and not resolved_capabilities.static_covariates:
            drop_static = True
            _append_static_issue(
                issues=errors if strict_mode else warnings,
                seen=warning_seen,
                strict_mode=strict_mode,
                model_name=model_name,
                model_family=model_family,
                series_id=series.id,
                names=sorted(profile.static_covariates),
            )

        if strict_mode:
            continue

        normalized_past = {
            name: values
            for name, values in past_covariates.items()
            if name not in drop_past
        }
        normalized_future = {
            name: values for name, values in future_covariates.items() if name not in drop_future
        }
        normalized_static = None if drop_static else (static_covariates or None)

        filtered.append(
            series.model_copy(
                update={
                    "past_covariates": normalized_past or None,
                    "future_covariates": normalized_future or None,
                    "static_covariates": normalized_static,
                },
            ),
        )

    if strict_mode and errors:
        raise ValueError("; ".join(errors))
    if strict_mode:
        return list(inputs), []
    return filtered, warnings


def _covariate_kind(values: CovariateValues) -> str:
    if not values:
        raise ValueError("covariate arrays must not be empty")

    has_numeric = False
    has_string = False
    for value in values:
        if isinstance(value, str):
            has_string = True
            continue
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            has_numeric = True
            continue
        raise ValueError(f"covariate contains unsupported value {value!r}")

    if has_numeric and has_string:
        raise ValueError("covariate arrays must not mix numeric and string values")
    if has_string:
        return "categorical"
    return "numeric"


def _infer_freq_from_timestamps(timestamps: list[str]) -> str | None:
    try:
        import pandas as pd
    except ImportError as exc:
        raise ValueError(
            "pandas is required to infer frequency when freq='auto'. "
            "Install pandas>=2.0,<3.0.",
        ) from exc

    try:
        index = pd.DatetimeIndex(timestamps)
        inferred = pd.infer_freq(index)
    except Exception:  # noqa: BLE001
        return None

    if isinstance(inferred, str) and inferred:
        return inferred
    return None


def _append_covariate_issue(
    *,
    issues: list[str],
    seen: set[str],
    strict_mode: bool,
    model_name: str,
    model_family: str,
    series_id: str,
    role: str,
    kind: str,
    name: str,
) -> None:
    prefix = "unsupported covariate" if strict_mode else "ignoring unsupported covariate"
    message = (
        f"{prefix}: model {model_name!r} (family={model_family}) does not support "
        f"{role} {kind} covariates; series={series_id!r} covariate={name!r}"
    )
    if strict_mode:
        issues.append(message)
        return
    if message in seen:
        return
    seen.add(message)
    issues.append(message)


def _append_static_issue(
    *,
    issues: list[str],
    seen: set[str],
    strict_mode: bool,
    model_name: str,
    model_family: str,
    series_id: str,
    names: list[str],
) -> None:
    joined = ", ".join(repr(name) for name in names)
    prefix = "unsupported covariate" if strict_mode else "ignoring unsupported covariate"
    message = (
        f"{prefix}: model {model_name!r} (family={model_family}) does not support static "
        f"covariates; series={series_id!r} covariates={joined}"
    )
    if strict_mode:
        issues.append(message)
        return
    if message in seen:
        return
    seen.add(message)
    issues.append(message)
