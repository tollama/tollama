"""Auto model-selection helpers for zero-config forecasting."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import fmean
from typing import Literal

from .recommend import recommend_models
from .registry import ModelSpec, list_registry_models
from .schemas import SeriesInput

AutoForecastStrategy = Literal["auto", "fastest", "best_accuracy", "ensemble"]
CovariatesType = Literal["numeric", "categorical"]

_FAMILY_SPEED_BONUS: dict[str, float] = {
    "mock": 80.0,
    "timesfm": 45.0,
    "torch": 25.0,
    "uni2ts": 18.0,
    "sundial": 16.0,
    "toto": 14.0,
}
_FAMILY_ACCURACY_BONUS: dict[str, float] = {
    "mock": -60.0,
    "timesfm": 14.0,
    "torch": 20.0,
    "uni2ts": 24.0,
    "sundial": 12.0,
    "toto": 22.0,
}


@dataclass(frozen=True, slots=True)
class AutoSeriesProfile:
    """Normalized shape/covariate profile used by auto-selection heuristics."""

    freq_hint: str
    series_count: int
    history_points: int
    average_history_length: float
    has_past_covariates: bool
    has_future_covariates: bool
    has_static_covariates: bool
    covariates_type: CovariatesType


@dataclass(frozen=True, slots=True)
class AutoCandidateScore:
    """One ranked model candidate for auto-forecast selection."""

    model: str
    family: str
    score: float
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class AutoSelection:
    """Selection output used by daemon auto-forecast orchestration."""

    strategy: AutoForecastStrategy
    profile: AutoSeriesProfile
    ranked_candidates: tuple[AutoCandidateScore, ...]
    selected_models: tuple[str, ...]
    chosen_model: str


def build_auto_series_profile(*, series: list[SeriesInput]) -> AutoSeriesProfile:
    """Build one merged request profile across all input series."""
    lengths = [len(item.target) for item in series]
    history_points = sum(lengths)
    avg_length = fmean(lengths) if lengths else 0.0
    has_past_covariates = any(bool(item.past_covariates) for item in series)
    has_future_covariates = any(bool(item.future_covariates) for item in series)
    has_static_covariates = any(bool(item.static_covariates) for item in series)
    covariates_type = _resolve_covariates_type(series)
    freq_hint = _resolve_frequency_hint(series)
    return AutoSeriesProfile(
        freq_hint=freq_hint,
        series_count=len(series),
        history_points=history_points,
        average_history_length=float(avg_length),
        has_past_covariates=has_past_covariates,
        has_future_covariates=has_future_covariates,
        has_static_covariates=has_static_covariates,
        covariates_type=covariates_type,
    )


def select_auto_models(
    *,
    series: list[SeriesInput],
    horizon: int,
    strategy: AutoForecastStrategy,
    include_models: list[str] | set[str] | tuple[str, ...],
    ensemble_top_k: int = 3,
    allow_restricted_license: bool = False,
) -> AutoSelection:
    """Rank installed-compatible models and choose one (or many) by strategy."""
    if horizon <= 0:
        raise ValueError("horizon must be > 0")
    if ensemble_top_k < 2:
        raise ValueError("ensemble_top_k must be >= 2")

    include_model_set = {
        item.strip()
        for item in include_models
        if isinstance(item, str) and item.strip()
    }
    if not include_model_set:
        raise ValueError("include_models must include at least one installed model")

    profile = build_auto_series_profile(series=series)
    base_payload = recommend_models(
        horizon=horizon,
        freq=profile.freq_hint,
        has_past_covariates=profile.has_past_covariates,
        has_future_covariates=profile.has_future_covariates,
        has_static_covariates=profile.has_static_covariates,
        covariates_type=profile.covariates_type,
        allow_restricted_license=allow_restricted_license,
        top_k=max(len(include_model_set), 1),
        include_models=include_model_set,
    )
    spec_by_model = {
        spec.name: spec
        for spec in list_registry_models()
        if spec.name in include_model_set
    }

    ranked_candidates: list[AutoCandidateScore] = []
    for item in base_payload["recommendations"]:
        model = str(item["model"])
        family = str(item["family"])
        base_score = float(item["score"])
        base_reasons = [str(reason) for reason in item.get("reasons", [])]
        spec = spec_by_model.get(model)
        bonus, bonus_reasons = _strategy_bonus(
            strategy=strategy,
            family=family,
            horizon=horizon,
            profile=profile,
            spec=spec,
        )
        ranked_candidates.append(
            AutoCandidateScore(
                model=model,
                family=family,
                score=round(base_score + bonus, 4),
                reasons=tuple([*base_reasons, *bonus_reasons]),
            ),
        )

    ranked = tuple(
        sorted(
            ranked_candidates,
            key=lambda item: (-item.score, item.model),
        ),
    )
    if not ranked:
        raise ValueError("no compatible installed models found for auto-forecast")

    if strategy == "ensemble":
        top_k = min(max(ensemble_top_k, 2), len(ranked))
        selected_models = tuple(item.model for item in ranked[:top_k])
    else:
        selected_models = (ranked[0].model,)

    return AutoSelection(
        strategy=strategy,
        profile=profile,
        ranked_candidates=ranked,
        selected_models=selected_models,
        chosen_model=selected_models[0],
    )


def _strategy_bonus(
    *,
    strategy: AutoForecastStrategy,
    family: str,
    horizon: int,
    profile: AutoSeriesProfile,
    spec: ModelSpec | None,
) -> tuple[float, list[str]]:
    speed_bonus = _FAMILY_SPEED_BONUS.get(family, 0.0)
    accuracy_bonus = _FAMILY_ACCURACY_BONUS.get(family, 0.0)
    profile_bonus = _profile_bonus(profile=profile, horizon=horizon, spec=spec, family=family)

    if strategy == "fastest":
        return speed_bonus + profile_bonus["speed"], [
            "strategy prefers lower-latency model families",
            *profile_bonus["speed_reasons"],
        ]
    if strategy == "best_accuracy":
        return accuracy_bonus + profile_bonus["accuracy"], [
            "strategy prefers higher-accuracy model families",
            *profile_bonus["accuracy_reasons"],
        ]
    if strategy == "ensemble":
        return (0.75 * accuracy_bonus) + (0.25 * speed_bonus) + profile_bonus["accuracy"], [
            "strategy ranks models by accuracy-first blend for weighted ensemble",
            *profile_bonus["accuracy_reasons"],
        ]
    return (
        (0.55 * accuracy_bonus) + (0.45 * speed_bonus) + profile_bonus["auto"],
        [
            "strategy balances speed and accuracy for default auto mode",
            *profile_bonus["auto_reasons"],
        ],
    )


def _profile_bonus(
    *,
    profile: AutoSeriesProfile,
    horizon: int,
    spec: ModelSpec | None,
    family: str,
) -> dict[str, float | list[str]]:
    speed_bonus = 0.0
    accuracy_bonus = 0.0
    speed_reasons: list[str] = []
    accuracy_reasons: list[str] = []

    if profile.average_history_length <= 64:
        if family in {"timesfm", "torch"}:
            accuracy_bonus += 3.0
            accuracy_reasons.append("short history favors robust zero-shot generalists")
        elif family == "mock":
            speed_bonus += 2.0
            speed_reasons.append("short history allows very fast fallback model")

    horizon_limit = _horizon_limit(spec)
    if horizon_limit is not None and horizon_limit > 0:
        utilization = horizon / float(horizon_limit)
        if utilization >= 0.85:
            accuracy_bonus -= 4.0
            speed_bonus -= 2.0
            accuracy_reasons.append("requested horizon is near declared model limit")
        elif utilization <= 0.35:
            accuracy_bonus += 2.0
            speed_bonus += 1.0
            accuracy_reasons.append("requested horizon has ample model headroom")

    if profile.has_future_covariates and family in {"timesfm", "torch", "uni2ts"}:
        accuracy_bonus += 2.0
        accuracy_reasons.append("future covariates available for covariate-aware models")

    if profile.series_count >= 4 and family in {"timesfm", "sundial", "toto"}:
        speed_bonus += 2.0
        speed_reasons.append("multi-series request favors batched model families")

    auto_reasons = [*accuracy_reasons, *speed_reasons]
    return {
        "speed": speed_bonus,
        "accuracy": accuracy_bonus,
        "auto": (0.6 * accuracy_bonus) + (0.4 * speed_bonus),
        "speed_reasons": speed_reasons,
        "accuracy_reasons": accuracy_reasons,
        "auto_reasons": auto_reasons,
    }


def _horizon_limit(spec: ModelSpec | None) -> int | None:
    if spec is None or not isinstance(spec.metadata, dict):
        return None
    for key in ("max_horizon", "prediction_length"):
        raw = spec.metadata.get(key)
        if isinstance(raw, bool):
            continue
        if isinstance(raw, int) and raw > 0:
            return raw
        if isinstance(raw, float) and raw > 0 and raw.is_integer():
            return int(raw)
    return None


def _resolve_frequency_hint(series: list[SeriesInput]) -> str:
    explicit = {
        item.freq.strip()
        for item in series
        if isinstance(item.freq, str) and item.freq.strip().lower() != "auto"
    }
    if len(explicit) == 1:
        return next(iter(explicit))
    return "auto"


def _resolve_covariates_type(series: list[SeriesInput]) -> CovariatesType:
    has_categorical = False
    has_numeric = False
    for item in series:
        for payload in (item.past_covariates or {}, item.future_covariates or {}):
            for values in payload.values():
                for value in values:
                    if isinstance(value, str):
                        has_categorical = True
                    elif isinstance(value, bool):
                        continue
                    elif isinstance(value, (int, float)):
                        has_numeric = True
    if has_categorical:
        return "categorical"
    if has_numeric:
        return "numeric"
    return "numeric"
