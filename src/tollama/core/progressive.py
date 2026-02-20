"""Helpers for progressive multi-stage forecasting model selection."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from tollama.core.auto_select import select_auto_models
from tollama.core.schemas import SeriesInput

ProgressiveStrategy = Literal["fastest", "best_accuracy", "explicit"]


@dataclass(frozen=True, slots=True)
class ProgressiveStage:
    """One stage in a progressive forecast execution plan."""

    index: int
    strategy: ProgressiveStrategy
    label: str
    model: str
    family: str | None


def build_progressive_stages(
    *,
    series: list[SeriesInput],
    horizon: int,
    include_models: list[str] | tuple[str, ...] | set[str],
    preferred_model: str | None = None,
    family_by_model: Mapping[str, str] | None = None,
) -> tuple[ProgressiveStage, ...]:
    """Build deterministic progressive stages: fastest pass, then refined pass."""
    candidates = _normalize_models(include_models)
    if not candidates:
        raise ValueError("include_models must include at least one model")

    if preferred_model is not None:
        normalized_preferred = preferred_model.strip()
        if normalized_preferred not in candidates:
            raise ValueError(f"preferred_model {preferred_model!r} is not in include_models")
        return (
            ProgressiveStage(
                index=1,
                strategy="explicit",
                label="explicit-pass",
                model=normalized_preferred,
                family=_resolve_family(family_by_model=family_by_model, model=normalized_preferred),
            ),
        )

    fastest_selection = select_auto_models(
        series=series,
        horizon=horizon,
        strategy="fastest",
        include_models=candidates,
        allow_restricted_license=True,
    )
    refined_selection = select_auto_models(
        series=series,
        horizon=horizon,
        strategy="best_accuracy",
        include_models=candidates,
        allow_restricted_license=True,
    )

    stage_specs: list[tuple[ProgressiveStrategy, str, str]] = [
        ("fastest", "fast-pass", fastest_selection.chosen_model),
    ]
    if refined_selection.chosen_model != fastest_selection.chosen_model:
        stage_specs.append(("best_accuracy", "refined-pass", refined_selection.chosen_model))

    stages: list[ProgressiveStage] = []
    for index, (strategy, label, model) in enumerate(stage_specs, start=1):
        stages.append(
            ProgressiveStage(
                index=index,
                strategy=strategy,
                label=label,
                model=model,
                family=_resolve_family(family_by_model=family_by_model, model=model),
            ),
        )
    return tuple(stages)


def _normalize_models(models: list[str] | tuple[str, ...] | set[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    ordered: list[str] = []
    for raw in models:
        if not isinstance(raw, str):
            continue
        normalized = raw.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(normalized)
    return tuple(ordered)


def _resolve_family(*, family_by_model: Mapping[str, str] | None, model: str) -> str | None:
    if family_by_model is None:
        return None
    raw = family_by_model.get(model)
    if isinstance(raw, str):
        normalized = raw.strip()
        if normalized:
            return normalized
    return None
