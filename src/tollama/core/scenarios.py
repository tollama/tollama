"""Scenario transform helpers for what-if forecasting."""

from __future__ import annotations

from tollama.core.schemas import CovariateValues, SeriesInput, WhatIfScenario, WhatIfTransform


def apply_scenario(
    *,
    series: list[SeriesInput],
    scenario: WhatIfScenario,
) -> list[SeriesInput]:
    """Apply one named scenario to a list of series inputs."""
    updated = [item.model_copy(deep=True) for item in series]

    for transform in scenario.transforms:
        matched = 0
        for index, item in enumerate(updated):
            if transform.series_id is not None and item.id != transform.series_id:
                continue
            updated[index] = _apply_transform_to_series(item=item, transform=transform)
            matched += 1

        if matched <= 0:
            raise ValueError(
                f"scenario {scenario.name!r} transform did not match any series "
                f"(series_id={transform.series_id!r})",
            )

    return updated


def _apply_transform_to_series(*, item: SeriesInput, transform: WhatIfTransform) -> SeriesInput:
    if transform.field == "target":
        return item.model_copy(update={"target": _apply_target_transform(item.target, transform)})

    assert transform.key is not None
    if transform.field == "past_covariates":
        covariates = dict(item.past_covariates or {})
        if transform.key not in covariates:
            raise ValueError(
                f"scenario transform references missing past covariate {transform.key!r} "
                f"for series {item.id!r}",
            )
        covariates[transform.key] = _apply_covariate_transform(covariates[transform.key], transform)
        return item.model_copy(update={"past_covariates": covariates})

    covariates = dict(item.future_covariates or {})
    if transform.key not in covariates:
        raise ValueError(
            f"scenario transform references missing future covariate {transform.key!r} "
            f"for series {item.id!r}",
        )
    covariates[transform.key] = _apply_covariate_transform(covariates[transform.key], transform)
    return item.model_copy(update={"future_covariates": covariates})


def _apply_target_transform(values: list[int | float], transform: WhatIfTransform) -> list[float]:
    operation = transform.operation
    if operation == "replace":
        replacement = _numeric_value(transform.value, operation=operation)
        return [replacement for _ in values]

    adjustment = _numeric_value(transform.value, operation=operation)
    transformed: list[float] = []
    for value in values:
        base = _numeric_value(value, operation=operation)
        if operation == "multiply":
            transformed.append(base * adjustment)
        elif operation == "add":
            transformed.append(base + adjustment)
        else:  # pragma: no cover - guarded by schema validation
            raise ValueError(f"unsupported operation {operation!r}")
    return transformed


def _apply_covariate_transform(
    values: CovariateValues,
    transform: WhatIfTransform,
) -> CovariateValues:
    operation = transform.operation
    if operation == "replace":
        return [transform.value for _ in values]

    adjustment = _numeric_value(transform.value, operation=operation)
    transformed: list[int | float] = []
    for value in values:
        base = _numeric_value(value, operation=operation)
        if operation == "multiply":
            transformed.append(base * adjustment)
        elif operation == "add":
            transformed.append(base + adjustment)
        else:  # pragma: no cover - guarded by schema validation
            raise ValueError(f"unsupported operation {operation!r}")
    return transformed


def _numeric_value(value: object, *, operation: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"operation {operation!r} requires numeric values")
    return float(value)
