"""Deterministic gate validation for real-data E2E responses."""

from __future__ import annotations

from typing import Any

try:
    from .payload_builder import best_effort_warning_required
except ImportError:  # pragma: no cover - direct script import fallback
    import sys
    from pathlib import Path

    _THIS_DIR = Path(__file__).resolve().parent
    if str(_THIS_DIR) not in sys.path:
        sys.path.append(str(_THIS_DIR))
    from payload_builder import best_effort_warning_required

UNHANDLED_STATUS_CODES = {502, 503}


def extract_warnings(payload: dict[str, Any] | None) -> list[str]:
    """Extract normalized warning strings from a forecast response payload."""
    if not isinstance(payload, dict):
        return []

    warnings: list[str] = []
    for key in ("warnings",):
        values = payload.get(key)
        if isinstance(values, list):
            warnings.extend(_coerce_warning(value) for value in values)

    filtered = [warning for warning in warnings if warning]
    deduped: list[str] = []
    for warning in filtered:
        if warning not in deduped:
            deduped.append(warning)
    return deduped


def extract_metrics(payload: dict[str, Any] | None) -> dict[str, float]:
    """Extract aggregate metrics from a forecast response payload."""
    if not isinstance(payload, dict):
        return {}

    metrics = payload.get("metrics")
    if not isinstance(metrics, dict):
        return {}

    aggregate = metrics.get("aggregate")
    if not isinstance(aggregate, dict):
        return {}

    normalized: dict[str, float] = {}
    for key, value in aggregate.items():
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            normalized[str(key)] = float(value)
    return normalized


def validate_forecast_shape(payload: dict[str, Any] | None, *, horizon: int) -> str | None:
    """Return a validation error string for malformed successful responses."""
    if not isinstance(payload, dict):
        return "missing JSON forecast payload"

    forecasts = payload.get("forecasts")
    if not isinstance(forecasts, list) or not forecasts:
        return "missing non-empty forecasts list"

    for index, forecast in enumerate(forecasts):
        if not isinstance(forecast, dict):
            return f"forecasts[{index}] is not an object"
        mean = forecast.get("mean")
        if not isinstance(mean, list):
            return f"forecasts[{index}].mean is missing"
        if len(mean) != horizon:
            return (
                f"forecasts[{index}].mean length mismatch: "
                f"expected {horizon}, got {len(mean)}"
            )
    return None


def evaluate_gate(
    *,
    scenario: str,
    model: str,
    expected_status: int,
    http_status: int | None,
    response_payload: dict[str, Any] | None,
    horizon: int,
    exception_detail: str | None,
) -> tuple[bool, str | None, list[str], dict[str, float]]:
    """Evaluate one scenario/model run against deterministic gate rules."""
    warnings = extract_warnings(response_payload)
    metrics = extract_metrics(response_payload)

    if exception_detail and http_status is None:
        return (
            False,
            f"request failed before status was returned: {exception_detail}",
            warnings,
            metrics,
        )

    if http_status in UNHANDLED_STATUS_CODES:
        return False, f"unhandled runner error status {http_status}", warnings, metrics

    if http_status != expected_status:
        return (
            False,
            f"status mismatch for {scenario}: expected {expected_status}, got {http_status}",
            warnings,
            metrics,
        )

    if expected_status == 200:
        shape_error = validate_forecast_shape(response_payload, horizon=horizon)
        if shape_error is not None:
            return False, shape_error, warnings, metrics

    if scenario == "contract_best_effort_covariates" and best_effort_warning_required(model):
        if not warnings:
            return (
                False,
                "expected best_effort warnings for unsupported covariates but got none",
                warnings,
                metrics,
            )

    return True, None, warnings, metrics


def _coerce_warning(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    return ""
