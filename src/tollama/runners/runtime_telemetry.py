"""Shared runtime telemetry helpers for runner forecast responses."""

from __future__ import annotations

import resource
import sys
from typing import Any

from tollama.core.schemas import ForecastResponse, ForecastTiming


def enrich_forecast_response(
    *,
    response: ForecastResponse,
    runner_name: str,
    inference_ms: float,
) -> ForecastResponse:
    """Add additive timing and usage metadata to runner responses."""
    usage: dict[str, Any] = dict(response.usage or {})
    usage.setdefault("runner", runner_name)
    usage.setdefault("device", "unknown")
    usage.setdefault("peak_memory_mb", _peak_memory_mb() or 0.0)

    existing_timing = response.timing or ForecastTiming()
    model_load_ms = (
        float(existing_timing.model_load_ms)
        if existing_timing.model_load_ms is not None
        else 0.0
    )
    resolved_inference_ms = (
        float(existing_timing.inference_ms)
        if existing_timing.inference_ms is not None
        else max(float(inference_ms), 0.0)
    )
    total_ms = (
        float(existing_timing.total_ms)
        if existing_timing.total_ms is not None
        else resolved_inference_ms
    )

    return response.model_copy(
        update={
            "usage": usage,
            "timing": ForecastTiming(
                model_load_ms=model_load_ms,
                inference_ms=resolved_inference_ms,
                total_ms=max(total_ms, 0.0),
            ),
        },
    )


def _peak_memory_mb() -> float | None:
    try:
        peak = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except (AttributeError, ValueError, OSError):
        return None

    if peak <= 0.0:
        return None

    divisor = 1024.0 * 1024.0 if sys.platform == "darwin" else 1024.0
    return round(peak / divisor, 3)
