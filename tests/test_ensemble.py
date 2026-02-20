"""Tests for ensemble forecast merge helpers."""

from __future__ import annotations

import pytest

from tollama.core.ensemble import EnsembleError, merge_forecast_responses
from tollama.core.schemas import ForecastResponse


def _response(model: str, values: list[float]) -> ForecastResponse:
    return ForecastResponse.model_validate(
        {
            "model": model,
            "forecasts": [
                {
                    "id": "s1",
                    "freq": "D",
                    "start_timestamp": "2025-01-04",
                    "mean": values,
                }
            ],
        }
    )


def test_merge_forecast_responses_weighted_mean() -> None:
    merged = merge_forecast_responses(
        [_response("a", [1.0, 3.0]), _response("b", [3.0, 5.0])],
        weights={"a": 1.0, "b": 3.0},
        method="mean",
    )

    assert merged.model == "ensemble"
    assert merged.forecasts[0].mean == [2.5, 4.5]


def test_merge_forecast_responses_weighted_median() -> None:
    merged = merge_forecast_responses(
        [_response("a", [1.0]), _response("b", [2.0]), _response("c", [10.0])],
        weights={"a": 1.0, "b": 2.0, "c": 1.0},
        method="median",
    )

    assert merged.forecasts[0].mean == [2.0]


def test_merge_forecast_responses_rejects_mismatched_series_ids() -> None:
    first = ForecastResponse.model_validate(
        {
            "model": "a",
            "forecasts": [
                {
                    "id": "left",
                    "freq": "D",
                    "start_timestamp": "2025-01-04",
                    "mean": [1.0],
                }
            ],
        }
    )
    second = ForecastResponse.model_validate(
        {
            "model": "b",
            "forecasts": [
                {
                    "id": "right",
                    "freq": "D",
                    "start_timestamp": "2025-01-04",
                    "mean": [1.0],
                }
            ],
        }
    )

    with pytest.raises(EnsembleError):
        merge_forecast_responses([first, second])
