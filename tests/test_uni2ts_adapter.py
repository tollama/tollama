"""Unit tests for Uni2TS adapter helper behavior without network access."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import pytest

from tollama.core.schemas import SeriesInput
from tollama.runners.uni2ts_runner.adapter import (
    build_quantile_payload,
    generate_future_timestamps,
    normalize_forecast_vector,
    resolve_context_length,
)
from tollama.runners.uni2ts_runner.errors import AdapterInputError


class _FakePandas:
    @staticmethod
    def to_datetime(values, *, utc: bool, errors: str):  # noqa: ANN001
        assert utc is True
        assert errors == "raise"
        return [datetime.fromisoformat(values[0]).replace(tzinfo=UTC)]

    @staticmethod
    def date_range(*, start: datetime, periods: int, freq: str):
        if freq != "D":
            raise ValueError("unsupported frequency")
        return [start + timedelta(days=index) for index in range(periods)]


@dataclass(frozen=True)
class _FakeForecast:
    mean: list[float]
    quantiles: dict[float, list[float]]

    def quantile(self, q: float) -> list[float]:
        return self.quantiles[q]


def _series(length: int) -> SeriesInput:
    start = datetime(2025, 1, 1, tzinfo=UTC)
    return SeriesInput.model_validate(
        {
            "id": "s1",
            "freq": "D",
            "timestamps": [(start + timedelta(days=i)).date().isoformat() for i in range(length)],
            "target": [100.0 + float(i) for i in range(length)],
        },
    )


def test_resolve_context_length_clamps_to_available_history() -> None:
    context = resolve_context_length(
        option_value=500,
        default_context_length=200,
        series_list=[_series(120), _series(80)],
    )
    assert context == 80


def test_resolve_context_length_rejects_too_short_series() -> None:
    with pytest.raises(AdapterInputError):
        resolve_context_length(
            option_value=None,
            default_context_length=200,
            series_list=[_series(1)],
        )


def test_normalize_forecast_vector_limits_to_horizon() -> None:
    normalized = normalize_forecast_vector([1, 2, 3, 4], horizon=3, label="mean")
    assert normalized == [1.0, 2.0, 3.0]


def test_build_quantile_payload_maps_requested_quantiles() -> None:
    forecast = _FakeForecast(
        mean=[1.0, 2.0, 3.0],
        quantiles={
            0.1: [0.8, 1.8, 2.8],
            0.5: [1.0, 2.0, 3.0],
            0.9: [1.2, 2.2, 3.2],
        },
    )
    payload = build_quantile_payload(
        forecast=forecast,
        requested_quantiles=[0.1, 0.9],
        horizon=3,
    )
    assert payload == {
        "0.1": [0.8, 1.8, 2.8],
        "0.9": [1.2, 2.2, 3.2],
    }


def test_generate_future_timestamps_daily() -> None:
    timestamps = generate_future_timestamps(
        last_timestamp="2025-01-31",
        freq="D",
        horizon=3,
        pandas_module=_FakePandas(),
    )
    assert timestamps == [
        "2025-02-01T00:00:00Z",
        "2025-02-02T00:00:00Z",
        "2025-02-03T00:00:00Z",
    ]
