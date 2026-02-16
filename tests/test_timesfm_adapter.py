"""Unit tests for TimesFM adapter helper behavior without network access."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from tollama.runners.timesfm_runner.adapter import (
    generate_future_timestamps,
    map_quantile_forecast,
    point_forecast_to_rows,
    truncate_target_to_max_context,
)
from tollama.runners.timesfm_runner.errors import AdapterInputError


class _FakePandas:
    @staticmethod
    def to_datetime(values, *, utc: bool, errors: str):  # noqa: ANN001
        assert utc is True
        assert errors == "raise"
        return [datetime.fromisoformat(values[0]).replace(tzinfo=UTC)]

    @staticmethod
    def date_range(*, start: datetime, periods: int, freq: str):
        if freq != "D":
            raise ValueError("unsupported frequency in fake pandas")
        return [start + timedelta(days=index) for index in range(periods)]


def test_truncate_target_to_max_context_keeps_latest_window() -> None:
    result = truncate_target_to_max_context([1, 2, 3, 4, 5], 3)
    assert result == [3.0, 4.0, 5.0]


def test_truncate_target_to_max_context_rejects_too_short_input() -> None:
    with pytest.raises(AdapterInputError):
        truncate_target_to_max_context([1], 4)


def test_map_quantile_forecast_selects_requested_quantiles() -> None:
    quantile_forecast = [
        [
            [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            [0.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9],
        ]
    ]

    payloads = map_quantile_forecast(
        quantile_forecast=quantile_forecast,
        requested_quantiles=[0.1, 0.5, 0.9],
        n_series=1,
        horizon=2,
    )

    assert payloads == [
        {
            "0.1": [0.1, 1.1],
            "0.5": [0.5, 1.5],
            "0.9": [0.9, 1.9],
        }
    ]


def test_map_quantile_forecast_errors_when_quantile_missing() -> None:
    quantile_forecast = [
        [
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9],
        ]
    ]
    with pytest.raises(AdapterInputError) as exc_info:
        map_quantile_forecast(
            quantile_forecast=quantile_forecast,
            requested_quantiles=[0.95],
            n_series=1,
            horizon=2,
        )

    assert "requested quantile" in str(exc_info.value)


def test_generate_future_timestamps_daily_frequency() -> None:
    future = generate_future_timestamps(
        last_timestamp="2025-01-31",
        freq="D",
        horizon=3,
        pandas_module=_FakePandas(),
    )
    assert future == [
        "2025-02-01T00:00:00Z",
        "2025-02-02T00:00:00Z",
        "2025-02-03T00:00:00Z",
    ]


def test_point_forecast_to_rows_normalizes_two_dimensional_output() -> None:
    rows = point_forecast_to_rows(
        point_forecast=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        n_series=2,
        horizon=2,
    )
    assert rows == [[1.0, 2.0], [4.0, 5.0]]
