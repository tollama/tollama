"""Unit tests for Sundial adapter helper behavior without network access."""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

from tollama.runners.sundial_runner.adapter import (
    compute_sample_statistics,
    generate_future_timestamps,
    truncate_target_to_max_context,
)


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


class _FakeTensor:
    def __init__(self, values):  # noqa: ANN001
        self._values = _deep_copy(values)

    def mean(self, dim: int):
        if dim != 0:
            raise AssertionError("fake tensor only supports dim=0")
        rows = _matrix(self._values)
        horizon = len(rows[0])
        means = [
            sum(row[column] for row in rows) / float(len(rows))
            for column in range(horizon)
        ]
        return _FakeTensor(means)

    def detach(self) -> _FakeTensor:
        return self

    def cpu(self) -> _FakeTensor:
        return self

    def tolist(self):  # noqa: ANN001
        return _deep_copy(self._values)


class _FakeTorch:
    float32 = "float32"

    @staticmethod
    def tensor(values, dtype=None):  # noqa: ANN001
        del dtype
        return _FakeTensor(values)

    @staticmethod
    def quantile(sample_tensor, quantiles, dim: int):  # noqa: ANN001
        if dim != 0:
            raise AssertionError("fake torch only supports dim=0")
        rows = _matrix(sample_tensor.tolist())
        q_values = _vector(quantiles.tolist() if hasattr(quantiles, "tolist") else quantiles)
        horizon = len(rows[0])
        result: list[list[float]] = []
        for quantile in q_values:
            quantile_row: list[float] = []
            for column in range(horizon):
                ordered = sorted(row[column] for row in rows)
                quantile_row.append(_linear_quantile(ordered, quantile))
            result.append(quantile_row)
        return _FakeTensor(result)


def test_truncate_target_to_max_context_keeps_latest_window() -> None:
    result = truncate_target_to_max_context([1, 2, 3, 4, 5], 3)
    assert result == [3.0, 4.0, 5.0]


def test_compute_sample_statistics_maps_requested_quantiles() -> None:
    mean, quantiles = compute_sample_statistics(
        samples=[
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
            [3.0, 4.0, 5.0],
            [4.0, 5.0, 6.0],
        ],
        requested_quantiles=[0.25, 0.5, 0.75],
        horizon=3,
        torch_module=_FakeTorch(),
    )

    assert mean == [2.5, 3.5, 4.5]
    assert quantiles == {
        "0.25": [1.75, 2.75, 3.75],
        "0.5": [2.5, 3.5, 4.5],
        "0.75": [3.25, 4.25, 5.25],
    }


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


def _deep_copy(value):  # noqa: ANN001
    if isinstance(value, list):
        return [_deep_copy(item) for item in value]
    return value


def _vector(values):  # noqa: ANN001
    return [float(value) for value in values]


def _matrix(values):  # noqa: ANN001
    rows = values if isinstance(values, list) else list(values)
    normalized: list[list[float]] = []
    for row in rows:
        normalized.append([float(item) for item in row])
    return normalized


def _linear_quantile(values: list[float], quantile: float) -> float:
    if len(values) == 1:
        return values[0]
    index = (len(values) - 1) * float(quantile)
    lower = int(math.floor(index))
    upper = int(math.ceil(index))
    if lower == upper:
        return values[lower]
    weight = index - lower
    return (1.0 - weight) * values[lower] + weight * values[upper]
