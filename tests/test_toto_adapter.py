"""Unit tests for Toto adapter helper behavior without network access."""

from __future__ import annotations

import math
from datetime import UTC, datetime

import pytest

from tollama.core.schemas import SeriesInput
from tollama.runners.toto_runner.adapter import (
    build_toto_variates,
    choose_samples_per_batch,
    generate_future_timestamps_from_interval,
    quantiles_from_target_samples,
    timestamps_to_unix_seconds,
    truncate_multivariate_to_max_context,
)
from tollama.runners.toto_runner.errors import AdapterInputError


class _FakePandas:
    @staticmethod
    def to_datetime(values, *, utc: bool, errors: str):  # noqa: ANN001
        assert utc is True
        assert errors == "raise"
        parsed: list[datetime] = []
        for value in values:
            parsed.append(datetime.fromisoformat(value).replace(tzinfo=UTC))
        return parsed


class _FakeTensor:
    def __init__(self, values):  # noqa: ANN001
        self._values = _deep_copy(values)

    def detach(self) -> _FakeTensor:
        return self

    def cpu(self) -> _FakeTensor:
        return self

    def tolist(self):  # noqa: ANN001
        return _deep_copy(self._values)


class _FakeTorch:
    float32 = "float32"

    @staticmethod
    def as_tensor(values, dtype=None, device=None):  # noqa: ANN001
        del dtype, device
        return _FakeTensor(values)

    @staticmethod
    def quantile(sample_tensor, quantiles, dim: int):  # noqa: ANN001
        if dim != 1:
            raise AssertionError("fake torch only supports dim=1")
        rows = _matrix(sample_tensor.tolist())
        if isinstance(quantiles, (float, int)):
            return _FakeTensor([_linear_quantile(sorted(row), float(quantiles)) for row in rows])

        values = quantiles.tolist() if hasattr(quantiles, "tolist") else quantiles
        out: list[list[float]] = []
        for value in values:
            out.append([_linear_quantile(sorted(row), float(value)) for row in rows])
        return _FakeTensor(out)


def test_truncate_multivariate_to_max_context_keeps_recent_window() -> None:
    matrix, timestamps = truncate_multivariate_to_max_context(
        matrix=[[1.0, 2.0, 3.0, 4.0], [11.0, 12.0, 13.0, 14.0]],
        timestamps=[10, 20, 30, 40],
        max_context=3,
    )
    assert matrix == [[2.0, 3.0, 4.0], [12.0, 13.0, 14.0]]
    assert timestamps == [20, 30, 40]


def test_truncate_multivariate_to_max_context_rejects_alignment_mismatch() -> None:
    with pytest.raises(AdapterInputError):
        truncate_multivariate_to_max_context(
            matrix=[[1.0, 2.0, 3.0], [11.0, 12.0]],
            timestamps=[10, 20, 30],
            max_context=3,
        )


def test_choose_samples_per_batch_picks_largest_divisor_at_or_below_desired() -> None:
    assert choose_samples_per_batch(num_samples=256, desired=128) == 128
    assert choose_samples_per_batch(num_samples=100, desired=64) == 50
    assert choose_samples_per_batch(num_samples=63, desired=10) == 9


def test_timestamps_to_unix_seconds_and_future_generation() -> None:
    parsed = timestamps_to_unix_seconds(
        timestamps=["2025-01-01T00:00:00", "2025-01-02T00:00:00", "2025-01-03T00:00:00"],
        pandas_module=_FakePandas(),
    )
    assert parsed == [1735689600, 1735776000, 1735862400]

    future = generate_future_timestamps_from_interval(
        last_timestamp="2025-01-03T00:00:00",
        interval_seconds=86400,
        horizon=3,
        pandas_module=_FakePandas(),
    )
    assert future == [
        "2025-01-04T00:00:00Z",
        "2025-01-05T00:00:00Z",
        "2025-01-06T00:00:00Z",
    ]


def test_build_toto_variates_includes_target_plus_numeric_covariates() -> None:
    series = SeriesInput.model_validate(
        {
            "id": "s1",
            "freq": "D",
            "timestamps": ["2025-01-01", "2025-01-02", "2025-01-03"],
            "target": [10.0, 11.0, 12.0],
            "past_covariates": {"promo": [0.0, 1.0, 0.0]},
        },
    )

    variates, timestamps, interval = build_toto_variates(
        series=series,
        max_context=16,
        pandas_module=_FakePandas(),
    )
    assert variates == [[10.0, 11.0, 12.0], [0.0, 1.0, 0.0]]
    assert timestamps == [1735689600, 1735776000, 1735862400]
    assert interval == 86400


def test_build_toto_variates_rejects_covariate_length_mismatch() -> None:
    series = SeriesInput.model_construct(
        id="s1",
        freq="D",
        timestamps=["2025-01-01", "2025-01-02", "2025-01-03"],
        target=[10.0, 11.0, 12.0],
        past_covariates={"promo": [0.0, 1.0]},
        future_covariates=None,
        static_covariates=None,
    )
    with pytest.raises(AdapterInputError):
        _ = build_toto_variates(
            series=series,
            max_context=16,
            pandas_module=_FakePandas(),
        )


def test_quantiles_from_target_samples_computes_requested_quantiles() -> None:
    payload = quantiles_from_target_samples(
        target_samples=[
            [1.0, 2.0, 3.0, 4.0],
            [2.0, 3.0, 4.0, 5.0],
            [3.0, 4.0, 5.0, 6.0],
        ],
        requested_quantiles=[0.1, 0.5, 0.9],
        torch_module=_FakeTorch(),
        horizon=3,
    )
    assert payload == {
        "0.1": [1.3, 2.3, 3.3],
        "0.5": [2.5, 3.5, 4.5],
        "0.9": [3.7, 4.7, 5.7],
    }


def _deep_copy(value):  # noqa: ANN001
    if isinstance(value, list):
        return [_deep_copy(item) for item in value]
    return value


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
