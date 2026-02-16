"""Unit tests for Uni2TS adapter helper behavior without network access."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import pytest

from tollama.core.schemas import SeriesInput
from tollama.runners.uni2ts_runner.adapter import (
    build_pandas_dataset,
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
        return [datetime.fromisoformat(value).replace(tzinfo=UTC) for value in values]

    @staticmethod
    def date_range(*, start: datetime, periods: int, freq: str):
        if freq != "D":
            raise ValueError("unsupported frequency")
        return [start + timedelta(days=index) for index in range(periods)]

    @staticmethod
    def DataFrame(payload, index=None):  # noqa: ANN001
        return {"payload": payload, "index": index}


class _FakeNumpy:
    @staticmethod
    def asarray(values, dtype=float):  # noqa: ANN001
        del dtype
        return _FakeArray([float(value) for value in values])


class _FakeArray(list[float]):
    def tolist(self) -> list[float]:
        return list(self)


class _CapturingPandasDataset:
    def __init__(self, frames, **kwargs):  # noqa: ANN001, ANN003
        self.frames = frames
        self.kwargs = kwargs
        self.num_feat_dynamic_real = len(kwargs.get("feat_dynamic_real") or [])
        self.num_past_feat_dynamic_real = len(kwargs.get("past_feat_dynamic_real") or [])


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


def test_build_pandas_dataset_maps_dynamic_and_past_dynamic_covariates() -> None:
    series = SeriesInput.model_validate(
        {
            "id": "s1",
            "freq": "D",
            "timestamps": ["2025-01-01", "2025-01-02", "2025-01-03"],
            "target": [10.0, 11.0, 12.0],
            "past_covariates": {
                "promo": [0.0, 1.0, 0.0],
                "temperature": [20.0, 21.0, 22.0],
            },
            "future_covariates": {
                "promo": [1.0, 1.0],
            },
        },
    )

    result = build_pandas_dataset(
        series_list=[series],
        pandas_module=_FakePandas(),
        numpy_module=_FakeNumpy(),
        pandas_dataset_cls=_CapturingPandasDataset,
        horizon=2,
    )

    assert result.feat_dynamic_real_dim == 1
    assert result.past_feat_dynamic_real_dim == 1
    dataset = result.dataset
    assert dataset.kwargs["feat_dynamic_real"] == ["promo"]
    assert dataset.kwargs["past_feat_dynamic_real"] == ["temperature"]
    frame = dataset.frames["s1"]["payload"]
    assert len(frame["target"]) == 5
    assert frame["promo"] == [0.0, 1.0, 0.0, 1.0, 1.0]
    assert frame["temperature"][:3] == [20.0, 21.0, 22.0]
