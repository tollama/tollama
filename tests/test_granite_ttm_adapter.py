"""Unit tests for Granite TTM adapter without network/model downloads."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from tollama.core.schemas import ForecastRequest
from tollama.runners.torch_runner.errors import AdapterInputError
from tollama.runners.torch_runner.granite_ttm_adapter import (
    GraniteTTMAdapter,
    _GraniteDependencies,
)


class _FakeSeries(list[float]):
    def tolist(self) -> list[float]:
        return list(self)


class _FakeIloc:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows = rows

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self._rows[index]


class _FakeDataFrame:
    def __init__(self, payload: list[dict[str, Any]] | dict[str, list[Any]]) -> None:
        if isinstance(payload, dict):
            keys = list(payload.keys())
            length = len(next(iter(payload.values()))) if payload else 0
            self._rows = [
                {key: payload[key][index] for key in keys}
                for index in range(length)
            ]
        else:
            self._rows = [dict(row) for row in payload]

        self.columns = list(self._rows[0].keys()) if self._rows else []
        self.iloc = _FakeIloc(self._rows)

    def tail(self, count: int) -> _FakeDataFrame:
        return _FakeDataFrame(self._rows[-count:])

    def copy(self) -> _FakeDataFrame:
        return _FakeDataFrame(self._rows)

    def __getitem__(self, key: str) -> _FakeSeries:
        return _FakeSeries([row[key] for row in self._rows])


class _FakePandas:
    DataFrame = _FakeDataFrame

    @staticmethod
    def to_datetime(values: list[str], *, utc: bool, errors: str) -> list[datetime]:
        assert utc is True
        assert errors == "raise"
        parsed: list[datetime] = []
        for value in values:
            parsed.append(datetime.fromisoformat(value).replace(tzinfo=UTC))
        return parsed

    @staticmethod
    def date_range(*, start: datetime, periods: int, freq: str) -> list[datetime]:
        if freq != "D":
            raise ValueError("unsupported freq in fake pandas")
        return [start + timedelta(days=offset) for offset in range(periods)]


class _FakeTorchCuda:
    @staticmethod
    def is_available() -> bool:
        return False


@dataclass(frozen=True)
class _FakeTorch:
    cuda: _FakeTorchCuda = _FakeTorchCuda()


class _FakePreprocessor:
    last_kwargs: dict[str, Any] | None = None
    last_train_payload: _FakeDataFrame | None = None

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        _FakePreprocessor.last_kwargs = kwargs

    def train(self, payload: _FakeDataFrame) -> _FakePreprocessor:
        _FakePreprocessor.last_train_payload = payload
        return self


class _FakeModel:
    @classmethod
    def from_pretrained(cls, path_or_repo: str, revision: str | None = None) -> _FakeModel:
        assert path_or_repo
        if path_or_repo.startswith("ibm-granite/"):
            assert revision == "90-30-ft-l1-r2.1"
        return cls()


class _FakePipeline:
    last_payload: _FakeDataFrame | None = None
    last_future_time_series: _FakeDataFrame | None = None
    last_init_kwargs: dict[str, Any] | None = None

    def __init__(
        self,
        model: Any,
        *,
        device: str,
        feature_extractor: Any,
        freq: str,
        batch_size: int,
    ) -> None:
        del model, feature_extractor
        assert device == "cpu"
        _FakePipeline.last_init_kwargs = {"device": device, "freq": freq, "batch_size": batch_size}
        assert batch_size == 1

    def __call__(
        self,
        payload: _FakeDataFrame,
        *,
        future_time_series: _FakeDataFrame | None = None,
    ) -> _FakeDataFrame:
        _FakePipeline.last_payload = payload
        _FakePipeline.last_future_time_series = future_time_series
        return _FakeDataFrame(
            [{"target_prediction": [float(index) for index in range(30)]}],
        )


def _build_request(*, horizon: int) -> ForecastRequest:
    start = datetime(2025, 1, 1, tzinfo=UTC)
    timestamps = [(start + timedelta(days=index)).date().isoformat() for index in range(120)]
    target = [100.0 + float(index) for index in range(120)]
    return ForecastRequest.model_validate(
        {
            "model": "granite-ttm-r2",
            "horizon": horizon,
            "quantiles": [],
            "series": [
                {
                    "id": "s0",
                    "freq": "D",
                    "timestamps": timestamps,
                    "target": target,
                }
            ],
            "options": {"device": "cpu"},
        },
    )


def _build_request_with_covariates(*, horizon: int) -> ForecastRequest:
    request = _build_request(horizon=horizon).model_dump(mode="json", exclude_none=True)
    series = request["series"][0]
    target = series["target"]
    history_length = len(target)
    series["past_covariates"] = {
        "promo": [0.0 if index % 2 == 0 else 1.0 for index in range(history_length)],
        "temperature": [20.0 + float(index) for index in range(history_length)],
    }
    series["future_covariates"] = {"promo": [1.0 for _ in range(horizon)]}
    return ForecastRequest.model_validate(request)


def _fake_dependencies() -> _GraniteDependencies:
    return _GraniteDependencies(
        torch=_FakeTorch(),
        pandas=_FakePandas(),
        forecasting_pipeline_cls=_FakePipeline,
        preprocessor_cls=_FakePreprocessor,
        model_cls=_FakeModel,
    )


def test_granite_adapter_slices_to_horizon_and_sets_start_timestamp(monkeypatch) -> None:
    adapter = GraniteTTMAdapter()
    monkeypatch.setattr(adapter, "_resolve_dependencies", _fake_dependencies)

    response = adapter.forecast(
        _build_request(horizon=12),
        model_local_dir=None,
        model_source={
            "type": "huggingface",
            "repo_id": "ibm-granite/granite-timeseries-ttm-r2",
            "revision": "90-30-ft-l1-r2.1",
        },
        model_metadata={
            "implementation": "granite_ttm",
            "context_length": 90,
            "prediction_length": 30,
        },
    )

    assert response.model == "granite-ttm-r2"
    assert len(response.forecasts) == 1
    forecast = response.forecasts[0]
    assert forecast.id == "s0"
    assert forecast.mean == [float(index) for index in range(12)]
    assert forecast.start_timestamp == "2025-05-01T00:00:00Z"
    assert forecast.quantiles is None


def test_granite_adapter_errors_when_horizon_exceeds_prediction_length(monkeypatch) -> None:
    adapter = GraniteTTMAdapter()
    monkeypatch.setattr(adapter, "_resolve_dependencies", _fake_dependencies)

    with pytest.raises(AdapterInputError) as exc_info:
        adapter.forecast(
            _build_request(horizon=31),
            model_local_dir=None,
            model_source={
                "type": "huggingface",
                "repo_id": "ibm-granite/granite-timeseries-ttm-r2",
                "revision": "90-30-ft-l1-r2.1",
            },
            model_metadata={
                "implementation": "granite_ttm",
                "context_length": 90,
                "prediction_length": 30,
            },
        )

    assert "Requested horizon exceeds model prediction_length" in str(exc_info.value)


def test_granite_adapter_passes_future_time_series_and_covariate_columns(monkeypatch) -> None:
    adapter = GraniteTTMAdapter()
    monkeypatch.setattr(adapter, "_resolve_dependencies", _fake_dependencies)
    _FakePreprocessor.last_kwargs = None
    _FakePreprocessor.last_train_payload = None
    _FakePipeline.last_future_time_series = None
    _FakePipeline.last_init_kwargs = None

    response = adapter.forecast(
        _build_request_with_covariates(horizon=4),
        model_local_dir=None,
        model_source={
            "type": "huggingface",
            "repo_id": "ibm-granite/granite-timeseries-ttm-r2",
            "revision": "90-30-ft-l1-r2.1",
        },
        model_metadata={
            "implementation": "granite_ttm",
            "context_length": 90,
            "prediction_length": 30,
        },
    )

    assert response.model == "granite-ttm-r2"
    assert _FakePreprocessor.last_kwargs is not None
    assert _FakePreprocessor.last_kwargs["freq"] == "D"
    assert _FakePreprocessor.last_kwargs["control_columns"] == ["promo"]
    assert _FakePreprocessor.last_kwargs["conditional_columns"] == ["temperature"]
    assert _FakePreprocessor.last_train_payload is not None
    assert len(_FakePreprocessor.last_train_payload._rows) == 90
    assert _FakePipeline.last_init_kwargs == {"device": "cpu", "freq": "D", "batch_size": 1}
    assert _FakePipeline.last_future_time_series is not None
    assert len(_FakePipeline.last_future_time_series._rows) == 4
    assert set(_FakePipeline.last_future_time_series.columns) >= {"id", "timestamp", "promo"}
