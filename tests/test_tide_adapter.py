from __future__ import annotations

from typing import Any

import pytest

from tollama.core.schemas import ForecastRequest
from tollama.runners.tide_runner.adapter import TiDEAdapter
from tollama.runners.tide_runner.errors import AdapterInputError


class _FakeDate:
    def __init__(self, iso: str) -> None:
        self._iso = iso

    def isoformat(self) -> str:
        return self._iso


class _FakeDatetime:
    def __init__(self, iso: str) -> None:
        self._iso = iso


class _FakePandas:
    @staticmethod
    def to_datetime(values: list[str], utc: bool = True, errors: str = "raise") -> list[_FakeDatetime]:
        del utc, errors
        return [_FakeDatetime(values[0])]

    @staticmethod
    def date_range(start: Any, periods: int, freq: str) -> list[_FakeDate]:
        del start
        if freq != "D":
            raise ValueError("unsupported")
        return [_FakeDate(f"2025-01-{index:02d}T00:00:00+00:00") for index in range(1, periods + 1)]


class _FakeTimeSeries:
    def __init__(self, timestamps: list[str], values: list[Any], columns: list[str] | None = None) -> None:
        self.timestamps = timestamps
        self.values = values
        self.columns = columns
        self.static_covariates: dict[str, Any] | None = None

    @classmethod
    def from_times_and_values(
        cls,
        timestamps: list[str],
        values: list[Any],
        columns: list[str] | None = None,
    ) -> _FakeTimeSeries:
        return cls(timestamps, values, columns=columns)

    def with_static_covariates(self, static: dict[str, Any]) -> _FakeTimeSeries:
        self.static_covariates = static
        return self


class _FakePrediction:
    def __init__(self, values: list[list[float]]) -> None:
        self._values = values

    def values(self) -> list[list[float]]:
        return self._values


class _FakeTiDEModel:
    init_calls: list[dict[str, Any]] = []

    def __init__(self, **kwargs: Any) -> None:
        type(self).init_calls.append(kwargs)
        self.fit_calls: list[dict[str, Any]] = []
        self.predict_calls: list[dict[str, Any]] = []

    def fit(self, **kwargs: Any) -> None:
        self.fit_calls.append(kwargs)

    def predict(self, **kwargs: Any) -> _FakePrediction:
        self.predict_calls.append(kwargs)
        horizon = int(kwargs["n"])
        return _FakePrediction([[float(index) for index in range(10, 10 + horizon)]])


def _request() -> ForecastRequest:
    return ForecastRequest.model_validate(
        {
            "model": "tide",
            "horizon": 2,
            "quantiles": [0.1, 0.9],
            "series": [
                {
                    "id": "s1",
                    "freq": "D",
                    "timestamps": ["2025-01-01", "2025-01-02", "2025-01-03"],
                    "target": [1.0, 2.0, 3.0],
                    "past_covariates": {"promo": [0, 1, 0]},
                    "future_covariates": {"promo": [1, 1]},
                    "static_covariates": {"store": 1},
                }
            ],
            "options": {},
        },
    )


def test_tide_adapter_forecast_smoke() -> None:
    adapter = TiDEAdapter()
    _FakeTiDEModel.init_calls = []
    adapter._dependencies = type(
        "D",
        (),
        {"pandas": _FakePandas(), "timeseries_cls": _FakeTimeSeries, "tide_model_cls": _FakeTiDEModel},
    )()

    response = adapter.forecast(_request())

    assert response.model == "tide"
    assert len(response.forecasts) == 1
    assert response.forecasts[0].mean == [10.0, 11.0]
    assert response.forecasts[0].quantiles is None
    assert response.forecasts[0].start_timestamp == "2025-01-02T00:00:00Z"
    assert response.usage is not None
    assert response.usage["runner"] == "tollama-tide"
    assert response.usage["series_count"] == 1
    assert response.warnings is not None
    assert "deterministic mean forecasts only" in response.warnings[0]

    assert _FakeTiDEModel.init_calls
    assert _FakeTiDEModel.init_calls[0]["output_chunk_length"] == 2


def test_tide_adapter_rejects_invalid_frequency() -> None:
    adapter = TiDEAdapter()
    adapter._dependencies = type(
        "D",
        (),
        {"pandas": _FakePandas(), "timeseries_cls": _FakeTimeSeries, "tide_model_cls": _FakeTiDEModel},
    )()

    request = _request()
    request.series[0].freq = "BAD"

    with pytest.raises(AdapterInputError, match="invalid frequency"):
        adapter.forecast(request)


def test_tide_adapter_rejects_non_numeric_covariates() -> None:
    adapter = TiDEAdapter()
    adapter._dependencies = type(
        "D",
        (),
        {"pandas": _FakePandas(), "timeseries_cls": _FakeTimeSeries, "tide_model_cls": _FakeTiDEModel},
    )()

    request = _request()
    request.series[0].past_covariates = {"promo": ["A", "B", "C"]}

    with pytest.raises(AdapterInputError, match="must be numeric"):
        adapter.forecast(request)
