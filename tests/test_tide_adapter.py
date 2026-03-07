from __future__ import annotations

from typing import Any

from tollama.core.schemas import ForecastRequest
from tollama.runners.tide_runner.adapter import TideAdapter


class _FakeDate:
    def __init__(self, iso: str) -> None:
        self._iso = iso

    def isoformat(self) -> str:
        return self._iso


class _FakePandas:
    @staticmethod
    def to_datetime(values: list[str], utc: bool = True, errors: str = "raise") -> list[str]:
        del utc, errors
        return values

    @staticmethod
    def date_range(start: Any, periods: int, freq: str) -> list[_FakeDate]:
        del start
        if freq != "D":
            raise ValueError("unsupported")
        return [_FakeDate(f"2025-01-{index:02d}T00:00:00+00:00") for index in range(1, periods + 1)]


class _FakeTimeSeries:
    def __init__(self, values: list[float]) -> None:
        self._values = values

    @classmethod
    def from_times_and_values(cls, times: Any, values: list[float], freq: str) -> _FakeTimeSeries:
        del times, freq
        return cls(values)

    def values(self) -> list[list[float]]:
        return [[value] for value in self._values]


class _FakePrediction:
    def __init__(self, mean: list[float], quantiles: dict[float, list[float]] | None) -> None:
        self._mean = mean
        self._quantiles = quantiles

    def values(self) -> list[list[float]]:
        return [[value] for value in self._mean]

    def quantile_timeseries(self, quantile: float) -> _FakeTimeSeries:
        if self._quantiles is None or quantile not in self._quantiles:
            raise ValueError("missing")
        return _FakeTimeSeries(self._quantiles[quantile])


class _FakeModelWithQuantiles:
    def predict(self, n: int, series: Any, num_samples: int) -> _FakePrediction:
        del n, series
        if num_samples == 1:
            return _FakePrediction([10.0, 11.0], None)
        return _FakePrediction([10.0, 11.0], {0.1: [9.0, 10.0], 0.9: [11.0, 12.0]})


class _FakeModelNoQuantiles:
    def predict(self, n: int, series: Any, num_samples: int) -> _FakePrediction:
        del n, series, num_samples
        return _FakePrediction([10.0, 11.0], None)


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
                }
            ],
            "options": {},
        },
    )


def _deps() -> Any:
    return type(
        "D",
        (),
        {
            "np": object(),
            "pd": _FakePandas(),
            "time_series_cls": _FakeTimeSeries,
            "tide_model_cls": object,
        },
    )()


def test_tide_adapter_returns_quantiles_when_runtime_supports_probabilistic_output(
    monkeypatch,
) -> None:
    adapter = TideAdapter()
    monkeypatch.setattr(adapter, "_resolve_dependencies", _deps)
    monkeypatch.setattr(adapter, "_get_or_load_model", lambda **kwargs: _FakeModelWithQuantiles())

    response = adapter.forecast(_request())

    assert response.forecasts[0].mean == [10.0, 11.0]
    assert response.forecasts[0].quantiles == {"0.1": [9.0, 10.0], "0.9": [11.0, 12.0]}
    assert response.warnings is None


def test_tide_adapter_falls_back_to_mean_only_when_quantiles_unavailable(monkeypatch) -> None:
    adapter = TideAdapter()
    monkeypatch.setattr(adapter, "_resolve_dependencies", _deps)
    monkeypatch.setattr(adapter, "_get_or_load_model", lambda **kwargs: _FakeModelNoQuantiles())

    response = adapter.forecast(_request())

    assert response.forecasts[0].mean == [10.0, 11.0]
    assert response.forecasts[0].quantiles is None
    assert response.warnings is not None
    assert "did not expose quantile outputs" in response.warnings[0]
