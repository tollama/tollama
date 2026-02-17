"""Unit tests for TimesFM adapter helper behavior without network access."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from tollama.core.schemas import ForecastRequest
from tollama.runners.timesfm_runner.adapter import (
    TimesFMAdapter,
    _TimesFMDependencies,
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


class _AmbiguousDatetimeIndex:
    def __init__(self, values: list[datetime]) -> None:
        self._values = values

    def __len__(self) -> int:
        return len(self._values)

    def __getitem__(self, index: int) -> datetime:
        return self._values[index]

    def __bool__(self) -> bool:
        raise ValueError("ambiguous truth value")


class _FakePandasAmbiguousBool:
    @staticmethod
    def to_datetime(values, *, utc: bool, errors: str):  # noqa: ANN001
        assert utc is True
        assert errors == "raise"
        return _AmbiguousDatetimeIndex([datetime.fromisoformat(values[0]).replace(tzinfo=UTC)])

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


def test_generate_future_timestamps_handles_ambiguous_index_truthiness() -> None:
    future = generate_future_timestamps(
        last_timestamp="2025-01-31",
        freq="D",
        horizon=2,
        pandas_module=_FakePandasAmbiguousBool(),
    )
    assert future == [
        "2025-02-01T00:00:00Z",
        "2025-02-02T00:00:00Z",
    ]


def test_point_forecast_to_rows_normalizes_two_dimensional_output() -> None:
    rows = point_forecast_to_rows(
        point_forecast=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        n_series=2,
        horizon=2,
    )
    assert rows == [[1.0, 2.0], [4.0, 5.0]]


class _FakeNumpy:
    @staticmethod
    def asarray(values, dtype=float):  # noqa: ANN001
        del dtype
        return [float(value) for value in values]


class _FakeForecastConfig:
    def __init__(self, **kwargs):  # noqa: ANN003
        self.kwargs = kwargs


class _FakeTimesFMModel:
    def __init__(self) -> None:
        self.compile_kwargs: dict[str, object] | None = None
        self.covariate_kwargs: dict[str, object] | None = None

    @classmethod
    def from_pretrained(cls, model_ref: str, **kwargs):  # noqa: ANN003
        del model_ref, kwargs
        return cls()

    def compile(self, forecast_config: _FakeForecastConfig) -> None:
        self.compile_kwargs = dict(forecast_config.kwargs)

    def forecast(self, *, horizon: int, inputs):  # noqa: ANN001
        del inputs
        return [[float(index + 1) for index in range(horizon)]]

    def forecast_with_covariates(self, **kwargs):  # noqa: ANN003
        self.covariate_kwargs = dict(kwargs)
        horizon = int(kwargs.get("horizon", 1))
        return [[float(index + 1) for index in range(horizon)]]


class _FakeTimesFMModule:
    ForecastConfig = _FakeForecastConfig
    TimesFM_2p5_200M_torch = _FakeTimesFMModel


class _FakeTimesFMNaNModel(_FakeTimesFMModel):
    def forecast(self, *, horizon: int, inputs):  # noqa: ANN001
        del inputs
        return [[float("nan") for _ in range(horizon)]]


class _FakeTimesFMNaNModule:
    ForecastConfig = _FakeForecastConfig
    TimesFM_2p5_200M_torch = _FakeTimesFMNaNModel


def _covariate_request() -> ForecastRequest:
    return ForecastRequest.model_validate(
        {
            "model": "timesfm-2.5-200m",
            "horizon": 2,
            "quantiles": [],
            "series": [
                {
                    "id": "s1",
                    "freq": "D",
                    "timestamps": ["2025-01-01", "2025-01-02", "2025-01-03"],
                    "target": [10.0, 11.0, 12.0],
                    "past_covariates": {
                        "promo": [0.0, 1.0, 0.0],
                        "event": ["off", "on", "off"],
                    },
                    "future_covariates": {
                        "promo": [1.0, 1.0],
                        "event": ["on", "off"],
                    },
                }
            ],
            "options": {},
            "parameters": {
                "timesfm": {
                    "xreg_mode": "xreg + timesfm",
                    "ridge": 0.25,
                    "force_on_cpu": True,
                }
            },
        },
    )


def test_timesfm_adapter_uses_forecast_with_covariates_and_return_backcast(monkeypatch) -> None:
    adapter = TimesFMAdapter()
    monkeypatch.setattr(
        adapter,
        "_resolve_dependencies",
        lambda: _TimesFMDependencies(
            numpy=_FakeNumpy(),
            pandas=_FakePandas(),
            timesfm=_FakeTimesFMModule(),
        ),
    )

    response = adapter.forecast(_covariate_request())

    assert response.model == "timesfm-2.5-200m"
    assert len(response.forecasts) == 1
    compiled_entry = next(iter(adapter._compiled_models.values()))  # noqa: SLF001
    model = compiled_entry.model
    assert isinstance(model, _FakeTimesFMModel)
    assert model.compile_kwargs is not None
    assert model.compile_kwargs.get("return_backcast") is True
    assert model.covariate_kwargs is not None
    dynamic = model.covariate_kwargs["dynamic_numerical_covariates"]
    assert dynamic["promo"][0] == [0.0, 1.0, 0.0, 1.0, 1.0]
    assert model.covariate_kwargs["xreg_mode"] == "xreg + timesfm"
    assert model.covariate_kwargs["ridge"] == 0.25
    assert model.covariate_kwargs["force_on_cpu"] is True
    assert response.warnings is not None
    assert "Ignoring categorical covariate" in response.warnings[0]


def test_timesfm_adapter_replaces_non_finite_forecasts(monkeypatch) -> None:
    adapter = TimesFMAdapter()
    monkeypatch.setattr(
        adapter,
        "_resolve_dependencies",
        lambda: _TimesFMDependencies(
            numpy=_FakeNumpy(),
            pandas=_FakePandas(),
            timesfm=_FakeTimesFMNaNModule(),
        ),
    )
    request = ForecastRequest.model_validate(
        {
            "model": "timesfm-2.5-200m",
            "horizon": 2,
            "quantiles": [],
            "series": [
                {
                    "id": "s1",
                    "freq": "D",
                    "timestamps": ["2025-01-01", "2025-01-02", "2025-01-03"],
                    "target": [10.0, 11.0, 12.0],
                }
            ],
            "options": {},
        },
    )

    response = adapter.forecast(request)

    assert response.forecasts[0].mean == [12.0, 12.0]
    assert response.warnings is not None
    assert "non-finite forecast values" in response.warnings[-1]
