from __future__ import annotations

from typing import Any

from tollama.core.schemas import ForecastRequest
from tollama.runners.nbeatsx_runner.adapter import NbeatsxAdapter


class _FakeDate:
    def __init__(self, iso: str) -> None:
        self._iso = iso

    def isoformat(self) -> str:
        return self._iso


class _FakeDataFrame:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows

    def __getitem__(self, key: str) -> list[Any]:
        return [row.get(key) for row in self.rows]

    def __setitem__(self, key: str, values: list[Any]) -> None:
        for row, value in zip(self.rows, values, strict=True):
            row[key] = value

    def to_dict(self, orient: str) -> list[dict[str, Any]]:
        assert orient == "records"
        return self.rows


class _FakePandas:
    @staticmethod
    def DataFrame(rows: list[dict[str, Any]]) -> _FakeDataFrame:
        return _FakeDataFrame(rows)

    @staticmethod
    def to_datetime(values: list[str], utc: bool = True, errors: str = "raise") -> list[str]:
        del utc, errors
        return values

    @staticmethod
    def infer_freq(values: list[str]) -> str | None:
        del values
        return "D"

    @staticmethod
    def date_range(start: Any, periods: int, freq: str) -> list[_FakeDate]:
        del start
        if freq not in {"D", "H"}:
            raise ValueError("unsupported")
        return [_FakeDate(f"2025-01-{index:02d}T00:00:00+00:00") for index in range(1, periods + 1)]


class _FakeNF:
    def __init__(self, models: list[Any], freq: str) -> None:
        self._models = models
        self._freq = freq
        self.fit_calls: list[dict[str, Any]] = []
        self.predict_calls: list[dict[str, Any]] = []

    def fit(self, train_df: _FakeDataFrame, static_df: _FakeDataFrame | None = None) -> None:
        self.fit_calls.append({"train_df": train_df, "static_df": static_df})

    def predict(self, h: int | None = None, futr_df: _FakeDataFrame | None = None) -> _FakeDataFrame:
        self.predict_calls.append({"h": h, "futr_df": futr_df})
        return _FakeDataFrame(
            [
                {"unique_id": "s1", "ds": "2025-01-04", "NBEATSx": 10.0},
                {"unique_id": "s1", "ds": "2025-01-05", "NBEATSx": 11.0},
                {"unique_id": "s2", "ds": "2025-01-04", "NBEATSx": 20.0},
                {"unique_id": "s2", "ds": "2025-01-05", "NBEATSx": 21.0},
            ],
        )


class _FakeNBEATSx:
    def __init__(self, h: int, input_size: int, **kwargs: Any) -> None:
        self.h = h
        self.input_size = input_size
        self.kwargs = kwargs


def _request() -> ForecastRequest:
    return ForecastRequest.model_validate(
        {
            "model": "nbeatsx",
            "horizon": 2,
            "quantiles": [0.1, 0.9],
            "series": [
                {
                    "id": "s1",
                    "freq": "D",
                    "timestamps": ["2025-01-01", "2025-01-02", "2025-01-03"],
                    "target": [1.0, 2.0, 3.0],
                },
                {
                    "id": "s2",
                    "freq": "D",
                    "timestamps": ["2025-01-01", "2025-01-02", "2025-01-03"],
                    "target": [3.0, 2.0, 1.0],
                    "past_covariates": {"promo": [0, 1, 1]},
                    "future_covariates": {"promo": [1, 0]},
                    "static_covariates": {"tier": 2},
                },
            ],
            "options": {},
        },
    )


def test_nbeatsx_adapter_forecast_smoke_multi_series(monkeypatch) -> None:
    adapter = NbeatsxAdapter()
    monkeypatch.setattr(
        adapter,
        "_resolve_dependencies",
        lambda: type(
            "D",
            (),
            {"pd": _FakePandas(), "neuralforecast_cls": _FakeNF, "nbeatsx_cls": _FakeNBEATSx},
        )(),
    )

    response = adapter.forecast(_request(), model_local_dir="/tmp")

    assert response.model == "nbeatsx"
    assert len(response.forecasts) == 2
    assert response.forecasts[0].mean == [10.0, 11.0]
    assert response.forecasts[1].mean == [20.0, 21.0]
    assert response.forecasts[0].quantiles is not None
    assert set(response.forecasts[0].quantiles) == {"0.1", "0.9"}
    assert response.usage is not None
    assert response.usage["runner"] == "tollama-nbeatsx"
    assert response.usage["covariates_hist_exog"] == 1
    assert response.warnings is not None
    assert any("calibrated quantile fallback" in warning for warning in response.warnings)


def test_nbeatsx_adapter_best_effort_drops_non_numeric_covariates(monkeypatch) -> None:
    adapter = NbeatsxAdapter()
    monkeypatch.setattr(
        adapter,
        "_resolve_dependencies",
        lambda: type(
            "D",
            (),
            {"pd": _FakePandas(), "neuralforecast_cls": _FakeNF, "nbeatsx_cls": _FakeNBEATSx},
        )(),
    )
    req = _request()
    req.series[1].past_covariates = {"promo": ["bad", "bad", "bad"]}
    req.series[1].future_covariates = {"promo": ["bad", "bad"]}

    response = adapter.forecast(req)

    assert response.warnings is not None
    assert any("non-numeric" in warning for warning in response.warnings)


def test_nbeatsx_adapter_strict_covariate_mode_rejects_non_numeric_covariates(monkeypatch) -> None:
    from tollama.runners.nbeatsx_runner.errors import AdapterInputError

    adapter = NbeatsxAdapter()
    monkeypatch.setattr(
        adapter,
        "_resolve_dependencies",
        lambda: type(
            "D",
            (),
            {"pd": _FakePandas(), "neuralforecast_cls": _FakeNF, "nbeatsx_cls": _FakeNBEATSx},
        )(),
    )
    req = _request()
    req.parameters.covariates_mode = "strict"
    req.series[1].past_covariates = {"promo": ["bad", "bad", "bad"]}

    try:
        adapter.forecast(req)
        raise AssertionError("expected AdapterInputError")
    except AdapterInputError as exc:
        assert "must be numeric" in str(exc)


def test_nbeatsx_adapter_invalid_frequency_maps_to_input_error(monkeypatch) -> None:
    from tollama.runners.nbeatsx_runner.errors import AdapterInputError

    adapter = NbeatsxAdapter()
    monkeypatch.setattr(
        adapter,
        "_resolve_dependencies",
        lambda: type(
            "D",
            (),
            {"pd": _FakePandas(), "neuralforecast_cls": _FakeNF, "nbeatsx_cls": _FakeNBEATSx},
        )(),
    )
    req = _request()
    req.series[0].freq = "BAD"

    try:
        adapter.forecast(req)
        raise AssertionError("expected AdapterInputError")
    except AdapterInputError as exc:
        assert "invalid frequency" in str(exc)


def test_nbeatsx_adapter_rejects_non_finite_target(monkeypatch) -> None:
    from tollama.runners.nbeatsx_runner.errors import AdapterInputError

    adapter = NbeatsxAdapter()
    monkeypatch.setattr(
        adapter,
        "_resolve_dependencies",
        lambda: type(
            "D",
            (),
            {"pd": _FakePandas(), "neuralforecast_cls": _FakeNF, "nbeatsx_cls": _FakeNBEATSx},
        )(),
    )
    req = _request()
    req.series[0].target[0] = float("nan")

    try:
        adapter.forecast(req)
        raise AssertionError("expected AdapterInputError")
    except AdapterInputError as exc:
        assert "must be finite" in str(exc)


def test_nbeatsx_adapter_rejects_mixed_multi_series_frequency(monkeypatch) -> None:
    from tollama.runners.nbeatsx_runner.errors import AdapterInputError

    adapter = NbeatsxAdapter()
    monkeypatch.setattr(
        adapter,
        "_resolve_dependencies",
        lambda: type(
            "D",
            (),
            {"pd": _FakePandas(), "neuralforecast_cls": _FakeNF, "nbeatsx_cls": _FakeNBEATSx},
        )(),
    )
    req = _request()
    req.series[1].freq = "H"

    try:
        adapter.forecast(req)
        raise AssertionError("expected AdapterInputError")
    except AdapterInputError as exc:
        assert "shared frequency" in str(exc)
