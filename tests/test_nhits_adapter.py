from __future__ import annotations

from typing import Any

from tollama.core.schemas import ForecastRequest
from tollama.runners.nhits_runner.adapter import NhitsAdapter
from tollama.runners.nhits_runner.errors import AdapterInputError


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
        self.models = models
        self.freq = freq
        self.fit_static_df: Any | None = None
        self.predict_futr_df: Any | None = None

    def fit(self, train_df: _FakeDataFrame, static_df: _FakeDataFrame | None = None) -> None:
        del train_df
        self.fit_static_df = static_df

    def predict(
        self,
        h: int | None = None,
        futr_df: _FakeDataFrame | None = None,
    ) -> _FakeDataFrame:
        del h
        self.predict_futr_df = futr_df
        return _FakeDataFrame(
            [
                {"unique_id": "s1", "ds": "2025-01-04", "NHITS": 10.0},
                {"unique_id": "s1", "ds": "2025-01-05", "NHITS": 11.0},
                {"unique_id": "s2", "ds": "2025-01-04", "NHITS": 20.0},
                {"unique_id": "s2", "ds": "2025-01-05", "NHITS": 21.0},
            ],
        )


class _FakeNFWithQuantiles(_FakeNF):
    def predict(
        self,
        h: int | None = None,
        futr_df: _FakeDataFrame | None = None,
    ) -> _FakeDataFrame:
        del h, futr_df
        return _FakeDataFrame(
            [
                {"unique_id": "s1", "ds": "2025-01-04", "NHITS": 10.0, "q0.1": 8.0, "q0.9": 12.0},
                {"unique_id": "s1", "ds": "2025-01-05", "NHITS": 11.0, "q0.1": 9.0, "q0.9": 13.0},
                {"unique_id": "s2", "ds": "2025-01-04", "NHITS": 20.0, "q0.1": 18.0, "q0.9": 22.0},
                {"unique_id": "s2", "ds": "2025-01-05", "NHITS": 21.0, "q0.1": 19.0, "q0.9": 23.0},
            ],
        )


class _FakeNHITS:
    def __init__(
        self,
        h: int,
        input_size: int,
        hist_exog_list: list[str] | None = None,
        futr_exog_list: list[str] | None = None,
        stat_exog_list: list[str] | None = None,
    ) -> None:
        self.h = h
        self.input_size = input_size
        self.hist_exog_list = hist_exog_list or []
        self.futr_exog_list = futr_exog_list or []
        self.stat_exog_list = stat_exog_list or []


class _RecordingNF(_FakeNF):
    last_instance: _RecordingNF | None = None

    def __init__(self, models: list[Any], freq: str) -> None:
        super().__init__(models=models, freq=freq)
        _RecordingNF.last_instance = self


def _request() -> ForecastRequest:
    return ForecastRequest.model_validate(
        {
            "model": "nhits",
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
                },
            ],
            "options": {},
        },
    )


def _patch_deps(monkeypatch, adapter: NhitsAdapter, nf_cls: Any = _FakeNF) -> None:
    monkeypatch.setattr(
        adapter,
        "_resolve_dependencies",
        lambda: type(
            "D",
            (),
            {
                "pd": _FakePandas(),
                "neuralforecast_cls": nf_cls,
                "nhits_cls": _FakeNHITS,
            },
        )(),
    )


def test_nhits_adapter_forecast_smoke_multi_series_uses_calibrated_quantile_fallback(
    monkeypatch,
) -> None:
    adapter = NhitsAdapter()
    _patch_deps(monkeypatch, adapter)

    response = adapter.forecast(_request(), model_local_dir="/tmp")

    assert response.model == "nhits"
    assert len(response.forecasts) == 2
    assert response.forecasts[0].mean == [10.0, 11.0]
    assert response.forecasts[1].mean == [20.0, 21.0]
    assert response.forecasts[0].quantiles is not None
    assert "0.1" in response.forecasts[0].quantiles
    assert response.usage is not None
    assert response.usage["runner"] == "tollama-nhits"
    assert response.warnings is not None
    assert any("calibrated quantile fallback" in warning for warning in response.warnings)


def test_nhits_adapter_returns_quantiles_when_backend_exposes_columns(monkeypatch) -> None:
    adapter = NhitsAdapter()
    _patch_deps(monkeypatch, adapter, _FakeNFWithQuantiles)

    response = adapter.forecast(_request())

    assert response.forecasts[0].quantiles == {"0.1": [8.0, 9.0], "0.9": [12.0, 13.0]}
    assert response.forecasts[1].quantiles == {"0.1": [18.0, 19.0], "0.9": [22.0, 23.0]}
    assert response.warnings is None or not any(
        "calibrated quantile fallback" in warning for warning in response.warnings
    )


def test_nhits_adapter_accepts_numeric_covariates_and_static_features(monkeypatch) -> None:
    adapter = NhitsAdapter()
    _patch_deps(monkeypatch, adapter, _RecordingNF)

    req = _request()
    req.series[0].past_covariates = {"promo": [0.0, 1.0, 0.0], "temp": [10.0, 11.0, 12.0]}
    req.series[0].future_covariates = {"promo": [1.0, 1.0]}
    req.series[0].static_covariates = {"store_size": 1.5}
    req.series[1].past_covariates = {"promo": [1.0, 0.0, 1.0], "temp": [8.0, 9.0, 10.0]}
    req.series[1].future_covariates = {"promo": [0.0, 0.0]}
    req.series[1].static_covariates = {"store_size": 2.5}

    response = adapter.forecast(req)

    model = _RecordingNF.last_instance.models[0]
    assert sorted(model.hist_exog_list) == ["promo", "temp"]
    assert sorted(model.futr_exog_list) == ["promo"]
    assert sorted(model.stat_exog_list) == ["store_size"]
    assert _RecordingNF.last_instance.fit_static_df is not None
    assert _RecordingNF.last_instance.predict_futr_df is not None
    assert response.usage["covariates_hist_exog"] == 2
    assert response.usage["covariates_futr_exog"] == 1
    assert response.usage["covariates_stat_exog"] == 1


def test_nhits_adapter_best_effort_drops_non_numeric_covariates_with_warning(monkeypatch) -> None:
    adapter = NhitsAdapter()
    _patch_deps(monkeypatch, adapter)

    req = _request()
    req.series[0].past_covariates = {"promo": ["bad", "bad", "bad"]}
    req.series[1].past_covariates = {"promo": [1.0, 0.0, 1.0]}

    response = adapter.forecast(req)

    assert response.warnings is not None
    assert any("best_effort mode" in warning for warning in response.warnings)


def test_nhits_adapter_strict_mode_rejects_non_numeric_covariates(monkeypatch) -> None:
    adapter = NhitsAdapter()
    _patch_deps(monkeypatch, adapter)

    req = _request()
    req.parameters.covariates_mode = "strict"
    req.series[0].past_covariates = {"promo": ["bad", "bad", "bad"]}
    req.series[1].past_covariates = {"promo": [1.0, 0.0, 1.0]}

    try:
        adapter.forecast(req)
        raise AssertionError("expected AdapterInputError")
    except AdapterInputError as exc:
        assert "must be numeric" in str(exc)


def test_nhits_adapter_invalid_frequency_maps_to_input_error(monkeypatch) -> None:
    adapter = NhitsAdapter()
    _patch_deps(monkeypatch, adapter)
    req = _request()
    req.series[0].freq = "BAD"

    try:
        adapter.forecast(req)
        raise AssertionError("expected AdapterInputError")
    except AdapterInputError as exc:
        assert "invalid frequency" in str(exc)


def test_nhits_adapter_rejects_non_finite_target(monkeypatch) -> None:
    adapter = NhitsAdapter()
    _patch_deps(monkeypatch, adapter)
    req = _request()
    req.series[0].target[0] = float("nan")

    try:
        adapter.forecast(req)
        raise AssertionError("expected AdapterInputError")
    except AdapterInputError as exc:
        assert "must be finite" in str(exc)


def test_nhits_adapter_rejects_mixed_multi_series_frequency(monkeypatch) -> None:
    adapter = NhitsAdapter()
    _patch_deps(monkeypatch, adapter)
    req = _request()
    req.series[1].freq = "H"

    try:
        adapter.forecast(req)
        raise AssertionError("expected AdapterInputError")
    except AdapterInputError as exc:
        assert "shared frequency" in str(exc)
