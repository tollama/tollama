"""Tests for the high-level Tollama Python SDK facade."""

from __future__ import annotations

import pandas as pd
import pytest

from tollama.core.schemas import ForecastRequest, ForecastResponse
from tollama.sdk import Tollama


def _single_series_response(*, series_id: str = "series_0") -> ForecastResponse:
    return ForecastResponse.model_validate(
        {
            "model": "chronos2",
            "forecasts": [
                {
                    "id": series_id,
                    "freq": "D",
                    "start_timestamp": "2025-01-06",
                    "mean": [15.1, 16.2, 17.3],
                    "quantiles": {
                        "0.1": [14.0, 15.0, 16.0],
                        "0.9": [16.0, 17.0, 18.0],
                    },
                }
            ],
        },
    )


def test_forecast_accepts_simple_series_dict_and_returns_convenience_accessors() -> None:
    captured: dict[str, ForecastRequest] = {}

    class _FakeClient:
        def forecast_response(self, request: ForecastRequest) -> ForecastResponse:
            captured["request"] = request
            return _single_series_response()

    sdk = Tollama(client=_FakeClient())  # type: ignore[arg-type]
    result = sdk.forecast(
        model="chronos2",
        series={"target": [10, 11, 12, 13, 14], "freq": "D"},
        horizon=3,
        quantiles=[0.1, 0.9],
    )

    request = captured["request"]
    assert request.model == "chronos2"
    assert request.series[0].id == "series_0"
    assert request.series[0].timestamps == ["0", "1", "2", "3", "4"]
    assert request.horizon == 3

    assert result.mean == [15.1, 16.2, 17.3]
    assert result.quantiles == {
        "0.1": [14.0, 15.0, 16.0],
        "0.9": [16.0, 17.0, 18.0],
    }

    frame = result.to_df()
    assert list(frame["id"]) == ["series_0", "series_0", "series_0"]
    assert list(frame["mean"]) == [15.1, 16.2, 17.3]
    assert "q0.1" in frame.columns
    assert "q0.9" in frame.columns


def test_forecast_accepts_pandas_series_input() -> None:
    captured: dict[str, ForecastRequest] = {}

    class _FakeClient:
        def forecast_response(self, request: ForecastRequest) -> ForecastResponse:
            captured["request"] = request
            return _single_series_response(series_id="sales")

    history = pd.Series(
        [10.0, 11.0, 12.0, 13.0, 14.0],
        index=pd.date_range("2025-01-01", periods=5, freq="D"),
        name="sales",
    )
    sdk = Tollama(client=_FakeClient())  # type: ignore[arg-type]
    result = sdk.forecast(model="chronos2", series=history, horizon=3)

    request = captured["request"]
    assert request.series[0].id == "sales"
    assert request.series[0].freq == "D"
    assert request.series[0].timestamps[0].startswith("2025-01-01")
    assert result.mean == [15.1, 16.2, 17.3]


def test_forecast_accepts_wide_pandas_dataframe_for_multi_series() -> None:
    captured: dict[str, ForecastRequest] = {}

    class _FakeClient:
        def forecast_response(self, request: ForecastRequest) -> ForecastResponse:
            captured["request"] = request
            return ForecastResponse.model_validate(
                {
                    "model": "timesfm-2.5-200m",
                    "forecasts": [
                        {
                            "id": "north",
                            "freq": "D",
                            "start_timestamp": "2025-01-04",
                            "mean": [4.0],
                        },
                        {
                            "id": "south",
                            "freq": "D",
                            "start_timestamp": "2025-01-04",
                            "mean": [7.0],
                        },
                    ],
                },
            )

    frame = pd.DataFrame(
        {"north": [1.0, 2.0, 3.0], "south": [4.0, 5.0, 6.0]},
        index=pd.date_range("2025-01-01", periods=3, freq="D"),
    )
    sdk = Tollama(client=_FakeClient())  # type: ignore[arg-type]
    result = sdk.forecast(model="timesfm-2.5-200m", series=frame, horizon=1)

    request = captured["request"]
    assert [series.id for series in request.series] == ["north", "south"]

    with pytest.raises(ValueError):
        _ = result.mean

    result_frame = result.to_df()
    assert sorted(result_frame["id"].unique().tolist()) == ["north", "south"]
    assert result_frame.shape[0] == 2


def test_tollama_export_is_available_from_package_root() -> None:
    from tollama import Tollama as ExportedTollama

    assert ExportedTollama is Tollama
    assert ExportedTollama is not None


def test_series_mapping_requires_target() -> None:
    sdk = Tollama(client=object())  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="target"):
        sdk.forecast(model="chronos2", series={"freq": "D"}, horizon=2)
