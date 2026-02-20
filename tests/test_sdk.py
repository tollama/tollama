"""Tests for the high-level Tollama Python SDK facade."""

from __future__ import annotations

import pandas as pd
import pytest

from tollama.core.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    AutoForecastRequest,
    AutoForecastResponse,
    ForecastRequest,
    ForecastResponse,
    PipelineRequest,
    PipelineResponse,
    WhatIfRequest,
    WhatIfResponse,
)
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


def test_analyze_accepts_series_dict_and_returns_typed_payload() -> None:
    captured: dict[str, AnalyzeRequest] = {}

    class _FakeClient:
        def analyze(self, request: AnalyzeRequest) -> AnalyzeResponse:
            captured["request"] = request
            return AnalyzeResponse.model_validate(
                {
                    "results": [
                        {
                            "id": "series_0",
                            "detected_frequency": "D",
                            "seasonality_periods": [2],
                            "trend": {"direction": "up", "slope": 0.1, "r2": 0.5},
                            "anomaly_indices": [],
                            "stationarity_flag": True,
                            "data_quality_score": 0.95,
                        }
                    ],
                },
            )

    sdk = Tollama(client=_FakeClient())  # type: ignore[arg-type]
    response = sdk.analyze(
        series={"target": [1.0, 2.0, 3.0, 4.0], "freq": "D"},
        parameters={"max_lag": 2},
    )

    request = captured["request"]
    assert request.series[0].id == "series_0"
    assert request.parameters.max_lag == 2
    assert response.results[0].id == "series_0"


def test_auto_forecast_accepts_series_dict_and_returns_typed_payload() -> None:
    captured: dict[str, AutoForecastRequest] = {}

    class _FakeClient:
        def auto_forecast(self, request: AutoForecastRequest) -> AutoForecastResponse:
            captured["request"] = request
            return AutoForecastResponse.model_validate(
                {
                    "strategy": "auto",
                    "selection": {
                        "strategy": "auto",
                        "chosen_model": "mock",
                        "selected_models": ["mock"],
                        "candidates": [
                            {
                                "model": "mock",
                                "family": "mock",
                                "rank": 1,
                                "score": 100.0,
                                "reasons": ["selected"],
                            }
                        ],
                        "rationale": ["selected"],
                        "fallback_used": False,
                    },
                    "response": {
                        "model": "mock",
                        "forecasts": [
                            {
                                "id": "series_0",
                                "freq": "D",
                                "start_timestamp": "2025-01-06",
                                "mean": [15.1, 16.2, 17.3],
                            }
                        ],
                    },
                },
            )

    sdk = Tollama(client=_FakeClient())  # type: ignore[arg-type]
    response = sdk.auto_forecast(
        series={"target": [1.0, 2.0, 3.0, 4.0], "freq": "D"},
        horizon=3,
        strategy="auto",
    )

    request = captured["request"]
    assert request.horizon == 3
    assert request.series[0].id == "series_0"
    assert response.selection.chosen_model == "mock"
    assert response.response.model == "mock"


def test_what_if_accepts_series_dict_and_returns_typed_payload() -> None:
    captured: dict[str, WhatIfRequest] = {}

    class _FakeClient:
        def what_if(self, request: WhatIfRequest) -> WhatIfResponse:
            captured["request"] = request
            return WhatIfResponse.model_validate(
                {
                    "model": "mock",
                    "horizon": 3,
                    "baseline": {
                        "model": "mock",
                        "forecasts": [
                            {
                                "id": "series_0",
                                "freq": "D",
                                "start_timestamp": "2025-01-06",
                                "mean": [15.1, 16.2, 17.3],
                            }
                        ],
                    },
                    "results": [
                        {
                            "scenario": "high_demand",
                            "ok": True,
                            "response": {
                                "model": "mock",
                                "forecasts": [
                                    {
                                        "id": "series_0",
                                        "freq": "D",
                                        "start_timestamp": "2025-01-06",
                                        "mean": [18.12, 19.44, 20.76],
                                    }
                                ],
                            },
                        }
                    ],
                    "summary": {
                        "requested_scenarios": 1,
                        "succeeded": 1,
                        "failed": 0,
                    },
                },
            )

    sdk = Tollama(client=_FakeClient())  # type: ignore[arg-type]
    response = sdk.what_if(
        model="mock",
        series={"target": [1.0, 2.0, 3.0, 4.0], "freq": "D"},
        horizon=3,
        scenarios=[
            {
                "name": "high_demand",
                "transforms": [
                    {"operation": "multiply", "field": "target", "value": 1.2},
                ],
            }
        ],
    )

    request = captured["request"]
    assert request.model == "mock"
    assert request.horizon == 3
    assert request.series[0].id == "series_0"
    assert request.scenarios[0].name == "high_demand"
    assert response.summary.succeeded == 1


def test_series_mapping_requires_target() -> None:
    sdk = Tollama(client=object())  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="target"):
        sdk.forecast(model="chronos2", series={"freq": "D"}, horizon=2)


def test_pipeline_accepts_series_dict_and_returns_typed_payload() -> None:
    captured: dict[str, PipelineRequest] = {}

    class _FakeClient:
        def pipeline(self, request: PipelineRequest) -> PipelineResponse:
            captured["request"] = request
            return PipelineResponse.model_validate(
                {
                    "analysis": {
                        "results": [
                            {
                                "id": "series_0",
                                "detected_frequency": "D",
                                "seasonality_periods": [2],
                                "trend": {"direction": "up", "slope": 0.1, "r2": 0.5},
                                "anomaly_indices": [],
                                "stationarity_flag": True,
                                "data_quality_score": 0.95,
                            }
                        ],
                    },
                    "recommendation": {
                        "request": {"horizon": 3, "freq": "D", "top_k": 3},
                        "recommendations": [
                            {"model": "mock", "family": "mock", "rank": 1, "score": 100}
                        ],
                        "excluded": [],
                        "total_candidates": 1,
                        "compatible_candidates": 1,
                    },
                    "pulled_model": None,
                    "auto_forecast": {
                        "strategy": "auto",
                        "selection": {
                            "strategy": "auto",
                            "chosen_model": "mock",
                            "selected_models": ["mock"],
                            "candidates": [
                                {
                                    "model": "mock",
                                    "family": "mock",
                                    "rank": 1,
                                    "score": 100.0,
                                    "reasons": ["selected"],
                                }
                            ],
                            "rationale": ["selected"],
                            "fallback_used": False,
                        },
                        "response": {
                            "model": "mock",
                            "forecasts": [
                                {
                                    "id": "series_0",
                                    "freq": "D",
                                    "start_timestamp": "2025-01-06",
                                    "mean": [15.1, 16.2, 17.3],
                                }
                            ],
                        },
                    },
                },
            )

    sdk = Tollama(client=_FakeClient())  # type: ignore[arg-type]
    response = sdk.pipeline(
        series={"target": [1.0, 2.0, 3.0, 4.0], "freq": "D"},
        horizon=3,
        strategy="auto",
        pull_if_missing=True,
    )

    request = captured["request"]
    assert request.horizon == 3
    assert request.series[0].id == "series_0"
    assert response.analysis.results[0].id == "series_0"
    assert response.auto_forecast.selection.chosen_model == "mock"
