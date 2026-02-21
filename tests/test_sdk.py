"""Tests for the high-level Tollama Python SDK facade."""

from __future__ import annotations

import pandas as pd
import pytest

from tollama.core.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    AutoForecastRequest,
    AutoForecastResponse,
    CounterfactualRequest,
    CounterfactualResponse,
    ForecastReport,
    ForecastRequest,
    ForecastResponse,
    GenerateRequest,
    GenerateResponse,
    PipelineRequest,
    PipelineResponse,
    ReportRequest,
    ScenarioTreeRequest,
    ScenarioTreeResponse,
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
    with pytest.warns(UserWarning, match="Inferred frequency 'D'"):
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
    with pytest.warns(UserWarning, match="Inferred frequency 'D'"):
        result = sdk.forecast(model="timesfm-2.5-200m", series=frame, horizon=1)

    request = captured["request"]
    assert [series.id for series in request.series] == ["north", "south"]

    with pytest.raises(ValueError):
        _ = result.mean

    result_frame = result.to_df()
    assert sorted(result_frame["id"].unique().tolist()) == ["north", "south"]
    assert result_frame.shape[0] == 2


def test_forecast_from_file_loads_csv_and_calls_forecast(tmp_path) -> None:
    captured: dict[str, ForecastRequest] = {}

    class _FakeClient:
        def forecast_response(self, request: ForecastRequest) -> ForecastResponse:
            captured["request"] = request
            return _single_series_response()

    path = tmp_path / "history.csv"
    path.write_text(
        "timestamp,target\n2025-01-01,10.0\n2025-01-02,11.0\n2025-01-03,12.0\n",
        encoding="utf-8",
    )

    sdk = Tollama(client=_FakeClient())  # type: ignore[arg-type]
    result = sdk.forecast_from_file(
        model="chronos2",
        path=path,
        horizon=3,
        format_hint="csv",
    )

    request = captured["request"]
    assert request.series[0].timestamps[0] == "2025-01-01"
    assert request.series[0].target == [10.0, 11.0, 12.0]
    assert result.mean == [15.1, 16.2, 17.3]


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
    assert request.ensemble_method == "mean"
    assert response.selection.chosen_model == "mock"
    assert response.response.model == "mock"


def test_generate_accepts_series_dict_and_returns_typed_payload() -> None:
    captured: dict[str, GenerateRequest] = {}

    class _FakeClient:
        def generate(self, request: GenerateRequest) -> GenerateResponse:
            captured["request"] = request
            return GenerateResponse.model_validate(
                {
                    "method": "statistical",
                    "generated": [
                        {
                            "id": "series_0_synthetic_1",
                            "source_id": "series_0",
                            "freq": "D",
                            "timestamps": [
                                "2025-01-01",
                                "2025-01-02",
                                "2025-01-03",
                                "2025-01-04",
                                "2025-01-05",
                            ],
                            "target": [1.1, 2.1, 2.9, 2.6, 3.7],
                        }
                    ],
                },
            )

    sdk = Tollama(client=_FakeClient())  # type: ignore[arg-type]
    response = sdk.generate(
        series={"target": [1.0, 2.0, 3.0, 2.5, 3.5], "freq": "D"},
        count=1,
        length=5,
        seed=7,
    )

    request = captured["request"]
    assert request.count == 1
    assert request.length == 5
    assert request.seed == 7
    assert request.series[0].id == "series_0"
    assert response.generated[0].source_id == "series_0"


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


def test_counterfactual_accepts_series_dict_and_returns_typed_payload() -> None:
    captured: dict[str, CounterfactualRequest] = {}

    class _FakeClient:
        def counterfactual(self, request: CounterfactualRequest) -> CounterfactualResponse:
            captured["request"] = request
            return CounterfactualResponse.model_validate(
                {
                    "model": "mock",
                    "horizon": 2,
                    "intervention_index": 3,
                    "baseline": {
                        "model": "mock",
                        "forecasts": [
                            {
                                "id": "series_0",
                                "freq": "D",
                                "start_timestamp": "2025-01-06",
                                "mean": [12.0, 13.0],
                            }
                        ],
                    },
                    "results": [
                        {
                            "id": "series_0",
                            "actual": [20.0, 22.0],
                            "counterfactual": [12.0, 13.0],
                            "delta": [8.0, 9.0],
                            "absolute_delta": [8.0, 9.0],
                            "mean_absolute_delta": 8.5,
                            "total_delta": 17.0,
                            "average_delta_pct": 40.0,
                            "direction": "above_counterfactual",
                        }
                    ],
                }
            )

    sdk = Tollama(client=_FakeClient())  # type: ignore[arg-type]
    response = sdk.counterfactual(
        model="mock",
        series={"target": [1.0, 2.0, 3.0, 6.0, 7.0], "freq": "D"},
        intervention_index=3,
    )

    request = captured["request"]
    assert request.model == "mock"
    assert request.intervention_index == 3
    assert response.results[0].direction == "above_counterfactual"


def test_scenario_tree_accepts_series_dict_and_returns_typed_payload() -> None:
    captured: dict[str, ScenarioTreeRequest] = {}

    class _FakeClient:
        def scenario_tree(self, request: ScenarioTreeRequest) -> ScenarioTreeResponse:
            captured["request"] = request
            return ScenarioTreeResponse.model_validate(
                {
                    "model": "mock",
                    "depth": 2,
                    "branch_quantiles": [0.1, 0.9],
                    "nodes": [
                        {
                            "node_id": "node_1",
                            "parent_id": None,
                            "series_id": "series_0",
                            "depth": 0,
                            "step": 0,
                            "branch": "root",
                            "value": 4.0,
                            "probability": 1.0,
                        },
                        {
                            "node_id": "node_2",
                            "parent_id": "node_1",
                            "series_id": "series_0",
                            "depth": 1,
                            "step": 1,
                            "branch": "q0.1",
                            "quantile": 0.1,
                            "value": 4.0,
                            "probability": 0.5,
                        },
                    ],
                }
            )

    sdk = Tollama(client=_FakeClient())  # type: ignore[arg-type]
    response = sdk.scenario_tree(
        model="mock",
        series={"target": [1.0, 2.0, 3.0, 4.0], "freq": "D"},
        horizon=4,
        depth=2,
        branch_quantiles=[0.1, 0.9],
    )

    request = captured["request"]
    assert request.horizon == 4
    assert request.depth == 2
    assert response.nodes[1].parent_id == "node_1"


def test_report_accepts_series_dict_and_returns_typed_payload() -> None:
    captured: dict[str, ReportRequest] = {}

    class _FakeClient:
        def report(self, request: ReportRequest) -> ForecastReport:
            captured["request"] = request
            return ForecastReport.model_validate(
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
                    "recommendation": {"request": {"horizon": 3, "freq": "D", "top_k": 3}},
                    "forecast": {
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
                                    "mean": [5.0, 6.0, 7.0],
                                }
                            ],
                        },
                    },
                    "baseline": {
                        "model": "mock",
                        "forecasts": [
                            {
                                "id": "series_0",
                                "freq": "D",
                                "start_timestamp": "2025-01-06",
                                "mean": [5.0, 6.0, 7.0],
                            }
                        ],
                    },
                }
            )

    sdk = Tollama(client=_FakeClient())  # type: ignore[arg-type]
    response = sdk.report(
        series={"target": [1.0, 2.0, 3.0, 4.0], "freq": "D"},
        horizon=3,
        include_baseline=True,
    )

    request = captured["request"]
    assert request.horizon == 3
    assert request.include_baseline is True
    assert response.forecast.selection.chosen_model == "mock"


def test_series_mapping_requires_target() -> None:
    sdk = Tollama(client=object())  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="target"):
        sdk.forecast(model="chronos2", series={"freq": "D"}, horizon=2)
    with pytest.raises(ValueError, match="Found keys: freq"):
        sdk.forecast(model="chronos2", series={"freq": "D"}, horizon=2)


def test_wide_dataframe_validation_includes_found_columns() -> None:
    sdk = Tollama(client=object())  # type: ignore[arg-type]
    frame = pd.DataFrame({"label": ["a", "b", "c"]})

    with pytest.raises(ValueError, match="Found columns: label"):
        sdk.forecast(model="chronos2", series=frame, horizon=2)


def test_series_freq_inference_emits_warning() -> None:
    class _FakeClient:
        def forecast_response(self, request: ForecastRequest) -> ForecastResponse:
            assert request.series[0].freq == "D"
            return _single_series_response(series_id=request.series[0].id)

    sdk = Tollama(client=_FakeClient())  # type: ignore[arg-type]
    series = pd.Series(
        [10.0, 11.0, 12.0],
        index=pd.date_range("2025-01-01", periods=3, freq="D"),
        name="demand",
    )

    with pytest.warns(UserWarning, match="Inferred frequency 'D'"):
        sdk.forecast(model="chronos2", series=series, horizon=2)


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
    assert request.ensemble_method == "mean"
    assert response.analysis.results[0].id == "series_0"
    assert response.auto_forecast.selection.chosen_model == "mock"
