"""Tests for forecast metrics computation."""

from __future__ import annotations

import pytest

from tollama.core.forecast_metrics import compute_forecast_metrics
from tollama.core.schemas import ForecastRequest, ForecastResponse


def test_compute_forecast_metrics_returns_all_requested_metrics_with_macro_aggregate() -> None:
    request = ForecastRequest.model_validate(
        {
            "model": "mock",
            "horizon": 2,
            "quantiles": [],
            "series": [
                {
                    "id": "s1",
                    "freq": "D",
                    "timestamps": ["2025-01-01", "2025-01-02"],
                    "target": [1.0, 3.0],
                    "actuals": [2.0, 4.0],
                },
                {
                    "id": "s2",
                    "freq": "D",
                    "timestamps": ["2025-01-01", "2025-01-02"],
                    "target": [2.0, 4.0],
                    "actuals": [4.0, 2.0],
                },
            ],
            "options": {},
            "parameters": {"metrics": {"names": ["mape", "mase", "mae", "rmse", "smape"]}},
        },
    )
    response = ForecastResponse.model_validate(
        {
            "model": "mock",
            "forecasts": [
                {
                    "id": "s1",
                    "freq": "D",
                    "start_timestamp": "2025-01-03",
                    "mean": [3.0, 3.0],
                },
                {
                    "id": "s2",
                    "freq": "D",
                    "start_timestamp": "2025-01-03",
                    "mean": [4.0, 4.0],
                },
            ],
        },
    )

    metrics, warnings = compute_forecast_metrics(request=request, response=response)

    assert warnings == []
    assert metrics is not None
    assert metrics.aggregate == pytest.approx(
        {
            "mape": 43.75,
            "mase": 0.5,
            "mae": 1.0,
            "rmse": 1.2071067811865475,
            "smape": 33.80952380952381,
        }
    )
    assert metrics.series[0].id == "s1"
    assert metrics.series[0].values == pytest.approx(
        {
            "mape": 37.5,
            "mase": 0.5,
            "mae": 1.0,
            "rmse": 1.0,
            "smape": 34.285714285714285,
        }
    )
    assert metrics.series[1].id == "s2"
    assert metrics.series[1].values == pytest.approx(
        {
            "mape": 50.0,
            "mase": 0.5,
            "mae": 1.0,
            "rmse": 1.4142135623730951,
            "smape": 33.333333333333336,
        }
    )


def test_compute_forecast_metrics_skips_undefined_mape_and_returns_warning() -> None:
    request = ForecastRequest.model_validate(
        {
            "model": "mock",
            "horizon": 2,
            "quantiles": [],
            "series": [
                {
                    "id": "s1",
                    "freq": "D",
                    "timestamps": ["2025-01-01", "2025-01-02"],
                    "target": [1.0, 3.0],
                    "actuals": [0.0, 0.0],
                }
            ],
            "options": {},
            "parameters": {"metrics": {"names": ["mape"]}},
        },
    )
    response = ForecastResponse.model_validate(
        {
            "model": "mock",
            "forecasts": [
                {
                    "id": "s1",
                    "freq": "D",
                    "start_timestamp": "2025-01-03",
                    "mean": [3.0, 3.0],
                }
            ],
        },
    )

    metrics, warnings = compute_forecast_metrics(request=request, response=response)

    assert metrics is None
    assert len(warnings) == 1
    assert "metrics.mape skipped" in warnings[0]


def test_compute_forecast_metrics_skips_undefined_mase_and_returns_warning() -> None:
    request = ForecastRequest.model_validate(
        {
            "model": "mock",
            "horizon": 1,
            "quantiles": [],
            "series": [
                {
                    "id": "s1",
                    "freq": "D",
                    "timestamps": ["2025-01-01", "2025-01-02"],
                    "target": [1.0, 1.0],
                    "actuals": [2.0],
                }
            ],
            "options": {},
            "parameters": {"metrics": {"names": ["mase"], "mase_seasonality": 1}},
        },
    )
    response = ForecastResponse.model_validate(
        {
            "model": "mock",
            "forecasts": [
                {
                    "id": "s1",
                    "freq": "D",
                    "start_timestamp": "2025-01-03",
                    "mean": [3.0],
                }
            ],
        },
    )

    metrics, warnings = compute_forecast_metrics(request=request, response=response)

    assert metrics is None
    assert len(warnings) == 1
    assert "metrics.mase skipped" in warnings[0]


def test_compute_forecast_metrics_uses_overlap_when_forecast_length_differs() -> None:
    request = ForecastRequest.model_validate(
        {
            "model": "mock",
            "horizon": 2,
            "quantiles": [],
            "series": [
                {
                    "id": "s1",
                    "freq": "D",
                    "timestamps": ["2025-01-01", "2025-01-02"],
                    "target": [1.0, 3.0],
                    "actuals": [2.0, 4.0],
                }
            ],
            "options": {},
            "parameters": {"metrics": {"names": ["mape"]}},
        },
    )
    response = ForecastResponse.model_validate(
        {
            "model": "mock",
            "forecasts": [
                {
                    "id": "s1",
                    "freq": "D",
                    "start_timestamp": "2025-01-03",
                    "mean": [3.0],
                }
            ],
        },
    )

    metrics, warnings = compute_forecast_metrics(request=request, response=response)

    assert metrics is not None
    assert metrics.aggregate == pytest.approx({"mape": 50.0})
    assert "uses 1 points" in warnings[0]


def test_compute_forecast_metrics_skips_undefined_smape_and_returns_warning() -> None:
    request = ForecastRequest.model_validate(
        {
            "model": "mock",
            "horizon": 2,
            "quantiles": [],
            "series": [
                {
                    "id": "s1",
                    "freq": "D",
                    "timestamps": ["2025-01-01", "2025-01-02"],
                    "target": [1.0, 3.0],
                    "actuals": [0.0, 0.0],
                }
            ],
            "options": {},
            "parameters": {"metrics": {"names": ["smape"]}},
        },
    )
    response = ForecastResponse.model_validate(
        {
            "model": "mock",
            "forecasts": [
                {
                    "id": "s1",
                    "freq": "D",
                    "start_timestamp": "2025-01-03",
                    "mean": [0.0, 0.0],
                }
            ],
        },
    )

    metrics, warnings = compute_forecast_metrics(request=request, response=response)

    assert metrics is None
    assert len(warnings) == 1
    assert "metrics.smape skipped" in warnings[0]
    assert "denominators are zero" in warnings[0]


def test_compute_forecast_metrics_supports_wape_rmsse_and_pinball() -> None:
    request = ForecastRequest.model_validate(
        {
            "model": "mock",
            "horizon": 2,
            "quantiles": [0.1, 0.9],
            "series": [
                {
                    "id": "s1",
                    "freq": "D",
                    "timestamps": ["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-04"],
                    "target": [1.0, 2.0, 3.0, 4.0],
                    "actuals": [3.0, 5.0],
                }
            ],
            "options": {},
            "parameters": {"metrics": {"names": ["wape", "rmsse", "pinball"]}},
        },
    )
    response = ForecastResponse.model_validate(
        {
            "model": "mock",
            "forecasts": [
                {
                    "id": "s1",
                    "freq": "D",
                    "start_timestamp": "2025-01-05",
                    "mean": [2.0, 6.0],
                    "quantiles": {
                        "0.1": [1.5, 4.0],
                        "0.9": [2.5, 6.5],
                    },
                }
            ],
        },
    )

    metrics, warnings = compute_forecast_metrics(request=request, response=response)

    assert warnings == []
    assert metrics is not None
    assert metrics.aggregate == pytest.approx({"wape": 25.0, "rmsse": 1.0, "pinball": 0.2125})
    assert metrics.series[0].values == pytest.approx(
        {"wape": 25.0, "rmsse": 1.0, "pinball": 0.2125},
    )


def test_compute_forecast_metrics_skips_pinball_when_quantiles_missing() -> None:
    request = ForecastRequest.model_validate(
        {
            "model": "mock",
            "horizon": 2,
            "quantiles": [0.1, 0.9],
            "series": [
                {
                    "id": "s1",
                    "freq": "D",
                    "timestamps": ["2025-01-01", "2025-01-02"],
                    "target": [1.0, 2.0],
                    "actuals": [3.0, 5.0],
                }
            ],
            "options": {},
            "parameters": {"metrics": {"names": ["pinball"]}},
        },
    )
    response = ForecastResponse.model_validate(
        {
            "model": "mock",
            "forecasts": [
                {
                    "id": "s1",
                    "freq": "D",
                    "start_timestamp": "2025-01-03",
                    "mean": [2.0, 6.0],
                }
            ],
        },
    )

    metrics, warnings = compute_forecast_metrics(request=request, response=response)

    assert metrics is None
    assert len(warnings) == 1
    assert "metrics.pinball skipped" in warnings[0]
    assert "quantile forecasts are required" in warnings[0]


def test_compute_forecast_metrics_skips_undefined_wape_and_rmsse_with_warnings() -> None:
    request = ForecastRequest.model_validate(
        {
            "model": "mock",
            "horizon": 2,
            "quantiles": [],
            "series": [
                {
                    "id": "s1",
                    "freq": "D",
                    "timestamps": ["2025-01-01", "2025-01-02"],
                    "target": [2.0, 2.0],
                    "actuals": [0.0, 0.0],
                }
            ],
            "options": {},
            "parameters": {"metrics": {"names": ["wape", "rmsse"]}},
        },
    )
    response = ForecastResponse.model_validate(
        {
            "model": "mock",
            "forecasts": [
                {
                    "id": "s1",
                    "freq": "D",
                    "start_timestamp": "2025-01-03",
                    "mean": [0.0, 0.0],
                }
            ],
        },
    )

    metrics, warnings = compute_forecast_metrics(request=request, response=response)

    assert metrics is None
    assert len(warnings) == 2
    assert "metrics.wape skipped" in warnings[0]
    assert "metrics.rmsse skipped" in warnings[1]
