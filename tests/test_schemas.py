"""Tests for forecast request/response schemas."""

import pytest
from pydantic import ValidationError

from tollama.core.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    ForecastRequest,
    ForecastResponse,
    SeriesForecast,
)


def _example_request_payload() -> dict[str, object]:
    return {
        "model": "naive",
        "horizon": 3,
        "quantiles": [0.1, 0.5, 0.9],
        "series": [
            {
                "id": "series-1",
                "freq": "D",
                "timestamps": ["2025-01-01", "2025-01-02", "2025-01-03"],
                "target": [10.0, 11.0, 12.0],
                "past_covariates": {"promo": [0.0, 1.0, 0.0]},
                "future_covariates": {"promo": [1.1, 1.2, 1.3]},
                "static_covariates": {"region": "us-east", "tier": 2},
            }
        ],
        "options": {"temperature": 0.1, "seed": 7},
    }


def _example_analyze_request_payload() -> dict[str, object]:
    return {
        "series": [
            {
                "id": "series-1",
                "freq": "D",
                "timestamps": ["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-04"],
                "target": [10.0, 11.0, 12.0, 11.5],
            }
        ],
        "parameters": {
            "max_points": 100,
            "max_lag": 16,
            "top_k_seasonality": 2,
            "anomaly_iqr_k": 1.5,
        },
    }


def test_forecast_request_roundtrip_is_lossless() -> None:
    request = ForecastRequest.model_validate(_example_request_payload())
    encoded = request.to_json()
    decoded = ForecastRequest.model_validate_json(encoded)
    assert decoded == request
    assert decoded.model_dump(mode="json", exclude_none=True) == request.model_dump(
        mode="json",
        exclude_none=True,
    )


def test_analyze_request_roundtrip_is_lossless() -> None:
    request = AnalyzeRequest.model_validate(_example_analyze_request_payload())
    encoded = request.to_json()
    decoded = AnalyzeRequest.model_validate_json(encoded)
    assert decoded == request
    assert decoded.parameters.max_points == 100


def test_forecast_request_freq_defaults_to_auto_and_preserves_explicit_freq() -> None:
    payload = _example_request_payload()
    del payload["series"][0]["freq"]
    request = ForecastRequest.model_validate(payload)
    assert request.series[0].freq == "auto"

    payload = _example_request_payload()
    payload["series"][0]["freq"] = "H"
    request = ForecastRequest.model_validate(payload)
    assert request.series[0].freq == "H"


def test_forecast_request_rejects_invalid_payloads() -> None:
    invalid_horizon = _example_request_payload()
    invalid_horizon["horizon"] = "3"
    with pytest.raises(ValidationError):
        ForecastRequest.model_validate(invalid_horizon)

    invalid_series = _example_request_payload()
    series = invalid_series["series"][0]
    series["target"] = [1.0, 2.0]
    with pytest.raises(ValidationError):
        ForecastRequest.model_validate(invalid_series)

    invalid_quantiles = _example_request_payload()
    invalid_quantiles["quantiles"] = [0.9, 0.1]
    with pytest.raises(ValidationError):
        ForecastRequest.model_validate(invalid_quantiles)


def test_analyze_request_rejects_invalid_parameters() -> None:
    payload = _example_analyze_request_payload()
    payload["parameters"]["max_points"] = 0  # type: ignore[index]
    with pytest.raises(ValidationError):
        AnalyzeRequest.model_validate(payload)


def test_analyze_request_requires_unique_series_ids() -> None:
    payload = _example_analyze_request_payload()
    payload["series"].append(  # type: ignore[union-attr]
        {
            "id": "series-1",
            "freq": "D",
            "timestamps": ["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-04"],
            "target": [1.0, 1.1, 1.2, 1.3],
        }
    )
    with pytest.raises(ValidationError):
        AnalyzeRequest.model_validate(payload)


def test_forecast_response_rejects_mismatched_quantile_lengths() -> None:
    payload = {
        "model": "naive",
        "forecasts": [
            {
                "id": "series-1",
                "freq": "D",
                "start_timestamp": "2025-01-04",
                "mean": [12.0, 13.0],
                "quantiles": {"0.5": [12.0]},
            }
        ],
    }

    with pytest.raises(ValidationError):
        ForecastResponse.model_validate(payload)


def test_analyze_response_rejects_unsorted_anomaly_indices() -> None:
    payload = {
        "results": [
            {
                "id": "series-1",
                "detected_frequency": "D",
                "seasonality_periods": [2, 4],
                "trend": {"direction": "up", "slope": 0.2, "r2": 0.8},
                "anomaly_indices": [3, 1],
                "stationarity_flag": False,
                "data_quality_score": 0.9,
            }
        ],
    }
    with pytest.raises(ValidationError):
        AnalyzeResponse.model_validate(payload)


def test_series_forecast_accepts_valid_quantiles() -> None:
    forecast = SeriesForecast.model_validate(
        {
            "id": "series-1",
            "freq": "D",
            "start_timestamp": "2025-01-04",
            "mean": [12.0, 13.0],
            "quantiles": {"0.1": [11.0, 12.0], "0.9": [13.0, 14.0]},
        }
    )
    assert forecast.quantiles is not None
    assert set(forecast.quantiles) == {"0.1", "0.9"}


def test_forecast_request_rejects_mixed_covariate_value_types() -> None:
    payload = _example_request_payload()
    series = payload["series"][0]
    series["past_covariates"] = {"promo": [1.0, "bad", 3.0]}
    with pytest.raises(ValidationError):
        ForecastRequest.model_validate(payload)


def test_forecast_request_rejects_future_only_covariates() -> None:
    payload = _example_request_payload()
    series = payload["series"][0]
    series["future_covariates"] = {"future_only": [1.0, 2.0, 3.0]}
    with pytest.raises(ValidationError):
        ForecastRequest.model_validate(payload)


def test_forecast_request_metrics_requires_actuals_and_horizon_alignment() -> None:
    payload = _example_request_payload()
    payload["parameters"] = {"metrics": {"names": ["mae", "rmse", "smape", "mape", "mase"]}}

    with pytest.raises(ValidationError):
        ForecastRequest.model_validate(payload)

    series = payload["series"][0]
    series["actuals"] = [12.0, 13.0]
    with pytest.raises(ValidationError):
        ForecastRequest.model_validate(payload)

    series["actuals"] = [12.0, 13.0, 14.0]
    request = ForecastRequest.model_validate(payload)
    assert request.parameters.metrics is not None
    assert request.parameters.metrics.names == ["mae", "rmse", "smape", "mape", "mase"]
    assert request.parameters.metrics.mase_seasonality == 1


def test_forecast_request_rejects_duplicate_metric_names() -> None:
    payload = _example_request_payload()
    payload["series"][0]["actuals"] = [12.0, 13.0, 14.0]
    payload["parameters"] = {"metrics": {"names": ["mape", "mape"]}}
    with pytest.raises(ValidationError):
        ForecastRequest.model_validate(payload)


def test_forecast_request_rejects_unknown_metric_names() -> None:
    payload = _example_request_payload()
    payload["series"][0]["actuals"] = [12.0, 13.0, 14.0]
    payload["parameters"] = {"metrics": {"names": ["mae", "unknown"]}}
    with pytest.raises(ValidationError):
        ForecastRequest.model_validate(payload)


def test_forecast_response_accepts_metrics_payload() -> None:
    payload = {
        "model": "naive",
        "forecasts": [
            {
                "id": "series-1",
                "freq": "D",
                "start_timestamp": "2025-01-04",
                "mean": [12.0, 13.0],
            }
        ],
        "metrics": {
            "aggregate": {"mape": 12.5, "mase": 0.8, "mae": 0.2, "rmse": 0.3, "smape": 14.0},
            "series": [
                {
                    "id": "series-1",
                    "values": {
                        "mape": 12.5,
                        "mase": 0.8,
                        "mae": 0.2,
                        "rmse": 0.3,
                        "smape": 14.0,
                    },
                }
            ],
        },
    }
    response = ForecastResponse.model_validate(payload)
    assert response.metrics is not None
    assert response.metrics.aggregate["mape"] == 12.5
    assert response.metrics.aggregate["smape"] == 14.0
