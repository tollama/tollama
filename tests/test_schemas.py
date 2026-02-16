"""Tests for forecast request/response schemas."""

import pytest
from pydantic import ValidationError

from tollama.core.schemas import ForecastRequest, ForecastResponse, SeriesForecast


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
                "future_covariates": {"price": [1.1, 1.2, 1.3, 1.4]},
                "static_covariates": {"region": "us-east", "tier": 2},
            }
        ],
        "options": {"temperature": 0.1, "seed": 7},
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
