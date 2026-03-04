"""Unit tests for Lag-Llama adapter helper behavior without network access."""

from __future__ import annotations

import pytest

from tollama.runners.lag_llama_runner.adapter import (
    build_covariate_warnings,
    build_quantile_payload,
    resolve_positive_int,
)
from tollama.runners.lag_llama_runner.errors import AdapterInputError


class _Forecast:
    def quantile(self, value: float):
        if value == 0.1:
            return [1.0, 2.0]
        if value == 0.9:
            return [3.0, 4.0]
        raise KeyError(value)


class _NoQuantileForecast:
    mean = [1.0, 2.0]


def test_resolve_positive_int_validates_option() -> None:
    assert resolve_positive_int(option_value=None, default_value=7, field_name="x") == 7
    with pytest.raises(AdapterInputError, match="must be an integer"):
        resolve_positive_int(option_value="3", default_value=7, field_name="x")


def test_build_quantile_payload_maps_requested_quantiles() -> None:
    payload = build_quantile_payload(
        forecast=_Forecast(),
        requested_quantiles=[0.1, 0.9],
        horizon=2,
    )
    assert payload == {"0.1": [1.0, 2.0], "0.9": [3.0, 4.0]}


def test_build_quantile_payload_requires_quantile_method() -> None:
    with pytest.raises(AdapterInputError, match="no quantile method"):
        build_quantile_payload(
            forecast=_NoQuantileForecast(),
            requested_quantiles=[0.5],
            horizon=2,
        )


def test_build_covariate_warnings_when_covariates_supplied() -> None:
    class _Series:
        past_covariates = {"x": [1.0, 2.0]}
        future_covariates = None
        static_covariates = None

    warnings = build_covariate_warnings([_Series()])
    assert warnings
    assert "ignores covariates" in warnings[0]
