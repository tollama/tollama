"""Tests for real-data payload builder helpers."""

from __future__ import annotations

import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

_MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "e2e_realdata"
    / "payload_builder.py"
)
_MODULE_SPEC = spec_from_file_location("scripts_e2e_realdata_payload_builder", _MODULE_PATH)
assert _MODULE_SPEC is not None and _MODULE_SPEC.loader is not None
_MODULE = module_from_spec(_MODULE_SPEC)
sys.modules[_MODULE_SPEC.name] = _MODULE
_MODULE_SPEC.loader.exec_module(_MODULE)

build_covariate_request = _MODULE.build_covariate_request
build_target_only_request = _MODULE.build_target_only_request
expected_status_for_strict_covariates = _MODULE.expected_status_for_strict_covariates
filter_covariates_for_model = _MODULE.filter_covariates_for_model


def _sample_series() -> dict[str, object]:
    return {
        "id": "s1",
        "freq": "D",
        "timestamps": ["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-04"],
        "target": [1.0, 2.0, 3.0, 4.0],
        "actuals": [5.0, 6.0],
    }


def test_build_target_only_request_includes_metrics() -> None:
    payload = build_target_only_request(
        model="chronos2",
        series=_sample_series(),
        horizon=2,
        timeout_seconds=900.0,
    )

    assert payload["model"] == "chronos2"
    assert payload["horizon"] == 2
    assert payload["parameters"]["metrics"]["names"] == ["mae", "rmse", "smape", "mape", "mase"]


def test_build_covariate_request_builds_known_future_covariates() -> None:
    payload = build_covariate_request(
        model="timesfm-2.5-200m",
        series=_sample_series(),
        horizon=2,
        covariates_mode="strict",
        timeout_seconds=900.0,
    )

    series = payload["series"][0]
    assert payload["parameters"]["covariates_mode"] == "strict"
    assert "calendar_num" in series["past_covariates"]
    assert "calendar_num" in series["future_covariates"]
    assert "calendar_cat" in series["past_covariates"]
    assert "calendar_cat" in series["future_covariates"]


def test_filter_covariates_for_model_applies_family_rules() -> None:
    past = {
        "calendar_num": [1.0, 2.0, 3.0],
        "calendar_cat": ["weekday", "weekday", "weekend"],
    }
    future = {
        "calendar_num": [4.0, 5.0],
        "calendar_cat": ["weekday", "weekend"],
    }

    toto_past, toto_future = filter_covariates_for_model(
        model="toto-open-base-1.0",
        past_covariates=past,
        future_covariates=future,
    )
    assert set(toto_past) == {"calendar_num"}
    assert toto_future == {}

    sundial_past, sundial_future = filter_covariates_for_model(
        model="sundial-base-128m",
        past_covariates=past,
        future_covariates=future,
    )
    assert sundial_past == {}
    assert sundial_future == {}


def test_expected_status_for_strict_covariates_mapping() -> None:
    assert expected_status_for_strict_covariates("chronos2") == 200
    assert expected_status_for_strict_covariates("timesfm-2.5-200m") == 400
