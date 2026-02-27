"""Tests for real-data gate validation helpers."""

from __future__ import annotations

import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

_MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "e2e_realdata"
    / "validate_results.py"
)
_MODULE_SPEC = spec_from_file_location("scripts_e2e_realdata_validate_results", _MODULE_PATH)
assert _MODULE_SPEC is not None and _MODULE_SPEC.loader is not None
_MODULE = module_from_spec(_MODULE_SPEC)
sys.modules[_MODULE_SPEC.name] = _MODULE
_MODULE_SPEC.loader.exec_module(_MODULE)

evaluate_gate = _MODULE.evaluate_gate
validate_forecast_shape = _MODULE.validate_forecast_shape


def _ok_payload() -> dict[str, object]:
    return {
        "model": "chronos2",
        "forecasts": [
            {
                "id": "s1",
                "freq": "D",
                "start_timestamp": "2025-01-05",
                "mean": [1.0, 2.0],
            }
        ],
        "warnings": [],
        "metrics": {
            "aggregate": {
                "mae": 0.1,
                "rmse": 0.2,
                "smape": 0.3,
            }
        },
    }


def test_validate_forecast_shape_rejects_horizon_mismatch() -> None:
    error = validate_forecast_shape(_ok_payload(), horizon=3)
    assert error is not None
    assert "length mismatch" in error


def test_evaluate_gate_accepts_valid_success_case() -> None:
    passed, error, warnings, metrics = evaluate_gate(
        scenario="benchmark_target_only",
        model="chronos2",
        expected_status=200,
        http_status=200,
        response_payload=_ok_payload(),
        horizon=2,
        exception_detail=None,
    )

    assert passed is True
    assert error is None
    assert warnings == []
    assert metrics["mae"] == 0.1


def test_evaluate_gate_rejects_strict_status_mismatch() -> None:
    passed, error, _, _ = evaluate_gate(
        scenario="contract_strict_covariates",
        model="timesfm-2.5-200m",
        expected_status=400,
        http_status=200,
        response_payload=_ok_payload(),
        horizon=2,
        exception_detail=None,
    )

    assert passed is False
    assert error is not None
    assert "status mismatch" in error


def test_evaluate_gate_requires_best_effort_warnings_for_unsupported_models() -> None:
    payload = _ok_payload()
    payload["warnings"] = []

    passed, error, _, _ = evaluate_gate(
        scenario="contract_best_effort_covariates",
        model="timesfm-2.5-200m",
        expected_status=200,
        http_status=200,
        response_payload=payload,
        horizon=2,
        exception_detail=None,
    )

    assert passed is False
    assert error is not None
    assert "expected best_effort warnings" in error


def test_evaluate_gate_rejects_unhandled_runner_errors() -> None:
    passed, error, _, _ = evaluate_gate(
        scenario="benchmark_target_only",
        model="chronos2",
        expected_status=200,
        http_status=503,
        response_payload={"detail": "runner unavailable"},
        horizon=2,
        exception_detail="forecast failed",
    )

    assert passed is False
    assert error is not None
    assert "unhandled runner error status" in error
