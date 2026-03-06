from __future__ import annotations

import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

_MODULE_PATH = Path(__file__).resolve().parents[1] / "benchmarks" / "cross_model_tsfm.py"
_MODULE_SPEC = spec_from_file_location("benchmarks_cross_model_tsfm", _MODULE_PATH)
assert _MODULE_SPEC is not None and _MODULE_SPEC.loader is not None
_MODULE = module_from_spec(_MODULE_SPEC)
sys.modules[_MODULE_SPEC.name] = _MODULE
_MODULE_SPEC.loader.exec_module(_MODULE)

ModelRun = _MODULE.ModelRun
_extract_mean_forecast = _MODULE._extract_mean_forecast
_mase = _MODULE._mase
_recommend_routing = _MODULE._recommend_routing


def test_extract_mean_forecast_truncates_to_horizon() -> None:
    payload = {"forecasts": [{"mean": [1.0, 2.0, 3.0, 4.0]}]}
    result = _extract_mean_forecast(payload, expected_horizon=3)
    assert result == [1.0, 2.0, 3.0]


def test_mase_returns_nan_when_denom_is_zero() -> None:
    actual = [1.0, 1.0]
    pred = [1.0, 1.0]
    insample = [2.0, 2.0, 2.0]
    value = _mase(actual, pred, insample, seasonality=1)
    assert value != value  # NaN check


def test_recommend_routing_prefers_quality_vs_latency() -> None:
    runs = [
        ModelRun(
            model="patchtst",
            dataset="d1",
            status="pass",
            error=None,
            quality={"smape": 10.0, "mase": 1.0},
            latency_ms={"p50": 140.0, "mean": 150.0, "p95": 190.0},
        ),
        ModelRun(
            model="nhits",
            dataset="d1",
            status="pass",
            error=None,
            quality={"smape": 12.0, "mase": 1.2},
            latency_ms={"p50": 80.0, "mean": 90.0, "p95": 120.0},
        ),
        ModelRun(
            model="nbeatsx",
            dataset="d1",
            status="pass",
            error=None,
            quality={"smape": 7.0, "mase": 0.8},
            latency_ms={"p50": 250.0, "mean": 260.0, "p95": 310.0},
        ),
    ]

    recommendation = _recommend_routing(runs)
    assert recommendation["default"] == "nbeatsx"
    assert recommendation["fast_path"] == "nhits"
    assert recommendation["high_accuracy"] == "nbeatsx"
