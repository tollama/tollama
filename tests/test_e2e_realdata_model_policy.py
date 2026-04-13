"""Tests for real-data HF starter model policy helpers."""

from __future__ import annotations

import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

_MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "e2e_realdata" / "model_policy.py"
_MODULE_SPEC = spec_from_file_location("scripts_e2e_realdata_model_policy", _MODULE_PATH)
assert _MODULE_SPEC is not None and _MODULE_SPEC.loader is not None
_MODULE = module_from_spec(_MODULE_SPEC)
sys.modules[_MODULE_SPEC.name] = _MODULE
_MODULE_SPEC.loader.exec_module(_MODULE)

CONTRACT_BEST_EFFORT = _MODULE.CONTRACT_BEST_EFFORT
CONTRACT_STRICT = _MODULE.CONTRACT_STRICT
resolve_models = _MODULE.resolve_models
scenario_policy_summary = _MODULE.scenario_policy_summary
scenarios_for_model = _MODULE.scenarios_for_model
strict_expected_status_for_model = _MODULE.strict_expected_status_for_model


def test_resolve_models_supports_all_aliases() -> None:
    assert resolve_models("all") == [
        "chronos2",
        "granite-ttm-r2",
        "timesfm-2.5-200m",
        "moirai-2.0-R-small",
        "sundial-base-128m",
        "toto-open-base-1.0",
    ]
    assert resolve_models("hf_all") == [
        "chronos2",
        "granite-ttm-r2",
        "timesfm-2.5-200m",
        "moirai-2.0-R-small",
        "sundial-base-128m",
        "toto-open-base-1.0",
        "lag-llama",
        "patchtst",
        "tide",
        "nhits",
        "nbeatsx",
    ]
    assert resolve_models("neural") == [
        "lag-llama",
        "patchtst",
        "tide",
        "nhits",
        "nbeatsx",
    ]


def test_scenarios_for_model_match_hf_starter_policy() -> None:
    assert scenarios_for_model("lag-llama") == ["benchmark_target_only"]
    assert scenarios_for_model("patchtst") == ["benchmark_target_only"]
    assert scenarios_for_model("tide") == ["benchmark_target_only"]
    assert scenarios_for_model("nhits") == [
        "benchmark_target_only",
        CONTRACT_BEST_EFFORT,
        CONTRACT_STRICT,
    ]


def test_strict_status_mapping_covers_neural_contract_models() -> None:
    assert strict_expected_status_for_model("chronos2") == 200
    assert strict_expected_status_for_model("nhits") == 400
    assert strict_expected_status_for_model("nbeatsx") == 400


def test_scenario_policy_summary_exposes_11_benchmark_and_8_contract_models() -> None:
    summary = scenario_policy_summary()

    assert len(summary["benchmark_target_only"]["models"]) == 11
    assert len(summary[CONTRACT_BEST_EFFORT]["models"]) == 8
    assert len(summary[CONTRACT_STRICT]["models"]) == 8
