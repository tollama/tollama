"""Tests for real-data Kaggle credential policy decisions."""

from __future__ import annotations

import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

_MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "e2e_realdata" / "prepare_data.py"
_MODULE_SPEC = spec_from_file_location("scripts_e2e_realdata_policy", _MODULE_PATH)
assert _MODULE_SPEC is not None and _MODULE_SPEC.loader is not None
_MODULE = module_from_spec(_MODULE_SPEC)
sys.modules[_MODULE_SPEC.name] = _MODULE
_MODULE_SPEC.loader.exec_module(_MODULE)

has_kaggle_credentials = _MODULE.has_kaggle_credentials
kaggle_policy_for_mode = _MODULE.kaggle_policy_for_mode


def test_has_kaggle_credentials_detects_both_fields() -> None:
    assert has_kaggle_credentials({"KAGGLE_USERNAME": "u", "KAGGLE_KEY": "k"}) is True
    assert has_kaggle_credentials({"KAGGLE_USERNAME": "u", "KAGGLE_KEY": ""}) is False


def test_pr_mode_falls_back_when_credentials_missing() -> None:
    policy = kaggle_policy_for_mode("pr", credentials_present=False)
    assert policy.include_kaggle is False
    assert policy.hard_fail_on_missing is False
    assert policy.message is not None


def test_nightly_mode_fails_when_credentials_missing() -> None:
    policy = kaggle_policy_for_mode("nightly", credentials_present=False)
    assert policy.include_kaggle is False
    assert policy.hard_fail_on_missing is True
    assert policy.message is not None


def test_nightly_mode_includes_kaggle_when_credentials_present() -> None:
    policy = kaggle_policy_for_mode("nightly", credentials_present=True)
    assert policy.include_kaggle is True
    assert policy.hard_fail_on_missing is False
    assert policy.message is None


def test_local_mode_requires_credentials_by_default() -> None:
    policy = kaggle_policy_for_mode("local", credentials_present=False)
    assert policy.include_kaggle is False
    assert policy.hard_fail_on_missing is True
    assert policy.message is not None


def test_local_mode_allows_explicit_fallback() -> None:
    policy = kaggle_policy_for_mode(
        "local",
        credentials_present=False,
        allow_local_fallback=True,
    )
    assert policy.include_kaggle is False
    assert policy.hard_fail_on_missing is False
    assert policy.message is not None
