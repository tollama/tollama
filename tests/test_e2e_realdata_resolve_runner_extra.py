"""Tests for resolving workflow runner extras from registry models."""

from __future__ import annotations

import subprocess
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest

_MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "e2e_realdata" / "resolve_runner_extra.py"
)
_MODULE_SPEC = spec_from_file_location("scripts_e2e_realdata_resolve_runner_extra", _MODULE_PATH)
assert _MODULE_SPEC is not None and _MODULE_SPEC.loader is not None
_MODULE = module_from_spec(_MODULE_SPEC)
sys.modules[_MODULE_SPEC.name] = _MODULE
_MODULE_SPEC.loader.exec_module(_MODULE)

resolve_runner_extra = _MODULE.resolve_runner_extra


@pytest.mark.parametrize(
    ("model", "expected_extra"),
    [
        ("chronos2", "runner_torch"),
        ("granite-ttm-r2", "runner_torch"),
        ("timesfm-2.5-200m", "runner_timesfm"),
        ("moirai-2.0-R-small", "runner_uni2ts"),
        ("sundial-base-128m", "runner_sundial"),
        ("toto-open-base-1.0", "runner_toto"),
        ("patchtst", "runner_patchtst"),
        ("timer-base", "runner_timer"),
    ],
)
def test_resolve_runner_extra_returns_expected_extra(
    model: str,
    expected_extra: str,
) -> None:
    assert resolve_runner_extra(model) == expected_extra


def test_resolve_runner_extra_rejects_local_models_without_runtime_extra() -> None:
    with pytest.raises(ValueError, match="does not declare a runner extra"):
        resolve_runner_extra("mock")


def test_resolve_runner_extra_script_runs_standalone() -> None:
    completed = subprocess.run(
        [sys.executable, str(_MODULE_PATH), "--model", "timesfm-2.5-200m"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert completed.stdout.strip() == "runner_timesfm"
    assert completed.stderr == ""
