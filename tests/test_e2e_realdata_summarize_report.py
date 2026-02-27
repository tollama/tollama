"""Tests for real-data summary aggregation/rendering."""

from __future__ import annotations

import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

_MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "e2e_realdata"
    / "summarize_report.py"
)
_MODULE_SPEC = spec_from_file_location("scripts_e2e_realdata_summarize_report", _MODULE_PATH)
assert _MODULE_SPEC is not None and _MODULE_SPEC.loader is not None
_MODULE = module_from_spec(_MODULE_SPEC)
sys.modules[_MODULE_SPEC.name] = _MODULE
_MODULE_SPEC.loader.exec_module(_MODULE)

render_markdown = _MODULE.render_markdown
summarize_entries = _MODULE.summarize_entries


def test_summarize_entries_counts_pass_fail_skip() -> None:
    entries = [
        {
            "model": "chronos2",
            "scenario": "benchmark_target_only",
            "status": "pass",
            "latency_ms": 100.0,
            "metrics": {"mae": 1.0, "rmse": 2.0, "smape": 3.0},
        },
        {
            "model": "chronos2",
            "scenario": "contract_best_effort_covariates",
            "status": "skip",
            "latency_ms": 0.0,
            "metrics": {},
        },
        {
            "model": "chronos2",
            "scenario": "contract_strict_covariates",
            "status": "fail",
            "latency_ms": 5.0,
            "metrics": {},
        },
    ]

    summary = summarize_entries(entries)

    assert summary["total_entries"] == 3
    assert summary["total_failed"] == 1
    assert summary["total_skipped"] == 1
    assert summary["gate_pass"] is False

    model = summary["models"]["chronos2"]
    assert model["passed"] == 1
    assert model["failed"] == 1
    assert model["skipped"] == 1
    assert model["benchmark_metrics"]["mae"] == 1.0
    assert model["scenarios"]["contract_best_effort_covariates"]["skip"] == 1


def test_render_markdown_includes_skip_columns() -> None:
    summary = {
        "gate_pass": True,
        "total_entries": 1,
        "total_failed": 0,
        "total_skipped": 1,
        "models": {
            "chronos2": {
                "passed": 0,
                "failed": 0,
                "skipped": 1,
                "total": 1,
                "mean_latency_ms": None,
                "benchmark_metrics": {"mae": None, "rmse": None, "smape": None},
            }
        },
    }

    markdown = render_markdown(summary)

    assert "Total skipped" in markdown
    assert "| chronos2 | 0/1 | 1 |" in markdown
