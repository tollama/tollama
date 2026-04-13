"""Tests for real-data benchmark report aggregation/rendering."""

from __future__ import annotations

import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

_MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "e2e_realdata" / "benchmark_report.py"
)
_MODULE_SPEC = spec_from_file_location("scripts_e2e_realdata_benchmark_report", _MODULE_PATH)
assert _MODULE_SPEC is not None and _MODULE_SPEC.loader is not None
_MODULE = module_from_spec(_MODULE_SPEC)
sys.modules[_MODULE_SPEC.name] = _MODULE
_MODULE_SPEC.loader.exec_module(_MODULE)

build_benchmark_report = _MODULE.build_benchmark_report
render_markdown = _MODULE.render_markdown


def _entry(
    *,
    model: str,
    dataset: str,
    scenario: str,
    status: str,
    series_id: str,
    latency_ms: float,
    metrics: dict[str, float] | None = None,
    error: str | None = None,
    http_status: int | None = 200,
) -> dict[str, object]:
    return {
        "model": model,
        "dataset": dataset,
        "series_id": series_id,
        "scenario": scenario,
        "status": status,
        "latency_ms": latency_ms,
        "retry_count": 0,
        "warnings": ["unsupported covariates"] if status == "fail" else [],
        "metrics": metrics or {},
        "error": error,
        "http_status": http_status,
    }


def test_build_benchmark_report_aggregates_rows_leaderboards_and_contracts() -> None:
    report = build_benchmark_report(
        {
            "run_id": "run-1",
            "finished_at": "2026-04-12T10:00:00+00:00",
            "base_url": "http://127.0.0.1:11435",
            "gate_profile": "hf_optional",
            "catalog_path": "/tmp/catalog.yaml",
            "models": ["chronos2", "nhits", "patchtst"],
            "datasets": ["ds1", "ds2"],
            "max_series_per_dataset": 1,
            "scenario_policy": {"benchmark_target_only": {"models": ["chronos2", "nhits"]}},
            "messages": ["local fallback enabled"],
            "entries": [
                _entry(
                    model="chronos2",
                    dataset="ds1",
                    scenario="benchmark_target_only",
                    status="pass",
                    series_id="s1",
                    latency_ms=100.0,
                    metrics={"mae": 1.0, "rmse": 2.0, "smape": 3.0, "mape": 4.0, "mase": 5.0},
                ),
                _entry(
                    model="chronos2",
                    dataset="ds2",
                    scenario="benchmark_target_only",
                    status="fail",
                    series_id="s2",
                    latency_ms=0.0,
                    error="status mismatch for benchmark_target_only: expected 200, got 400",
                    http_status=400,
                ),
                _entry(
                    model="nhits",
                    dataset="ds1",
                    scenario="benchmark_target_only",
                    status="pass",
                    series_id="s3",
                    latency_ms=300.0,
                    metrics={"mae": 2.0, "rmse": 3.0, "smape": 4.0, "mape": 5.0, "mase": 6.0},
                ),
                _entry(
                    model="nhits",
                    dataset="ds2",
                    scenario="benchmark_target_only",
                    status="pass",
                    series_id="s4",
                    latency_ms=500.0,
                    metrics={"mae": 4.0, "rmse": 5.0, "smape": 6.0, "mape": 7.0, "mase": 8.0},
                ),
                _entry(
                    model="patchtst",
                    dataset="ds2",
                    scenario="benchmark_target_only",
                    status="skip",
                    series_id="s5",
                    latency_ms=0.0,
                    error="payload_build_error: unsupported model",
                    http_status=None,
                ),
                _entry(
                    model="chronos2",
                    dataset="ds1",
                    scenario="contract_best_effort_covariates",
                    status="pass",
                    series_id="s1",
                    latency_ms=10.0,
                ),
                _entry(
                    model="nhits",
                    dataset="ds1",
                    scenario="contract_strict_covariates",
                    status="fail",
                    series_id="s3",
                    latency_ms=20.0,
                    error="status mismatch for contract_strict_covariates: expected 400, got 200",
                    http_status=200,
                ),
            ],
        }
    )

    assert report["run_id"] == "run-1"
    assert report["benchmark_metric_names"] == ["mae", "rmse", "smape", "mape", "mase"]
    assert len(report["benchmark_rows"]) == 5

    chronos2 = next(row for row in report["model_leaderboard"] if row["model"] == "chronos2")
    nhits = next(row for row in report["model_leaderboard"] if row["model"] == "nhits")
    patchtst = next(row for row in report["model_leaderboard"] if row["model"] == "patchtst")

    assert chronos2["rank"] == 1
    assert chronos2["passed"] == 1
    assert chronos2["failed"] == 1
    assert chronos2["success_rate"] == 0.5
    assert chronos2["mean_metrics"]["smape"] == 3.0

    assert nhits["rank"] == 2
    assert nhits["mean_latency_ms"] == 400.0
    assert nhits["latency_p50_ms"] == 400.0
    assert nhits["latency_p95_ms"] == 490.0

    assert patchtst["rank"] is None
    assert patchtst["skipped"] == 1

    ds2 = next(row for row in report["dataset_breakdown"] if row["dataset"] == "ds2")
    assert ds2["rows"] == 3
    assert ds2["success_rate"] == 0.333333
    assert ds2["failure_classification_counts"] == {"BAD_REQUEST": 1}

    assert report["failure_summary"]["benchmark_by_error_classification"] == {"BAD_REQUEST": 1}
    assert report["contract_summary"]["contract_best_effort_covariates"] == {
        "pass": 1,
        "fail": 0,
        "skip": 0,
        "total": 1,
    }
    assert report["contract_summary"]["contract_strict_covariates"] == {
        "pass": 0,
        "fail": 1,
        "skip": 0,
        "total": 1,
    }


def test_render_markdown_includes_leaderboard_and_failure_sections() -> None:
    report = build_benchmark_report(
        {
            "run_id": "run-2",
            "finished_at": "2026-04-12T10:00:00+00:00",
            "base_url": "http://127.0.0.1:11435",
            "gate_profile": "hf_optional",
            "catalog_path": "/tmp/catalog.yaml",
            "models": ["chronos2"],
            "datasets": ["ds1"],
            "max_series_per_dataset": 1,
            "scenario_policy": {},
            "entries": [
                _entry(
                    model="chronos2",
                    dataset="ds1",
                    scenario="benchmark_target_only",
                    status="fail",
                    series_id="s1",
                    latency_ms=0.0,
                    error="dependency_missing: install extra",
                    http_status=503,
                )
            ],
        }
    )

    markdown = render_markdown(report)

    assert "## Model Leaderboard" in markdown
    assert "## Dataset Breakdown" in markdown
    assert "## Benchmark Rows" in markdown
    assert "## Contract Summary" in markdown
    assert "DEPENDENCY_GATED" in markdown
