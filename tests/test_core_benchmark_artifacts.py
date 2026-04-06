"""Tests for Core benchmark artifact generation."""

from __future__ import annotations

import csv
import json

from tollama.core.benchmark import (
    BenchmarkSummary,
    ModelBenchmarkResult,
    format_operator_summary,
    recommend_routing,
    save_benchmark_bundle,
)
from tollama.core.routing import load_routing_manifest
from tollama.core.storage import TollamaPaths


def _summary() -> BenchmarkSummary:
    return BenchmarkSummary(
        dataset_fingerprint="demo1234abcd5678",
        horizon=4,
        num_folds=2,
        metric_names=["mase", "mae", "rmse"],
        learned_weights={
            "accurate": 0.55,
            "balanced": 0.3,
            "fast": 0.15,
        },
        models=[
            ModelBenchmarkResult(
                model="accurate",
                metrics={"mase": 0.82, "mae": 1.1, "rmse": 1.5},
                latency_ms=260.0,
                folds_evaluated=2,
            ),
            ModelBenchmarkResult(
                model="balanced",
                metrics={"mase": 0.95, "mae": 1.3, "rmse": 1.7},
                latency_ms=140.0,
                folds_evaluated=2,
            ),
            ModelBenchmarkResult(
                model="fast",
                metrics={"mase": 1.15, "mae": 1.8, "rmse": 2.2},
                latency_ms=60.0,
                folds_evaluated=2,
            ),
        ],
    )


def test_recommend_routing_prefers_accuracy_and_latency_roles() -> None:
    recommendation = recommend_routing(_summary())

    assert recommendation["high_accuracy"] == "accurate"
    assert recommendation["fast_path"] == "fast"
    assert recommendation["default"] == "accurate"
    assert recommendation["ranking"][0]["model"] == recommendation["default"]


def test_save_benchmark_bundle_writes_core_artifacts(monkeypatch, tmp_path) -> None:
    summary = _summary()
    output_dir = tmp_path / "bundle"
    artifacts = save_benchmark_bundle(summary, output_dir)

    result_path = artifacts["result"]
    routing_path = artifacts["routing_manifest"]
    leaderboard_path = artifacts["leaderboard"]
    summary_path = artifacts["operator_summary"]
    legacy_path = artifacts["legacy_summary"]

    assert result_path.exists()
    assert routing_path.exists()
    assert leaderboard_path.exists()
    assert summary_path.exists()
    assert legacy_path.exists()

    result_payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert result_payload["artifact_kind"] == "tollama_core_benchmark"
    assert result_payload["schema_version"] == 1
    assert result_payload["source"] == "tollama.core.benchmark"
    assert result_payload["run_id"].startswith("core-benchmark-demo1234abcd5678-")
    assert result_payload["quality_metric_priority"][0] == "mase"
    assert result_payload["routing_recommendation"]["high_accuracy"] == "accurate"
    assert result_payload["artifact_mapping"]["result_json"] == "result.json"
    assert result_payload["artifact_mapping"]["routing_manifest"] == "routing.json"
    assert result_payload["artifact_mapping"]["operator_summary_md"] == "summary.md"

    routing_payload = json.loads(routing_path.read_text(encoding="utf-8"))
    assert routing_payload["version"] == 1
    assert routing_payload["source"] == "tollama.core.benchmark"
    assert routing_payload["routing"]["default"] == "accurate"
    assert routing_payload["routing"]["fast_path"] == "fast"

    with leaderboard_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["model"] == "accurate"
    assert rows[0]["rank"] == "1"

    summary_markdown = summary_path.read_text(encoding="utf-8")
    assert "# Operator Summary" in summary_markdown
    assert "`accurate`" in summary_markdown
    assert "Fast path" in summary_markdown

    monkeypatch.setenv("TOLLAMA_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("TOLLAMA_ROUTING_MANIFEST", str(result_path))
    manifest = load_routing_manifest(paths=TollamaPaths.default())

    assert manifest is not None
    assert manifest.routing.default == "accurate"
    assert manifest.routing.fast_path == "fast"


def test_format_operator_summary_surfaces_lane_recommendations() -> None:
    rendered = format_operator_summary(_summary())

    assert "Recommendation summary:" in rendered
    assert "default: accurate" in rendered
    assert "fast_path: fast" in rendered
    assert "high_accuracy: accurate" in rendered
    assert "Caveats:" in rendered
