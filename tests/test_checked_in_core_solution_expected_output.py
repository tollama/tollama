"""Validation for the checked-in expected-output example bundle."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from tollama.core.routing import load_routing_manifest_from_path


def test_checked_in_core_solution_expected_output_bundle_is_consistent() -> None:
    bundle_dir = Path(__file__).resolve().parents[1] / "examples" / "core_solution_expected_output"

    result_payload = json.loads((bundle_dir / "result.json").read_text(encoding="utf-8"))
    routing_payload = json.loads((bundle_dir / "routing.json").read_text(encoding="utf-8"))
    summary_text = (bundle_dir / "summary.md").read_text(encoding="utf-8")

    assert result_payload["artifact_kind"] == "tollama_core_benchmark"
    assert result_payload["schema_version"] == 1
    assert result_payload["eval_ref"] == result_payload["run_id"]
    assert result_payload["forecast_id"].startswith("core-routing-candidate:")
    assert result_payload["routing_recommendation"]["default"] == "chronos2"
    assert result_payload["routing_recommendation"]["fast_path"] == "timesfm-2.5-200m"
    assert result_payload["routing_rationale"]["high_accuracy"]["model"] == "chronos2"

    assert routing_payload["eval_ref"] == result_payload["eval_ref"]
    assert routing_payload["forecast_id"] == result_payload["forecast_id"]
    assert (
        routing_payload["routing"]["default"] == result_payload["routing_recommendation"]["default"]
    )
    assert routing_payload["routing_rationale"]["fast_path"]["model"] == "timesfm-2.5-200m"

    manifest = load_routing_manifest_from_path(bundle_dir / "routing.json")
    assert manifest.routing.default == "chronos2"
    assert manifest.routing.fast_path == "timesfm-2.5-200m"
    assert manifest.routing_rationale["default"]["reason"] == (
        "Best balanced benchmark profile for general workloads."
    )

    with (bundle_dir / "leaderboard.csv").open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["model"] == "chronos2"
    assert rows[-1]["model"] == "timesfm-2.5-200m"
    assert "Default lane: `chronos2`" in summary_text
    assert "Fast path: `timesfm-2.5-200m`" in summary_text
