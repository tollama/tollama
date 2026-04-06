"""Validation for the checked-in concrete-solution benchmark input."""

from __future__ import annotations

import json
from pathlib import Path

from tollama.core.backtest import generate_folds
from tollama.core.schemas import SeriesInput


def test_checked_in_core_solution_input_matches_hourly_demand_profile() -> None:
    input_path = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "core_solution_hourly_input.json"
    )

    payload = json.loads(input_path.read_text(encoding="utf-8"))

    assert payload["artifact_kind"] == "tollama_core_benchmark_input"
    assert payload["schema_version"] == 1
    assert payload["solution_profile"] == "hourly_demand_operations"
    assert payload["selected_dataset"] == "checked_in_hourly_demo"
    assert payload["recommended_horizon"] == 24
    assert payload["recommended_models"] == [
        "chronos2",
        "granite-ttm-r2",
        "timesfm-2.5-200m",
        "moirai-2.0-R-small",
    ]
    assert payload["dataset_summary"]["freq_values"] == ["H"]
    assert payload["dataset_summary"]["series_count"] == 1

    series = [SeriesInput.model_validate(item) for item in payload["series"]]
    assert len(series) == 1
    assert series[0].id == "ops-demand:site-a"
    assert series[0].freq == "H"
    assert len(series[0].timestamps or []) == len(series[0].target)
    assert len(series[0].target) == 96

    folds = generate_folds(
        series_length=len(series[0].target),
        horizon=int(payload["recommended_horizon"]),
        num_folds=2,
    )
    assert len(folds) == 2
