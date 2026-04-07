"""Tests for exporting concrete-solution benchmark input payloads."""

from __future__ import annotations

import json
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

_MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "e2e_realdata"
    / "export_core_solution_input.py"
)
_MODULE_SPEC = spec_from_file_location(
    "scripts_e2e_realdata_export_core_solution_input",
    _MODULE_PATH,
)
assert _MODULE_SPEC is not None and _MODULE_SPEC.loader is not None
_MODULE = module_from_spec(_MODULE_SPEC)
sys.modules[_MODULE_SPEC.name] = _MODULE
_MODULE_SPEC.loader.exec_module(_MODULE)

build_solution_payload = _MODULE.build_solution_payload
prepared_series_to_benchmark_series = _MODULE.prepared_series_to_benchmark_series
select_solution_dataset_name = _MODULE.select_solution_dataset_name


def _prepared_series(*, series_id: str = "pjm_hourly:AEP") -> dict[str, object]:
    return {
        "id": series_id,
        "freq": "H",
        "timestamps": [
            "2024-01-01T00:00:00",
            "2024-01-01T01:00:00",
            "2024-01-01T02:00:00",
        ],
        "target": [100.0, 101.0, 102.0],
        "actuals": [103.0, 104.0],
    }


def test_select_solution_dataset_name_prefers_preferred_dataset() -> None:
    selected, fallback_used = select_solution_dataset_name(
        {
            "pjm_hourly_energy": [_prepared_series()],
            "m4_daily": [_prepared_series(series_id="m4_daily:D1")],
        },
        preferred="pjm_hourly_energy",
        fallback="m4_daily",
    )

    assert selected == "pjm_hourly_energy"
    assert fallback_used is False


def test_select_solution_dataset_name_falls_back_when_needed() -> None:
    selected, fallback_used = select_solution_dataset_name(
        {"m4_daily": [_prepared_series(series_id="m4_daily:D1")]},
        preferred="pjm_hourly_energy",
        fallback="m4_daily",
    )

    assert selected == "m4_daily"
    assert fallback_used is True


def test_prepared_series_to_benchmark_series_concatenates_actuals() -> None:
    converted = prepared_series_to_benchmark_series(_prepared_series())

    assert converted["id"] == "pjm_hourly:AEP"
    assert converted["freq"] == "H"
    assert converted["target"] == [100.0, 101.0, 102.0, 103.0, 104.0]
    assert len(converted["timestamps"]) == len(converted["target"])
    assert converted["timestamps"][-1] == "2024-01-01T04:00:00"


def test_build_solution_payload_writes_core_input_metadata(tmp_path: Path) -> None:
    payload = build_solution_payload(
        dataset_name="pjm_hourly_energy",
        selected_rows=[_prepared_series(), _prepared_series(series_id="pjm_hourly:COMED")],
        preferred_dataset="pjm_hourly_energy",
        fallback_dataset="m4_daily",
        fallback_used=False,
        messages=["demo message"],
    )

    assert payload["artifact_kind"] == "tollama_core_benchmark_input"
    assert payload["schema_version"] == 1
    assert payload["solution_profile"] == "hourly_demand_operations"
    assert payload["selected_dataset"] == "pjm_hourly_energy"
    assert payload["recommended_horizon"] == 2
    assert payload["recommended_models"][0] == "chronos2"
    assert payload["dataset_summary"]["series_count"] == 2
    assert payload["dataset_summary"]["freq_values"] == ["H"]
    assert len(payload["series"]) == 2
    assert payload["messages"] == ["demo message"]

    output_path = tmp_path / "benchmark_input.json"
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    roundtrip = json.loads(output_path.read_text(encoding="utf-8"))
    assert roundtrip["series"][0]["target"][-2:] == [103.0, 104.0]
