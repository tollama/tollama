"""Tests for real-data dataset preparation helpers."""

from __future__ import annotations

import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

_MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "e2e_realdata" / "prepare_data.py"
_MODULE_SPEC = spec_from_file_location("scripts_e2e_realdata_prepare_data", _MODULE_PATH)
assert _MODULE_SPEC is not None and _MODULE_SPEC.loader is not None
_MODULE = module_from_spec(_MODULE_SPEC)
sys.modules[_MODULE_SPEC.name] = _MODULE
_MODULE_SPEC.loader.exec_module(_MODULE)

parse_kaggle_hourly_directory = _MODULE.parse_kaggle_hourly_directory
parse_m4_daily_files = _MODULE.parse_m4_daily_files
sample_size_for_mode = _MODULE.sample_size_for_mode


def test_sample_size_for_mode_is_deterministic() -> None:
    assert sample_size_for_mode("pr") == 1
    assert sample_size_for_mode("nightly") == 4
    assert sample_size_for_mode("local") == 2


def test_parse_m4_daily_files_applies_horizon_and_context(tmp_path: Path) -> None:
    train_path = tmp_path / "Daily-train.csv"
    test_path = tmp_path / "Daily-test.csv"

    train_path.write_text(
        "V1,V2,V3,V4,V5\n"
        "D1,1,2,3,4\n"
        "D2,10,11,12,13\n"
        "D3,20,21,22,23\n",
        encoding="utf-8",
    )
    test_path.write_text(
        "V1,V2,V3\n"
        "D1,5,6\n"
        "D2,14,15\n"
        "D3,24,25\n",
        encoding="utf-8",
    )

    first = parse_m4_daily_files(
        train_path=train_path,
        test_path=test_path,
        horizon=2,
        context_cap=3,
        max_series=2,
        seed=42,
    )
    second = parse_m4_daily_files(
        train_path=train_path,
        test_path=test_path,
        horizon=2,
        context_cap=3,
        max_series=2,
        seed=42,
    )

    assert first == second
    assert len(first) == 2
    for item in first:
        assert len(item["target"]) <= 3
        assert len(item["actuals"]) == 2
        assert len(item["timestamps"]) == len(item["target"])


def test_parse_kaggle_hourly_directory_parses_csv_files(tmp_path: Path) -> None:
    csv_a = tmp_path / "AEP_hourly.csv"
    csv_b = tmp_path / "COMED_hourly.csv"

    csv_a.write_text(
        "Datetime,AEP_MW\n"
        "2024-01-01 00:00:00,100\n"
        "2024-01-01 01:00:00,101\n"
        "2024-01-01 02:00:00,102\n"
        "2024-01-01 03:00:00,103\n"
        "2024-01-01 04:00:00,104\n",
        encoding="utf-8",
    )
    csv_b.write_text(
        "Datetime,COMED_MW\n"
        "2024-01-01 00:00:00,200\n"
        "2024-01-01 01:00:00,201\n"
        "2024-01-01 02:00:00,202\n"
        "2024-01-01 03:00:00,203\n"
        "2024-01-01 04:00:00,204\n",
        encoding="utf-8",
    )

    series = parse_kaggle_hourly_directory(
        data_dir=tmp_path,
        horizon=2,
        context_cap=3,
        max_series=1,
        seed=42,
    )

    assert len(series) == 1
    row = series[0]
    assert row["freq"] == "H"
    assert len(row["target"]) == 3
    assert len(row["actuals"]) == 2
    assert len(row["timestamps"]) == len(row["target"])
