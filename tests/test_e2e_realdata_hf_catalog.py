"""Tests for HuggingFace dataset catalog gathering helpers."""

from __future__ import annotations

import py_compile
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

_MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "e2e_realdata" / "gather_hf_datasets.py"
)
_MODULE_SPEC = spec_from_file_location("scripts_e2e_realdata_gather_hf_datasets", _MODULE_PATH)
assert _MODULE_SPEC is not None and _MODULE_SPEC.loader is not None
_MODULE = module_from_spec(_MODULE_SPEC)
sys.modules[_MODULE_SPEC.name] = _MODULE
_MODULE_SPEC.loader.exec_module(_MODULE)

assess_schema_rows = _MODULE.assess_schema_rows
infer_schema_from_rows = _MODULE.infer_schema_from_rows
is_schema_acceptable = _MODULE.is_schema_acceptable


def test_gather_hf_script_compiles() -> None:
    py_compile.compile(str(_MODULE_PATH), doraise=True)


def test_infer_schema_from_rows_is_deterministic() -> None:
    rows = [
        {"date": "2024-01-01", "value": "1.0", "note": "a"},
        {"date": "2024-01-02", "value": "2.0", "note": "b"},
        {"date": "2024-01-03", "value": "3.0", "note": "c"},
    ]

    first = infer_schema_from_rows(rows)
    second = infer_schema_from_rows(rows)

    assert first == second
    assert first == ("date", "value")


def test_schema_quality_rejects_non_time_non_numeric_mapping() -> None:
    rows = [
        {"instruction": "predict tomorrow", "input": "series a"},
        {"instruction": "predict next week", "input": "series b"},
        {"instruction": "predict next month", "input": "series c"},
    ]

    timestamp_col, target_col = infer_schema_from_rows(rows)
    assert timestamp_col is not None
    assert target_col is not None

    assessment = assess_schema_rows(
        rows=rows,
        split="train",
        timestamp_column=timestamp_col,
        target_column=target_col,
    )
    accepted, detail = is_schema_acceptable(
        assessment=assessment,
        min_timestamp_ratio=0.90,
        min_target_ratio=0.90,
        min_contiguous_rows=5,
    )

    assert accepted is False
    assert detail is not None
