"""Tests for CSV/Parquet ingest helpers."""

from __future__ import annotations

import pandas as pd
import pytest

from tollama.core.ingest import (
    IngestDependencyError,
    IngestError,
    load_series_inputs_from_bytes,
    load_series_inputs_from_data_url,
    load_series_inputs_from_path,
    series_inputs_from_frame,
)


def test_load_series_inputs_from_csv_path_groups_by_series_id(tmp_path) -> None:
    frame = pd.DataFrame(
        {
            "timestamp": ["2025-01-01", "2025-01-02", "2025-01-01", "2025-01-02"],
            "id": ["north", "north", "south", "south"],
            "target": [1.0, 2.0, 3.0, 4.0],
        }
    )
    path = tmp_path / "history.csv"
    frame.to_csv(path, index=False)

    series = load_series_inputs_from_path(path)

    assert [item.id for item in series] == ["north", "south"]
    assert [item.target for item in series] == [[1.0, 2.0], [3.0, 4.0]]


def test_load_series_inputs_from_data_url_rejects_remote_by_default() -> None:
    with pytest.raises(IngestError):
        load_series_inputs_from_data_url("https://example.com/history.csv")


def test_load_series_inputs_from_bytes_csv() -> None:
    payload = b"timestamp,target\n2025-01-01,1.0\n2025-01-02,2.0\n"
    series = load_series_inputs_from_bytes(payload, filename="upload.csv")

    assert len(series) == 1
    assert series[0].id == "series_0"
    assert series[0].target == [1.0, 2.0]


def test_series_inputs_from_frame_requires_explicit_target_when_ambiguous() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": ["2025-01-01", "2025-01-02"],
            "sales": [1.0, 2.0],
            "orders": [3.0, 4.0],
        }
    )

    with pytest.raises(IngestError, match="multiple numeric columns"):
        series_inputs_from_frame(frame)


def test_load_series_inputs_from_path_parquet_requires_optional_dependency(
    monkeypatch,
    tmp_path,
) -> None:
    path = tmp_path / "history.parquet"
    path.write_bytes(b"dummy")

    def _raise_import_error(*args, **kwargs):  # noqa: ANN002, ANN003
        del args, kwargs
        raise ImportError("pyarrow missing")

    monkeypatch.setattr("pandas.read_parquet", _raise_import_error)

    with pytest.raises(IngestDependencyError):
        load_series_inputs_from_path(path)
