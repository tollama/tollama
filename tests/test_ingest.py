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


def test_series_inputs_from_frame_detects_series_column_as_series_id() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": [
                "2025-01-01T00:00:00",
                "2025-01-01T01:00:00",
                "2025-01-01T02:00:00",
                "2025-01-01T00:00:00",
                "2025-01-01T01:00:00",
                "2025-01-01T02:00:00",
            ],
            "series": ["north", "north", "north", "south", "south", "south"],
            "target": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }
    )

    series = series_inputs_from_frame(frame)

    assert [item.id for item in series] == ["north", "south"]
    assert [item.freq for item in series] == ["h", "h"]
    assert [item.target for item in series] == [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]


def test_load_series_inputs_from_data_url_rejects_remote_by_default() -> None:
    with pytest.raises(IngestError):
        load_series_inputs_from_data_url("https://example.com/history.csv")


def test_load_series_inputs_from_bytes_csv() -> None:
    payload = b"timestamp,target\n2025-01-01,1.0\n2025-01-02,2.0\n"
    series = load_series_inputs_from_bytes(payload, filename="upload.csv")

    assert len(series) == 1
    assert series[0].id == "series_0"
    assert series[0].target == [1.0, 2.0]


def test_series_inputs_from_frame_applies_explicit_frequency() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": ["2025-01-01", "2025-01-03", "2025-01-04"],
            "target": [1.0, 2.0, 3.0],
        }
    )

    series = series_inputs_from_frame(frame, freq="D")

    assert series[0].freq == "D"


def test_series_inputs_from_frame_detects_date_and_ot_columns() -> None:
    frame = pd.DataFrame(
        {
            "Date": ["2025-01-01", "2025-01-02", "2025-01-03"],
            "HUFL": [10.0, 11.0, 12.0],
            "OT": [1.0, 2.0, 3.0],
        }
    )

    series = series_inputs_from_frame(frame)

    assert len(series) == 1
    assert series[0].timestamps == ["2025-01-01", "2025-01-02", "2025-01-03"]
    assert series[0].target == [1.0, 2.0, 3.0]


def test_load_series_inputs_from_bytes_handles_bom_date_header() -> None:
    payload = "\ufeffDate,Number of flights\n2025-01-01,10\n2025-01-02,11\n".encode()

    series = load_series_inputs_from_bytes(payload, filename="flights.csv")

    assert series[0].timestamps == ["2025-01-01", "2025-01-02"]
    assert series[0].target == [10, 11]


def test_series_inputs_from_frame_detects_air_quality_pm25_target() -> None:
    frame = pd.DataFrame(
        {
            "datetime": ["2025-01-01 00:00:00", "2025-01-01 01:00:00"],
            "so2": [4.0, 5.0],
            "co": [0.2, 0.3],
            "pm2.5": [12.0, 13.0],
        }
    )

    series = series_inputs_from_frame(frame)

    assert series[0].target == [12.0, 13.0]


def test_series_inputs_from_frame_detects_opsd_load_actual_target() -> None:
    frame = pd.DataFrame(
        {
            "utc_timestamp": ["2025-01-01T00:00:00Z", "2025-01-01T01:00:00Z"],
            "AT_load_actual_entsoe_transparency": [100.0, 101.0],
            "AT_solar_generation_actual": [10.0, 11.0],
            "DE_load_actual_entsoe_transparency": [200.0, 201.0],
        }
    )

    series = series_inputs_from_frame(frame)

    assert series[0].target == [100.0, 101.0]


def test_series_inputs_from_frame_drops_null_target_rows() -> None:
    frame = pd.DataFrame(
        {
            "observation_date": ["2025-01-01", "2025-01-02", "2025-01-03"],
            "CPIAUCSL": [300.0, None, 302.0],
        }
    )

    series = series_inputs_from_frame(frame)

    assert series[0].timestamps == ["2025-01-01", "2025-01-03"]
    assert series[0].target == [300.0, 302.0]


def test_series_inputs_from_frame_rejects_all_null_target_column() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": ["2025-01-01", "2025-01-02"],
            "target": [None, None],
        }
    )

    with pytest.raises(IngestError, match="only null"):
        series_inputs_from_frame(frame)


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
