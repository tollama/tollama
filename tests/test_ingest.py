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
    series_inputs_result_from_frame,
)
from tollama.core.schemas import IngestPreprocessingOptions, MissingValuePreprocessingOptions


def _missing_preprocessing(
    *,
    method: str = "linear",
    max_missing_ratio: float = 0.30,
    max_gap: int | None = 24,
    edge_strategy: str = "nearest",
    seasonal_period: int | None = None,
) -> IngestPreprocessingOptions:
    return IngestPreprocessingOptions(
        missing=MissingValuePreprocessingOptions(
            enabled=True,
            method=method,
            max_missing_ratio=max_missing_ratio,
            max_gap=max_gap,
            edge_strategy=edge_strategy,
            seasonal_period=seasonal_period,
        )
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


def test_load_series_inputs_from_bytes_reads_semicolon_csv() -> None:
    payload = (
        b'"timestamp";"target"\n'
        b'"2025-01-01 00:00:00";1.0\n'
        b'"2025-01-01 01:00:00";2.0\n'
        b'"2025-01-01 02:00:00";3.0\n'
    )

    series = load_series_inputs_from_bytes(payload, filename="upload.csv")

    assert series[0].freq == "h"
    assert series[0].target == [1.0, 2.0, 3.0]


def test_load_series_inputs_from_bytes_skips_world_bank_preamble_csv() -> None:
    payload = (
        b'"Data Source","World Development Indicators"\n'
        b"\n"
        b'"Last Updated Date","2026-04-08"\n'
        b"\n"
        b'"Country Name","Country Code","Indicator Name","Indicator Code","1960","1961","1962"\n'
        b'"Aruba","ABW","GDP per capita","NY.GDP.PCAP.CD","1.0","2.0","3.0"\n'
        b'"Afghanistan","AFG","GDP per capita","NY.GDP.PCAP.CD","","5.0","6.0"\n'
    )

    series = load_series_inputs_from_bytes(payload, filename="world_bank.csv")

    assert [item.id for item in series] == ["ABW", "AFG"]
    assert series[0].timestamps == ["1960-01-01", "1961-01-01", "1962-01-01"]
    assert series[0].target == [1.0, 2.0, 3.0]
    assert series[1].timestamps == ["1961-01-01", "1962-01-01"]
    assert series[1].target == [5.0, 6.0]


def test_load_series_inputs_from_bytes_reports_malformed_csv() -> None:
    payload = (
        b"timestamp,target,extra\n"
        b"2025-01-01,1.0,x\n"
        b"2025-01-02,2.0,y\n"
        b"broken,row,with,too,many,fields\n"
    )

    with pytest.raises(IngestError, match="unable to parse CSV uploaded CSV"):
        load_series_inputs_from_bytes(payload, filename="upload.csv")


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


def test_series_inputs_from_frame_infers_dominant_frequency_after_null_target_drop() -> None:
    frame = pd.DataFrame(
        {
            "datetime": [
                "2025-01-01 00:00:00",
                "2025-01-01 01:00:00",
                "2025-01-01 02:00:00",
                "2025-01-01 03:00:00",
                "2025-01-01 04:00:00",
                "2025-01-01 05:00:00",
                "2025-01-01 06:00:00",
            ],
            "pm2.5": [10.0, 11.0, None, 13.0, 14.0, 15.0, 16.0],
        }
    )

    series = series_inputs_from_frame(frame)

    assert series[0].timestamps == [
        "2025-01-01 00:00:00",
        "2025-01-01 01:00:00",
        "2025-01-01 03:00:00",
        "2025-01-01 04:00:00",
        "2025-01-01 05:00:00",
        "2025-01-01 06:00:00",
    ]
    assert series[0].freq == "h"


def test_series_inputs_from_frame_infers_frequency_before_null_target_drop() -> None:
    frame = pd.DataFrame(
        {
            "datetime": [
                "2025-01-01 00:00:00",
                "2025-01-01 01:00:00",
                "2025-01-01 02:00:00",
                "2025-01-01 03:00:00",
                "2025-01-01 04:00:00",
                "2025-01-01 05:00:00",
                "2025-01-01 06:00:00",
            ],
            "pm2.5": [10.0, None, 12.0, None, 14.0, None, 16.0],
        }
    )

    series = series_inputs_from_frame(frame)

    assert series[0].timestamps == [
        "2025-01-01 00:00:00",
        "2025-01-01 02:00:00",
        "2025-01-01 04:00:00",
        "2025-01-01 06:00:00",
    ]
    assert series[0].freq == "h"


def test_series_inputs_from_frame_treats_freq_auto_as_infer_request() -> None:
    frame = pd.DataFrame(
        {
            "datetime": [
                "2025-01-01 00:00:00",
                "2025-01-01 01:00:00",
                "2025-01-01 02:00:00",
                "2025-01-01 03:00:00",
            ],
            "pm2.5": [10.0, None, 12.0, 13.0],
        }
    )

    series = series_inputs_from_frame(frame, freq="auto")

    assert series[0].freq == "h"


def test_series_inputs_from_frame_infers_large_mostly_regular_frequency() -> None:
    timestamps = []
    current = pd.Timestamp("2025-01-01 00:00:00")
    for index in range(1_201):
        timestamps.append(current.strftime("%Y-%m-%d %H:%M:%S"))
        step_hours = 2 if index % 3 == 2 else 1
        current += pd.Timedelta(hours=step_hours)
    frame = pd.DataFrame(
        {
            "datetime": timestamps,
            "pm2.5": [float(index) for index in range(len(timestamps))],
        }
    )

    series = series_inputs_from_frame(frame)

    assert series[0].freq == "h"


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


def test_series_inputs_result_default_has_no_preprocessing_diagnostics() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": ["2025-01-01", "2025-01-02", "2025-01-03"],
            "target": [1.0, None, 3.0],
        }
    )

    result = series_inputs_result_from_frame(frame)

    assert result.preprocessing == []
    assert result.series[0].target == [1.0, 3.0]


def test_series_inputs_result_linear_preprocessing_regularizes_missing_values() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": [
                "2025-01-01 00:00:00",
                "2025-01-01 01:00:00",
                "2025-01-01 02:00:00",
                "2025-01-01 04:00:00",
                "2025-01-01 05:00:00",
            ],
            "target": [10.0, None, 12.0, 14.0, 15.0],
        }
    )

    result = series_inputs_result_from_frame(
        frame,
        preprocessing=_missing_preprocessing(method="linear", max_missing_ratio=0.5),
    )

    series = result.series[0]
    assert series.freq == "h"
    assert series.target == [10.0, 11.0, 12.0, 13.0, 14.0, 15.0]
    diagnostic = result.preprocessing[0]
    assert diagnostic.original_row_count == 5
    assert diagnostic.regularized_row_count == 6
    assert diagnostic.raw_null_target_count == 1
    assert diagnostic.missing_timestamp_count == 1
    assert diagnostic.imputed_point_count == 2
    assert diagnostic.used_method == "linear"


def test_series_inputs_result_bspline_preprocessing_fills_internal_gap() -> None:
    pytest.importorskip("scipy")
    frame = pd.DataFrame(
        {
            "timestamp": [
                "2025-01-01 00:00:00",
                "2025-01-01 01:00:00",
                "2025-01-01 02:00:00",
                "2025-01-01 03:00:00",
                "2025-01-01 04:00:00",
            ],
            "target": [1.0, 2.0, None, 4.0, 5.0],
        }
    )

    result = series_inputs_result_from_frame(
        frame,
        preprocessing=_missing_preprocessing(method="bspline"),
    )

    assert len(result.series[0].target) == 5
    assert result.preprocessing[0].used_method == "bspline"
    assert result.preprocessing[0].imputed_point_count == 1


def test_series_inputs_result_auto_falls_back_to_linear_without_scipy(monkeypatch) -> None:
    def _raise_missing_dependency(values):  # noqa: ANN001
        del values
        raise IngestDependencyError("scipy missing")

    monkeypatch.setattr("tollama.core.ingest._bspline_interpolate", _raise_missing_dependency)
    frame = pd.DataFrame(
        {
            "timestamp": [
                "2025-01-01 00:00:00",
                "2025-01-01 01:00:00",
                "2025-01-01 02:00:00",
                "2025-01-01 03:00:00",
                "2025-01-01 04:00:00",
            ],
            "target": [1.0, 2.0, None, 4.0, 5.0],
        }
    )

    result = series_inputs_result_from_frame(
        frame,
        preprocessing=_missing_preprocessing(method="auto"),
    )

    assert result.preprocessing[0].used_method == "linear"
    assert result.preprocessing[0].warnings == [
        "scipy unavailable for auto missing preprocessing; used linear"
    ]


def test_series_inputs_result_explicit_bspline_surfaces_dependency_error(monkeypatch) -> None:
    def _raise_missing_dependency(values):  # noqa: ANN001
        del values
        raise IngestDependencyError("scipy missing")

    monkeypatch.setattr("tollama.core.ingest._bspline_interpolate", _raise_missing_dependency)
    frame = pd.DataFrame(
        {
            "timestamp": [
                "2025-01-01 00:00:00",
                "2025-01-01 01:00:00",
                "2025-01-01 02:00:00",
                "2025-01-01 03:00:00",
                "2025-01-01 04:00:00",
            ],
            "target": [1.0, 2.0, None, 4.0, 5.0],
        }
    )

    with pytest.raises(IngestDependencyError, match="scipy missing"):
        series_inputs_result_from_frame(
            frame,
            preprocessing=_missing_preprocessing(method="bspline"),
        )


def test_series_inputs_result_seasonal_preprocessing_uses_period() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": [f"2025-01-01 {hour:02d}:00:00" for hour in range(8)],
            "target": [1.0, 10.0, 2.0, 20.0, None, 30.0, 4.0, 40.0],
        }
    )

    result = series_inputs_result_from_frame(
        frame,
        preprocessing=_missing_preprocessing(method="seasonal", seasonal_period=2),
    )

    assert result.series[0].target[4] == 7.0 / 3.0
    assert result.preprocessing[0].used_method == "seasonal"


def test_series_inputs_result_rejects_edge_missing_when_configured() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": ["2025-01-01", "2025-01-02", "2025-01-03"],
            "target": [None, 2.0, 3.0],
        }
    )

    with pytest.raises(IngestError, match="leading or trailing"):
        series_inputs_result_from_frame(
            frame,
            preprocessing=_missing_preprocessing(
                method="linear",
                max_missing_ratio=1.0,
                edge_strategy="reject",
            ),
        )


def test_series_inputs_result_rejects_all_null_target_with_preprocessing() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": ["2025-01-01", "2025-01-02"],
            "target": [None, None],
        }
    )

    with pytest.raises(IngestError, match="only null"):
        series_inputs_result_from_frame(
            frame,
            preprocessing=_missing_preprocessing(method="linear"),
        )


def test_series_inputs_result_rejects_missing_gap_over_limit() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": [
                "2025-01-01 00:00:00",
                "2025-01-01 01:00:00",
                "2025-01-01 02:00:00",
                "2025-01-01 03:00:00",
                "2025-01-01 04:00:00",
            ],
            "target": [1.0, None, None, None, 5.0],
        }
    )

    with pytest.raises(IngestError, match="max missing gap"):
        series_inputs_result_from_frame(
            frame,
            preprocessing=_missing_preprocessing(
                method="linear",
                max_missing_ratio=1.0,
                max_gap=2,
            ),
        )


def test_series_inputs_result_rejects_missing_ratio_over_limit() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": ["2025-01-01", "2025-01-02", "2025-01-03"],
            "target": [1.0, None, 3.0],
        }
    )

    with pytest.raises(IngestError, match="missing ratio"):
        series_inputs_result_from_frame(
            frame,
            preprocessing=_missing_preprocessing(method="linear", max_missing_ratio=0.30),
        )


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
