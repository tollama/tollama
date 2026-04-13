"""Tests for real-data dataset preparation helpers."""

from __future__ import annotations

import sys
import zipfile
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest

_MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "e2e_realdata" / "prepare_data.py"
_MODULE_SPEC = spec_from_file_location("scripts_e2e_realdata_prepare_data", _MODULE_PATH)
assert _MODULE_SPEC is not None and _MODULE_SPEC.loader is not None
_MODULE = module_from_spec(_MODULE_SPEC)
sys.modules[_MODULE_SPEC.name] = _MODULE
_MODULE_SPEC.loader.exec_module(_MODULE)

parse_kaggle_hourly_directory = _MODULE.parse_kaggle_hourly_directory
parse_huggingface_dataset = _MODULE.parse_huggingface_dataset
parse_m4_daily_files = _MODULE.parse_m4_daily_files
sample_size_for_mode = _MODULE.sample_size_for_mode
ensure_kaggle_dataset = _MODULE._ensure_kaggle_dataset


def test_sample_size_for_mode_is_deterministic() -> None:
    assert sample_size_for_mode("pr") == 1
    assert sample_size_for_mode("nightly") == 4
    assert sample_size_for_mode("local") == 2


def test_parse_m4_daily_files_applies_horizon_and_context(tmp_path: Path) -> None:
    train_path = tmp_path / "Daily-train.csv"
    test_path = tmp_path / "Daily-test.csv"

    train_path.write_text(
        "V1,V2,V3,V4,V5\nD1,1,2,3,4\nD2,10,11,12,13\nD3,20,21,22,23\n",
        encoding="utf-8",
    )
    test_path.write_text(
        "V1,V2,V3\nD1,5,6\nD2,14,15\nD3,24,25\n",
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
    assert set(row.keys()) == {"id", "freq", "timestamps", "target", "actuals"}
    assert row["freq"] == "H"
    assert row["timestamps"] == sorted(row["timestamps"])
    assert len(row["target"]) == 3
    assert len(row["actuals"]) == 2
    assert len(row["timestamps"]) == len(row["target"])
    assert all(isinstance(value, float) for value in row["target"])
    assert all(isinstance(value, float) for value in row["actuals"])


def test_ensure_kaggle_dataset_uses_cached_extraction(tmp_path: Path, monkeypatch) -> None:
    dataset_cache = tmp_path / "cache"
    extract_dir = dataset_cache / "extracted"
    extract_dir.mkdir(parents=True)
    (extract_dir / "cached.csv").write_text(
        "Datetime,value\n2024-01-01 00:00:00,1\n",
        encoding="utf-8",
    )

    def _fail_download(**_: object) -> None:
        raise AssertionError("download should not be called when extracted cache exists")

    monkeypatch.setattr(_MODULE, "_download_kaggle_archive", _fail_download)

    resolved = ensure_kaggle_dataset(
        dataset_ref="robikscube/hourly-energy-consumption",
        dataset_cache=dataset_cache,
        timeout_seconds=30,
    )

    assert resolved == extract_dir
    assert (resolved / "cached.csv").exists()


def test_ensure_kaggle_dataset_downloads_and_extracts(tmp_path: Path, monkeypatch) -> None:
    dataset_cache = tmp_path / "cache"

    def _fake_download(*, dataset_ref: str, archive_path: Path, timeout_seconds: int) -> None:
        assert dataset_ref == "robikscube/hourly-energy-consumption"
        assert timeout_seconds == 30
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive_path, "w") as archive:
            archive.writestr(
                "AEP_hourly.csv",
                "Datetime,AEP_MW\n2024-01-01 00:00:00,100\n2024-01-01 01:00:00,101\n",
            )

    monkeypatch.setattr(_MODULE, "_download_kaggle_archive", _fake_download)

    resolved = ensure_kaggle_dataset(
        dataset_ref="robikscube/hourly-energy-consumption",
        dataset_cache=dataset_cache,
        timeout_seconds=30,
    )

    assert (resolved / "AEP_hourly.csv").exists()


def test_parse_huggingface_dataset_normalizes_timestamps_and_infers_hourly(
    monkeypatch,
) -> None:
    class _FakeSplit(list):
        @property
        def column_names(self) -> list[str]:
            return ["ts", "value"]

    fake_rows = _FakeSplit(
        [
            {"ts": "2024-01-01 00:00:00", "value": "1"},
            {"ts": "2024/01/01 01:00:00", "value": "2"},
            {"ts": "2024-01-01T02:00:00Z", "value": "3"},
            {"ts": "01/01/2024 03:00:00", "value": "4"},
            {"ts": "2024-01-01 04:00:00", "value": "5"},
        ]
    )

    def _fake_loader(*, hf_id: str, split_name: str | None):  # noqa: ANN202
        assert hf_id == "org/ds"
        assert split_name == "train"
        return {"train": fake_rows}

    monkeypatch.setattr(_MODULE, "_load_hf_dataset", _fake_loader)

    series = parse_huggingface_dataset(
        hf_id="org/ds",
        split_name="train",
        timestamp_column="ts",
        target_column="value",
        horizon=2,
        context_cap=3,
        max_series=1,
        seed=42,
    )

    assert len(series) == 1
    item = series[0]
    assert item["freq"] == "H"
    assert len(item["target"]) == 3
    assert len(item["actuals"]) == 2
    assert all("T" in stamp for stamp in item["timestamps"])
    assert item["timestamps"] == sorted(item["timestamps"])


def test_parse_huggingface_dataset_requires_contiguous_windows(monkeypatch) -> None:
    # With horizon=2 and context_cap=3, required_rows=5. Providing only 3 rows
    # ensures both the strict-contiguity path and the index-fallback path fail.
    class _FakeSplit(list):
        @property
        def column_names(self) -> list[str]:
            return ["ts", "value"]

    fake_rows = _FakeSplit(
        [
            {"ts": "2024-01-01 00:00:00", "value": "1"},
            {"ts": "2024-01-01 04:00:00", "value": "2"},
            {"ts": "2024-01-01 08:00:00", "value": "3"},
        ]
    )

    monkeypatch.setattr(
        _MODULE,
        "_load_hf_dataset",
        lambda *, hf_id, split_name: {"train": fake_rows},
    )

    with pytest.raises(RuntimeError, match="(no contiguous chunks|insufficient parsed rows)"):
        parse_huggingface_dataset(
            hf_id="org/ds",
            split_name="train",
            timestamp_column="ts",
            target_column="value",
            horizon=2,
            context_cap=3,
            max_series=1,
            seed=42,
        )


def test_parse_huggingface_dataset_sampling_is_deterministic(monkeypatch) -> None:
    class _FakeSplit(list):
        @property
        def column_names(self) -> list[str]:
            return ["ts", "value"]

    rows = _FakeSplit(
        [
            {
                "ts": f"2024-01-01 {hour:02d}:00:00",
                "value": str(hour),
            }
            for hour in range(20)
        ]
    )

    monkeypatch.setattr(
        _MODULE,
        "_load_hf_dataset",
        lambda *, hf_id, split_name: {"train": rows},
    )

    first = parse_huggingface_dataset(
        hf_id="org/ds",
        split_name="train",
        timestamp_column="ts",
        target_column="value",
        horizon=2,
        context_cap=3,
        max_series=2,
        seed=42,
    )
    second = parse_huggingface_dataset(
        hf_id="org/ds",
        split_name="train",
        timestamp_column="ts",
        target_column="value",
        horizon=2,
        context_cap=3,
        max_series=2,
        seed=42,
    )

    assert first == second


def test_parse_huggingface_dataset_reads_snapshot_zip_with_explicit_series_ids(
    tmp_path: Path,
    monkeypatch,
) -> None:
    snapshot_path = tmp_path / "app_flow.zip"
    with zipfile.ZipFile(snapshot_path, "w") as archive:
        archive.writestr(
            "app_zone_rpc_hour_encrypted.csv",
            "\n".join(
                [
                    "time,value,app_name",
                    "2024-01-01 00:00:00,1.0,rpc",
                    "2024-01-01 01:00:00,2.0,rpc",
                    "2024-01-01 02:00:00,3.0,rpc",
                    "2024-01-01 03:00:00,4.0,rpc",
                    "2024-01-01 04:00:00,5.0,rpc",
                ]
            ),
        )

    monkeypatch.setattr(
        _MODULE,
        "_resolve_hf_snapshot_file",
        lambda *, hf_id, snapshot_file: snapshot_path,
    )

    series = parse_huggingface_dataset(
        hf_id="kashif/App_Flow",
        split_name=None,
        timestamp_column="time",
        target_column="value",
        horizon=2,
        context_cap=3,
        max_series=1,
        seed=42,
        snapshot_file="app_flow.zip",
        archive_member="app_zone_rpc_hour_encrypted.csv",
        series_id_columns=["app_name"],
    )

    assert len(series) == 1
    item = series[0]
    assert item["id"].startswith("hf:app_flow_rpc_")
    assert item["freq"] == "H"
    assert len(item["target"]) == 3
    assert len(item["actuals"]) == 2


def test_parse_huggingface_dataset_expands_array_targets_from_snapshot_rows(
    tmp_path: Path,
    monkeypatch,
) -> None:
    snapshot_path = tmp_path / "train.parquet"

    monkeypatch.setattr(
        _MODULE,
        "_resolve_hf_snapshot_file",
        lambda *, hf_id, snapshot_file: snapshot_path,
    )
    monkeypatch.setattr(
        _MODULE,
        "_snapshot_column_names",
        lambda path, *, archive_member: ["dt", "hours_sale", "store_id"],
    )
    monkeypatch.setattr(
        _MODULE,
        "_iter_snapshot_rows",
        lambda path, *, archive_member, selected_columns: iter(
            [
                {
                    "dt": "2024-01-01 00:00:00",
                    "hours_sale": [1.0, 2.0, 3.0],
                    "store_id": "store-1",
                },
                {
                    "dt": "2024-01-01 03:00:00",
                    "hours_sale": [4.0, 5.0, 6.0],
                    "store_id": "store-1",
                },
            ]
        ),
    )

    series = parse_huggingface_dataset(
        hf_id="Dingdong-Inc/FreshRetailNet-50K",
        split_name=None,
        timestamp_column="dt",
        target_column="hours_sale",
        horizon=2,
        context_cap=3,
        max_series=1,
        seed=42,
        snapshot_file="data/train.parquet",
        series_id_columns=["store_id"],
        target_array_freq="H",
    )

    assert len(series) == 1
    item = series[0]
    assert item["id"].startswith("hf:freshretailnet-50k_store_1_")
    assert item["freq"] == "H"
    assert item["target"] == [1.0, 2.0, 3.0]
    assert item["actuals"] == [4.0, 5.0]
