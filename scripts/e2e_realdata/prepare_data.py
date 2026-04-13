"""Dataset fetch and normalization helpers for real-data E2E runs."""

from __future__ import annotations

import base64
import csv
import io
import os
import random
import re
import shutil
import urllib.error
import urllib.request
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from statistics import median
from typing import Any

import yaml

SAMPLE_SIZE_BY_MODE: dict[str, int] = {
    "pr": 1,
    "nightly": 4,
    "local": 2,
}


@dataclass(frozen=True)
class KagglePolicy:
    """Resolved dataset policy for Kaggle credentials vs run mode."""

    include_kaggle: bool
    hard_fail_on_missing: bool
    message: str | None = None


@dataclass(frozen=True)
class PreparedDataResult:
    """Prepared datasets plus non-fatal informational messages."""

    datasets: dict[str, list[dict[str, Any]]]
    messages: list[str]


def sample_size_for_mode(mode: str) -> int:
    """Return the deterministic sample size for the selected run mode."""
    return resolve_sample_size(mode)


def resolve_sample_size(mode: str, *, max_series_per_dataset: int | None = None) -> int:
    """Return the deterministic per-dataset sample size for one run."""
    if max_series_per_dataset is not None:
        if max_series_per_dataset <= 0:
            raise ValueError("max_series_per_dataset must be positive")
        return int(max_series_per_dataset)

    normalized = mode.strip().lower()
    if normalized not in SAMPLE_SIZE_BY_MODE:
        raise ValueError(f"unsupported mode: {mode!r}")
    return SAMPLE_SIZE_BY_MODE[normalized]


def has_kaggle_credentials(env: Mapping[str, str] | None = None) -> bool:
    """Return whether Kaggle API credentials are available."""
    source = os.environ if env is None else env
    username = source.get("KAGGLE_USERNAME", "").strip()
    key = source.get("KAGGLE_KEY", "").strip()
    return bool(username and key)


def kaggle_policy_for_mode(
    mode: str,
    credentials_present: bool,
    *,
    allow_local_fallback: bool = False,
) -> KagglePolicy:
    """Resolve Kaggle policy based on mode and credential availability."""
    normalized = mode.strip().lower()
    if normalized not in SAMPLE_SIZE_BY_MODE:
        raise ValueError(f"unsupported mode: {mode!r}")

    if credentials_present:
        return KagglePolicy(include_kaggle=True, hard_fail_on_missing=False, message=None)

    if normalized == "nightly":
        return KagglePolicy(
            include_kaggle=False,
            hard_fail_on_missing=True,
            message="nightly mode requires KAGGLE_USERNAME and KAGGLE_KEY",
        )

    if normalized == "local":
        if allow_local_fallback:
            return KagglePolicy(
                include_kaggle=False,
                hard_fail_on_missing=False,
                message="Kaggle credentials missing; local fallback enabled (open datasets only)",
            )
        return KagglePolicy(
            include_kaggle=False,
            hard_fail_on_missing=True,
            message=(
                "local mode requires KAGGLE_USERNAME and KAGGLE_KEY; "
                "use --allow-kaggle-fallback to run open datasets only"
            ),
        )

    return KagglePolicy(
        include_kaggle=False,
        hard_fail_on_missing=False,
        message="Kaggle credentials missing; falling back to open datasets only",
    )


def load_dataset_catalog(
    path: Path,
    *,
    require_unique_hf_ids: bool = False,
) -> dict[str, Any]:
    """Load and minimally validate the YAML dataset catalog."""
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("dataset catalog must be a mapping")

    datasets = payload.get("datasets")
    if not isinstance(datasets, list) or not datasets:
        raise ValueError("dataset catalog must contain a non-empty datasets list")

    seen_names: set[str] = set()
    seen_hf_ids: set[str] = set()

    for item in datasets:
        if not isinstance(item, dict):
            raise ValueError("each dataset catalog entry must be a mapping")
        if not isinstance(item.get("name"), str) or not item["name"].strip():
            raise ValueError("dataset catalog entry missing name")
        if not isinstance(item.get("kind"), str) or not item["kind"].strip():
            raise ValueError("dataset catalog entry missing kind")

        name = str(item["name"]).strip()
        if name in seen_names:
            raise ValueError(f"duplicate dataset name: {name}")
        seen_names.add(name)

        kind = str(item["kind"]).strip()
        horizon = item.get("horizon")
        if not isinstance(horizon, int) or horizon <= 0:
            raise ValueError(f"dataset {name!r} has invalid horizon")

        if kind == "huggingface_dataset":
            for field in ("hf_id", "timestamp_column", "target_column"):
                value = item.get(field)
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(f"dataset {name!r} missing {field}")
            hf_id = str(item["hf_id"]).strip()
            if require_unique_hf_ids:
                if hf_id in seen_hf_ids:
                    raise ValueError(f"duplicate hf_id: {hf_id}")
                seen_hf_ids.add(hf_id)
        elif kind == "open_m4_daily":
            for field in ("train_url", "test_url"):
                value = item.get(field)
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(f"dataset {name!r} missing {field}")
        elif kind == "kaggle_hourly_energy":
            value = item.get("kaggle_dataset")
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"dataset {name!r} missing kaggle_dataset")
    return payload


def prepare_datasets(
    *,
    mode: str,
    catalog_path: Path,
    cache_dir: Path,
    include_kaggle: bool,
    require_kaggle: bool,
    seed: int = 42,
    context_cap: int = 512,
    timeout_seconds: int = 120,
    max_series_per_dataset: int | None = None,
) -> PreparedDataResult:
    """Fetch and normalize configured datasets for the selected mode."""
    catalog = load_dataset_catalog(catalog_path)
    sample_size = resolve_sample_size(
        mode,
        max_series_per_dataset=max_series_per_dataset,
    )

    datasets: dict[str, list[dict[str, Any]]] = {}
    messages: list[str] = []
    cache_dir.mkdir(parents=True, exist_ok=True)

    for item in catalog["datasets"]:
        name = str(item["name"])
        kind = str(item["kind"])
        horizon = int(item.get("horizon", 0))

        if horizon <= 0:
            raise ValueError(f"dataset {name!r} has invalid horizon")

        if kind == "open_m4_daily":
            train_url = str(item["train_url"])
            test_url = str(item["test_url"])
            dataset_cache = cache_dir / name
            dataset_cache.mkdir(parents=True, exist_ok=True)
            train_path = dataset_cache / "Daily-train.csv"
            test_path = dataset_cache / "Daily-test.csv"
            _download_file(train_url, train_path, timeout_seconds=timeout_seconds)
            _download_file(test_url, test_path, timeout_seconds=timeout_seconds)
            datasets[name] = parse_m4_daily_files(
                train_path=train_path,
                test_path=test_path,
                horizon=horizon,
                context_cap=context_cap,
                max_series=sample_size,
                seed=seed,
            )
            continue

        if kind == "huggingface_dataset":
            hf_id = str(item["hf_id"])
            timestamp_column = str(item["timestamp_column"])
            target_column = str(item["target_column"])
            split_name_value = item.get("split")
            split_name = str(split_name_value) if isinstance(split_name_value, str) else None
            snapshot_file_value = item.get("snapshot_file")
            snapshot_file = (
                str(snapshot_file_value).strip()
                if isinstance(snapshot_file_value, str) and snapshot_file_value.strip()
                else None
            )
            archive_member_value = item.get("archive_member")
            archive_member = (
                str(archive_member_value).strip()
                if isinstance(archive_member_value, str) and archive_member_value.strip()
                else None
            )
            target_array_freq_value = item.get("target_array_freq")
            target_array_freq = (
                str(target_array_freq_value).strip()
                if isinstance(target_array_freq_value, str) and target_array_freq_value.strip()
                else None
            )
            raw_series_id_columns = item.get("series_id_columns")
            series_id_columns = None
            if isinstance(raw_series_id_columns, list):
                normalized_columns = [
                    str(column).strip()
                    for column in raw_series_id_columns
                    if isinstance(column, str) and str(column).strip()
                ]
                if normalized_columns:
                    series_id_columns = normalized_columns

            try:
                datasets[name] = parse_huggingface_dataset(
                    hf_id=hf_id,
                    split_name=split_name,
                    timestamp_column=timestamp_column,
                    target_column=target_column,
                    horizon=horizon,
                    context_cap=context_cap,
                    max_series=sample_size,
                    seed=seed,
                    snapshot_file=snapshot_file,
                    archive_member=archive_member,
                    series_id_columns=series_id_columns,
                    target_array_freq=target_array_freq,
                )
            except Exception as exc:
                reason = f"skipped dataset {name}: failed to load or parse from HuggingFace ({exc})"
                messages.append(reason)
            continue

        if kind == "kaggle_hourly_energy":
            if not include_kaggle:
                reason = f"skipped dataset {name}: kaggle credentials unavailable"
                messages.append(reason)
                if require_kaggle:
                    raise RuntimeError(reason)
                continue

            dataset_cache = cache_dir / name
            extracted_dir = _ensure_kaggle_dataset(
                dataset_ref=str(item["kaggle_dataset"]),
                dataset_cache=dataset_cache,
                timeout_seconds=timeout_seconds,
            )
            datasets[name] = parse_kaggle_hourly_directory(
                data_dir=extracted_dir,
                horizon=horizon,
                context_cap=context_cap,
                max_series=sample_size,
                seed=seed,
            )
            continue

        raise ValueError(f"unsupported dataset kind: {kind!r}")

    if not datasets:
        raise RuntimeError("no datasets were prepared")

    return PreparedDataResult(datasets=datasets, messages=messages)


def _hf_row_scan_limit(*, required_rows: int, max_series: int, explicit_series_ids: bool) -> int:
    baseline = max(required_rows * max_series * 20, 20_000)
    if not explicit_series_ids:
        return baseline
    # Explicit panel keys often interleave many series in timestamp order, so
    # the first contiguous window may not appear until far deeper in the file.
    return max(baseline, required_rows * max_series * 1200, 300_000)


def parse_huggingface_dataset(
    *,
    hf_id: str,
    split_name: str | None,
    timestamp_column: str,
    target_column: str,
    horizon: int,
    context_cap: int,
    max_series: int,
    seed: int,
    snapshot_file: str | None = None,
    archive_member: str | None = None,
    series_id_columns: list[str] | None = None,
    target_array_freq: str | None = None,
) -> list[dict[str, Any]]:
    """Parse a HuggingFace dataset into normalized request series."""
    if snapshot_file is not None:
        snapshot_path = _resolve_hf_snapshot_file(hf_id=hf_id, snapshot_file=snapshot_file)
        available_cols = _snapshot_column_names(
            snapshot_path,
            archive_member=archive_member,
        )
        rows = _iter_snapshot_rows(
            snapshot_path,
            archive_member=archive_member,
            selected_columns=_selected_snapshot_columns(
                timestamp_column=timestamp_column,
                target_column=target_column,
                series_id_columns=series_id_columns,
            ),
        )
        return _parse_hf_rows(
            rows=rows,
            available_cols=available_cols,
            hf_id=hf_id,
            resolved_split=f"snapshot:{snapshot_file}",
            timestamp_column=timestamp_column,
            target_column=target_column,
            horizon=horizon,
            context_cap=context_cap,
            max_series=max_series,
            seed=seed,
            series_id_columns=series_id_columns,
            target_array_freq=target_array_freq,
        )

    try:
        import datasets as hf_datasets
    except ModuleNotFoundError:
        hf_datasets = None
    else:
        # Suppress progress bars when the datasets package is available.
        hf_datasets.utils.logging.disable_progress_bar()

    loaded_dataset = _load_hf_dataset(hf_id=hf_id, split_name=split_name)
    resolved_split = split_name
    if hasattr(loaded_dataset, "keys"):
        split_names = [str(name) for name in loaded_dataset.keys()]
        if not split_names:
            raise RuntimeError(f"dataset {hf_id} has no splits")
        chosen_split = resolved_split or sorted(split_names)[0]
        if chosen_split not in split_names:
            raise RuntimeError(
                f"split {chosen_split!r} not found for {hf_id}; available={sorted(split_names)}"
            )
        split_data = loaded_dataset[chosen_split]
        resolved_split = chosen_split
    else:
        split_data = loaded_dataset
        resolved_split = resolved_split or "default"

    available_cols = list(getattr(split_data, "column_names", []))
    return _parse_hf_rows(
        rows=split_data,
        available_cols=available_cols,
        hf_id=hf_id,
        resolved_split=resolved_split,
        timestamp_column=timestamp_column,
        target_column=target_column,
        horizon=horizon,
        context_cap=context_cap,
        max_series=max_series,
        seed=seed,
        series_id_columns=series_id_columns,
        target_array_freq=target_array_freq,
    )


def _parse_hf_rows(
    *,
    rows: Any,
    available_cols: list[str],
    hf_id: str,
    resolved_split: str,
    timestamp_column: str,
    target_column: str,
    horizon: int,
    context_cap: int,
    max_series: int,
    seed: int,
    series_id_columns: list[str] | None = None,
    target_array_freq: str | None = None,
) -> list[dict[str, Any]]:
    if available_cols and (
        timestamp_column not in available_cols or target_column not in available_cols
    ):
        raise RuntimeError(
            f"columns {timestamp_column!r}/{target_column!r} not in {available_cols}"
        )
    if series_id_columns:
        missing_series_columns = [
            column
            for column in series_id_columns
            if available_cols and column not in available_cols
        ]
        if missing_series_columns:
            raise RuntimeError(
                f"series id columns {missing_series_columns!r} not in {available_cols}"
            )

    required_rows = context_cap + horizon
    candidates: list[dict[str, Any]] = []
    parsed_rows: list[tuple[datetime, float, dict[str, str | None]]] = []
    resolved_series_id_columns = list(series_id_columns or [])
    series_id_column: str | None = None
    series_id_candidates = (
        resolved_series_id_columns
        if resolved_series_id_columns
        else _hf_series_id_candidates(
            available_cols,
            timestamp_column=timestamp_column,
            target_column=target_column,
        )
    )
    series_id_value_counts: dict[str, dict[str, int]] = (
        {} if resolved_series_id_columns else {column: {} for column in series_id_candidates}
    )
    chunk_index = 0
    # Scan enough rows to produce stable deterministic candidates while staying bounded.
    row_limit = _hf_row_scan_limit(
        required_rows=required_rows,
        max_series=max_series,
        explicit_series_ids=bool(series_id_columns),
    )
    array_step = _step_for_freq(target_array_freq) if target_array_freq is not None else None

    for row_idx, item in enumerate(rows):
        if row_idx >= row_limit:
            break
        try:
            if not isinstance(item, Mapping):
                continue

            ts_raw = item.get(timestamp_column)
            if ts_raw is None:
                continue

            parsed_dt = _parse_timestamp_value(ts_raw)
            if parsed_dt is None:
                continue

            series_values: dict[str, str | None] = {}
            for column in series_id_candidates:
                raw_series_value = item.get(column)
                if raw_series_value is None:
                    series_values[column] = None
                    continue
                normalized_series_value = str(raw_series_value).strip()
                if not normalized_series_value:
                    series_values[column] = None
                    continue
                series_values[column] = normalized_series_value
                if not resolved_series_id_columns:
                    counts = series_id_value_counts[column]
                    counts[normalized_series_value] = counts.get(normalized_series_value, 0) + 1

            target_points = _extract_target_points(
                raw=item.get(target_column),
                parsed_dt=parsed_dt,
                array_step=array_step,
            )
            for target_dt, parsed_value in target_points:
                parsed_rows.append((target_dt, parsed_value, dict(series_values)))
        except Exception:
            continue

    if len(parsed_rows) < required_rows:
        raise RuntimeError(
            f"insufficient parsed rows for {hf_id}: {len(parsed_rows)} < {required_rows}"
        )

    grouped_rows: dict[str, list[tuple[datetime, float]]] = {}
    if resolved_series_id_columns:
        for parsed_dt, parsed_value, series_values in parsed_rows:
            series_key = _compose_series_key(
                series_values=series_values,
                series_id_columns=resolved_series_id_columns,
            )
            if series_key is None:
                continue
            grouped_rows.setdefault(series_key, []).append((parsed_dt, parsed_value))
    else:
        series_id_column = _select_hf_series_id_column(
            series_id_value_counts,
            required_rows=required_rows,
        )
        for parsed_dt, parsed_value, series_values in parsed_rows:
            series_key = "__single__"
            if series_id_column is not None:
                candidate_series_key = series_values.get(series_id_column)
                if candidate_series_key is None:
                    continue
                series_key = candidate_series_key
            grouped_rows.setdefault(series_key, []).append((parsed_dt, parsed_value))

    for series_key in sorted(grouped_rows):
        series_rows = grouped_rows[series_key]
        if len(series_rows) < required_rows:
            continue

        series_rows.sort(key=lambda item: item[0])
        series_rows = _deduplicate_timestamps(series_rows)
        inferred_freq = _infer_frequency_from_datetimes([point[0] for point in series_rows])
        expected_step = _step_for_freq(inferred_freq)
        contiguous_segments = _split_contiguous_rows(series_rows, expected_step=expected_step)

        for segment in contiguous_segments:
            if len(segment) < required_rows:
                continue
            for start in range(0, len(segment) - required_rows + 1, required_rows):
                window = segment[start : start + required_rows]
                history = window[:-horizon]
                future = window[-horizon:]

                series_suffix = (
                    f"_{_sanitize_series_component(series_key)}"
                    if resolved_series_id_columns or series_id_column is not None
                    else ""
                )
                candidates.append(
                    {
                        "id": f"hf:{hf_id.split('/')[-1].lower()}{series_suffix}_{chunk_index}",
                        "freq": inferred_freq,
                        "timestamps": [point[0].isoformat() for point in history],
                        "target": [point[1] for point in history],
                        "actuals": [point[1] for point in future],
                        "split": resolved_split,
                    }
                )
                chunk_index += 1
                if len(candidates) >= max_series * 5:
                    break
            if len(candidates) >= max_series * 5:
                break
        if len(candidates) >= max_series * 5:
            break

    if not candidates:
        # Optional HF datasets are often sparse/irregular panel data. If strict contiguity
        # yields no windows, fall back to deterministic index windows from parsed rows.
        for series_key in sorted(grouped_rows):
            series_rows = grouped_rows[series_key]
            if len(series_rows) < required_rows:
                continue

            series_rows.sort(key=lambda item: item[0])
            series_rows = _deduplicate_timestamps(series_rows)
            inferred_freq = _infer_frequency_from_datetimes([point[0] for point in series_rows])
            for start in range(0, len(series_rows) - required_rows + 1, required_rows):
                window = series_rows[start : start + required_rows]
                history = window[:-horizon]
                future = window[-horizon:]

                series_suffix = (
                    f"_{_sanitize_series_component(series_key)}"
                    if resolved_series_id_columns or series_id_column is not None
                    else ""
                )
                candidates.append(
                    {
                        "id": f"hf:{hf_id.split('/')[-1].lower()}{series_suffix}_{chunk_index}",
                        "freq": inferred_freq,
                        "timestamps": [point[0].isoformat() for point in history],
                        "target": [point[1] for point in history],
                        "actuals": [point[1] for point in future],
                        "split": resolved_split,
                    }
                )
                chunk_index += 1
                if len(candidates) >= max_series * 5:
                    break
            if len(candidates) >= max_series * 5:
                break

    if not candidates:
        raise RuntimeError(
            f"no contiguous chunks (size {required_rows}) found in {hf_id} split={resolved_split}"
        )

    return _sample_series(candidates, max_series=max_series, seed=seed)


def _selected_snapshot_columns(
    *,
    timestamp_column: str,
    target_column: str,
    series_id_columns: list[str] | None,
) -> list[str]:
    ordered = [timestamp_column, target_column, *(series_id_columns or [])]
    unique: list[str] = []
    seen: set[str] = set()
    for column in ordered:
        if column not in seen:
            seen.add(column)
            unique.append(column)
    return unique


def _resolve_hf_snapshot_file(*, hf_id: str, snapshot_file: str) -> Path:
    repo_dir = (
        Path.home() / ".cache" / "huggingface" / "hub" / f"datasets--{hf_id.replace('/', '--')}"
    )
    snapshots_dir = repo_dir / "snapshots"
    if not snapshots_dir.is_dir():
        raise RuntimeError(f"no cached snapshots found for {hf_id}")

    main_ref = repo_dir / "refs" / "main"
    if main_ref.is_file():
        revision = main_ref.read_text(encoding="utf-8").strip()
        preferred = snapshots_dir / revision / snapshot_file
        if preferred.exists():
            return preferred

    snapshot_dirs = sorted(path for path in snapshots_dir.iterdir() if path.is_dir())
    for snapshot_dir in reversed(snapshot_dirs):
        candidate = snapshot_dir / snapshot_file
        if candidate.exists():
            return candidate
    raise RuntimeError(f"snapshot file {snapshot_file!r} not found for {hf_id}")


def _snapshot_column_names(snapshot_path: Path, *, archive_member: str | None) -> list[str]:
    suffix = snapshot_path.suffix.lower()
    if archive_member is not None or suffix == ".zip":
        member_name = _resolve_zip_member(snapshot_path, archive_member=archive_member)
        with zipfile.ZipFile(snapshot_path) as archive:
            with archive.open(member_name) as handle:
                text_handle = io.TextIOWrapper(handle, encoding="utf-8", errors="ignore")
                reader = csv.DictReader(text_handle)
                return list(reader.fieldnames or [])

    if suffix == ".csv":
        with snapshot_path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
            reader = csv.DictReader(handle)
            return list(reader.fieldnames or [])

    if suffix == ".parquet":
        import pyarrow.parquet as pq

        return list(pq.ParquetFile(snapshot_path).schema_arrow.names)

    raise RuntimeError(f"unsupported snapshot file format: {snapshot_path.suffix}")


def _iter_snapshot_rows(
    snapshot_path: Path,
    *,
    archive_member: str | None,
    selected_columns: list[str] | None,
):
    suffix = snapshot_path.suffix.lower()
    if archive_member is not None or suffix == ".zip":
        member_name = _resolve_zip_member(snapshot_path, archive_member=archive_member)
        with zipfile.ZipFile(snapshot_path) as archive:
            with archive.open(member_name) as handle:
                text_handle = io.TextIOWrapper(handle, encoding="utf-8", errors="ignore")
                reader = csv.DictReader(text_handle)
                for row in reader:
                    if selected_columns is None:
                        yield row
                    else:
                        yield {column: row.get(column) for column in selected_columns}
        return

    if suffix == ".csv":
        with snapshot_path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if selected_columns is None:
                    yield row
                else:
                    yield {column: row.get(column) for column in selected_columns}
        return

    if suffix == ".parquet":
        import pyarrow.parquet as pq

        parquet_file = pq.ParquetFile(snapshot_path)
        columns = selected_columns or None
        for batch in parquet_file.iter_batches(columns=columns, batch_size=50_000):
            payload = batch.to_pydict()
            column_names = list(payload.keys())
            if not column_names:
                continue
            row_count = len(payload[column_names[0]])
            for index in range(row_count):
                yield {column: payload[column][index] for column in column_names}
        return

    raise RuntimeError(f"unsupported snapshot file format: {snapshot_path.suffix}")


def _resolve_zip_member(snapshot_path: Path, *, archive_member: str | None) -> str:
    with zipfile.ZipFile(snapshot_path) as archive:
        if archive_member is not None:
            if archive_member not in archive.namelist():
                raise RuntimeError(
                    f"archive member {archive_member!r} not found in {snapshot_path.name!r}"
                )
            return archive_member

        csv_members = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if len(csv_members) == 1:
            return csv_members[0]
        raise RuntimeError(
            f"archive member required for {snapshot_path.name!r}; available={csv_members[:20]}"
        )


def _parse_timestamp_value(raw: Any) -> datetime | None:
    if raw is None:
        return None
    if isinstance(raw, datetime):
        return raw.replace(tzinfo=None) if raw.tzinfo else raw
    return _parse_timestamp(str(raw))


def _extract_target_points(
    *,
    raw: Any,
    parsed_dt: datetime,
    array_step: timedelta | None,
) -> list[tuple[datetime, float]]:
    if array_step is None:
        parsed_value = _coerce_float_value(raw)
        if parsed_value is None:
            return []
        return [(parsed_dt, parsed_value)]

    parsed_values = _parse_float_sequence(raw)
    return [(parsed_dt + (array_step * index), value) for index, value in enumerate(parsed_values)]


def _coerce_float_value(raw: Any) -> float | None:
    if raw is None or isinstance(raw, bool):
        return None
    if isinstance(raw, (int, float)):
        try:
            return float(raw)
        except (TypeError, ValueError):
            return None
    return _parse_float(str(raw))


def _parse_float_sequence(raw: Any) -> list[float]:
    if raw is None or isinstance(raw, (str, bytes, bytearray)):
        return []
    if hasattr(raw, "tolist"):
        try:
            raw = raw.tolist()
        except Exception:
            return []
    if not isinstance(raw, (list, tuple)):
        return []

    parsed: list[float] = []
    for item in raw:
        value = _coerce_float_value(item)
        if value is not None:
            parsed.append(value)
    return parsed


def _compose_series_key(
    *,
    series_values: Mapping[str, str | None],
    series_id_columns: list[str],
) -> str | None:
    parts: list[str] = []
    for column in series_id_columns:
        value = series_values.get(column)
        if value is None:
            return None
        parts.append(value)
    return "|".join(parts)


def _hf_series_id_candidates(
    columns: list[str],
    *,
    timestamp_column: str,
    target_column: str,
) -> list[str]:
    hints = (
        "series_id",
        "unique_id",
        "item_id",
        "entity_id",
        "store_id",
        "sku",
        "symbol",
        "ticker",
        "station",
        "sensor",
        "node",
        "store",
        "entity",
        "id",
    )
    excluded = {timestamp_column, target_column}
    ranked: list[tuple[int, str]] = []
    for column in columns:
        if column in excluded:
            continue
        normalized = column.strip().lower().replace("-", "_").replace(" ", "_")
        if not normalized:
            continue
        rank = 10_000
        for idx, hint in enumerate(hints):
            if normalized == hint:
                rank = idx
                break
            if hint in normalized:
                rank = min(rank, len(hints) + idx)
        if rank < 10_000:
            ranked.append((rank, column))

    ranked.sort(key=lambda item: (item[0], item[1]))
    return [column for _, column in ranked]


def _select_hf_series_id_column(
    value_counts: Mapping[str, Mapping[str, int]],
    *,
    required_rows: int,
) -> str | None:
    candidates: list[tuple[str, int, int]] = []
    for column, counts in value_counts.items():
        distinct_count = len(counts)
        if distinct_count <= 1:
            continue
        max_group_rows = max(counts.values(), default=0)
        if max_group_rows < required_rows:
            continue
        candidates.append((column, distinct_count, max_group_rows))

    if not candidates:
        return None

    candidates.sort(key=lambda item: (-item[2], item[1], item[0]))
    return candidates[0][0]


def _sanitize_series_component(raw: str) -> str:
    text = raw.strip().lower()
    if not text:
        return "series"
    cleaned = "".join(char if char.isalnum() else "_" for char in text)
    trimmed = cleaned.strip("_")
    if not trimmed:
        return "series"
    return trimmed[:48]


def parse_m4_daily_files(
    *,
    train_path: Path,
    test_path: Path,
    horizon: int,
    context_cap: int,
    max_series: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Parse M4 daily train/test files into normalized request series."""
    train_map = _parse_m4_matrix(train_path)
    test_map = _parse_m4_matrix(test_path)
    ids = sorted(set(train_map).intersection(test_map))

    candidates: list[dict[str, Any]] = []
    for series_id in ids:
        history = train_map[series_id]
        future = test_map[series_id][:horizon]
        if len(history) < 2 or len(future) < horizon:
            continue

        trimmed_history = history[-context_cap:]
        timestamps = _synthetic_timestamps(freq="D", count=len(trimmed_history))
        candidates.append(
            {
                "id": f"m4_daily:{series_id}",
                "freq": "D",
                "timestamps": timestamps,
                "target": trimmed_history,
                "actuals": future,
            }
        )

    if not candidates:
        raise RuntimeError("no valid series found in M4 daily files")

    return _sample_series(candidates, max_series=max_series, seed=seed)


def parse_kaggle_hourly_directory(
    *,
    data_dir: Path,
    horizon: int,
    context_cap: int,
    max_series: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Parse Kaggle hourly-energy CSV files into normalized request series."""
    csv_files = sorted(data_dir.rglob("*.csv"))
    if not csv_files:
        raise RuntimeError(f"no csv files found under {data_dir}")

    candidates: list[dict[str, Any]] = []
    for csv_path in csv_files:
        rows = _parse_hourly_csv(csv_path)
        if len(rows) <= horizon:
            continue

        window = rows[-(context_cap + horizon) :]
        history = window[:-horizon]
        future = window[-horizon:]
        if len(history) < 2 or len(future) < horizon:
            continue

        candidates.append(
            {
                "id": f"pjm_hourly:{csv_path.stem.lower()}",
                "freq": "H",
                "timestamps": [point[0] for point in history],
                "target": [point[1] for point in history],
                "actuals": [point[1] for point in future],
            }
        )

    if not candidates:
        raise RuntimeError("no valid series found in Kaggle hourly-energy files")

    return _sample_series(candidates, max_series=max_series, seed=seed)


def _sample_series(
    series: list[dict[str, Any]],
    *,
    max_series: int,
    seed: int,
) -> list[dict[str, Any]]:
    ordered = sorted(series, key=lambda item: str(item["id"]))
    if max_series <= 0:
        raise ValueError("max_series must be > 0")
    if len(ordered) <= max_series:
        return ordered

    rng = random.Random(seed)
    sampled = rng.sample(ordered, k=max_series)
    return sorted(sampled, key=lambda item: str(item["id"]))


def _parse_m4_matrix(path: Path) -> dict[str, list[float]]:
    parsed: dict[str, list[float]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        _ = next(reader, None)
        for row in reader:
            if not row:
                continue
            series_id = row[0].strip()
            if not series_id:
                continue
            values = [_parse_float(cell) for cell in row[1:]]
            cleaned = [value for value in values if value is not None]
            if cleaned:
                parsed[series_id] = cleaned
    return parsed


def _parse_hourly_csv(path: Path) -> list[tuple[str, float]]:
    rows: list[tuple[str, float]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            return rows

        timestamp_column = _choose_timestamp_column(reader.fieldnames)
        value_column = _choose_value_column(reader.fieldnames, timestamp_column)
        if timestamp_column is None or value_column is None:
            return rows

        for raw in reader:
            timestamp_text = str(raw.get(timestamp_column, "")).strip()
            value_text = str(raw.get(value_column, "")).strip()
            parsed_value = _parse_float(value_text)
            if not timestamp_text or parsed_value is None:
                continue

            parsed_dt = _parse_timestamp(timestamp_text)
            if parsed_dt is None:
                continue
            rows.append((parsed_dt.isoformat(), parsed_value))

    rows.sort(key=lambda item: item[0])
    return rows


def _choose_timestamp_column(columns: list[str]) -> str | None:
    preferred = {"datetime", "timestamp", "date", "time"}
    for name in columns:
        if name.strip().lower() in preferred:
            return name
    return columns[0] if columns else None


def _choose_value_column(columns: list[str], timestamp_column: str | None) -> str | None:
    for name in columns:
        if name == timestamp_column:
            continue
        return name
    return None


def _parse_float(raw: str) -> float | None:
    text = raw.strip().replace(",", "")
    if not text or text.lower() in {"nan", "na", "null", "none"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _parse_timestamp(raw: str) -> datetime | None:
    text = raw.strip()
    if not text:
        return None

    if text.isdigit():
        try:
            epoch = int(text)
            if len(text) >= 13:
                epoch = epoch // 1000
            return datetime.fromtimestamp(epoch)
        except (OverflowError, OSError, ValueError):
            pass

    normalized = text.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
        return parsed.replace(tzinfo=None) if parsed.tzinfo else parsed
    except ValueError:
        pass

    formats = (
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%Y/%m/%d %H:%M",
        "%Y/%m/%d %H:%M:%S",
        "%d-%m-%Y",
        "%d/%m/%Y",
        "%m/%d/%Y %H:%M",
        "%m/%d/%Y %H:%M:%S",
        "%m/%d/%Y",
    )
    for fmt in formats:
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def _load_hf_dataset(*, hf_id: str, split_name: str | None) -> Any:
    from datasets import load_dataset

    if split_name:
        return load_dataset(hf_id, split=split_name)
    return load_dataset(hf_id)


def _infer_frequency_from_datetimes(points: list[datetime]) -> str:
    if len(points) < 2:
        return "D"

    deltas = sorted(
        (points[index] - points[index - 1]).total_seconds()
        for index in range(1, len(points))
        if points[index] > points[index - 1]
    )
    if not deltas:
        return "D"

    median_seconds = float(median(deltas))
    canonical_freqs = (
        (1.0, "S"),
        (60.0, "MIN"),
        (30.0 * 60.0, "30MIN"),
        (60.0 * 60.0, "H"),
        (24.0 * 60.0 * 60.0, "D"),
        (7.0 * 24.0 * 60.0 * 60.0, "W"),
        (30.0 * 24.0 * 60.0 * 60.0, "M"),
    )
    for seconds, freq in canonical_freqs:
        tolerance = max(1.0, seconds * 0.25)
        if abs(median_seconds - seconds) <= tolerance:
            return freq

    if median_seconds < 30.0 * 60.0:
        return "MIN"
    if median_seconds < 12.0 * 60.0 * 60.0:
        return "H"
    if median_seconds < 5.0 * 24.0 * 60.0 * 60.0:
        return "D"
    if median_seconds < 14.0 * 24.0 * 60.0 * 60.0:
        return "W"
    return "M"


def _step_for_freq(freq: str) -> timedelta:
    match = re.fullmatch(r"\s*(\d+)?\s*([A-Za-z]+)\s*", freq or "")
    if match is None:
        return timedelta(days=1)

    multiplier = int(match.group(1) or "1")
    token = match.group(2).strip().upper()
    if token.startswith("S"):
        return timedelta(seconds=multiplier)
    if token in {"T", "MIN", "MINS", "MINUTE", "MINUTES"}:
        return timedelta(minutes=multiplier)
    if token.startswith("H"):
        return timedelta(hours=multiplier)
    if token.startswith("W"):
        return timedelta(weeks=multiplier)
    if token in {"M", "MS", "ME", "MONTH", "MONTHS"}:
        return timedelta(days=30 * multiplier)
    return timedelta(days=multiplier)


def _deduplicate_timestamps(
    rows: list[tuple[datetime, float]],
) -> list[tuple[datetime, float]]:
    """Remove duplicate timestamps, keeping the first occurrence.

    Some HuggingFace datasets contain rows with identical timestamps. Keeping the
    first occurrence after a stable sort preserves temporal order and prevents
    non-increasing timestamp rejections downstream.
    """
    seen: set[datetime] = set()
    deduped: list[tuple[datetime, float]] = []
    for point in rows:
        ts = point[0]
        if ts not in seen:
            seen.add(ts)
            deduped.append(point)
    return deduped


def _split_contiguous_rows(
    rows: list[tuple[datetime, float]],
    *,
    expected_step: timedelta,
) -> list[list[tuple[datetime, float]]]:
    if not rows:
        return []

    expected_seconds = expected_step.total_seconds()
    tolerance = max(expected_seconds * 0.5, 1.0)
    segments: list[list[tuple[datetime, float]]] = []
    current: list[tuple[datetime, float]] = [rows[0]]

    for item in rows[1:]:
        prev_dt = current[-1][0]
        delta_seconds = (item[0] - prev_dt).total_seconds()
        if abs(delta_seconds - expected_seconds) <= tolerance:
            current.append(item)
            continue

        segments.append(current)
        current = [item]

    if current:
        segments.append(current)
    return segments


def _synthetic_timestamps(*, freq: str, count: int) -> list[str]:
    if count <= 0:
        return []
    base = datetime(2020, 1, 1)
    if freq == "H":
        step = timedelta(hours=1)
        return [(base + step * index).isoformat() for index in range(count)]
    step = timedelta(days=1)
    return [(base + step * index).date().isoformat() for index in range(count)]


def _download_file(url: str, destination: Path, *, timeout_seconds: int) -> None:
    if destination.exists() and destination.stat().st_size > 0:
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url)
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            payload = response.read()
    except urllib.error.URLError as exc:
        raise RuntimeError(f"failed to download {url}: {exc}") from exc

    destination.write_bytes(payload)


def _ensure_kaggle_dataset(
    *,
    dataset_ref: str,
    dataset_cache: Path,
    timeout_seconds: int,
) -> Path:
    archive_path = dataset_cache / "dataset.zip"
    extract_dir = dataset_cache / "extracted"

    existing_files = list(extract_dir.rglob("*.csv"))
    if existing_files:
        return extract_dir

    dataset_cache.mkdir(parents=True, exist_ok=True)
    _download_kaggle_archive(
        dataset_ref=dataset_ref,
        archive_path=archive_path,
        timeout_seconds=timeout_seconds,
    )

    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(archive_path, "r") as archive:
        archive.extractall(extract_dir)

    return extract_dir


def _download_kaggle_archive(
    *,
    dataset_ref: str,
    archive_path: Path,
    timeout_seconds: int,
) -> None:
    if archive_path.exists() and archive_path.stat().st_size > 0:
        return

    username = os.environ.get("KAGGLE_USERNAME", "").strip()
    key = os.environ.get("KAGGLE_KEY", "").strip()
    if not username or not key:
        raise RuntimeError("kaggle credentials are required to download dataset")

    token = base64.b64encode(f"{username}:{key}".encode()).decode("ascii")
    url = f"https://www.kaggle.com/api/v1/datasets/download/{dataset_ref}"
    request = urllib.request.Request(url)
    request.add_header("Authorization", f"Basic {token}")

    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            payload = response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"kaggle download failed ({exc.code}): {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"kaggle download failed: {exc}") from exc

    archive_path.write_bytes(payload)
