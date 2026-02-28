"""Dataset fetch and normalization helpers for real-data E2E runs."""

from __future__ import annotations

import base64
import csv
import os
import random
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


def load_dataset_catalog(path: Path) -> dict[str, Any]:
    """Load and minimally validate the YAML dataset catalog."""
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("dataset catalog must be a mapping")

    datasets = payload.get("datasets")
    if not isinstance(datasets, list) or not datasets:
        raise ValueError("dataset catalog must contain a non-empty datasets list")

    for item in datasets:
        if not isinstance(item, dict):
            raise ValueError("each dataset catalog entry must be a mapping")
        if not isinstance(item.get("name"), str) or not item["name"].strip():
            raise ValueError("dataset catalog entry missing name")
        if not isinstance(item.get("kind"), str) or not item["kind"].strip():
            raise ValueError("dataset catalog entry missing kind")
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
) -> PreparedDataResult:
    """Fetch and normalize configured datasets for the selected mode."""
    catalog = load_dataset_catalog(catalog_path)
    sample_size = sample_size_for_mode(mode)

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
) -> list[dict[str, Any]]:
    """Parse a HuggingFace dataset into normalized request series."""
    import datasets as hf_datasets

    # Suppress progress bars
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
    if available_cols and (
        timestamp_column not in available_cols or target_column not in available_cols
    ):
        raise RuntimeError(
            f"columns {timestamp_column!r}/{target_column!r} not in {available_cols}"
        )

    required_rows = context_cap + horizon
    candidates: list[dict[str, Any]] = []
    parsed_rows: list[tuple[datetime, float, dict[str, str | None]]] = []
    series_id_candidates = _hf_series_id_candidates(
        available_cols,
        timestamp_column=timestamp_column,
        target_column=target_column,
    )
    series_id_value_counts: dict[str, dict[str, int]] = {
        column: {} for column in series_id_candidates
    }
    chunk_index = 0
    # Scan enough rows to produce stable deterministic candidates while staying bounded.
    row_limit = max(required_rows * max_series * 20, 20_000)

    for row_idx, item in enumerate(split_data):
        if row_idx >= row_limit:
            break
        try:
            if not isinstance(item, Mapping):
                continue

            ts_raw = item.get(timestamp_column)
            target_raw = item.get(target_column)
            if ts_raw is None or target_raw is None:
                continue

            parsed_dt = _parse_timestamp(str(ts_raw))
            parsed_value = _parse_float(str(target_raw))
            if parsed_dt is None or parsed_value is None:
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
                counts = series_id_value_counts[column]
                counts[normalized_series_value] = counts.get(normalized_series_value, 0) + 1

            parsed_rows.append((parsed_dt, parsed_value, series_values))
        except Exception:
            continue

    if len(parsed_rows) < required_rows:
        raise RuntimeError(
            f"insufficient parsed rows for {hf_id}: "
            f"{len(parsed_rows)} < {required_rows}"
        )

    series_id_column = _select_hf_series_id_column(
        series_id_value_counts,
        required_rows=required_rows,
    )

    grouped_rows: dict[str, list[tuple[datetime, float]]] = {}
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
                    if series_id_column is not None
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
                    if series_id_column is not None
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
    text = raw.strip()
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
    if 1800.0 <= median_seconds <= 5400.0:
        return "H"
    if 64800.0 <= median_seconds <= 108000.0:
        return "D"
    return "D"


def _step_for_freq(freq: str) -> timedelta:
    return timedelta(hours=1) if freq.strip().upper().startswith("H") else timedelta(days=1)


def _deduplicate_timestamps(
    rows: list[tuple[datetime, float]],
) -> list[tuple[datetime, float]]:
    """Remove duplicate timestamps, keeping the first occurrence.

    Some HuggingFace datasets contain rows with identical timestamps (e.g.
    FreshRetailNet panel data where multiple records share the same time).
    Duplicate timestamps cause the daemon to reject the series with
    "timestamps must be strictly increasing".  Keeping the first occurrence
    after a stable sort preserves temporal order and eliminates the
    duplicates.
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
