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


def kaggle_policy_for_mode(mode: str, credentials_present: bool) -> KagglePolicy:
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

    normalized = text.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(normalized)
    except ValueError:
        pass

    formats = (
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d",
        "%m/%d/%Y %H:%M",
        "%m/%d/%Y %H:%M:%S",
    )
    for fmt in formats:
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


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
