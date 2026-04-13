#!/usr/bin/env python3
"""Collect random Hugging Face time-series datasets via public REST API only.

This collector avoids direct dependency on ``huggingface_hub`` / ``datasets`` by using
Hugging Face HTTP endpoints and direct file downloads.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import random
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, quote, urlparse
from urllib.request import Request, urlopen

try:
    from pyarrow import ipc as pa_ipc
    from pyarrow import parquet as pa_parquet

    HAS_PYARROW = True
except Exception:  # pragma: no cover
    pa_ipc = None  # type: ignore[assignment]
    pa_parquet = None  # type: ignore[assignment]
    HAS_PYARROW = False

MAX_ROWS = 1_000_000
MAX_STREAM_ATTEMPTS = 3
DEFAULT_CONTEXT_CAP = 512
DEFAULT_HORIZON = 24
DEFAULT_SAMPLE_ROWS = 600
DEFAULT_MIN_RATIO = 0.90
MAX_RAW_BYTES = 1_000_000
MAX_TEXT_SOURCE_BYTES = 20 * 1024 * 1024
HEAD_TIMEOUT_SECONDS = 8
MAX_SOURCE_FILES_PER_DATASET = 20
HTTP_TIMEOUT_SECONDS = 45
DEFAULT_HTTP_PAGE_SIZE = 250
DEFAULT_LIST_ATTEMPTS = 20_000
MAX_PARQUET_ARROW_BYTES = 120 * 1024 * 1024


KNOWN_BAD_DATASETS = {
    "Numerati/numerai-datasets",
    "MLNTeam-Unical/NFT-70M_transactions",
    "MLNTeam-Unical/NFT-70M_text",
    "MLNTeam-Unical/NFT-70M_image",
    "rajistics/electricity_demand",
    "davidwisdom/fake_railroad_company",
    "raeidsaqur/nifty-rl",
    "louisbrulenaudet/obis",
    "hereldav/TimeAware",
    "Maple728/Time-300B",
    "saluslab/HM-SYNC",
    "mirai-ml/wids2023_competition_weather_forecasting",
    "traqq/weekly_activity_data_jul24_jul28",
    "ChengsenWang/TSQA",
    "mschi/blogspot_raw",
    "ChengsenWang/ChatTime-1-Pretrain-1M",
    "ChengsenWang/ChatTime-1-Finetune-100K",
}

TIMESTAMP_NAME_HINTS = (
    "timestamp",
    "datetime",
    "date_time",
    "dateutc",
    "date",
    "time",
    "dt",
    "period",
    "start",
    "ds",
    "tstmp",
)

TEXT_EXTENSIONS = {"json", "jsonl", "csv", "tsv"}
BINARY_EXTENSIONS = {"parquet", "arrow"}

NON_PAYLOAD_HINTS = (
    "assets/",
    "asset/",
    "/asset/",
    "/assets/",
    "README",
    "readme",
    "license",
    "dataset_info",
    "dataset_infos",
    "dataset_info.json",
    "dataset_infos.json",
    "README.md",
    "README.txt",
    "configs/",
    "config.",
    "pyproject.",
    "requirements.",
    "limits.json",
)

TARGET_NAME_HINTS = (
    "target",
    "value",
    "actual",
    "actuals",
    "y",
    "close",
    "price",
    "demand",
    "sales",
    "sale_amount",
    "label",
)

ID_COLUMN_HINTS = (
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

INDUSTRY_HINTS = {
    "finance": (
        "stock",
        "forex",
        "crypto",
        "trading",
        "exchange",
        "market",
        "price",
        "index",
        "nasdaq",
        "dow",
        "spx",
        "futures",
    ),
    "energy": (
        "energy",
        "electric",
        "electricity",
        "power",
        "solar",
        "wind",
        "grid",
        "load",
        "gas",
    ),
    "retail": (
        "sales",
        "retail",
        "ecommerce",
        "customer",
        "order",
        "store",
        "demand",
        "purchase",
        "shopping",
    ),
    "weather": (
        "weather",
        "temperature",
        "precip",
        "humidity",
        "rain",
        "climate",
        "windspeed",
        "meteo",
        "climate",
    ),
    "mobility": (
        "traffic",
        "trip",
        "vehicle",
        "flight",
        "transport",
        "route",
        "gps",
        "fleet",
        "uber",
    ),
    "iot": (
        "iot",
        "sensor",
        "telemetry",
        "device",
        "machine",
        "pump",
        "vibration",
        "pressure",
        "voltage",
    ),
}


HF_LIST_URL = "https://huggingface.co/api/datasets"
HF_DATASET_URL = "https://huggingface.co/api/datasets/{hf_id}"
HF_FILE_URL = "https://huggingface.co/datasets/{hf_id}/resolve/main/{path}"
USER_AGENT = "tollama-hf-rest-collector/1.0"


@dataclass(frozen=True)
class Candidate:
    hf_id: str
    split: str
    source_path: str
    source_config: str | None
    timestamp_column: str
    target_column: str
    interval_bucket: str
    interval_seconds: float | None
    shape_bucket: str
    has_multi_id: bool
    industry_hint: str
    tags: list[str]
    desc_snippet: str
    columns: list[str]


@dataclass(frozen=True)
class SourceFile:
    path: str
    split: str
    config_name: str | None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect random HF time-series datasets via HTTP API."
    )
    parser.add_argument("--out-dir", default="hf_data", help="Output directory (default: hf_data)")
    parser.add_argument("--count", type=int, default=300, help="Number of datasets to collect")
    parser.add_argument(
        "--page-size", type=int, default=DEFAULT_HTTP_PAGE_SIZE, help="HF list page size"
    )
    parser.add_argument(
        "--attempt-limit",
        type=int,
        default=DEFAULT_LIST_ATTEMPTS,
        help="Hard cap on candidate attempts",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--max-rows", type=int, default=MAX_ROWS, help="Per-dataset row cap (1,000,000)"
    )
    parser.add_argument(
        "--sample-rows",
        type=int,
        default=DEFAULT_SAMPLE_ROWS,
        help="Sample rows per split for filtering",
    )
    parser.add_argument("--context-cap", type=int, default=DEFAULT_CONTEXT_CAP)
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    parser.add_argument("--min-timestamp-ratio", type=float, default=DEFAULT_MIN_RATIO)
    parser.add_argument("--min-target-ratio", type=float, default=DEFAULT_MIN_RATIO)
    parser.add_argument("--min-contiguous", type=int, default=None)
    parser.add_argument(
        "--min-pool-multiplier", type=float, default=2.0, help="Candidate pool target multiplier"
    )
    parser.add_argument("--force", action="store_true", help="Overwrite non-empty output directory")
    parser.add_argument(
        "--allow-binary",
        action="store_true",
        help="Allow parquet/arrow sources (default: text formats only)",
    )
    parser.add_argument(
        "--raw-max-bytes",
        type=int,
        default=MAX_RAW_BYTES,
        help="Per-dataset raw file size cap in bytes (default: 1,000,000)",
    )
    parser.add_argument(
        "--source-max-bytes",
        type=int,
        default=MAX_TEXT_SOURCE_BYTES,
        help="Max source file size for sampling (default: 20,000,000)",
    )
    parser.add_argument(
        "--filter",
        default="task_categories:time-series-forecasting",
        help="Hugging Face dataset filter expression",
    )
    parser.add_argument(
        "--source-url",
        default=None,
        help="Optional HF datasets URL to parse task_categories filter from",
    )
    return parser.parse_args(argv)


def resolve_filter_query(filter_value: str, source_url: str | None) -> str:
    if not source_url:
        return filter_value

    query = source_url.split("?")[1] if "?" in source_url else ""
    if not query:
        return filter_value

    import urllib.parse

    params = urllib.parse.parse_qs(query)
    task_categories = params.get("task_categories")
    if not task_categories:
        return filter_value

    candidate = task_categories[0].strip()
    if not candidate:
        return filter_value
    if candidate.startswith("task_categories:"):
        return candidate
    return f"task_categories:{candidate}"


def parse_datetime(raw: Any) -> datetime | None:
    if isinstance(raw, datetime):
        return raw
    if raw is None:
        return None

    text = str(raw).strip()
    if not text:
        return None
    if text.isdigit():
        try:
            epoch = int(text)
            if len(text) >= 13:
                epoch = epoch // 1000
            return datetime.fromtimestamp(epoch, tz=UTC).replace(tzinfo=None)
        except (OverflowError, OSError, ValueError):
            pass

    normalized = text.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
        return parsed.replace(tzinfo=None) if parsed.tzinfo else parsed
    except ValueError:
        pass

    for fmt in (
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d %H:%M:%S",
        "%Y/%m/%d %H:%M",
        "%Y/%m/%d %H:%M:%S",
        "%d-%m-%Y",
        "%d/%m/%Y",
        "%m/%d/%Y",
        "%m/%d/%Y %H:%M",
        "%m/%d/%Y %H:%M:%S",
    ):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue

    return None


def parse_float(raw: Any) -> float | None:
    if raw is None:
        return None
    if isinstance(raw, bool):
        return None
    if isinstance(raw, (int, float)):
        return float(raw)
    text = str(raw).strip()
    if not text or text.lower() in {"nan", "na", "null", "none"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def normalize_name(column: str) -> str:
    return column.strip().lower().replace(" ", "_")


def name_priority(column: str, hints: tuple[str, ...]) -> int:
    normalized = normalize_name(column)
    for idx, hint in enumerate(hints):
        if normalized == hint:
            return idx
    for idx, hint in enumerate(hints):
        if hint in normalized:
            return len(hints) + idx
    return 10_000


def infer_schema_from_rows(rows: list[dict[str, Any]]) -> tuple[str | None, str | None]:
    if not rows:
        return None, None

    columns = sorted({key for row in rows for key in row.keys() if isinstance(key, str) and key})
    if not columns:
        return None, None

    ts_scores: list[tuple[str, float, int]] = []
    target_scores: list[tuple[str, float, int]] = []
    total_rows = len(rows)

    for column in columns:
        ts_ok = 0
        target_ok = 0
        for row in rows:
            if parse_datetime(row.get(column)) is not None:
                ts_ok += 1
            if parse_float(row.get(column)) is not None:
                target_ok += 1

        ts_scores.append((column, ts_ok / total_rows, name_priority(column, TIMESTAMP_NAME_HINTS)))
        target_scores.append(
            (column, target_ok / total_rows, name_priority(column, TARGET_NAME_HINTS))
        )

    ts_scores.sort(key=lambda item: (-item[1], item[2], item[0]))
    target_scores.sort(key=lambda item: (-item[1], item[2], item[0]))

    timestamp_column = ts_scores[0][0] if ts_scores else None
    target_column = None
    for candidate, _, _ in target_scores:
        if candidate != timestamp_column:
            target_column = candidate
            break
    return timestamp_column, target_column


def assess_sample_rows(
    *,
    rows: list[dict[str, Any]],
    timestamp_column: str,
    target_column: str,
    min_timestamp_ratio: float,
    min_target_ratio: float,
    min_contiguous_rows: int,
) -> tuple[bool, str]:
    total = len(rows)
    if total <= 0:
        return False, "empty_sample"

    timestamp_ok = 0
    target_ok = 0
    max_run = 0
    current = 0

    for row in rows:
        has_timestamp = parse_datetime(row.get(timestamp_column)) is not None
        has_target = parse_float(row.get(target_column)) is not None

        if has_timestamp:
            timestamp_ok += 1
        if has_target:
            target_ok += 1

        if has_timestamp and has_target:
            current += 1
            max_run = max(max_run, current)
        else:
            current = 0

    timestamp_ratio = timestamp_ok / total
    target_ratio = target_ok / total

    if timestamp_ratio < min_timestamp_ratio:
        return False, f"low_timestamp_ratio={timestamp_ratio:.3f} < {min_timestamp_ratio:.3f}"
    if target_ratio < min_target_ratio:
        return False, f"low_target_ratio={target_ratio:.3f} < {min_target_ratio:.3f}"
    if max_run < min_contiguous_rows:
        return False, f"insufficient_contiguous_rows={max_run} < {min_contiguous_rows}"
    return True, ""


def infer_interval_bucket(dts: list[datetime]) -> tuple[str, float | None]:
    if len(dts) < 2:
        return "unknown", None

    points = sorted(dts)
    deltas = [
        (points[i].timestamp() - points[i - 1].timestamp())
        for i in range(1, len(points))
        if points[i] > points[i - 1]
    ]
    if not deltas:
        return "unknown", None

    median_sec = float(sorted(deltas)[len(deltas) // 2])
    if median_sec <= 60:
        return "minute_or_less", median_sec
    if median_sec <= 3600:
        return "hourly", median_sec
    if median_sec <= 86400:
        return "daily", median_sec
    if median_sec <= 7 * 86400:
        return "weekly", median_sec
    if median_sec <= 31 * 86400:
        return "monthly", median_sec
    if median_sec <= 365 * 86400:
        return "yearly", median_sec
    return "irregular", median_sec


def infer_shape_and_ids(
    rows: list[dict[str, Any]],
    columns: list[str],
    timestamp_column: str,
    target_column: str,
) -> tuple[str, bool]:
    id_candidates = ranked_id_columns(columns, timestamp_column, target_column)
    has_multi_id = False
    for column in id_candidates:
        values = {
            str(row[column])
            for row in rows
            if isinstance(row, dict) and row.get(column) not in (None, "")
        }
        if len(values) > 1:
            has_multi_id = True
            break

    numeric_columns = 0
    feature_columns = [col for col in columns if col not in {timestamp_column, target_column}]
    for col in feature_columns:
        parsed = 0
        for row in rows:
            if parse_float(row.get(col)) is not None:
                parsed += 1
        if parsed >= max(1, len(rows) // 2):
            numeric_columns += 1

    if has_multi_id:
        shape = "multi_series"
    elif numeric_columns >= 1:
        shape = "multivariate"
    elif len(feature_columns) >= 1:
        shape = "single_series_with_extras"
    else:
        shape = "univariate"

    return shape, has_multi_id


def infer_industry(hf_id: str, tags: list[str], description: str) -> str:
    joined = " ".join([hf_id, *tags, description]).lower()
    for industry, keywords in INDUSTRY_HINTS.items():
        if any(keyword in joined for keyword in keywords):
            return industry
    return "general"


def safe_path_name(hf_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", hf_id.replace("/", "__"))


def ranked_id_columns(
    columns: list[str],
    timestamp_column: str,
    target_column: str,
) -> list[str]:
    excluded = {timestamp_column, target_column}
    ranked = []

    for column in columns:
        if column in excluded:
            continue
        normalized = normalize_name(column)
        if not normalized:
            continue
        rank = 10_000
        for idx, hint in enumerate(ID_COLUMN_HINTS):
            if normalized == hint:
                rank = idx
                break
            if hint in normalized:
                rank = min(rank, len(ID_COLUMN_HINTS) + idx)
        if rank < 10_000:
            ranked.append((rank, column))

    ranked.sort(key=lambda item: (item[0], item[1]))
    return [column for _, column in ranked]


def serialize_json_line(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def get_json(url: str) -> Any:
    try:
        req = Request(url, headers={"User-Agent": USER_AGENT})
        with urlopen(req, timeout=HTTP_TIMEOUT_SECONDS) as response:
            if response.status >= 400:
                raise RuntimeError(f"request failed status={response.status}: {url}")
            payload = json.loads(response.read().decode("utf-8"))
        return payload
    except Exception as exc:
        return _get_json_via_curl(url, exc)


def get_json_with_headers(url: str) -> tuple[Any, dict[str, str]]:
    try:
        req = Request(url, headers={"User-Agent": USER_AGENT})
        with urlopen(req, timeout=HTTP_TIMEOUT_SECONDS) as response:
            if response.status >= 400:
                raise RuntimeError(f"request failed status={response.status}: {url}")
            payload = json.loads(response.read().decode("utf-8"))
            headers = {key: value for key, value in response.headers.items()}
            return payload, headers
    except Exception as exc:
        payload = _get_json_via_curl(url, exc)
        return payload, {}


def parse_next_cursor(headers: dict[str, str]) -> str | None:
    link = headers.get("Link")
    if not link:
        return None

    for part in link.split(","):
        if 'rel="next"' not in part:
            continue
        start = part.find("<")
        end = part.find(">", start + 1)
        if start == -1 or end == -1:
            continue
        href = part[start + 1 : end]
        query = parse_qs(urlparse(href).query)
        cursor_values = query.get("cursor")
        if cursor_values:
            return cursor_values[0]
    return None


def _run_curl(
    args: list[str],
    *,
    max_output: int | None = None,
    timeout_seconds: int = HTTP_TIMEOUT_SECONDS,
) -> subprocess.CompletedProcess[bytes]:
    completed = subprocess.run(
        [
            "curl",
            "-sS",
            "--max-time",
            str(HTTP_TIMEOUT_SECONDS),
            "-H",
            f"User-Agent: {USER_AGENT}",
        ]
        + args,
        check=False,
        capture_output=True,
        timeout=timeout_seconds + 10,
    )
    if completed.returncode != 0:
        stderr_text = completed.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"curl failed status={completed.returncode}: {stderr_text}")
    if max_output is not None and len(completed.stdout) > max_output:
        raise RuntimeError(f"response exceeded output cap: {len(completed.stdout)} > {max_output}")
    return completed


def _run_curl_text(url: str) -> bytes:
    completed = _run_curl(["--compressed", "-L", url])
    return completed.stdout


def _run_curl_to_file(output_file: Path, url: str, *, max_bytes: int | None = None) -> None:
    cmd = [
        "curl",
        "-sS",
        "--max-time",
        str(HTTP_TIMEOUT_SECONDS),
        "--compressed",
        "-L",
        "-o",
        str(output_file),
        "-H",
        f"User-Agent: {USER_AGENT}",
        url,
    ]
    if max_bytes is not None:
        cmd.extend(["--max-filesize", str(max_bytes)])
    completed = subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        timeout=HTTP_TIMEOUT_SECONDS + 20,
    )
    if completed.returncode != 0:
        stderr_text = completed.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"curl download failed status={completed.returncode}: {stderr_text}")


def _get_content_length_via_curl(url: str) -> int | None:
    headers = (
        _run_curl(["-I", "--head", "-L", url], timeout_seconds=HEAD_TIMEOUT_SECONDS)
        .stdout.decode("utf-8", errors="replace")
        .splitlines()
    )
    for line in headers:
        lower = line.lower()
        if lower.startswith("content-length:"):
            try:
                return int(line.split(":", 1)[1].strip())
            except ValueError:
                return None
    return None


def _get_json_via_curl(url: str, original_error: Exception) -> Any:
    completed = _run_curl([url])
    try:
        return json.loads(completed.stdout.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"json fetch failed via curl for {url}: {original_error}") from exc


def get_content_length(url: str) -> int | None:
    try:
        req = Request(url, method="HEAD", headers={"User-Agent": USER_AGENT})
        with urlopen(req, timeout=HEAD_TIMEOUT_SECONDS) as response:
            if response.status >= 400:
                return None
            length = response.getheader("Content-Length")
            if length is None:
                return None
            try:
                return int(length)
            except ValueError:
                return None
    except Exception:
        return _get_content_length_via_curl(url)


def urlsafe_hf_id(hf_id: str) -> str:
    return quote(hf_id, safe="/")


def normalize_ext(path: str) -> str:
    text = path.lower()
    if text.endswith(".gz"):
        text = text[:-3]
    return text.rsplit(".", maxsplit=1)[-1] if "." in text else ""


def is_supported_data_file(path: str) -> bool:
    ext = normalize_ext(path)
    return ext in TEXT_EXTENSIONS | BINARY_EXTENSIONS


def is_supported_data_file_with_options(path: str, *, allow_binary: bool) -> bool:
    ext = normalize_ext(path)
    if ext in TEXT_EXTENSIONS:
        return True
    return bool(allow_binary and ext in BINARY_EXTENSIONS)


def is_likely_payload_path(path: str) -> bool:
    lower_path = path.lower()
    if any(hint in lower_path for hint in NON_PAYLOAD_HINTS):
        return False
    return True


def dataset_listing_url(
    filter_query: str,
    page_size: int,
    cursor: str | None,
) -> str:
    query = f"{HF_LIST_URL}?filter={quote(filter_query, safe=':')}&limit={page_size}&full=true"
    if cursor is not None:
        query = f"{query}&cursor={quote(cursor, safe='')}"
    return query


def dataset_info_url(hf_id: str) -> str:
    return HF_DATASET_URL.format(hf_id=urlsafe_hf_id(hf_id))


def file_url(hf_id: str, path: str) -> str:
    return HF_FILE_URL.format(hf_id=urlsafe_hf_id(hf_id), path=quote(path, safe="/"))


def hash_path(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:12]


def iter_dataset_ids(
    filter_query: str,
    max_ids: int,
    page_size: int,
    seed: int,
) -> list[str]:
    ids: list[str] = []
    seen = set[str]()
    cursor: str | None = None
    page_count = 0

    while len(ids) < max_ids:
        payload, headers = get_json_with_headers(
            dataset_listing_url(filter_query, page_size, cursor=cursor)
        )
        if not isinstance(payload, list) or not payload:
            break
        new_found = 0
        for item in payload:
            hf_id = item.get("id") if isinstance(item, dict) else None
            if not isinstance(hf_id, str) or not hf_id.strip():
                continue
            if hf_id in seen:
                continue
            seen.add(hf_id)
            ids.append(hf_id)
            new_found += 1

        if new_found == 0:
            break
        page_count += 1
        cursor = parse_next_cursor(headers)
        if not cursor:
            break

        if page_count >= (max_ids // page_size + 1):
            break

    rng = random.Random(seed)
    rng.shuffle(ids)
    return ids


def parse_dataset_info(
    hf_id: str, allow_binary: bool
) -> tuple[list[str], str, bool, bool, set[str], list[SourceFile]]:
    payload = get_json(dataset_info_url(hf_id))
    if not isinstance(payload, dict):
        raise ValueError("dataset info payload is not object")

    tags = [str(tag) for tag in (payload.get("tags") or []) if isinstance(tag, str)]
    desc = str(payload.get("description", "") or "")
    if not desc:
        card_data = payload.get("cardData")
        if isinstance(card_data, dict):
            desc = str(card_data.get("pretty_print", "") or card_data.get("description", "") or "")

    source_files: list[SourceFile] = []
    card_data = payload.get("cardData")
    if isinstance(card_data, dict):
        configs = card_data.get("configs")
        if isinstance(configs, list):
            for cfg in configs:
                if not isinstance(cfg, dict):
                    continue
                cfg_name = cfg.get("config_name")
                config_name = (
                    str(cfg_name) if isinstance(cfg_name, str) and cfg_name.strip() else None
                )
                data_files = cfg.get("data_files")
                if isinstance(data_files, list):
                    for item in data_files:
                        if not isinstance(item, dict):
                            continue
                        path = item.get("path")
                        if len(source_files) >= MAX_SOURCE_FILES_PER_DATASET:
                            break
                        split = item.get("split", "train")
                        if not isinstance(path, str) or not path.strip():
                            continue
                        if not is_likely_payload_path(path):
                            continue
                        if not is_supported_data_file_with_options(path, allow_binary=allow_binary):
                            continue
                        if len(source_files) >= MAX_SOURCE_FILES_PER_DATASET:
                            break
                        source_files.append(
                            SourceFile(path=path, split=str(split), config_name=config_name)
                        )

    if not source_files:
        for sibling in payload.get("siblings", []):
            if not isinstance(sibling, dict):
                continue
            path = sibling.get("rfilename")
            if not isinstance(path, str) or not path.strip():
                continue
            if not is_likely_payload_path(path):
                continue
            if not is_supported_data_file_with_options(path, allow_binary=allow_binary):
                continue
            if len(source_files) >= MAX_SOURCE_FILES_PER_DATASET:
                break
            source_files.append(SourceFile(path=path, split="train", config_name=None))

    unique_sources: list[SourceFile] = []
    seen_paths: set[str] = set()
    for source in source_files:
        key = f"{source.path}|{source.split}|{source.config_name or ''}"
        if key in seen_paths:
            continue
        if source.path.endswith(".py"):
            continue
        seen_paths.add(key)
        unique_sources.append(source)

    return (
        tags,
        desc[:500],
        bool(payload.get("gated")),
        bool(payload.get("private")),
        set(
            str(s.get("rfilename", "")) for s in payload.get("siblings", []) if isinstance(s, dict)
        ),
        unique_sources,
    )


def ensure_text_reader(response, *, gzip_compressed: bool) -> io.TextIOWrapper:
    if gzip_compressed:
        return io.TextIOWrapper(gzip.GzipFile(fileobj=response), encoding="utf-8", errors="replace")
    return io.TextIOWrapper(response, encoding="utf-8", errors="replace")


def iter_rows_from_text_file(
    *,
    response,
    ext: str,
    sample_rows: int,
    split_limit: int,
    is_gzip: bool = False,
) -> list[dict[str, Any]]:
    # ``split_limit`` exists only for signature consistency with binary loaders.
    _ = split_limit
    if ext == "jsonl":
        rows: list[dict[str, Any]] = []
        reader = ensure_text_reader(response, gzip_compressed=is_gzip)
        for line in reader:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                value = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                rows.append(value)
            if len(rows) >= sample_rows:
                break
        return rows

    if ext in {"csv", "tsv"}:
        delimiter = "," if ext == "csv" else "\t"
        reader = ensure_text_reader(response, gzip_compressed=is_gzip)
        csv_reader = csv.DictReader(reader, delimiter=delimiter)
        rows: list[dict[str, Any]] = []
        for row in csv_reader:
            if isinstance(row, dict):
                rows.append({k: v for k, v in row.items() if k is not None})
            if len(rows) >= sample_rows:
                break
        return rows

    if ext == "json":
        reader = ensure_text_reader(response, gzip_compressed=is_gzip)
        payload = json.load(reader)
        rows = []
        if isinstance(payload, list):
            for value in payload:
                if isinstance(value, dict):
                    rows.append(value)
                if len(rows) >= sample_rows:
                    break
        elif isinstance(payload, dict):
            candidate = payload.get("data")
            if isinstance(candidate, list):
                for value in candidate:
                    if isinstance(value, dict):
                        rows.append(value)
                    if len(rows) >= sample_rows:
                        break
        return rows

    raise ValueError(f"unsupported text extension: {ext}")


def iter_rows_from_binary_file(path: Path, ext: str, sample_rows: int) -> list[dict[str, Any]]:
    if not HAS_PYARROW or pa_parquet is None or pa_ipc is None:
        raise RuntimeError("pyarrow is required for parquet/arrow files")

    if ext == "parquet":
        reader = pa_parquet.ParquetFile(str(path))
        rows: list[dict[str, Any]] = []
        for rg in range(reader.num_row_groups):
            table = reader.read_row_group(rg)
            for value in table.to_pylist():
                if isinstance(value, dict):
                    rows.append(value)
                if len(rows) >= sample_rows:
                    return rows
        return rows

    if ext == "arrow":
        with open(path, "rb") as fp:
            try:
                file_reader = pa_ipc.RecordBatchFileReader(fp)
            except Exception:
                fp.seek(0)
                file_reader = pa_ipc.RecordBatchStreamReader(fp)
            rows: list[dict[str, Any]] = []
            for batch in file_reader:
                for value in batch.to_pylist():
                    if isinstance(value, dict):
                        rows.append(value)
                    if len(rows) >= sample_rows:
                        return rows
            return rows

    raise ValueError(f"unsupported binary extension: {ext}")


def stream_rows_from_file(
    hf_id: str,
    source: SourceFile,
    sample_rows: int,
    *,
    max_bytes: int | None = None,
    source_max_bytes: int | None = None,
) -> list[dict[str, Any]]:
    path = source.path
    base_ext = normalize_ext(path)
    url = file_url(hf_id, path)

    request = Request(url, headers={"User-Agent": USER_AGENT})
    is_gzip = path.endswith(".gz")

    if source_max_bytes is not None:
        content_length = get_content_length(url)
        if content_length is None:
            raise RuntimeError(f"source file size unavailable: {url}")
        if content_length is not None and content_length > source_max_bytes:
            raise RuntimeError(f"source file too large: {content_length} > {source_max_bytes}")

    if base_ext in {"jsonl", "json", "csv", "tsv"}:
        try:
            with urlopen(request, timeout=HTTP_TIMEOUT_SECONDS) as response:
                if response.status >= 400:
                    raise RuntimeError(f"download failed {response.status}: {url}")
                return iter_rows_from_text_file(
                    response=response,
                    ext=base_ext,
                    sample_rows=sample_rows,
                    split_limit=sample_rows,
                    is_gzip=is_gzip,
                )
        except Exception:
            try:
                fallback_bytes = _run_curl_text(url)
            except Exception as exc2:
                raise RuntimeError(
                    f"download failed and curl fallback failed {url}: {exc2}"
                ) from exc2
            with io.BytesIO(fallback_bytes) as response:
                return iter_rows_from_text_file(
                    response=response,
                    ext=base_ext,
                    sample_rows=sample_rows,
                    split_limit=sample_rows,
                    is_gzip=is_gzip,
                )

    if base_ext not in {"parquet", "arrow"}:
        raise ValueError(f"unsupported data extension: {path}")

    with tempfile.NamedTemporaryFile(suffix=f".{base_ext}", delete=False) as tmp:
        bytes_read = 0
        try:
            with urlopen(request, timeout=HTTP_TIMEOUT_SECONDS) as response:
                if response.status >= 400:
                    raise RuntimeError(f"download failed {response.status}: {url}")
                content_length = response.getheader("Content-Length")
                if content_length is not None:
                    try:
                        estimated_size = int(content_length)
                        if max_bytes is not None and estimated_size > max_bytes:
                            raise RuntimeError(
                                "binary file size too large for preview "
                                f"{estimated_size} > {max_bytes}"
                            )
                    except ValueError:
                        pass
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    bytes_read += len(chunk)
                    if max_bytes is not None and bytes_read > max_bytes:
                        raise RuntimeError(
                            f"binary file too large for safe read {bytes_read} > {max_bytes}"
                        )
                    tmp.write(chunk)
        except Exception:
            tmp.seek(0)
            tmp.truncate()
            _run_curl_to_file(Path(tmp.name), url, max_bytes=max_bytes)
        tmp.flush()
        local_path = Path(tmp.name)
    try:
        return iter_rows_from_binary_file(local_path, base_ext, sample_rows)
    finally:
        local_path.unlink(missing_ok=True)


def sample_candidate_rows(
    hf_id: str, source: SourceFile, sample_rows: int, *, source_max_bytes: int | None = None
) -> list[dict[str, Any]]:
    return stream_rows_from_file(
        hf_id=hf_id,
        source=source,
        sample_rows=sample_rows,
        max_bytes=MAX_PARQUET_ARROW_BYTES,
        source_max_bytes=source_max_bytes,
    )


def evaluate_dataset(
    hf_id: str,
    sample_rows: int,
    min_timestamp_ratio: float,
    min_target_ratio: float,
    min_contiguous_rows: int,
    allow_binary: bool,
    source_max_bytes: int,
) -> tuple[Candidate | None, list[dict[str, Any]]]:
    rejections: list[dict[str, Any]] = []

    if hf_id in KNOWN_BAD_DATASETS:
        return None, [
            {
                "hf_id": hf_id,
                "split": "-",
                "reason": "blocklisted",
                "detail": "in KNOWN_BAD_DATASETS",
            }
        ]

    try:
        tags, desc, gated, private, sibling_files, source_files = parse_dataset_info(
            hf_id, allow_binary
        )
    except Exception as exc:
        return None, [
            {"hf_id": hf_id, "split": "-", "reason": "metadata_error", "detail": str(exc)}
        ]

    if gated or private:
        return None, [
            {
                "hf_id": hf_id,
                "split": "-",
                "reason": "restricted_dataset",
                "detail": "gated or private",
            }
        ]

    if any(file_path.endswith(".py") for file_path in sibling_files):
        return (
            None,
            [
                {
                    "hf_id": hf_id,
                    "split": "-",
                    "reason": "scripted_dataset",
                    "detail": "contains .py file in repository",
                }
            ],
        )

    if not source_files:
        return None, [
            {
                "hf_id": hf_id,
                "split": "-",
                "reason": "no_supported_files",
                "detail": "no supported source files found",
            }
        ]

    best_candidate: Candidate | None = None
    for source in source_files:
        try:
            rows = sample_candidate_rows(
                hf_id, source, sample_rows=sample_rows, source_max_bytes=source_max_bytes
            )
        except Exception as exc:
            rejections.append(
                {
                    "hf_id": hf_id,
                    "split": source.split,
                    "reason": "sample_load_error",
                    "detail": str(exc),
                }
            )
            continue

        if not rows:
            rejections.append(
                {
                    "hf_id": hf_id,
                    "split": source.split,
                    "reason": "empty_sample",
                    "detail": "no rows returned",
                }
            )
            continue

        ts_col, target_col = infer_schema_from_rows(rows)
        if not ts_col or not target_col:
            rejections.append(
                {
                    "hf_id": hf_id,
                    "split": source.split,
                    "reason": "schema_inference_failed",
                    "detail": "timestamp/target not inferred",
                }
            )
            continue

        ok, detail = assess_sample_rows(
            rows=rows,
            timestamp_column=ts_col,
            target_column=target_col,
            min_timestamp_ratio=min_timestamp_ratio,
            min_target_ratio=min_target_ratio,
            min_contiguous_rows=min_contiguous_rows,
        )
        if not ok:
            rejections.append(
                {
                    "hf_id": hf_id,
                    "split": source.split,
                    "reason": "schema_quality_failed",
                    "detail": detail,
                }
            )
            continue

        columns = sorted(
            {key for row in rows for key in row.keys() if isinstance(key, str) and key}
        )
        parsed_dts = [parse_datetime(row.get(ts_col)) for row in rows]
        dts = [dt for dt in parsed_dts if dt is not None]
        interval_bucket, interval_seconds = infer_interval_bucket(dts)
        shape, has_multi_id = infer_shape_and_ids(rows, columns, ts_col, target_col)

        candidate = Candidate(
            hf_id=hf_id,
            split=source.split,
            source_path=source.path,
            source_config=source.config_name,
            timestamp_column=ts_col,
            target_column=target_col,
            interval_bucket=interval_bucket,
            interval_seconds=interval_seconds,
            shape_bucket=shape,
            has_multi_id=has_multi_id,
            industry_hint=infer_industry(hf_id, tags, desc),
            tags=tags,
            desc_snippet=desc,
            columns=columns,
        )

        if best_candidate is None:
            best_candidate = candidate
            continue
        if best_candidate.interval_bucket == "unknown" and candidate.interval_bucket != "unknown":
            best_candidate = candidate

    if best_candidate is None:
        return None, rejections
    return best_candidate, rejections


def select_diverse(
    candidates: list[Candidate],
    target_count: int,
    rng: random.Random,
) -> list[Candidate]:
    if len(candidates) <= target_count:
        return candidates

    selected: list[Candidate] = []
    remaining = candidates.copy()
    interval_counts: dict[str, int] = {}
    shape_counts: dict[str, int] = {}
    industry_counts: dict[str, int] = {}
    multi_counts = {True: 0, False: 0}

    for _ in range(target_count):
        best_idx = -1
        best_score = -1.0

        for idx, candidate in enumerate(remaining):
            score = 0.0
            score += 3.0 / (1.0 + interval_counts.get(candidate.interval_bucket, 0))
            score += 2.0 / (1.0 + shape_counts.get(candidate.shape_bucket, 0))
            score += 2.0 / (1.0 + industry_counts.get(candidate.industry_hint, 0))
            score += 1.3 / (1.0 + multi_counts.get(candidate.has_multi_id, 0))
            score += rng.random() * 0.25

            if score > best_score:
                best_score = score
                best_idx = idx

        if best_idx < 0:
            break

        selected_item = remaining.pop(best_idx)
        selected.append(selected_item)
        interval_counts[selected_item.interval_bucket] = (
            interval_counts.get(selected_item.interval_bucket, 0) + 1
        )
        shape_counts[selected_item.shape_bucket] = (
            shape_counts.get(selected_item.shape_bucket, 0) + 1
        )
        industry_counts[selected_item.industry_hint] = (
            industry_counts.get(selected_item.industry_hint, 0) + 1
        )
        multi_counts[selected_item.has_multi_id] = (
            multi_counts.get(selected_item.has_multi_id, 0) + 1
        )

    return selected


def save_raw_dataset(
    hf_id: str,
    source_path: str,
    split: str,
    output_dir: Path,
    max_rows: int,
    max_bytes: int,
    source_max_bytes: int,
) -> tuple[int, int]:
    source = SourceFile(path=source_path, split=split, config_name=None)
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / "rows.jsonl"

    if not source.path:
        raise ValueError("empty source path")

    # We only need a deterministic subset here; preserve order.
    rows = sample_candidate_rows(
        hf_id, source, sample_rows=max_rows, source_max_bytes=source_max_bytes
    )
    raw_size = 0

    with raw_path.open("w", encoding="utf-8") as handle:
        saved = 0
        for row in rows[:max_rows]:
            if not isinstance(row, dict):
                continue
            line = serialize_json_line(row) + "\n"
            encoded = line.encode("utf-8")
            if raw_size + len(encoded) > max_bytes:
                break
            handle.write(line)
            saved += 1
            raw_size += len(encoded)
    return saved, raw_size


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.count <= 0:
        raise SystemExit("--count must be > 0")
    if args.raw_max_bytes <= 0:
        raise SystemExit("--raw-max-bytes must be > 0")
    if args.source_max_bytes <= 0:
        raise SystemExit("--source-max-bytes must be > 0")

    out_dir = Path(args.out_dir).resolve()
    if out_dir.exists() and any(out_dir.iterdir()):
        if not args.force:
            raise SystemExit("output directory is not empty; use --force to overwrite")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    min_contiguous = (
        args.context_cap + args.horizon if args.min_contiguous is None else args.min_contiguous
    )
    rng = random.Random(args.seed)
    filter_query = resolve_filter_query(args.filter, args.source_url)
    attempts = 0
    candidate_pool_target = max(args.count, int(args.count * args.min_pool_multiplier))
    max_ids_to_scan = min(
        args.attempt_limit,
        max(args.count * 20, candidate_pool_target),
    )

    print(
        f"collecting datasets via REST (filter={filter_query}, page_size={args.page_size}, "
        f"target_pool={candidate_pool_target})"
    )
    ids = iter_dataset_ids(
        filter_query,
        max_ids=max_ids_to_scan,
        page_size=args.page_size,
        seed=args.seed,
    )
    if not ids:
        raise SystemExit("no datasets returned from HF REST listing")
    ids = ids[: max(args.count * 20, candidate_pool_target)]

    candidates: list[Candidate] = []
    rejections: list[dict[str, Any]] = []
    started_at = time.time()

    for hf_id in ids:
        if attempts >= args.attempt_limit:
            break
        attempts += 1
        if attempts == 1 or attempts % 10 == 0:
            elapsed = int(time.time() - started_at)
            print(f"evaluating candidate #{attempts} / {len(ids)} (time={elapsed}s)")
        candidate, cand_rejections = evaluate_dataset(
            hf_id=hf_id,
            sample_rows=args.sample_rows,
            min_timestamp_ratio=args.min_timestamp_ratio,
            min_target_ratio=args.min_target_ratio,
            min_contiguous_rows=min_contiguous,
            allow_binary=args.allow_binary,
            source_max_bytes=args.source_max_bytes,
        )
        rejections.extend(cand_rejections)
        if candidate is None:
            continue
        candidates.append(candidate)

        print(
            f"accepted {candidate.hf_id} | split={candidate.split} | "
            f"interval={candidate.interval_bucket} "
            f"shape={candidate.shape_bucket} | multi_id={candidate.has_multi_id} "
            f"source={candidate.source_path}"
        )
        if len(candidates) >= candidate_pool_target:
            break

    if not candidates:
        raise SystemExit("no candidate datasets accepted")

    selected = select_diverse(candidates, args.count, rng)

    manifest_items: list[dict[str, Any]] = []
    saved_count = 0
    for candidate in selected:
        dataset_dir = out_dir / safe_path_name(candidate.hf_id)
        dataset_dir.mkdir(parents=True, exist_ok=True)
        meta_path = dataset_dir / "meta.json"

        try:
            saved_rows, raw_size = save_raw_dataset(
                hf_id=candidate.hf_id,
                source_path=candidate.source_path,
                split=candidate.split,
                output_dir=dataset_dir,
                max_rows=args.max_rows,
                max_bytes=args.raw_max_bytes,
                source_max_bytes=args.source_max_bytes,
            )
        except Exception as exc:
            rejections.append(
                {
                    "hf_id": candidate.hf_id,
                    "split": candidate.split,
                    "reason": "save_error",
                    "detail": str(exc),
                }
            )
            continue

        if saved_rows <= 0:
            rejections.append(
                {
                    "hf_id": candidate.hf_id,
                    "split": candidate.split,
                    "reason": "empty_save",
                    "detail": "dataset produced no rows",
                }
            )
            continue

        if raw_size > args.raw_max_bytes:
            rejections.append(
                {
                    "hf_id": candidate.hf_id,
                    "split": candidate.split,
                    "reason": "raw_size_exceeded",
                    "detail": f"raw bytes {raw_size} > limit {args.raw_max_bytes}",
                }
            )
            continue

        raw_path = dataset_dir / "raw" / "rows.jsonl"

        meta = {
            "hf_id": candidate.hf_id,
            "selected_split": candidate.split,
            "selected_file": candidate.source_path,
            "selected_config": candidate.source_config,
            "source_file_url": file_url(candidate.hf_id, candidate.source_path),
            "timestamp_column": candidate.timestamp_column,
            "target_column": candidate.target_column,
            "num_rows_saved": saved_rows,
            "max_rows_cap": args.max_rows,
            "raw_file": "raw/rows.jsonl",
            "raw_file_size": raw_size,
            "interval_bucket": candidate.interval_bucket,
            "interval_seconds": candidate.interval_seconds,
            "shape": candidate.shape_bucket,
            "industry_hint": candidate.industry_hint,
            "has_multiple_id": candidate.has_multi_id,
            "columns": candidate.columns,
            "tags": candidate.tags,
            "desc_snippet": candidate.desc_snippet,
            "collected_at_utc": datetime.now(tz=UTC).replace(microsecond=0).isoformat(),
        }
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

        manifest_items.append(
            {
                "hf_id": candidate.hf_id,
                "split": candidate.split,
                "source_path": candidate.source_path,
                "interval_bucket": candidate.interval_bucket,
                "shape_bucket": candidate.shape_bucket,
                "industry_hint": candidate.industry_hint,
                "has_multiple_id": candidate.has_multi_id,
                "num_rows_saved": saved_rows,
                "raw_file": str(raw_path.relative_to(out_dir)),
                "meta_file": str(meta_path.relative_to(out_dir)),
            }
        )
        saved_count += 1

    index = {
        "generated_at_utc": datetime.now(tz=UTC).replace(microsecond=0).isoformat(),
        "seed": args.seed,
        "target_count": args.count,
        "candidate_pool_target": candidate_pool_target,
        "attempts": attempts,
        "candidate_pool": len(candidates),
        "saved_count": saved_count,
        "max_rows": args.max_rows,
        "min_contiguous": min_contiguous,
        "method": "hf-rest",
        "items": manifest_items,
    }

    (out_dir / "_index.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (out_dir / "_rejections.jsonl").open("w", encoding="utf-8") as handle:
        for row in rejections:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")

    interval_summary: dict[str, int] = {}
    industry_summary: dict[str, int] = {}
    shape_summary: dict[str, int] = {}
    for item in manifest_items:
        interval_summary[item["interval_bucket"]] = (
            interval_summary.get(item["interval_bucket"], 0) + 1
        )
        industry_summary[item["industry_hint"]] = industry_summary.get(item["industry_hint"], 0) + 1
        shape_summary[item["shape_bucket"]] = shape_summary.get(item["shape_bucket"], 0) + 1

    print(f"saved={saved_count}")
    print(f"intervals={interval_summary}")
    print(f"industries={industry_summary}")
    print(f"shapes={shape_summary}")
    print(f"index={out_dir / '_index.json'}")

    return 0 if saved_count >= args.count else 1


if __name__ == "__main__":
    raise SystemExit(main())
