#!/usr/bin/env python3
"""Collect random Hugging Face time-series datasets and store raw data + metadata.

Each collected dataset is written under `hf_data/<safe_hf_id>/` with:
- raw/rows.jsonl
- meta.json
- `_index.json` and `_rejections.jsonl` in the root output directory.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from datasets import load_dataset
from huggingface_hub import HfApi

MAX_ROWS = 1_000_000
DEFAULT_CONTEXT_CAP = 512
DEFAULT_HORIZON = 24
DEFAULT_SAMPLE_ROWS = 600
DEFAULT_MIN_RATIO = 0.90
DEFAULT_TRUST_REMOTE_CODE = False

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


@dataclass(frozen=True)
class Candidate:
    hf_id: str
    split: str
    timestamp_column: str
    target_column: str
    num_rows_est: int | None
    interval_bucket: str
    interval_seconds: float | None
    shape_bucket: str
    has_multi_id: bool
    industry_hint: str
    tags: list[str]
    desc_snippet: str
    columns: list[str]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect random HF time-series datasets.")
    parser.add_argument("--out-dir", default="hf_data", help="Output directory (default: hf_data)")
    parser.add_argument("--count", type=int, default=300, help="Number of datasets to collect")
    parser.add_argument("--limit", type=int, default=2500, help="HF dataset query limit")
    parser.add_argument(
        "--attempt-limit", type=int, default=40_000, help="Hard cap on candidate attempts"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--max-rows", type=int, default=MAX_ROWS, help="Per-dataset row cap (1,000,000)"
    )
    parser.add_argument(
        "--sample-rows", type=int, default=DEFAULT_SAMPLE_ROWS, help="Sample rows per split"
    )
    parser.add_argument("--context-cap", type=int, default=DEFAULT_CONTEXT_CAP)
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    parser.add_argument("--min-timestamp-ratio", type=float, default=DEFAULT_MIN_RATIO)
    parser.add_argument("--min-target-ratio", type=float, default=DEFAULT_MIN_RATIO)
    parser.add_argument("--min-contiguous", type=int, default=None)
    parser.add_argument(
        "--min-pool-multiplier", type=float, default=2.0, help="Candidate pool size multiplier"
    )
    parser.add_argument("--force", action="store_true", help="Overwrite non-empty output directory")
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
    """Resolve filter expression from CLI filter and optional source URL."""
    if not source_url:
        return filter_value

    parsed = urlparse(source_url)
    query = parse_qs(parsed.query)
    task_categories = query.get("task_categories")
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

    formats = (
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
    )
    for fmt in formats:
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
            if row.get(column) not in (None, "") and isinstance(row, dict)
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


def discover_splits(hf_id: str) -> list[str]:
    dataset = load_dataset(hf_id, streaming=True, trust_remote_code=DEFAULT_TRUST_REMOTE_CODE)
    if hasattr(dataset, "keys"):
        return sorted([str(key) for key in dataset.keys()])
    return ["train"]


def stream_rows(hf_id: str, split: str, sample_rows: int) -> list[dict[str, Any]]:
    dataset = load_dataset(
        hf_id, split=split, streaming=True, trust_remote_code=DEFAULT_TRUST_REMOTE_CODE
    )
    rows: list[dict[str, Any]] = []
    for item in dataset:
        if not isinstance(item, dict):
            continue
        rows.append(item)
        if len(rows) >= sample_rows:
            break
    return rows


def load_dataset_metadata(api: HfApi, hf_id: str) -> tuple[list[str], str, int | None, set[str]]:
    info = api.dataset_info(hf_id, files_metadata=False)
    tags = [str(tag) for tag in (getattr(info, "tags", []) or [])]

    desc = str(getattr(info, "description", "") or "")
    if not desc:
        card_data = getattr(info, "card_data", None)
        if isinstance(card_data, dict):
            desc = str(card_data.get("pretty_print", "") or card_data.get("description", "") or "")

    row_est = None
    card_data = getattr(info, "card_data", None)
    dataset_info = None
    if isinstance(card_data, dict):
        dataset_info = card_data.get("dataset_info")
    if isinstance(dataset_info, dict):
        splits = dataset_info.get("splits")
        if isinstance(splits, dict):
            total = 0
            for split_info in splits.values():
                if not isinstance(split_info, dict):
                    continue
                value = split_info.get("num_examples")
                if isinstance(value, int):
                    total += value
                if total > 0:
                    row_est = total

    sibling_files = {
        str(getattr(sibling, "rfilename", ""))
        for sibling in getattr(info, "siblings", [])
        if getattr(sibling, "rfilename", None)
    }

    return tags, desc[:500], row_est, sibling_files


def evaluate_dataset(
    *,
    api: HfApi,
    hf_id: str,
    sample_rows: int,
    min_timestamp_ratio: float,
    min_target_ratio: float,
    min_contiguous_rows: int,
    max_rows: int,
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
        tags, desc, row_est, sibling_files = load_dataset_metadata(api, hf_id)
    except Exception as exc:
        return None, [
            {"hf_id": hf_id, "split": "-", "reason": "metadata_error", "detail": str(exc)}
        ]

    if any(file.endswith(".py") for file in sibling_files):
        return None, [
            {
                "hf_id": hf_id,
                "split": "-",
                "reason": "scripted_dataset",
                "detail": "contains .py dataset script",
            }
        ]

    standard_extensions = (
        ".arrow",
        ".parquet",
        ".csv",
        ".csv.gz",
        ".json",
        ".jsonl",
        ".tsv",
        ".tsv.gz",
    )
    if not any(file.endswith(ext) for file in sibling_files for ext in standard_extensions):
        return (
            None,
            [
                {
                    "hf_id": hf_id,
                    "split": "-",
                    "reason": "unsupported_file_type",
                    "detail": "no supported data file extension found",
                }
            ],
        )

    if isinstance(row_est, int) and row_est > max_rows:
        return (
            None,
            [
                {
                    "hf_id": hf_id,
                    "split": "-",
                    "reason": "too_many_rows",
                    "detail": f"num_rows_est={row_est} > max_rows={max_rows}",
                }
            ],
        )

    try:
        splits = discover_splits(hf_id)
    except Exception as exc:
        return None, [
            {"hf_id": hf_id, "split": "-", "reason": "split_discovery_error", "detail": str(exc)}
        ]

    best_candidate: Candidate | None = None

    for split in splits:
        try:
            rows = stream_rows(hf_id, split, sample_rows)
        except Exception as exc:
            rejections.append(
                {"hf_id": hf_id, "split": split, "reason": "sample_load_error", "detail": str(exc)}
            )
            continue

        if not rows:
            rejections.append(
                {
                    "hf_id": hf_id,
                    "split": split,
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
                    "split": split,
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
                    "split": split,
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
            split=split,
            timestamp_column=ts_col,
            target_column=target_col,
            num_rows_est=row_est,
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
            continue

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
        multi_counts[selected_item.has_multi_id] = multi_counts[selected_item.has_multi_id] + 1

    return selected


def serialize_json_line(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def save_raw_dataset(
    hf_id: str,
    split: str,
    output_dir: Path,
    max_rows: int,
) -> int:
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / "rows.jsonl"

    saved = 0
    dataset = load_dataset(
        hf_id,
        split=split,
        streaming=True,
        trust_remote_code=DEFAULT_TRUST_REMOTE_CODE,
    )

    with raw_path.open("w", encoding="utf-8") as handle:
        for item in dataset:
            if not isinstance(item, dict):
                continue
            handle.write(serialize_json_line(item))
            handle.write("\n")
            saved += 1
            if saved >= max_rows:
                break

    return saved


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.count <= 0:
        raise SystemExit("--count must be > 0")

    out_dir = Path(args.out_dir).resolve()
    if out_dir.exists() and any(out_dir.iterdir()):
        if not args.force:
            raise SystemExit("output directory is not empty; use --force to overwrite")
        shutil.rmtree(out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    if args.min_contiguous is not None:
        min_contiguous = args.min_contiguous
    else:
        min_contiguous = args.context_cap + args.horizon

    rng = random.Random(args.seed)
    api = HfApi()

    candidate_pool_target = max(args.count, int(args.count * args.min_pool_multiplier))
    filter_query = resolve_filter_query(args.filter, args.source_url)

    print(
        "querying HF time-series-forecasting datasets "
        f"(limit={args.limit}, target_pool={candidate_pool_target})"
    )
    try:
        listed = api.list_datasets(filter=filter_query, limit=args.limit)
        dataset_ids = sorted({str(item.id) for item in listed if getattr(item, "id", None)})
    except Exception as exc:
        raise SystemExit(f"HF query failed: {exc}")

    ids = list(dataset_ids)
    rng.shuffle(ids)

    candidates: list[Candidate] = []
    rejections: list[dict[str, Any]] = []

    attempts = 0
    for hf_id in ids:
        if len(candidates) >= candidate_pool_target:
            break
        if attempts >= args.attempt_limit:
            break

        attempts += 1
        candidate, cand_rejections = evaluate_dataset(
            api=api,
            hf_id=hf_id,
            sample_rows=args.sample_rows,
            min_timestamp_ratio=args.min_timestamp_ratio,
            min_target_ratio=args.min_target_ratio,
            min_contiguous_rows=min_contiguous,
            max_rows=args.max_rows,
        )
        rejections.extend(cand_rejections)

        if candidate is None:
            continue

        candidates.append(candidate)
        print(
            f"accepted {candidate.hf_id} | split={candidate.split} | "
            f"interval={candidate.interval_bucket} shape={candidate.shape_bucket} | "
            f"multi_id={candidate.has_multi_id}"
        )

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
            saved_rows = save_raw_dataset(
                hf_id=candidate.hf_id,
                split=candidate.split,
                output_dir=dataset_dir,
                max_rows=args.max_rows,
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

        raw_path = dataset_dir / "raw" / "rows.jsonl"
        raw_size = raw_path.stat().st_size
        meta = {
            "hf_id": candidate.hf_id,
            "selected_split": candidate.split,
            "timestamp_column": candidate.timestamp_column,
            "target_column": candidate.target_column,
            "num_rows_estimate": candidate.num_rows_est,
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
        "attempts": attempts,
        "candidate_pool": len(candidates),
        "saved_count": saved_count,
        "max_rows": args.max_rows,
        "min_contiguous": min_contiguous,
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
