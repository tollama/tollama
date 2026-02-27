"""Gather HuggingFace time-series datasets into a deterministic local catalog.

This script is intended for optional local/manual E2E runs. It keeps catalog quality
high by validating timestamp/target parseability on streamed samples before adding an
entry to ``hf_dataset_catalog.yaml``.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml
from huggingface_hub import HfApi

MAX_ROWS = 1_000_000
DEFAULT_CONTEXT_CAP = 512
DEFAULT_HORIZON = 24
DEFAULT_SAMPLE_ROWS = 600
DEFAULT_MIN_RATIO = 0.90


# Datasets known to be huge, malformed, or irrelevant for scalar forecasting.
KNOWN_BAD_DATASETS: set[str] = {
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


@dataclass(frozen=True)
class SchemaAssessment:
    split: str
    timestamp_column: str
    target_column: str
    timestamp_ratio: float
    target_ratio: float
    contiguous_valid_rows: int

    @property
    def score(self) -> tuple[int, float, float, str, str]:
        return (
            self.contiguous_valid_rows,
            self.timestamp_ratio,
            self.target_ratio,
            self.timestamp_column,
            self.target_column,
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gather HuggingFace datasets for local E2E.")
    parser.add_argument("--limit", type=int, default=300)
    parser.add_argument("--target-count", type=int, default=50)
    parser.add_argument("--max-rows", type=int, default=MAX_ROWS)
    parser.add_argument("--context-cap", type=int, default=DEFAULT_CONTEXT_CAP)
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    parser.add_argument("--sample-rows", type=int, default=DEFAULT_SAMPLE_ROWS)
    parser.add_argument("--min-timestamp-ratio", type=float, default=DEFAULT_MIN_RATIO)
    parser.add_argument("--min-target-ratio", type=float, default=DEFAULT_MIN_RATIO)
    parser.add_argument(
        "--output",
        type=str,
        default=str(Path(__file__).parent / "hf_dataset_catalog.yaml"),
    )
    parser.add_argument(
        "--rejections-output",
        type=str,
        default=str(Path(__file__).parent / "hf_dataset_rejections.json"),
    )
    return parser.parse_args(argv)


def get_dataset_num_rows(api: HfApi, hf_id: str) -> int | None:
    """Try to read total row count from dataset metadata without download."""
    try:
        info = api.dataset_info(hf_id, files_metadata=False)
    except Exception:
        return None

    card_data = getattr(info, "card_data", None)
    if card_data is None:
        return None

    # ``card_data`` can be a mapping-like object. We only consume known shapes.
    raw_dataset_info = None
    if isinstance(card_data, dict):
        raw_dataset_info = card_data.get("dataset_info")
    elif hasattr(card_data, "get"):
        raw_dataset_info = card_data.get("dataset_info")
    elif hasattr(card_data, "dataset_info"):
        raw_dataset_info = card_data.dataset_info

    if raw_dataset_info is None:
        return None

    return _sum_num_examples(raw_dataset_info)


def _sum_num_examples(payload: Any) -> int | None:
    total = 0

    if isinstance(payload, dict):
        splits = payload.get("splits")
        if isinstance(splits, dict):
            for split_info in splits.values():
                total += _extract_num_examples(split_info)
            return total if total > 0 else None

    if isinstance(payload, list):
        for item in payload:
            item_total = _sum_num_examples(item)
            if isinstance(item_total, int):
                total += item_total
        return total if total > 0 else None

    return None


def _extract_num_examples(split_info: Any) -> int:
    if not isinstance(split_info, dict):
        return 0
    value = split_info.get("num_examples")
    if isinstance(value, int) and value >= 0:
        return value
    return 0


def _parse_timestamp(raw: Any) -> datetime | None:
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


def _parse_float(raw: Any) -> float | None:
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


def _normalized_name(column: str) -> str:
    return column.strip().lower().replace(" ", "_")


def _name_priority(column: str, hints: tuple[str, ...]) -> int:
    normalized = _normalized_name(column)
    for index, hint in enumerate(hints):
        if normalized == hint:
            return index
    for index, hint in enumerate(hints):
        if hint in normalized:
            return len(hints) + index
    return 10_000


def infer_schema_from_rows(rows: list[dict[str, Any]]) -> tuple[str | None, str | None]:
    """Infer the best timestamp/target columns from sampled rows."""
    if not rows:
        return None, None

    columns = sorted({key for row in rows for key in row.keys() if isinstance(key, str) and key})
    if not columns:
        return None, None

    timestamp_scores: list[tuple[str, float, int]] = []
    target_scores: list[tuple[str, float, int]] = []
    total_rows = len(rows)

    for column in columns:
        ts_ok = 0
        target_ok = 0
        for row in rows:
            if _parse_timestamp(row.get(column)) is not None:
                ts_ok += 1
            if _parse_float(row.get(column)) is not None:
                target_ok += 1

        timestamp_scores.append(
            (
                column,
                ts_ok / total_rows,
                _name_priority(column, TIMESTAMP_NAME_HINTS),
            )
        )
        target_scores.append(
            (
                column,
                target_ok / total_rows,
                _name_priority(column, TARGET_NAME_HINTS),
            )
        )

    timestamp_scores.sort(key=lambda item: (-item[1], item[2], item[0]))
    target_scores.sort(key=lambda item: (-item[1], item[2], item[0]))

    timestamp_column = timestamp_scores[0][0] if timestamp_scores else None
    target_column = None
    for candidate, _, _ in target_scores:
        if candidate != timestamp_column:
            target_column = candidate
            break

    return timestamp_column, target_column


def assess_schema_rows(
    *,
    rows: list[dict[str, Any]],
    split: str,
    timestamp_column: str,
    target_column: str,
) -> SchemaAssessment:
    """Measure parseability and contiguous valid run length for a schema pair."""
    total = len(rows)
    timestamp_ok = 0
    target_ok = 0
    max_run = 0
    current_run = 0

    for row in rows:
        ts_valid = _parse_timestamp(row.get(timestamp_column)) is not None
        target_valid = _parse_float(row.get(target_column)) is not None

        if ts_valid:
            timestamp_ok += 1
        if target_valid:
            target_ok += 1

        if ts_valid and target_valid:
            current_run += 1
            if current_run > max_run:
                max_run = current_run
        else:
            current_run = 0

    return SchemaAssessment(
        split=split,
        timestamp_column=timestamp_column,
        target_column=target_column,
        timestamp_ratio=(timestamp_ok / total) if total else 0.0,
        target_ratio=(target_ok / total) if total else 0.0,
        contiguous_valid_rows=max_run,
    )


def is_schema_acceptable(
    *,
    assessment: SchemaAssessment,
    min_timestamp_ratio: float,
    min_target_ratio: float,
    min_contiguous_rows: int,
) -> tuple[bool, str | None]:
    if assessment.timestamp_ratio < min_timestamp_ratio:
        return (
            False,
            f"timestamp_ratio={assessment.timestamp_ratio:.3f} < {min_timestamp_ratio:.3f}",
        )
    if assessment.target_ratio < min_target_ratio:
        return (
            False,
            f"target_ratio={assessment.target_ratio:.3f} < {min_target_ratio:.3f}",
        )
    if assessment.contiguous_valid_rows < min_contiguous_rows:
        return (
            False,
            f"contiguous_valid_rows={assessment.contiguous_valid_rows} < {min_contiguous_rows}",
        )
    return True, None


def _stream_sample_rows(*, hf_id: str, split: str, sample_rows: int) -> list[dict[str, Any]]:
    stream = _load_dataset(hf_id, split=split, streaming=True, trust_remote_code=True)
    rows: list[dict[str, Any]] = []
    for item in stream:
        if not isinstance(item, dict):
            continue
        rows.append(item)
        if len(rows) >= sample_rows:
            break
    return rows


def _discover_splits(hf_id: str) -> list[str]:
    dataset = _load_dataset(hf_id, streaming=True, trust_remote_code=True)

    if hasattr(dataset, "keys"):
        keys = [str(key) for key in dataset.keys()]
        return sorted(keys)

    return ["train"]


def _load_dataset(hf_id: str, **kwargs: Any) -> Any:
    from datasets import load_dataset

    return load_dataset(hf_id, **kwargs)


def _build_dataset_entry(
    *,
    hf_id: str,
    num_rows: int | None,
    horizon: int,
    assessment: SchemaAssessment,
) -> dict[str, Any]:
    return {
        "name": hf_id.replace("/", "_"),
        "kind": "huggingface_dataset",
        "hf_id": hf_id,
        "split": assessment.split,
        "num_rows": num_rows,
        "freq": "H",
        "horizon": horizon,
        "timestamp_column": assessment.timestamp_column,
        "target_column": assessment.target_column,
    }


def _rejection_record(*, hf_id: str, split: str, reason: str, detail: str) -> dict[str, Any]:
    return {
        "hf_id": hf_id,
        "split": split,
        "reason": reason,
        "detail": detail,
    }


def _evaluate_dataset(
    *,
    hf_id: str,
    num_rows: int | None,
    sample_rows: int,
    horizon: int,
    context_cap: int,
    min_timestamp_ratio: float,
    min_target_ratio: float,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    min_contiguous = context_cap + horizon
    rejections: list[dict[str, Any]] = []

    try:
        splits = _discover_splits(hf_id)
    except Exception as exc:  # noqa: BLE001
        return None, [
            _rejection_record(
                hf_id=hf_id,
                split="-",
                reason="split_discovery_error",
                detail=str(exc),
            )
        ]

    best_assessment: SchemaAssessment | None = None

    for split in splits:
        try:
            rows = _stream_sample_rows(hf_id=hf_id, split=split, sample_rows=sample_rows)
        except Exception as exc:  # noqa: BLE001
            rejections.append(
                _rejection_record(
                    hf_id=hf_id,
                    split=split,
                    reason="sample_load_error",
                    detail=str(exc),
                )
            )
            continue

        if not rows:
            rejections.append(
                _rejection_record(
                    hf_id=hf_id,
                    split=split,
                    reason="empty_sample",
                    detail="no rows returned",
                )
            )
            continue

        timestamp_column, target_column = infer_schema_from_rows(rows)
        if not timestamp_column or not target_column:
            rejections.append(
                _rejection_record(
                    hf_id=hf_id,
                    split=split,
                    reason="schema_inference_failed",
                    detail="unable to infer timestamp/target columns",
                )
            )
            continue

        assessment = assess_schema_rows(
            rows=rows,
            split=split,
            timestamp_column=timestamp_column,
            target_column=target_column,
        )
        accepted, detail = is_schema_acceptable(
            assessment=assessment,
            min_timestamp_ratio=min_timestamp_ratio,
            min_target_ratio=min_target_ratio,
            min_contiguous_rows=min_contiguous,
        )
        if not accepted:
            reason = "schema_quality_failed"
            if detail is not None and detail.startswith("timestamp_ratio="):
                reason = "low_timestamp_ratio"
            elif detail is not None and detail.startswith("target_ratio="):
                reason = "low_target_ratio"
            elif detail is not None and detail.startswith("contiguous_valid_rows="):
                reason = "insufficient_contiguous_rows"
            rejections.append(
                _rejection_record(
                    hf_id=hf_id,
                    split=split,
                    reason=reason,
                    detail=detail or "schema quality threshold not met",
                )
            )
            continue

        if best_assessment is None or assessment.score > best_assessment.score:
            best_assessment = assessment

    if best_assessment is None:
        if not rejections:
            rejections.append(
                _rejection_record(
                    hf_id=hf_id,
                    split="-",
                    reason="no_valid_split",
                    detail="no split met schema quality thresholds",
                )
            )
        return None, rejections

    return (
        _build_dataset_entry(
            hf_id=hf_id,
            num_rows=num_rows,
            horizon=horizon,
            assessment=best_assessment,
        ),
        rejections,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    api = HfApi()

    required_contiguous = args.context_cap + args.horizon
    print(
        "Scanning HuggingFace datasets "
        f"(limit={args.limit}, target_count={args.target_count}, max_rows={args.max_rows:,}, "
        f"sample_rows={args.sample_rows}, min_contiguous={required_contiguous})"
    )

    listed = list(
        api.list_datasets(
            filter="task_categories:time-series-forecasting",
            limit=args.limit,
        )
    )
    dataset_ids = sorted({str(item.id) for item in listed if getattr(item, "id", None)})

    catalog_entries: list[dict[str, Any]] = []
    rejections: list[dict[str, Any]] = []

    for index, hf_id in enumerate(dataset_ids, start=1):
        if len(catalog_entries) >= args.target_count:
            break

        print(f"[{index}/{len(dataset_ids)}] {hf_id}")

        if hf_id in KNOWN_BAD_DATASETS:
            rejections.append(
                _rejection_record(
                    hf_id=hf_id,
                    split="-",
                    reason="blocklisted",
                    detail="dataset listed in KNOWN_BAD_DATASETS",
                )
            )
            continue

        num_rows = get_dataset_num_rows(api, hf_id)
        if isinstance(num_rows, int) and num_rows > args.max_rows:
            rejections.append(
                _rejection_record(
                    hf_id=hf_id,
                    split="-",
                    reason="too_many_rows",
                    detail=f"num_rows={num_rows} > max_rows={args.max_rows}",
                )
            )
            continue

        entry, entry_rejections = _evaluate_dataset(
            hf_id=hf_id,
            num_rows=num_rows,
            sample_rows=args.sample_rows,
            horizon=args.horizon,
            context_cap=args.context_cap,
            min_timestamp_ratio=args.min_timestamp_ratio,
            min_target_ratio=args.min_target_ratio,
        )
        rejections.extend(entry_rejections)

        if entry is None:
            print("  -> rejected")
            continue

        catalog_entries.append(entry)
        print(
            "  -> accepted "
            f"split={entry['split']} ts={entry['timestamp_column']} target={entry['target_column']}"
        )

    catalog_entries.sort(key=lambda item: str(item["hf_id"]))
    rejections.sort(key=lambda item: (str(item["hf_id"]), str(item["split"]), str(item["reason"])))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        yaml.safe_dump({"datasets": catalog_entries}, sort_keys=False),
        encoding="utf-8",
    )

    rejections_path = Path(args.rejections_output)
    rejections_path.parent.mkdir(parents=True, exist_ok=True)
    rejections_path.write_text(
        json.dumps({"rejections": rejections}, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print(f"Accepted datasets: {len(catalog_entries)}")
    print(f"Rejected datasets: {len(rejections)}")
    print(f"Catalog: {output_path}")
    print(f"Rejections: {rejections_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
