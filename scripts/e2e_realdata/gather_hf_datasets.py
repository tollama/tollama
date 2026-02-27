"""Gather time-series datasets from HuggingFace and generate a catalog.

Only includes datasets with fewer than MAX_ROWS total rows to avoid
accidentally queueing massive downloads during E2E testing.
"""

import argparse
from pathlib import Path

import yaml
from datasets import load_dataset
from huggingface_hub import HfApi, DatasetCard
from huggingface_hub.utils import EntryNotFoundError


MAX_ROWS = 1_000_000  # skip datasets larger than this


def get_dataset_num_rows(hf_id: str) -> int | None:
    """Return the total row count of a HuggingFace dataset via API (no download).

    Uses dataset_info endpoint which returns metadata including num_rows per split.
    Returns None if the info is not available.
    """
    try:
        api = HfApi()
        info = api.dataset_info(hf_id, files_metadata=False)
        if info.card_data and hasattr(info.card_data, "dataset_info"):
            ds_info = info.card_data.dataset_info
            if isinstance(ds_info, dict):
                total = 0
                # dataset_info may be a flat dict with splits
                for split_info in ds_info.get("splits", {}).values():
                    n = split_info.get("num_examples", 0) if isinstance(split_info, dict) else 0
                    total += n
                if total > 0:
                    return total
            elif isinstance(ds_info, list):
                # some cards have a list of config sub-info
                total = 0
                for cfg in ds_info:
                    for split_info in cfg.get("splits", {}).values() if isinstance(cfg, dict) else []:
                        n = split_info.get("num_examples", 0) if isinstance(split_info, dict) else 0
                        total += n
                if total > 0:
                    return total
        return None
    except Exception:
        return None


def infer_schema(sample_row: dict) -> tuple[str | None, str | None]:
    """Infer the timestamp and target columns from a sample row's keys."""
    # Filter out hidden metadata keys
    keys = [k for k in sample_row.keys() if not k.startswith("_")]

    # Heuristics for timestamp column
    timestamp_candidates = [
        "date", "datetime", "timestamp", "time", "dt", "period", "ds",
        "Date", "DateTime", "Timestamp", "Time", "DateUTC", "date_time",
        "TSTMP", "rate_date",
    ]
    timestamp_column = None
    for candidate in timestamp_candidates:
        for k in keys:
            if k.lower() == candidate.lower():
                timestamp_column = k
                break
        if timestamp_column:
            break

    # Heuristics for target column
    target_candidates = [
        "target", "value", "actual", "sale_amount", "y", "close", "price",
        "demand", "Close", "Value", "Target", "Actual", "label",
    ]
    target_column = None
    for candidate in target_candidates:
        for k in keys:
            if k.lower() == candidate.lower():
                target_column = k
                break
        if target_column:
            break

    # Fallbacks if we can't find perfect matches, but we need at least 2 columns
    if not timestamp_column and len(keys) >= 1:
        timestamp_column = keys[0]

    if not target_column and len(keys) >= 2:
        target_column = keys[1] if keys[1] != timestamp_column else keys[0]

    return timestamp_column, target_column


def main():
    parser = argparse.ArgumentParser(description="Gather HuggingFace TSFM datasets.")
    parser.add_argument(
        "--limit",
        type=int,
        default=300,
        help="Number of datasets to scan from HuggingFace (we filter down from this pool).",
    )
    parser.add_argument(
        "--target-count",
        type=int,
        default=100,
        help="Target number of datasets to include in the catalog.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=MAX_ROWS,
        help="Maximum allowed total row count for a dataset.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(Path(__file__).parent / "hf_dataset_catalog.yaml"),
        help="Path to output YAML catalog.",
    )
    args = parser.parse_args()

    api = HfApi()
    print(
        f"Fetching up to {args.limit} time-series-forecasting datasets from Hugging Face "
        f"(target: {args.target_count}, max rows: {args.max_rows:,})..."
    )
    all_datasets = list(
        api.list_datasets(filter="task_categories:time-series-forecasting", limit=args.limit)
    )

# Datasets known to be extremely large, broken, or irrelevant despite passing metadata checks
KNOWN_BAD_DATASETS: set[str] = {
    "Numerati/numerai-datasets",           # ~100M rows ML competition dataset
    "MLNTeam-Unical/NFT-70M_transactions", # 70M rows
    "MLNTeam-Unical/NFT-70M_text",         # 70M rows
    "MLNTeam-Unical/NFT-70M_image",        # 70M rows
    "rajistics/electricity_demand",        # schema mismatch
    "davidwisdom/fake_railroad_company",   # not real time series
    "raeidsaqur/nifty-rl",                 # RL dataset, not TS
    "louisbrulenaudet/obis",               # biological data, not TS
    "hereldav/TimeAware",                  # schema mismatch
    "Maple728/Time-300B",                  # metadata-only
    "saluslab/HM-SYNC",                    # image format
    "mirai-ml/wids2023_competition_weather_forecasting",  # GPS coords, not values
    "traqq/weekly_activity_data_jul24_jul28",  # non-numeric TS
    "ChengsenWang/TSQA",                   # QA format, not TS values
    "mschi/blogspot_raw",                  # blog text
    "ChengsenWang/ChatTime-1-Pretrain-1M", # image/text
    "ChengsenWang/ChatTime-1-Finetune-100K", # image/text
}

    catalog_entries = []

    for i, ds in enumerate(all_datasets):
        if len(catalog_entries) >= args.target_count:
            break

        print(f"[{i+1}/{len(all_datasets)}] Processing {ds.id}...")

        # Skip known bad datasets
        if ds.id in KNOWN_BAD_DATASETS:
            print(f"  -> Skipped: in blocklist")
            continue

        # --- Row-count gate: use HF API metadata first (no download) ---
        num_rows = get_dataset_num_rows(ds.id)
        if num_rows is not None:
            if num_rows > args.max_rows:
                print(f"  -> Skipped: {num_rows:,} rows exceeds limit ({args.max_rows:,})")
                continue
            else:
                print(f"  -> Row count OK: {num_rows:,}")
        else:
            print(f"  -> Row count unknown, will probe schema via streaming (will abort if too large)")

        try:
            # Use streaming to peek at the first row only - avoids downloading large files
            dataset = load_dataset(ds.id, streaming=True, trust_remote_code=True)
            split_name = list(dataset.keys())[0] if isinstance(dataset, dict) else "train"
            split_data = dataset[split_name] if isinstance(dataset, dict) else dataset

            # Get the first row
            sample_row = next(iter(split_data))

            timestamp_col, target_col = infer_schema(sample_row)

            if timestamp_col and target_col:
                catalog_entries.append({
                    "name": ds.id.replace("/", "_"),
                    "kind": "huggingface_dataset",
                    "hf_id": ds.id,
                    "num_rows": num_rows,  # store metadata for reference
                    "freq": "H",
                    "horizon": 24,
                    "timestamp_column": timestamp_col,
                    "target_column": target_col,
                })
                print(f"  -> Inferred schema: timestamp='{timestamp_col}', target='{target_col}'")
            else:
                print(f"  -> Could not infer schema from columns: {list(sample_row.keys())}")

        except Exception as e:
            print(f"  -> Failed to load or parse: {e}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    catalog = {"datasets": catalog_entries}
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(catalog, f, sort_keys=False, default_flow_style=False)

    print(f"\nSaved {len(catalog_entries)} datasets to {output_path}")


if __name__ == "__main__":
    main()
