#!/usr/bin/env python3
"""Randomly pick N unique HF datasets from the full catalog and write a subset YAML."""

import argparse
import random
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", default="scripts/e2e_realdata/hf_dataset_catalog.yaml")
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output", default="/tmp/hf_subset_catalog.yaml")
    args = parser.parse_args()

    data = yaml.safe_load(Path(args.catalog).read_text())
    entries = data["datasets"]

    # De-duplicate by hf_id, keep first occurrence of each
    seen: dict[str, dict] = {}
    for e in entries:
        hf_id = e["hf_id"]
        if hf_id not in seen:
            seen[hf_id] = e

    unique_entries = list(seen.values())
    print(f"Total unique datasets in catalog: {len(unique_entries)}")

    rng = random.Random(args.seed)
    n = min(args.n, len(unique_entries))
    chosen = rng.sample(unique_entries, k=n)

    print(f"\nRandomly selected {n} datasets (seed={args.seed}):")
    for i, e in enumerate(chosen, 1):
        print(f"  {i:2d}. {e['hf_id']}  (freq={e['freq']}, horizon={e['horizon']})")

    # Rename so names are clean (strip _rNN suffix on names)
    for i, e in enumerate(chosen, 1):
        base = e["hf_id"].replace("/", "_")
        e = dict(e)
        e["name"] = f"{base}_s{i:02d}"
        chosen[i - 1] = e

    subset = {"datasets": chosen}
    out = Path(args.output)
    out.write_text(yaml.dump(subset, sort_keys=False, allow_unicode=True))
    print(f"\nSubset catalog written to: {out}")


if __name__ == "__main__":
    main()
