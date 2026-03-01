#!/usr/bin/env python3
"""Import a pre-collected hf_data bundle into the local repository.

The source bundle should be produced from another environment using the
`collect_random_hf_timeseries.py` output layout.
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ImportResult:
    copied: int
    skipped: int
    missing: list[str]
    errors: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Import hf_data datasets from an externally collected bundle."
    )
    parser.add_argument(
        "--source-dir",
        required=True,
        help="Path to source hf_data bundle directory (other machine).",
    )
    parser.add_argument(
        "--out-dir",
        default="hf_data",
        help="Destination directory (default: hf_data)",
    )
    parser.add_argument(
        "--source-index",
        default="_index.json",
        help="Index filename under source root (default: _index.json)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace destination directory if it already exists.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional max number of dataset folders to import.",
    )
    return parser.parse_args()


def read_index(source_root: Path, index_name: str) -> dict:
    index_path = source_root / index_name
    if not index_path.exists():
        return {"items": []}
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("index is not a JSON object")
    return payload


def iter_source_entries(
    source_root: Path, index: dict, limit: int | None
) -> list[tuple[str, Path]]:
    raw_items = []
    for item in index.get("items", []):
        if isinstance(item, dict):
            hf_id = item.get("hf_id")
            if not isinstance(hf_id, str) or not hf_id.strip():
                continue
            raw_items.append((hf_id, source_root / hf_id.replace("/", "__")))

    if raw_items:
        return raw_items[:limit] if limit else raw_items

    # Fallback: import every dataset folder in source root excluding index/rejection files.
    folders: list[tuple[str, Path]] = []
    for child in sorted(source_root.iterdir()):
        if not child.is_dir():
            continue
        folders.append((child.name, child))

    return folders if limit is None else folders[:limit]


def copy_one(source_root: Path, dest_root: Path, source_path: Path) -> None:
    if not source_path.exists():
        raise FileNotFoundError(f"missing source path: {source_path}")

    if source_path.name in {"", "__pycache__"}:
        return

    target = dest_root / source_path.name
    if target.exists():
        if target.is_dir():
            shutil.rmtree(target)
        else:
            target.unlink()
    shutil.copytree(source_path, target)


def import_bundle(
    *,
    source_dir: Path,
    out_dir: Path,
    index_name: str,
    force: bool,
    limit: int | None,
) -> ImportResult:
    if not source_dir.is_dir():
        raise NotADirectoryError(f"source-dir is not a directory: {source_dir}")

    if out_dir.exists() and force:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    index = read_index(source_dir, index_name)
    entries = iter_source_entries(source_dir, index, limit)

    copied = 0
    skipped = 0
    missing: list[str] = []
    errors: list[str] = []

    for hf_id, dataset_dir in entries:
        if not dataset_dir.exists():
            missing.append(hf_id)
            continue

        try:
            copy_one(source_dir, out_dir, dataset_dir)
            copied += 1
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{hf_id}: {exc}")

    # Copy root metadata files if present.
    for filename in ["_index.json", "_rejections.jsonl", "README.md"]:
        source_file = source_dir / filename
        if source_file.exists():
            target_file = out_dir / filename
            target_file.write_text(source_file.read_text(encoding="utf-8"), encoding="utf-8")
            copied += 1

    if limit is not None and copied > limit:
        skipped = copied - limit
        copied = limit

    return ImportResult(copied=copied, skipped=skipped, missing=missing, errors=errors)


def main() -> int:
    args = parse_args()
    source_dir = Path(args.source_dir).resolve()
    out_dir = Path(args.out_dir).resolve()

    result = import_bundle(
        source_dir=source_dir,
        out_dir=out_dir,
        index_name=args.source_index,
        force=args.force,
        limit=args.limit,
    )

    if result.errors:
        print(f"errors={len(result.errors)}")
        for item in result.errors[:20]:
            print(f"  - {item}")
        if len(result.errors) > 20:
            print(f"  ... and {len(result.errors) - 20} more")

    if result.missing:
        print(f"missing={len(result.missing)}")
        for item in result.missing[:20]:
            print(f"  - {item}")
        if len(result.missing) > 20:
            print(f"  ... and {len(result.missing) - 20} more")

    print(f"copied={result.copied}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
