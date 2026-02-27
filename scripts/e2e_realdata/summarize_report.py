"""Summary helpers for real-data E2E artifacts."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from statistics import fmean
from typing import Any


def summarize_entries(entries: list[dict[str, Any]]) -> dict[str, Any]:
    """Build aggregate model/scenario summary payload from raw entries."""
    by_model: dict[str, list[dict[str, Any]]] = {}
    for item in entries:
        model = str(item.get("model", "unknown"))
        by_model.setdefault(model, []).append(item)

    models: dict[str, Any] = {}
    total_failed = 0
    for model, model_entries in sorted(by_model.items()):
        total = len(model_entries)
        passed = sum(1 for item in model_entries if item.get("status") == "pass")
        failed = total - passed
        total_failed += failed

        latencies = [
            float(item["latency_ms"])
            for item in model_entries
            if item.get("status") == "pass" and isinstance(item.get("latency_ms"), (int, float))
        ]

        benchmark_metrics: dict[str, list[float]] = {"mae": [], "rmse": [], "smape": []}
        for item in model_entries:
            if item.get("status") != "pass":
                continue
            if item.get("scenario") != "benchmark_target_only":
                continue
            metrics = item.get("metrics")
            if not isinstance(metrics, dict):
                continue
            for key in benchmark_metrics:
                value = metrics.get(key)
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    benchmark_metrics[key].append(float(value))

        scenario_counts: dict[str, dict[str, int]] = {}
        for item in model_entries:
            scenario = str(item.get("scenario", "unknown"))
            counter = scenario_counts.setdefault(scenario, {"pass": 0, "fail": 0})
            counter["pass" if item.get("status") == "pass" else "fail"] += 1

        models[model] = {
            "total": total,
            "passed": passed,
            "failed": failed,
            "mean_latency_ms": round(fmean(latencies), 3) if latencies else None,
            "benchmark_metrics": {
                key: (round(fmean(values), 6) if values else None)
                for key, values in benchmark_metrics.items()
            },
            "scenarios": scenario_counts,
        }

    return {
        "gate_pass": total_failed == 0,
        "total_entries": len(entries),
        "total_failed": total_failed,
        "models": models,
    }


def render_markdown(summary: dict[str, Any]) -> str:
    """Render a concise Markdown summary report."""
    header = [
        "# Real-Data E2E Summary",
        "",
        f"- Gate pass: **{summary.get('gate_pass')}**",
        f"- Total entries: **{summary.get('total_entries')}**",
        f"- Total failed: **{summary.get('total_failed')}**",
        "",
        "| Model | Pass/Total | Mean Latency (ms) | MAE | RMSE | SMAPE |",
        "| --- | --- | --- | --- | --- | --- |",
    ]

    rows: list[str] = []
    models = summary.get("models")
    if isinstance(models, dict):
        for model, payload in models.items():
            if not isinstance(payload, dict):
                continue
            metrics = payload.get("benchmark_metrics", {})
            if not isinstance(metrics, dict):
                metrics = {}
            rows.append(
                "| {model} | {passed}/{total} | {latency} | {mae} | {rmse} | {smape} |".format(
                    model=model,
                    passed=payload.get("passed", 0),
                    total=payload.get("total", 0),
                    latency=_fmt_number(payload.get("mean_latency_ms")),
                    mae=_fmt_number(metrics.get("mae")),
                    rmse=_fmt_number(metrics.get("rmse")),
                    smape=_fmt_number(metrics.get("smape")),
                )
            )

    if not rows:
        rows.append("| (none) | 0/0 | - | - | - | - |")

    return "\n".join([*header, *rows, ""])


def _fmt_number(value: Any) -> str:
    if isinstance(value, bool):
        return "-"
    if isinstance(value, (int, float)):
        return f"{float(value):.6f}".rstrip("0").rstrip(".")
    return "-"


def load_entry_list(paths: list[Path]) -> list[dict[str, Any]]:
    """Load and merge result entries from one or more result JSON files."""
    merged: list[dict[str, Any]] = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            entries = payload.get("entries")
            if isinstance(entries, list):
                merged.extend(item for item in entries if isinstance(item, dict))
                continue
        if isinstance(payload, list):
            merged.extend(item for item in payload if isinstance(item, dict))
    return merged


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize real-data E2E result artifacts.")
    parser.add_argument(
        "--result-path",
        action="append",
        default=[],
        help="Path to a result.json file. Can be repeated.",
    )
    parser.add_argument(
        "--input-glob",
        default="",
        help="Glob expression resolving to result.json files.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where summary.json and summary.md will be written.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    result_paths = [Path(item) for item in args.result_path]
    if args.input_glob:
        result_paths.extend(Path(item) for item in glob.glob(args.input_glob, recursive=True))

    deduped_paths = sorted({path.resolve() for path in result_paths})
    if not deduped_paths:
        print("no input result files found")
        return 2

    entries = load_entry_list(deduped_paths)
    summary = summarize_entries(entries)
    markdown = render_markdown(summary)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_dir / "summary.md").write_text(markdown, encoding="utf-8")

    print(f"wrote summary for {len(entries)} entries to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
