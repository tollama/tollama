"""Export a prepared real-data workload as a Core benchmark input payload.

This bridges the existing real-data harness to the canonical
``tollama benchmark`` CLI by converting prepared series
(``target`` + ``actuals``) into a benchmark-ready payload with one
contiguous ``target`` history.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    _THIS_DIR = Path(__file__).resolve().parent
    if str(_THIS_DIR) not in sys.path:
        sys.path.append(str(_THIS_DIR))
    import prepare_data as prepare_data  # noqa: PLC0414
else:
    from . import prepare_data

DEFAULT_PREFERRED_DATASET = "pjm_hourly_energy"
DEFAULT_FALLBACK_DATASET = "m4_daily"
DEFAULT_OUTPUT = "artifacts/core-solution/benchmark_input.json"
DEFAULT_CACHE_DIR = "/tmp/tollama-e2e-realdata-cache"
RECOMMENDED_MODELS = [
    "chronos2",
    "granite-ttm-r2",
    "timesfm-2.5-200m",
    "moirai-2.0-R-small",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a prepared real-data workload as a Tollama Core benchmark input.",
    )
    parser.add_argument("--mode", choices=("pr", "nightly", "local"), default="local")
    parser.add_argument("--dataset", default=DEFAULT_PREFERRED_DATASET)
    parser.add_argument("--fallback-dataset", default=DEFAULT_FALLBACK_DATASET)
    parser.add_argument(
        "--catalog-path",
        default=str(Path(__file__).resolve().parent / "dataset_catalog.yaml"),
    )
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--context-cap", type=int, default=512)
    parser.add_argument("--timeout-seconds", type=int, default=120)
    parser.add_argument(
        "--allow-kaggle-fallback",
        action="store_true",
        help="Allow local mode to fall back to open datasets when Kaggle credentials are missing.",
    )
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def select_solution_dataset_name(
    datasets: dict[str, list[dict[str, Any]]],
    *,
    preferred: str,
    fallback: str,
) -> tuple[str, bool]:
    """Choose the preferred dataset when available, else a stable fallback."""
    preferred_rows = datasets.get(preferred)
    if preferred_rows:
        return preferred, False

    fallback_rows = datasets.get(fallback)
    if fallback_rows:
        return fallback, True

    for name in sorted(datasets):
        if datasets.get(name):
            return name, name != preferred

    raise RuntimeError("no prepared datasets contained any benchmarkable series")


def prepared_series_to_benchmark_series(series: dict[str, Any]) -> dict[str, Any]:
    """Convert one prepared real-data series into benchmark CLI input format."""
    history = [float(value) for value in series.get("target", [])]
    actuals = [float(value) for value in series.get("actuals", [])]
    timestamps = [str(value) for value in series.get("timestamps", [])]
    freq = str(series.get("freq", "D"))

    if not history:
        raise ValueError(f"prepared series {series.get('id')!r} is missing target history")
    if not actuals:
        raise ValueError(f"prepared series {series.get('id')!r} is missing actuals")
    if len(timestamps) != len(history):
        raise ValueError(
            f"prepared series {series.get('id')!r} has mismatched timestamps and target lengths",
        )

    full_timestamps = extend_timestamps(timestamps, horizon=len(actuals), freq=freq)
    return {
        "id": str(series["id"]),
        "freq": freq,
        "timestamps": full_timestamps,
        "target": history + actuals,
    }


def build_solution_payload(
    *,
    dataset_name: str,
    selected_rows: list[dict[str, Any]],
    preferred_dataset: str,
    fallback_dataset: str,
    fallback_used: bool,
    messages: list[str],
) -> dict[str, Any]:
    """Build the exported benchmark payload plus concrete-solution metadata."""
    if not selected_rows:
        raise ValueError("selected_rows must contain at least one series")

    benchmark_series = [prepared_series_to_benchmark_series(item) for item in selected_rows]
    horizons = {len(item.get("actuals", [])) for item in selected_rows}
    if len(horizons) != 1:
        raise ValueError("all selected series must share the same forecast horizon")

    history_lengths = [len(item["target"]) for item in benchmark_series]
    freq_values = sorted({str(item["freq"]) for item in benchmark_series})

    return {
        "artifact_kind": "tollama_core_benchmark_input",
        "schema_version": 1,
        "generated_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "solution_profile": "hourly_demand_operations",
        "preferred_dataset": preferred_dataset,
        "fallback_dataset": fallback_dataset,
        "selected_dataset": dataset_name,
        "fallback_used": fallback_used,
        "recommended_horizon": horizons.pop(),
        "recommended_models": list(RECOMMENDED_MODELS),
        "dataset_summary": {
            "series_count": len(benchmark_series),
            "freq_values": freq_values,
            "min_history_points": min(history_lengths),
            "max_history_points": max(history_lengths),
        },
        "messages": list(messages),
        "series": benchmark_series,
    }


def extend_timestamps(
    timestamps: list[str],
    *,
    horizon: int,
    freq: str,
) -> list[str]:
    """Extend history timestamps to cover the held-out horizon."""
    if horizon <= 0:
        return list(timestamps)
    if not timestamps:
        return prepare_data._synthetic_timestamps(freq=_normalize_freq(freq), count=horizon)  # noqa: SLF001

    normalized_freq = _normalize_freq(freq)
    try:
        last_timestamp = _parse_timestamp(timestamps[-1])
    except ValueError:
        return prepare_data._synthetic_timestamps(  # noqa: SLF001
            freq=normalized_freq,
            count=len(timestamps) + horizon,
        )

    step = _step_for_freq(normalized_freq)
    future = [
        _render_timestamp(last_timestamp + step * index, freq=normalized_freq)
        for index in range(1, horizon + 1)
    ]
    return list(timestamps) + future


def _normalize_freq(freq: str) -> str:
    normalized = freq.strip().upper()
    if normalized.startswith("H"):
        return "H"
    if normalized.startswith("W"):
        return "W"
    if normalized.startswith("M"):
        return "M"
    return "D"


def _parse_timestamp(raw: str) -> datetime:
    normalized = raw.strip().replace("Z", "+00:00")
    if not normalized:
        raise ValueError("timestamp cannot be empty")
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(UTC).replace(tzinfo=None)
    return parsed


def _step_for_freq(freq: str) -> timedelta:
    if freq == "H":
        return timedelta(hours=1)
    if freq == "W":
        return timedelta(weeks=1)
    if freq == "M":
        # Month support is approximate here because the concrete solution path
        # currently targets hourly and daily workloads.
        return timedelta(days=30)
    return timedelta(days=1)


def _render_timestamp(value: datetime, *, freq: str) -> str:
    if freq == "H":
        return value.replace(microsecond=0).isoformat()
    return value.date().isoformat()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    credentials_present = prepare_data.has_kaggle_credentials()
    policy = prepare_data.kaggle_policy_for_mode(
        args.mode,
        credentials_present,
        allow_local_fallback=args.allow_kaggle_fallback,
    )
    if policy.hard_fail_on_missing and not policy.include_kaggle:
        print(policy.message or "required Kaggle credentials are missing", file=sys.stderr)
        return 1

    try:
        prepared = prepare_data.prepare_datasets(
            mode=args.mode,
            catalog_path=Path(args.catalog_path),
            cache_dir=Path(args.cache_dir),
            include_kaggle=policy.include_kaggle,
            require_kaggle=policy.hard_fail_on_missing,
            seed=int(args.seed),
            context_cap=int(args.context_cap),
            timeout_seconds=max(int(args.timeout_seconds), 30),
        )
    except Exception as exc:  # noqa: BLE001
        print(f"failed to prepare datasets: {exc}", file=sys.stderr)
        return 1

    selected_dataset, fallback_used = select_solution_dataset_name(
        prepared.datasets,
        preferred=str(args.dataset),
        fallback=str(args.fallback_dataset),
    )
    selected_rows = prepared.datasets[selected_dataset]
    all_messages = list(prepared.messages)
    if policy.message:
        all_messages.insert(0, policy.message)

    try:
        payload = build_solution_payload(
            dataset_name=selected_dataset,
            selected_rows=selected_rows,
            preferred_dataset=str(args.dataset),
            fallback_dataset=str(args.fallback_dataset),
            fallback_used=fallback_used,
            messages=all_messages,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"failed to build benchmark payload: {exc}", file=sys.stderr)
        return 1

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    print(f"Wrote Core solution benchmark input to {output_path}")
    print(f"Selected dataset: {selected_dataset}")
    print(f"Series count: {payload['dataset_summary']['series_count']}")
    print(f"Recommended horizon: {payload['recommended_horizon']}")
    print(f"Recommended models: {', '.join(payload['recommended_models'])}")
    if fallback_used:
        print("Dataset fallback was used.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
