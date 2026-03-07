"""Cross-model TSFM benchmark for quality/latency routing defaults.

This benchmark compares five target models in Tollama:
- lag-llama
- patchtst
- tide
- nhits
- nbeatsx

Protocol is deterministic and reproducible by default:
- synthetic benchmark datasets generated from a fixed seed
- fixed train/test split by horizon
- repeated non-stream forecast calls for latency profiling
- standardized quality metrics and markdown/json artifacts
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

from tollama.client import TollamaClient
from tollama.client.exceptions import TollamaClientError

MODELS = ["lag-llama", "patchtst", "tide", "nhits", "nbeatsx"]
DEFAULT_DATASETS = ["seasonal_daily", "trend_weekly", "intermittent_daily"]


@dataclass(frozen=True, slots=True)
class BenchmarkSeries:
    dataset: str
    freq: str
    context: list[float]
    actuals: list[float]


@dataclass(frozen=True, slots=True)
class ModelRun:
    model: str
    dataset: str
    status: str
    error: str | None
    quality: dict[str, float]
    latency_ms: dict[str, float]
    error_classification: str | None = None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cross-model TSFM benchmark + routing defaults")
    parser.add_argument("--base-url", default="http://127.0.0.1:11435")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--models", default=",".join(MODELS))
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--context-length", type=int, default=96)
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="artifacts/benchmarks/cross_model")
    parser.add_argument(
        "--skip-pull",
        action="store_true",
        help="Skip pull preflight (useful when models are already installed)",
    )
    parser.add_argument(
        "--template-only",
        action="store_true",
        help="Write protocol/report template without contacting daemon",
    )
    return parser.parse_args(argv)


def _generate_series(
    *, dataset: str, context_length: int, horizon: int, seed: int
) -> BenchmarkSeries:
    total = context_length + horizon
    if dataset == "seasonal_daily":
        values = [
            50.0
            + 0.15 * i
            + 10.0 * math.sin((2.0 * math.pi * i) / 7.0)
            + 2.0 * math.cos((2.0 * math.pi * i) / 30.0)
            for i in range(total)
        ]
        freq = "D"
    elif dataset == "trend_weekly":
        values = [30.0 + 0.6 * i + 3.0 * math.sin((2.0 * math.pi * i) / 12.0) for i in range(total)]
        freq = "W"
    elif dataset == "intermittent_daily":
        # Deterministic pseudo-intermittent pattern (no randomness dependency in output shape).
        values = []
        for i in range(total):
            base = 12.0 + 0.05 * i
            burst = 14.0 if (i + seed) % 17 in {0, 1, 2} else 0.0
            valley = -6.0 if (i + seed) % 19 == 0 else 0.0
            values.append(max(0.0, base + burst + valley))
        freq = "D"
    else:
        raise ValueError(f"unsupported dataset: {dataset}")

    return BenchmarkSeries(
        dataset=dataset,
        freq=freq,
        context=[float(v) for v in values[:context_length]],
        actuals=[float(v) for v in values[context_length:]],
    )


def _mae(actual: list[float], pred: list[float]) -> float:
    return statistics.fmean(abs(a - p) for a, p in zip(actual, pred))


def _rmse(actual: list[float], pred: list[float]) -> float:
    return math.sqrt(statistics.fmean((a - p) ** 2 for a, p in zip(actual, pred)))


def _mape(actual: list[float], pred: list[float]) -> float:
    eps = 1e-9
    return 100.0 * statistics.fmean(abs((a - p) / max(abs(a), eps)) for a, p in zip(actual, pred))


def _smape(actual: list[float], pred: list[float]) -> float:
    eps = 1e-9
    return 100.0 * statistics.fmean(
        (2.0 * abs(a - p)) / max(abs(a) + abs(p), eps) for a, p in zip(actual, pred)
    )


def _mase(
    actual: list[float], pred: list[float], insample: list[float], seasonality: int = 1
) -> float:
    if len(insample) <= seasonality:
        return float("nan")
    naive_errors = [
        abs(insample[i] - insample[i - seasonality]) for i in range(seasonality, len(insample))
    ]
    denom = statistics.fmean(naive_errors) if naive_errors else 0.0
    if denom <= 1e-12:
        return float("nan")
    return _mae(actual, pred) / denom


def _extract_mean_forecast(payload: dict[str, Any], expected_horizon: int) -> list[float]:
    forecasts = payload.get("forecasts")
    if not isinstance(forecasts, list) or not forecasts:
        raise ValueError("response missing forecasts[]")
    first = forecasts[0]
    if not isinstance(first, dict):
        raise ValueError("response forecasts[0] is not object")
    mean = first.get("mean")
    if not isinstance(mean, list):
        raise ValueError("response forecasts[0].mean missing")
    values = [float(v) for v in mean]
    if len(values) < expected_horizon:
        raise ValueError(
            f"forecast shorter than horizon: len={len(values)} horizon={expected_horizon}"
        )
    return values[:expected_horizon]


def _score_run(*, quality: dict[str, float], p50_ms: float) -> tuple[float, float]:
    # Lower-is-better unified quality score + latency cost.
    quality_score = quality["smape"] + 10.0 * quality["mase"]
    latency_score = p50_ms / 1000.0
    return quality_score, latency_score


def _recommend_routing(runs: list[ModelRun]) -> dict[str, Any]:
    successful = [run for run in runs if run.status == "pass"]
    if not successful:
        return {
            "default": None,
            "fast_path": None,
            "high_accuracy": None,
            "policy": "no successful benchmark runs",
            "caveats": ["all model runs failed; keep existing routing unchanged"],
        }

    per_model: dict[str, dict[str, list[float]]] = {}
    for run in successful:
        bucket = per_model.setdefault(run.model, {"quality": [], "latency": []})
        quality_score, latency_score = _score_run(
            quality=run.quality,
            p50_ms=float(run.latency_ms["p50"]),
        )
        bucket["quality"].append(quality_score)
        bucket["latency"].append(latency_score)

    ranked = []
    for model, scores in per_model.items():
        ranked.append(
            {
                "model": model,
                "quality_score": statistics.fmean(scores["quality"]),
                "latency_score": statistics.fmean(scores["latency"]),
            }
        )

    quality_sorted = sorted(
        ranked, key=lambda item: (item["quality_score"], item["latency_score"], item["model"])
    )
    latency_sorted = sorted(
        ranked, key=lambda item: (item["latency_score"], item["quality_score"], item["model"])
    )

    # Balanced objective with stronger quality weight.
    balanced_sorted = sorted(
        ranked,
        key=lambda item: (0.7 * item["quality_score"] + 0.3 * item["latency_score"], item["model"]),
    )

    default_model = balanced_sorted[0]["model"]
    fast_model = latency_sorted[0]["model"]
    accurate_model = quality_sorted[0]["model"]

    return {
        "default": default_model,
        "fast_path": fast_model,
        "high_accuracy": accurate_model,
        "policy": (
            "Use default for general workloads; route latency-sensitive requests to fast_path; "
            "route mission-critical accuracy requests to high_accuracy."
        ),
        "ranking": balanced_sorted,
        "caveats": [
            (
                "Benchmark uses deterministic synthetic datasets by default; "
                "validate against production data."
            ),
            "Latency depends on hardware/runtime bootstrap state and may shift after warm-up.",
        ],
    }


def _classify_run_error(error: str) -> str:
    text = error.lower()
    if "dependency_missing" in text:
        return "DEPENDENCY_GATED"
    if "no module named" in text or "not installed" in text:
        return "DEPENDENCY_GATED"
    if "runner family" in text and "is not supported" in text:
        return "UNSUPPORTED_FAMILY_REGRESSION"
    return "EXECUTION_ERROR"


def _run_single(
    *,
    client: TollamaClient,
    model: str,
    series: BenchmarkSeries,
    horizon: int,
    repeats: int,
) -> ModelRun:
    payload = {
        "model": model,
        "horizon": horizon,
        "series": [
            {
                "id": f"{series.dataset}-s1",
                "freq": series.freq,
                "timestamps": [str(i + 1) for i in range(len(series.context))],
                "target": series.context,
            }
        ],
        "options": {},
    }

    latencies: list[float] = []
    latest_payload: dict[str, Any] | None = None

    try:
        for _ in range(max(1, repeats)):
            started = perf_counter()
            response = client.forecast(payload, stream=False)
            latencies.append((perf_counter() - started) * 1000.0)
            if not isinstance(response, dict):
                raise ValueError("forecast response is not object")
            latest_payload = response

        assert latest_payload is not None
        prediction = _extract_mean_forecast(latest_payload, expected_horizon=horizon)

        quality = {
            "mae": round(_mae(series.actuals, prediction), 6),
            "rmse": round(_rmse(series.actuals, prediction), 6),
            "mape": round(_mape(series.actuals, prediction), 6),
            "smape": round(_smape(series.actuals, prediction), 6),
            "mase": round(_mase(series.actuals, prediction, series.context, seasonality=1), 6),
        }
        latency = {
            "p50": round(float(statistics.median(latencies)), 3),
            "mean": round(float(statistics.fmean(latencies)), 3),
            "p95": round(float(_percentile(latencies, 95.0)), 3),
        }
        return ModelRun(
            model=model,
            dataset=series.dataset,
            status="pass",
            error=None,
            error_classification=None,
            quality=quality,
            latency_ms=latency,
        )
    except Exception as exc:  # noqa: BLE001
        error = str(exc)
        return ModelRun(
            model=model,
            dataset=series.dataset,
            status="fail",
            error=error,
            error_classification=_classify_run_error(error),
            quality={},
            latency_ms={},
        )


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    data = sorted(float(v) for v in values)
    if len(data) == 1:
        return data[0]
    position = (len(data) - 1) * (q / 100.0)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return data[lower]
    weight = position - lower
    return data[lower] * (1.0 - weight) + data[upper] * weight


def _markdown_report(payload: dict[str, Any]) -> str:
    protocol = payload["protocol"]
    lines = [
        "# Cross-Model TSFM Benchmark Report",
        "",
        f"- run_id: `{payload['run_id']}`",
        f"- generated_at: `{payload['generated_at']}`",
        f"- base_url: `{protocol['base_url']}`",
        f"- models: {', '.join(protocol['models'])}",
        f"- datasets: {', '.join(protocol['datasets'])}",
        f"- split: context={protocol['context_length']}, horizon={protocol['horizon']}",
        f"- repeats per run: {protocol['repeats']}",
        "",
        "## Per-run results",
        "",
        "| model | dataset | status | error class | sMAPE | MASE | "
        "p50 latency (ms) | p95 latency (ms) | error |",
        "|---|---|---|---|---:|---:|---:|---:|---|",
    ]
    for row in payload["runs"]:
        q = row.get("quality", {})
        latency = row.get("latency_ms", {})
        row_template = (
            "| {model} | {dataset} | {status} | {error_class} | {smape} | {mase} | "
            "{p50} | {p95} | {error} |"
        )
        lines.append(
            row_template.format(
                model=row["model"],
                dataset=row["dataset"],
                status=row["status"],
                error_class=row.get("error_classification") or "-",
                smape=q.get("smape", "-"),
                mase=q.get("mase", "-"),
                p50=latency.get("p50", "-"),
                p95=latency.get("p95", "-"),
                error=(row.get("error") or "").replace("|", "/"),
            )
        )

    failure_counts: dict[str, int] = {}
    for row in payload["runs"]:
        if row.get("status") != "fail":
            continue
        key = str(row.get("error_classification") or "EXECUTION_ERROR")
        failure_counts[key] = failure_counts.get(key, 0) + 1

    if failure_counts:
        lines.extend(["", "## Failure classification summary", ""])
        for key, value in sorted(failure_counts.items()):
            lines.append(f"- {key}: {value}")

    routing = payload["routing_recommendation"]
    lines.extend(
        [
            "",
            "## Routing recommendation",
            "",
            f"- default: `{routing.get('default')}`",
            f"- fast_path: `{routing.get('fast_path')}`",
            f"- high_accuracy: `{routing.get('high_accuracy')}`",
            f"- policy: {routing.get('policy')}",
            "",
            "### Caveats",
        ]
    )
    lines.extend(f"- {item}" for item in routing.get("caveats", []))
    return "\n".join(lines) + "\n"


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    models = [item.strip() for item in args.models.split(",") if item.strip()]
    datasets = [item.strip() for item in args.datasets.split(",") if item.strip()]
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")

    protocol = {
        "base_url": args.base_url,
        "models": models,
        "datasets": datasets,
        "context_length": int(args.context_length),
        "horizon": int(args.horizon),
        "metrics": ["mae", "rmse", "mape", "smape", "mase", "latency_p50", "latency_p95"],
        "split": "last horizon points as holdout; remaining as context",
        "repeats": int(args.repeats),
        "seed": int(args.seed),
    }

    if args.template_only:
        template_payload = {
            "run_id": "<template-run-id>",
            "generated_at": "<template-generated-at>",
            "protocol": protocol,
            "runs": [],
            "routing_recommendation": {
                "default": "<to-be-filled>",
                "fast_path": "<to-be-filled>",
                "high_accuracy": "<to-be-filled>",
                "policy": "Populate after benchmark execution.",
                "caveats": [
                    "Template-only run. No daemon/model execution performed.",
                    "Fill with actual results after running with a live tollama daemon.",
                ],
            },
        }
        _write(
            out_dir / "report_template.json", json.dumps(template_payload, indent=2, sort_keys=True)
        )
        _write(out_dir / "report_template.md", _markdown_report(template_payload))
        print(f"template artifacts written: {out_dir}")
        return 0

    client = TollamaClient(base_url=args.base_url, timeout=float(args.timeout))
    try:
        _ = client.health()
    except TollamaClientError as exc:
        print(f"daemon unreachable: {exc}")
        return 2

    if not args.skip_pull:
        for model in models:
            try:
                _ = client.pull_model(name=model, stream=False, accept_license=True)
            except TollamaClientError as exc:
                print(f"warning: pull failed for {model}: {exc}")

    series_list = [
        _generate_series(
            dataset=dataset,
            context_length=int(args.context_length),
            horizon=int(args.horizon),
            seed=int(args.seed),
        )
        for dataset in datasets
    ]

    runs: list[ModelRun] = []
    for model in models:
        for series in series_list:
            runs.append(
                _run_single(
                    client=client,
                    model=model,
                    series=series,
                    horizon=int(args.horizon),
                    repeats=int(args.repeats),
                )
            )

    routing = _recommend_routing(runs)
    payload = {
        "run_id": run_id,
        "generated_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "protocol": protocol,
        "runs": [
            {
                "model": run.model,
                "dataset": run.dataset,
                "status": run.status,
                "error": run.error,
                "error_classification": run.error_classification,
                "quality": run.quality,
                "latency_ms": run.latency_ms,
            }
            for run in runs
        ],
        "routing_recommendation": routing,
    }

    _write(out_dir / "result.json", json.dumps(payload, indent=2, sort_keys=True))
    _write(out_dir / "result.md", _markdown_report(payload))
    print(f"benchmark artifacts written: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
