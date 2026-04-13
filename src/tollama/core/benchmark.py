"""Benchmark engine for evaluating model performance on datasets.

Runs all (or selected) models against a dataset with known actuals,
collects per-model MASE/CRPS/latency, and produces a summary table
for comparison and regression detection.
"""

from __future__ import annotations

import csv
import json
import logging
import math
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .backtest import (
    ModelFoldResult,
    aggregate_backtest_results,
    build_fold_request,
    compute_dataset_fingerprint,
    derive_ensemble_weights,
    evaluate_fold,
    generate_folds,
)
from .schemas import ForecastResponse, SeriesInput

logger = logging.getLogger(__name__)

_QUALITY_METRIC_PRIORITY = (
    "mase",
    "smape",
    "mae",
    "rmse",
    "rmsse",
    "wape",
    "mape",
    "pinball",
)


@dataclass(frozen=True)
class ModelBenchmarkResult:
    """Performance summary for one model on the benchmark dataset."""

    model: str
    metrics: dict[str, float]
    latency_ms: float
    folds_evaluated: int
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class BenchmarkSummary:
    """Full benchmark output across all models."""

    dataset_fingerprint: str
    horizon: int
    num_folds: int
    models: list[ModelBenchmarkResult]
    learned_weights: dict[str, float]
    metric_names: list[str]


def run_benchmark(
    *,
    series: list[SeriesInput],
    models: list[str],
    horizon: int,
    num_folds: int = 3,
    forecast_fn: Any,
    metric_names: list[str] | None = None,
    quantiles: list[float] | None = None,
) -> BenchmarkSummary:
    """Run a benchmark evaluation across models.

    Parameters
    ----------
    series:
        Dataset with actuals in the ``target`` field.
    models:
        List of model names to evaluate.
    horizon:
        Forecast horizon.
    num_folds:
        Number of walk-forward folds.
    forecast_fn:
        Callable ``(request) -> ForecastResponse`` that runs a forecast.
    metric_names:
        Metrics to compute. Defaults to ``["mase", "mae", "rmse"]``.
    quantiles:
        Optional quantiles to request.
    """
    if metric_names is None:
        metric_names = ["mase", "mae", "rmse"]

    fingerprint = compute_dataset_fingerprint(series)

    # Use first series length as representative for fold generation
    representative_length = max(len(s.target) for s in series)
    folds = generate_folds(
        series_length=representative_length,
        horizon=horizon,
        num_folds=num_folds,
    )

    if not folds:
        raise ValueError(
            f"insufficient data for backtesting: need at least "
            f"{horizon * 2 + horizon} points, got {representative_length}"
        )

    all_fold_results: list[ModelFoldResult] = []
    model_results: list[ModelBenchmarkResult] = []

    for model in models:
        model_warnings: list[str] = []
        model_started = time.perf_counter()
        folds_evaluated = 0

        for fold in folds:
            request = build_fold_request(
                series=series,
                fold=fold,
                model=model,
                horizon=horizon,
                quantiles=quantiles,
            )

            try:
                response = forecast_fn(request)
                if not isinstance(response, ForecastResponse):
                    response = ForecastResponse.model_validate(response)
            except Exception as exc:  # noqa: BLE001
                model_warnings.append(f"fold {fold.fold_index}: {type(exc).__name__}: {exc}")
                continue

            fold_result = evaluate_fold(
                request=request,
                response=response,
                fold_index=fold.fold_index,
            )
            all_fold_results.append(fold_result)
            model_warnings.extend(fold_result.warnings)
            folds_evaluated += 1

        model_latency = (time.perf_counter() - model_started) * 1000.0

        # Compute aggregate metrics for this model
        model_fold_results = [r for r in all_fold_results if r.model == model]
        if model_fold_results:
            agg = aggregate_backtest_results(
                fold_results=model_fold_results,
                metric_names=metric_names,
            )
            model_metrics = agg.get(model, {})
        else:
            model_metrics = {}

        model_results.append(
            ModelBenchmarkResult(
                model=model,
                metrics=model_metrics,
                latency_ms=model_latency,
                folds_evaluated=folds_evaluated,
                warnings=model_warnings,
            )
        )

    # Derive ensemble weights from all results
    full_aggregate = aggregate_backtest_results(
        fold_results=all_fold_results,
        metric_names=metric_names,
    )
    learned_weights = derive_ensemble_weights(full_aggregate)

    return BenchmarkSummary(
        dataset_fingerprint=fingerprint,
        horizon=horizon,
        num_folds=len(folds),
        models=model_results,
        learned_weights=learned_weights,
        metric_names=metric_names,
    )


def format_benchmark_table(summary: BenchmarkSummary) -> str:
    """Format benchmark results as a human-readable table."""
    if not summary.models:
        return "No models evaluated."

    # Build header
    metric_cols = summary.metric_names
    header = ["Model", *[m.upper() for m in metric_cols], "Latency(ms)", "Folds", "Weight"]
    rows: list[list[str]] = []

    for result in sorted(summary.models, key=lambda r: r.metrics.get("mase", float("inf"))):
        row = [result.model]
        for metric in metric_cols:
            val = result.metrics.get(metric)
            row.append(f"{val:.4f}" if val is not None else "N/A")
        row.append(f"{result.latency_ms:.0f}")
        row.append(str(result.folds_evaluated))
        weight = summary.learned_weights.get(result.model)
        row.append(f"{weight:.4f}" if weight is not None else "N/A")
        rows.append(row)

    # Compute column widths
    col_widths = [len(h) for h in header]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(cell))

    # Format
    sep = "+" + "+".join("-" * (w + 2) for w in col_widths) + "+"
    lines = [sep]
    lines.append("| " + " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(header)) + " |")
    lines.append(sep)
    for row in rows:
        lines.append("| " + " | ".join(c.ljust(col_widths[i]) for i, c in enumerate(row)) + " |")
    lines.append(sep)
    return "\n".join(lines)


def save_benchmark_results(
    summary: BenchmarkSummary,
    output_dir: Path,
) -> Path:
    """Save benchmark results to a JSON file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"benchmark_{summary.dataset_fingerprint}.json"
    payload = {
        "fingerprint": summary.dataset_fingerprint,
        "horizon": summary.horizon,
        "num_folds": summary.num_folds,
        "metric_names": summary.metric_names,
        "models": [
            {
                "model": r.model,
                "metrics": r.metrics,
                "latency_ms": r.latency_ms,
                "folds_evaluated": r.folds_evaluated,
                "warnings": r.warnings,
            }
            for r in summary.models
        ],
        "learned_weights": summary.learned_weights,
    }
    path.write_text(json.dumps(payload, indent=2))
    logger.info("saved benchmark results to %s", path)
    return path


def recommend_routing(summary: BenchmarkSummary) -> dict[str, Any]:
    """Derive benchmark-backed routing defaults from a benchmark summary."""
    quality_metric_names = _ordered_quality_metric_names(summary.metric_names)
    successful = [
        result
        for result in summary.models
        if result.folds_evaluated > 0 and _has_quality_metrics(result, quality_metric_names)
    ]

    if not successful:
        return {
            "default": None,
            "fast_path": None,
            "high_accuracy": None,
            "policy": "no successful benchmark runs",
            "ranking": [],
            "caveats": [
                "All benchmarked models failed or produced no comparable quality metrics.",
                "Keep existing routing defaults until a successful benchmark run is available.",
            ],
        }

    quality_sorted = sorted(
        successful,
        key=lambda result: (
            _quality_sort_key(result, quality_metric_names),
            result.latency_ms,
            result.model,
        ),
    )
    latency_sorted = sorted(
        successful,
        key=lambda result: (
            result.latency_ms,
            _quality_sort_key(result, quality_metric_names),
            result.model,
        ),
    )

    quality_rank = {result.model: index for index, result in enumerate(quality_sorted, start=1)}
    latency_rank = {result.model: index for index, result in enumerate(latency_sorted, start=1)}

    ranking = [
        {
            "model": result.model,
            "quality_rank": quality_rank[result.model],
            "latency_rank": latency_rank[result.model],
            "balanced_score": round(
                0.7 * quality_rank[result.model] + 0.3 * latency_rank[result.model],
                4,
            ),
            "primary_metric": quality_metric_names[0],
            "primary_metric_value": result.metrics.get(quality_metric_names[0]),
            "latency_ms": round(result.latency_ms, 4),
        }
        for result in successful
    ]
    balanced_sorted = sorted(
        ranking,
        key=lambda item: (
            item["balanced_score"],
            item["quality_rank"],
            item["latency_rank"],
            item["model"],
        ),
    )

    return {
        "default": balanced_sorted[0]["model"],
        "fast_path": latency_sorted[0].model,
        "high_accuracy": quality_sorted[0].model,
        "policy": (
            "Use default for general workloads; route latency-sensitive requests to "
            "fast_path; route accuracy-critical requests to high_accuracy."
        ),
        "ranking": balanced_sorted,
        "caveats": [
            "Routing recommendation is only as good as the benchmark dataset and fold design.",
            "Latency can shift with hardware, daemon warm-up, and runner cache state.",
        ],
    }


def build_benchmark_result_payload(
    summary: BenchmarkSummary,
    *,
    generated_at: str,
    run_id: str,
) -> dict[str, Any]:
    """Build the canonical Core benchmark result payload."""
    routing_recommendation = recommend_routing(summary)
    routing_rationale = build_routing_rationale(summary)
    legacy_filename = f"benchmark_{summary.dataset_fingerprint}.json"
    forecast_id = _build_forecast_selection_id(
        run_id=run_id,
        default_model=routing_recommendation.get("default"),
    )

    return {
        "artifact_kind": "tollama_core_benchmark",
        "schema_version": 1,
        "generated_at": generated_at,
        "run_id": run_id,
        "eval_ref": run_id,
        "forecast_id": forecast_id,
        "source": "tollama.core.benchmark",
        "dataset_fingerprint": summary.dataset_fingerprint,
        "horizon": summary.horizon,
        "num_folds": summary.num_folds,
        "metric_names": summary.metric_names,
        "quality_metric_priority": _ordered_quality_metric_names(summary.metric_names),
        "models": [
            {
                "model": result.model,
                "metrics": result.metrics,
                "latency_ms": result.latency_ms,
                "folds_evaluated": result.folds_evaluated,
                "warnings": result.warnings,
                "learned_weight": summary.learned_weights.get(result.model),
            }
            for result in summary.models
        ],
        "leaderboard": build_benchmark_leaderboard(summary),
        "learned_weights": summary.learned_weights,
        "routing_recommendation": {
            **routing_recommendation,
            "rationale": routing_rationale,
        },
        "preprocessing_metadata": {
            "available": False,
            "source": "benchmark_input",
            "note": (
                "No structured preprocessing metadata was attached to this Core benchmark run."
            ),
        },
        "routing_rationale": routing_rationale,
        "artifact_mapping": {
            "result_json": "result.json",
            "routing_manifest": "routing.json",
            "leaderboard_csv": "leaderboard.csv",
            "operator_summary_md": "summary.md",
            "legacy_summary_json": legacy_filename,
            "rich_eval_artifacts": (
                "Use tollama-eval for results.json, details.json, and report.html."
            ),
        },
    }


def build_routing_manifest_payload(
    summary: BenchmarkSummary,
    *,
    generated_at: str,
    run_id: str,
) -> dict[str, Any]:
    """Build a routing manifest from a benchmark summary."""
    routing = recommend_routing(summary)
    routing_rationale = build_routing_rationale(summary)
    return {
        "version": 1,
        "generated_at": generated_at,
        "run_id": run_id,
        "eval_ref": run_id,
        "forecast_id": _build_forecast_selection_id(
            run_id=run_id,
            default_model=routing.get("default"),
        ),
        "source": "tollama.core.benchmark",
        "routing": {
            "default": routing.get("default"),
            "fast_path": routing.get("fast_path"),
            "high_accuracy": routing.get("high_accuracy"),
        },
        "policy": routing.get("policy"),
        "caveats": list(routing.get("caveats", [])),
        "preprocessing_metadata": {
            "available": False,
            "source": "benchmark_input",
            "note": (
                "No structured preprocessing metadata was attached to this Core benchmark run."
            ),
        },
        "routing_rationale": routing_rationale,
    }


def build_operator_summary(summary: BenchmarkSummary) -> dict[str, Any]:
    """Build a deterministic operator-facing recommendation summary."""
    routing = recommend_routing(summary)
    primary_metric = _ordered_quality_metric_names(summary.metric_names)[0]
    by_model = {result.model: result for result in summary.models}
    ranking = {
        item["model"]: item
        for item in routing.get("ranking", [])
        if isinstance(item, dict) and isinstance(item.get("model"), str)
    }

    def _lane_summary(lane: str, model_name: str | None) -> dict[str, Any]:
        if model_name is None:
            return {
                "lane": lane,
                "model": None,
                "reason": "No successful benchmark recommendation is available yet.",
                "latency_ms": None,
                "primary_metric": primary_metric,
                "primary_metric_value": None,
                "quality_rank": None,
                "latency_rank": None,
            }

        result = by_model.get(model_name)
        ranking_item = ranking.get(model_name, {})
        latency_ms = round(result.latency_ms, 4) if result is not None else None
        metric_value = result.metrics.get(primary_metric) if result is not None else None
        if lane == "default":
            reason = "Best balanced benchmark profile for general workloads."
        elif lane == "fast_path":
            reason = "Lowest observed latency among successful benchmark runs."
        else:
            reason = f"Best {primary_metric.upper()} among successful benchmark runs."
        return {
            "lane": lane,
            "model": model_name,
            "reason": reason,
            "latency_ms": latency_ms,
            "primary_metric": primary_metric,
            "primary_metric_value": metric_value,
            "quality_rank": ranking_item.get("quality_rank"),
            "latency_rank": ranking_item.get("latency_rank"),
        }

    return {
        "primary_metric": primary_metric,
        "policy": routing.get("policy"),
        "caveats": list(routing.get("caveats", [])),
        "default": _lane_summary("default", routing.get("default")),
        "fast_path": _lane_summary("fast_path", routing.get("fast_path")),
        "high_accuracy": _lane_summary("high_accuracy", routing.get("high_accuracy")),
    }


def build_routing_rationale(summary: BenchmarkSummary) -> dict[str, dict[str, Any]]:
    """Build a compact lane-by-lane rationale for downstream handoff."""
    payload = build_operator_summary(summary)
    rationale: dict[str, dict[str, Any]] = {}
    for lane_name in ("default", "fast_path", "high_accuracy"):
        entry = payload[lane_name]
        rationale[lane_name] = {
            "model": entry.get("model"),
            "reason": entry.get("reason"),
            "primary_metric": entry.get("primary_metric"),
            "primary_metric_value": entry.get("primary_metric_value"),
            "latency_ms": entry.get("latency_ms"),
            "quality_rank": entry.get("quality_rank"),
            "latency_rank": entry.get("latency_rank"),
        }
    return rationale


def _build_forecast_selection_id(*, run_id: str, default_model: Any) -> str:
    model_name = default_model if isinstance(default_model, str) and default_model else "none"
    return f"core-routing-candidate:{run_id}:{model_name}"


def format_operator_summary(summary: BenchmarkSummary) -> str:
    """Format an operator-facing summary for terminal output."""
    payload = build_operator_summary(summary)

    def _metric_text(entry: dict[str, Any]) -> str:
        value = entry.get("primary_metric_value")
        metric_name = str(entry.get("primary_metric", "metric")).upper()
        if value is None or not isinstance(value, (int, float)) or not math.isfinite(value):
            return f"{metric_name}=N/A"
        return f"{metric_name}={value:.4f}"

    def _latency_text(entry: dict[str, Any]) -> str:
        value = entry.get("latency_ms")
        if value is None or not isinstance(value, (int, float)) or not math.isfinite(value):
            return "latency=N/A"
        return f"latency={value:.1f}ms"

    lines = ["Recommendation summary:"]
    for lane_name in ("default", "fast_path", "high_accuracy"):
        entry = payload[lane_name]
        model_name = entry.get("model") or "n/a"
        lines.append(
            f"  {lane_name}: {model_name} "
            f"({_metric_text(entry)}; {_latency_text(entry)}; {entry['reason']})"
        )

    caveats = payload.get("caveats", [])
    if caveats:
        lines.append("Caveats:")
        for caveat in caveats:
            lines.append(f"  - {caveat}")

    return "\n".join(lines)


def build_operator_summary_markdown(summary: BenchmarkSummary) -> str:
    """Render the operator-facing recommendation summary as Markdown."""
    payload = build_operator_summary(summary)

    def _metric_text(entry: dict[str, Any]) -> str:
        value = entry.get("primary_metric_value")
        metric_name = str(entry.get("primary_metric", "metric")).upper()
        if value is None or not isinstance(value, (int, float)) or not math.isfinite(value):
            return f"{metric_name}=N/A"
        return f"{metric_name}={value:.4f}"

    def _latency_text(entry: dict[str, Any]) -> str:
        value = entry.get("latency_ms")
        if value is None or not isinstance(value, (int, float)) or not math.isfinite(value):
            return "latency=N/A"
        return f"latency={value:.1f}ms"

    lines = [
        "# Operator Summary",
        "",
        f"- Default lane: `{payload['default']['model'] or 'n/a'}`",
        f"  - {_metric_text(payload['default'])}",
        f"  - {_latency_text(payload['default'])}",
        f"  - {payload['default']['reason']}",
        f"- Fast path: `{payload['fast_path']['model'] or 'n/a'}`",
        f"  - {_metric_text(payload['fast_path'])}",
        f"  - {_latency_text(payload['fast_path'])}",
        f"  - {payload['fast_path']['reason']}",
        f"- High accuracy: `{payload['high_accuracy']['model'] or 'n/a'}`",
        f"  - {_metric_text(payload['high_accuracy'])}",
        f"  - {_latency_text(payload['high_accuracy'])}",
        f"  - {payload['high_accuracy']['reason']}",
        "",
        "## Policy",
        "",
        str(payload.get("policy") or "No routing policy is available."),
    ]

    caveats = payload.get("caveats", [])
    if caveats:
        lines.extend(["", "## Caveats", ""])
        lines.extend(f"- {caveat}" for caveat in caveats)

    return "\n".join(lines) + "\n"


def export_benchmark_leaderboard_csv(
    summary: BenchmarkSummary,
    output_path: Path,
) -> Path:
    """Export a compact leaderboard CSV for the Core benchmark bundle."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    leaderboard = build_benchmark_leaderboard(summary)
    fieldnames = [
        "rank",
        "model",
        *summary.metric_names,
        "latency_ms",
        "folds_evaluated",
        "learned_weight",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in leaderboard:
            writer.writerow({name: row.get(name) for name in fieldnames})
    logger.info("exported benchmark leaderboard CSV to %s", output_path)
    return output_path


def save_benchmark_bundle(
    summary: BenchmarkSummary,
    output_dir: Path,
) -> dict[str, Path]:
    """Persist the canonical Core benchmark bundle plus legacy compatibility output."""
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now(UTC).replace(microsecond=0).isoformat()
    run_id = datetime.now(UTC).strftime(
        f"core-benchmark-{summary.dataset_fingerprint}-%Y%m%dT%H%M%SZ"
    )

    legacy_summary_path = save_benchmark_results(summary, output_dir)
    result_path = output_dir / "result.json"
    routing_path = output_dir / "routing.json"
    leaderboard_path = output_dir / "leaderboard.csv"
    operator_summary_path = output_dir / "summary.md"

    result_payload = build_benchmark_result_payload(
        summary,
        generated_at=generated_at,
        run_id=run_id,
    )
    result_path.write_text(json.dumps(result_payload, indent=2, sort_keys=True), encoding="utf-8")

    routing_payload = build_routing_manifest_payload(
        summary,
        generated_at=generated_at,
        run_id=run_id,
    )
    routing_path.write_text(json.dumps(routing_payload, indent=2, sort_keys=True), encoding="utf-8")

    export_benchmark_leaderboard_csv(summary, leaderboard_path)
    operator_summary_path.write_text(
        build_operator_summary_markdown(summary),
        encoding="utf-8",
    )
    logger.info("saved benchmark bundle to %s", output_dir)
    return {
        "result": result_path,
        "routing_manifest": routing_path,
        "leaderboard": leaderboard_path,
        "operator_summary": operator_summary_path,
        "legacy_summary": legacy_summary_path,
    }


def build_benchmark_leaderboard(summary: BenchmarkSummary) -> list[dict[str, Any]]:
    """Build a quality-first leaderboard for benchmark artifacts."""
    quality_metric_names = _ordered_quality_metric_names(summary.metric_names)
    sorted_models = sorted(
        summary.models,
        key=lambda result: (
            _quality_sort_key(result, quality_metric_names),
            result.latency_ms,
            result.model,
        ),
    )
    leaderboard: list[dict[str, Any]] = []
    for index, result in enumerate(sorted_models, start=1):
        row: dict[str, Any] = {
            "rank": index,
            "model": result.model,
            "latency_ms": round(result.latency_ms, 4),
            "folds_evaluated": result.folds_evaluated,
            "learned_weight": summary.learned_weights.get(result.model),
        }
        for metric_name in summary.metric_names:
            row[metric_name] = result.metrics.get(metric_name)
        leaderboard.append(row)
    return leaderboard


def _ordered_quality_metric_names(metric_names: list[str]) -> list[str]:
    ordered = [name for name in _QUALITY_METRIC_PRIORITY if name in metric_names]
    ordered.extend(name for name in metric_names if name not in ordered)
    return ordered or ["mase"]


def _has_quality_metrics(
    result: ModelBenchmarkResult,
    metric_names: list[str],
) -> bool:
    return any(
        (value := result.metrics.get(name)) is not None and math.isfinite(value)
        for name in metric_names
    )


def _quality_sort_key(
    result: ModelBenchmarkResult,
    metric_names: list[str],
) -> tuple[float, ...]:
    values: list[float] = []
    for name in metric_names:
        value = result.metrics.get(name)
        if value is None or not math.isfinite(value):
            values.append(float("inf"))
        else:
            values.append(float(value))
    return tuple(values)
