"""Benchmark engine for evaluating model performance on datasets.

Runs all (or selected) models against a dataset with known actuals,
collects per-model MASE/CRPS/latency, and produces a summary table
for comparison and regression detection.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
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
                model_warnings.append(
                    f"fold {fold.fold_index}: {type(exc).__name__}: {exc}"
                )
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
        model_fold_results = [
            r for r in all_fold_results if r.model == model
        ]
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
    lines.append(
        "| " + " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(header)) + " |"
    )
    lines.append(sep)
    for row in rows:
        lines.append(
            "| " + " | ".join(c.ljust(col_widths[i]) for i, c in enumerate(row)) + " |"
        )
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
