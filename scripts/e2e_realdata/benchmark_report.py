"""Detailed benchmark reporting for real-data E2E artifacts."""

from __future__ import annotations

import math
import statistics
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    _THIS_DIR = Path(__file__).resolve().parent
    if str(_THIS_DIR) not in sys.path:
        sys.path.append(str(_THIS_DIR))
    import model_policy as model_policy  # noqa: PLC0414
else:
    from . import model_policy


def build_benchmark_report(result_payload: dict[str, Any]) -> dict[str, Any]:
    """Build a detailed benchmark report payload from one result payload."""
    entries = _coerce_entry_list(result_payload.get("entries"))
    benchmark_rows = [
        _benchmark_row(entry)
        for entry in entries
        if entry.get("scenario") == model_policy.BENCHMARK_TARGET_ONLY
    ]
    benchmark_rows.sort(
        key=lambda row: (str(row["model"]), str(row["dataset"]), str(row["series_id"] or "")),
    )

    return {
        "run_id": result_payload.get("run_id"),
        "generated_at": result_payload.get("finished_at"),
        "base_url": result_payload.get("base_url"),
        "gate_profile": result_payload.get("gate_profile"),
        "catalog_path": result_payload.get("catalog_path"),
        "models": list(_coerce_string_list(result_payload.get("models"))),
        "datasets": list(_coerce_string_list(result_payload.get("datasets"))),
        "max_series_per_dataset": result_payload.get("max_series_per_dataset"),
        "scenario_policy": _coerce_mapping(result_payload.get("scenario_policy")),
        "benchmark_metric_names": list(model_policy.BENCHMARK_METRIC_NAMES),
        "primary_ranking_metric": model_policy.PRIMARY_RANKING_METRIC,
        "infra_error": _optional_string(result_payload.get("infra_error")),
        "messages": list(_coerce_string_list(result_payload.get("messages"))),
        "benchmark_rows": benchmark_rows,
        "model_leaderboard": _build_model_leaderboard(
            models=_coerce_string_list(result_payload.get("models")),
            benchmark_rows=benchmark_rows,
        ),
        "dataset_breakdown": _build_dataset_breakdown(
            datasets=_coerce_string_list(result_payload.get("datasets")),
            benchmark_rows=benchmark_rows,
        ),
        "failure_summary": _build_failure_summary(entries=entries, benchmark_rows=benchmark_rows),
        "contract_summary": _build_contract_summary(entries=entries),
    }


def render_markdown(report: dict[str, Any]) -> str:
    """Render a human-readable Markdown benchmark report."""
    lines = [
        "# HF Real-Data Benchmark Report",
        "",
        f"- run_id: `{report.get('run_id')}`",
        f"- generated_at: `{report.get('generated_at')}`",
        f"- base_url: `{report.get('base_url')}`",
        f"- gate_profile: `{report.get('gate_profile')}`",
        f"- catalog_path: `{report.get('catalog_path')}`",
        f"- max_series_per_dataset: `{report.get('max_series_per_dataset')}`",
        f"- primary_ranking_metric: `{report.get('primary_ranking_metric')}`",
        "",
        f"- models: {', '.join(_coerce_string_list(report.get('models')))}",
        f"- datasets: {', '.join(_coerce_string_list(report.get('datasets')))}",
    ]

    infra_error = _optional_string(report.get("infra_error"))
    if infra_error:
        lines.extend(["", f"- infra_error: **{infra_error}**"])

    lines.extend(
        [
            "",
            "## Model Leaderboard",
            "",
            "| Rank | Model | Pass/Rows | Fail | Skip | Success Rate | "
            "Mean Latency (ms) | P50 (ms) | P95 (ms) | MAE | RMSE | SMAPE | MAPE | MASE |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    leaderboard = report.get("model_leaderboard")
    if isinstance(leaderboard, list) and leaderboard:
        for row in leaderboard:
            if not isinstance(row, dict):
                continue
            metrics = _coerce_mapping(row.get("mean_metrics"))
            lines.append(
                (
                    "| {rank} | {model} | {passed}/{rows} | {failed} | {skipped} | "
                    "{success_rate} | {mean_latency} | {p50} | {p95} | {mae} | "
                    "{rmse} | {smape} | {mape} | {mase} |"
                ).format(
                    rank=row.get("rank") if row.get("rank") is not None else "-",
                    model=row.get("model"),
                    passed=row.get("passed", 0),
                    rows=row.get("rows", 0),
                    failed=row.get("failed", 0),
                    skipped=row.get("skipped", 0),
                    success_rate=_fmt_number(row.get("success_rate")),
                    mean_latency=_fmt_number(row.get("mean_latency_ms")),
                    p50=_fmt_number(row.get("latency_p50_ms")),
                    p95=_fmt_number(row.get("latency_p95_ms")),
                    mae=_fmt_number(metrics.get("mae")),
                    rmse=_fmt_number(metrics.get("rmse")),
                    smape=_fmt_number(metrics.get("smape")),
                    mape=_fmt_number(metrics.get("mape")),
                    mase=_fmt_number(metrics.get("mase")),
                )
            )
    else:
        lines.append("| - | (none) | 0/0 | 0 | 0 | - | - | - | - | - | - | - | - | - |")

    lines.extend(
        [
            "",
            "## Dataset Breakdown",
            "",
            "| Dataset | Rows | Success Rate | Mean Latency (ms) | MAE | RMSE | SMAPE | "
            "MAPE | MASE | Failure Classes |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    breakdown = report.get("dataset_breakdown")
    if isinstance(breakdown, list) and breakdown:
        for row in breakdown:
            if not isinstance(row, dict):
                continue
            metrics = _coerce_mapping(row.get("mean_metrics"))
            failure_counts = _coerce_mapping(row.get("failure_classification_counts"))
            failure_text = (
                ", ".join(f"{key}:{value}" for key, value in sorted(failure_counts.items())) or "-"
            )
            lines.append(
                "| {dataset} | {rows} | {success_rate} | {latency} | {mae} | {rmse} | {smape} | "
                "{mape} | {mase} | {failures} |".format(
                    dataset=row.get("dataset"),
                    rows=row.get("rows", 0),
                    success_rate=_fmt_number(row.get("success_rate")),
                    latency=_fmt_number(row.get("mean_latency_ms")),
                    mae=_fmt_number(metrics.get("mae")),
                    rmse=_fmt_number(metrics.get("rmse")),
                    smape=_fmt_number(metrics.get("smape")),
                    mape=_fmt_number(metrics.get("mape")),
                    mase=_fmt_number(metrics.get("mase")),
                    failures=failure_text,
                )
            )
    else:
        lines.append("| (none) | 0 | - | - | - | - | - | - | - | - |")

    lines.extend(
        [
            "",
            "## Benchmark Rows",
            "",
            "| Model | Dataset | Series ID | Status | Error Class | Latency (ms) | MAE | RMSE | "
            "SMAPE | MAPE | MASE | Error |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    benchmark_rows = report.get("benchmark_rows")
    if isinstance(benchmark_rows, list) and benchmark_rows:
        for row in benchmark_rows:
            if not isinstance(row, dict):
                continue
            lines.append(
                "| {model} | {dataset} | {series_id} | {status} | {error_class} | {latency} | "
                "{mae} | {rmse} | {smape} | {mape} | {mase} | {error} |".format(
                    model=row.get("model"),
                    dataset=row.get("dataset"),
                    series_id=row.get("series_id") or "-",
                    status=row.get("status"),
                    error_class=row.get("error_classification") or "-",
                    latency=_fmt_number(row.get("latency_ms")),
                    mae=_fmt_number(row.get("mae")),
                    rmse=_fmt_number(row.get("rmse")),
                    smape=_fmt_number(row.get("smape")),
                    mape=_fmt_number(row.get("mape")),
                    mase=_fmt_number(row.get("mase")),
                    error=(row.get("error") or "").replace("|", "/"),
                )
            )
    else:
        lines.append("| (none) | - | - | - | - | - | - | - | - | - | - | - |")

    lines.extend(
        [
            "",
            "## Contract Summary",
            "",
            "| Scenario | Pass | Fail | Skip | Total |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    contract_summary = _coerce_mapping(report.get("contract_summary"))
    if contract_summary:
        for scenario in (model_policy.CONTRACT_BEST_EFFORT, model_policy.CONTRACT_STRICT):
            row = _coerce_mapping(contract_summary.get(scenario))
            lines.append(
                "| {scenario} | {pass_count} | {fail_count} | {skip_count} | {total} |".format(
                    scenario=scenario,
                    pass_count=row.get("pass", 0),
                    fail_count=row.get("fail", 0),
                    skip_count=row.get("skip", 0),
                    total=row.get("total", 0),
                )
            )
    else:
        lines.append("| (none) | 0 | 0 | 0 | 0 |")

    lines.extend(["", "## Failure Classification Summary", ""])
    failure_summary = _coerce_mapping(report.get("failure_summary"))
    benchmark_failures = _coerce_mapping(failure_summary.get("benchmark_by_error_classification"))
    if benchmark_failures:
        lines.append("### Benchmark Rows")
        lines.append("")
        for key, value in sorted(benchmark_failures.items()):
            lines.append(f"- {key}: {value}")
    else:
        lines.extend(["### Benchmark Rows", "", "- none"])

    lines.extend(["", "### All Scenarios", ""])
    by_scenario = _coerce_mapping(failure_summary.get("all_by_scenario"))
    if by_scenario:
        for scenario, counts in sorted(by_scenario.items()):
            row = _coerce_mapping(counts)
            lines.append(
                f"- {scenario}: pass={row.get('pass', 0)} "
                f"fail={row.get('fail', 0)} skip={row.get('skip', 0)}"
            )
    else:
        lines.append("- none")

    return "\n".join(lines) + "\n"


def _benchmark_row(entry: dict[str, Any]) -> dict[str, Any]:
    metrics = _coerce_mapping(entry.get("metrics"))
    error = _optional_string(entry.get("error"))
    return {
        "model": entry.get("model"),
        "dataset": entry.get("dataset"),
        "series_id": entry.get("series_id"),
        "status": entry.get("status"),
        "error": error,
        "error_classification": _classify_error(
            error=error,
            http_status=entry.get("http_status"),
            status=entry.get("status"),
        ),
        "latency_ms": entry.get("latency_ms"),
        "retry_count": entry.get("retry_count"),
        "warnings": list(_coerce_string_list(entry.get("warnings"))),
        "mae": _maybe_number(metrics.get("mae")),
        "rmse": _maybe_number(metrics.get("rmse")),
        "smape": _maybe_number(metrics.get("smape")),
        "mape": _maybe_number(metrics.get("mape")),
        "mase": _maybe_number(metrics.get("mase")),
    }


def _build_model_leaderboard(
    *,
    models: list[str],
    benchmark_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows_by_model: dict[str, list[dict[str, Any]]] = {model: [] for model in models}
    for row in benchmark_rows:
        rows_by_model.setdefault(str(row.get("model")), []).append(row)

    leaderboard: list[dict[str, Any]] = []
    for model in models:
        rows = rows_by_model.get(model, [])
        successful = [row for row in rows if row.get("status") == "pass"]
        latencies = [
            float(row["latency_ms"])
            for row in successful
            if isinstance(row.get("latency_ms"), (int, float))
        ]
        metrics = {
            metric: _mean(_maybe_number(row.get(metric)) for row in successful)
            for metric in model_policy.BENCHMARK_METRIC_NAMES
        }
        leaderboard.append(
            {
                "model": model,
                "rows": len(rows),
                "passed": sum(1 for row in rows if row.get("status") == "pass"),
                "failed": sum(1 for row in rows if row.get("status") == "fail"),
                "skipped": sum(1 for row in rows if row.get("status") == "skip"),
                "success_rate": _success_rate(rows),
                "mean_latency_ms": _mean(latencies),
                "latency_p50_ms": _percentile(latencies, 50.0),
                "latency_p95_ms": _percentile(latencies, 95.0),
                "mean_metrics": metrics,
                "rank": None,
            }
        )

    ranked_models = [
        row
        for row in leaderboard
        if _maybe_number(
            _coerce_mapping(row["mean_metrics"]).get(model_policy.PRIMARY_RANKING_METRIC)
        )
        is not None
    ]
    ranked_models.sort(
        key=lambda row: (
            float(_coerce_mapping(row["mean_metrics"])[model_policy.PRIMARY_RANKING_METRIC]),
            _sortable_latency(row.get("latency_p50_ms")),
            str(row["model"]),
        )
    )
    for index, row in enumerate(ranked_models, start=1):
        row["rank"] = index

    return leaderboard


def _build_dataset_breakdown(
    *,
    datasets: list[str],
    benchmark_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows_by_dataset: dict[str, list[dict[str, Any]]] = {dataset: [] for dataset in datasets}
    for row in benchmark_rows:
        rows_by_dataset.setdefault(str(row.get("dataset")), []).append(row)

    breakdown: list[dict[str, Any]] = []
    for dataset in datasets:
        rows = rows_by_dataset.get(dataset, [])
        successful = [row for row in rows if row.get("status") == "pass"]
        latencies = [
            float(row["latency_ms"])
            for row in successful
            if isinstance(row.get("latency_ms"), (int, float))
        ]
        failure_counts: dict[str, int] = {}
        for row in rows:
            if row.get("status") != "fail":
                continue
            key = str(row.get("error_classification") or "EXECUTION_ERROR")
            failure_counts[key] = failure_counts.get(key, 0) + 1
        breakdown.append(
            {
                "dataset": dataset,
                "rows": len(rows),
                "participating_models": sorted(
                    {str(row.get("model")) for row in rows if isinstance(row.get("model"), str)}
                ),
                "success_rate": _success_rate(rows),
                "mean_latency_ms": _mean(latencies),
                "mean_metrics": {
                    metric: _mean(_maybe_number(row.get(metric)) for row in successful)
                    for metric in model_policy.BENCHMARK_METRIC_NAMES
                },
                "failure_classification_counts": failure_counts,
            }
        )
    return breakdown


def _build_failure_summary(
    *,
    entries: list[dict[str, Any]],
    benchmark_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    benchmark_failures: dict[str, int] = {}
    for row in benchmark_rows:
        if row.get("status") != "fail":
            continue
        key = str(row.get("error_classification") or "EXECUTION_ERROR")
        benchmark_failures[key] = benchmark_failures.get(key, 0) + 1

    by_scenario: dict[str, dict[str, int]] = {}
    for entry in entries:
        scenario = str(entry.get("scenario", "unknown"))
        bucket = by_scenario.setdefault(scenario, {"pass": 0, "fail": 0, "skip": 0})
        status = str(entry.get("status", "fail"))
        if status == "pass":
            bucket["pass"] += 1
        elif status == "skip":
            bucket["skip"] += 1
        else:
            bucket["fail"] += 1

    return {
        "benchmark_by_error_classification": benchmark_failures,
        "all_by_scenario": by_scenario,
    }


def _build_contract_summary(*, entries: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    summary: dict[str, dict[str, int]] = {}
    for scenario in (model_policy.CONTRACT_BEST_EFFORT, model_policy.CONTRACT_STRICT):
        rows = [entry for entry in entries if entry.get("scenario") == scenario]
        summary[scenario] = {
            "pass": sum(1 for row in rows if row.get("status") == "pass"),
            "fail": sum(1 for row in rows if row.get("status") == "fail"),
            "skip": sum(1 for row in rows if row.get("status") == "skip"),
            "total": len(rows),
        }
    return summary


def _classify_error(*, error: str | None, http_status: Any, status: Any) -> str | None:
    if status != "fail":
        return None
    if isinstance(http_status, int):
        if http_status == 400:
            return "BAD_REQUEST"
        if http_status in {502, 503}:
            if error and "dependency_missing" in error.lower():
                return "DEPENDENCY_GATED"
            if error and "missing optional" in error.lower():
                return "DEPENDENCY_GATED"
            return "RUNNER_ERROR"

    text = (error or "").lower()
    if not text:
        return "EXECUTION_ERROR"
    if "dependency_missing" in text or "missing optional" in text:
        return "DEPENDENCY_GATED"
    if "no module named" in text or "not installed" in text:
        return "DEPENDENCY_GATED"
    if "timed out" in text or "timeout" in text:
        return "TIMEOUT"
    if "payload_build_error" in text:
        return "PAYLOAD_BUILD_ERROR"
    if "status mismatch" in text:
        return "CONTRACT_MISMATCH"
    return "EXECUTION_ERROR"


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    data = sorted(float(value) for value in values)
    if len(data) == 1:
        return round(data[0], 3)
    position = (len(data) - 1) * (q / 100.0)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return round(data[lower], 3)
    weight = position - lower
    return round(data[lower] * (1.0 - weight) + data[upper] * weight, 3)


def _mean(values: Any) -> float | None:
    numbers = [float(value) for value in values if isinstance(value, (int, float))]
    if not numbers:
        return None
    return round(float(statistics.fmean(numbers)), 6)


def _success_rate(rows: list[dict[str, Any]]) -> float | None:
    if not rows:
        return None
    return round(sum(1 for row in rows if row.get("status") == "pass") / len(rows), 6)


def _sortable_latency(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    return float("inf")


def _fmt_number(value: Any) -> str:
    if isinstance(value, bool):
        return "-"
    if isinstance(value, (int, float)):
        return f"{float(value):.6f}".rstrip("0").rstrip(".")
    return "-"


def _optional_string(value: Any) -> str | None:
    if isinstance(value, str):
        normalized = value.strip()
        if normalized:
            return normalized
    return None


def _maybe_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _coerce_mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _coerce_entry_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _coerce_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if isinstance(item, str) and item]
