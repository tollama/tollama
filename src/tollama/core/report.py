"""Composite report helpers for structured LLM-ready intelligence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from tollama.core.pipeline import run_pipeline_analysis
from tollama.core.schemas import (
    AnalyzeResponse,
    AutoForecastResponse,
    ForecastMetrics,
    ForecastReport,
    PipelineRequest,
    ReportNarrative,
    ReportRequest,
)


@dataclass(frozen=True, slots=True)
class ReportInsights:
    """Deterministic pre-forecast report stage outputs."""

    analysis: AnalyzeResponse
    recommendation: dict[str, Any]
    preferred_model: str | None


def run_report_analysis(payload: ReportRequest) -> ReportInsights:
    """Run analyze + recommend stages for report generation."""
    pipeline_payload = pipeline_request_from_report(payload=payload)
    insights = run_pipeline_analysis(pipeline_payload)
    return ReportInsights(
        analysis=insights.analysis,
        recommendation=insights.recommendation,
        preferred_model=insights.preferred_model,
    )


def pipeline_request_from_report(*, payload: ReportRequest) -> PipelineRequest:
    """Convert a report request into the equivalent pipeline request payload."""
    source = payload.model_dump(mode="python", exclude_none=True)
    source.pop("include_baseline", None)
    return PipelineRequest.model_validate(source)


def build_report_narrative(*, report: ForecastReport) -> ReportNarrative:
    """Build deterministic top-level narrative summary for report payloads."""
    chosen_model = report.forecast.selection.chosen_model
    anomaly_count = sum(len(item.anomaly_indices) for item in report.analysis.results)

    key_insights: list[str] = [
        f"Analyzed {len(report.analysis.results)} series.",
        f"Auto-forecast selected model {chosen_model!r}.",
    ]
    metrics = _metrics_summary(report.metrics)
    if metrics is not None:
        key_insights.append(metrics)
    if report.recommendation.get("recommendations"):
        key_insights.append("Recommendation stage returned ranked model candidates.")

    warnings_count = len(report.warnings or [])
    return ReportNarrative(
        summary=(
            f"Composite report selected model {chosen_model!r} with "
            f"{anomaly_count} detected anomalies across analyzed series."
        ),
        chosen_model=chosen_model,
        anomaly_count=anomaly_count,
        key_insights=key_insights,
        warnings_count=warnings_count,
    )


def _metrics_summary(metrics: ForecastMetrics | None) -> str | None:
    if metrics is None:
        return None
    aggregate = metrics.aggregate
    if not aggregate:
        return None
    first_name = sorted(aggregate)[0]
    value = aggregate[first_name]
    return f"Forecast metrics available (example: {first_name}={value:.4f})."


def build_forecast_report(
    *,
    analysis: AnalyzeResponse,
    recommendation: dict[str, Any],
    forecast: AutoForecastResponse,
    include_baseline: bool,
    warnings: list[str] | None,
) -> ForecastReport:
    """Assemble a canonical report payload from computed sections."""
    baseline = forecast.response if include_baseline else None
    return ForecastReport(
        analysis=analysis,
        recommendation=recommendation,
        forecast=forecast,
        baseline=baseline,
        metrics=forecast.response.metrics,
        warnings=warnings or None,
    )
