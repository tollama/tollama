"""Core helpers for end-to-end autonomous forecasting pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from .recommend import recommend_models
from .schemas import AnalyzeRequest, AnalyzeResponse, PipelineRequest, SeriesInput
from .series_analysis import analyze_series_request

_CovariatesType = Literal["numeric", "categorical"]


@dataclass(frozen=True, slots=True)
class PipelineInsights:
    """Deterministic analysis + recommendation stage outputs."""

    analysis: AnalyzeResponse
    recommendation: dict[str, Any]
    preferred_model: str | None


def run_pipeline_analysis(payload: PipelineRequest) -> PipelineInsights:
    """Run analyze + recommend stages and choose a preferred model candidate."""
    analyze_response = analyze_series_request(
        AnalyzeRequest(
            series=payload.series,
            parameters=payload.analyze_parameters,
        ),
    )
    recommendation = recommend_models(
        horizon=payload.horizon,
        freq=_recommendation_freq(payload=payload, analysis=analyze_response),
        has_past_covariates=_has_past_covariates(payload.series),
        has_future_covariates=_has_future_covariates(payload.series),
        has_static_covariates=_has_static_covariates(payload.series),
        covariates_type=_covariates_type(payload.series),
        allow_restricted_license=payload.allow_restricted_license,
        top_k=payload.recommend_top_k,
    )
    return PipelineInsights(
        analysis=analyze_response,
        recommendation=recommendation,
        preferred_model=_preferred_model(payload=payload, recommendation=recommendation),
    )


def _recommendation_freq(*, payload: PipelineRequest, analysis: AnalyzeResponse) -> str | None:
    if analysis.results:
        detected = analysis.results[0].detected_frequency.strip()
        if detected and detected.lower() != "auto":
            return detected

    for series in payload.series:
        freq = series.freq.strip()
        if freq and freq.lower() != "auto":
            return freq
    return None


def _has_past_covariates(series_items: list[SeriesInput]) -> bool:
    return any(bool(item.past_covariates) for item in series_items)


def _has_future_covariates(series_items: list[SeriesInput]) -> bool:
    return any(bool(item.future_covariates) for item in series_items)


def _has_static_covariates(series_items: list[SeriesInput]) -> bool:
    return any(bool(item.static_covariates) for item in series_items)


def _covariates_type(series_items: list[SeriesInput]) -> _CovariatesType:
    has_categorical = False
    for series in series_items:
        for covariates in (series.past_covariates or {}, series.future_covariates or {}):
            for values in covariates.values():
                if any(isinstance(value, str) for value in values):
                    has_categorical = True
                    break
            if has_categorical:
                break
        if has_categorical:
            break
    return "categorical" if has_categorical else "numeric"


def _preferred_model(*, payload: PipelineRequest, recommendation: dict[str, Any]) -> str | None:
    if payload.model is not None:
        return payload.model

    recommendations = recommendation.get("recommendations")
    if not isinstance(recommendations, list) or not recommendations:
        return None

    first = recommendations[0]
    if not isinstance(first, dict):
        return None
    candidate = first.get("model")
    if not isinstance(candidate, str) or not candidate.strip():
        return None
    return candidate
