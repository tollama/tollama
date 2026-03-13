"""Evidence packaging helpers for the v3.8 decision explanation facade."""

from __future__ import annotations

import math

from tollama.core.decision_schemas import (
    DecisionExplanationResponse,
    DecisionPolicyExplanation,
    DecisionPolicyInput,
    ExplanationEvidence,
    ForecastDecomposition,
    InputExplanation,
    ModelCandidateExplanation,
    PlanExplanation,
    PlanSeriesExplanation,
    SignalTrustExplanation,
    SignalTrustInput,
    TemporalImportancePoint,
)
from tollama.core.schemas import (
    AutoSelectionInfo,
    ConfidenceLevel,
    ForecastRequest,
    ForecastResponse,
    SeriesForecast,
    SeriesInput,
    TrendDirection,
)


def build_decision_explanation(
    *,
    request: ForecastRequest,
    response: ForecastResponse,
    selection: AutoSelectionInfo | None = None,
    signal_trust: list[SignalTrustInput] | None = None,
    policy: DecisionPolicyInput | None = None,
) -> DecisionExplanationResponse:
    """Package existing forecast, selection, and policy evidence into one explanation."""
    request_series_by_id = {series.id: series for series in request.series}
    forecast_series_by_id = {series.id: series for series in response.forecasts}

    input_explanation = _build_input_explanation(
        request=request,
        signal_trust=signal_trust or [],
    )

    plan_series: list[PlanSeriesExplanation] = []
    for series_id, request_series in request_series_by_id.items():
        forecast_series = forecast_series_by_id.get(series_id)
        if forecast_series is None:
            continue
        plan_series.append(
            _build_plan_series_explanation(
                request_series=request_series,
                forecast_series=forecast_series,
            )
        )

    plan_explanation = _build_plan_explanation(
        request=request,
        response=response,
        selection=selection,
        series=plan_series,
    )
    decision_policy = _build_policy_explanation(
        policy=policy,
        plan_explanation=plan_explanation,
    )

    return DecisionExplanationResponse(
        input_explanation=input_explanation,
        plan_explanation=plan_explanation,
        decision_policy=decision_policy,
    )


def _build_input_explanation(
    *,
    request: ForecastRequest,
    signal_trust: list[SignalTrustInput],
) -> InputExplanation:
    signal_explanations = [_build_signal_explanation(signal) for signal in signal_trust]

    internal_signals: set[str] = {"internal_ts"}
    for series in request.series:
        if series.past_covariates:
            internal_signals.update(sorted(series.past_covariates))
        if series.future_covariates:
            internal_signals.update(sorted(series.future_covariates))
        if series.static_covariates:
            internal_signals.update(sorted(series.static_covariates))

    signals_used = sorted({*internal_signals, *(signal.name for signal in signal_trust)})

    why_this_input = [
        f"used {len(request.series)} internal time series as the primary evidence base",
    ]
    if signal_trust:
        why_this_input.append(
            f"attached {len(signal_trust)} external probability signal"
            f"{'s' if len(signal_trust) != 1 else ''} with explicit trust scores",
        )
    else:
        why_this_input.append(
            "no external signal trust payload was provided; explanation is internal-data only"
        )

    covariate_count = sum(
        len(series.past_covariates or {})
        + len(series.future_covariates or {})
        + len(series.static_covariates or {})
        for series in request.series
    )
    if covariate_count:
        why_this_input.append(
            f"detected {covariate_count} covariate attachment"
            f"{'s' if covariate_count != 1 else ''} across the request payload",
        )

    return InputExplanation(
        signals_used=signals_used,
        signal_explanations=signal_explanations,
        why_this_input=why_this_input,
    )


def _build_signal_explanation(signal: SignalTrustInput) -> SignalTrustExplanation:
    trust_score = float(signal.trust_score)
    if trust_score >= 0.8:
        level: ConfidenceLevel = "high"
    elif trust_score >= 0.6:
        level = "medium"
    elif trust_score > 0:
        level = "low"
    else:
        level = "unknown"

    evidence: list[ExplanationEvidence] = []
    for metric_name, metric_value in sorted(signal.metrics.items()):
        evidence.append(
            ExplanationEvidence(
                kind="metric",
                label=metric_name,
                value=float(metric_value),
                detail=f"{metric_name}={float(metric_value):.4f}",
                source=signal.source,
            )
        )
    for rationale in signal.rationale:
        evidence.append(
            ExplanationEvidence(
                kind="signal",
                label=signal.name,
                detail=rationale,
                source=signal.source,
            )
        )

    metric_bits = [
        f"{name}={float(value):.4f}"
        for name, value in sorted(signal.metrics.items())
    ]
    detail_suffix = f" ({', '.join(metric_bits)})" if metric_bits else ""
    why_trusted = (
        f"signal {signal.name!r} received trust_score={trust_score:.2f}"
        f"{detail_suffix}"
    )
    return SignalTrustExplanation(
        name=signal.name,
        trust_score=trust_score,
        confidence_level=level,
        why_trusted=why_trusted,
        evidence=evidence,
    )


def _build_plan_explanation(
    *,
    request: ForecastRequest,
    response: ForecastResponse,
    selection: AutoSelectionInfo | None,
    series: list[PlanSeriesExplanation],
) -> PlanExplanation:
    if selection is not None:
        model_selected = selection.chosen_model
        selection_rationale = list(selection.rationale)
        candidates = [
            ModelCandidateExplanation(
                model=item.model,
                rank=item.rank,
                score=float(item.score),
                reasons=list(item.reasons),
            )
            for item in selection.candidates
        ]
        why_this_model = (
            f"auto-selection chose {selection.chosen_model!r} from "
            f"{len(selection.candidates)} candidate models"
        )
    else:
        model_selected = response.model
        selection_rationale = ["response came from an explicit forecast invocation"]
        candidates = [
            ModelCandidateExplanation(
                model=response.model,
                rank=1,
                score=1.0,
                reasons=["selected directly by the caller"],
            )
        ]
        why_this_model = (
            f"forecast response was produced by explicitly requested model {response.model!r}"
        )

    if len(request.series) > 1:
        selection_rationale.append(
            f"plan covers {len(request.series)} series and packages per-series evidence"
        )

    return PlanExplanation(
        model_selected=model_selected,
        why_this_model=why_this_model,
        selection_rationale=selection_rationale,
        candidates=candidates,
        series=series,
    )


def _build_plan_series_explanation(
    *,
    request_series: SeriesInput,
    forecast_series: SeriesForecast,
) -> PlanSeriesExplanation:
    trend_direction = _trend_direction(forecast_series.mean)
    spread_ratio = _spread_ratio(forecast_series)
    confidence_level = _confidence_level_from_spread(spread_ratio)
    decomposition = _forecast_decomposition(
        history=request_series.target,
        forecast=forecast_series.mean,
    )
    temporal_importance = _temporal_importance(
        timestamps=request_series.timestamps,
        history=request_series.target,
    )
    evidence = _series_evidence(
        request_series=request_series,
        forecast_series=forecast_series,
        trend_direction=trend_direction,
        spread_ratio=spread_ratio,
    )
    return PlanSeriesExplanation(
        id=request_series.id,
        trend_direction=trend_direction,
        confidence_level=confidence_level,
        temporal_importance=temporal_importance,
        forecast_decomposition=decomposition,
        evidence=evidence,
    )


def _build_policy_explanation(
    *,
    policy: DecisionPolicyInput | None,
    plan_explanation: PlanExplanation,
) -> DecisionPolicyExplanation:
    if policy is None:
        return DecisionPolicyExplanation(
            auto_executed=False,
            confidence=None,
            threshold=None,
            reason=(
                "no explicit decision policy was supplied; default posture is human review "
                f"for model {plan_explanation.model_selected!r}"
            ),
            human_override=True,
            evidence=[
                ExplanationEvidence(
                    kind="policy",
                    label="policy_mode",
                    detail="human review by default",
                    value="manual_review",
                ),
            ],
        )

    evidence: list[ExplanationEvidence] = []
    if policy.confidence is not None:
        evidence.append(
            ExplanationEvidence(
                kind="policy",
                label="confidence",
                value=float(policy.confidence),
                detail=f"confidence={float(policy.confidence):.2f}",
            )
        )
    if policy.threshold is not None:
        evidence.append(
            ExplanationEvidence(
                kind="policy",
                label="threshold",
                value=float(policy.threshold),
                detail=f"threshold={float(policy.threshold):.2f}",
            )
        )
    for rationale in policy.rationale:
        evidence.append(
            ExplanationEvidence(
                kind="policy",
                label="policy_rationale",
                detail=rationale,
            )
        )

    auto_executed = bool(policy.auto_execute)
    if policy.confidence is not None and policy.threshold is not None:
        auto_executed = auto_executed and float(policy.confidence) >= float(policy.threshold)
        comparator = ">=" if float(policy.confidence) >= float(policy.threshold) else "<"
        reason = (
            f"confidence {float(policy.confidence):.2f} {comparator} "
            f"threshold {float(policy.threshold):.2f}"
        )
    elif policy.auto_execute:
        reason = "policy explicitly requested auto execution without a numeric threshold"
    else:
        reason = "policy requested manual review"

    return DecisionPolicyExplanation(
        auto_executed=auto_executed,
        confidence=float(policy.confidence) if policy.confidence is not None else None,
        threshold=float(policy.threshold) if policy.threshold is not None else None,
        reason=reason,
        human_override=bool(policy.human_override or policy.override_available),
        evidence=evidence,
    )


def _series_evidence(
    *,
    request_series: SeriesInput,
    forecast_series: SeriesForecast,
    trend_direction: TrendDirection,
    spread_ratio: float | None,
) -> list[ExplanationEvidence]:
    history = [float(item) for item in request_series.target]
    forecast = [float(item) for item in forecast_series.mean]
    evidence = [
        ExplanationEvidence(
            kind="forecast",
            label="trend_direction",
            detail=f"mean forecast direction is {trend_direction}",
            value=trend_direction,
        ),
        ExplanationEvidence(
            kind="forecast",
            label="history_points",
            detail=f"history length={len(history)}",
            value=len(history),
        ),
        ExplanationEvidence(
            kind="forecast",
            label="forecast_horizon",
            detail=f"horizon={len(forecast)}",
            value=len(forecast),
        ),
    ]
    if history:
        history_mean = sum(history) / len(history)
        evidence.append(
            ExplanationEvidence(
                kind="metric",
                label="historical_mean",
                detail=f"historical_mean={history_mean:.4f}",
                value=history_mean,
            )
        )
    if forecast:
        forecast_mean = sum(forecast) / len(forecast)
        evidence.append(
            ExplanationEvidence(
                kind="metric",
                label="forecast_mean",
                detail=f"forecast_mean={forecast_mean:.4f}",
                value=forecast_mean,
            )
        )
    if spread_ratio is not None:
        evidence.append(
            ExplanationEvidence(
                kind="metric",
                label="spread_ratio",
                detail=f"quantile spread ratio={spread_ratio:.4f}",
                value=spread_ratio,
            )
        )
    return evidence


def _trend_direction(mean: list[int | float]) -> TrendDirection:
    if not mean:
        return "flat"
    first = float(mean[0])
    last = float(mean[-1])
    delta = last - first
    tolerance = max(abs(first) * 0.01, 1e-6)
    if delta > tolerance:
        return "up"
    if delta < -tolerance:
        return "down"
    return "flat"


def _spread_ratio(forecast: SeriesForecast) -> float | None:
    quantiles = forecast.quantiles
    if not quantiles or len(quantiles) < 2 or not forecast.mean:
        return None

    parsed: list[tuple[float, list[int | float]]] = []
    for key, values in quantiles.items():
        try:
            parsed.append((float(key), list(values)))
        except (TypeError, ValueError):
            return None

    parsed.sort(key=lambda item: item[0])
    low = parsed[0][1]
    high = parsed[-1][1]
    spreads = [abs(float(hi) - float(lo)) for lo, hi in zip(low, high, strict=True)]
    if not spreads:
        return None
    baseline = sum(abs(float(item)) for item in forecast.mean) / len(forecast.mean)
    scale = baseline if baseline > 0 else 1.0
    return (sum(spreads) / len(spreads)) / scale


def _confidence_level_from_spread(spread_ratio: float | None) -> ConfidenceLevel:
    if spread_ratio is None:
        return "unknown"
    if spread_ratio < 0.2:
        return "high"
    if spread_ratio < 0.5:
        return "medium"
    return "low"


def _temporal_importance(
    *,
    timestamps: list[str],
    history: list[int | float],
    top_k: int = 8,
) -> list[TemporalImportancePoint]:
    values = [float(item) for item in history]
    if not values:
        return []

    diffs: list[float] = []
    for index, value in enumerate(values):
        prev = values[index - 1] if index > 0 else value
        diffs.append(abs(value - prev))

    center = sum(values) / len(values)
    magnitudes = [abs(value - center) for value in values]
    n = len(values)
    raw_scores: list[float] = []
    for index, (diff, magnitude) in enumerate(zip(diffs, magnitudes, strict=True)):
        recency = (index + 1) / n
        raw_scores.append(
            0.45 * recency
            + 0.35 * _safe_norm(diff, diffs)
            + 0.20 * _safe_norm(magnitude, magnitudes)
        )

    max_score = max(raw_scores) if raw_scores else 1.0
    normalized = [score / max_score if max_score > 0 else 0.0 for score in raw_scores]
    ranked_indices = sorted(range(n), key=lambda idx: normalized[idx], reverse=True)[
        : max(1, min(top_k, n))
    ]
    selected_indices = sorted(ranked_indices)

    points: list[TemporalImportancePoint] = []
    for idx in selected_indices:
        lag = n - idx
        value = values[idx]
        prev = values[idx - 1] if idx > 0 else value
        delta = value - prev
        if delta > 0:
            direction = "positive"
        elif delta < 0:
            direction = "negative"
        else:
            direction = "neutral"

        if lag == 1:
            reason = "most recent observation anchors the near-term forecast"
        elif normalized[idx] > 0.75:
            reason = "large local change and recency made this point influential"
        else:
            reason = "point contributed through recency and deviation from baseline"

        points.append(
            TemporalImportancePoint(
                lag=lag,
                timestamp=timestamps[idx] if idx < len(timestamps) else None,
                value=float(value),
                importance=round(float(normalized[idx]), 4),
                direction=direction,
                reason=reason,
            )
        )
    return points


def _forecast_decomposition(
    *,
    history: list[int | float],
    forecast: list[int | float],
) -> ForecastDecomposition:
    history_values = [float(item) for item in history]
    forecast_values = [float(item) for item in forecast]
    if not history_values and not forecast_values:
        return ForecastDecomposition(
            trend=0.34,
            seasonal=0.33,
            residual=0.33,
            dominant_driver="insufficient history",
        )

    trend_strength = abs(
        (forecast_values[-1] if forecast_values else 0.0)
        - (history_values[-1] if history_values else 0.0)
    )
    seasonal_strength = _seasonality_strength(history_values)
    residual_strength = _volatility_strength(history_values)

    total = trend_strength + seasonal_strength + residual_strength
    if total <= 0:
        shares = (0.34, 0.33, 0.33)
    else:
        shares = (
            trend_strength / total,
            seasonal_strength / total,
            residual_strength / total,
        )

    labels = {
        "trend": shares[0],
        "seasonal": shares[1],
        "residual": shares[2],
    }
    dominant_driver = max(labels, key=labels.get)

    return ForecastDecomposition(
        trend=round(shares[0], 4),
        seasonal=round(shares[1], 4),
        residual=round(shares[2], 4),
        dominant_driver=dominant_driver,
    )


def _seasonality_strength(history: list[float]) -> float:
    if len(history) < 6:
        return 0.0
    best = 0.0
    for lag in (7, 12, 24, 30):
        if len(history) < lag * 2:
            continue
        left = history[:-lag]
        right = history[lag:]
        if not left or not right:
            continue
        correlation = _correlation(left, right)
        best = max(best, max(0.0, correlation))
    return best


def _volatility_strength(history: list[float]) -> float:
    if len(history) < 3:
        return 0.0
    diffs = [history[index] - history[index - 1] for index in range(1, len(history))]
    mean = sum(diffs) / len(diffs)
    variance = sum((item - mean) ** 2 for item in diffs) / len(diffs)
    return math.sqrt(variance)


def _safe_norm(value: float, population: list[float]) -> float:
    if not population:
        return 0.0
    denom = max(population)
    return value / denom if denom > 0 else 0.0


def _correlation(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        return 0.0
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum(
        (a - left_mean) * (b - right_mean)
        for a, b in zip(left, right, strict=True)
    )
    left_var = sum((a - left_mean) ** 2 for a in left)
    right_var = sum((b - right_mean) ** 2 for b in right)
    denom = math.sqrt(left_var * right_var)
    if denom <= 0:
        return 0.0
    return numerator / denom
