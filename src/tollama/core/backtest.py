"""Backtest engine for evaluating models and learning ensemble weights.

Runs a walk-forward evaluation where historical data is split into
rolling train/test windows. Each installed model is forecast on each
window, and per-model accuracy metrics are aggregated to derive
optimal ensemble weights.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path

from .forecast_metrics import compute_forecast_metrics
from .schemas import (
    ForecastParameters,
    ForecastRequest,
    ForecastResponse,
    MetricsParameters,
    SeriesInput,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BacktestFold:
    """One walk-forward evaluation window."""

    fold_index: int
    train_end: int  # index into target where test starts
    test_start: int
    test_end: int


@dataclass(frozen=True)
class ModelFoldResult:
    """Metrics from one model on one fold."""

    model: str
    fold_index: int
    metrics: dict[str, float]
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class BacktestResult:
    """Full backtest output across all models and folds."""

    models: list[str]
    folds: list[BacktestFold]
    fold_results: list[ModelFoldResult]
    aggregate_metrics: dict[str, dict[str, float]]  # model -> metric -> value
    learned_weights: dict[str, float]  # model -> weight
    dataset_fingerprint: str
    metric_names: list[str]


# ---------------------------------------------------------------------------
# Walk-forward fold generation
# ---------------------------------------------------------------------------


def generate_folds(
    *,
    series_length: int,
    horizon: int,
    num_folds: int = 3,
    min_train_length: int | None = None,
) -> list[BacktestFold]:
    """Generate walk-forward backtest folds.

    Each fold uses an expanding window: the test set is always ``horizon``
    points, and the training set grows as we advance through the series.
    """
    if min_train_length is None:
        min_train_length = max(horizon * 2, 32)

    folds: list[BacktestFold] = []
    # Place folds from the end of the series backwards
    for i in range(num_folds):
        test_end = series_length - (i * horizon)
        test_start = test_end - horizon
        train_end = test_start

        if train_end < min_train_length:
            break
        if test_start < 0 or test_end > series_length:
            break

        folds.append(
            BacktestFold(
                fold_index=num_folds - 1 - i,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
            )
        )

    folds.reverse()
    return folds


def build_fold_request(
    *,
    series: list[SeriesInput],
    fold: BacktestFold,
    model: str,
    horizon: int,
    quantiles: list[float] | None = None,
) -> ForecastRequest:
    """Build a ForecastRequest for one backtest fold.

    Truncates each series target to ``fold.train_end`` and sets actuals
    to the test window so metrics can be computed.
    """
    fold_series: list[SeriesInput] = []
    for s in series:
        truncated_target = list(s.target[: fold.train_end])
        test_actuals = list(s.target[fold.test_start : fold.test_end])

        fold_series.append(
            SeriesInput(
                id=s.id,
                freq=s.freq,
                timestamps=s.timestamps[: fold.train_end] if s.timestamps else None,
                target=truncated_target,
                actuals=test_actuals,
                past_covariates=s.past_covariates,
                future_covariates=s.future_covariates,
                static_covariates=s.static_covariates,
            )
        )

    return ForecastRequest(
        model=model,
        horizon=horizon,
        series=fold_series,
        quantiles=quantiles,
        parameters=ForecastParameters(
            metrics=MetricsParameters(names=["mase", "mae", "rmse"]),
        ),
    )


# ---------------------------------------------------------------------------
# Metrics aggregation & weight derivation
# ---------------------------------------------------------------------------


def evaluate_fold(
    *,
    request: ForecastRequest,
    response: ForecastResponse,
    fold_index: int,
) -> ModelFoldResult:
    """Compute metrics for one model on one fold."""
    metrics_result, warnings = compute_forecast_metrics(
        request=request,
        response=response,
    )
    metric_values: dict[str, float] = {}
    if metrics_result is not None and metrics_result.aggregate:
        metric_values = dict(metrics_result.aggregate)

    return ModelFoldResult(
        model=response.model,
        fold_index=fold_index,
        metrics=metric_values,
        warnings=warnings,
    )


def aggregate_backtest_results(
    *,
    fold_results: list[ModelFoldResult],
    metric_names: list[str],
) -> dict[str, dict[str, float]]:
    """Average per-fold metrics into per-model aggregates."""
    model_metrics: dict[str, dict[str, list[float]]] = {}
    for result in fold_results:
        if result.model not in model_metrics:
            model_metrics[result.model] = {name: [] for name in metric_names}
        for name in metric_names:
            value = result.metrics.get(name)
            if value is not None and math.isfinite(value):
                model_metrics[result.model][name].append(value)

    aggregated: dict[str, dict[str, float]] = {}
    for model, metrics_lists in model_metrics.items():
        aggregated[model] = {}
        for name, values in metrics_lists.items():
            if values:
                aggregated[model][name] = sum(values) / len(values)
    return aggregated


def derive_ensemble_weights(
    aggregate_metrics: dict[str, dict[str, float]],
    *,
    primary_metric: str = "mase",
) -> dict[str, float]:
    """Derive ensemble weights from aggregate metrics.

    Uses inverse-error weighting on the primary metric: models with lower
    error get higher weight. Falls back to equal weights if the metric
    is unavailable for any model.
    """
    scores: dict[str, float] = {}
    for model, metrics in aggregate_metrics.items():
        value = metrics.get(primary_metric)
        if value is not None and value > 0 and math.isfinite(value):
            scores[model] = 1.0 / value
        else:
            # Fall back to equal weights
            return {model: 1.0 for model in aggregate_metrics}

    if not scores:
        return {model: 1.0 for model in aggregate_metrics}

    total = sum(scores.values())
    if total <= 0:
        return {model: 1.0 for model in aggregate_metrics}

    return {model: round(score / total, 6) for model, score in scores.items()}


# ---------------------------------------------------------------------------
# Dataset fingerprinting & persistence
# ---------------------------------------------------------------------------


def compute_dataset_fingerprint(series: list[SeriesInput]) -> str:
    """Compute a stable fingerprint for a dataset based on series IDs and lengths."""
    parts: list[str] = []
    for s in sorted(series, key=lambda x: x.id):
        parts.append(f"{s.id}:{len(s.target)}:{s.freq}")
    content = "|".join(parts)
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def save_learned_weights(
    *,
    weights: dict[str, float],
    fingerprint: str,
    storage_dir: Path,
) -> Path:
    """Persist learned ensemble weights to a JSON file."""
    storage_dir.mkdir(parents=True, exist_ok=True)
    path = storage_dir / f"backtest_weights_{fingerprint}.json"
    payload = {"fingerprint": fingerprint, "weights": weights}
    path.write_text(json.dumps(payload, indent=2))
    logger.info("saved learned weights to %s", path)
    return path


def load_learned_weights(
    *,
    fingerprint: str,
    storage_dir: Path,
) -> dict[str, float] | None:
    """Load previously learned ensemble weights for a dataset fingerprint."""
    path = storage_dir / f"backtest_weights_{fingerprint}.json"
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
        if payload.get("fingerprint") != fingerprint:
            return None
        weights = payload.get("weights")
        if isinstance(weights, dict):
            return {str(k): float(v) for k, v in weights.items()}
    except (json.JSONDecodeError, KeyError, ValueError):
        logger.warning("failed to load learned weights from %s", path)
    return None
