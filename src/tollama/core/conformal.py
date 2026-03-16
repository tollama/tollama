"""Conformal prediction for calibrated forecast intervals.

Provides split and adaptive conformal methods to produce
distribution-free prediction intervals with guaranteed coverage.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np

from tollama.core.schemas import ForecastResponse, SeriesForecast

ConformalMethod = Literal["split", "adaptive"]

_EPSILON = 1e-8


@dataclass(frozen=True)
class ConformalCalibration:
    """Calibration state produced by the conformal calibration step.

    Attributes:
        scores: Nonconformity scores computed from the calibration set.
        method: The conformal method used (``"split"`` or ``"adaptive"``).
        coverage: Target coverage level in (0, 1).
        horizon: Number of forecast steps the calibration covers.
        q_hat: The conformal quantile threshold for interval construction.
    """

    scores: np.ndarray
    method: ConformalMethod
    coverage: float
    horizon: int
    q_hat: float


def calibrate(
    actuals: np.ndarray,
    predictions: np.ndarray,
    coverage: float = 0.9,
    method: ConformalMethod = "split",
) -> ConformalCalibration:
    """Compute conformal calibration from a held-out calibration set.

    Parameters
    ----------
    actuals:
        Array of actual values with shape ``(n,)`` or ``(n, horizon)``.
    predictions:
        Array of point forecasts with the same shape as *actuals*.
    coverage:
        Desired coverage level, e.g. ``0.9`` for 90 %. Must be in (0, 1).
    method:
        ``"split"`` for standard split conformal or ``"adaptive"`` for
        locally-weighted scores normalised by prediction magnitude.

    Returns
    -------
    ConformalCalibration
        Frozen dataclass that stores the calibration state needed by
        :func:`predict_intervals`.

    Raises
    ------
    ValueError
        If inputs have mismatched shapes, coverage is out of range, or the
        calibration set is empty.
    """

    actuals = np.asarray(actuals, dtype=np.float64)
    predictions = np.asarray(predictions, dtype=np.float64)

    if actuals.shape != predictions.shape:
        msg = (
            f"Shape mismatch: actuals {actuals.shape} vs "
            f"predictions {predictions.shape}"
        )
        raise ValueError(msg)

    if actuals.size == 0:
        raise ValueError("Calibration set must not be empty")

    if not 0 < coverage < 1:
        raise ValueError(f"Coverage must be in (0, 1), got {coverage}")

    # Flatten to 1-D when a single horizon dimension is provided so that
    # the quantile computation operates over all calibration residuals.
    if actuals.ndim == 2:
        horizon = actuals.shape[1]
    elif actuals.ndim == 1:
        horizon = 1
    else:
        raise ValueError(f"Expected 1-D or 2-D arrays, got ndim={actuals.ndim}")

    residuals = np.abs(actuals - predictions)

    if method == "adaptive":
        scores = residuals / (np.abs(predictions) + _EPSILON)
    else:
        scores = residuals

    scores_flat = scores.ravel()
    n = len(scores_flat)
    alpha = 1.0 - coverage
    quantile_level = min(math.ceil((n + 1) * (1.0 - alpha)) / n, 1.0)
    q_hat = float(np.quantile(scores_flat, quantile_level))

    return ConformalCalibration(
        scores=scores_flat,
        method=method,
        coverage=coverage,
        horizon=horizon,
        q_hat=q_hat,
    )


def predict_intervals(
    mean_forecast: list[float],
    calibration: ConformalCalibration,
) -> dict[str, list[float]]:
    """Construct prediction intervals from a point forecast and calibration.

    Returns a quantile dictionary compatible with
    ``SeriesForecast.quantiles``.  For example, 90 % coverage yields keys
    ``"0.05"`` and ``"0.95"``.

    Parameters
    ----------
    mean_forecast:
        Point forecast values for the prediction horizon.
    calibration:
        Calibration state from :func:`calibrate`.

    Returns
    -------
    dict[str, list[float]]
        Mapping from quantile label to per-step values.
    """

    mean = np.asarray(mean_forecast, dtype=np.float64)
    alpha = 1.0 - calibration.coverage
    lower_q = f"{alpha / 2:.2f}"
    upper_q = f"{1.0 - alpha / 2:.2f}"

    if calibration.method == "adaptive":
        # Scale the conformal width by local prediction magnitude.
        width = calibration.q_hat * (np.abs(mean) + _EPSILON)
    else:
        width = calibration.q_hat

    lower = mean - width
    upper = mean + width

    return {
        lower_q: lower.tolist(),
        upper_q: upper.tolist(),
    }


def apply_conformal_to_response(
    response: ForecastResponse,
    calibration: ConformalCalibration,
) -> ForecastResponse:
    """Post-process a :class:`ForecastResponse` to add conformal intervals.

    Existing quantiles on each :class:`SeriesForecast` are preserved; the
    conformal bands are merged in.  A warning is appended when the
    calibration horizon does not match a forecast's length.

    Parameters
    ----------
    response:
        The original forecast response (not mutated).
    calibration:
        Calibration state from :func:`calibrate`.

    Returns
    -------
    ForecastResponse
        A new response with conformal quantile bands added.
    """

    warnings: list[str] = list(response.warnings) if response.warnings else []
    updated_forecasts: list[SeriesForecast] = []

    for forecast in response.forecasts:
        forecast_horizon = len(forecast.mean)

        if calibration.horizon > 1 and forecast_horizon != calibration.horizon:
            warnings.append(
                f"Series '{forecast.id}': calibration horizon "
                f"({calibration.horizon}) differs from forecast horizon "
                f"({forecast_horizon}); intervals may be approximate"
            )

        intervals = predict_intervals(list(forecast.mean), calibration)

        merged_quantiles: dict[str, list[float]] = {}
        if forecast.quantiles is not None:
            merged_quantiles.update(forecast.quantiles)
        merged_quantiles.update(intervals)

        updated_forecasts.append(
            forecast.model_copy(update={"quantiles": merged_quantiles})
        )

    return response.model_copy(
        update={
            "forecasts": updated_forecasts,
            "warnings": warnings or None,
        }
    )
