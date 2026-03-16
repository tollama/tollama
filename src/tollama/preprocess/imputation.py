"""Missing value imputation strategies for time series.

Provides multiple imputation methods beyond the existing spline-based approach:
forward fill, backward fill, linear interpolation, and seasonal interpolation.

All functions operate on plain numpy arrays with NaN representing missing values.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

ImputationMethod = Literal[
    "forward_fill",
    "backward_fill",
    "linear",
    "seasonal",
]


def forward_fill(values: np.ndarray) -> np.ndarray:
    """Fill missing values by propagating the last valid observation forward.

    Leading NaNs remain unchanged (no previous value to propagate).
    """
    result = values.copy()
    mask = np.isnan(result)
    if not mask.any():
        return result
    for i in range(1, len(result)):
        if mask[i] and not mask[i - 1]:
            result[i] = result[i - 1]
            mask[i] = False
        elif mask[i] and not np.isnan(result[i - 1]):
            result[i] = result[i - 1]
            mask[i] = False
    return result


def backward_fill(values: np.ndarray) -> np.ndarray:
    """Fill missing values by propagating the next valid observation backward.

    Trailing NaNs remain unchanged (no subsequent value to propagate).
    """
    result = values.copy()
    mask = np.isnan(result)
    if not mask.any():
        return result
    for i in range(len(result) - 2, -1, -1):
        if mask[i] and not np.isnan(result[i + 1]):
            result[i] = result[i + 1]
            mask[i] = False
    return result


def linear_interpolation(values: np.ndarray) -> np.ndarray:
    """Fill missing values using linear interpolation between valid points.

    Leading and trailing NaNs are filled by nearest valid value (constant
    extrapolation) to avoid unbounded extrapolation.
    """
    result = values.copy()
    mask = np.isnan(result)
    if not mask.any():
        return result

    valid_idx = np.where(~mask)[0]
    if len(valid_idx) == 0:
        return result
    if len(valid_idx) == 1:
        result[:] = result[valid_idx[0]]
        return result

    missing_idx = np.where(mask)[0]
    result[missing_idx] = np.interp(
        missing_idx,
        valid_idx,
        result[valid_idx],
    )
    return result


def seasonal_interpolation(
    values: np.ndarray,
    period: int | None = None,
) -> np.ndarray:
    """Fill missing values using seasonal (periodic) interpolation.

    For each missing value at index ``i``, looks at values at ``i - period``,
    ``i - 2*period``, etc. and takes their mean. Falls back to linear
    interpolation when no seasonal neighbours are available.

    If ``period`` is None, attempts to auto-detect via dominant frequency in
    the non-missing portion of the series.
    """
    result = values.copy()
    mask = np.isnan(result)
    if not mask.any():
        return result

    if period is None:
        period = _detect_period(result[~mask])

    if period is None or period < 2:
        return linear_interpolation(values)

    missing_idx = np.where(mask)[0]
    for idx in missing_idx:
        seasonal_vals = []
        # Look backward
        k = idx - period
        while k >= 0:
            if not np.isnan(result[k]):
                seasonal_vals.append(result[k])
            k -= period
        # Look forward
        k = idx + period
        while k < len(result):
            if not np.isnan(result[k]):
                seasonal_vals.append(result[k])
            k += period

        if seasonal_vals:
            result[idx] = float(np.mean(seasonal_vals))

    # Fall back to linear interpolation for any remaining NaNs
    remaining_mask = np.isnan(result)
    if remaining_mask.any():
        result = linear_interpolation(result)

    return result


def _detect_period(values: np.ndarray) -> int | None:
    """Auto-detect the dominant period via FFT on non-missing values."""
    if len(values) < 6:
        return None
    centered = values - np.mean(values)
    fft_vals = np.abs(np.fft.rfft(centered))
    # Skip DC component (index 0) and very high frequencies
    if len(fft_vals) < 3:
        return None
    fft_vals[0] = 0
    dominant_freq_idx = int(np.argmax(fft_vals[1:])) + 1
    if dominant_freq_idx == 0:
        return None
    period = len(values) // dominant_freq_idx
    if period < 2 or period > len(values) // 2:
        return None
    return period


def impute(
    values: np.ndarray,
    method: ImputationMethod = "linear",
    *,
    period: int | None = None,
) -> np.ndarray:
    """Apply the specified imputation method to fill NaN values.

    Parameters
    ----------
    values:
        1-D array with NaN for missing entries.
    method:
        One of ``"forward_fill"``, ``"backward_fill"``, ``"linear"``,
        ``"seasonal"``.
    period:
        Seasonal period (only used when ``method="seasonal"``).  Auto-detected
        when ``None``.

    Returns
    -------
    numpy.ndarray
        Copy of *values* with NaNs filled.
    """
    if method == "forward_fill":
        return forward_fill(values)
    if method == "backward_fill":
        return backward_fill(values)
    if method == "linear":
        return linear_interpolation(values)
    if method == "seasonal":
        return seasonal_interpolation(values, period=period)
    raise ValueError(f"unknown imputation method: {method!r}")
