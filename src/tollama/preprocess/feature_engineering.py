"""Automated feature engineering for time-series data.

Extracts calendar, lag, rolling-window, and Fourier features from timestamps
and series values.  All functions operate on plain numpy arrays and return
feature dictionaries mapping ``feature_name -> np.ndarray``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class FeatureConfig:
    """Controls which feature families are extracted.

    Parameters
    ----------
    calendar:
        Extract calendar features (hour, day-of-week, month, etc.) from
        timestamps.
    lags:
        Specific lag offsets to generate.  ``None`` triggers auto-detection.
    rolling:
        Rolling-window sizes for mean/std features.  ``None`` triggers
        auto-detection.
    fourier:
        Number of Fourier sin/cos pairs per period.  ``0`` triggers
        auto-detection (defaults to 3 pairs).
    differences:
        Include first-difference of the series as a feature.
    """

    calendar: bool = True
    lags: list[int] | None = None
    rolling: list[int] | None = None
    fourier: int = 0
    differences: bool = False


# ---------------------------------------------------------------------------
# Timestamp helpers
# ---------------------------------------------------------------------------


def _parse_timestamps(timestamps: np.ndarray | list[str]) -> np.ndarray:
    """Convert *timestamps* to an array of ``numpy.datetime64[s]``.

    Accepts ISO-8601 strings, Python ``datetime`` objects, or arrays that
    are already ``datetime64``.
    """
    arr = np.asarray(timestamps)
    if np.issubdtype(arr.dtype, np.datetime64):
        return arr.astype("datetime64[s]")
    # Treat as ISO string or Python datetime
    parsed: list[np.datetime64] = []
    for item in arr.flat:
        if isinstance(item, (datetime, np.datetime64)):
            parsed.append(np.datetime64(item, "s"))
        else:
            parsed.append(np.datetime64(str(item), "s"))
    return np.array(parsed, dtype="datetime64[s]")


def _datetime_components(dt64: np.ndarray) -> dict[str, np.ndarray]:
    """Extract raw integer calendar components from a ``datetime64[s]`` array."""
    # Convert to Python datetimes for reliable component extraction.
    epoch = np.datetime64(0, "s")
    one_second = np.timedelta64(1, "s")
    stamps = ((dt64 - epoch) / one_second).astype(float)

    components: dict[str, np.ndarray] = {}
    hours = np.empty(len(stamps), dtype=float)
    dow = np.empty_like(hours)
    dom = np.empty_like(hours)
    month = np.empty_like(hours)
    quarter = np.empty_like(hours)
    woy = np.empty_like(hours)
    is_weekend = np.empty_like(hours)

    for i, ts in enumerate(stamps):
        dt = datetime.fromtimestamp(float(ts), tz=UTC)
        hours[i] = dt.hour
        dow[i] = dt.weekday()  # 0=Monday
        dom[i] = dt.day
        month[i] = dt.month
        quarter[i] = (dt.month - 1) // 3 + 1
        woy[i] = dt.isocalendar()[1]
        is_weekend[i] = 1.0 if dt.weekday() >= 5 else 0.0

    components["hour_of_day"] = hours
    components["day_of_week"] = dow
    components["day_of_month"] = dom
    components["month"] = month
    components["quarter"] = quarter
    components["week_of_year"] = woy
    components["is_weekend"] = is_weekend
    return components


# ---------------------------------------------------------------------------
# Feature extractors
# ---------------------------------------------------------------------------


def extract_calendar_features(
    timestamps: np.ndarray | list[str],
) -> dict[str, np.ndarray]:
    """Extract normalised calendar features from *timestamps*.

    Each feature is scaled to ``[0, 1]``.  Features that are constant across
    the series (e.g., ``hour_of_day`` when data is daily) are omitted.

    Returns a dict mapping feature names to 1-D float arrays.
    """
    dt64 = _parse_timestamps(timestamps)
    raw = _datetime_components(dt64)

    # Normalisation divisors (max possible value for each component).
    divisors: dict[str, float] = {
        "hour_of_day": 23.0,
        "day_of_week": 6.0,
        "day_of_month": 31.0,
        "month": 12.0,
        "quarter": 4.0,
        "week_of_year": 53.0,
        "is_weekend": 1.0,
    }

    features: dict[str, np.ndarray] = {}
    for name, values in raw.items():
        # Skip constant features.
        if np.nanmin(values) == np.nanmax(values):
            continue
        divisor = divisors.get(name, 1.0)
        if divisor == 0.0:
            divisor = 1.0
        features[name] = values / divisor

    return features


def extract_lag_features(
    values: np.ndarray,
    lags: list[int] | None = None,
) -> dict[str, np.ndarray]:
    """Create lag features for *values*.

    Parameters
    ----------
    values:
        1-D numeric series.
    lags:
        Specific lag offsets (positive integers).  When ``None``, defaults to
        ``[1, 2, 3, 7, 14, 28]`` filtered to those less than
        ``len(values) // 3``.

    Returns a dict ``{"lag_1": array, "lag_7": array, ...}`` where earlier
    positions are filled with ``NaN``.
    """
    values = np.asarray(values, dtype=float)
    n = len(values)
    max_lag = n // 3

    if lags is None:
        lags = [k for k in [1, 2, 3, 7, 14, 28] if k < max_lag]
    else:
        lags = [k for k in lags if 0 < k < n]

    features: dict[str, np.ndarray] = {}
    for lag in lags:
        lagged = np.empty(n, dtype=float)
        lagged[:lag] = np.nan
        lagged[lag:] = values[: n - lag]
        features[f"lag_{lag}"] = lagged

    return features


def extract_rolling_features(
    values: np.ndarray,
    windows: list[int] | None = None,
) -> dict[str, np.ndarray]:
    """Compute rolling mean and standard deviation features.

    Parameters
    ----------
    values:
        1-D numeric series.
    windows:
        Window sizes.  When ``None``, defaults to ``[7, 14, 28]`` filtered to
        those less than ``len(values) // 3``.

    Returns a dict with keys like ``rolling_mean_7``, ``rolling_std_7``.
    Positions where the full window is not yet available are ``NaN``.
    """
    values = np.asarray(values, dtype=float)
    n = len(values)
    max_win = n // 3

    if windows is None:
        windows = [w for w in [7, 14, 28] if w < max_win]
    else:
        windows = [w for w in windows if 1 < w < n]

    features: dict[str, np.ndarray] = {}
    for w in windows:
        r_mean = np.empty(n, dtype=float)
        r_std = np.empty(n, dtype=float)
        r_mean[: w - 1] = np.nan
        r_std[: w - 1] = np.nan
        for i in range(w - 1, n):
            window_slice = values[i - w + 1 : i + 1]
            r_mean[i] = np.nanmean(window_slice)
            r_std[i] = np.nanstd(window_slice)
        features[f"rolling_mean_{w}"] = r_mean
        features[f"rolling_std_{w}"] = r_std

    return features


def extract_fourier_features(
    n: int,
    periods: list[float] | None = None,
    num_pairs: int = 3,
) -> dict[str, np.ndarray]:
    """Generate Fourier sin/cos pairs for seasonal modelling.

    Parameters
    ----------
    n:
        Length of the series (number of time steps).
    periods:
        Seasonal periods in the same time-step units.  ``None`` defaults to
        ``[7, 30.44, 365.25]`` (weekly, monthly, yearly).
    num_pairs:
        Number of harmonic pairs per period.

    Returns a dict with keys like ``sin_7.0_1``, ``cos_7.0_1``, etc.
    """
    if periods is None:
        periods = [7.0, 30.44, 365.25]

    t = np.arange(n, dtype=float)
    features: dict[str, np.ndarray] = {}
    for period in periods:
        if period <= 0:
            continue
        for k in range(1, num_pairs + 1):
            angle = 2.0 * np.pi * k * t / period
            features[f"sin_{period}_{k}"] = np.sin(angle)
            features[f"cos_{period}_{k}"] = np.cos(angle)

    return features


# ---------------------------------------------------------------------------
# Master function
# ---------------------------------------------------------------------------


def auto_engineer_features(
    values: np.ndarray,
    timestamps: np.ndarray | list[str] | None = None,
    config: FeatureConfig | None = None,
) -> dict[str, np.ndarray]:
    """Automatically extract a comprehensive feature set from a time series.

    Combines calendar, lag, rolling-window, Fourier, and differencing features
    into a single dict.  Every returned array has the same length as *values*.

    Parameters
    ----------
    values:
        1-D numeric target series.
    timestamps:
        Optional timestamps aligned with *values*.  Required when calendar
        features are enabled.
    config:
        Feature engineering configuration.  Defaults are applied when ``None``.

    Returns
    -------
    dict[str, np.ndarray]
        Mapping of ``feature_name -> feature_array``.
    """
    cfg = config or FeatureConfig()
    values = np.asarray(values, dtype=float)
    n = len(values)

    features: dict[str, np.ndarray] = {}

    # Calendar features
    if cfg.calendar and timestamps is not None:
        features.update(extract_calendar_features(timestamps))

    # Lag features
    features.update(extract_lag_features(values, lags=cfg.lags))

    # Rolling features
    features.update(extract_rolling_features(values, windows=cfg.rolling))

    # Fourier features
    num_pairs = cfg.fourier if cfg.fourier > 0 else 3
    features.update(extract_fourier_features(n, num_pairs=num_pairs))

    # First differences
    if cfg.differences:
        diff = np.empty(n, dtype=float)
        diff[0] = np.nan
        diff[1:] = np.diff(values)
        features["diff_1"] = diff

    return features
