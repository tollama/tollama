"""Sliding window generation for supervised time-series learning."""

from __future__ import annotations

import numpy as np


def make_windows(
    series: np.ndarray,
    lookback: int,
    horizon: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Create supervised windows from a 1D series.

    Returns:
        X: [batch, lookback, 1]
        y: [batch, horizon]
    """
    s = np.asarray(series, dtype=float)
    if s.ndim != 1:
        raise ValueError(f"series must be 1D, got {s.shape}")
    if np.isnan(s).any() or np.isinf(s).any():
        raise ValueError("series contains NaN/Inf")
    if lookback <= 0 or horizon <= 0:
        raise ValueError("lookback and horizon must be positive")

    n = len(s) - lookback - horizon + 1
    if n <= 0:
        raise ValueError(
            f"not enough points ({len(s)}) for lookback={lookback} + horizon={horizon}"
        )

    X = np.zeros((n, lookback, 1), dtype=np.float32)
    y = np.zeros((n, horizon), dtype=np.float32)

    for i in range(n):
        X[i, :, 0] = s[i : i + lookback]
        y[i, :] = s[i + lookback : i + lookback + horizon]

    return X, y


def make_windows_multivariate(
    features: np.ndarray,
    target: np.ndarray,
    lookback: int,
    horizon: int = 1,
    future_features: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Create windows for multivariate inputs with target-only labels.

    Args:
        features: [time, n_features] past feature matrix.
        target: [time] target series.
        lookback: window width for past features.
        horizon: prediction horizon.
        future_features: optional [time, n_future_features] known-future covariates.

    Returns:
        (X, y) when future features are absent.
        (X, y, X_future) when future features are provided.
    """
    f = np.asarray(features, dtype=float)
    t = np.asarray(target, dtype=float).reshape(-1)
    if f.ndim != 2:
        raise ValueError(f"features must be 2D [time, n_features], got {f.shape}")
    if len(f) != len(t):
        raise ValueError("features and target must have same time length")

    ff = None
    if future_features is not None:
        ff = np.asarray(future_features, dtype=float)
        if ff.ndim != 2:
            raise ValueError(f"future_features must be 2D, got {ff.shape}")
        if len(ff) != len(t):
            raise ValueError("future_features and target must have same time length")

    if np.isnan(f).any() or np.isinf(f).any() or np.isnan(t).any() or np.isinf(t).any():
        raise ValueError("features/target contains NaN/Inf")
    if ff is not None and (np.isnan(ff).any() or np.isinf(ff).any()):
        raise ValueError("future_features contains NaN/Inf")

    if lookback <= 0 or horizon <= 0:
        raise ValueError("lookback and horizon must be positive")

    n = len(f) - lookback - horizon + 1
    if n <= 0:
        if ff is None:
            return np.array([]), np.array([])
        return np.array([]), np.array([]), None

    X = np.zeros((n, lookback, f.shape[1]), dtype=np.float32)
    y = np.zeros((n, horizon), dtype=np.float32)
    X_fut = None if ff is None else np.zeros((n, horizon, ff.shape[1]), dtype=np.float32)

    for i in range(n):
        X[i, :, :] = f[i : i + lookback, :]
        y[i, :] = t[i + lookback : i + lookback + horizon]
        if X_fut is not None and ff is not None:
            X_fut[i, :, :] = ff[i + lookback : i + lookback + horizon, :]

    if ff is None:
        return X, y
    return X, y, X_fut
