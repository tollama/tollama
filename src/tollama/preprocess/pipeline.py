"""In-memory preprocessing pipeline orchestrator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .schemas import PreprocessConfig
from .spline import SplinePreprocessor
from .transforms import build_scaler, chronological_split
from .validators import validate_series
from .window import make_windows


@dataclass(frozen=True, slots=True)
class PreprocessResult:
    """Immutable result of a preprocessing pipeline run."""

    raw: np.ndarray
    interpolated: np.ndarray
    smoothed: np.ndarray
    scaled: np.ndarray
    X: np.ndarray  # [batch, lookback, 1]
    y: np.ndarray  # [batch, horizon]
    scaler: Any  # StandardScaler1D | MinMaxScaler1D | None
    spline: SplinePreprocessor
    train_end: int
    val_end: int


def run_pipeline(
    timestamps: np.ndarray,
    target: np.ndarray,
    config: PreprocessConfig | None = None,
) -> PreprocessResult:
    """Run validate -> interpolate -> smooth -> scale (train-fit) -> window.

    Args:
        timestamps: numeric x-axis positions (e.g., np.arange(n)).
        target: 1D target values (may contain NaN for interpolation).
        config: preprocessing configuration (defaults applied if None).

    Returns:
        PreprocessResult with all intermediate and final arrays.
    """
    cfg = config or PreprocessConfig()

    # 1. Validate
    validate_series(timestamps, target, config=cfg.validation)

    target = np.asarray(target, dtype=float)

    # 2. Interpolate missing values
    spline = SplinePreprocessor(
        degree=cfg.spline.degree,
        smoothing_factor=cfg.spline.smoothing_factor,
        num_knots=cfg.spline.num_knots,
        knot_strategy=cfg.spline.knot_strategy,
        smoothing_method=cfg.spline.smoothing_method,
    )
    interpolated = spline.interpolate_missing(target)

    # 3. Smooth
    smoothed = spline.smooth(interpolated, window=cfg.spline.smoothing_window)

    # 4. Split and scale (fit on train only — prevents data leakage)
    train, _, _, (train_end, val_end) = chronological_split(
        smoothed,
        train_ratio=cfg.train_ratio,
        val_ratio=cfg.val_ratio,
    )

    scaler = None
    if cfg.scaling != "none":
        scaler = build_scaler(cfg.scaling)
        scaler.fit(train)
        scaled = scaler.transform(smoothed)
    else:
        scaled = smoothed.copy()

    # 5. Window
    X, y = make_windows(scaled, lookback=cfg.lookback, horizon=cfg.horizon)

    return PreprocessResult(
        raw=target.copy(),
        interpolated=interpolated,
        smoothed=smoothed,
        scaled=scaled,
        X=X,
        y=y,
        scaler=scaler,
        spline=spline,
        train_end=train_end,
        val_end=val_end,
    )
