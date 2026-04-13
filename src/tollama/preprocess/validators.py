"""Data validation for time-series preprocessing."""

from __future__ import annotations

import numpy as np

from .schemas import ValidationConfig


def _max_consecutive_true(mask: np.ndarray) -> int:
    """Return the longest consecutive run of True values."""
    max_len = cur = 0
    for v in mask.astype(bool):
        if v:
            cur += 1
            if cur > max_len:
                max_len = cur
        else:
            cur = 0
    return int(max_len)


def validate_series(
    timestamps: np.ndarray,
    target: np.ndarray,
    *,
    config: ValidationConfig | None = None,
) -> None:
    """Validate a time series for preprocessing.

    Checks: monotonic timestamps, numeric/finite target,
    missing ratio, max consecutive gap, non-constant target.

    Raises ValueError with descriptive messages on failure.
    """
    cfg = config or ValidationConfig()
    ts = np.asarray(timestamps)
    tgt = np.asarray(target, dtype=float)

    if ts.ndim != 1 or tgt.ndim != 1:
        raise ValueError("timestamps and target must be 1D arrays")

    if len(ts) != len(tgt):
        raise ValueError(f"timestamps length ({len(ts)}) must match target length ({len(tgt)})")

    if len(ts) == 0:
        raise ValueError("timestamps must not be empty")

    # Monotonicity (works for numeric and datetime-like)
    if len(ts) > 1:
        try:
            diffs = np.diff(ts.astype(float))
            if np.any(diffs <= 0):
                raise ValueError("timestamps must be strictly increasing")
        except (TypeError, ValueError) as exc:
            if "strictly increasing" in str(exc):
                raise
            # If timestamps can't be cast to float, skip monotonicity check
            # (e.g., string timestamps that need pandas parsing)

    # Inf check
    if np.any(np.isinf(tgt)):
        raise ValueError("target contains Inf/-Inf")

    # Missing ratio
    missing_mask = np.isnan(tgt)
    if missing_mask.any():
        miss_ratio = float(missing_mask.mean())
        if miss_ratio > cfg.missing_ratio_max:
            raise ValueError(
                f"target missing ratio {miss_ratio:.4f} exceeds limit {cfg.missing_ratio_max:.4f}"
            )

        if cfg.max_gap is not None:
            gap = _max_consecutive_true(missing_mask)
            if gap > cfg.max_gap:
                raise ValueError(f"target max missing gap {gap} exceeds limit {cfg.max_gap}")

    # Non-constant check (on valid values)
    valid = tgt[~missing_mask]
    if valid.size == 0:
        raise ValueError("target has no valid numeric values")
    if np.nanstd(valid) == 0.0:
        raise ValueError("target is constant (zero variance)")

    # Minimum length
    if cfg.min_length is not None and len(tgt) < cfg.min_length:
        raise ValueError(f"series length {len(tgt)} is below minimum {cfg.min_length}")
