"""Lazy imports for optional dependencies."""

from __future__ import annotations


def require_scipy() -> tuple:
    """Return (scipy.interpolate, scipy.signal.savgol_filter) or raise ImportError."""
    try:
        from scipy import interpolate
        from scipy.signal import savgol_filter

        return interpolate, savgol_filter
    except ImportError as exc:
        raise ImportError(
            "scipy is required for spline preprocessing. "
            "Install it with: pip install 'tollama[preprocess]'"
        ) from exc
