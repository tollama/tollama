"""Spline-based interpolation, smoothing, and feature extraction for time series."""

from __future__ import annotations

import logging

import numpy as np

from ._compat import require_scipy

logger = logging.getLogger(__name__)

KNOT_STRATEGIES = ("auto", "curvature", "uniform")
SMOOTHING_METHODS = ("legacy", "pspline")


class SplinePreprocessor:
    """Cubic spline preprocessor for interpolation, smoothing, and feature extraction.

    Supports three knot strategies (auto, curvature, uniform) and two smoothing
    methods (legacy Savitzky-Golay, P-spline).
    """

    def __init__(
        self,
        degree: int = 3,
        smoothing_factor: float = 0.5,
        num_knots: int = 10,
        knot_strategy: str = "auto",
        smoothing_method: str = "legacy",
    ) -> None:
        if smoothing_factor < 0:
            raise ValueError("smoothing_factor must be >= 0")
        if knot_strategy not in KNOT_STRATEGIES:
            raise ValueError(f"knot_strategy must be one of {KNOT_STRATEGIES}")
        if smoothing_method not in SMOOTHING_METHODS:
            raise ValueError(f"smoothing_method must be one of {SMOOTHING_METHODS}")

        self.degree = degree
        self.smoothing_factor = smoothing_factor
        self.num_knots = num_knots
        self.knot_strategy = knot_strategy
        self.smoothing_method = smoothing_method
        self._spline = None
        self._fitted = False

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_1d_float_array(arr: np.ndarray, name: str) -> np.ndarray:
        out = np.asarray(arr, dtype=float)
        if out.ndim != 1:
            raise ValueError(f"{name} must be 1D, got shape {out.shape}")
        return out

    @staticmethod
    def _select_knots_uniform(x: np.ndarray, max_knots: int) -> np.ndarray:
        """Place interior knots evenly across the data range."""
        n_interior = max(1, min(max_knots, len(x) // 4))
        return np.linspace(x[1], x[-2], n_interior)

    @staticmethod
    def _select_knots_curvature(
        x: np.ndarray,
        y: np.ndarray,
        max_knots: int,
        degree: int,
    ) -> np.ndarray:
        """Concentrate knots in high-curvature regions via inverse-CDF sampling."""
        interpolate_mod, _ = require_scipy()
        try:
            preliminary = interpolate_mod.UnivariateSpline(x, y, k=degree, s=len(y))
            d2 = np.abs(preliminary(x, nu=2))
        except Exception:
            return SplinePreprocessor._select_knots_uniform(x, max_knots)

        d2 = np.maximum(d2, 1e-12)
        cdf = np.cumsum(d2)
        cdf = cdf / cdf[-1]
        n_interior = max(1, min(max_knots, len(x) // 4))
        targets = np.linspace(0, 1, n_interior + 2)[1:-1]
        knot_indices = np.searchsorted(cdf, targets).clip(1, len(x) - 2)
        return np.unique(x[knot_indices])

    # ------------------------------------------------------------------
    # Fit / transform
    # ------------------------------------------------------------------

    def fit(self, x: np.ndarray, y: np.ndarray) -> SplinePreprocessor:
        """Fit a spline to the data.

        Tries (in order): P-spline, LSQUnivariateSpline, legacy UnivariateSpline,
        linear interp1d as fallback.
        """
        interpolate_mod, _ = require_scipy()
        x = self._to_1d_float_array(x, "x")
        y = self._to_1d_float_array(y, "y")

        if len(x) != len(y):
            raise ValueError("x and y must have the same length")
        if np.any(np.isnan(x)) or np.any(np.isinf(x)):
            raise ValueError("x contains NaN/Inf")

        # Mask NaN in y
        valid = ~np.isnan(y)
        xv, yv = x[valid], y[valid]
        if len(xv) < 2:
            raise ValueError("need at least 2 valid points to fit a spline")

        # Ensure strictly increasing x
        order = np.argsort(xv)
        xv, yv = xv[order], yv[order]
        unique_mask = np.concatenate([[True], np.diff(xv) > 0])
        xv, yv = xv[unique_mask], yv[unique_mask]
        if len(xv) < 2:
            raise ValueError("need at least 2 unique x values to fit a spline")

        # Path 1: P-spline (scipy >= 1.11)
        if self.smoothing_method == "pspline":
            try:
                make_smoothing_spline = getattr(interpolate_mod, "make_smoothing_spline", None)
                if make_smoothing_spline is not None:
                    self._spline = make_smoothing_spline(xv, yv)
                    self._fitted = True
                    return self
            except Exception:
                logger.debug("P-spline failed, falling back to LSQ")

        degree = min(self.degree, len(xv) - 1)

        # Path 2: LSQUnivariateSpline with knot selection
        if len(xv) > 2 * (degree + 1):
            try:
                if self.knot_strategy == "curvature":
                    knots = self._select_knots_curvature(xv, yv, self.num_knots, degree)
                else:
                    knots = self._select_knots_uniform(xv, self.num_knots)

                # Filter knots to interior
                knots = knots[(knots > xv[0]) & (knots < xv[-1])]
                if len(knots) > 0:
                    self._spline = interpolate_mod.LSQUnivariateSpline(xv, yv, knots, k=degree)
                    self._fitted = True
                    return self
            except Exception:
                logger.debug("LSQ spline failed, falling back to UnivariateSpline")

        # Path 3: UnivariateSpline
        if len(xv) > degree:
            try:
                self._spline = interpolate_mod.UnivariateSpline(
                    xv, yv, k=degree, s=self.smoothing_factor * len(xv)
                )
                self._fitted = True
                return self
            except Exception:
                logger.debug("UnivariateSpline failed, falling back to linear interp")

        # Path 4: Linear interpolation fallback
        self._spline = interpolate_mod.interp1d(xv, yv, kind="linear", fill_value="extrapolate")
        self._fitted = True
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the fitted spline at given x positions."""
        if not self._fitted:
            raise RuntimeError("SplinePreprocessor is not fitted")
        x = self._to_1d_float_array(x, "x")
        return np.asarray(self._spline(x), dtype=float)

    def fit_transform(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Fit and evaluate in one call."""
        self.fit(x, y)
        return self.transform(x)

    # ------------------------------------------------------------------
    # Missing value handling
    # ------------------------------------------------------------------

    def interpolate_missing(
        self,
        y: np.ndarray,
        missing_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """Fill NaN values via spline interpolation on valid positions."""
        interpolate_mod, _ = require_scipy()
        y = self._to_1d_float_array(y, "y")
        result = y.copy()

        if missing_mask is not None:
            mask = np.asarray(missing_mask, dtype=bool)
        else:
            mask = np.isnan(y)

        if not mask.any():
            return result

        valid = ~mask
        if valid.sum() < 2:
            raise ValueError("need at least 2 valid points for interpolation")

        x_all = np.arange(len(y), dtype=float)
        x_valid = x_all[valid]
        y_valid = y[valid]

        f = interpolate_mod.interp1d(
            x_valid,
            y_valid,
            kind="cubic" if len(x_valid) > 3 else "linear",
            fill_value="extrapolate",
        )
        result[mask] = f(x_all[mask])
        return result

    # ------------------------------------------------------------------
    # Smoothing
    # ------------------------------------------------------------------

    def smooth(self, y: np.ndarray, window: int = 5) -> np.ndarray:
        """Apply smoothing filter.

        For P-spline method, smoothing is already done during fit so this
        returns the input unchanged. For legacy method, applies Savitzky-Golay.
        """
        y = self._to_1d_float_array(y, "y")
        if self.smoothing_method == "pspline":
            return y.copy()

        _, savgol_filter = require_scipy()
        win = min(window, len(y))
        if win % 2 == 0:
            win = max(3, win - 1)
        if win < 3 or len(y) < win:
            return y.copy()
        poly_order = min(2, win - 1)
        return np.asarray(savgol_filter(y, win, poly_order), dtype=float)

    # ------------------------------------------------------------------
    # Advanced methods
    # ------------------------------------------------------------------

    def extrapolate(self, x_future: np.ndarray) -> np.ndarray:
        """Evaluate spline at future x positions."""
        if not self._fitted:
            raise RuntimeError("SplinePreprocessor is not fitted")
        x_future = self._to_1d_float_array(x_future, "x_future")
        return np.asarray(self._spline(x_future), dtype=float)

    def evaluate_derivatives(self, x: np.ndarray, order: int = 1) -> np.ndarray:
        """Compute analytical or numerical derivatives of the fitted spline."""
        if not self._fitted:
            raise RuntimeError("SplinePreprocessor is not fitted")
        x = self._to_1d_float_array(x, "x")

        # Try analytical derivative via spline's built-in
        try:
            return np.asarray(self._spline(x, nu=order), dtype=float)
        except (TypeError, AttributeError):
            pass

        # Fall back to numerical differentiation
        vals = self.transform(x)
        for _ in range(order):
            vals = np.gradient(vals, x)
        return vals

    def compute_residuals(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Return y - spline(x)."""
        if not self._fitted:
            raise RuntimeError("SplinePreprocessor is not fitted")
        y = self._to_1d_float_array(y, "y")
        fitted_vals = self.transform(x)
        return y - fitted_vals

    def extract_features(self, y: np.ndarray) -> dict[str, float]:
        """Compute summary statistics, optionally with trend from a temporary spline."""
        y = self._to_1d_float_array(y, "y")
        features: dict[str, float] = {
            "mean": float(np.nanmean(y)),
            "std": float(np.nanstd(y)),
            "min": float(np.nanmin(y)),
            "max": float(np.nanmax(y)),
        }

        if len(y) >= 4:
            x = np.arange(len(y), dtype=float)
            tmp = SplinePreprocessor(
                degree=self.degree,
                smoothing_factor=self.smoothing_factor,
                num_knots=min(self.num_knots, len(y) // 2),
                knot_strategy=self.knot_strategy,
                smoothing_method=self.smoothing_method,
            )
            try:
                tmp.fit(x, y)
                trend = tmp.transform(x)
                features["trend_start"] = float(trend[0])
                features["trend_end"] = float(trend[-1])
                features["trend_range"] = float(trend[-1] - trend[0])
            except Exception:
                pass

        return features
