"""Bridge between tollama.core.schemas.SeriesInput and preprocessing arrays."""

from __future__ import annotations

import numpy as np

from tollama.core.schemas import SeriesInput

from .pipeline import PreprocessResult, run_pipeline
from .schemas import PreprocessConfig


def series_input_to_arrays(
    series: SeriesInput,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Convert a SeriesInput to (timestamps_numeric, target, covariates_dict).

    timestamps_numeric: float64 array of 0-based ordinal positions.
    target: float64 array of target values.
    covariates_dict: {name: float64 array} for numeric past_covariates.
    """
    target = np.array(series.target, dtype=float)
    x = np.arange(len(target), dtype=float)

    covariates: dict[str, np.ndarray] = {}
    if series.past_covariates:
        for name, values in series.past_covariates.items():
            numeric = []
            for v in values:
                try:
                    numeric.append(float(v))
                except (TypeError, ValueError):
                    numeric.append(float("nan"))
            covariates[name] = np.array(numeric, dtype=float)

    return x, target, covariates


def preprocess_series_input(
    series: SeriesInput,
    config: PreprocessConfig | None = None,
) -> PreprocessResult:
    """Run the full preprocessing pipeline on a SeriesInput."""
    x, target, _ = series_input_to_arrays(series)
    return run_pipeline(x, target, config=config)
