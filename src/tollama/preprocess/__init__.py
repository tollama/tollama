"""Reusable time-series preprocessing pipeline.

Components can be used independently or orchestrated via ``run_pipeline``.
The core modules (validators, spline, transforms, window, pipeline) work with
plain numpy arrays and have no dependency on tollama internals.  The ``bridge``
module provides optional integration with ``tollama.core.schemas.SeriesInput``.
"""

from .imputation import ImputationMethod, impute
from .pipeline import PreprocessResult, run_pipeline
from .schemas import PreprocessConfig, SplineConfig, ValidationConfig
from .spline import SplinePreprocessor
from .transforms import (
    DifferencingTransform,
    LogTransform,
    MinMaxScaler1D,
    StandardScaler1D,
    build_scaler,
    chronological_split,
)
from .validators import validate_series
from .window import make_windows, make_windows_multivariate

__all__ = [
    "DifferencingTransform",
    "ImputationMethod",
    "LogTransform",
    "MinMaxScaler1D",
    "PreprocessConfig",
    "PreprocessResult",
    "SplineConfig",
    "SplinePreprocessor",
    "StandardScaler1D",
    "ValidationConfig",
    "build_scaler",
    "chronological_split",
    "impute",
    "make_windows",
    "make_windows_multivariate",
    "run_pipeline",
    "validate_series",
]
