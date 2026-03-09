"""Scalers, transforms, and chronological splitting for time-series preprocessing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Scaler(Protocol):
    """Protocol for fit/transform scalers."""

    fitted_: bool

    def fit(self, y: np.ndarray) -> Scaler: ...

    def transform(self, y: np.ndarray) -> np.ndarray: ...

    def inverse_transform(self, y: np.ndarray) -> np.ndarray: ...

    def to_dict(self) -> dict[str, float | str]: ...


def _to_1d_float(y: np.ndarray, name: str = "y") -> np.ndarray:
    arr = np.asarray(y, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1D, got shape {arr.shape}")
    return arr


@dataclass
class StandardScaler1D:
    """Z-score normalization: (y - mean) / std."""

    fitted_: bool = False
    mean_: float = 0.0
    std_: float = 1.0

    def fit(self, y: np.ndarray) -> StandardScaler1D:
        arr = _to_1d_float(y)
        self.mean_ = float(np.nanmean(arr))
        self.std_ = float(np.nanstd(arr))
        if self.std_ == 0.0:
            self.std_ = 1.0
        self.fitted_ = True
        return self

    def transform(self, y: np.ndarray) -> np.ndarray:
        if not self.fitted_:
            raise RuntimeError("StandardScaler1D is not fitted")
        arr = _to_1d_float(y)
        return (arr - self.mean_) / self.std_

    def inverse_transform(self, y: np.ndarray) -> np.ndarray:
        if not self.fitted_:
            raise RuntimeError("StandardScaler1D is not fitted")
        arr = _to_1d_float(y)
        return arr * self.std_ + self.mean_

    def to_dict(self) -> dict[str, float | str]:
        return {"method": "standard", "mean": self.mean_, "std": self.std_}


@dataclass
class MinMaxScaler1D:
    """Min-max scaling to [0, 1]."""

    fitted_: bool = False
    min_: float = 0.0
    max_: float = 1.0

    def fit(self, y: np.ndarray) -> MinMaxScaler1D:
        arr = _to_1d_float(y)
        self.min_ = float(np.nanmin(arr))
        self.max_ = float(np.nanmax(arr))
        if self.min_ == self.max_:
            self.max_ = self.min_ + 1.0
        self.fitted_ = True
        return self

    def transform(self, y: np.ndarray) -> np.ndarray:
        if not self.fitted_:
            raise RuntimeError("MinMaxScaler1D is not fitted")
        arr = _to_1d_float(y)
        return (arr - self.min_) / (self.max_ - self.min_)

    def inverse_transform(self, y: np.ndarray) -> np.ndarray:
        if not self.fitted_:
            raise RuntimeError("MinMaxScaler1D is not fitted")
        arr = _to_1d_float(y)
        return arr * (self.max_ - self.min_) + self.min_

    def to_dict(self) -> dict[str, float | str]:
        return {"method": "minmax", "min": self.min_, "max": self.max_}


@dataclass
class DifferencingTransform:
    """First-order differencing for stationarity."""

    fitted_: bool = False
    first_value_: float = 0.0

    def fit(self, y: np.ndarray) -> DifferencingTransform:
        arr = _to_1d_float(y)
        if len(arr) < 2:
            raise ValueError("need at least 2 values for differencing")
        self.first_value_ = float(arr[0])
        self.fitted_ = True
        return self

    def transform(self, y: np.ndarray) -> np.ndarray:
        if not self.fitted_:
            raise RuntimeError("DifferencingTransform is not fitted")
        arr = _to_1d_float(y)
        return np.diff(arr)

    def inverse_transform(self, y: np.ndarray) -> np.ndarray:
        if not self.fitted_:
            raise RuntimeError("DifferencingTransform is not fitted")
        arr = _to_1d_float(y)
        return np.concatenate([[self.first_value_], np.cumsum(arr) + self.first_value_])

    def to_dict(self) -> dict[str, float | str]:
        return {"method": "differencing", "first_value": self.first_value_}


@dataclass
class LogTransform:
    """Log1p / expm1 for variance stabilization."""

    fitted_: bool = False

    def fit(self, y: np.ndarray) -> LogTransform:
        arr = _to_1d_float(y)
        if np.any(arr < 0):
            raise ValueError("LogTransform requires non-negative values")
        self.fitted_ = True
        return self

    def transform(self, y: np.ndarray) -> np.ndarray:
        if not self.fitted_:
            raise RuntimeError("LogTransform is not fitted")
        arr = _to_1d_float(y)
        return np.log1p(arr)

    def inverse_transform(self, y: np.ndarray) -> np.ndarray:
        if not self.fitted_:
            raise RuntimeError("LogTransform is not fitted")
        arr = _to_1d_float(y)
        return np.expm1(arr)

    def to_dict(self) -> dict[str, float | str]:
        return {"method": "log"}


def build_scaler(method: str) -> StandardScaler1D | MinMaxScaler1D:
    """Factory for creating scalers by name."""
    key = method.strip().lower()
    if key == "standard":
        return StandardScaler1D()
    if key == "minmax":
        return MinMaxScaler1D()
    raise ValueError(f"unknown scaling method {method!r}; expected 'standard' or 'minmax'")


def chronological_split(
    y: np.ndarray,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, int]]:
    """Split a 1D time series chronologically into train / val / test.

    Returns:
        (train, val, test, (train_end_idx, val_end_idx))
    """
    arr = _to_1d_float(y)
    n = len(arr)
    if n < 3:
        raise ValueError(f"need at least 3 values for splitting, got {n}")
    if train_ratio + val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be < 1.0")

    train_end = max(1, int(n * train_ratio))
    val_end = max(train_end + 1, int(n * (train_ratio + val_ratio)))
    val_end = min(val_end, n - 1)

    return arr[:train_end], arr[train_end:val_end], arr[val_end:], (train_end, val_end)
