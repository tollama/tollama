"""Pydantic schemas for preprocessing configuration."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

ScalingMethod = Literal["standard", "minmax", "none"]
KnotStrategy = Literal["auto", "curvature", "uniform"]
SmoothingMethod = Literal["legacy", "pspline"]


class ValidationConfig(BaseModel):
    """Controls for data validation step."""

    model_config = ConfigDict(extra="forbid", strict=True)

    missing_ratio_max: float = Field(default=0.30, ge=0.0, le=1.0)
    max_gap: int | None = Field(default=24, ge=1)
    min_length: int | None = None


class SplineConfig(BaseModel):
    """Spline fitting parameters."""

    model_config = ConfigDict(extra="forbid", strict=True)

    degree: int = Field(default=3, ge=1, le=5)
    smoothing_factor: float = Field(default=0.5, ge=0.0)
    num_knots: int = Field(default=10, ge=1)
    knot_strategy: KnotStrategy = "auto"
    smoothing_method: SmoothingMethod = "legacy"
    smoothing_window: int = Field(default=5, ge=3)


class PreprocessConfig(BaseModel):
    """Full preprocessing pipeline configuration."""

    model_config = ConfigDict(extra="forbid", strict=True)

    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    spline: SplineConfig = Field(default_factory=SplineConfig)
    scaling: ScalingMethod = "standard"
    lookback: int = Field(default=24, gt=0)
    horizon: int = Field(default=1, gt=0)
    train_ratio: float = Field(default=0.7, gt=0.0, lt=1.0)
    val_ratio: float = Field(default=0.15, ge=0.0, lt=1.0)
