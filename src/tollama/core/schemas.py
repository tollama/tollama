"""Canonical API schemas for forecast requests and responses."""

from __future__ import annotations

import json
from typing import Annotated

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    StrictFloat,
    StrictInt,
    StrictStr,
    ValidationInfo,
    field_validator,
    model_validator,
)

NonEmptyStr = Annotated[StrictStr, Field(min_length=1)]
PositiveInt = Annotated[StrictInt, Field(gt=0)]
Quantile = Annotated[StrictFloat, Field(gt=0.0, lt=1.0)]
NumericValue = StrictInt | StrictFloat
SequenceValues = list[NumericValue]


class CanonicalModel(BaseModel):
    """Base model with strict validation and deterministic JSON serialization."""

    model_config = ConfigDict(extra="forbid", strict=True, ser_json_inf_nan="null")

    def to_json(self) -> str:
        """Serialize model to canonical JSON with sorted keys."""
        payload = self.model_dump(mode="json", exclude_none=True)
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


class SeriesInput(CanonicalModel):
    """Input series with optional covariates."""

    id: NonEmptyStr
    freq: NonEmptyStr
    timestamps: list[NonEmptyStr] = Field(min_length=1)
    target: SequenceValues = Field(min_length=1)
    past_covariates: dict[NonEmptyStr, SequenceValues] | None = None
    future_covariates: dict[NonEmptyStr, SequenceValues] | None = None
    static_covariates: dict[NonEmptyStr, JsonValue] | None = None

    @model_validator(mode="after")
    def validate_lengths(self) -> SeriesInput:
        expected = len(self.timestamps)
        if len(self.target) != expected:
            raise ValueError("target length must match timestamps length")

        if self.past_covariates:
            for name, values in self.past_covariates.items():
                if len(values) != expected:
                    raise ValueError(
                        f"past_covariates[{name!r}] length must match timestamps length",
                    )

        if self.future_covariates:
            for name, values in self.future_covariates.items():
                if len(values) < expected:
                    raise ValueError(
                        f"future_covariates[{name!r}] length must be >= timestamps length",
                    )

        return self


class ForecastRequest(CanonicalModel):
    """Unified forecast request payload."""

    model: NonEmptyStr
    horizon: PositiveInt
    quantiles: list[Quantile] = Field(default_factory=list)
    series: list[SeriesInput] = Field(min_length=1)
    options: dict[NonEmptyStr, JsonValue] = Field(default_factory=dict)

    @field_validator("quantiles")
    @classmethod
    def validate_quantiles(cls, value: list[Quantile]) -> list[Quantile]:
        if value != sorted(value):
            raise ValueError("quantiles must be sorted in ascending order")
        if len(value) != len(set(value)):
            raise ValueError("quantiles must be unique")
        return value


class SeriesForecast(CanonicalModel):
    """Forecast output for one series."""

    id: NonEmptyStr
    freq: NonEmptyStr
    start_timestamp: NonEmptyStr
    mean: SequenceValues = Field(min_length=1)
    quantiles: dict[NonEmptyStr, SequenceValues] | None = None

    @field_validator("quantiles")
    @classmethod
    def validate_quantile_keys(
        cls,
        quantiles: dict[NonEmptyStr, SequenceValues] | None,
    ) -> dict[NonEmptyStr, SequenceValues] | None:
        if quantiles is None:
            return None

        for key in quantiles:
            try:
                quantile = float(key)
            except ValueError as exc:
                raise ValueError(f"invalid quantile key: {key!r}") from exc

            if not 0.0 < quantile < 1.0:
                raise ValueError(f"quantile key out of range: {key!r}")

        return quantiles

    @field_validator("quantiles")
    @classmethod
    def validate_quantile_lengths(
        cls,
        quantiles: dict[NonEmptyStr, SequenceValues] | None,
        info: ValidationInfo,
    ) -> dict[NonEmptyStr, SequenceValues] | None:
        if quantiles is None:
            return None

        mean = info.data.get("mean")
        if not isinstance(mean, list):
            return quantiles

        expected = len(mean)
        for key, values in quantiles.items():
            if len(values) != expected:
                raise ValueError(f"quantiles[{key!r}] length must match mean length")
        return quantiles


class ForecastResponse(CanonicalModel):
    """Unified forecast response payload."""

    model: NonEmptyStr
    forecasts: list[SeriesForecast] = Field(min_length=1)
    usage: dict[NonEmptyStr, JsonValue] | None = None
