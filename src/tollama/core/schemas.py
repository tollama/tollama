"""Canonical API schemas for forecast requests and responses."""

from __future__ import annotations

import json
from typing import Annotated, Literal

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    StrictBool,
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
CovariateValue = NumericValue | StrictStr
CovariateValues = list[CovariateValue]
CovariateMode = Literal["best_effort", "strict"]
MetricName = Literal["mape", "mase", "mae", "rmse", "smape"]


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
    freq: NonEmptyStr = "auto"
    timestamps: list[NonEmptyStr] = Field(min_length=1)
    target: SequenceValues = Field(min_length=1)
    actuals: SequenceValues | None = None
    past_covariates: dict[NonEmptyStr, CovariateValues] | None = None
    future_covariates: dict[NonEmptyStr, CovariateValues] | None = None
    static_covariates: dict[NonEmptyStr, JsonValue] | None = None

    @model_validator(mode="after")
    def validate_lengths(self) -> SeriesInput:
        expected = len(self.timestamps)
        if len(self.target) != expected:
            raise ValueError("target length must match timestamps length")

        if self.past_covariates:
            for name, values in self.past_covariates.items():
                _validate_covariate_values(name=name, values=values, location="past_covariates")
                if len(values) != expected:
                    raise ValueError(
                        f"past_covariates[{name!r}] length must match timestamps length",
                    )

        if self.future_covariates:
            for name, values in self.future_covariates.items():
                _validate_covariate_values(name=name, values=values, location="future_covariates")

        return self


class TimesFMParameters(CanonicalModel):
    """TimesFM-specific xreg knobs."""

    xreg_mode: NonEmptyStr = "xreg + timesfm"
    ridge: StrictInt | StrictFloat = 0.0
    force_on_cpu: StrictBool = False


class MetricsParameters(CanonicalModel):
    """Forecast accuracy metric parameters."""

    names: list[MetricName] = Field(min_length=1)
    mase_seasonality: PositiveInt = 1

    @field_validator("names")
    @classmethod
    def validate_names(cls, value: list[MetricName]) -> list[MetricName]:
        if len(value) != len(set(value)):
            raise ValueError("metrics.names must be unique")
        return value


class ForecastParameters(CanonicalModel):
    """Shared forecast parameters independent of model family options."""

    covariates_mode: CovariateMode = "best_effort"
    timesfm: TimesFMParameters | None = None
    metrics: MetricsParameters | None = None


class ForecastRequest(CanonicalModel):
    """Unified forecast request payload."""

    model: NonEmptyStr
    horizon: PositiveInt
    quantiles: list[Quantile] = Field(default_factory=list)
    series: list[SeriesInput] = Field(min_length=1)
    options: dict[NonEmptyStr, JsonValue] = Field(
        default_factory=dict,
        validation_alias=AliasChoices("options"),
    )
    timeout: float | None = Field(default=None, gt=0.0)
    parameters: ForecastParameters = Field(
        default_factory=ForecastParameters,
        validation_alias=AliasChoices("parameters"),
    )

    @field_validator("quantiles")
    @classmethod
    def validate_quantiles(cls, value: list[Quantile]) -> list[Quantile]:
        if value != sorted(value):
            raise ValueError("quantiles must be sorted in ascending order")
        if len(value) != len(set(value)):
            raise ValueError("quantiles must be unique")
        return value

    @model_validator(mode="after")
    def validate_covariates(self) -> ForecastRequest:
        metrics_parameters = self.parameters.metrics
        for series in self.series:
            past_covariates = series.past_covariates or {}
            future_covariates = series.future_covariates or {}

            for name, values in future_covariates.items():
                if len(values) != self.horizon:
                    raise ValueError(
                        f"future_covariates[{name!r}] length must match horizon ({self.horizon})",
                    )

            future_only = set(future_covariates) - set(past_covariates)
            if future_only:
                first = sorted(future_only)[0]
                raise ValueError(
                    "future_covariates must also exist in past_covariates; "
                    f"missing past values for covariate {first!r}",
                )

            for name in set(past_covariates).intersection(future_covariates):
                past_kind = _covariate_kind(past_covariates[name])
                future_kind = _covariate_kind(future_covariates[name])
                if past_kind != future_kind:
                    raise ValueError(
                        "covariate type must be consistent between past and future values; "
                        f"covariate {name!r} has past={past_kind} future={future_kind}",
                    )

            if metrics_parameters is not None:
                actuals = series.actuals
                if actuals is None:
                    raise ValueError(
                        "series.actuals is required when parameters.metrics is provided; "
                        f"missing for series {series.id!r}",
                    )
                if len(actuals) != self.horizon:
                    raise ValueError(
                        f"series {series.id!r} actuals length must match horizon "
                        f"({self.horizon}) when parameters.metrics is provided",
                    )

        return self


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


class SeriesMetrics(CanonicalModel):
    """Calculated metrics for one series."""

    id: NonEmptyStr
    values: dict[NonEmptyStr, StrictFloat] = Field(min_length=1)


class ForecastMetrics(CanonicalModel):
    """Calculated metrics payload for one forecast response."""

    aggregate: dict[NonEmptyStr, StrictFloat] = Field(min_length=1)
    series: list[SeriesMetrics] = Field(min_length=1)


class ForecastResponse(CanonicalModel):
    """Unified forecast response payload."""

    model: NonEmptyStr
    forecasts: list[SeriesForecast] = Field(min_length=1)
    metrics: ForecastMetrics | None = None
    usage: dict[NonEmptyStr, JsonValue] | None = None
    warnings: list[NonEmptyStr] | None = None


def _validate_covariate_values(
    *,
    name: str,
    values: CovariateValues,
    location: str,
) -> None:
    try:
        _covariate_kind(values)
    except ValueError as exc:
        raise ValueError(f"{location}[{name!r}] {exc}") from exc


def _covariate_kind(values: CovariateValues) -> str:
    if not values:
        raise ValueError("must not be empty")

    has_numeric = False
    has_string = False
    for value in values:
        if isinstance(value, str):
            has_string = True
            continue
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            has_numeric = True
            continue
        raise ValueError(f"contains unsupported value type {type(value).__name__!r}")

    if has_numeric and has_string:
        raise ValueError("must contain either only numeric values or only string values")
    if has_string:
        return "categorical"
    return "numeric"
