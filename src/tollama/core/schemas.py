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
MetricName = Literal["mape", "mase", "mae", "rmse", "smape", "wape", "rmsse", "pinball"]
TrendDirection = Literal["up", "down", "flat"]
AutoForecastStrategy = Literal["auto", "fastest", "best_accuracy", "ensemble"]
ScenarioOperation = Literal["multiply", "add", "replace"]
ScenarioTargetField = Literal["target", "past_covariates", "future_covariates"]


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


class CompareRequest(CanonicalModel):
    """Compare request payload for running one forecast request across multiple models."""

    models: list[NonEmptyStr] = Field(min_length=2)
    horizon: PositiveInt
    quantiles: list[Quantile] = Field(default_factory=list)
    series: list[SeriesInput] = Field(min_length=1)
    options: dict[NonEmptyStr, JsonValue] = Field(
        default_factory=dict,
        validation_alias=AliasChoices("options"),
    )
    timeout: float | None = Field(default=None, gt=0.0)
    keep_alive: StrictStr | StrictInt | StrictFloat | None = None
    parameters: ForecastParameters = Field(
        default_factory=ForecastParameters,
        validation_alias=AliasChoices("parameters"),
    )

    @field_validator("models")
    @classmethod
    def validate_models(cls, value: list[NonEmptyStr]) -> list[NonEmptyStr]:
        if len(value) != len(set(value)):
            raise ValueError("models must be unique")
        return value

    @field_validator("quantiles")
    @classmethod
    def validate_quantiles(cls, value: list[Quantile]) -> list[Quantile]:
        if value != sorted(value):
            raise ValueError("quantiles must be sorted in ascending order")
        if len(value) != len(set(value)):
            raise ValueError("quantiles must be unique")
        return value

    @model_validator(mode="after")
    def validate_forecast_compatibility(self) -> CompareRequest:
        validation_payload = {
            "model": self.models[0],
            "horizon": self.horizon,
            "quantiles": self.quantiles,
            "series": [series.model_dump(mode="python") for series in self.series],
            "options": self.options,
            "timeout": self.timeout,
            "parameters": self.parameters.model_dump(mode="python"),
        }
        ForecastRequest.model_validate(validation_payload)
        return self


class AutoForecastRequest(CanonicalModel):
    """Auto-forecast request payload with optional model override."""

    model: NonEmptyStr | None = None
    allow_fallback: StrictBool = False
    strategy: AutoForecastStrategy = "auto"
    ensemble_top_k: StrictInt = Field(default=3, ge=2, le=8)
    horizon: PositiveInt
    quantiles: list[Quantile] = Field(default_factory=list)
    series: list[SeriesInput] = Field(min_length=1)
    options: dict[NonEmptyStr, JsonValue] = Field(
        default_factory=dict,
        validation_alias=AliasChoices("options"),
    )
    timeout: float | None = Field(default=None, gt=0.0)
    keep_alive: StrictStr | StrictInt | StrictFloat | None = None
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
    def validate_forecast_compatibility(self) -> AutoForecastRequest:
        validation_payload = {
            "model": self.model or "__auto__",
            "horizon": self.horizon,
            "quantiles": self.quantiles,
            "series": [series.model_dump(mode="python") for series in self.series],
            "options": self.options,
            "timeout": self.timeout,
            "parameters": self.parameters.model_dump(mode="python"),
        }
        ForecastRequest.model_validate(validation_payload)
        return self


class AnalyzeParameters(CanonicalModel):
    """Controls for bounded and deterministic series analysis."""

    max_points: PositiveInt = 5000
    max_lag: PositiveInt = 365
    top_k_seasonality: PositiveInt = 3
    anomaly_iqr_k: StrictFloat = Field(default=1.5, gt=0.0)


class AnalyzeRequest(CanonicalModel):
    """Analyze one or more input time series."""

    series: list[SeriesInput] = Field(min_length=1)
    parameters: AnalyzeParameters = Field(
        default_factory=AnalyzeParameters,
        validation_alias=AliasChoices("parameters"),
    )

    @field_validator("series")
    @classmethod
    def validate_unique_ids(cls, value: list[SeriesInput]) -> list[SeriesInput]:
        ids = [item.id for item in value]
        if len(ids) != len(set(ids)):
            raise ValueError("series ids must be unique")
        return value


class TrendAnalysis(CanonicalModel):
    """Trend summary for one series."""

    direction: TrendDirection
    slope: StrictFloat
    r2: StrictFloat = Field(ge=0.0, le=1.0)


class SeriesAnalysis(CanonicalModel):
    """Analysis output for one series."""

    id: NonEmptyStr
    detected_frequency: NonEmptyStr
    seasonality_periods: list[PositiveInt] = Field(default_factory=list)
    trend: TrendAnalysis
    anomaly_indices: list[StrictInt] = Field(default_factory=list)
    stationarity_flag: StrictBool | None = None
    data_quality_score: StrictFloat = Field(ge=0.0, le=1.0)
    warnings: list[NonEmptyStr] | None = None

    @field_validator("seasonality_periods")
    @classmethod
    def validate_seasonality_periods(cls, value: list[PositiveInt]) -> list[PositiveInt]:
        if value != sorted(value):
            raise ValueError("seasonality_periods must be sorted in ascending order")
        if len(value) != len(set(value)):
            raise ValueError("seasonality_periods must be unique")
        return value

    @field_validator("anomaly_indices")
    @classmethod
    def validate_anomaly_indices(cls, value: list[StrictInt]) -> list[StrictInt]:
        if any(index < 0 for index in value):
            raise ValueError("anomaly_indices must be non-negative")
        if value != sorted(value):
            raise ValueError("anomaly_indices must be sorted in ascending order")
        if len(value) != len(set(value)):
            raise ValueError("anomaly_indices must be unique")
        return value


class AnalyzeResponse(CanonicalModel):
    """Response payload for series analysis requests."""

    results: list[SeriesAnalysis] = Field(min_length=1)
    warnings: list[NonEmptyStr] | None = None

    @field_validator("results")
    @classmethod
    def validate_unique_ids(cls, value: list[SeriesAnalysis]) -> list[SeriesAnalysis]:
        ids = [item.id for item in value]
        if len(ids) != len(set(ids)):
            raise ValueError("results ids must be unique")
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


class SeriesMetrics(CanonicalModel):
    """Calculated metrics for one series."""

    id: NonEmptyStr
    values: dict[NonEmptyStr, StrictFloat] = Field(min_length=1)


class ForecastMetrics(CanonicalModel):
    """Calculated metrics payload for one forecast response."""

    aggregate: dict[NonEmptyStr, StrictFloat] = Field(min_length=1)
    series: list[SeriesMetrics] = Field(min_length=1)


class ForecastTiming(CanonicalModel):
    """Timing metadata for one forecast response."""

    model_load_ms: StrictFloat | None = Field(default=None, ge=0.0)
    inference_ms: StrictFloat | None = Field(default=None, ge=0.0)
    total_ms: StrictFloat | None = Field(default=None, ge=0.0)


class SeriesForecastExplanation(CanonicalModel):
    """Deterministic explainability payload for one forecasted series."""

    id: NonEmptyStr
    trend_direction: TrendDirection
    confidence_assessment: NonEmptyStr
    historical_comparison: NonEmptyStr
    notable_patterns: list[NonEmptyStr] = Field(default_factory=list)


class ForecastExplanation(CanonicalModel):
    """Structured explainability payload for a forecast response."""

    series: list[SeriesForecastExplanation] = Field(min_length=1)


class ForecastResponse(CanonicalModel):
    """Unified forecast response payload."""

    model: NonEmptyStr
    forecasts: list[SeriesForecast] = Field(min_length=1)
    metrics: ForecastMetrics | None = None
    timing: ForecastTiming | None = None
    explanation: ForecastExplanation | None = None
    usage: dict[NonEmptyStr, JsonValue] | None = None
    warnings: list[NonEmptyStr] | None = None


class AutoSelectionScore(CanonicalModel):
    """One ranked model score used by auto-forecast selection."""

    model: NonEmptyStr
    family: NonEmptyStr
    rank: PositiveInt
    score: StrictFloat
    reasons: list[NonEmptyStr] = Field(min_length=1)


class AutoSelectionInfo(CanonicalModel):
    """Selection metadata for auto-forecast requests."""

    strategy: AutoForecastStrategy
    chosen_model: NonEmptyStr
    selected_models: list[NonEmptyStr] = Field(min_length=1)
    candidates: list[AutoSelectionScore] = Field(min_length=1)
    rationale: list[NonEmptyStr] = Field(min_length=1)
    fallback_used: StrictBool = False

    @field_validator("selected_models")
    @classmethod
    def validate_selected_models(cls, value: list[NonEmptyStr]) -> list[NonEmptyStr]:
        if len(value) != len(set(value)):
            raise ValueError("selected_models must be unique")
        return value


class AutoForecastResponse(CanonicalModel):
    """Response payload for auto-forecast requests."""

    strategy: AutoForecastStrategy
    selection: AutoSelectionInfo
    response: ForecastResponse

    @model_validator(mode="after")
    def validate_strategy_consistency(self) -> AutoForecastResponse:
        if self.strategy != self.selection.strategy:
            raise ValueError("strategy and selection.strategy must match")
        return self


class WhatIfTransform(CanonicalModel):
    """One deterministic mutation applied to one request field."""

    operation: ScenarioOperation
    field: ScenarioTargetField
    key: NonEmptyStr | None = None
    value: CovariateValue
    series_id: NonEmptyStr | None = None

    @model_validator(mode="after")
    def validate_shape(self) -> WhatIfTransform:
        if self.field == "target":
            if self.key is not None:
                raise ValueError("key must be null when field is 'target'")
        elif self.key is None:
            raise ValueError("key is required when field is a covariates object")

        if self.operation in {"multiply", "add"}:
            if not isinstance(self.value, (int, float)) or isinstance(self.value, bool):
                raise ValueError(f"value must be numeric for operation {self.operation!r}")

        if self.field == "target" and (
            not isinstance(self.value, (int, float)) or isinstance(self.value, bool)
        ):
            raise ValueError("value must be numeric when field is 'target'")

        return self


class WhatIfScenario(CanonicalModel):
    """Named what-if scenario with one or more transforms."""

    name: NonEmptyStr
    transforms: list[WhatIfTransform] = Field(min_length=1)


class WhatIfRequest(CanonicalModel):
    """Scenario analysis request payload built on top of forecast inputs."""

    model: NonEmptyStr
    horizon: PositiveInt
    quantiles: list[Quantile] = Field(default_factory=list)
    series: list[SeriesInput] = Field(min_length=1)
    scenarios: list[WhatIfScenario] = Field(min_length=1)
    continue_on_error: StrictBool = True
    options: dict[NonEmptyStr, JsonValue] = Field(
        default_factory=dict,
        validation_alias=AliasChoices("options"),
    )
    timeout: float | None = Field(default=None, gt=0.0)
    keep_alive: StrictStr | StrictInt | StrictFloat | None = None
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

    @field_validator("scenarios")
    @classmethod
    def validate_scenario_names(cls, value: list[WhatIfScenario]) -> list[WhatIfScenario]:
        names = [item.name for item in value]
        if len(names) != len(set(names)):
            raise ValueError("scenario names must be unique")
        return value

    @model_validator(mode="after")
    def validate_forecast_compatibility(self) -> WhatIfRequest:
        validation_payload = {
            "model": self.model,
            "horizon": self.horizon,
            "quantiles": self.quantiles,
            "series": [series.model_dump(mode="python") for series in self.series],
            "options": self.options,
            "timeout": self.timeout,
            "parameters": self.parameters.model_dump(mode="python"),
        }
        ForecastRequest.model_validate(validation_payload)

        series_ids = {series.id for series in self.series}
        for scenario in self.scenarios:
            for transform in scenario.transforms:
                if transform.series_id is not None and transform.series_id not in series_ids:
                    raise ValueError(
                        "unknown transform.series_id "
                        f"{transform.series_id!r} in scenario {scenario.name!r}",
                    )
        return self


class CompareError(CanonicalModel):
    """Error payload for one failed model comparison result."""

    category: NonEmptyStr
    status_code: StrictInt = Field(ge=400, le=599)
    message: NonEmptyStr


class CompareResult(CanonicalModel):
    """One model result in a compare response."""

    model: NonEmptyStr
    ok: StrictBool
    response: ForecastResponse | None = None
    error: CompareError | None = None

    @model_validator(mode="after")
    def validate_shape(self) -> CompareResult:
        if self.ok and self.response is None:
            raise ValueError("response is required when ok is true")
        if self.ok and self.error is not None:
            raise ValueError("error must be null when ok is true")
        if not self.ok and self.error is None:
            raise ValueError("error is required when ok is false")
        if not self.ok and self.response is not None:
            raise ValueError("response must be null when ok is false")
        return self


class CompareSummary(CanonicalModel):
    """Summary for compare endpoint outcomes."""

    requested_models: PositiveInt
    succeeded: StrictInt = Field(ge=0)
    failed: StrictInt = Field(ge=0)

    @model_validator(mode="after")
    def validate_counts(self) -> CompareSummary:
        if self.succeeded + self.failed != self.requested_models:
            raise ValueError("succeeded + failed must equal requested_models")
        return self


class CompareResponse(CanonicalModel):
    """Response payload for model comparison requests."""

    models: list[NonEmptyStr] = Field(min_length=1)
    horizon: PositiveInt
    results: list[CompareResult] = Field(min_length=1)
    summary: CompareSummary


class WhatIfError(CanonicalModel):
    """Error payload for one failed scenario result."""

    category: NonEmptyStr
    status_code: StrictInt = Field(ge=400, le=599)
    message: NonEmptyStr


class WhatIfResult(CanonicalModel):
    """One scenario result in a what-if response."""

    scenario: NonEmptyStr
    ok: StrictBool
    response: ForecastResponse | None = None
    error: WhatIfError | None = None

    @model_validator(mode="after")
    def validate_shape(self) -> WhatIfResult:
        if self.ok and self.response is None:
            raise ValueError("response is required when ok is true")
        if self.ok and self.error is not None:
            raise ValueError("error must be null when ok is true")
        if not self.ok and self.error is None:
            raise ValueError("error is required when ok is false")
        if not self.ok and self.response is not None:
            raise ValueError("response must be null when ok is false")
        return self


class WhatIfSummary(CanonicalModel):
    """Summary for what-if endpoint outcomes."""

    requested_scenarios: PositiveInt
    succeeded: StrictInt = Field(ge=0)
    failed: StrictInt = Field(ge=0)

    @model_validator(mode="after")
    def validate_counts(self) -> WhatIfSummary:
        if self.succeeded + self.failed != self.requested_scenarios:
            raise ValueError("succeeded + failed must equal requested_scenarios")
        return self


class WhatIfResponse(CanonicalModel):
    """Response payload for scenario analysis requests."""

    model: NonEmptyStr
    horizon: PositiveInt
    baseline: ForecastResponse
    results: list[WhatIfResult] = Field(min_length=1)
    summary: WhatIfSummary


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
