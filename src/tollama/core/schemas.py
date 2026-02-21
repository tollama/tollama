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
ConfidenceLevel = Literal["high", "medium", "low", "unknown"]
VolatilityChange = Literal["lower", "stable", "higher", "unknown"]
AutoForecastStrategy = Literal["auto", "fastest", "best_accuracy", "ensemble"]
EnsembleMethod = Literal["mean", "median"]
GenerateMethod = Literal["statistical"]
ProgressiveStageStrategy = Literal["fastest", "best_accuracy", "explicit"]
ProgressiveEventStatus = Literal["selected", "running", "completed", "failed"]
ScenarioOperation = Literal["multiply", "add", "replace"]
ScenarioTargetField = Literal["target", "past_covariates", "future_covariates"]
AnomalySeverity = Literal["low", "medium", "high"]
AnomalyType = Literal["spike", "dip", "shift", "trend_break"]
CounterfactualDirection = Literal["above_counterfactual", "below_counterfactual", "neutral"]
TabularFormat = Literal["csv", "parquet"]


def _default_branch_quantiles() -> list[float]:
    return [0.1, 0.5, 0.9]


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


class ResponseOptions(CanonicalModel):
    """Optional response enrichments for deterministic payload extension."""

    narrative: StrictBool = False


class IngestOptions(CanonicalModel):
    """CSV/Parquet ingest options for data_url based forecasting."""

    format: TabularFormat | None = None
    timestamp_column: NonEmptyStr | None = None
    series_id_column: NonEmptyStr | None = None
    target_column: NonEmptyStr | None = None
    freq_column: NonEmptyStr | None = None


class ForecastRequest(CanonicalModel):
    """Unified forecast request payload."""

    model: NonEmptyStr
    horizon: PositiveInt
    modelfile: NonEmptyStr | None = None
    data_url: NonEmptyStr | None = None
    quantiles: list[Quantile] = Field(default_factory=list)
    series: list[SeriesInput] = Field(default_factory=list)
    options: dict[NonEmptyStr, JsonValue] = Field(
        default_factory=dict,
        validation_alias=AliasChoices("options"),
    )
    timeout: float | None = Field(default=None, gt=0.0)
    ingest: IngestOptions | None = None
    parameters: ForecastParameters = Field(
        default_factory=ForecastParameters,
        validation_alias=AliasChoices("parameters"),
    )
    response_options: ResponseOptions = Field(default_factory=ResponseOptions)

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
        if not self.series and self.data_url is None:
            raise ValueError("either series or data_url is required")
        if self.series and self.data_url is not None:
            raise ValueError("series and data_url cannot be combined")

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
    response_options: ResponseOptions = Field(default_factory=ResponseOptions)

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
            "response_options": self.response_options.model_dump(mode="python"),
        }
        ForecastRequest.model_validate(validation_payload)
        return self


class AutoForecastRequest(CanonicalModel):
    """Auto-forecast request payload with optional model override."""

    model: NonEmptyStr | None = None
    allow_fallback: StrictBool = False
    strategy: AutoForecastStrategy = "auto"
    ensemble_top_k: StrictInt = Field(default=3, ge=2, le=8)
    ensemble_method: EnsembleMethod = "mean"
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
    response_options: ResponseOptions = Field(default_factory=ResponseOptions)

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
    response_options: ResponseOptions = Field(default_factory=ResponseOptions)

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


class AnomalyRecord(CanonicalModel):
    """Structured anomaly record with severity and handling hints."""

    index: StrictInt = Field(ge=0)
    type: AnomalyType
    severity: AnomalySeverity
    value: StrictFloat
    range_start: StrictInt = Field(ge=0)
    range_end: StrictInt = Field(ge=0)
    suggested_handling: NonEmptyStr

    @model_validator(mode="after")
    def validate_range(self) -> AnomalyRecord:
        if self.range_start > self.range_end:
            raise ValueError("range_start must be less than or equal to range_end")
        if self.index < self.range_start or self.index > self.range_end:
            raise ValueError("index must be within [range_start, range_end]")
        return self


class SeriesAnalysis(CanonicalModel):
    """Analysis output for one series."""

    id: NonEmptyStr
    detected_frequency: NonEmptyStr
    seasonality_periods: list[PositiveInt] = Field(default_factory=list)
    trend: TrendAnalysis
    anomaly_indices: list[StrictInt] = Field(default_factory=list)
    anomalies: list[AnomalyRecord] = Field(default_factory=list)
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

    @field_validator("anomalies")
    @classmethod
    def validate_anomalies(cls, value: list[AnomalyRecord]) -> list[AnomalyRecord]:
        indices = [item.index for item in value]
        if indices != sorted(indices):
            raise ValueError("anomalies must be sorted by index")
        if len(indices) != len(set(indices)):
            raise ValueError("anomalies must be unique by index")
        return value


class AnalysisNarrativeEntry(CanonicalModel):
    """Deterministic narrative summary for one analyzed series."""

    id: NonEmptyStr
    summary: NonEmptyStr
    trend_direction: TrendDirection
    dominant_seasonality_period: PositiveInt | None = None
    anomaly_count: StrictInt = Field(ge=0)
    data_quality_score: StrictFloat = Field(ge=0.0, le=1.0)
    key_risks: list[NonEmptyStr] = Field(default_factory=list)


class AnalysisNarrative(CanonicalModel):
    """Narrative-ready summary for analyze responses."""

    series: list[AnalysisNarrativeEntry] = Field(min_length=1)


class AnalyzeResponse(CanonicalModel):
    """Response payload for series analysis requests."""

    results: list[SeriesAnalysis] = Field(min_length=1)
    warnings: list[NonEmptyStr] | None = None
    narrative: AnalysisNarrative | None = None

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


class NarrativeTrend(CanonicalModel):
    """Trend summary for one forecast narrative series."""

    direction: TrendDirection
    strength: StrictFloat = Field(ge=0.0, le=1.0)


class NarrativeConfidence(CanonicalModel):
    """Confidence summary for one forecast narrative series."""

    level: ConfidenceLevel
    spread_ratio: StrictFloat | None = Field(default=None, ge=0.0)
    reason: NonEmptyStr


class NarrativeSeasonality(CanonicalModel):
    """Seasonality summary for one forecast narrative series."""

    detected: StrictBool
    period: PositiveInt | None = None


class NarrativeAnomalies(CanonicalModel):
    """Anomaly summary for one forecast narrative series."""

    count: StrictInt = Field(ge=0)
    indices: list[StrictInt] = Field(default_factory=list)

    @field_validator("indices")
    @classmethod
    def validate_indices(cls, value: list[StrictInt]) -> list[StrictInt]:
        if any(index < 0 for index in value):
            raise ValueError("indices must be non-negative")
        if value != sorted(value):
            raise ValueError("indices must be sorted in ascending order")
        if len(value) != len(set(value)):
            raise ValueError("indices must be unique")
        return value


class NarrativeHistoryComparison(CanonicalModel):
    """Forecast-vs-history comparison summary for one forecast narrative series."""

    mean_delta_pct: StrictFloat | None = None
    volatility_change: VolatilityChange


class SeriesForecastNarrative(CanonicalModel):
    """Deterministic narrative summary for one forecasted series."""

    id: NonEmptyStr
    summary: NonEmptyStr
    trend: NarrativeTrend
    confidence: NarrativeConfidence
    seasonality: NarrativeSeasonality
    anomalies: NarrativeAnomalies
    key_insight: NonEmptyStr
    comparison_to_history: NarrativeHistoryComparison


class ForecastNarrative(CanonicalModel):
    """Narrative-ready summary for forecast responses."""

    series: list[SeriesForecastNarrative] = Field(min_length=1)


class ForecastResponse(CanonicalModel):
    """Unified forecast response payload."""

    model: NonEmptyStr
    forecasts: list[SeriesForecast] = Field(min_length=1)
    metrics: ForecastMetrics | None = None
    timing: ForecastTiming | None = None
    explanation: ForecastExplanation | None = None
    narrative: ForecastNarrative | None = None
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


class ProgressiveForecastEvent(CanonicalModel):
    """Structured progressive forecast event payload for SSE streams."""

    event: NonEmptyStr
    stage: PositiveInt
    strategy: ProgressiveStageStrategy
    model: NonEmptyStr
    family: NonEmptyStr | None = None
    status: ProgressiveEventStatus
    final: StrictBool = False
    response: ForecastResponse | None = None
    error: NonEmptyStr | None = None

    @model_validator(mode="after")
    def validate_shape(self) -> ProgressiveForecastEvent:
        if self.status == "completed":
            if self.response is None:
                raise ValueError("response is required when status is 'completed'")
            if self.error is not None:
                raise ValueError("error must be null when status is 'completed'")
        elif self.status == "failed":
            if self.error is None:
                raise ValueError("error is required when status is 'failed'")
            if self.response is not None:
                raise ValueError("response must be null when status is 'failed'")
        else:
            if self.response is not None:
                raise ValueError("response must be null when status is not terminal")
            if self.error is not None:
                raise ValueError("error must be null when status is not terminal")
        return self


class GenerateVariation(CanonicalModel):
    """Variation controls for synthetic time-series generation."""

    level_jitter: StrictFloat = Field(default=0.2, ge=0.0, le=2.0)
    trend_jitter: StrictFloat = Field(default=0.2, ge=0.0, le=2.0)
    seasonality_jitter: StrictFloat = Field(default=0.2, ge=0.0, le=2.0)
    noise_scale: StrictFloat = Field(default=1.0, ge=0.0, le=5.0)
    respect_non_negative: StrictBool = True


class GenerateRequest(CanonicalModel):
    """Request payload for deterministic synthetic series generation."""

    series: list[SeriesInput] = Field(min_length=1, max_length=32)
    count: StrictInt = Field(default=1, ge=1, le=50)
    length: StrictInt | None = Field(default=None, ge=3, le=5000)
    method: GenerateMethod = "statistical"
    seed: StrictInt | None = None
    variation: GenerateVariation = Field(default_factory=GenerateVariation)

    @field_validator("series")
    @classmethod
    def validate_unique_ids(cls, value: list[SeriesInput]) -> list[SeriesInput]:
        ids = [item.id for item in value]
        if len(ids) != len(set(ids)):
            raise ValueError("series ids must be unique")
        return value

    @model_validator(mode="after")
    def validate_series_lengths(self) -> GenerateRequest:
        for series in self.series:
            if len(series.target) < 3:
                raise ValueError(
                    "series.target must include at least 3 points for synthetic generation; "
                    f"series {series.id!r} has {len(series.target)}",
                )
        return self


class GeneratedSeries(CanonicalModel):
    """One generated synthetic series."""

    id: NonEmptyStr
    source_id: NonEmptyStr
    freq: NonEmptyStr
    timestamps: list[NonEmptyStr] = Field(min_length=1)
    target: SequenceValues = Field(min_length=1)

    @model_validator(mode="after")
    def validate_lengths(self) -> GeneratedSeries:
        if len(self.target) != len(self.timestamps):
            raise ValueError("target length must match timestamps length")
        return self


class GenerateResponse(CanonicalModel):
    """Response payload for synthetic generation requests."""

    method: GenerateMethod
    generated: list[GeneratedSeries] = Field(min_length=1)
    warnings: list[NonEmptyStr] | None = None

    @field_validator("generated")
    @classmethod
    def validate_unique_ids(cls, value: list[GeneratedSeries]) -> list[GeneratedSeries]:
        ids = [item.id for item in value]
        if len(ids) != len(set(ids)):
            raise ValueError("generated series ids must be unique")
        return value


class CounterfactualRequest(CanonicalModel):
    """Request payload for intervention counterfactual generation."""

    model: NonEmptyStr
    series: list[SeriesInput] = Field(min_length=1)
    intervention_index: StrictInt = Field(ge=1)
    intervention_label: NonEmptyStr | None = None
    quantiles: list[Quantile] = Field(default_factory=list)
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
    response_options: ResponseOptions = Field(default_factory=ResponseOptions)

    @field_validator("series")
    @classmethod
    def validate_unique_ids(cls, value: list[SeriesInput]) -> list[SeriesInput]:
        ids = [item.id for item in value]
        if len(ids) != len(set(ids)):
            raise ValueError("series ids must be unique")
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
    def validate_intervention_window(self) -> CounterfactualRequest:
        post_lengths: set[int] = set()
        for series in self.series:
            if self.intervention_index >= len(series.target):
                raise ValueError(
                    "intervention_index must be less than each series length; "
                    f"series {series.id!r} has length {len(series.target)}",
                )
            post_lengths.add(len(series.target) - self.intervention_index)

        if len(post_lengths) > 1:
            raise ValueError("all series must share the same post-intervention horizon")
        return self


class CounterfactualSeriesResult(CanonicalModel):
    """Counterfactual divergence payload for one series."""

    id: NonEmptyStr
    actual: SequenceValues = Field(min_length=1)
    counterfactual: SequenceValues = Field(min_length=1)
    delta: SequenceValues = Field(min_length=1)
    absolute_delta: SequenceValues = Field(min_length=1)
    mean_absolute_delta: StrictFloat = Field(ge=0.0)
    total_delta: StrictFloat
    average_delta_pct: StrictFloat | None = None
    direction: CounterfactualDirection

    @model_validator(mode="after")
    def validate_lengths(self) -> CounterfactualSeriesResult:
        expected = len(self.actual)
        if len(self.counterfactual) != expected:
            raise ValueError("counterfactual length must match actual length")
        if len(self.delta) != expected:
            raise ValueError("delta length must match actual length")
        if len(self.absolute_delta) != expected:
            raise ValueError("absolute_delta length must match actual length")
        return self


class CounterfactualResponse(CanonicalModel):
    """Response payload for intervention counterfactual requests."""

    model: NonEmptyStr
    horizon: PositiveInt
    intervention_index: StrictInt = Field(ge=1)
    intervention_label: NonEmptyStr | None = None
    baseline: ForecastResponse
    results: list[CounterfactualSeriesResult] = Field(min_length=1)
    warnings: list[NonEmptyStr] | None = None

    @field_validator("results")
    @classmethod
    def validate_unique_ids(
        cls,
        value: list[CounterfactualSeriesResult],
    ) -> list[CounterfactualSeriesResult]:
        ids = [item.id for item in value]
        if len(ids) != len(set(ids)):
            raise ValueError("results ids must be unique")
        return value


class ScenarioTreeRequest(CanonicalModel):
    """Request payload for probabilistic scenario tree generation."""

    model: NonEmptyStr
    horizon: PositiveInt
    series: list[SeriesInput] = Field(min_length=1)
    depth: StrictInt = Field(default=2, ge=1, le=6)
    branch_quantiles: list[Quantile] = Field(
        default_factory=_default_branch_quantiles,
        min_length=2,
        max_length=7,
    )
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
    response_options: ResponseOptions = Field(default_factory=ResponseOptions)

    @field_validator("series")
    @classmethod
    def validate_unique_ids(cls, value: list[SeriesInput]) -> list[SeriesInput]:
        ids = [item.id for item in value]
        if len(ids) != len(set(ids)):
            raise ValueError("series ids must be unique")
        return value

    @field_validator("branch_quantiles")
    @classmethod
    def validate_branch_quantiles(cls, value: list[Quantile]) -> list[Quantile]:
        if value != sorted(value):
            raise ValueError("branch_quantiles must be sorted in ascending order")
        if len(value) != len(set(value)):
            raise ValueError("branch_quantiles must be unique")
        return value

    @model_validator(mode="after")
    def validate_depth(self) -> ScenarioTreeRequest:
        if self.depth > self.horizon:
            raise ValueError("depth must be less than or equal to horizon")
        return self


class ScenarioTreeNode(CanonicalModel):
    """One node in a flattened probabilistic scenario tree."""

    node_id: NonEmptyStr
    parent_id: NonEmptyStr | None = None
    series_id: NonEmptyStr
    depth: StrictInt = Field(ge=0)
    step: StrictInt = Field(ge=0)
    branch: NonEmptyStr
    quantile: Quantile | None = None
    value: StrictFloat | None = None
    probability: StrictFloat = Field(gt=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_structure(self) -> ScenarioTreeNode:
        if self.depth != self.step:
            raise ValueError("depth and step must match")
        if self.depth == 0:
            if self.parent_id is not None:
                raise ValueError("root node must have parent_id=null")
            return self
        if self.parent_id is None:
            raise ValueError("non-root nodes must have parent_id")
        if self.quantile is None:
            raise ValueError("non-root nodes must include quantile")
        if self.value is None:
            raise ValueError("non-root nodes must include value")
        return self


class ScenarioTreeResponse(CanonicalModel):
    """Response payload for scenario-tree generation."""

    model: NonEmptyStr
    depth: PositiveInt
    branch_quantiles: list[Quantile] = Field(min_length=2)
    nodes: list[ScenarioTreeNode] = Field(min_length=1)
    warnings: list[NonEmptyStr] | None = None

    @field_validator("branch_quantiles")
    @classmethod
    def validate_branch_quantiles(cls, value: list[Quantile]) -> list[Quantile]:
        if value != sorted(value):
            raise ValueError("branch_quantiles must be sorted in ascending order")
        if len(value) != len(set(value)):
            raise ValueError("branch_quantiles must be unique")
        return value

    @field_validator("nodes")
    @classmethod
    def validate_unique_ids(cls, value: list[ScenarioTreeNode]) -> list[ScenarioTreeNode]:
        ids = [item.node_id for item in value]
        if len(ids) != len(set(ids)):
            raise ValueError("nodes must be unique by node_id")
        return value


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
    response_options: ResponseOptions = Field(default_factory=ResponseOptions)

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
            "response_options": self.response_options.model_dump(mode="python"),
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


class CompareNarrativeEntry(CanonicalModel):
    """Narrative ranking entry for one compared model."""

    model: NonEmptyStr
    rank: PositiveInt
    score: StrictFloat | None = None


class CompareNarrative(CanonicalModel):
    """Narrative-ready summary for compare responses."""

    summary: NonEmptyStr
    criterion: NonEmptyStr
    best_model: NonEmptyStr | None = None
    rankings: list[CompareNarrativeEntry] = Field(default_factory=list)


class CompareResponse(CanonicalModel):
    """Response payload for model comparison requests."""

    models: list[NonEmptyStr] = Field(min_length=1)
    horizon: PositiveInt
    results: list[CompareResult] = Field(min_length=1)
    summary: CompareSummary
    narrative: CompareNarrative | None = None


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


class PipelineRequest(CanonicalModel):
    """Full-flow pipeline request payload for analyze/recommend/forecast orchestration."""

    model: NonEmptyStr | None = None
    allow_fallback: StrictBool = False
    strategy: AutoForecastStrategy = "auto"
    ensemble_top_k: StrictInt = Field(default=3, ge=2, le=8)
    ensemble_method: EnsembleMethod = "mean"
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
    analyze_parameters: AnalyzeParameters = Field(default_factory=AnalyzeParameters)
    recommend_top_k: StrictInt = Field(default=3, ge=1, le=20)
    allow_restricted_license: StrictBool = False
    pull_if_missing: StrictBool = True
    accept_license: StrictBool = False
    response_options: ResponseOptions = Field(default_factory=ResponseOptions)

    @field_validator("quantiles")
    @classmethod
    def validate_quantiles(cls, value: list[Quantile]) -> list[Quantile]:
        if value != sorted(value):
            raise ValueError("quantiles must be sorted in ascending order")
        if len(value) != len(set(value)):
            raise ValueError("quantiles must be unique")
        return value

    @model_validator(mode="after")
    def validate_compatibility(self) -> PipelineRequest:
        auto_payload = {
            "model": self.model,
            "allow_fallback": self.allow_fallback,
            "strategy": self.strategy,
            "ensemble_top_k": self.ensemble_top_k,
            "ensemble_method": self.ensemble_method,
            "horizon": self.horizon,
            "quantiles": self.quantiles,
            "series": [series.model_dump(mode="python") for series in self.series],
            "options": self.options,
            "timeout": self.timeout,
            "keep_alive": self.keep_alive,
            "parameters": self.parameters.model_dump(mode="python"),
            "response_options": self.response_options.model_dump(mode="python"),
        }
        AutoForecastRequest.model_validate(auto_payload)
        AnalyzeRequest.model_validate(
            {
                "series": [series.model_dump(mode="python") for series in self.series],
                "parameters": self.analyze_parameters.model_dump(mode="python"),
                "response_options": self.response_options.model_dump(mode="python"),
            },
        )
        return self


class PipelineNarrative(CanonicalModel):
    """Narrative-ready summary for pipeline responses."""

    summary: NonEmptyStr
    chosen_model: NonEmptyStr
    pulled_model: NonEmptyStr | None = None
    warnings_count: StrictInt = Field(ge=0)


class PipelineResponse(CanonicalModel):
    """Response payload for one full autonomous forecasting pipeline run."""

    analysis: AnalyzeResponse
    recommendation: dict[NonEmptyStr, JsonValue]
    pulled_model: NonEmptyStr | None = None
    auto_forecast: AutoForecastResponse
    warnings: list[NonEmptyStr] | None = None
    narrative: PipelineNarrative | None = None


class ReportRequest(PipelineRequest):
    """Composite report request payload."""

    include_baseline: StrictBool = True


class ReportNarrative(CanonicalModel):
    """Narrative-ready summary for report responses."""

    summary: NonEmptyStr
    chosen_model: NonEmptyStr
    anomaly_count: StrictInt = Field(ge=0)
    key_insights: list[NonEmptyStr] = Field(default_factory=list)
    warnings_count: StrictInt = Field(ge=0)


class ForecastReport(CanonicalModel):
    """Composite report response payload."""

    analysis: AnalyzeResponse
    recommendation: dict[NonEmptyStr, JsonValue]
    forecast: AutoForecastResponse
    baseline: ForecastResponse | None = None
    metrics: ForecastMetrics | None = None
    warnings: list[NonEmptyStr] | None = None
    narrative: ReportNarrative | None = None


class DashboardStateWarning(CanonicalModel):
    """Warning entry for partial dashboard-state aggregation failures."""

    source: Literal["info", "ps", "usage"]
    status_code: StrictInt = Field(ge=400, le=599)
    detail: NonEmptyStr


class DashboardStateResponse(CanonicalModel):
    """Aggregated payload for dashboard bootstrap state."""

    info: dict[NonEmptyStr, JsonValue] | None = None
    ps: dict[NonEmptyStr, JsonValue] | None = None
    usage: dict[NonEmptyStr, JsonValue] | None = None
    warnings: list[DashboardStateWarning] = Field(default_factory=list)


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


def _default_field_description(field_name: str) -> str:
    readable = field_name.replace("_", " ").strip()
    if not readable:
        return "Field value."
    return f"{readable[0].upper()}{readable[1:]}."


def _populate_missing_field_descriptions() -> None:
    """Populate missing model field descriptions for OpenAPI schema generation."""
    for candidate in globals().values():
        if not isinstance(candidate, type) or not issubclass(candidate, BaseModel):
            continue
        updated = False
        for field_name, field in candidate.model_fields.items():
            if field.description is not None:
                continue
            field.description = _default_field_description(field_name)
            updated = True
        if updated:
            candidate.model_rebuild(force=True)


_populate_missing_field_descriptions()
