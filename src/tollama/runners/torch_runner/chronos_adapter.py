"""Chronos forecasting adapter used by the torch runner."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from tollama.core.schemas import ForecastRequest, ForecastResponse, SeriesForecast, SeriesInput

CHRONOS_MODEL_REGISTRY = {
    "chronos2": {
        "repo_id": "amazon/chronos-2",
        "revision": "main",
    },
}


class DependencyMissingError(RuntimeError):
    """Raised when optional torch runner dependencies are missing."""


class UnsupportedModelError(ValueError):
    """Raised when a request targets an unsupported torch runner model."""


@dataclass(frozen=True)
class _ChronosDependencies:
    chronos_pipeline: Any
    pandas: Any


class ChronosAdapter:
    """Adapter that maps canonical requests/responses to Chronos2 predict_df I/O."""

    def __init__(self) -> None:
        self._dependencies: _ChronosDependencies | None = None
        self._pipelines: dict[str, Any] = {}

    def load(self, model_name: str) -> None:
        """Load one model pipeline into memory if needed."""
        model_source = _source_for_model(model_name)
        if model_name in self._pipelines:
            return

        dependencies = self._resolve_dependencies()
        pipeline_cls = dependencies.chronos_pipeline
        try:
            pipeline = pipeline_cls.from_pretrained(
                model_source["repo_id"],
                revision=model_source["revision"],
            )
        except TypeError:
            pipeline = pipeline_cls.from_pretrained(model_source["repo_id"])
        self._pipelines[model_name] = pipeline

    def unload(self, model_name: str | None = None) -> None:
        """Unload one model or all models from memory."""
        if model_name is None:
            self._pipelines.clear()
            return
        self._pipelines.pop(model_name, None)

    def forecast(self, request: ForecastRequest) -> ForecastResponse:
        """Generate a forecast from canonical request data."""
        self.load(request.model)
        dependencies = self._resolve_dependencies()
        pandas = dependencies.pandas
        pipeline = self._pipelines[request.model]

        context_df, future_df = _build_chronos_frames(pandas=pandas, request=request)
        pred_df = pipeline.predict_df(
            context_df,
            future_df=future_df,
            prediction_length=request.horizon,
            quantile_levels=list(request.quantiles),
            id_column="id",
            timestamp_column="timestamp",
            target="target",
        )

        forecasts = _response_forecasts_from_pred_df(
            request=request,
            pred_df=pred_df,
        )
        return ForecastResponse(
            model=request.model,
            forecasts=forecasts,
            usage={
                "runner": "tollama-torch",
                "series_count": len(request.series),
                "horizon": request.horizon,
            },
        )

    def _resolve_dependencies(self) -> _ChronosDependencies:
        cached = self._dependencies
        if cached is not None:
            return cached

        missing_packages: list[str] = []
        try:
            import pandas as pd
        except ModuleNotFoundError:
            missing_packages.append("pandas")
            pd = None

        try:
            import numpy as np  # noqa: F401
        except ModuleNotFoundError:
            missing_packages.append("numpy")

        try:
            from chronos import Chronos2Pipeline
        except ModuleNotFoundError as exc:
            missing_name = exc.name or "chronos-forecasting"
            missing_packages.append(missing_name)
            Chronos2Pipeline = None

        if missing_packages:
            unique = sorted(set(missing_packages))
            joined = ", ".join(unique)
            raise DependencyMissingError(
                "missing optional torch runner dependencies "
                f"({joined}); install them with `pip install -e \".[dev,runner_torch]\"`",
            )

        assert pd is not None
        assert Chronos2Pipeline is not None
        resolved = _ChronosDependencies(chronos_pipeline=Chronos2Pipeline, pandas=pd)
        self._dependencies = resolved
        return resolved


def _source_for_model(model_name: str) -> dict[str, str]:
    try:
        return CHRONOS_MODEL_REGISTRY[model_name]
    except KeyError as exc:
        supported = ", ".join(sorted(CHRONOS_MODEL_REGISTRY))
        raise UnsupportedModelError(
            f"unsupported torch runner model {model_name!r}; supported models: {supported}",
        ) from exc


def _build_chronos_frames(
    *,
    pandas: Any,
    request: ForecastRequest,
) -> tuple[Any, Any | None]:
    context_rows: list[dict[str, Any]] = []
    future_rows: list[dict[str, Any]] = []

    for series in request.series:
        timestamps = pandas.to_datetime(series.timestamps, utc=True, errors="raise")
        history_length = len(series.timestamps)

        for index in range(history_length):
            row: dict[str, Any] = {
                "id": series.id,
                "timestamp": timestamps[index],
                "target": _to_python_number(series.target[index]),
            }
            if series.past_covariates:
                for name, values in series.past_covariates.items():
                    row[name] = _to_python_number(values[index])
            if series.future_covariates:
                for name, values in series.future_covariates.items():
                    row[name] = _to_python_number(values[index])
            context_rows.append(row)

        if series.future_covariates:
            _validate_future_covariates(series=series, horizon=request.horizon)
            future_timestamps = _future_timestamps(
                pandas=pandas,
                last_timestamp=timestamps[-1],
                freq=series.freq,
                horizon=request.horizon,
            )
            for step in range(request.horizon):
                future_row: dict[str, Any] = {
                    "id": series.id,
                    "timestamp": future_timestamps[step],
                }
                for name, values in series.future_covariates.items():
                    offset = history_length + step
                    future_row[name] = _to_python_number(values[offset])
                future_rows.append(future_row)

    context_df = pandas.DataFrame(context_rows)
    future_df = pandas.DataFrame(future_rows) if future_rows else None
    return context_df, future_df


def _validate_future_covariates(*, series: SeriesInput, horizon: int) -> None:
    future_covariates = series.future_covariates
    if not future_covariates:
        return

    min_length = len(series.timestamps) + horizon
    for name, values in future_covariates.items():
        if len(values) < min_length:
            raise ValueError(
                f"future_covariates[{name!r}] must include at least "
                f"timestamps + horizon values ({min_length})",
            )


def _future_timestamps(
    *,
    pandas: Any,
    last_timestamp: Any,
    freq: str,
    horizon: int,
) -> list[Any]:
    try:
        date_range = pandas.date_range(
            start=last_timestamp,
            periods=horizon + 1,
            freq=freq,
        )
    except ValueError as exc:
        raise ValueError(f"invalid frequency {freq!r} for Chronos forecast") from exc
    return list(date_range[1:])


def _response_forecasts_from_pred_df(
    *,
    request: ForecastRequest,
    pred_df: Any,
) -> list[SeriesForecast]:
    if pred_df is None:
        raise ValueError("Chronos predict_df returned no predictions")
    required_columns = {"id", "timestamp", "predictions"}
    missing_columns = required_columns - set(pred_df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Chronos predict_df output missing required columns: {missing}")

    grouped = pred_df.groupby("id")
    forecasts: list[SeriesForecast] = []
    for series in request.series:
        if series.id not in grouped.groups:
            raise ValueError(f"Chronos predict_df missing predictions for series id {series.id!r}")

        frame = grouped.get_group(series.id).sort_values("timestamp")
        mean = [_to_python_number(value) for value in frame["predictions"].tolist()]
        quantiles_payload: dict[str, list[int | float]] | None = None

        if request.quantiles:
            quantiles_payload = {}
            for quantile in request.quantiles:
                column_name = str(quantile)
                if column_name not in frame.columns:
                    continue
                quantiles_payload[format(quantile, "g")] = [
                    _to_python_number(value) for value in frame[column_name].tolist()
                ]
            if not quantiles_payload:
                quantiles_payload = None

        first_timestamp = frame.iloc[0]["timestamp"]
        forecasts.append(
            SeriesForecast(
                id=series.id,
                freq=series.freq,
                start_timestamp=_to_iso_timestamp(first_timestamp),
                mean=mean,
                quantiles=quantiles_payload,
            ),
        )

    return forecasts


def _to_iso_timestamp(value: Any) -> str:
    if isinstance(value, datetime):
        return value.astimezone(UTC).isoformat().replace("+00:00", "Z")
    return str(value)


def _to_python_number(value: Any) -> int | float:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return value
    if hasattr(value, "item"):
        scalar = value.item()
        if isinstance(scalar, (int, float)):
            return scalar
    raise ValueError(f"non-numeric value in forecast payload: {value!r}")
