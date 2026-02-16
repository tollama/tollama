"""Granite TTM forecasting adapter used by the torch runner."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tollama.core.schemas import ForecastRequest, ForecastResponse, SeriesForecast

from .errors import AdapterInputError, DependencyMissingError, UnsupportedModelError

GRANITE_MODEL_REGISTRY: dict[str, dict[str, Any]] = {
    "granite-ttm-r2": {
        "repo_id": "ibm-granite/granite-timeseries-ttm-r2",
        "revision": "90-30-ft-l1-r2.1",
        "context_length": 90,
        "prediction_length": 30,
        "implementation": "granite_ttm",
    },
}


@dataclass(frozen=True)
class _GraniteDependencies:
    torch: Any
    pandas: Any
    forecasting_pipeline_cls: Any
    preprocessor_cls: Any
    model_cls: Any


@dataclass(frozen=True)
class _LoadedGraniteModel:
    model: Any
    repo_id: str
    revision: str
    context_length: int
    prediction_length: int


class GraniteTTMAdapter:
    """Adapter that maps canonical request/response to Granite TTM inference."""

    def __init__(self) -> None:
        self._dependencies: _GraniteDependencies | None = None
        self._models: dict[str, _LoadedGraniteModel] = {}

    def load(
        self,
        model_name: str,
        *,
        model_local_dir: str | None = None,
        model_source: dict[str, Any] | None = None,
        model_metadata: dict[str, Any] | None = None,
    ) -> None:
        """Load one Granite TTM model and preprocessor into memory."""
        if model_name in self._models:
            return

        dependencies = self._resolve_dependencies()
        repo_id, revision, context_length, prediction_length = _resolve_model_runtime_config(
            model_name=model_name,
            model_source=model_source,
            model_metadata=model_metadata,
        )

        model_path = _existing_local_model_path(model_local_dir)
        if model_path is not None:
            model = dependencies.model_cls.from_pretrained(model_path)
        else:
            model = dependencies.model_cls.from_pretrained(repo_id, revision=revision)

        self._models[model_name] = _LoadedGraniteModel(
            model=model,
            repo_id=repo_id,
            revision=revision,
            context_length=context_length,
            prediction_length=prediction_length,
        )

    def unload(self, model_name: str | None = None) -> None:
        """Unload one model or all loaded Granite models."""
        if model_name is None:
            self._models.clear()
            return
        self._models.pop(model_name, None)

    def forecast(
        self,
        request: ForecastRequest,
        *,
        model_local_dir: str | None = None,
        model_source: dict[str, Any] | None = None,
        model_metadata: dict[str, Any] | None = None,
    ) -> ForecastResponse:
        """Generate a Granite TTM point forecast for one canonical request."""
        if len(request.series) != 1:
            raise AdapterInputError(
                "granite_ttm currently supports exactly one input series per request",
            )

        self.load(
            request.model,
            model_local_dir=model_local_dir,
            model_source=model_source,
            model_metadata=model_metadata,
        )
        dependencies = self._resolve_dependencies()
        loaded = self._models[request.model]
        series = request.series[0]

        if request.horizon > loaded.prediction_length:
            raise AdapterInputError(
                "Requested horizon exceeds model prediction_length. "
                "Pull a different TTM revision.",
            )

        if (
            len(series.timestamps) < loaded.context_length
            or len(series.target) < loaded.context_length
        ):
            raise AdapterInputError(
                "input series length is shorter than model context_length "
                f"({loaded.context_length})",
            )

        timestamps = dependencies.pandas.to_datetime(series.timestamps, utc=True, errors="raise")
        past_covariates = series.past_covariates or {}
        future_covariates = series.future_covariates or {}
        known_future_columns = sorted(set(past_covariates).intersection(future_covariates))
        past_only_columns = sorted(set(past_covariates) - set(known_future_columns))
        future_only = set(future_covariates) - set(past_covariates)
        if future_only:
            first = sorted(future_only)[0]
            raise AdapterInputError(
                f"future_covariates[{first!r}] must also be present in past_covariates "
                f"for series {series.id!r}",
            )

        context_rows = [
            {
                "id": series.id,
                "timestamp": timestamps[index],
                "target": float(series.target[index]),
                **{
                    name: _to_numeric_covariate(past_covariates[name][index], covariate=name)
                    for name in sorted(past_covariates)
                },
            }
            for index in range(len(series.timestamps))
        ]
        context_df = dependencies.pandas.DataFrame(context_rows)
        context_tail = context_df.tail(loaded.context_length).copy()

        preprocessor = _build_preprocessor(
            dependencies=dependencies,
            context_length=loaded.context_length,
            prediction_length=loaded.prediction_length,
            control_columns=known_future_columns,
            conditional_columns=past_only_columns,
        )

        pipeline = dependencies.forecasting_pipeline_cls(
            loaded.model,
            device=_resolve_device(
                torch_module=dependencies.torch,
                requested_device=_requested_device(request.options),
            ),
            feature_extractor=preprocessor,
            batch_size=1,
        )
        future_time_series = _build_future_time_series(
            series=series,
            pandas=dependencies.pandas,
            timestamps=timestamps,
            horizon=request.horizon,
            known_future_columns=known_future_columns,
        )
        if future_time_series is None:
            pred_df = pipeline(context_tail)
        else:
            pred_df = pipeline(context_tail, future_time_series=future_time_series)

        prediction_column = _prediction_column_name(pred_df)
        raw_vector = pred_df.iloc[-1][prediction_column]
        prediction_values = _to_float_list(raw_vector)
        if len(prediction_values) < request.horizon:
            raise AdapterInputError(
                "Granite TTM prediction output length is shorter than requested horizon",
            )

        mean = prediction_values[: request.horizon]
        future_timestamps = dependencies.pandas.date_range(
            start=timestamps[-1],
            periods=request.horizon + 1,
            freq=series.freq,
        )
        start_timestamp = _to_iso_timestamp(future_timestamps[1])

        return ForecastResponse(
            model=request.model,
            forecasts=[
                SeriesForecast(
                    id=series.id,
                    freq=series.freq,
                    start_timestamp=start_timestamp,
                    mean=mean,
                    quantiles=None,
                )
            ],
            usage={
                "runner": "tollama-torch",
                "implementation": "granite_ttm",
                "series_count": 1,
                "horizon": request.horizon,
                "control_columns": known_future_columns,
                "conditional_columns": past_only_columns,
            },
        )

    def _resolve_dependencies(self) -> _GraniteDependencies:
        cached = self._dependencies
        if cached is not None:
            return cached

        missing_packages: list[str] = []
        try:
            import torch
        except ModuleNotFoundError:
            missing_packages.append("torch")
            torch = None

        try:
            import pandas as pd
        except ModuleNotFoundError:
            missing_packages.append("pandas")
            pd = None

        try:
            from tsfm_public import (
                TimeSeriesForecastingPipeline,
                TimeSeriesPreprocessor,
                TinyTimeMixerForPrediction,
            )
        except ModuleNotFoundError as exc:
            missing_packages.append(exc.name or "granite-tsfm")
            TimeSeriesForecastingPipeline = None
            TimeSeriesPreprocessor = None
            TinyTimeMixerForPrediction = None

        if missing_packages:
            unique = ", ".join(sorted(set(missing_packages)))
            raise DependencyMissingError(
                "missing optional torch runner dependencies "
                f"({unique}); install them with `pip install -e \".[dev,runner_torch]\"`",
            )

        assert torch is not None
        assert pd is not None
        assert TimeSeriesForecastingPipeline is not None
        assert TimeSeriesPreprocessor is not None
        assert TinyTimeMixerForPrediction is not None

        resolved = _GraniteDependencies(
            torch=torch,
            pandas=pd,
            forecasting_pipeline_cls=TimeSeriesForecastingPipeline,
            preprocessor_cls=TimeSeriesPreprocessor,
            model_cls=TinyTimeMixerForPrediction,
        )
        self._dependencies = resolved
        return resolved


def _resolve_model_runtime_config(
    *,
    model_name: str,
    model_source: dict[str, Any] | None,
    model_metadata: dict[str, Any] | None,
) -> tuple[str, str, int, int]:
    base = GRANITE_MODEL_REGISTRY.get(model_name)
    if base is None:
        raise UnsupportedModelError(
            f"unsupported granite_ttm model {model_name!r}; "
            f"supported models: {', '.join(sorted(GRANITE_MODEL_REGISTRY))}",
        )

    repo_id = _dict_str(model_source, "repo_id") or str(base["repo_id"])
    revision = _dict_str(model_source, "revision") or str(base["revision"])
    context_length = _dict_positive_int(model_metadata, "context_length") or int(
        base["context_length"],
    )
    prediction_length = _dict_positive_int(model_metadata, "prediction_length") or int(
        base["prediction_length"],
    )
    return repo_id, revision, context_length, prediction_length


def _prediction_column_name(pred_df: Any) -> str:
    columns = list(getattr(pred_df, "columns", []))
    if "target_prediction" in columns:
        return "target_prediction"
    for column in columns:
        if isinstance(column, str) and column.endswith("_prediction"):
            return column
    raise AdapterInputError("Granite TTM output is missing a prediction column")


def _requested_device(options: dict[str, Any]) -> str | None:
    value = options.get("device")
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower()
    if not normalized:
        return None
    return normalized


def _resolve_device(*, torch_module: Any, requested_device: str | None) -> str:
    if requested_device == "cpu":
        return "cpu"
    cuda = getattr(torch_module, "cuda", None)
    is_available = getattr(cuda, "is_available", None)
    if callable(is_available) and bool(is_available()):
        return "cuda"
    return "cpu"


def _to_float_list(value: Any) -> list[float]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (tuple, list)):
        numbers: list[float] = []
        for item in value:
            if isinstance(item, bool):
                numbers.append(float(int(item)))
                continue
            if isinstance(item, (int, float)):
                numbers.append(float(item))
                continue
            if hasattr(item, "item"):
                scalar = item.item()
                if isinstance(scalar, (int, float)):
                    numbers.append(float(scalar))
                    continue
            raise AdapterInputError(f"non-numeric prediction value: {item!r}")
        return numbers
    raise AdapterInputError("Granite TTM prediction output is not list-like")


def _build_preprocessor(
    *,
    dependencies: _GraniteDependencies,
    context_length: int,
    prediction_length: int,
    control_columns: list[str],
    conditional_columns: list[str],
) -> Any:
    kwargs: dict[str, Any] = {
        "timestamp_column": "timestamp",
        "id_columns": ["id"],
        "target_columns": ["target"],
        "observable_columns": [],
        "control_columns": control_columns,
        "conditional_columns": conditional_columns,
        "context_length": context_length,
        "prediction_length": prediction_length,
        "scaling": True,
        "encode_categorical": False,
        "scaler_type": "standard",
    }
    try:
        return dependencies.preprocessor_cls(**kwargs)
    except TypeError:
        # Backward-compatible fallback for older preprocessor signatures.
        fallback_kwargs = dict(kwargs)
        fallback_kwargs.pop("control_columns", None)
        fallback_kwargs.pop("conditional_columns", None)
        fallback_kwargs["observable_columns"] = sorted(
            set(control_columns).union(conditional_columns),
        )
        return dependencies.preprocessor_cls(**fallback_kwargs)


def _build_future_time_series(
    *,
    series: Any,
    pandas: Any,
    timestamps: Any,
    horizon: int,
    known_future_columns: list[str],
) -> Any | None:
    if not known_future_columns:
        return None

    future_covariates = series.future_covariates or {}
    future_timestamps = pandas.date_range(
        start=timestamps[-1],
        periods=horizon + 1,
        freq=series.freq,
    )
    rows: list[dict[str, Any]] = []
    for step in range(horizon):
        row: dict[str, Any] = {
            "id": series.id,
            "timestamp": future_timestamps[step + 1],
        }
        for name in known_future_columns:
            values = future_covariates[name]
            row[name] = _to_numeric_covariate(values[step], covariate=name)
        rows.append(row)
    return pandas.DataFrame(rows)


def _to_numeric_covariate(value: Any, *, covariate: str) -> float:
    if isinstance(value, bool):
        return float(int(value))
    if isinstance(value, (int, float)):
        return float(value)
    if hasattr(value, "item"):
        scalar = value.item()
        if isinstance(scalar, bool):
            return float(int(scalar))
        if isinstance(scalar, (int, float)):
            return float(scalar)
    raise AdapterInputError(
        f"Granite TTM supports only numeric covariates; got {value!r} for {covariate!r}",
    )


def _to_iso_timestamp(value: Any) -> str:
    if isinstance(value, datetime):
        return value.astimezone(UTC).isoformat().replace("+00:00", "Z")
    return str(value)


def _existing_local_model_path(model_local_dir: str | None) -> str | None:
    if not model_local_dir:
        return None
    normalized = model_local_dir.strip()
    if not normalized:
        return None
    path = Path(normalized)
    if not path.exists():
        return None
    return str(path)


def _dict_str(payload: dict[str, Any] | None, key: str) -> str | None:
    if not isinstance(payload, dict):
        return None
    value = payload.get(key)
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    if not normalized:
        return None
    return normalized


def _dict_positive_int(payload: dict[str, Any] | None, key: str) -> int | None:
    if not isinstance(payload, dict):
        return None
    value = payload.get(key)
    if not isinstance(value, int):
        return None
    if value <= 0:
        return None
    return value
