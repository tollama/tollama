"""Moirai forecasting adapter used by the uni2ts runner."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tollama.core.schemas import ForecastRequest, ForecastResponse, SeriesForecast, SeriesInput

from .errors import AdapterInputError, DependencyMissingError, UnsupportedModelError

_DEFAULT_CONTEXT_LENGTH = 1680
_DEFAULT_BATCH_SIZE = 32
_UNI2TS_MODELS: dict[str, dict[str, Any]] = {
    "moirai-2.0-R-small": {
        "repo_id": "Salesforce/moirai-2.0-R-small",
        "revision": "main",
        "implementation": "moirai_2p0",
        "default_context_length": 1680,
    },
}


@dataclass(frozen=True)
class _Uni2TSDependencies:
    pandas: Any
    numpy: Any
    pandas_dataset_cls: Any
    moirai_forecast_cls: Any
    moirai_module_cls: Any


@dataclass(frozen=True)
class _RuntimeConfig:
    model_name: str
    repo_id: str
    revision: str
    implementation: str
    default_context_length: int


@dataclass(frozen=True)
class _PredictorConfig:
    prediction_length: int
    context_length: int
    batch_size: int


@dataclass(frozen=True)
class _DatasetBuildResult:
    dataset: Any
    ordered_series: list[SeriesInput]
    feat_dynamic_real_dim: int
    past_feat_dynamic_real_dim: int


class MoiraiAdapter:
    """Adapter that maps canonical request/response to Moirai probabilistic inference."""

    def __init__(self) -> None:
        self._dependencies: _Uni2TSDependencies | None = None
        self._module_cache: dict[tuple[str, str, str], Any] = {}

    def load(
        self,
        model_name: str,
        *,
        model_local_dir: str | None = None,
        model_source: dict[str, Any] | None = None,
        model_metadata: dict[str, Any] | None = None,
    ) -> None:
        """Load one Moirai module into cache."""
        runtime = _resolve_runtime_config(
            model_name=model_name,
            model_source=model_source,
            model_metadata=model_metadata,
        )
        self._get_or_load_module(runtime=runtime, model_local_dir=model_local_dir)

    def unload(self, model_name: str | None = None) -> None:
        """Unload one cached module or clear all cached modules."""
        if model_name is None:
            self._module_cache.clear()
            return
        self._module_cache = {
            key: value
            for key, value in self._module_cache.items()
            if key[0] != model_name
        }

    def forecast(
        self,
        request: ForecastRequest,
        *,
        model_local_dir: str | None = None,
        model_source: dict[str, Any] | None = None,
        model_metadata: dict[str, Any] | None = None,
    ) -> ForecastResponse:
        """Generate probabilistic Moirai forecasts mapped to canonical response schema."""
        runtime = _resolve_runtime_config(
            model_name=request.model,
            model_source=model_source,
            model_metadata=model_metadata,
        )
        dependencies = self._resolve_dependencies()
        module = self._get_or_load_module(runtime=runtime, model_local_dir=model_local_dir)

        if "num_samples" in request.options:
            raise AdapterInputError("moirai-2.0 does not support 'num_samples' execution option")
        if "patch_size" in request.options:
            raise AdapterInputError("moirai-2.0 does not support 'patch_size' execution option")

        context_length = resolve_context_length(
            option_value=request.options.get("context_length"),
            default_context_length=runtime.default_context_length,
            series_list=request.series,
        )
        batch_size = resolve_positive_int(
            option_value=request.options.get("batch_size"),
            default_value=_DEFAULT_BATCH_SIZE,
            field_name="batch_size",
        )
        predictor_config = _PredictorConfig(
            prediction_length=request.horizon,
            context_length=context_length,
            batch_size=batch_size,
        )

        dataset_result = build_pandas_dataset(
            series_list=request.series,
            pandas_module=dependencies.pandas,
            numpy_module=dependencies.numpy,
            pandas_dataset_cls=dependencies.pandas_dataset_cls,
            horizon=request.horizon,
        )
        dataset = dataset_result.dataset
        ordered_series = dataset_result.ordered_series
        feat_dynamic_real_dim = dataset_result.feat_dynamic_real_dim
        past_feat_dynamic_real_dim = dataset_result.past_feat_dynamic_real_dim
        if hasattr(dataset, "num_feat_dynamic_real"):
            value = getattr(dataset, "num_feat_dynamic_real")
            if isinstance(value, int):
                feat_dynamic_real_dim = value
        if hasattr(dataset, "num_past_feat_dynamic_real"):
            value = getattr(dataset, "num_past_feat_dynamic_real")
            if isinstance(value, int):
                past_feat_dynamic_real_dim = value

        moirai_model = dependencies.moirai_forecast_cls(
            module=module,
            prediction_length=predictor_config.prediction_length,
            context_length=predictor_config.context_length,
            target_dim=1,
            feat_dynamic_real_dim=feat_dynamic_real_dim,
            past_feat_dynamic_real_dim=past_feat_dynamic_real_dim,
        )
        predictor = moirai_model.create_predictor(batch_size=predictor_config.batch_size)
        forecast_iter = predictor.predict(dataset)
        forecasts = list(forecast_iter)
        if len(forecasts) != len(ordered_series):
            raise AdapterInputError(
                "Moirai predictor returned a different number of series forecasts than requested",
            )

        payloads: list[SeriesForecast] = []
        for series, forecast in zip(ordered_series, forecasts, strict=True):
            mean = normalize_forecast_vector(forecast.mean, horizon=request.horizon, label="mean")
            quantiles = (
                build_quantile_payload(
                    forecast=forecast,
                    requested_quantiles=list(request.quantiles),
                    horizon=request.horizon,
                )
                if request.quantiles
                else None
            )
            start_timestamp = resolve_forecast_start_timestamp(
                forecast=forecast,
                series=series,
                pandas_module=dependencies.pandas,
                horizon=request.horizon,
            )
            payloads.append(
                SeriesForecast(
                    id=series.id,
                    freq=series.freq,
                    start_timestamp=start_timestamp,
                    mean=mean,
                    quantiles=quantiles,
                ),
            )

        return ForecastResponse(
            model=request.model,
            forecasts=payloads,
            usage={
                "runner": "tollama-uni2ts",
                "implementation": runtime.implementation,
                "series_count": len(payloads),
                "horizon": request.horizon,
                "context_length": context_length,
            },
        )

    def _get_or_load_module(
        self,
        *,
        runtime: _RuntimeConfig,
        model_local_dir: str | None,
    ) -> Any:
        dependencies = self._resolve_dependencies()
        local_model_path = _existing_local_model_path(model_local_dir)
        model_ref = local_model_path or runtime.repo_id
        module_key = (runtime.model_name, model_ref, runtime.revision)
        cached = self._module_cache.get(module_key)
        if cached is not None:
            return cached

        if local_model_path is not None:
            try:
                module = dependencies.moirai_module_cls.from_pretrained(local_model_path)
            except TypeError:
                module = dependencies.moirai_module_cls.from_pretrained(local_model_path)
        else:
            try:
                module = dependencies.moirai_module_cls.from_pretrained(
                    runtime.repo_id,
                    revision=runtime.revision,
                )
            except TypeError:
                module = dependencies.moirai_module_cls.from_pretrained(runtime.repo_id)
        self._module_cache[module_key] = module
        return module

    def _resolve_dependencies(self) -> _Uni2TSDependencies:
        cached = self._dependencies
        if cached is not None:
            return cached

        missing_packages: list[str] = []
        try:
            import numpy as np
        except ModuleNotFoundError:
            missing_packages.append("numpy")
            np = None

        try:
            import pandas as pd
        except ModuleNotFoundError:
            missing_packages.append("pandas")
            pd = None

        try:
            from gluonts.dataset.pandas import PandasDataset
        except ModuleNotFoundError as exc:
            missing_packages.append(exc.name or "gluonts")
            PandasDataset = None

        try:
            from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module
        except ModuleNotFoundError as exc:
            missing_packages.append(exc.name or "uni2ts")
            MoiraiForecast = None
            MoiraiModule = None
        else:
            MoiraiForecast = Moirai2Forecast
            MoiraiModule = Moirai2Module

        if missing_packages:
            joined = ", ".join(sorted(set(missing_packages)))
            raise DependencyMissingError(
                "missing optional uni2ts runner dependencies "
                f"({joined}); install them with `pip install -e \".[dev,runner_uni2ts]\"`",
            )

        assert np is not None
        assert pd is not None
        assert PandasDataset is not None
        assert MoiraiForecast is not None
        assert MoiraiModule is not None
        resolved = _Uni2TSDependencies(
            pandas=pd,
            numpy=np,
            pandas_dataset_cls=PandasDataset,
            moirai_forecast_cls=MoiraiForecast,
            moirai_module_cls=MoiraiModule,
        )
        self._dependencies = resolved
        return resolved


def resolve_context_length(
    *,
    option_value: Any,
    default_context_length: int,
    series_list: Sequence[SeriesInput],
) -> int:
    """Resolve context length from options/defaults and clamp to available history."""
    candidate = resolve_positive_int(
        option_value=option_value,
        default_value=(
            default_context_length if default_context_length > 0 else _DEFAULT_CONTEXT_LENGTH
        ),
        field_name="context_length",
    )
    min_length = min(len(series.target) for series in series_list) if series_list else 0
    if min_length < 2:
        raise AdapterInputError("each input series must include at least two target points")
    return max(2, min(candidate, min_length))


def resolve_positive_int(*, option_value: Any, default_value: int, field_name: str) -> int:
    """Resolve one positive integer option or use the provided default."""
    if option_value is None:
        if default_value <= 0:
            raise AdapterInputError(f"{field_name} must be greater than zero")
        return default_value
    if isinstance(option_value, bool) or not isinstance(option_value, int):
        raise AdapterInputError(f"{field_name} option must be an integer")
    if option_value <= 0:
        raise AdapterInputError(f"{field_name} option must be greater than zero")
    return option_value


def build_pandas_dataset(
    *,
    series_list: Sequence[SeriesInput],
    pandas_module: Any,
    numpy_module: Any,
    pandas_dataset_cls: Any,
    horizon: int = 1,
) -> _DatasetBuildResult:
    """Build a GluonTS PandasDataset with dynamic covariate feature fields."""
    return build_pandas_dataset_with_horizon(
        series_list=series_list,
        pandas_module=pandas_module,
        numpy_module=numpy_module,
        pandas_dataset_cls=pandas_dataset_cls,
        horizon=horizon,
    )


def build_pandas_dataset_with_horizon(
    *,
    series_list: Sequence[SeriesInput],
    pandas_module: Any,
    numpy_module: Any,
    pandas_dataset_cls: Any,
    horizon: int,
) -> _DatasetBuildResult:
    """Build a GluonTS PandasDataset from canonical series payloads."""
    if horizon <= 0:
        raise AdapterInputError("horizon must be greater than zero")

    frames: dict[str, Any] = {}
    known_future_columns: set[str] = set()
    past_only_columns: set[str] = set()

    for index, series in enumerate(series_list):
        if len(series.timestamps) != len(series.target):
            raise AdapterInputError(
                f"series {series.id!r} timestamps and target lengths must match",
            )
        timestamps = pandas_module.to_datetime(series.timestamps, utc=True, errors="raise")
        if len(timestamps) < 2:
            raise AdapterInputError(
                f"series {series.id!r} must include at least two timestamps",
            )

        past_covariates = series.past_covariates or {}
        future_covariates = series.future_covariates or {}
        known_future = set(past_covariates).intersection(future_covariates)
        future_only = set(future_covariates) - set(past_covariates)
        if future_only:
            first = sorted(future_only)[0]
            raise AdapterInputError(
                f"future_covariates[{first!r}] must also be present in past_covariates "
                f"for series {series.id!r}",
            )
        past_only = set(past_covariates) - known_future

        known_future_columns.update(known_future)
        past_only_columns.update(past_only)

        try:
            future_index = pandas_module.date_range(
                start=timestamps[-1],
                periods=horizon + 1,
                freq=series.freq,
            )[1:]
        except ValueError as exc:
            raise AdapterInputError(
                f"invalid frequency {series.freq!r} for series {series.id!r}",
            ) from exc
        full_index = list(timestamps) + list(future_index)
        target_values = numpy_module.asarray(series.target, dtype=float).tolist() + [
            float("nan")
        ] * horizon
        frame_payload: dict[str, list[float]] = {"target": target_values}

        for name in sorted(past_covariates):
            history_values = _to_numeric_covariate_sequence(
                values=past_covariates[name],
                series_id=series.id,
                covariate=name,
            )
            if name in known_future:
                future_values = _to_numeric_covariate_sequence(
                    values=future_covariates[name],
                    series_id=series.id,
                    covariate=name,
                )
                frame_payload[name] = history_values + future_values
            else:
                frame_payload[name] = history_values + [float("nan")] * horizon

        dataframe = pandas_module.DataFrame(frame_payload, index=full_index)
        column_name = _unique_column_name(series.id, index, frames)
        frames[column_name] = dataframe

    feat_dynamic_real_columns = sorted(known_future_columns)
    past_feat_dynamic_real_columns = sorted(past_only_columns)
    dataset = pandas_dataset_cls(
        frames,
        target="target",
        feat_dynamic_real=feat_dynamic_real_columns or None,
        past_feat_dynamic_real=past_feat_dynamic_real_columns or None,
    )
    return _DatasetBuildResult(
        dataset=dataset,
        ordered_series=list(series_list),
        feat_dynamic_real_dim=len(feat_dynamic_real_columns),
        past_feat_dynamic_real_dim=len(past_feat_dynamic_real_columns),
    )


def _to_numeric_covariate_sequence(
    *,
    values: Sequence[Any],
    series_id: str,
    covariate: str,
) -> list[float]:
    converted: list[float] = []
    for value in values:
        if isinstance(value, bool):
            converted.append(float(int(value)))
            continue
        if isinstance(value, (int, float)):
            converted.append(float(value))
            continue
        if hasattr(value, "item"):
            scalar = value.item()
            if isinstance(scalar, bool):
                converted.append(float(int(scalar)))
                continue
            if isinstance(scalar, (int, float)):
                converted.append(float(scalar))
                continue
        raise AdapterInputError(
            f"Moirai covariates must be numeric; series={series_id!r} "
            f"covariate={covariate!r} value={value!r}",
        )
    return converted


def build_quantile_payload(
    *,
    forecast: Any,
    requested_quantiles: Sequence[float],
    horizon: int,
) -> dict[str, list[float]]:
    """Build quantile payload from a GluonTS-style forecast object."""
    payload: dict[str, list[float]] = {}
    for quantile in requested_quantiles:
        try:
            quantile_values = forecast.quantile(float(quantile))
        except Exception as exc:  # noqa: BLE001
            formatted = format(float(quantile), "g")
            raise AdapterInputError(
                f"requested quantile {formatted} is unavailable in Moirai forecast output",
            ) from exc
        values = normalize_forecast_vector(
            quantile_values,
            horizon=horizon,
            label=f"quantile {format(float(quantile), 'g')}",
        )
        payload[format(float(quantile), "g")] = values
    return payload


def normalize_forecast_vector(values: Any, *, horizon: int, label: str) -> list[float]:
    """Normalize a forecast vector to list[float] and enforce requested horizon length."""
    if horizon <= 0:
        raise AdapterInputError("horizon must be greater than zero")

    vector = _coerce_forecast_values(values)
    if len(vector) < horizon:
        raise AdapterInputError(
            f"{label} output is shorter than requested horizon ({len(vector)} < {horizon})",
        )
    return vector[:horizon]


def resolve_forecast_start_timestamp(
    *,
    forecast: Any,
    series: SeriesInput,
    pandas_module: Any,
    horizon: int,
) -> str:
    """Resolve forecast start timestamp from forecast metadata or fallback to freq stepping."""
    start_date = getattr(forecast, "start_date", None)
    if start_date is not None:
        direct = _coerce_timestamp_like(start_date)
        if direct is not None:
            return direct
        to_timestamp = getattr(start_date, "to_timestamp", None)
        if callable(to_timestamp):
            converted = to_timestamp()
            direct = _coerce_timestamp_like(converted)
            if direct is not None:
                return direct

    fallback = generate_future_timestamps(
        last_timestamp=series.timestamps[-1],
        freq=series.freq,
        horizon=horizon,
        pandas_module=pandas_module,
    )
    return fallback[0]


def generate_future_timestamps(
    *,
    last_timestamp: str,
    freq: str,
    horizon: int,
    pandas_module: Any,
) -> list[str]:
    """Generate future timestamps from the final observed timestamp and frequency."""
    if horizon <= 0:
        raise AdapterInputError("horizon must be greater than zero")

    parsed = pandas_module.to_datetime([last_timestamp], utc=True, errors="raise")
    if not parsed:
        raise AdapterInputError("series timestamp parsing returned no values")
    start = parsed[0]
    try:
        generated = pandas_module.date_range(start=start, periods=horizon + 1, freq=freq)
    except ValueError as exc:
        raise AdapterInputError(f"invalid frequency {freq!r} for moirai forecast") from exc
    return [_to_iso_timestamp(value) for value in list(generated[1:])]


def _resolve_runtime_config(
    *,
    model_name: str,
    model_source: dict[str, Any] | None,
    model_metadata: dict[str, Any] | None,
) -> _RuntimeConfig:
    defaults = _UNI2TS_MODELS.get(model_name, {})
    repo_id = _dict_str(model_source, "repo_id") or _string_or_none(defaults.get("repo_id"))
    revision = _dict_str(model_source, "revision") or _string_or_none(defaults.get("revision"))
    implementation = _dict_str(model_metadata, "implementation") or _string_or_none(
        defaults.get("implementation"),
    )
    default_context_length = _dict_positive_int(
        model_metadata, "default_context_length"
    ) or _int_or_none(defaults.get("default_context_length"))

    if implementation is None:
        implementation = "moirai_2p0"
    if default_context_length is None:
        default_context_length = _DEFAULT_CONTEXT_LENGTH

    if repo_id is None:
        raise UnsupportedModelError(f"unsupported uni2ts model {model_name!r}; missing repo_id")
    if revision is None:
        revision = "main"

    return _RuntimeConfig(
        model_name=model_name,
        repo_id=repo_id,
        revision=revision,
        implementation=implementation,
        default_context_length=default_context_length,
    )


def _coerce_forecast_values(values: Any) -> list[float]:
    if hasattr(values, "tolist"):
        values = values.tolist()
    if not isinstance(values, list):
        if isinstance(values, tuple):
            values = list(values)
        else:
            raise AdapterInputError("forecast output vector is not list-like")

    normalized: list[float] = []
    for value in values:
        if isinstance(value, bool):
            normalized.append(float(int(value)))
            continue
        if isinstance(value, (int, float)):
            normalized.append(float(value))
            continue
        if hasattr(value, "item"):
            scalar = value.item()
            if isinstance(scalar, (int, float)):
                normalized.append(float(scalar))
                continue
        raise AdapterInputError(f"forecast output contains non-numeric value: {value!r}")
    return normalized


def _coerce_timestamp_like(value: Any) -> str | None:
    if isinstance(value, datetime):
        return _to_iso_timestamp(value)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _to_iso_timestamp(value: Any) -> str:
    if isinstance(value, datetime):
        return value.astimezone(UTC).isoformat().replace("+00:00", "Z")
    return str(value)


def _unique_column_name(series_id: str, index: int, existing: dict[str, Any]) -> str:
    if series_id not in existing:
        return series_id
    return f"{series_id}__{index}"


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
    return _string_or_none(payload.get(key))


def _string_or_none(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    if not normalized:
        return None
    return normalized


def _dict_positive_int(payload: dict[str, Any] | None, key: str) -> int | None:
    if not isinstance(payload, dict):
        return None
    return _int_or_none(payload.get(key))


def _int_or_none(value: Any) -> int | None:
    if not isinstance(value, int):
        return None
    if value <= 0:
        return None
    return value
