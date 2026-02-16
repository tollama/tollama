"""Moirai forecasting adapter used by the uni2ts runner."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tollama.core.schemas import ForecastRequest, ForecastResponse, SeriesForecast, SeriesInput

from .errors import AdapterInputError, DependencyMissingError, UnsupportedModelError

_DEFAULT_CONTEXT_LENGTH = 200
_DEFAULT_NUM_SAMPLES = 200
_DEFAULT_PATCH_SIZE = "auto"
_DEFAULT_BATCH_SIZE = 32
_UNI2TS_MODELS: dict[str, dict[str, Any]] = {
    "moirai-1.1-R-base": {
        "repo_id": "Salesforce/moirai-1.1-R-base",
        "revision": "main",
        "implementation": "moirai_1p1",
        "default_num_samples": 200,
        "default_context_length": 200,
        "default_patch_size": "auto",
    },
    "moirai1p1-base": {
        "repo_id": "salesforce/moirai-1.1-base",
        "revision": "main",
        "implementation": "moirai_1p1",
        "default_num_samples": 200,
        "default_context_length": 200,
        "default_patch_size": "auto",
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
    default_num_samples: int
    default_context_length: int
    default_patch_size: str


@dataclass(frozen=True)
class _PredictorConfig:
    prediction_length: int
    context_length: int
    patch_size: str
    num_samples: int
    batch_size: int


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

        context_length = resolve_context_length(
            option_value=request.options.get("context_length"),
            default_context_length=runtime.default_context_length,
            series_list=request.series,
        )
        patch_size = resolve_patch_size(
            option_value=request.options.get("patch_size"),
            default_patch_size=runtime.default_patch_size,
        )
        num_samples = resolve_positive_int(
            option_value=request.options.get("num_samples"),
            default_value=runtime.default_num_samples,
            field_name="num_samples",
        )
        batch_size = resolve_positive_int(
            option_value=request.options.get("batch_size"),
            default_value=_DEFAULT_BATCH_SIZE,
            field_name="batch_size",
        )
        predictor_config = _PredictorConfig(
            prediction_length=request.horizon,
            context_length=context_length,
            patch_size=patch_size,
            num_samples=num_samples,
            batch_size=batch_size,
        )

        dataset, ordered_series = build_pandas_dataset(
            series_list=request.series,
            pandas_module=dependencies.pandas,
            numpy_module=dependencies.numpy,
            pandas_dataset_cls=dependencies.pandas_dataset_cls,
        )
        moirai_model = dependencies.moirai_forecast_cls(
            module=module,
            prediction_length=predictor_config.prediction_length,
            context_length=predictor_config.context_length,
            patch_size=predictor_config.patch_size,
            num_samples=predictor_config.num_samples,
            target_dim=1,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0,
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
                "num_samples": num_samples,
                "context_length": context_length,
                "patch_size": patch_size,
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
            from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
        except ModuleNotFoundError as exc:
            missing_packages.append(exc.name or "uni2ts")
            MoiraiForecast = None
            MoiraiModule = None

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


def resolve_patch_size(*, option_value: Any, default_patch_size: str) -> str:
    """Resolve patch size option with a default fallback."""
    if option_value is None:
        return default_patch_size or _DEFAULT_PATCH_SIZE
    if not isinstance(option_value, str):
        raise AdapterInputError("patch_size option must be a string")
    normalized = option_value.strip()
    if not normalized:
        raise AdapterInputError("patch_size option cannot be empty")
    return normalized


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
) -> tuple[Any, list[SeriesInput]]:
    """Build a GluonTS PandasDataset from canonical series payloads."""
    columns: dict[str, Any] = {}
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
        values = numpy_module.asarray(series.target, dtype=float)
        column_name = _unique_column_name(series.id, index, columns)
        columns[column_name] = pandas_module.Series(values, index=timestamps, name=column_name)

    dataframe = pandas_module.DataFrame(columns).sort_index()
    dataset = pandas_dataset_cls(dict(dataframe))
    return dataset, list(series_list)


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
    default_num_samples = _dict_positive_int(model_metadata, "default_num_samples") or _int_or_none(
        defaults.get("default_num_samples"),
    )
    default_context = _dict_positive_int(model_metadata, "default_context_length") or _int_or_none(
        defaults.get("default_context_length"),
    )
    default_patch_size = _dict_str(model_metadata, "default_patch_size") or _string_or_none(
        defaults.get("default_patch_size"),
    )

    if repo_id is None:
        raise UnsupportedModelError(f"unsupported uni2ts model {model_name!r}; missing repo_id")
    if revision is None:
        revision = "main"
    if implementation is None:
        implementation = "moirai_1p1"
    if default_num_samples is None:
        default_num_samples = _DEFAULT_NUM_SAMPLES
    if default_context is None:
        default_context = _DEFAULT_CONTEXT_LENGTH
    if default_patch_size is None:
        default_patch_size = _DEFAULT_PATCH_SIZE
    return _RuntimeConfig(
        model_name=model_name,
        repo_id=repo_id,
        revision=revision,
        implementation=implementation,
        default_num_samples=default_num_samples,
        default_context_length=default_context,
        default_patch_size=default_patch_size,
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
