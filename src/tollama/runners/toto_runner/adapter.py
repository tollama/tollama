"""Toto forecasting adapter used by the toto runner."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from tollama.core.schemas import ForecastRequest, ForecastResponse, SeriesForecast, SeriesInput

from .errors import AdapterInputError, DependencyMissingError, UnsupportedModelError

_DEFAULT_MAX_CONTEXT = 4096
_DEFAULT_MAX_HORIZON = 720
_DEFAULT_NUM_SAMPLES = 256
_DEFAULT_SAMPLES_PER_BATCH = 256
_DEFAULT_USE_KV_CACHE = True
_DEFAULT_MATMUL_PRECISION = "high"
_DEFAULT_DEVICE = "auto"
_DEFAULT_DTYPE = "float32"

_TOTO_MODELS: dict[str, dict[str, Any]] = {
    "toto-open-base-1.0": {
        "repo_id": "Datadog/Toto-Open-Base-1.0",
        "revision": "main",
        "implementation": "toto_open_base",
        "max_context": _DEFAULT_MAX_CONTEXT,
        "max_horizon": _DEFAULT_MAX_HORIZON,
        "default_num_samples": _DEFAULT_NUM_SAMPLES,
        "default_samples_per_batch": _DEFAULT_SAMPLES_PER_BATCH,
        "default_use_kv_cache": _DEFAULT_USE_KV_CACHE,
        "device": _DEFAULT_DEVICE,
        "matmul_precision": _DEFAULT_MATMUL_PRECISION,
        "torch_compile": False,
        "dtype": _DEFAULT_DTYPE,
    },
}


@dataclass(frozen=True)
class _TotoDependencies:
    torch: Any
    numpy: Any
    pandas: Any
    toto_cls: Any
    toto_forecaster_cls: Any
    masked_timeseries_cls: Any


@dataclass(frozen=True)
class _RuntimeConfig:
    model_name: str
    repo_id: str
    revision: str
    implementation: str
    max_context: int
    max_horizon: int
    default_num_samples: int
    default_samples_per_batch: int
    default_use_kv_cache: bool
    device: str
    matmul_precision: str
    torch_compile: bool
    dtype: str


@dataclass(frozen=True)
class _PreparedTotoInput:
    input_payload: Any
    variate_count: int
    interval_seconds: int


@dataclass(frozen=True)
class _LoadedTotoModel:
    model: Any
    forecaster: Any
    runtime: _RuntimeConfig
    model_ref: str
    device: str
    dtype: str
    torch_compile: bool


class TotoAdapter:
    """Adapter that maps canonical request/response to Toto probabilistic inference."""

    def __init__(self) -> None:
        self._dependencies: _TotoDependencies | None = None
        self._models: dict[str, _LoadedTotoModel] = {}

    def load(
        self,
        model_name: str,
        *,
        model_local_dir: str | None = None,
        model_source: dict[str, Any] | None = None,
        model_metadata: dict[str, Any] | None = None,
    ) -> None:
        """Load one Toto model into cache."""
        runtime = _resolve_runtime_config(
            model_name=model_name,
            model_source=model_source,
            model_metadata=model_metadata,
        )
        dependencies = self._resolve_dependencies()
        device = resolve_device(
            option_value=None,
            default_device=runtime.device,
            torch_module=dependencies.torch,
        )
        _ = self._get_or_load_model(
            runtime=runtime,
            model_local_dir=model_local_dir,
            device=device,
            dtype=runtime.dtype,
            torch_compile=runtime.torch_compile,
            matmul_precision=runtime.matmul_precision,
        )

    def unload(self, model_name: str | None = None) -> None:
        """Unload one cached model or clear all cached models."""
        if model_name is None:
            self._models.clear()
        else:
            self._models.pop(model_name, None)

        dependencies = self._dependencies
        if dependencies is None:
            return
        cuda_module = getattr(dependencies.torch, "cuda", None)
        if cuda_module is None:
            return
        is_available = getattr(cuda_module, "is_available", None)
        empty_cache = getattr(cuda_module, "empty_cache", None)
        if callable(is_available) and callable(empty_cache) and is_available():
            empty_cache()

    def forecast(
        self,
        request: ForecastRequest,
        *,
        model_local_dir: str | None = None,
        model_source: dict[str, Any] | None = None,
        model_metadata: dict[str, Any] | None = None,
    ) -> ForecastResponse:
        """Generate probabilistic Toto forecasts mapped to canonical response schema."""
        runtime = _resolve_runtime_config(
            model_name=request.model,
            model_source=model_source,
            model_metadata=model_metadata,
        )
        if request.horizon > runtime.max_horizon:
            raise AdapterInputError(
                f"requested horizon {request.horizon} exceeds model max_horizon "
                f"({runtime.max_horizon})",
            )

        dependencies = self._resolve_dependencies()
        options = request.options
        device = resolve_device(
            option_value=options.get("device"),
            default_device=runtime.device,
            torch_module=dependencies.torch,
        )
        dtype = resolve_dtype(
            option_value=options.get("dtype"),
            default_dtype=runtime.dtype,
        )
        matmul_precision = resolve_matmul_precision(
            option_value=options.get("matmul_precision"),
            default_precision=runtime.matmul_precision,
        )
        torch_compile = resolve_bool_option(
            option_value=options.get("torch_compile"),
            default_value=runtime.torch_compile,
            field_name="torch_compile",
        )
        use_kv_cache = resolve_bool_option(
            option_value=options.get("use_kv_cache"),
            default_value=runtime.default_use_kv_cache,
            field_name="use_kv_cache",
        )
        num_samples = resolve_positive_int_option(
            option_value=options.get("num_samples"),
            default_value=runtime.default_num_samples,
            field_name="num_samples",
        )
        desired_samples_per_batch = resolve_positive_int_option(
            option_value=options.get("samples_per_batch"),
            default_value=runtime.default_samples_per_batch,
            field_name="samples_per_batch",
        )
        samples_per_batch = choose_samples_per_batch(
            num_samples=num_samples,
            desired=desired_samples_per_batch,
        )

        loaded = self._get_or_load_model(
            runtime=runtime,
            model_local_dir=model_local_dir,
            device=device,
            dtype=dtype,
            torch_compile=torch_compile,
            matmul_precision=matmul_precision,
        )

        warnings = build_unsupported_covariate_warnings(request.series)
        forecasts: list[SeriesForecast] = []
        for series in request.series:
            prepared = build_masked_timeseries_input(
                series=series,
                max_context=runtime.max_context,
                device=device,
                dtype=dtype,
                torch_module=dependencies.torch,
                pandas_module=dependencies.pandas,
                masked_timeseries_cls=dependencies.masked_timeseries_cls,
            )

            forecast_output = _run_toto_forecast(
                forecaster=loaded.forecaster,
                inputs=prepared.input_payload,
                horizon=request.horizon,
                num_samples=num_samples,
                samples_per_batch=samples_per_batch,
                use_kv_cache=use_kv_cache,
            )
            point_forecast = extract_point_forecast(
                forecast=forecast_output,
                horizon=request.horizon,
                torch_module=dependencies.torch,
            )
            quantiles = extract_quantiles(
                forecast=forecast_output,
                requested_quantiles=list(request.quantiles),
                horizon=request.horizon,
                torch_module=dependencies.torch,
            )
            future_timestamps = generate_future_timestamps_from_interval(
                last_timestamp=series.timestamps[-1],
                interval_seconds=prepared.interval_seconds,
                horizon=request.horizon,
                pandas_module=dependencies.pandas,
            )
            forecasts.append(
                SeriesForecast(
                    id=series.id,
                    freq=series.freq,
                    start_timestamp=future_timestamps[0],
                    mean=point_forecast,
                    quantiles=quantiles,
                ),
            )

        return ForecastResponse(
            model=request.model,
            forecasts=forecasts,
            usage={
                "runner": "tollama-toto",
                "implementation": runtime.implementation,
                "series_count": len(request.series),
                "horizon": request.horizon,
                "num_samples": num_samples,
                "samples_per_batch": samples_per_batch,
                "use_kv_cache": use_kv_cache,
            },
            warnings=warnings or None,
        )

    def _get_or_load_model(
        self,
        *,
        runtime: _RuntimeConfig,
        model_local_dir: str | None,
        device: str,
        dtype: str,
        torch_compile: bool,
        matmul_precision: str,
    ) -> _LoadedTotoModel:
        dependencies = self._resolve_dependencies()
        _configure_matmul_precision(dependencies.torch, matmul_precision=matmul_precision)

        local_model_path = _existing_local_model_path(model_local_dir)
        model_ref = local_model_path or runtime.repo_id

        cached = self._models.get(runtime.model_name)
        if (
            cached is not None
            and cached.model_ref == model_ref
            and cached.runtime.revision == runtime.revision
            and cached.device == device
            and cached.dtype == dtype
            and cached.torch_compile == torch_compile
        ):
            return cached

        load_kwargs = {"map_location": "cpu"}
        if local_model_path is not None:
            model = _call_from_pretrained(
                model_cls=dependencies.toto_cls,
                model_ref=local_model_path,
                revision=None,
                kwargs=load_kwargs,
            )
        else:
            model = _call_from_pretrained(
                model_cls=dependencies.toto_cls,
                model_ref=runtime.repo_id,
                revision=runtime.revision,
                kwargs=load_kwargs,
            )

        model_dtype = _torch_dtype_from_name(dependencies.torch, dtype)
        _move_model_to_device(
            model=model,
            device=device,
            dtype=model_dtype,
        )
        eval_method = getattr(model, "eval", None)
        if callable(eval_method):
            eval_method()

        forecaster_model = getattr(model, "model", model)
        if torch_compile:
            forecaster_model = _compile_model(
                torch_module=dependencies.torch,
                model=forecaster_model,
            )
            if hasattr(model, "model"):
                try:
                    setattr(model, "model", forecaster_model)
                except Exception:  # noqa: BLE001
                    pass

        forecaster = dependencies.toto_forecaster_cls(forecaster_model)
        loaded = _LoadedTotoModel(
            model=model,
            forecaster=forecaster,
            runtime=runtime,
            model_ref=model_ref,
            device=device,
            dtype=dtype,
            torch_compile=torch_compile,
        )
        self._models[runtime.model_name] = loaded
        return loaded

    def _resolve_dependencies(self) -> _TotoDependencies:
        cached = self._dependencies
        if cached is not None:
            return cached

        missing_packages: list[str] = []
        try:
            import torch
        except ModuleNotFoundError as exc:
            missing_packages.append(exc.name or "torch")
            torch = None

        try:
            import numpy as np
        except ModuleNotFoundError as exc:
            missing_packages.append(exc.name or "numpy")
            np = None

        try:
            import pandas as pd
        except ModuleNotFoundError as exc:
            missing_packages.append(exc.name or "pandas")
            pd = None

        try:
            from toto.data.util.dataset import MaskedTimeseries
            from toto.inference.forecaster import TotoForecaster
            from toto.model.toto import Toto
        except ModuleNotFoundError as exc:
            missing_packages.append(exc.name or "toto-ts")
            Toto = None
            TotoForecaster = None
            MaskedTimeseries = None

        if missing_packages:
            joined = ", ".join(sorted(set(missing_packages)))
            raise DependencyMissingError(
                "missing optional toto runner dependencies "
                f"({joined}); install them with `pip install -e \".[dev,runner_toto]\"`",
            )

        assert torch is not None
        assert np is not None
        assert pd is not None
        assert Toto is not None
        assert TotoForecaster is not None
        assert MaskedTimeseries is not None
        resolved = _TotoDependencies(
            torch=torch,
            numpy=np,
            pandas=pd,
            toto_cls=Toto,
            toto_forecaster_cls=TotoForecaster,
            masked_timeseries_cls=MaskedTimeseries,
        )
        self._dependencies = resolved
        return resolved


def build_masked_timeseries_input(
    *,
    series: SeriesInput,
    max_context: int,
    device: str,
    dtype: str,
    torch_module: Any,
    pandas_module: Any,
    masked_timeseries_cls: Any,
) -> _PreparedTotoInput:
    """Build a Toto MaskedTimeseries payload for one request series."""
    variates, timestamp_seconds, interval_seconds = build_toto_variates(
        series=series,
        max_context=max_context,
        pandas_module=pandas_module,
    )
    cleaned_variates, padding_mask = build_clean_values_and_mask(variates)
    id_mask = [[0 for _ in row] for row in cleaned_variates]
    timestamp_matrix = [list(timestamp_seconds) for _ in cleaned_variates]
    time_interval = [interval_seconds for _ in cleaned_variates]

    torch_dtype = _torch_dtype_from_name(torch_module, dtype)
    series_tensor = _torch_tensor(
        torch_module=torch_module,
        values=[cleaned_variates],
        dtype=torch_dtype,
        device=device,
    )
    padding_tensor = _torch_tensor(
        torch_module=torch_module,
        values=[padding_mask],
        dtype=getattr(torch_module, "bool", None),
        device=device,
    )
    id_tensor = _torch_tensor(
        torch_module=torch_module,
        values=[id_mask],
        dtype=getattr(torch_module, "int64", None),
        device=device,
    )
    timestamp_tensor = _torch_tensor(
        torch_module=torch_module,
        values=[timestamp_matrix],
        dtype=getattr(torch_module, "int64", None),
        device=device,
    )
    interval_tensor = _torch_tensor(
        torch_module=torch_module,
        values=[time_interval],
        dtype=getattr(torch_module, "int64", None),
        device=device,
    )

    payload = masked_timeseries_cls(
        series=series_tensor,
        padding_mask=padding_tensor,
        id_mask=id_tensor,
        timestamp_seconds=timestamp_tensor,
        time_interval_seconds=interval_tensor,
    )
    return _PreparedTotoInput(
        input_payload=payload,
        variate_count=len(cleaned_variates),
        interval_seconds=interval_seconds,
    )


def build_toto_variates(
    *,
    series: SeriesInput,
    max_context: int,
    pandas_module: Any,
) -> tuple[list[list[float]], list[int], int]:
    """Construct target+past-covariate variates and aligned timestamps."""
    target = to_numeric_sequence(
        values=series.target,
        field_name="target",
        series_id=series.id,
    )
    if len(target) < 2:
        raise AdapterInputError("each input series must include at least two target points")

    covariates = series.past_covariates or {}
    variates = [target]
    for name in sorted(covariates):
        values = covariates[name]
        if len(values) != len(target):
            raise AdapterInputError(
                f"series {series.id!r} past_covariates[{name!r}] length must match target length",
            )
        variates.append(
            to_numeric_sequence(
                values=values,
                field_name=f"past_covariates[{name!r}]",
                series_id=series.id,
            ),
        )

    timestamp_seconds = timestamps_to_unix_seconds(
        timestamps=series.timestamps,
        pandas_module=pandas_module,
    )
    truncated_variates, truncated_timestamps = truncate_multivariate_to_max_context(
        matrix=variates,
        timestamps=timestamp_seconds,
        max_context=max_context,
    )
    interval_seconds = infer_time_interval_seconds(truncated_timestamps)
    return truncated_variates, truncated_timestamps, interval_seconds


def truncate_multivariate_to_max_context(
    *,
    matrix: Sequence[Sequence[float]],
    timestamps: Sequence[int],
    max_context: int,
) -> tuple[list[list[float]], list[int]]:
    """Truncate multivariate history to the most recent max_context points."""
    if max_context <= 1:
        raise AdapterInputError("max_context must be greater than one")
    if not matrix:
        raise AdapterInputError("input matrix must include at least one variate")

    expected = len(timestamps)
    if expected < 2:
        raise AdapterInputError("timestamps must include at least two points")
    for row in matrix:
        if len(row) != expected:
            raise AdapterInputError("all variates must align to the same timestamp length")

    start = max(0, expected - max_context)
    truncated_rows = [list(row[start:]) for row in matrix]
    truncated_timestamps = list(timestamps[start:])
    if len(truncated_timestamps) < 2:
        raise AdapterInputError("truncated context must include at least two time steps")
    return truncated_rows, truncated_timestamps


def choose_samples_per_batch(*, num_samples: int, desired: int | None) -> int:
    """Choose the largest divisor of num_samples that does not exceed desired."""
    if num_samples <= 0:
        raise AdapterInputError("num_samples must be greater than zero")
    if desired is None:
        return num_samples
    if desired <= 0:
        raise AdapterInputError("samples_per_batch must be greater than zero")

    candidate = min(desired, num_samples)
    while candidate > 1:
        if num_samples % candidate == 0:
            return candidate
        candidate -= 1
    return 1


def timestamps_to_unix_seconds(*, timestamps: Sequence[str], pandas_module: Any) -> list[int]:
    """Convert timestamp strings to UTC POSIX seconds."""
    parsed = pandas_module.to_datetime(timestamps, utc=True, errors="raise")
    values = list(parsed)
    if len(values) < 2:
        raise AdapterInputError("series must include at least two timestamps")

    converted = [_timestamp_to_unix_seconds(value) for value in values]
    return converted


def infer_time_interval_seconds(timestamp_seconds: Sequence[int]) -> int:
    """Infer one representative time interval in seconds from consecutive timestamps."""
    if len(timestamp_seconds) < 2:
        raise AdapterInputError("timestamps must include at least two points")

    deltas = [
        int(right - left)
        for left, right in zip(timestamp_seconds[:-1], timestamp_seconds[1:], strict=False)
        if int(right - left) > 0
    ]
    if not deltas:
        raise AdapterInputError("timestamps must be strictly increasing")

    sorted_deltas = sorted(deltas)
    mid = len(sorted_deltas) // 2
    if len(sorted_deltas) % 2 == 1:
        interval = sorted_deltas[mid]
    else:
        interval = int(round((sorted_deltas[mid - 1] + sorted_deltas[mid]) / 2.0))
    if interval <= 0:
        raise AdapterInputError("derived time interval must be positive")
    return interval


def generate_future_timestamps_from_interval(
    *,
    last_timestamp: str,
    interval_seconds: int,
    horizon: int,
    pandas_module: Any,
) -> list[str]:
    """Generate future timestamps from the last observed timestamp and inferred interval."""
    if horizon <= 0:
        raise AdapterInputError("horizon must be greater than zero")
    if interval_seconds <= 0:
        raise AdapterInputError("time interval seconds must be greater than zero")

    parsed = pandas_module.to_datetime([last_timestamp], utc=True, errors="raise")
    values = list(parsed)
    if not values:
        raise AdapterInputError("series timestamp parsing returned no values")
    start = _to_datetime(values[0])

    generated: list[str] = []
    for step in range(1, horizon + 1):
        generated.append(
            _to_iso_timestamp(start + timedelta(seconds=interval_seconds * step)),
        )
    return generated


def build_clean_values_and_mask(
    variates: Sequence[Sequence[float]],
) -> tuple[list[list[float]], list[list[bool]]]:
    """Replace non-finite values with zero and build Toto-style valid-data padding mask."""
    cleaned: list[list[float]] = []
    mask: list[list[bool]] = []
    for row in variates:
        clean_row: list[float] = []
        mask_row: list[bool] = []
        for value in row:
            numeric = float(value)
            finite = math.isfinite(numeric)
            clean_row.append(numeric if finite else 0.0)
            mask_row.append(bool(finite))
        cleaned.append(clean_row)
        mask.append(mask_row)
    return cleaned, mask


def extract_point_forecast(*, forecast: Any, horizon: int, torch_module: Any) -> list[float]:
    """Extract one point forecast vector, preferring median over samples."""
    median_attr = getattr(forecast, "median", None)
    if median_attr is not None:
        return _extract_target_vector(median_attr, horizon=horizon, label="median")

    quantile_method = getattr(forecast, "quantile", None)
    if callable(quantile_method):
        try:
            value = quantile_method(0.5)
        except Exception:  # noqa: BLE001
            value = None
        if value is not None:
            return _extract_target_vector(value, horizon=horizon, label="median")

    samples_attr = getattr(forecast, "samples", None)
    if samples_attr is not None:
        target_samples = _extract_target_samples(
            samples=samples_attr,
            horizon=horizon,
        )
        return quantiles_from_target_samples(
            target_samples=target_samples,
            requested_quantiles=[0.5],
            torch_module=torch_module,
            horizon=horizon,
        )["0.5"]

    mean_attr = getattr(forecast, "mean", None)
    if mean_attr is not None:
        return _extract_target_vector(mean_attr, horizon=horizon, label="mean")
    raise AdapterInputError("Toto forecast output is missing both median and mean")


def extract_quantiles(
    *,
    forecast: Any,
    requested_quantiles: Sequence[float],
    horizon: int,
    torch_module: Any,
) -> dict[str, list[float]] | None:
    """Extract requested quantiles from Toto output."""
    if not requested_quantiles:
        return None

    quantile_method = getattr(forecast, "quantile", None)
    if callable(quantile_method):
        payload: dict[str, list[float]] = {}
        for value in requested_quantiles:
            try:
                quantile_output = quantile_method(float(value))
            except Exception as exc:  # noqa: BLE001
                raise AdapterInputError(
                    f"requested quantile {format(float(value), 'g')} is unavailable",
                ) from exc
            payload[format(float(value), "g")] = _extract_target_vector(
                quantile_output,
                horizon=horizon,
                label=f"quantile {format(float(value), 'g')}",
            )
        return payload

    samples_attr = getattr(forecast, "samples", None)
    if samples_attr is None:
        raise AdapterInputError("quantiles requested but Toto output does not include samples")
    target_samples = _extract_target_samples(samples=samples_attr, horizon=horizon)
    return quantiles_from_target_samples(
        target_samples=target_samples,
        requested_quantiles=[float(item) for item in requested_quantiles],
        torch_module=torch_module,
        horizon=horizon,
    )


def quantiles_from_target_samples(
    *,
    target_samples: Sequence[Sequence[float]],
    requested_quantiles: Sequence[float],
    torch_module: Any,
    horizon: int,
) -> dict[str, list[float]]:
    """Compute horizon-wise quantiles from [horizon, samples] target sample matrix."""
    if not requested_quantiles:
        return {}
    if not target_samples:
        raise AdapterInputError("target samples must not be empty")
    if horizon <= 0:
        raise AdapterInputError("horizon must be greater than zero")

    normalized_rows: list[list[float]] = []
    sample_count = 0
    for row in target_samples:
        values = [_to_float(value) for value in list(row)]
        if sample_count == 0:
            sample_count = len(values)
        if len(values) != sample_count:
            raise AdapterInputError("target sample rows must all have the same sample count")
        normalized_rows.append(values)
    if len(normalized_rows) < horizon:
        raise AdapterInputError("target sample output is shorter than requested horizon")

    sample_tensor = _torch_tensor(
        torch_module=torch_module,
        values=normalized_rows[:horizon],
        dtype=getattr(torch_module, "float32", None),
        device=None,
    )
    quantile_fn = getattr(torch_module, "quantile", None)
    if not callable(quantile_fn):
        raise AdapterInputError("torch dependency does not expose quantile")

    payload: dict[str, list[float]] = {}
    for quantile in requested_quantiles:
        quantile_value = float(quantile)
        try:
            quantile_tensor = quantile_fn(sample_tensor, quantile_value, dim=1)
        except TypeError:
            quantile_tensor = quantile_fn(
                sample_tensor,
                _torch_tensor(
                    torch_module=torch_module,
                    values=[quantile_value],
                    dtype=getattr(torch_module, "float32", None),
                    device=None,
                ),
                dim=1,
            )
        values = _to_nested_list(quantile_tensor)
        if isinstance(values, list) and values and isinstance(values[0], list):
            values = values[0]
        payload[format(quantile_value, "g")] = _to_float_vector(
            values,
            horizon=horizon,
            label=f"quantile {format(quantile_value, 'g')}",
        )
    return payload


def to_numeric_sequence(*, values: Sequence[Any], field_name: str, series_id: str) -> list[float]:
    """Convert one sequence to float while validating scalar numeric entries."""
    converted: list[float] = []
    for value in values:
        try:
            converted.append(_to_float(value))
        except AdapterInputError as exc:
            raise AdapterInputError(
                f"series {series_id!r} {field_name} contains non-numeric value {value!r}",
            ) from exc
    return converted


def build_unsupported_covariate_warnings(series_list: Sequence[SeriesInput]) -> list[str]:
    """Warn when unsupported covariates are supplied directly to the runner."""
    saw_future = False
    saw_static = False
    for series in series_list:
        if series.future_covariates:
            saw_future = True
        if series.static_covariates:
            saw_static = True

    warnings: list[str] = []
    if saw_future:
        warnings.append(
            "Toto runner ignores future covariates; using target + past numeric covariates",
        )
    if saw_static:
        warnings.append("Toto runner ignores static covariates")
    return warnings


def resolve_device(*, option_value: Any, default_device: str, torch_module: Any) -> str:
    """Resolve runtime device from options/default metadata."""
    chosen = option_value if option_value is not None else default_device
    if not isinstance(chosen, str) or not chosen.strip():
        raise AdapterInputError("device option must be a non-empty string")
    normalized = chosen.strip().lower()
    if normalized == "auto":
        cuda_module = getattr(torch_module, "cuda", None)
        is_available = (
            getattr(cuda_module, "is_available", None) if cuda_module is not None else None
        )
        if callable(is_available) and is_available():
            return "cuda"
        return "cpu"
    return normalized


def resolve_dtype(*, option_value: Any, default_dtype: str) -> str:
    """Resolve dtype option."""
    chosen = option_value if option_value is not None else default_dtype
    if not isinstance(chosen, str) or not chosen.strip():
        raise AdapterInputError("dtype option must be a non-empty string")
    normalized = chosen.strip().lower()
    allowed = {"float32", "float16", "bfloat16"}
    if normalized not in allowed:
        joined = ", ".join(sorted(allowed))
        raise AdapterInputError(f"dtype must be one of {joined}")
    return normalized


def resolve_matmul_precision(*, option_value: Any, default_precision: str) -> str:
    """Resolve torch float32 matmul precision value."""
    chosen = option_value if option_value is not None else default_precision
    if not isinstance(chosen, str) or not chosen.strip():
        raise AdapterInputError("matmul_precision must be a non-empty string")
    normalized = chosen.strip().lower()
    allowed = {"highest", "high", "medium"}
    if normalized not in allowed:
        joined = ", ".join(sorted(allowed))
        raise AdapterInputError(f"matmul_precision must be one of {joined}")
    return normalized


def resolve_bool_option(*, option_value: Any, default_value: bool, field_name: str) -> bool:
    """Resolve strict boolean option."""
    if option_value is None:
        return bool(default_value)
    if not isinstance(option_value, bool):
        raise AdapterInputError(f"{field_name} option must be a boolean")
    return option_value


def resolve_positive_int_option(*, option_value: Any, default_value: int, field_name: str) -> int:
    """Resolve strict positive integer option."""
    if option_value is None:
        if default_value <= 0:
            raise AdapterInputError(f"{field_name} must be greater than zero")
        return int(default_value)
    if isinstance(option_value, bool) or not isinstance(option_value, int):
        raise AdapterInputError(f"{field_name} option must be an integer")
    if option_value <= 0:
        raise AdapterInputError(f"{field_name} option must be greater than zero")
    return option_value


def _resolve_runtime_config(
    *,
    model_name: str,
    model_source: dict[str, Any] | None,
    model_metadata: dict[str, Any] | None,
) -> _RuntimeConfig:
    defaults = _TOTO_MODELS.get(model_name, {})
    repo_id = _dict_str(model_source, "repo_id") or _string_or_none(defaults.get("repo_id"))
    revision = _dict_str(model_source, "revision") or _string_or_none(defaults.get("revision"))
    implementation = _dict_str(model_metadata, "implementation") or _string_or_none(
        defaults.get("implementation"),
    )
    max_context = _dict_positive_int(model_metadata, "max_context") or _int_or_none(
        defaults.get("max_context"),
    )
    max_horizon = _dict_positive_int(model_metadata, "max_horizon") or _int_or_none(
        defaults.get("max_horizon"),
    )
    default_num_samples = _dict_positive_int(
        model_metadata,
        "default_num_samples",
    ) or _int_or_none(defaults.get("default_num_samples"))
    default_samples_per_batch = _dict_positive_int(
        model_metadata,
        "default_samples_per_batch",
    ) or _int_or_none(defaults.get("default_samples_per_batch"))
    default_use_kv_cache = _dict_bool(
        model_metadata,
        "default_use_kv_cache",
    )
    if default_use_kv_cache is None:
        default_use_kv_cache = _bool_or_none(defaults.get("default_use_kv_cache"))
    device = _dict_str(model_metadata, "device") or _string_or_none(defaults.get("device"))
    matmul_precision = _dict_str(model_metadata, "matmul_precision") or _string_or_none(
        defaults.get("matmul_precision"),
    )
    torch_compile = _dict_bool(model_metadata, "torch_compile")
    if torch_compile is None:
        torch_compile = _bool_or_none(defaults.get("torch_compile"))
    dtype = _dict_str(model_metadata, "dtype") or _string_or_none(defaults.get("dtype"))

    if repo_id is None:
        raise UnsupportedModelError(f"unsupported toto model {model_name!r}; missing repo_id")
    if revision is None:
        revision = "main"
    if implementation is None:
        implementation = "toto_open_base"
    if max_context is None:
        max_context = _DEFAULT_MAX_CONTEXT
    if max_horizon is None:
        max_horizon = _DEFAULT_MAX_HORIZON
    if default_num_samples is None:
        default_num_samples = _DEFAULT_NUM_SAMPLES
    if default_samples_per_batch is None:
        default_samples_per_batch = _DEFAULT_SAMPLES_PER_BATCH
    if default_use_kv_cache is None:
        default_use_kv_cache = _DEFAULT_USE_KV_CACHE
    if device is None:
        device = _DEFAULT_DEVICE
    if matmul_precision is None:
        matmul_precision = _DEFAULT_MATMUL_PRECISION
    if torch_compile is None:
        torch_compile = False
    if dtype is None:
        dtype = _DEFAULT_DTYPE

    return _RuntimeConfig(
        model_name=model_name,
        repo_id=repo_id,
        revision=revision,
        implementation=implementation,
        max_context=max_context,
        max_horizon=max_horizon,
        default_num_samples=default_num_samples,
        default_samples_per_batch=default_samples_per_batch,
        default_use_kv_cache=default_use_kv_cache,
        device=device,
        matmul_precision=matmul_precision,
        torch_compile=torch_compile,
        dtype=dtype,
    )


def _run_toto_forecast(
    *,
    forecaster: Any,
    inputs: Any,
    horizon: int,
    num_samples: int,
    samples_per_batch: int,
    use_kv_cache: bool,
) -> Any:
    kwargs: dict[str, Any] = {
        "prediction_length": horizon,
        "num_samples": num_samples,
        "samples_per_batch": samples_per_batch,
        "use_kv_cache": use_kv_cache,
    }
    try:
        return forecaster.forecast(inputs, **kwargs)
    except TypeError:
        fallback_kwargs = dict(kwargs)
        fallback_kwargs.pop("samples_per_batch", None)
        try:
            return forecaster.forecast(inputs, **fallback_kwargs)
        except TypeError as exc:
            raise AdapterInputError(
                "Toto forecaster.forecast signature is incompatible with tollama runner",
            ) from exc


def _extract_target_vector(value: Any, *, horizon: int, label: str) -> list[float]:
    payload = _to_nested_list(value)
    if not isinstance(payload, list):
        raise AdapterInputError(f"Toto {label} output is not list-like")
    if payload and isinstance(payload[0], list):
        if payload[0] and isinstance(payload[0][0], list):
            payload = payload[0][0]
        else:
            payload = payload[0]
    return _to_float_vector(payload, horizon=horizon, label=label)


def _extract_target_samples(*, samples: Any, horizon: int) -> list[list[float]]:
    payload = _to_nested_list(samples)
    if not isinstance(payload, list):
        raise AdapterInputError("Toto sample output is not list-like")
    if payload and isinstance(payload[0], list):
        if payload[0] and isinstance(payload[0][0], list):
            if payload[0][0] and isinstance(payload[0][0][0], list):
                payload = payload[0][0]
            else:
                payload = payload[0]
    if not isinstance(payload, list):
        raise AdapterInputError("Toto sample output has unexpected shape")

    rows: list[list[float]] = []
    for row in payload:
        if not isinstance(row, list):
            raise AdapterInputError("Toto sample output row is not list-like")
        rows.append([_to_float(value) for value in row])
    if len(rows) < horizon:
        raise AdapterInputError("Toto sample output is shorter than requested horizon")
    return rows[:horizon]


def _to_float_vector(value: Any, *, horizon: int, label: str) -> list[float]:
    if not isinstance(value, list):
        raise AdapterInputError(f"Toto {label} output is not list-like")
    if len(value) < horizon:
        raise AdapterInputError(f"Toto {label} output is shorter than requested horizon")
    return [_to_float(item) for item in value[:horizon]]


def _timestamp_to_unix_seconds(value: Any) -> int:
    as_datetime = _to_datetime(value)
    return int(as_datetime.timestamp())


def _to_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value.astimezone(UTC)
    to_pydatetime = getattr(value, "to_pydatetime", None)
    if callable(to_pydatetime):
        converted = to_pydatetime()
        if isinstance(converted, datetime):
            return converted.astimezone(UTC)
    if isinstance(value, str):
        normalized = value.strip()
        if normalized.endswith("Z"):
            normalized = normalized.replace("Z", "+00:00")
        try:
            parsed = datetime.fromisoformat(normalized)
        except ValueError as exc:
            raise AdapterInputError(f"invalid timestamp value: {value!r}") from exc
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        return parsed.astimezone(UTC)
    raise AdapterInputError(f"unsupported timestamp value: {value!r}")


def _to_float(value: Any) -> float:
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
    raise AdapterInputError(f"value is not numeric: {value!r}")


def _to_iso_timestamp(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _to_nested_list(value: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def _torch_tensor(*, torch_module: Any, values: Any, dtype: Any, device: str | None) -> Any:
    constructor = getattr(torch_module, "as_tensor", None)
    if not callable(constructor):
        constructor = getattr(torch_module, "tensor", None)
    if not callable(constructor):
        raise AdapterInputError("torch dependency does not expose tensor construction")

    kwargs: dict[str, Any] = {}
    if dtype is not None:
        kwargs["dtype"] = dtype
    if device is not None:
        kwargs["device"] = device
    try:
        return constructor(values, **kwargs)
    except TypeError:
        fallback_kwargs = dict(kwargs)
        fallback_kwargs.pop("device", None)
        tensor = constructor(values, **fallback_kwargs)
        to_method = getattr(tensor, "to", None)
        if device is not None and callable(to_method):
            try:
                return to_method(device=device)
            except TypeError:
                return to_method(device)
        return tensor


def _configure_matmul_precision(torch_module: Any, *, matmul_precision: str) -> None:
    setter = getattr(torch_module, "set_float32_matmul_precision", None)
    if callable(setter):
        setter(matmul_precision)


def _compile_model(*, torch_module: Any, model: Any) -> Any:
    model_compile = getattr(model, "compile", None)
    if callable(model_compile):
        compiled = model_compile()
        return model if compiled is None else compiled

    torch_compile = getattr(torch_module, "compile", None)
    if callable(torch_compile):
        try:
            return torch_compile(model)
        except Exception as exc:  # noqa: BLE001
            raise AdapterInputError(f"torch.compile failed: {exc}") from exc
    raise AdapterInputError("torch_compile requested but compile is unavailable")


def _call_from_pretrained(
    *,
    model_cls: Any,
    model_ref: str,
    revision: str | None,
    kwargs: dict[str, Any],
) -> Any:
    from_pretrained = getattr(model_cls, "from_pretrained", None)
    if not callable(from_pretrained):
        raise DependencyMissingError(
            "toto-ts Toto class does not expose from_pretrained; "
            "install with `pip install -e \".[dev,runner_toto]\"`",
        )

    resolved_kwargs = dict(kwargs)
    if revision is not None:
        resolved_kwargs["revision"] = revision
    try:
        return from_pretrained(model_ref, **resolved_kwargs)
    except TypeError:
        resolved_kwargs.pop("revision", None)
        try:
            return from_pretrained(model_ref, **resolved_kwargs)
        except TypeError:
            resolved_kwargs.pop("map_location", None)
            return from_pretrained(model_ref, **resolved_kwargs)


def _move_model_to_device(*, model: Any, device: str, dtype: Any) -> None:
    to_method = getattr(model, "to", None)
    if not callable(to_method):
        return

    kwargs: dict[str, Any] = {"device": device}
    if dtype is not None:
        kwargs["dtype"] = dtype
    try:
        to_method(**kwargs)
        return
    except TypeError:
        kwargs.pop("dtype", None)
    try:
        to_method(**kwargs)
        return
    except TypeError:
        to_method(device)


def _torch_dtype_from_name(torch_module: Any, dtype_name: str) -> Any:
    mapping = {
        "float32": getattr(torch_module, "float32", None),
        "float16": getattr(torch_module, "float16", None),
        "bfloat16": getattr(torch_module, "bfloat16", None),
    }
    resolved = mapping.get(dtype_name)
    if resolved is None:
        raise AdapterInputError(
            f"unsupported dtype {dtype_name!r} for current torch build",
        )
    return resolved


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


def _dict_bool(payload: dict[str, Any] | None, key: str) -> bool | None:
    if not isinstance(payload, dict):
        return None
    return _bool_or_none(payload.get(key))


def _bool_or_none(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    return None
