"""TimesFM forecasting adapter used by the timesfm runner."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tollama.core.schemas import ForecastRequest, ForecastResponse, SeriesForecast, SeriesInput

from .errors import AdapterInputError, DependencyMissingError, UnsupportedModelError

_DEFAULT_QUANTILES = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
_TIMESFM_MODELS: dict[str, dict[str, Any]] = {
    "timesfm-2.5-200m": {
        "repo_id": "google/timesfm-2.5-200m-pytorch",
        "revision": "main",
        "implementation": "timesfm_2p5_torch",
        "max_context": 1024,
        "max_horizon": 256,
    },
    "timesfm2p5": {
        "repo_id": "google/timesfm-2.5",
        "revision": "main",
        "implementation": "timesfm_2p5_torch",
        "max_context": 1024,
        "max_horizon": 256,
    },
}


@dataclass(frozen=True)
class _TimesFMDependencies:
    numpy: Any
    pandas: Any
    timesfm: Any


@dataclass(frozen=True)
class _RuntimeConfig:
    model_name: str
    repo_id: str
    revision: str
    implementation: str
    max_context: int
    max_horizon: int


@dataclass(frozen=True)
class _CompileKey:
    model_name: str
    model_ref: str
    revision: str
    implementation: str
    max_context: int
    max_horizon: int
    use_quantile_head: bool
    torch_compile: bool
    infer_is_positive: bool


@dataclass(frozen=True)
class _CompiledTimesFMModel:
    model: Any
    runtime: _RuntimeConfig


class TimesFMAdapter:
    """Adapter that maps canonical request/response to TimesFM 2.5 inference."""

    def __init__(self) -> None:
        self._dependencies: _TimesFMDependencies | None = None
        self._compiled_models: dict[_CompileKey, _CompiledTimesFMModel] = {}

    def load(
        self,
        model_name: str,
        *,
        model_local_dir: str | None = None,
        model_source: dict[str, Any] | None = None,
        model_metadata: dict[str, Any] | None = None,
    ) -> None:
        """Load and compile one TimesFM model into cache."""
        runtime = _resolve_runtime_config(
            model_name=model_name,
            model_source=model_source,
            model_metadata=model_metadata,
        )
        request_horizon = runtime.max_horizon
        self._get_or_compile_model(
            model_name=model_name,
            runtime=runtime,
            model_local_dir=model_local_dir,
            request_horizon=request_horizon,
            requested_quantiles=[],
            options={},
        )

    def unload(self, model_name: str | None = None) -> None:
        """Unload one model or clear all compiled TimesFM entries."""
        if model_name is None:
            self._compiled_models.clear()
            return
        self._compiled_models = {
            key: value
            for key, value in self._compiled_models.items()
            if key.model_name != model_name
        }

    def forecast(
        self,
        request: ForecastRequest,
        *,
        model_local_dir: str | None = None,
        model_source: dict[str, Any] | None = None,
        model_metadata: dict[str, Any] | None = None,
    ) -> ForecastResponse:
        """Generate a TimesFM forecast from canonical request data."""
        runtime = _resolve_runtime_config(
            model_name=request.model,
            model_source=model_source,
            model_metadata=model_metadata,
        )
        compiled = self._get_or_compile_model(
            model_name=request.model,
            runtime=runtime,
            model_local_dir=model_local_dir,
            request_horizon=request.horizon,
            requested_quantiles=list(request.quantiles),
            options=request.options,
        )
        dependencies = self._resolve_dependencies()
        inputs = build_timesfm_inputs(
            series_list=request.series,
            max_context=compiled.runtime.max_context,
            numpy_module=dependencies.numpy,
        )

        forecast_output = compiled.model.forecast(horizon=request.horizon, inputs=inputs)
        point_forecast, quantile_forecast = _split_forecast_output(forecast_output)
        point_values = point_forecast_to_rows(
            point_forecast=point_forecast,
            n_series=len(request.series),
            horizon=request.horizon,
        )
        quantile_payloads = map_quantile_forecast(
            quantile_forecast=quantile_forecast,
            requested_quantiles=list(request.quantiles),
            n_series=len(request.series),
            horizon=request.horizon,
        )

        forecasts: list[SeriesForecast] = []
        for index, series in enumerate(request.series):
            future_timestamps = generate_future_timestamps(
                last_timestamp=series.timestamps[-1],
                freq=series.freq,
                horizon=request.horizon,
                pandas_module=dependencies.pandas,
            )
            forecasts.append(
                SeriesForecast(
                    id=series.id,
                    freq=series.freq,
                    start_timestamp=future_timestamps[0],
                    mean=point_values[index],
                    quantiles=quantile_payloads[index],
                ),
            )

        return ForecastResponse(
            model=request.model,
            forecasts=forecasts,
            usage={
                "runner": "tollama-timesfm",
                "implementation": compiled.runtime.implementation,
                "series_count": len(request.series),
                "horizon": request.horizon,
            },
        )

    def _get_or_compile_model(
        self,
        *,
        model_name: str,
        runtime: _RuntimeConfig,
        model_local_dir: str | None,
        request_horizon: int,
        requested_quantiles: list[float],
        options: dict[str, Any],
    ) -> _CompiledTimesFMModel:
        dependencies = self._resolve_dependencies()
        model_ref = _existing_local_model_path(model_local_dir) or runtime.repo_id
        model_revision = runtime.revision
        max_horizon = max(runtime.max_horizon, request_horizon)
        use_quantile_head = bool(requested_quantiles)
        torch_compile = (
            bool(options.get("torch_compile")) if "torch_compile" in options else False
        )
        infer_is_positive = (
            bool(options.get("infer_is_positive")) if "infer_is_positive" in options else False
        )
        key = _CompileKey(
            model_name=model_name,
            model_ref=model_ref,
            revision=model_revision,
            implementation=runtime.implementation,
            max_context=runtime.max_context,
            max_horizon=max_horizon,
            use_quantile_head=use_quantile_head,
            torch_compile=torch_compile,
            infer_is_positive=infer_is_positive,
        )
        cached = self._compiled_models.get(key)
        if cached is not None:
            return cached

        model = _load_timesfm_model(
            timesfm_module=dependencies.timesfm,
            model_ref=model_ref,
            revision=model_revision,
            torch_compile=torch_compile,
            local_model_path=_existing_local_model_path(model_local_dir) is not None,
        )
        forecast_config = dependencies.timesfm.ForecastConfig(
            max_context=runtime.max_context,
            max_horizon=max_horizon,
            normalize_inputs=True,
            use_continuous_quantile_head=use_quantile_head,
            force_flip_invariance=True,
            infer_is_positive=infer_is_positive,
            fix_quantile_crossing=True,
        )
        model.compile(forecast_config)
        compiled = _CompiledTimesFMModel(model=model, runtime=runtime)
        self._compiled_models[key] = compiled
        return compiled

    def _resolve_dependencies(self) -> _TimesFMDependencies:
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
            import timesfm
        except ModuleNotFoundError as exc:
            missing_packages.append(exc.name or "timesfm")
            timesfm = None

        if missing_packages:
            joined = ", ".join(sorted(set(missing_packages)))
            raise DependencyMissingError(
                "missing optional timesfm runner dependencies "
                f"({joined}); install them with `pip install -e \".[dev,runner_timesfm]\"`",
            )

        assert np is not None
        assert pd is not None
        assert timesfm is not None
        resolved = _TimesFMDependencies(numpy=np, pandas=pd, timesfm=timesfm)
        self._dependencies = resolved
        return resolved


def truncate_target_to_max_context(target: Sequence[int | float], max_context: int) -> list[float]:
    """Convert target values to float and truncate to the configured context window."""
    if max_context <= 0:
        raise AdapterInputError("max_context must be greater than zero")
    if len(target) < 2:
        raise AdapterInputError("each series must include at least two target points")

    window = target[-max_context:]
    values: list[float] = []
    for value in window:
        if isinstance(value, bool):
            values.append(float(int(value)))
            continue
        if isinstance(value, (int, float)):
            values.append(float(value))
            continue
        raise AdapterInputError(f"target contains a non-numeric value: {value!r}")
    return values


def build_timesfm_inputs(
    *,
    series_list: Sequence[SeriesInput],
    max_context: int,
    numpy_module: Any,
) -> list[Any]:
    """Build one TimesFM input array per series."""
    arrays: list[Any] = []
    for series in series_list:
        truncated = truncate_target_to_max_context(series.target, max_context)
        arrays.append(numpy_module.asarray(truncated, dtype=float))
    return arrays


def point_forecast_to_rows(
    *,
    point_forecast: Any,
    n_series: int,
    horizon: int,
) -> list[list[float]]:
    """Normalize point forecast output to list[list[float]] with shape [n_series, horizon]."""
    try:
        rows_3d = _to_float_3d(point_forecast)
    except AdapterInputError:
        rows_3d = []
    if rows_3d and len(rows_3d) == n_series and rows_3d[0] and len(rows_3d[0][0]) == 1:
        normalized_3d: list[list[float]] = []
        for row in rows_3d:
            if len(row) < horizon:
                raise AdapterInputError(
                    "TimesFM point forecast output is shorter than requested horizon",
                )
            normalized_3d.append([row[index][0] for index in range(horizon)])
        return normalized_3d

    rows_2d = _to_float_2d(point_forecast)
    if len(rows_2d) != n_series:
        raise AdapterInputError(
            "TimesFM point forecast output series dimension does not match request series count",
        )

    normalized: list[list[float]] = []
    for row in rows_2d:
        if len(row) < horizon:
            raise AdapterInputError(
                "TimesFM point forecast output is shorter than requested horizon",
            )
        normalized.append(row[:horizon])
    return normalized


def map_quantile_forecast(
    *,
    quantile_forecast: Any,
    requested_quantiles: list[float],
    n_series: int,
    horizon: int,
) -> list[dict[str, list[float]] | None]:
    """Map TimesFM quantile output channels to the canonical quantile dictionary payload."""
    if not requested_quantiles:
        return [None] * n_series
    if quantile_forecast is None:
        raise AdapterInputError(
            "TimesFM quantile output is unavailable for requested quantiles",
        )

    rows = _to_float_3d(quantile_forecast)
    if len(rows) != n_series:
        raise AdapterInputError(
            "TimesFM quantile output series dimension does not match request series count",
        )

    channel_map = _quantile_channel_index(rows)
    requested = [_normalize_quantile(value) for value in requested_quantiles]
    missing = [value for value in requested if value not in channel_map]
    if missing:
        first = format(missing[0], "g")
        raise AdapterInputError(
            f"requested quantile {first} is not available from TimesFM output",
        )

    payloads: list[dict[str, list[float]] | None] = []
    for series_rows in rows:
        if len(series_rows) < horizon:
            raise AdapterInputError(
                "TimesFM quantile output is shorter than requested horizon",
            )
        mapped: dict[str, list[float]] = {}
        for quantile in requested:
            channel = channel_map[quantile]
            mapped[format(quantile, "g")] = [
                float(series_rows[index][channel]) for index in range(horizon)
            ]
        payloads.append(mapped)
    return payloads


def generate_future_timestamps(
    *,
    last_timestamp: str,
    freq: str,
    horizon: int,
    pandas_module: Any,
) -> list[str]:
    """Generate horizon future timestamps from the final observed timestamp and frequency."""
    if horizon <= 0:
        raise AdapterInputError("horizon must be greater than zero")

    parsed = pandas_module.to_datetime([last_timestamp], utc=True, errors="raise")
    if not parsed:
        raise AdapterInputError("series timestamp parsing returned no values")
    start = parsed[0]
    try:
        generated = pandas_module.date_range(start=start, periods=horizon + 1, freq=freq)
    except ValueError as exc:
        raise AdapterInputError(f"invalid frequency {freq!r} for TimesFM forecast") from exc
    return [_to_iso_timestamp(value) for value in list(generated[1:])]


def _resolve_runtime_config(
    *,
    model_name: str,
    model_source: dict[str, Any] | None,
    model_metadata: dict[str, Any] | None,
) -> _RuntimeConfig:
    defaults = _TIMESFM_MODELS.get(model_name, {})
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

    if repo_id is None:
        raise UnsupportedModelError(
            f"unsupported timesfm model {model_name!r}; missing repo_id metadata",
        )
    if revision is None:
        revision = "main"
    if implementation is None:
        implementation = "timesfm_2p5_torch"
    if max_context is None:
        max_context = 1024
    if max_horizon is None:
        max_horizon = 256
    return _RuntimeConfig(
        model_name=model_name,
        repo_id=repo_id,
        revision=revision,
        implementation=implementation,
        max_context=max_context,
        max_horizon=max_horizon,
    )


def _load_timesfm_model(
    *,
    timesfm_module: Any,
    model_ref: str,
    revision: str,
    torch_compile: bool,
    local_model_path: bool,
) -> Any:
    model_cls = getattr(timesfm_module, "TimesFM_2p5_200M_torch", None)
    if model_cls is None:
        raise DependencyMissingError(
            "timesfm package does not expose TimesFM_2p5_200M_torch; "
            "install with `pip install -e \".[dev,runner_timesfm]\"`",
        )

    kwargs: dict[str, Any] = {}
    if torch_compile:
        kwargs["torch_compile"] = True

    if local_model_path:
        try:
            return model_cls.from_pretrained(model_ref, **kwargs)
        except TypeError:
            return model_cls.from_pretrained(model_ref)

    try:
        return model_cls.from_pretrained(model_ref, revision=revision, **kwargs)
    except TypeError:
        if kwargs:
            try:
                return model_cls.from_pretrained(model_ref, revision=revision)
            except TypeError:
                return model_cls.from_pretrained(model_ref)
        return model_cls.from_pretrained(model_ref)


def _split_forecast_output(output: Any) -> tuple[Any, Any | None]:
    if isinstance(output, tuple):
        if len(output) >= 2:
            return output[0], output[1]
        if len(output) == 1:
            return output[0], None
    if isinstance(output, dict):
        point = output.get("point_forecast")
        if point is None:
            point = output.get("mean")
        if point is None:
            point = output.get("forecast")

        quantiles = output.get("quantile_forecast")
        if quantiles is None:
            quantiles = output.get("quantiles")
        if point is None:
            raise AdapterInputError("TimesFM forecast output is missing point forecasts")
        return point, quantiles
    return output, None


def _quantile_channel_index(rows: list[list[list[float]]]) -> dict[float, int]:
    if not rows or not rows[0]:
        raise AdapterInputError("TimesFM quantile output is empty")

    channels = len(rows[0][0])
    if channels == 9:
        offset = 0
    elif channels == 10:
        offset = 1
    else:
        raise AdapterInputError(
            "TimesFM quantile output has unsupported channel count "
            f"({channels}); expected 9 or 10",
        )
    return {
        _normalize_quantile(value): index + offset
        for index, value in enumerate(_DEFAULT_QUANTILES)
    }


def _to_float_2d(value: Any) -> list[list[float]]:
    rows = _to_nested_list(value)
    if not isinstance(rows, list):
        raise AdapterInputError("TimesFM output must be list-like")

    normalized: list[list[float]] = []
    for row in rows:
        if not isinstance(row, list):
            raise AdapterInputError("TimesFM output row is not list-like")
        values: list[float] = []
        for item in row:
            values.append(_to_float(item))
        normalized.append(values)
    return normalized


def _to_float_3d(value: Any) -> list[list[list[float]]]:
    outer = _to_nested_list(value)
    if not isinstance(outer, list):
        raise AdapterInputError("TimesFM output must be list-like")

    normalized: list[list[list[float]]] = []
    for matrix in outer:
        if not isinstance(matrix, list):
            raise AdapterInputError("TimesFM output matrix is not list-like")
        rows: list[list[float]] = []
        for row in matrix:
            if not isinstance(row, list):
                raise AdapterInputError("TimesFM output row is not list-like")
            values: list[float] = [_to_float(item) for item in row]
            rows.append(values)
        normalized.append(rows)
    return normalized


def _to_nested_list(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def _to_float(value: Any) -> float:
    if isinstance(value, bool):
        return float(int(value))
    if isinstance(value, (int, float)):
        return float(value)
    if hasattr(value, "item"):
        scalar = value.item()
        if isinstance(scalar, (int, float)):
            return float(scalar)
    raise AdapterInputError(f"TimesFM output contains non-numeric value: {value!r}")


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


def _normalize_quantile(value: float) -> float:
    return round(float(value), 6)
