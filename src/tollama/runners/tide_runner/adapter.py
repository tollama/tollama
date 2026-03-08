"""TiDE forecasting adapter used by the tide runner."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tollama.core.schemas import ForecastRequest, ForecastResponse, SeriesForecast, SeriesInput

from .errors import AdapterInputError, DependencyMissingError, UnsupportedModelError

_TIDE_MODELS: dict[str, dict[str, Any]] = {
    "tide": {
        "repo_id": "tollama/tide-runner",
        "revision": "main",
        "implementation": "tide",
    },
}
_DEFAULT_QUANTILE_SAMPLES = 200


@dataclass(frozen=True)
class _Dependencies:
    np: Any
    pd: Any
    time_series_cls: Any
    tide_model_cls: Any


@dataclass(frozen=True)
class _RuntimeConfig:
    model_name: str
    repo_id: str
    revision: str
    implementation: str


class TideAdapter:
    def __init__(self) -> None:
        self._dependencies: _Dependencies | None = None
        self._model_cache: dict[tuple[str, str, str], Any] = {}

    def unload(self, model_name: str | None = None) -> None:
        if model_name is None:
            self._model_cache.clear()
            return
        self._model_cache = {
            key: value
            for key, value in self._model_cache.items()
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
        runtime = _resolve_runtime_config(
            model_name=request.model,
            model_source=model_source,
            model_metadata=model_metadata,
        )
        deps = self._resolve_dependencies()
        model = self._get_or_load_model(runtime=runtime, model_local_dir=model_local_dir)

        quantile_samples = _resolve_quantile_samples(request.options.get("quantile_samples"))
        forecasts: list[SeriesForecast] = []
        warnings: list[str] = []
        quantile_fallback = False

        for series in request.series:
            target_series = _build_target_series(
                series=series,
                pd_module=deps.pd,
                np_module=deps.np,
                ts_cls=deps.time_series_cls,
            )
            model_for_series = _prepare_model_for_series(
                model=model,
                tide_model_cls=deps.tide_model_cls,
                series_length=len(series.target),
                horizon=request.horizon,
                options=request.options,
                target_series=target_series,
            )
            mean, quantiles = _forecast_one_series(
                model=model_for_series,
                target_series=target_series,
                horizon=request.horizon,
                requested_quantiles=list(request.quantiles),
                quantile_samples=quantile_samples,
            )
            if request.quantiles and quantiles is None:
                quantile_fallback = True

            forecasts.append(
                SeriesForecast(
                    id=series.id,
                    freq=series.freq,
                    start_timestamp=_future_start_timestamp(
                        series=series,
                        horizon=request.horizon,
                        pd_module=deps.pd,
                    ),
                    mean=mean,
                    quantiles=quantiles,
                ),
            )

        if quantile_fallback:
            warnings.append(
                "TiDE runtime did not expose quantile outputs for requested levels; "
                "returning mean forecasts only",
            )

        return ForecastResponse(
            model=request.model,
            forecasts=forecasts,
            usage={
                "runner": "tollama-tide",
                "implementation": runtime.implementation,
                "series_count": len(forecasts),
                "horizon": request.horizon,
                "quantile_samples": quantile_samples,
            },
            warnings=warnings or None,
        )

    def _get_or_load_model(self, *, runtime: _RuntimeConfig, model_local_dir: str | None) -> Any:
        deps = self._resolve_dependencies()
        model_ref = _existing_local_model_path(model_local_dir) or runtime.repo_id
        key = (runtime.model_name, model_ref, runtime.revision)
        cached = self._model_cache.get(key)
        if cached is not None:
            return cached

        model = None
        if hasattr(deps.tide_model_cls, "load"):
            try:
                model = deps.tide_model_cls.load(model_ref)
            except Exception:
                model = None

        if model is None and hasattr(deps.tide_model_cls, "from_pretrained"):
            try:
                model = deps.tide_model_cls.from_pretrained(model_ref)
            except Exception:
                model = None

        # Local TiDE integration can run without a serialized checkpoint.
        # In that case we keep the class and build+fit per-series at runtime.
        if model is None:
            model = deps.tide_model_cls

        self._model_cache[key] = model
        return model

    def _resolve_dependencies(self) -> _Dependencies:
        if self._dependencies is not None:
            return self._dependencies

        missing: list[str] = []
        try:
            import numpy as np
        except ModuleNotFoundError as exc:
            missing.append(exc.name or "numpy")
            np = None

        try:
            import pandas as pd
        except ModuleNotFoundError as exc:
            missing.append(exc.name or "pandas")
            pd = None

        time_series_cls = None
        tide_model_cls = None
        try:
            from darts import TimeSeries
            from darts.models import TiDEModel

            time_series_cls = TimeSeries
            tide_model_cls = TiDEModel
        except ModuleNotFoundError as exc:
            missing.append(exc.name or "darts")

        if missing or time_series_cls is None or tide_model_cls is None or np is None or pd is None:
            joined = ", ".join(sorted(set(missing or ["darts"])))
            raise DependencyMissingError(
                "missing optional tide runner dependencies "
                f"({joined}); install them with `pip install -e \".[dev,runner_tide]\"`",
            )

        deps = _Dependencies(
            np=np,
            pd=pd,
            time_series_cls=time_series_cls,
            tide_model_cls=tide_model_cls,
        )
        self._dependencies = deps
        return deps


def _prepare_model_for_series(
    *,
    model: Any,
    tide_model_cls: Any,
    series_length: int,
    horizon: int,
    options: dict[str, Any],
    target_series: Any,
) -> Any:
    if isinstance(model, type):
        model = _build_runtime_tide_model(
            tide_model_cls=tide_model_cls,
            series_length=series_length,
            horizon=horizon,
            options=options,
        )

    fit_fn = getattr(model, "fit", None)
    if callable(fit_fn):
        try:
            fit_fn(series=target_series, verbose=False)
        except TypeError:
            try:
                fit_fn(target_series, verbose=False)
            except TypeError:
                fit_fn(target_series)
        except Exception as exc:  # noqa: BLE001
            raise AdapterInputError(f"TiDE fit failed: {exc}") from exc

    return model


def _build_runtime_tide_model(
    *,
    tide_model_cls: Any,
    series_length: int,
    horizon: int,
    options: dict[str, Any],
) -> Any:
    input_chunk_length = _resolve_positive_int_option(
        options.get("context_length"),
        default=min(64, max(2, series_length - 1)),
        field_name="context_length",
    )
    if input_chunk_length >= series_length:
        input_chunk_length = max(2, series_length - 1)

    output_chunk_length = _resolve_positive_int_option(
        options.get("output_chunk_length"),
        default=max(1, min(horizon, 32)),
        field_name="output_chunk_length",
    )

    n_epochs = _resolve_positive_int_option(
        options.get("tide_epochs"),
        default=1,
        field_name="tide_epochs",
    )

    trial_kwargs = {
        "input_chunk_length": input_chunk_length,
        "output_chunk_length": output_chunk_length,
        "n_epochs": n_epochs,
        "random_state": 42,
    }
    try:
        return tide_model_cls(**trial_kwargs)
    except TypeError:
        try:
            return tide_model_cls(
                input_chunk_length=input_chunk_length,
                output_chunk_length=output_chunk_length,
            )
        except Exception as exc:  # noqa: BLE001
            raise AdapterInputError(f"failed to initialize TiDE model: {exc}") from exc


def _resolve_positive_int_option(raw: Any, *, default: int, field_name: str) -> int:
    if raw is None:
        return default
    if isinstance(raw, bool) or not isinstance(raw, int) or raw <= 0:
        raise AdapterInputError(f"{field_name} option must be a positive integer")
    return raw


def _forecast_one_series(
    *,
    model: Any,
    target_series: Any,
    horizon: int,
    requested_quantiles: list[float],
    quantile_samples: int,
) -> tuple[list[float], dict[str, list[float]] | None]:
    mean_prediction = _predict(model=model, series=target_series, horizon=horizon, num_samples=1)
    mean = _timeseries_to_vector(mean_prediction, label="mean", horizon=horizon)

    if not requested_quantiles:
        return mean, None

    probabilistic_prediction = _predict(
        model=model,
        series=target_series,
        horizon=horizon,
        num_samples=quantile_samples,
    )
    quantiles = _extract_quantiles(
        probabilistic_prediction,
        requested_quantiles=requested_quantiles,
        horizon=horizon,
    )
    return mean, quantiles


def _predict(*, model: Any, series: Any, horizon: int, num_samples: int) -> Any:
    try:
        return model.predict(n=horizon, series=series, num_samples=num_samples)
    except TypeError:
        return model.predict(n=horizon, num_samples=num_samples)
    except Exception as exc:  # noqa: BLE001
        raise AdapterInputError(f"TiDE prediction failed: {exc}") from exc


def _extract_quantiles(
    prediction: Any,
    *,
    requested_quantiles: list[float],
    horizon: int,
) -> dict[str, list[float]] | None:
    quantile_ts_fn = getattr(prediction, "quantile_timeseries", None)
    if not callable(quantile_ts_fn):
        return None

    payload: dict[str, list[float]] = {}
    for value in requested_quantiles:
        q = float(value)
        try:
            quantile_series = quantile_ts_fn(q)
        except Exception:
            return None
        payload[format(q, "g")] = _timeseries_to_vector(
            quantile_series,
            label=f"quantile {format(q, 'g')}",
            horizon=horizon,
        )
    return payload


def _timeseries_to_vector(series: Any, *, label: str, horizon: int) -> list[float]:
    values_fn = getattr(series, "values", None)
    if callable(values_fn):
        values = values_fn()
    else:
        values = getattr(series, "values", None)

    if values is None and hasattr(series, "to_numpy"):
        values = series.to_numpy()

    if values is None:
        raise AdapterInputError(f"TiDE {label} output is unavailable")

    if hasattr(values, "tolist"):
        values = values.tolist()

    flattened: list[float] = []
    if isinstance(values, list):
        while values and isinstance(values[0], list):
            values = [row[0] if isinstance(row, list) and row else row for row in values]
        for item in values:
            if isinstance(item, bool):
                flattened.append(float(int(item)))
            elif isinstance(item, (int, float)):
                flattened.append(float(item))
            elif hasattr(item, "item"):
                flattened.append(float(item.item()))
            else:
                raise AdapterInputError(f"TiDE {label} output contains non-numeric values")
    else:
        raise AdapterInputError(f"TiDE {label} output has unexpected shape")

    if len(flattened) < horizon:
        raise AdapterInputError(
            f"TiDE {label} output is shorter than requested horizon "
            f"({len(flattened)} < {horizon})",
        )
    return flattened[:horizon]


def _build_target_series(
    *,
    series: SeriesInput,
    pd_module: Any,
    np_module: Any,
    ts_cls: Any,
) -> Any:
    if len(series.timestamps) != len(series.target):
        raise AdapterInputError(f"series {series.id!r} timestamps and target lengths must match")
    if len(series.target) < 2:
        raise AdapterInputError("each input series must include at least two target points")

    try:
        index = pd_module.to_datetime(series.timestamps, utc=True, errors="raise")
    except Exception as exc:  # noqa: BLE001
        raise AdapterInputError(f"invalid timestamps for series {series.id!r}: {exc}") from exc

    try:
        frame = pd_module.DataFrame(
            {"target": [float(v) for v in series.target]},
            index=index,
            dtype="float32",
        )
        try:
            frame = frame.resample(series.freq).asfreq()
            frame["target"] = frame["target"].fillna(0.0)
        except Exception:
            pass
        return ts_cls.from_times_and_values(
            frame.index,
            frame["target"].values,
            freq=series.freq,
            fill_missing_dates=True,
            fillna_value=0.0,
        )
    except Exception as exc:  # noqa: BLE001
        raise AdapterInputError(
            f"failed to build TiDE target series for {series.id!r}: {exc}",
        ) from exc


def _future_start_timestamp(*, series: SeriesInput, horizon: int, pd_module: Any) -> str:
    try:
        parsed = pd_module.to_datetime([series.timestamps[-1]], utc=True, errors="raise")
        generated = pd_module.date_range(start=parsed[0], periods=horizon + 1, freq=series.freq)
    except Exception as exc:  # noqa: BLE001
        raise AdapterInputError(f"invalid frequency {series.freq!r} for tide forecast") from exc
    return _to_iso_timestamp(generated[1])


def _to_iso_timestamp(value: Any) -> str:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=UTC)
        return value.astimezone(UTC).isoformat().replace("+00:00", "Z")
    isoformat = getattr(value, "isoformat", None)
    if callable(isoformat):
        return str(isoformat()).replace("+00:00", "Z")
    return str(value)


def _resolve_quantile_samples(raw: Any) -> int:
    if raw is None:
        return _DEFAULT_QUANTILE_SAMPLES
    if isinstance(raw, bool) or not isinstance(raw, int) or raw <= 1:
        raise AdapterInputError("quantile_samples option must be an integer greater than 1")
    return raw


def _resolve_runtime_config(
    *,
    model_name: str,
    model_source: dict[str, Any] | None,
    model_metadata: dict[str, Any] | None,
) -> _RuntimeConfig:
    defaults = _TIDE_MODELS.get(model_name, {})
    repo_id = _dict_str(model_source, "repo_id") or _string_or_none(defaults.get("repo_id"))
    revision = (
        _dict_str(model_source, "revision")
        or _string_or_none(defaults.get("revision"))
        or "main"
    )
    implementation = (
        _dict_str(model_metadata, "implementation")
        or _string_or_none(defaults.get("implementation"))
        or "tide"
    )

    if repo_id is None:
        raise UnsupportedModelError(f"unsupported tide model {model_name!r}; missing repo_id")

    return _RuntimeConfig(
        model_name=model_name,
        repo_id=repo_id,
        revision=revision,
        implementation=implementation,
    )


def _existing_local_model_path(model_local_dir: str | None) -> str | None:
    if not model_local_dir:
        return None
    path = Path(model_local_dir.strip())
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
