"""Lag-Llama forecasting adapter used by the lag_llama runner."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tollama.core.schemas import ForecastRequest, ForecastResponse, SeriesForecast, SeriesInput

from .errors import AdapterInputError, DependencyMissingError, UnsupportedModelError

_DEFAULT_CONTEXT_LENGTH = 1024
_DEFAULT_NUM_SAMPLES = 100
_DEFAULT_BATCH_SIZE = 32
_DEFAULT_CHECKPOINT_FILENAME = "lag-llama.ckpt"

_LAG_LLAMA_MODELS: dict[str, dict[str, Any]] = {
    "lag-llama": {
        "repo_id": "time-series-foundation-models/Lag-Llama",
        "revision": "main",
        "implementation": "lag_llama",
        "default_context_length": _DEFAULT_CONTEXT_LENGTH,
        "default_num_samples": _DEFAULT_NUM_SAMPLES,
        "default_batch_size": _DEFAULT_BATCH_SIZE,
        "checkpoint_filename": _DEFAULT_CHECKPOINT_FILENAME,
    },
}


@dataclass(frozen=True)
class _Dependencies:
    pandas: Any
    hf_hub_download: Any
    pandas_dataset_cls: Any
    lag_llama_estimator_cls: Any


@dataclass(frozen=True)
class _RuntimeConfig:
    model_name: str
    repo_id: str
    revision: str
    implementation: str
    default_context_length: int
    default_num_samples: int
    default_batch_size: int
    checkpoint_filename: str


@dataclass(frozen=True)
class _PredictorConfig:
    prediction_length: int
    context_length: int
    num_samples: int
    batch_size: int


class LagLlamaAdapter:
    """Adapter that maps canonical request/response to Lag-Llama predictor calls."""

    def __init__(self) -> None:
        self._dependencies: _Dependencies | None = None
        self._predictor_cache: dict[tuple[str, str, str, int, int, int], Any] = {}

    def load(
        self,
        model_name: str,
        *,
        model_local_dir: str | None = None,
        model_source: dict[str, Any] | None = None,
        model_metadata: dict[str, Any] | None = None,
    ) -> None:
        runtime = _resolve_runtime_config(
            model_name=model_name,
            model_source=model_source,
            model_metadata=model_metadata,
        )
        self._get_or_build_predictor(
            runtime=runtime,
            model_local_dir=model_local_dir,
            predictor_config=_PredictorConfig(
                prediction_length=1,
                context_length=runtime.default_context_length,
                num_samples=runtime.default_num_samples,
                batch_size=runtime.default_batch_size,
            ),
        )

    def unload(self, model_name: str | None = None) -> None:
        if model_name is None:
            self._predictor_cache.clear()
            return
        self._predictor_cache = {
            key: value for key, value in self._predictor_cache.items() if key[0] != model_name
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

        context_length = resolve_positive_int(
            option_value=request.options.get("context_length"),
            default_value=runtime.default_context_length,
            field_name="context_length",
        )
        num_samples = resolve_positive_int(
            option_value=request.options.get("num_samples"),
            default_value=runtime.default_num_samples,
            field_name="num_samples",
        )
        batch_size = resolve_positive_int(
            option_value=request.options.get("batch_size"),
            default_value=runtime.default_batch_size,
            field_name="batch_size",
        )
        predictor_config = _PredictorConfig(
            prediction_length=request.horizon,
            context_length=context_length,
            num_samples=num_samples,
            batch_size=batch_size,
        )

        dependencies = self._resolve_dependencies()
        predictor = self._get_or_build_predictor(
            runtime=runtime,
            model_local_dir=model_local_dir,
            predictor_config=predictor_config,
        )
        dataset, ordered_series = build_pandas_dataset(
            series_list=request.series,
            pandas_module=dependencies.pandas,
            pandas_dataset_cls=dependencies.pandas_dataset_cls,
        )
        forecasts = list(predictor.predict(dataset))
        if len(forecasts) != len(ordered_series):
            raise AdapterInputError(
                "Lag-Llama predictor returned a different number of series forecasts than requested",
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

        warnings = build_covariate_warnings(request.series)
        return ForecastResponse(
            model=request.model,
            forecasts=payloads,
            usage={
                "runner": "tollama-lag-llama",
                "implementation": runtime.implementation,
                "series_count": len(payloads),
                "horizon": request.horizon,
                "context_length": context_length,
                "num_samples": num_samples,
            },
            warnings=warnings or None,
        )

    def _get_or_build_predictor(
        self,
        *,
        runtime: _RuntimeConfig,
        model_local_dir: str | None,
        predictor_config: _PredictorConfig,
    ) -> Any:
        deps = self._resolve_dependencies()
        ckpt_path = resolve_checkpoint_path(
            dependencies=deps,
            runtime=runtime,
            model_local_dir=model_local_dir,
        )
        key = (
            runtime.model_name,
            ckpt_path,
            runtime.revision,
            predictor_config.prediction_length,
            predictor_config.context_length,
            predictor_config.num_samples,
        )
        cached = self._predictor_cache.get(key)
        if cached is not None:
            return cached

        estimator = create_lag_llama_estimator(
            estimator_cls=deps.lag_llama_estimator_cls,
            ckpt_path=ckpt_path,
            prediction_length=predictor_config.prediction_length,
            context_length=predictor_config.context_length,
            num_samples=predictor_config.num_samples,
        )
        create_predictor = getattr(estimator, "create_predictor", None)
        if not callable(create_predictor):
            raise AdapterInputError("LagLlamaEstimator does not expose create_predictor")
        try:
            predictor = create_predictor(batch_size=predictor_config.batch_size)
        except TypeError:
            try:
                transformation = estimator.create_transformation()
                module = estimator.create_lightning_module()
                predictor = create_predictor(transformation, module)
            except TypeError:
                predictor = create_predictor()

        self._predictor_cache[key] = predictor
        return predictor

    def _resolve_dependencies(self) -> _Dependencies:
        cached = self._dependencies
        if cached is not None:
            return cached

        missing_packages: list[str] = []
        try:
            import pandas as pd
        except ModuleNotFoundError as exc:
            missing_packages.append(exc.name or "pandas")
            pd = None

        try:
            from huggingface_hub import hf_hub_download
        except ModuleNotFoundError as exc:
            missing_packages.append(exc.name or "huggingface_hub")
            hf_hub_download = None

        try:
            from gluonts.dataset.pandas import PandasDataset
        except ModuleNotFoundError as exc:
            missing_packages.append(exc.name or "gluonts")
            PandasDataset = None

        estimator_cls = None
        for path in (
            "lag_llama.gluon.estimator:LagLlamaEstimator",
            "lag_llama.estimator:LagLlamaEstimator",
        ):
            module_name, class_name = path.split(":", 1)
            try:
                module = __import__(module_name, fromlist=[class_name])
                estimator_cls = getattr(module, class_name, None)
                if estimator_cls is not None:
                    break
            except ModuleNotFoundError as exc:
                missing_packages.append(exc.name or "lag-llama")

        if estimator_cls is None:
            missing_packages.append("lag-llama")

        if missing_packages:
            joined = ", ".join(sorted(set(missing_packages)))
            raise DependencyMissingError(
                "missing optional lag-llama runner dependencies "
                f"({joined}); install them with `pip install -e \".[dev,runner_lag_llama]\"`",
            )

        assert pd is not None
        assert hf_hub_download is not None
        assert PandasDataset is not None
        assert estimator_cls is not None
        resolved = _Dependencies(
            pandas=pd,
            hf_hub_download=hf_hub_download,
            pandas_dataset_cls=PandasDataset,
            lag_llama_estimator_cls=estimator_cls,
        )
        self._dependencies = resolved
        return resolved


def create_lag_llama_estimator(
    *,
    estimator_cls: Any,
    ckpt_path: str,
    prediction_length: int,
    context_length: int,
    num_samples: int,
) -> Any:
    checkpoint_hparams = load_checkpoint_hparams(ckpt_path)
    model_kwargs = checkpoint_hparams.get("model_kwargs", {})

    ckpt_context_length = _int_or_none(model_kwargs.get("context_length")) or _int_or_none(
        checkpoint_hparams.get("context_length"),
    )
    resolved_context_length = ckpt_context_length or context_length

    ckpt_max_context_length = _int_or_none(model_kwargs.get("max_context_length"))
    resolved_max_context_length = max(context_length, ckpt_max_context_length or resolved_context_length)

    architecture_overrides: dict[str, Any] = {
        "context_length": resolved_context_length,
        "max_context_length": resolved_max_context_length,
        "input_size": _int_or_none(model_kwargs.get("input_size")),
        "n_layer": _int_or_none(model_kwargs.get("n_layer")),
        "n_embd_per_head": _int_or_none(model_kwargs.get("n_embd_per_head")),
        "n_head": _int_or_none(model_kwargs.get("n_head")),
        "scaling": _string_or_none(model_kwargs.get("scaling")),
        "rope_scaling": model_kwargs.get("rope_scaling"),
        "time_feat": model_kwargs.get("time_feat") if isinstance(model_kwargs.get("time_feat"), bool) else None,
        "dropout": float(model_kwargs.get("dropout")) if isinstance(model_kwargs.get("dropout"), (int, float)) else None,
        "lags_seq": list(model_kwargs.get("lags_seq"))
        if isinstance(model_kwargs.get("lags_seq"), list)
        and all(isinstance(item, str) for item in model_kwargs.get("lags_seq"))
        else None,
    }
    architecture_overrides = {key: value for key, value in architecture_overrides.items() if value is not None}

    base_variants = (
        {
            "ckpt_path": ckpt_path,
            "prediction_length": prediction_length,
            "context_length": context_length,
            "num_samples": num_samples,
        },
        {
            "checkpoint_path": ckpt_path,
            "prediction_length": prediction_length,
            "context_length": context_length,
            "num_samples": num_samples,
        },
        {
            "model_path": ckpt_path,
            "prediction_length": prediction_length,
            "context_length": context_length,
            "num_samples": num_samples,
        },
        {
            "ckpt_path": ckpt_path,
            "prediction_length": prediction_length,
            "context_length": context_length,
            "num_parallel_samples": num_samples,
        },
        {
            "ckpt_path": ckpt_path,
            "prediction_length": prediction_length,
            "max_context_length": context_length,
            "num_parallel_samples": num_samples,
        },
        {
            "checkpoint_path": ckpt_path,
            "prediction_length": prediction_length,
            "max_context_length": context_length,
            "num_parallel_samples": num_samples,
        },
        {
            "model_path": ckpt_path,
            "prediction_length": prediction_length,
            "max_context_length": context_length,
            "num_parallel_samples": num_samples,
        },
    )
    variants = tuple({**kwargs, **architecture_overrides} for kwargs in base_variants)

    last_error: TypeError | None = None
    for kwargs in variants:
        try:
            return estimator_cls(**kwargs)
        except TypeError as exc:
            last_error = exc
    raise AdapterInputError(
        "LagLlamaEstimator constructor signature is incompatible with tollama runner",
    ) from last_error


def load_checkpoint_hparams(ckpt_path: str) -> dict[str, Any]:
    try:
        import torch
    except ModuleNotFoundError:
        return {}

    try:
        checkpoint = torch.load(ckpt_path, map_location="cpu")
    except Exception:  # noqa: BLE001
        return {}
    if not isinstance(checkpoint, dict):
        return {}

    hyper_parameters = checkpoint.get("hyper_parameters")
    return hyper_parameters if isinstance(hyper_parameters, dict) else {}


def resolve_checkpoint_path(
    *,
    dependencies: _Dependencies,
    runtime: _RuntimeConfig,
    model_local_dir: str | None,
) -> str:
    local_ckpt = _existing_local_checkpoint_path(
        model_local_dir=model_local_dir,
        checkpoint_filename=runtime.checkpoint_filename,
    )
    if local_ckpt is not None:
        return local_ckpt

    try:
        return str(
            dependencies.hf_hub_download(
                repo_id=runtime.repo_id,
                filename=runtime.checkpoint_filename,
                revision=runtime.revision,
            ),
        )
    except TypeError:
        return str(
            dependencies.hf_hub_download(
                repo_id=runtime.repo_id,
                filename=runtime.checkpoint_filename,
            ),
        )


def build_pandas_dataset(
    *,
    series_list: Sequence[SeriesInput],
    pandas_module: Any,
    pandas_dataset_cls: Any,
) -> tuple[Any, list[SeriesInput]]:
    frames: dict[str, Any] = {}
    dataset_freq: str | None = None
    for index, series in enumerate(series_list):
        if len(series.timestamps) != len(series.target):
            raise AdapterInputError(
                f"series {series.id!r} timestamps and target lengths must match",
            )
        if len(series.target) < 2:
            raise AdapterInputError("each input series must include at least two target points")

        if dataset_freq is None:
            dataset_freq = series.freq
        elif series.freq != dataset_freq:
            raise AdapterInputError("all input series must use the same frequency for lag-llama")

        timestamps = pandas_module.to_datetime(series.timestamps, utc=True, errors="raise")
        frame = pandas_module.DataFrame(
            {"target": [float(value) for value in series.target]},
            index=timestamps,
            dtype="float32",
        )
        key = _unique_column_name(series.id, index, frames)
        frames[key] = frame

    return pandas_dataset_cls(frames, target="target", freq=dataset_freq), list(series_list)


def resolve_positive_int(*, option_value: Any, default_value: int, field_name: str) -> int:
    if option_value is None:
        if default_value <= 0:
            raise AdapterInputError(f"{field_name} must be greater than zero")
        return default_value
    if isinstance(option_value, bool) or not isinstance(option_value, int):
        raise AdapterInputError(f"{field_name} option must be an integer")
    if option_value <= 0:
        raise AdapterInputError(f"{field_name} option must be greater than zero")
    return option_value


def normalize_forecast_vector(values: Any, *, horizon: int, label: str) -> list[float]:
    vector = _coerce_forecast_values(values)
    if len(vector) < horizon:
        raise AdapterInputError(
            f"{label} output is shorter than requested horizon ({len(vector)} < {horizon})",
        )
    return vector[:horizon]


def build_quantile_payload(
    *,
    forecast: Any,
    requested_quantiles: Sequence[float],
    horizon: int,
) -> dict[str, list[float]]:
    payload: dict[str, list[float]] = {}
    quantile_fn = getattr(forecast, "quantile", None)
    if not callable(quantile_fn):
        raise AdapterInputError("quantiles requested but Lag-Llama output has no quantile method")

    for quantile in requested_quantiles:
        q = float(quantile)
        try:
            values = quantile_fn(q)
        except Exception as exc:  # noqa: BLE001
            raise AdapterInputError(f"requested quantile {format(q, 'g')} is unavailable") from exc
        payload[format(q, "g")] = normalize_forecast_vector(
            values,
            horizon=horizon,
            label=f"quantile {format(q, 'g')}",
        )
    return payload


def resolve_forecast_start_timestamp(
    *,
    forecast: Any,
    series: SeriesInput,
    pandas_module: Any,
    horizon: int,
) -> str:
    start_date = getattr(forecast, "start_date", None)
    if isinstance(start_date, datetime):
        return _to_iso_timestamp(start_date)
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
    parsed = pandas_module.to_datetime([last_timestamp], utc=True, errors="raise")
    start = parsed[0]
    try:
        generated = pandas_module.date_range(start=start, periods=horizon + 1, freq=freq)
    except ValueError as exc:
        raise AdapterInputError(f"invalid frequency {freq!r} for lag-llama forecast") from exc
    return [_to_iso_timestamp(value) for value in list(generated[1:])]


def build_covariate_warnings(series_list: Sequence[SeriesInput]) -> list[str]:
    for series in series_list:
        if series.past_covariates or series.future_covariates or series.static_covariates:
            return [
                "Lag-Llama runner ignores covariates and static features; using target-only history",
            ]
    return []


def _resolve_runtime_config(
    *,
    model_name: str,
    model_source: dict[str, Any] | None,
    model_metadata: dict[str, Any] | None,
) -> _RuntimeConfig:
    defaults = _LAG_LLAMA_MODELS.get(model_name, {})
    repo_id = _dict_str(model_source, "repo_id") or _string_or_none(defaults.get("repo_id"))
    revision = _dict_str(model_source, "revision") or _string_or_none(defaults.get("revision"))
    implementation = _dict_str(model_metadata, "implementation") or _string_or_none(
        defaults.get("implementation"),
    )
    default_context_length = _dict_positive_int(
        model_metadata,
        "default_context_length",
    ) or _int_or_none(defaults.get("default_context_length"))
    default_num_samples = _dict_positive_int(
        model_metadata,
        "default_num_samples",
    ) or _int_or_none(defaults.get("default_num_samples"))
    default_batch_size = _dict_positive_int(
        model_metadata,
        "default_batch_size",
    ) or _int_or_none(defaults.get("default_batch_size"))
    checkpoint_filename = _dict_str(
        model_metadata,
        "checkpoint_filename",
    ) or _string_or_none(defaults.get("checkpoint_filename"))

    if repo_id is None:
        raise UnsupportedModelError(f"unsupported lag-llama model {model_name!r}; missing repo_id")
    if revision is None:
        revision = "main"
    if implementation is None:
        implementation = "lag_llama"
    if default_context_length is None:
        default_context_length = _DEFAULT_CONTEXT_LENGTH
    if default_num_samples is None:
        default_num_samples = _DEFAULT_NUM_SAMPLES
    if default_batch_size is None:
        default_batch_size = _DEFAULT_BATCH_SIZE
    if checkpoint_filename is None:
        checkpoint_filename = _DEFAULT_CHECKPOINT_FILENAME

    return _RuntimeConfig(
        model_name=model_name,
        repo_id=repo_id,
        revision=revision,
        implementation=implementation,
        default_context_length=default_context_length,
        default_num_samples=default_num_samples,
        default_batch_size=default_batch_size,
        checkpoint_filename=checkpoint_filename,
    )


def _existing_local_checkpoint_path(
    *,
    model_local_dir: str | None,
    checkpoint_filename: str,
) -> str | None:
    if not model_local_dir:
        return None
    base = Path(model_local_dir.strip())
    if not base.exists():
        return None
    if base.is_file():
        return str(base)

    candidate = base / checkpoint_filename
    if candidate.is_file():
        return str(candidate)
    for glob in ("*.ckpt", "*.pt", "*.pth"):
        matches = sorted(base.glob(glob))
        if matches:
            return str(matches[0])
    return None


def _coerce_forecast_values(values: Any) -> list[float]:
    if hasattr(values, "tolist"):
        values = values.tolist()
    if isinstance(values, tuple):
        values = list(values)
    if not isinstance(values, list):
        raise AdapterInputError("forecast output vector is not list-like")

    normalized: list[float] = []
    for value in values:
        if isinstance(value, bool):
            normalized.append(float(int(value)))
        elif isinstance(value, (int, float)):
            normalized.append(float(value))
        elif hasattr(value, "item") and isinstance(value.item(), (int, float, bool)):
            scalar = value.item()
            normalized.append(float(int(scalar)) if isinstance(scalar, bool) else float(scalar))
        else:
            raise AdapterInputError(f"forecast output contains non-numeric value: {value!r}")
    return normalized


def _to_iso_timestamp(value: Any) -> str:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=UTC)
        return value.astimezone(UTC).isoformat().replace("+00:00", "Z")
    return str(value)


def _unique_column_name(series_id: str, index: int, existing: dict[str, Any]) -> str:
    if series_id not in existing:
        return series_id
    return f"{series_id}__{index}"


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
    if not isinstance(value, int) or value <= 0:
        return None
    return value
