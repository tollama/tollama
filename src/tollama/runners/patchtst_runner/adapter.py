"""PatchTST forecasting adapter used by the patchtst runner."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tollama.core.schemas import ForecastRequest, ForecastResponse, SeriesForecast, SeriesInput

from .errors import AdapterInputError, DependencyMissingError, UnsupportedModelError

_DEFAULT_CONTEXT_LENGTH = 512
_PATCHTST_MODELS: dict[str, dict[str, Any]] = {
    "patchtst": {
        "repo_id": "ibm-granite/granite-timeseries-patchtst",
        "revision": "main",
        "implementation": "patchtst",
        "default_context_length": _DEFAULT_CONTEXT_LENGTH,
    },
}


@dataclass(frozen=True)
class _Dependencies:
    torch: Any
    pandas: Any
    model_loader_cls: Any


@dataclass(frozen=True)
class _RuntimeConfig:
    model_name: str
    repo_id: str
    revision: str
    implementation: str
    default_context_length: int


class PatchTSTAdapter:
    """Adapter that maps canonical request/response to PatchTST predictor calls."""

    def __init__(self) -> None:
        self._dependencies: _Dependencies | None = None
        self._model_cache: dict[tuple[str, str, str], Any] = {}

    def unload(self, model_name: str | None = None) -> None:
        if model_name is None:
            self._model_cache.clear()
            return
        self._model_cache = {
            key: value for key, value in self._model_cache.items() if key[0] != model_name
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
        dependencies = self._resolve_dependencies()
        model = self._get_or_load_model(
            runtime=runtime,
            model_local_dir=model_local_dir,
        )

        context_length = resolve_positive_int(
            option_value=request.options.get("context_length"),
            default_value=runtime.default_context_length,
            field_name="context_length",
        )

        forecasts: list[SeriesForecast] = []
        warnings = build_covariate_warnings(request.series)
        quantile_warning = False
        for series in request.series:
            mean, quantiles = _forecast_one_series(
                series=series,
                horizon=request.horizon,
                requested_quantiles=list(request.quantiles),
                context_length=context_length,
                model=model,
                torch_module=dependencies.torch,
                pandas_module=dependencies.pandas,
            )
            if request.quantiles and quantiles is None:
                quantile_warning = True

            forecasts.append(
                SeriesForecast(
                    id=series.id,
                    freq=series.freq,
                    start_timestamp=_future_start_timestamp(
                        series=series,
                        horizon=request.horizon,
                        pandas_module=dependencies.pandas,
                    ),
                    mean=mean,
                    quantiles=quantiles,
                ),
            )

        if quantile_warning:
            warnings.append(
                "PatchTST backend did not expose quantile outputs for this runtime; "
                "returning mean forecasts only",
            )

        return ForecastResponse(
            model=request.model,
            forecasts=forecasts,
            usage={
                "runner": "tollama-patchtst",
                "implementation": runtime.implementation,
                "series_count": len(forecasts),
                "horizon": request.horizon,
                "context_length": context_length,
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

        kwargs = {"revision": runtime.revision}
        try:
            model = deps.model_loader_cls.from_pretrained(model_ref, **kwargs)
        except TypeError:
            model = deps.model_loader_cls.from_pretrained(model_ref)
        model.eval()
        self._model_cache[key] = model
        return model

    def _resolve_dependencies(self) -> _Dependencies:
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
            import pandas as pd
        except ModuleNotFoundError as exc:
            missing_packages.append(exc.name or "pandas")
            pd = None

        model_loader_cls = None
        try:
            import transformers

            model_loader_cls = getattr(transformers, "PatchTSTForPrediction", None)
            if model_loader_cls is None:
                model_loader_cls = getattr(transformers, "AutoModelForTimeSeriesPrediction", None)
            if model_loader_cls is None:
                model_loader_cls = getattr(transformers, "AutoModel", None)
        except ModuleNotFoundError as exc:
            missing_packages.append(exc.name or "transformers")

        if model_loader_cls is None:
            missing_packages.append("transformers")

        if missing_packages:
            joined = ", ".join(sorted(set(missing_packages)))
            raise DependencyMissingError(
                "missing optional patchtst runner dependencies "
                f"({joined}); install them with `pip install -e \".[dev,runner_patchtst]\"`",
            )

        assert torch is not None
        assert pd is not None
        assert model_loader_cls is not None
        resolved = _Dependencies(
            torch=torch,
            pandas=pd,
            model_loader_cls=model_loader_cls,
        )
        self._dependencies = resolved
        return resolved


def _forecast_one_series(
    *,
    series: SeriesInput,
    horizon: int,
    requested_quantiles: list[float],
    context_length: int,
    model: Any,
    torch_module: Any,
    pandas_module: Any,
) -> tuple[list[float], dict[str, list[float]] | None]:
    if len(series.timestamps) != len(series.target):
        raise AdapterInputError(f"series {series.id!r} timestamps and target lengths must match")
    if len(series.target) < 2:
        raise AdapterInputError("each input series must include at least two target points")

    _validate_frequency(series.freq, pandas_module)

    history = [float(value) for value in series.target]
    window = history[-context_length:] if context_length < len(history) else history
    tensor = torch_module.tensor([window], dtype=torch_module.float32)

    outputs = _invoke_forecast(
        model=model,
        tensor=tensor,
        horizon=horizon,
    )
    mean = _extract_mean(outputs, horizon=horizon)
    quantiles = _extract_quantiles(outputs, requested_quantiles=requested_quantiles, horizon=horizon)
    return mean, quantiles


def _invoke_forecast(*, model: Any, tensor: Any, horizon: int) -> Any:
    call_variants = (
        {"past_values": tensor, "prediction_length": horizon},
        {"past_values": tensor},
        {"past_target": tensor, "prediction_length": horizon},
        {"past_target": tensor},
        {"inputs": tensor},
    )

    generate = getattr(model, "generate", None)
    if callable(generate):
        for kwargs in call_variants:
            try:
                return generate(**kwargs)
            except TypeError:
                continue
            except Exception as exc:  # noqa: BLE001
                raise AdapterInputError(f"PatchTST generate failed: {exc}") from exc

    for kwargs in call_variants:
        try:
            return model(**kwargs)
        except TypeError:
            continue
        except Exception as exc:  # noqa: BLE001
            raise AdapterInputError(f"PatchTST forward pass failed: {exc}") from exc

    raise AdapterInputError("PatchTST model signature is incompatible with runner adapter")


def _extract_mean(outputs: Any, *, horizon: int) -> list[float]:
    candidates = (
        "mean",
        "prediction_outputs",
        "predictions",
        "sequences",
        "forecast",
        "logits",
    )

    for key in candidates:
        value = _extract_attr_or_key(outputs, key)
        vector = _flatten_forecast_vector(value)
        if vector:
            return _cut_to_horizon(vector, horizon=horizon, label=key)

    vector = _flatten_forecast_vector(outputs)
    if vector:
        return _cut_to_horizon(vector, horizon=horizon, label="output")

    raise AdapterInputError("PatchTST output does not contain a forecast vector")


def _extract_quantiles(
    outputs: Any,
    *,
    requested_quantiles: list[float],
    horizon: int,
) -> dict[str, list[float]] | None:
    if not requested_quantiles:
        return None

    quantiles_output = _extract_attr_or_key(outputs, "quantiles")
    if quantiles_output is None:
        quantile_fn = getattr(outputs, "quantile", None)
        if callable(quantile_fn):
            payload: dict[str, list[float]] = {}
            for q in requested_quantiles:
                value = quantile_fn(float(q))
                vector = _flatten_forecast_vector(value)
                if not vector:
                    return None
                payload[format(float(q), "g")] = _cut_to_horizon(
                    vector,
                    horizon=horizon,
                    label=f"quantile {format(float(q), 'g')}",
                )
            return payload
        return None

    if isinstance(quantiles_output, dict):
        payload: dict[str, list[float]] = {}
        for q in requested_quantiles:
            key = format(float(q), "g")
            source = quantiles_output.get(key)
            if source is None:
                source = quantiles_output.get(float(q))
            if source is None:
                return None
            vector = _flatten_forecast_vector(source)
            if not vector:
                return None
            payload[key] = _cut_to_horizon(vector, horizon=horizon, label=f"quantile {key}")
        return payload

    return None


def _flatten_forecast_vector(value: Any) -> list[float]:
    if value is None:
        return []
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    if hasattr(value, "tolist"):
        value = value.tolist()

    while isinstance(value, list) and value and isinstance(value[0], list):
        value = value[-1]

    if isinstance(value, tuple):
        value = list(value)
    if not isinstance(value, list):
        return []

    normalized: list[float] = []
    for item in value:
        if isinstance(item, bool):
            normalized.append(float(int(item)))
        elif isinstance(item, (int, float)):
            normalized.append(float(item))
        elif hasattr(item, "item"):
            scalar = item.item()
            if isinstance(scalar, bool):
                normalized.append(float(int(scalar)))
            elif isinstance(scalar, (int, float)):
                normalized.append(float(scalar))
            else:
                return []
        else:
            return []
    return normalized


def _cut_to_horizon(values: list[float], *, horizon: int, label: str) -> list[float]:
    if len(values) < horizon:
        raise AdapterInputError(f"{label} output is shorter than requested horizon ({len(values)} < {horizon})")
    return values[:horizon]


def _extract_attr_or_key(payload: Any, key: str) -> Any | None:
    if payload is None:
        return None
    if isinstance(payload, dict):
        return payload.get(key)
    return getattr(payload, key, None)


def _future_start_timestamp(*, series: SeriesInput, horizon: int, pandas_module: Any) -> str:
    _validate_frequency(series.freq, pandas_module)
    parsed = pandas_module.to_datetime([series.timestamps[-1]], utc=True, errors="raise")
    generated = pandas_module.date_range(start=parsed[0], periods=horizon + 1, freq=series.freq)
    return _to_iso_timestamp(generated[1])


def _validate_frequency(freq: str, pandas_module: Any) -> None:
    try:
        pandas_module.date_range(start="2025-01-01", periods=2, freq=freq)
    except ValueError as exc:
        raise AdapterInputError(f"invalid frequency {freq!r} for patchtst forecast") from exc


def build_covariate_warnings(series_list: list[SeriesInput]) -> list[str]:
    for series in series_list:
        if series.past_covariates or series.future_covariates or series.static_covariates:
            return [
                "PatchTST runner currently ignores covariates and static features; using target-only history",
            ]
    return []


def _resolve_runtime_config(
    *,
    model_name: str,
    model_source: dict[str, Any] | None,
    model_metadata: dict[str, Any] | None,
) -> _RuntimeConfig:
    defaults = _PATCHTST_MODELS.get(model_name, {})
    repo_id = _dict_str(model_source, "repo_id") or _string_or_none(defaults.get("repo_id"))
    revision = _dict_str(model_source, "revision") or _string_or_none(defaults.get("revision"))
    implementation = _dict_str(model_metadata, "implementation") or _string_or_none(
        defaults.get("implementation"),
    )
    default_context_length = _dict_positive_int(
        model_metadata,
        "default_context_length",
    ) or _int_or_none(defaults.get("default_context_length"))

    if repo_id is None:
        raise UnsupportedModelError(f"unsupported patchtst model {model_name!r}; missing repo_id")
    if revision is None:
        revision = "main"
    if implementation is None:
        implementation = "patchtst"
    if default_context_length is None:
        default_context_length = _DEFAULT_CONTEXT_LENGTH

    return _RuntimeConfig(
        model_name=model_name,
        repo_id=repo_id,
        revision=revision,
        implementation=implementation,
        default_context_length=default_context_length,
    )


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


def _existing_local_model_path(model_local_dir: str | None) -> str | None:
    if not model_local_dir:
        return None
    path = Path(model_local_dir.strip())
    if not path.exists():
        return None
    return str(path)


def _to_iso_timestamp(value: Any) -> str:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=UTC)
        return value.astimezone(UTC).isoformat().replace("+00:00", "Z")
    isoformat = getattr(value, "isoformat", None)
    if callable(isoformat):
        return str(isoformat()).replace("+00:00", "Z")
    return str(value)


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
