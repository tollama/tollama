"""Sundial forecasting adapter used by the sundial runner."""

from __future__ import annotations

import math
from collections.abc import Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tollama.core.schemas import ForecastRequest, ForecastResponse, SeriesForecast, SeriesInput

from .errors import AdapterInputError, DependencyMissingError, UnsupportedModelError

_DEFAULT_NUM_SAMPLES = 20
_SUNDIAL_MODELS: dict[str, dict[str, Any]] = {
    "sundial-base-128m": {
        "repo_id": "thuml/sundial-base-128m",
        "revision": "main",
        "implementation": "sundial_base",
        "max_context": 2880,
        "max_horizon": 720,
        "default_num_samples": _DEFAULT_NUM_SAMPLES,
    },
}


@dataclass(frozen=True)
class _SundialDependencies:
    torch: Any
    numpy: Any
    pandas: Any
    transformers: Any


@dataclass(frozen=True)
class _RuntimeConfig:
    model_name: str
    repo_id: str
    revision: str
    implementation: str
    max_context: int
    max_horizon: int
    default_num_samples: int


@dataclass(frozen=True)
class _LoadedSundialModel:
    model: Any
    runtime: _RuntimeConfig
    model_ref: str


class SundialAdapter:
    """Adapter that maps canonical request/response to Sundial sample generation."""

    def __init__(self) -> None:
        self._dependencies: _SundialDependencies | None = None
        self._models: dict[str, _LoadedSundialModel] = {}

    def load(
        self,
        model_name: str,
        *,
        model_local_dir: str | None = None,
        model_source: dict[str, Any] | None = None,
        model_metadata: dict[str, Any] | None = None,
    ) -> None:
        """Load one Sundial model into cache."""
        runtime = _resolve_runtime_config(
            model_name=model_name,
            model_source=model_source,
            model_metadata=model_metadata,
        )
        self._get_or_load_model(runtime=runtime, model_local_dir=model_local_dir)

    def unload(self, model_name: str | None = None) -> None:
        """Unload one model or clear all cached Sundial entries."""
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
        """Generate probabilistic Sundial forecasts mapped to canonical response schema."""
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
        loaded_model = self._get_or_load_model(runtime=runtime, model_local_dir=model_local_dir)
        num_samples = resolve_num_samples(
            option_value=request.options.get("num_samples"),
            requested_quantiles=list(request.quantiles),
            default_value=runtime.default_num_samples,
        )

        forecasts: list[SeriesForecast] = []
        for series in request.series:
            truncated_target = truncate_target_to_max_context(
                target=series.target,
                max_context=runtime.max_context,
            )
            generated = generate_sundial_samples(
                model=loaded_model.model,
                target_values=truncated_target,
                horizon=request.horizon,
                num_samples=num_samples,
                torch_module=dependencies.torch,
            )
            sample_matrix = normalize_generated_samples(
                generated=generated,
                horizon=request.horizon,
            )
            mean, quantiles = compute_sample_statistics(
                samples=sample_matrix,
                requested_quantiles=list(request.quantiles),
                horizon=request.horizon,
                torch_module=dependencies.torch,
            )
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
                    mean=mean,
                    quantiles=quantiles,
                ),
            )

        warnings = build_covariate_warnings(request.series)
        return ForecastResponse(
            model=request.model,
            forecasts=forecasts,
            usage={
                "runner": "tollama-sundial",
                "implementation": runtime.implementation,
                "series_count": len(forecasts),
                "horizon": request.horizon,
                "num_samples": num_samples,
            },
            warnings=warnings or None,
        )

    def _get_or_load_model(
        self,
        *,
        runtime: _RuntimeConfig,
        model_local_dir: str | None,
    ) -> _LoadedSundialModel:
        dependencies = self._resolve_dependencies()
        local_model_path = _existing_local_model_path(model_local_dir)
        model_ref = local_model_path or runtime.repo_id

        cached = self._models.get(runtime.model_name)
        if (
            cached is not None
            and cached.model_ref == model_ref
            and cached.runtime.revision == runtime.revision
        ):
            return cached

        auto_model_cls = getattr(dependencies.transformers, "AutoModelForCausalLM", None)
        if auto_model_cls is None:
            raise DependencyMissingError(
                "transformers package does not expose AutoModelForCausalLM; "
                "install with `pip install -e \".[dev,runner_sundial]\"`",
            )
        from_pretrained = getattr(auto_model_cls, "from_pretrained", None)
        if not callable(from_pretrained):
            raise DependencyMissingError(
                "transformers AutoModelForCausalLM.from_pretrained is unavailable; "
                "install with `pip install -e \".[dev,runner_sundial]\"`",
            )

        if local_model_path is not None:
            model = from_pretrained(local_model_path, trust_remote_code=True)
        else:
            try:
                model = from_pretrained(
                    runtime.repo_id,
                    revision=runtime.revision,
                    trust_remote_code=True,
                )
            except TypeError:
                model = from_pretrained(runtime.repo_id, trust_remote_code=True)

        eval_method = getattr(model, "eval", None)
        if callable(eval_method):
            eval_method()

        loaded = _LoadedSundialModel(
            model=model,
            runtime=runtime,
            model_ref=model_ref,
        )
        self._models[runtime.model_name] = loaded
        return loaded

    def _resolve_dependencies(self) -> _SundialDependencies:
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
            import transformers
        except ModuleNotFoundError as exc:
            missing_packages.append(exc.name or "transformers")
            transformers = None

        if missing_packages:
            joined = ", ".join(sorted(set(missing_packages)))
            raise DependencyMissingError(
                "missing optional sundial runner dependencies "
                f"({joined}); install them with `pip install -e \".[dev,runner_sundial]\"`",
            )

        assert torch is not None
        assert np is not None
        assert pd is not None
        assert transformers is not None
        resolved = _SundialDependencies(
            torch=torch,
            numpy=np,
            pandas=pd,
            transformers=transformers,
        )
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
        values.append(_to_float(value))
    return values


def resolve_num_samples(
    *,
    option_value: Any,
    requested_quantiles: Sequence[float],
    default_value: int,
) -> int:
    """Resolve sample count from options/defaults and requested quantiles."""
    if option_value is not None:
        if isinstance(option_value, bool) or not isinstance(option_value, int):
            raise AdapterInputError("num_samples option must be an integer")
        if option_value <= 0:
            raise AdapterInputError("num_samples option must be greater than zero")
        return option_value

    if default_value <= 0:
        default_value = _DEFAULT_NUM_SAMPLES
    if not requested_quantiles:
        return default_value

    min_tail_probability = min(
        min(float(value), 1.0 - float(value)) for value in requested_quantiles
    )
    if min_tail_probability <= 0.0:
        return default_value
    recommended = int(math.ceil(1.0 / min_tail_probability))
    return max(default_value, recommended)


def generate_sundial_samples(
    *,
    model: Any,
    target_values: Sequence[float],
    horizon: int,
    num_samples: int,
    torch_module: Any,
) -> Any:
    """Call Sundial generation with a single-series input tensor."""
    inputs = _torch_tensor(torch_module=torch_module, values=[list(target_values)])
    inference_mode = getattr(torch_module, "inference_mode", None)
    context_manager = inference_mode() if callable(inference_mode) else nullcontext()
    with context_manager:
        try:
            return model.generate(inputs, max_new_tokens=horizon, num_samples=num_samples)
        except TypeError:
            try:
                return model.generate(
                    inputs=inputs,
                    max_new_tokens=horizon,
                    num_samples=num_samples,
                )
            except TypeError as exc:
                raise AdapterInputError(
                    "Sundial model.generate does not support max_new_tokens and num_samples",
                ) from exc


def normalize_generated_samples(*, generated: Any, horizon: int) -> list[list[float]]:
    """Normalize generated outputs to [num_samples, horizon] float rows."""
    if horizon <= 0:
        raise AdapterInputError("horizon must be greater than zero")

    payload = _to_nested_list(generated)
    if not isinstance(payload, list) or not payload:
        raise AdapterInputError("Sundial generate output is empty")

    rows: list[Any]
    if isinstance(payload[0], list) and payload[0] and isinstance(payload[0][0], list):
        if len(payload) != 1:
            raise AdapterInputError(
                "Sundial generate output has unexpected batch dimension for one series",
            )
        rows = payload[0]
    elif isinstance(payload[0], list):
        rows = payload
    else:
        raise AdapterInputError("Sundial generate output is not list-like")

    normalized: list[list[float]] = []
    for row in rows:
        if not isinstance(row, list):
            raise AdapterInputError("Sundial sample row is not list-like")
        if len(row) < horizon:
            raise AdapterInputError("Sundial sample row is shorter than requested horizon")
        normalized.append([_to_float(item) for item in row[:horizon]])
    if not normalized:
        raise AdapterInputError("Sundial generate output has no samples")
    return normalized


def compute_sample_statistics(
    *,
    samples: Sequence[Sequence[float]],
    requested_quantiles: Sequence[float],
    horizon: int,
    torch_module: Any,
) -> tuple[list[float], dict[str, list[float]] | None]:
    """Compute mean and optional quantiles over sample trajectories."""
    if not samples:
        raise AdapterInputError("Sundial sample matrix must include at least one sample")
    if horizon <= 0:
        raise AdapterInputError("horizon must be greater than zero")

    normalized_samples: list[list[float]] = []
    for row in samples:
        values = [_to_float(item) for item in list(row)]
        if len(values) < horizon:
            raise AdapterInputError("Sundial sample row is shorter than requested horizon")
        normalized_samples.append(values[:horizon])

    sample_tensor = _torch_tensor(torch_module=torch_module, values=normalized_samples)
    mean_values = _to_float_vector(sample_tensor.mean(dim=0), horizon=horizon, label="mean")
    if not requested_quantiles:
        return mean_values, None

    quantile_values = [float(value) for value in requested_quantiles]
    quantile_tensor = _torch_quantile(
        torch_module=torch_module,
        sample_tensor=sample_tensor,
        quantiles=quantile_values,
    )
    quantile_rows = _to_nested_list(quantile_tensor)
    if isinstance(quantile_rows, list) and quantile_rows and not isinstance(quantile_rows[0], list):
        quantile_rows = [quantile_rows]
    if not isinstance(quantile_rows, list) or len(quantile_rows) != len(quantile_values):
        raise AdapterInputError("Sundial quantile output shape does not match requested quantiles")

    payload: dict[str, list[float]] = {}
    for index, quantile in enumerate(quantile_values):
        payload[format(quantile, "g")] = _to_float_vector(
            quantile_rows[index],
            horizon=horizon,
            label=f"quantile {format(quantile, 'g')}",
        )
    return mean_values, payload


def build_covariate_warnings(series_list: Sequence[SeriesInput]) -> list[str]:
    """Return one warning when covariates are supplied for Sundial requests."""
    for series in series_list:
        if series.past_covariates or series.future_covariates or series.static_covariates:
            return [
                "Sundial runner ignores covariates and static features; using target-only history",
            ]
    return []


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
        raise AdapterInputError(f"invalid frequency {freq!r} for Sundial forecast") from exc
    return [_to_iso_timestamp(value) for value in list(generated[1:])]


def _resolve_runtime_config(
    *,
    model_name: str,
    model_source: dict[str, Any] | None,
    model_metadata: dict[str, Any] | None,
) -> _RuntimeConfig:
    defaults = _SUNDIAL_MODELS.get(model_name, {})
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
    default_num_samples = _dict_positive_int(model_metadata, "default_num_samples") or _int_or_none(
        defaults.get("default_num_samples"),
    )

    if repo_id is None:
        raise UnsupportedModelError(
            f"unsupported sundial model {model_name!r}; missing repo_id metadata",
        )
    if revision is None:
        revision = "main"
    if implementation is None:
        implementation = "sundial_base"
    if max_context is None:
        max_context = 2880
    if max_horizon is None:
        max_horizon = 720
    if default_num_samples is None:
        default_num_samples = _DEFAULT_NUM_SAMPLES

    return _RuntimeConfig(
        model_name=model_name,
        repo_id=repo_id,
        revision=revision,
        implementation=implementation,
        max_context=max_context,
        max_horizon=max_horizon,
        default_num_samples=default_num_samples,
    )


def _torch_tensor(*, torch_module: Any, values: Any) -> Any:
    constructor = getattr(torch_module, "as_tensor", None)
    dtype = getattr(torch_module, "float32", None)
    if callable(constructor):
        if dtype is not None:
            return constructor(values, dtype=dtype)
        return constructor(values)
    constructor = getattr(torch_module, "tensor", None)
    if callable(constructor):
        if dtype is not None:
            return constructor(values, dtype=dtype)
        return constructor(values)
    raise AdapterInputError("torch dependency does not expose tensor construction")


def _torch_quantile(*, torch_module: Any, sample_tensor: Any, quantiles: Sequence[float]) -> Any:
    quantile_fn = getattr(torch_module, "quantile", None)
    if not callable(quantile_fn):
        raise AdapterInputError("torch dependency does not expose quantile")
    quantile_tensor = _torch_tensor(torch_module=torch_module, values=list(quantiles))
    try:
        return quantile_fn(sample_tensor, quantile_tensor, dim=0)
    except TypeError:
        return quantile_fn(sample_tensor, list(quantiles), dim=0)


def _to_float_vector(value: Any, *, horizon: int, label: str) -> list[float]:
    payload = _to_nested_list(value)
    if not isinstance(payload, list):
        raise AdapterInputError(f"Sundial {label} output is not list-like")
    if len(payload) < horizon:
        raise AdapterInputError(f"Sundial {label} output is shorter than requested horizon")
    return [_to_float(item) for item in payload[:horizon]]


def _to_nested_list(value: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
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
        if isinstance(scalar, bool):
            return float(int(scalar))
        if isinstance(scalar, (int, float)):
            return float(scalar)
    raise AdapterInputError(f"value is not numeric: {value!r}")


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
