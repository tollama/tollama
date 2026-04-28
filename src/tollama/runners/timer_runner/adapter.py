"""Timer model adapter for the tollama runner protocol.

Timer (THUML, ICML 2024) is a generative pre-trained Transformer for
time series forecasting with strong zero-shot transfer capabilities.
"""

from __future__ import annotations

import logging
from typing import Any

from tollama.core.schemas import (
    ForecastRequest,
    ForecastResponse,
    SeriesForecast,
)

from .errors import AdapterInputError, DependencyMissingError

logger = logging.getLogger(__name__)

_TIMER_MODELS: dict[str, dict[str, Any]] = {
    "timer-base": {
        "repo_id": "thuml/timer-base-84m",
        "revision": "main",
        "implementation": "timer_base",
        "max_context": 2880,
        "max_horizon": 720,
        "default_num_samples": 100,
    },
}


class TimerAdapter:
    """Inference adapter for Timer models."""

    def __init__(self) -> None:
        self._loaded_models: dict[str, Any] = {}

    def load(
        self,
        model_name: str,
        model_local_dir: str | None = None,
        model_source: dict[str, Any] | None = None,
        model_metadata: dict[str, Any] | None = None,
    ) -> None:
        """Pre-load a Timer model into memory."""
        config = _resolve_runtime_config(model_name, model_source, model_metadata)
        if model_name in self._loaded_models:
            return

        try:
            import torch  # noqa: F401
            from transformers import AutoModelForCausalLM  # noqa: F401
        except ImportError as exc:
            raise DependencyMissingError(
                "Timer runner requires torch and transformers. "
                "Install with: pip install -e '.[dev,runner_timer]'"
            ) from exc

        repo_id = model_local_dir or config["repo_id"]
        logger.info("loading Timer model %s from %s", model_name, repo_id)
        # Actual model loading deferred to forecast() for lazy initialization
        self._loaded_models[model_name] = {"config": config, "repo_id": repo_id}

    def unload(self, model_name: str | None = None) -> None:
        """Unload one or all models."""
        if model_name is not None:
            self._loaded_models.pop(model_name, None)
        else:
            self._loaded_models.clear()

    def forecast(
        self,
        request: ForecastRequest,
        model_local_dir: str | None = None,
        model_source: dict[str, Any] | None = None,
        model_metadata: dict[str, Any] | None = None,
    ) -> ForecastResponse:
        """Run Timer inference and return canonical forecast response."""
        model_name = request.model
        config = _resolve_runtime_config(model_name, model_source, model_metadata)

        try:
            import torch
            from transformers import AutoModelForCausalLM
        except ImportError as exc:
            raise DependencyMissingError(
                "Timer runner requires torch, numpy, and transformers"
            ) from exc
        _ensure_transformers_cache_compatibility()

        max_horizon = config.get("max_horizon", 720)
        if request.horizon > max_horizon:
            raise AdapterInputError(
                f"requested horizon {request.horizon} exceeds Timer max_horizon {max_horizon}"
            )

        max_context = config.get("max_context", 2880)
        repo_id = model_local_dir or config["repo_id"]
        revision = None if model_local_dir else config.get("revision")

        forecasts: list[SeriesForecast] = []
        warnings: list[str] = []

        model = self._get_or_load_model(
            model_name=model_name,
            repo_id=repo_id,
            revision=revision,
            config=config,
            model_cls=AutoModelForCausalLM,
        )
        input_token_len = _model_input_token_len(model)

        for series in request.series:
            target = [float(v) for v in series.target]
            if len(target) > max_context:
                target = target[-max_context:]
                warnings.append(f"series {series.id!r}: truncated to last {max_context} points")
            target = _trim_to_timer_token_boundary(
                target=target,
                input_token_len=input_token_len,
                series_id=series.id,
                warnings=warnings,
            )

            input_tensor = torch.tensor([target], dtype=torch.float32)

            try:
                with torch.no_grad():
                    predicted = _predict_timer_values(
                        model=model,
                        input_tensor=input_tensor,
                        context_length=len(target),
                        horizon=request.horizon,
                        torch_module=torch,
                        input_token_len=input_token_len,
                    )
            except Exception as exc:
                raise ValueError(f"Timer inference failed for series {series.id!r}: {exc}") from exc

            # Pad if output is shorter than requested horizon
            if len(predicted) < request.horizon:
                last_val = predicted[-1] if predicted else target[-1]
                predicted.extend([last_val] * (request.horizon - len(predicted)))

            mean_values = [round(float(v), 8) for v in predicted[: request.horizon]]

            forecasts.append(
                SeriesForecast(
                    id=series.id,
                    freq=series.freq,
                    start_timestamp=_forecast_start_timestamp(series),
                    mean=mean_values,
                    quantiles=None,
                )
            )

        return ForecastResponse(
            model=model_name,
            forecasts=forecasts,
            warnings=warnings or None,
        )

    def _get_or_load_model(
        self,
        *,
        model_name: str,
        repo_id: str,
        revision: str | None,
        config: dict[str, Any],
        model_cls: Any,
    ) -> Any:
        loaded = self._loaded_models.get(model_name, {})
        if "model" in loaded:
            return loaded["model"]

        model = model_cls.from_pretrained(
            repo_id,
            revision=revision,
            trust_remote_code=True,
        )
        model.eval()
        if model_name not in self._loaded_models:
            self._loaded_models[model_name] = {"config": config, "repo_id": repo_id}
        self._loaded_models[model_name]["model"] = model
        return model


def _resolve_runtime_config(
    model_name: str,
    model_source: dict[str, Any] | None = None,
    model_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Merge registry defaults with runtime overrides."""
    config = dict(_TIMER_MODELS.get(model_name, _TIMER_MODELS.get("timer-base", {})))
    if model_source:
        for key in ("repo_id", "revision"):
            if key in model_source:
                config[key] = model_source[key]
    if model_metadata:
        config.update(model_metadata)
    return config


def _model_input_token_len(model: Any) -> int:
    config = getattr(model, "config", None)
    raw = getattr(config, "input_token_len", 1)
    if isinstance(raw, bool):
        return 1
    if isinstance(raw, int) and raw > 0:
        return raw
    return 1


def _trim_to_timer_token_boundary(
    *,
    target: list[float],
    input_token_len: int,
    series_id: str,
    warnings: list[str],
) -> list[float]:
    if input_token_len <= 1:
        return target
    usable_length = (len(target) // input_token_len) * input_token_len
    if usable_length <= 0:
        raise AdapterInputError(
            f"series {series_id!r}: Timer requires at least {input_token_len} history points"
        )
    if usable_length != len(target):
        warnings.append(
            f"series {series_id!r}: truncated to last {usable_length} points "
            f"to match Timer input_token_len {input_token_len}"
        )
    return target[-usable_length:]


def _forecast_start_timestamp(series: Any) -> str:
    timestamps = getattr(series, "timestamps", None)
    if timestamps:
        return str(timestamps[-1])
    return "1970-01-01T00:00:00Z"


def _predict_timer_values(
    *,
    model: Any,
    input_tensor: Any,
    context_length: int,
    horizon: int,
    torch_module: Any,
    input_token_len: int,
) -> list[float]:
    if callable(model):
        try:
            output = model(input_tensor, max_output_length=horizon, revin=True)
        except TypeError:
            output = None
        else:
            return _extract_prediction_values(
                output,
                context_length=context_length,
                horizon=horizon,
            )

    attention_mask = torch_module.ones((1, context_length // max(input_token_len, 1)))
    output = model.generate(
        input_tensor,
        max_new_tokens=horizon,
        attention_mask=attention_mask,
    )
    return _extract_prediction_values(
        output,
        context_length=context_length,
        horizon=horizon,
    )


def _extract_prediction_values(
    output: Any,
    *,
    context_length: int,
    horizon: int,
) -> list[float]:
    values = _coerce_prediction_values(getattr(output, "logits", output))
    if len(values) >= context_length + horizon:
        return values[context_length : context_length + horizon]
    return values[:horizon]


def _coerce_prediction_values(payload: Any) -> list[float]:
    if hasattr(payload, "detach"):
        payload = payload.detach()
    if hasattr(payload, "cpu"):
        payload = payload.cpu()
    if hasattr(payload, "tolist"):
        payload = payload.tolist()
    if isinstance(payload, list):
        return _flatten_numeric_values(payload)
    return list(payload)


def _flatten_numeric_values(values: list[Any]) -> list[float]:
    flattened = values
    while flattened and isinstance(flattened[0], list):
        flattened = flattened[0]
    return [float(value) for value in flattened]


def _ensure_transformers_cache_compatibility() -> None:
    """Restore legacy cache APIs expected by Timer's remote model code."""
    try:
        from transformers.cache_utils import DynamicCache
    except ImportError:
        return

    if not hasattr(DynamicCache, "seen_tokens"):
        DynamicCache.seen_tokens = property(_dynamic_cache_seen_tokens)  # type: ignore[attr-defined]
    if not hasattr(DynamicCache, "from_legacy_cache"):
        DynamicCache.from_legacy_cache = classmethod(_dynamic_cache_from_legacy_cache)  # type: ignore[attr-defined]
    if not hasattr(DynamicCache, "to_legacy_cache"):
        DynamicCache.to_legacy_cache = _dynamic_cache_to_legacy_cache  # type: ignore[attr-defined]
    if not hasattr(DynamicCache, "get_max_length"):
        DynamicCache.get_max_length = _dynamic_cache_get_max_length  # type: ignore[attr-defined]
    if not hasattr(DynamicCache, "get_usable_length"):
        DynamicCache.get_usable_length = _dynamic_cache_get_usable_length  # type: ignore[attr-defined]


def _dynamic_cache_seen_tokens(cache: Any) -> int:
    get_seq_length = getattr(cache, "get_seq_length", None)
    if not callable(get_seq_length):
        return 0
    return int(get_seq_length())


def _dynamic_cache_from_legacy_cache(cls: Any, past_key_values: Any = None) -> Any:
    if past_key_values is None:
        return cls()
    return cls(past_key_values)


def _dynamic_cache_to_legacy_cache(cache: Any) -> tuple[Any, ...]:
    legacy_cache: list[tuple[Any, Any]] = []
    for layer in getattr(cache, "layers", ()):
        key_states = getattr(layer, "keys", None)
        value_states = getattr(layer, "values", None)
        if key_states is not None and value_states is not None:
            legacy_cache.append((key_states, value_states))
    return tuple(legacy_cache)


def _dynamic_cache_get_max_length(cache: Any) -> int | None:
    get_max_cache_shape = getattr(cache, "get_max_cache_shape", None)
    if not callable(get_max_cache_shape):
        return None
    max_cache_shape = get_max_cache_shape()
    if max_cache_shape is None or int(max_cache_shape) < 0:
        return None
    return int(max_cache_shape)


def _dynamic_cache_get_usable_length(
    cache: Any,
    new_seq_length: int,
    layer_idx: int = 0,
) -> int:
    get_seq_length = getattr(cache, "get_seq_length", None)
    if not callable(get_seq_length):
        return 0

    previous_seq_length = int(get_seq_length(layer_idx))
    get_max_length = getattr(cache, "get_max_length", None)
    max_length = get_max_length() if callable(get_max_length) else None
    if max_length is not None and previous_seq_length + new_seq_length > int(max_length):
        return int(max_length) - new_seq_length
    return previous_seq_length
