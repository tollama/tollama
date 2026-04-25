"""TimeMixer model adapter for the tollama runner protocol.

TimeMixer uses channel-mixing architecture and excels at multivariate
forecasting tasks with dedicated mixing of temporal and channel dimensions.
"""

from __future__ import annotations

import logging
from typing import Any

from tollama.core.schemas import (
    ForecastRequest,
    ForecastResponse,
    SeriesForecast,
)

from .errors import AdapterInputError, DependencyMissingError, UnsupportedModelError

logger = logging.getLogger(__name__)

_TIMEMIXER_MODELS: dict[str, dict[str, Any]] = {
    "timemixer-base": {
        "repo_id": "tollama/timemixer-runner",
        "revision": "main",
        "implementation": "timemixer_base",
        "max_context": 1536,
        "max_horizon": 720,
    },
}


class TimeMixerAdapter:
    """Inference adapter for TimeMixer models."""

    def __init__(self) -> None:
        self._loaded_models: dict[str, Any] = {}

    def load(
        self,
        model_name: str,
        model_local_dir: str | None = None,
        model_source: dict[str, Any] | None = None,
        model_metadata: dict[str, Any] | None = None,
    ) -> None:
        """Pre-load a TimeMixer model into memory."""
        config = _resolve_runtime_config(model_name, model_source, model_metadata)
        if model_name in self._loaded_models:
            return

        try:
            import torch  # noqa: F401
        except ImportError as exc:
            raise DependencyMissingError(
                "TimeMixer runner requires torch. "
                "Install with: pip install -e '.[dev,runner_timemixer]'"
            ) from exc

        repo_id = model_local_dir or config["repo_id"]
        logger.info("loading TimeMixer model %s from %s", model_name, repo_id)
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
        """Run TimeMixer inference and return canonical forecast response."""
        model_name = request.model
        config = _resolve_runtime_config(model_name, model_source, model_metadata)
        if _is_manifest_only_source(model_source):
            raise UnsupportedModelError(
                "TimeMixer is registered as manifest-only because no dedicated "
                "TimeMixer Hugging Face model snapshot is published for this runner. "
                "Use timer-base or sundial-base-128m for downloadable THUML models."
            )

        try:
            import torch
        except ImportError as exc:
            raise DependencyMissingError("TimeMixer runner requires torch and numpy") from exc

        max_horizon = config.get("max_horizon", 720)
        if request.horizon > max_horizon:
            raise AdapterInputError(
                f"requested horizon {request.horizon} exceeds TimeMixer max_horizon {max_horizon}"
            )

        max_context = config.get("max_context", 1536)
        repo_id = model_local_dir or config["repo_id"]
        revision = None if model_local_dir else config.get("revision")

        forecasts: list[SeriesForecast] = []
        warnings: list[str] = []

        for series in request.series:
            target = [float(v) for v in series.target]
            if len(target) > max_context:
                target = target[-max_context:]
                warnings.append(f"series {series.id!r}: truncated to last {max_context} points")

            input_tensor = torch.tensor([target], dtype=torch.float32)

            try:
                loaded = self._loaded_models.get(model_name, {})
                if "model" not in loaded:
                    from transformers import AutoModel

                    model = AutoModel.from_pretrained(
                        repo_id,
                        revision=revision,
                        trust_remote_code=True,
                    )
                    model.eval()
                    if model_name not in self._loaded_models:
                        self._loaded_models[model_name] = {"config": config, "repo_id": repo_id}
                    self._loaded_models[model_name]["model"] = model
                else:
                    model = self._loaded_models[model_name]["model"]

                with torch.no_grad():
                    output = model(input_tensor, prediction_length=request.horizon)

                if hasattr(output, "predictions"):
                    predicted = output.predictions[0].tolist()
                else:
                    predicted = output[0, : request.horizon].tolist()
            except Exception as exc:
                raise ValueError(
                    f"TimeMixer inference failed for series {series.id!r}: {exc}"
                ) from exc

            if len(predicted) < request.horizon:
                last_val = predicted[-1] if predicted else target[-1]
                predicted.extend([last_val] * (request.horizon - len(predicted)))

            mean_values = [round(float(v), 8) for v in predicted[: request.horizon]]

            forecasts.append(
                SeriesForecast(
                    id=series.id,
                    freq=series.freq,
                    start_timestamp=None,
                    mean=mean_values,
                    quantiles=None,
                )
            )

        return ForecastResponse(
            model=model_name,
            forecasts=forecasts,
            warnings=warnings or None,
        )


def _resolve_runtime_config(
    model_name: str,
    model_source: dict[str, Any] | None = None,
    model_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Merge registry defaults with runtime overrides."""
    config = dict(_TIMEMIXER_MODELS.get(model_name, _TIMEMIXER_MODELS.get("timemixer-base", {})))
    if model_source:
        for key in ("repo_id", "revision"):
            if key in model_source:
                config[key] = model_source[key]
    if model_metadata:
        config.update(model_metadata)
    return config


def _is_manifest_only_source(model_source: dict[str, Any] | None) -> bool:
    if not isinstance(model_source, dict):
        return False
    return model_source.get("type") == "local"
