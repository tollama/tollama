"""ForecastPFN model adapter for the tollama runner protocol.

ForecastPFN is a zero-shot Bayesian forecaster that produces well-calibrated
probabilistic predictions without any training data. It uses a prior-fitted
network trained on synthetic time series, making it an instant probabilistic
baseline for any univariate forecasting task.
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

_FORECASTPFN_MODELS: dict[str, dict[str, Any]] = {
    "forecastpfn": {
        "repo_id": "abacusai/ForecastPFN",
        "revision": "main",
        "implementation": "forecastpfn",
        "max_context": 1000,
        "max_horizon": 300,
    },
}


class ForecastPFNAdapter:
    """Inference adapter for ForecastPFN models."""

    def __init__(self) -> None:
        self._loaded_models: dict[str, Any] = {}

    def load(
        self,
        model_name: str,
        model_local_dir: str | None = None,
        model_source: dict[str, Any] | None = None,
        model_metadata: dict[str, Any] | None = None,
    ) -> None:
        """Pre-load a ForecastPFN model into memory."""
        config = _resolve_runtime_config(model_name, model_source, model_metadata)
        if model_name in self._loaded_models:
            return

        try:
            import torch  # noqa: F401
        except ImportError as exc:
            raise DependencyMissingError(
                "ForecastPFN runner requires torch. "
                "Install with: pip install -e '.[dev,runner_forecastpfn]'"
            ) from exc

        repo_id = model_local_dir or config["repo_id"]
        logger.info("loading ForecastPFN model %s from %s", model_name, repo_id)
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
        """Run ForecastPFN inference and return canonical forecast response."""
        model_name = request.model
        config = _resolve_runtime_config(model_name, model_source, model_metadata)

        try:
            import numpy as np
            import torch
        except ImportError as exc:
            raise DependencyMissingError("ForecastPFN runner requires torch and numpy") from exc

        max_horizon = config.get("max_horizon", 300)
        if request.horizon > max_horizon:
            raise AdapterInputError(
                f"requested horizon {request.horizon} exceeds ForecastPFN max_horizon {max_horizon}"
            )

        max_context = config.get("max_context", 1000)
        repo_id = model_local_dir or config["repo_id"]

        forecasts: list[SeriesForecast] = []
        warnings: list[str] = []

        for series in request.series:
            target = [float(v) for v in series.target]
            if len(target) > max_context:
                target = target[-max_context:]
                warnings.append(f"series {series.id!r}: truncated to last {max_context} points")

            input_array = np.array(target, dtype=np.float32)

            try:
                loaded = self._loaded_models.get(model_name, {})
                if "model" not in loaded:
                    from ForecastPFN import ForecastPFN as ForecastPFNModel

                    model = ForecastPFNModel()
                    if model_name not in self._loaded_models:
                        self._loaded_models[model_name] = {"config": config, "repo_id": repo_id}
                    self._loaded_models[model_name]["model"] = model
                else:
                    model = self._loaded_models[model_name]["model"]

                with torch.no_grad():
                    output = model.predict(input_array, prediction_length=request.horizon)

                if hasattr(output, "mean"):
                    predicted = output.mean.tolist()
                elif isinstance(output, np.ndarray):
                    predicted = output[: request.horizon].tolist()
                elif isinstance(output, tuple):
                    predicted = output[0][: request.horizon].tolist()
                else:
                    predicted = list(output)[: request.horizon]
            except Exception as exc:
                raise ValueError(
                    f"ForecastPFN inference failed for series {series.id!r}: {exc}"
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
    config = dict(_FORECASTPFN_MODELS.get(model_name, _FORECASTPFN_MODELS.get("forecastpfn", {})))
    if model_source:
        for key in ("repo_id", "revision"):
            if key in model_source:
                config[key] = model_source[key]
    if model_metadata:
        config.update(model_metadata)
    return config
