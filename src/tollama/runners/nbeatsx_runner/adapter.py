"""N-BEATSx forecasting adapter used by the nbeatsx runner."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tollama.core.schemas import ForecastRequest, ForecastResponse, SeriesForecast, SeriesInput

from .errors import AdapterInputError, DependencyMissingError, UnsupportedModelError

_NBEATSX_MODELS: dict[str, dict[str, Any]] = {
    "nbeatsx": {
        "repo_id": "cchallu/nbeatsx-air-passengers",
        "revision": "main",
        "implementation": "nbeatsx",
    },
}


@dataclass(frozen=True)
class _Dependencies:
    pd: Any
    neuralforecast_cls: Any
    nbeatsx_cls: Any


@dataclass(frozen=True)
class _RuntimeConfig:
    model_name: str
    repo_id: str
    revision: str
    implementation: str


class NbeatsxAdapter:
    """Adapter mapping canonical forecast payloads to NeuralForecast N-BEATSx."""

    def __init__(self) -> None:
        self._dependencies: _Dependencies | None = None

    def unload(self, model_name: str | None = None) -> None:
        del model_name

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

        resolved_freq = _resolve_shared_frequency(request=request, pd_module=deps.pd)
        train_df = _request_to_training_frame(request=request, pd_module=deps.pd)
        model = _build_model(request=request, runtime=runtime, nbeatsx_cls=deps.nbeatsx_cls)
        predictor = deps.neuralforecast_cls(models=[model], freq=resolved_freq)

        try:
            predictor.fit(train_df)
            prediction = predictor.predict(h=request.horizon)
        except Exception as exc:  # noqa: BLE001
            raise AdapterInputError(f"N-BEATSx forecast failed: {exc}") from exc

        by_id = _prediction_to_id_map(prediction)
        forecasts: list[SeriesForecast] = []
        for series in request.series:
            mean = by_id.get(series.id)
            if mean is None:
                raise AdapterInputError(
                    f"N-BEATSx output missing forecast for series {series.id!r}",
                )
            forecasts.append(
                SeriesForecast(
                    id=series.id,
                    freq=resolved_freq,
                    start_timestamp=_future_start_timestamp(
                        series=series,
                        horizon=request.horizon,
                        pd_module=deps.pd,
                        resolved_freq=resolved_freq,
                    ),
                    mean=mean,
                    quantiles=None,
                ),
            )

        warnings = _build_limitations_warnings(
            request=request,
            model_local_dir=model_local_dir,
        )

        return ForecastResponse(
            model=request.model,
            forecasts=forecasts,
            usage={
                "runner": "tollama-nbeatsx",
                "implementation": runtime.implementation,
                "series_count": len(forecasts),
                "horizon": request.horizon,
                "repo_id": runtime.repo_id,
                "revision": runtime.revision,
            },
            warnings=warnings or None,
        )

    def _resolve_dependencies(self) -> _Dependencies:
        if self._dependencies is not None:
            return self._dependencies

        missing: list[str] = []
        try:
            import pandas as pd
        except ModuleNotFoundError as exc:
            missing.append(exc.name or "pandas")
            pd = None

        neuralforecast_cls = None
        nbeatsx_cls = None
        try:
            from neuralforecast import NeuralForecast
            from neuralforecast.models import NBEATSx

            neuralforecast_cls = NeuralForecast
            nbeatsx_cls = NBEATSx
        except ModuleNotFoundError as exc:
            missing.append(exc.name or "neuralforecast")

        if missing or pd is None or neuralforecast_cls is None or nbeatsx_cls is None:
            joined = ", ".join(sorted(set(missing or ["neuralforecast"])))
            raise DependencyMissingError(
                "missing optional nbeatsx runner dependencies "
                f"({joined}); install them with `pip install -e \".[dev,runner_nbeatsx]\"`",
            )

        deps = _Dependencies(
            pd=pd,
            neuralforecast_cls=neuralforecast_cls,
            nbeatsx_cls=nbeatsx_cls,
        )
        self._dependencies = deps
        return deps


def _request_to_training_frame(*, request: ForecastRequest, pd_module: Any) -> Any:
    rows: list[dict[str, Any]] = []
    for series in request.series:
        if len(series.timestamps) != len(series.target):
            raise AdapterInputError(f"series {series.id!r} timestamps and target lengths must match")
        if len(series.target) < 2:
            raise AdapterInputError("each input series must include at least two target points")

        for index, (timestamp, value) in enumerate(zip(series.timestamps, series.target, strict=True)):
            rows.append(
                {
                    "unique_id": series.id,
                    "ds": timestamp,
                    "y": _coerce_finite_float(value, field=f"series {series.id!r} target[{index}]"),
                },
            )

    frame = pd_module.DataFrame(rows)
    try:
        frame["ds"] = pd_module.to_datetime(frame["ds"], utc=True, errors="raise")
    except Exception as exc:  # noqa: BLE001
        raise AdapterInputError(f"invalid timestamps for nbeatsx request: {exc}") from exc
    return frame


def _build_model(*, request: ForecastRequest, runtime: _RuntimeConfig, nbeatsx_cls: Any) -> Any:
    input_size = max(2 * request.horizon, 4)
    try:
        return nbeatsx_cls(h=request.horizon, input_size=input_size)
    except Exception as exc:  # noqa: BLE001
        raise AdapterInputError(
            f"failed to initialize N-BEATSx model for {runtime.model_name!r}: {exc}",
        ) from exc


def _prediction_to_id_map(prediction: Any) -> dict[str, list[float]]:
    if hasattr(prediction, "to_dict"):
        rows = prediction.to_dict(orient="records")
    elif isinstance(prediction, list):
        rows = prediction
    else:
        raise AdapterInputError("N-BEATSx prediction output has unexpected shape")

    by_id: dict[str, list[float]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        series_id = row.get("unique_id")
        if not isinstance(series_id, str) or not series_id:
            continue
        value = _extract_prediction_value(row)
        if value is None:
            continue
        by_id.setdefault(series_id, []).append(value)

    return by_id


def _extract_prediction_value(row: dict[str, Any]) -> float | None:
    for key, value in row.items():
        if key in {"unique_id", "ds"}:
            continue
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
    return None


def _build_limitations_warnings(*, request: ForecastRequest, model_local_dir: str | None) -> list[str]:
    warnings: list[str] = []

    if request.quantiles:
        warnings.append(
            "N-BEATSx baseline currently returns point forecasts only; requested quantiles were omitted",
        )

    if any(
        series.past_covariates or series.future_covariates or series.static_covariates
        for series in request.series
    ):
        warnings.append(
            "N-BEATSx baseline currently ignores covariates and static features; "
            "using target-only history",
        )

    if model_local_dir:
        path = Path(model_local_dir.strip())
        if path.exists():
            warnings.append(
                "N-BEATSx baseline trains from request history at runtime; "
                "model_local_dir is currently ignored",
            )

    return warnings


def _future_start_timestamp(*, series: SeriesInput, horizon: int, pd_module: Any, resolved_freq: str) -> str:
    try:
        parsed = pd_module.to_datetime([series.timestamps[-1]], utc=True, errors="raise")
        generated = pd_module.date_range(start=parsed[0], periods=horizon + 1, freq=resolved_freq)
    except Exception as exc:  # noqa: BLE001
        raise AdapterInputError(f"invalid frequency {series.freq!r} for nbeatsx forecast") from exc
    return _to_iso_timestamp(generated[1])


def _resolve_shared_frequency(*, request: ForecastRequest, pd_module: Any) -> str:
    resolved: str | None = None
    for series in request.series:
        freq = _normalize_frequency(series=series, pd_module=pd_module)
        if resolved is None:
            resolved = freq
            continue
        if freq != resolved:
            raise AdapterInputError(
                "N-BEATSx currently requires one shared frequency across all series in a request",
            )
    if resolved is None:
        raise AdapterInputError("nbeatsx forecast requires at least one input series")
    return resolved


def _normalize_frequency(*, series: SeriesInput, pd_module: Any) -> str:
    freq = (series.freq or "").strip()
    if freq.lower() == "auto":
        infer_freq = getattr(pd_module, "infer_freq", None)
        if not callable(infer_freq):
            raise AdapterInputError(
                f"unable to infer frequency for series {series.id!r}; provide explicit freq",
            )
        try:
            parsed = pd_module.to_datetime(series.timestamps, utc=True, errors="raise")
            inferred = infer_freq(parsed)
        except Exception as exc:  # noqa: BLE001
            raise AdapterInputError(f"invalid timestamps for series {series.id!r}: {exc}") from exc
        if not inferred:
            raise AdapterInputError(
                f"unable to infer frequency for series {series.id!r}; provide explicit freq",
            )
        freq = str(inferred)

    try:
        parsed = pd_module.to_datetime([series.timestamps[-1]], utc=True, errors="raise")
        pd_module.date_range(start=parsed[0], periods=2, freq=freq)
    except Exception as exc:  # noqa: BLE001
        raise AdapterInputError(f"invalid frequency {series.freq!r} for nbeatsx forecast") from exc
    return freq


def _coerce_finite_float(value: Any, *, field: str) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise AdapterInputError(f"{field} must be numeric") from exc
    if not math.isfinite(numeric):
        raise AdapterInputError(f"{field} must be finite")
    return numeric


def _to_iso_timestamp(value: Any) -> str:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=UTC)
        return value.astimezone(UTC).isoformat().replace("+00:00", "Z")
    isoformat = getattr(value, "isoformat", None)
    if callable(isoformat):
        return str(isoformat()).replace("+00:00", "Z")
    return str(value)


def _resolve_runtime_config(
    *,
    model_name: str,
    model_source: dict[str, Any] | None,
    model_metadata: dict[str, Any] | None,
) -> _RuntimeConfig:
    defaults = _NBEATSX_MODELS.get(model_name, {})
    repo_id = _dict_str(model_source, "repo_id") or _string_or_none(defaults.get("repo_id"))
    revision = _dict_str(model_source, "revision") or _string_or_none(defaults.get("revision")) or "main"
    implementation = (
        _dict_str(model_metadata, "implementation")
        or _string_or_none(defaults.get("implementation"))
        or "nbeatsx"
    )

    if repo_id is None:
        raise UnsupportedModelError(f"unsupported nbeatsx model {model_name!r}; missing repo_id")

    return _RuntimeConfig(
        model_name=model_name,
        repo_id=repo_id,
        revision=revision,
        implementation=implementation,
    )


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
