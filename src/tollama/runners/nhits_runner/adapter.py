"""N-HiTS forecasting adapter used by the nhits runner."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tollama.core.schemas import ForecastRequest, ForecastResponse, SeriesForecast, SeriesInput

from .errors import (
    AdapterInputError,
    AdapterRuntimeError,
    DependencyMissingError,
    UnsupportedModelError,
)

_NHITS_MODELS: dict[str, dict[str, Any]] = {
    "nhits": {
        "repo_id": "cchallu/nhits-air-passengers",
        "revision": "main",
        "implementation": "nhits",
    },
}


@dataclass(frozen=True)
class _Dependencies:
    pd: Any
    neuralforecast_cls: Any
    nhits_cls: Any


@dataclass(frozen=True)
class _RuntimeConfig:
    model_name: str
    repo_id: str
    revision: str
    implementation: str


class NhitsAdapter:
    """Adapter mapping canonical forecast payloads to NeuralForecast N-HiTS."""

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
        model = _build_model(request=request, runtime=runtime, nhits_cls=deps.nhits_cls)
        predictor = deps.neuralforecast_cls(models=[model], freq=resolved_freq)

        try:
            predictor.fit(train_df)
            prediction = predictor.predict(h=request.horizon)
        except Exception as exc:  # noqa: BLE001
            raise AdapterRuntimeError(f"N-HiTS runtime forecast failed: {exc}") from exc

        parsed = _prediction_to_series_map(prediction, request_quantiles=request.quantiles)
        forecasts: list[SeriesForecast] = []
        for series in request.series:
            output = parsed.get(series.id)
            if output is None:
                raise AdapterInputError(
                    f"N-HiTS output missing forecast for series {series.id!r}",
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
                    mean=output["mean"],
                    quantiles=output["quantiles"] or None,
                ),
            )

        warnings = _build_limitations_warnings(
            request=request,
            model_local_dir=model_local_dir,
            quantile_support=parsed.quantile_support,
        )

        return ForecastResponse(
            model=request.model,
            forecasts=forecasts,
            usage={
                "runner": "tollama-nhits",
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
        nhits_cls = None
        try:
            from neuralforecast import NeuralForecast
            from neuralforecast.models import NHITS

            neuralforecast_cls = NeuralForecast
            nhits_cls = NHITS
        except ModuleNotFoundError as exc:
            missing.append(exc.name or "neuralforecast")

        if missing or pd is None or neuralforecast_cls is None or nhits_cls is None:
            joined = ", ".join(sorted(set(missing or ["neuralforecast"])))
            raise DependencyMissingError(
                "missing optional nhits runner dependencies "
                f"({joined}); install them with `pip install -e \".[dev,runner_nhits]\"`",
            )

        deps = _Dependencies(
            pd=pd,
            neuralforecast_cls=neuralforecast_cls,
            nhits_cls=nhits_cls,
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
        raise AdapterInputError(f"invalid timestamps for nhits request: {exc}") from exc
    return frame


def _build_model(*, request: ForecastRequest, runtime: _RuntimeConfig, nhits_cls: Any) -> Any:
    input_size = max(2 * request.horizon, 4)
    try:
        return nhits_cls(h=request.horizon, input_size=input_size)
    except Exception as exc:  # noqa: BLE001
        raise AdapterInputError(
            f"failed to initialize N-HiTS model for {runtime.model_name!r}: {exc}",
        ) from exc


class _SeriesMap(dict[str, dict[str, Any]]):
    quantile_support: str


def _prediction_to_series_map(prediction: Any, *, request_quantiles: list[float]) -> _SeriesMap:
    if hasattr(prediction, "to_dict"):
        rows = prediction.to_dict(orient="records")
    elif isinstance(prediction, list):
        rows = prediction
    else:
        raise AdapterInputError("N-HiTS prediction output has unexpected shape")

    by_id: _SeriesMap = _SeriesMap()
    saw_quantile = False
    requested = [float(q) for q in request_quantiles]

    for row in rows:
        if not isinstance(row, dict):
            continue
        series_id = row.get("unique_id")
        if not isinstance(series_id, str) or not series_id:
            continue

        mean = _extract_prediction_value(row)
        if mean is None:
            continue

        series_out = by_id.setdefault(series_id, {"mean": [], "quantiles": {}})
        series_out["mean"].append(mean)

        if not requested:
            continue

        row_quantiles = _extract_row_quantiles(row=row, requested=requested)
        if row_quantiles:
            saw_quantile = True
        for key, value in row_quantiles.items():
            series_out["quantiles"].setdefault(key, []).append(value)

    quantile_support = "not_requested"
    if requested:
        quantile_support = "full" if saw_quantile and _all_quantiles_present(by_id, requested) else "none"
    by_id.quantile_support = quantile_support

    if quantile_support != "full":
        for output in by_id.values():
            output["quantiles"] = {}

    return by_id


def _all_quantiles_present(by_id: dict[str, dict[str, Any]], requested: list[float]) -> bool:
    expected_keys = {_format_quantile(q) for q in requested}
    for output in by_id.values():
        quantiles = output.get("quantiles") or {}
        mean = output.get("mean") or []
        if set(quantiles.keys()) != expected_keys:
            return False
        if any(len(values) != len(mean) for values in quantiles.values()):
            return False
    return bool(by_id)


def _extract_row_quantiles(*, row: dict[str, Any], requested: list[float]) -> dict[str, float]:
    parsed_columns: dict[float, float] = {}
    for key, raw in row.items():
        if key in {"unique_id", "ds"}:
            continue
        quantile = _parse_quantile_key(key)
        if quantile is None:
            continue
        value = _coerce_scalar_float(raw)
        if value is None:
            continue
        parsed_columns[quantile] = value

    extracted: dict[str, float] = {}
    for quantile in requested:
        matched = _lookup_quantile(parsed_columns, quantile)
        if matched is None:
            continue
        extracted[_format_quantile(quantile)] = matched
    return extracted


def _lookup_quantile(values: dict[float, float], quantile: float) -> float | None:
    for candidate, value in values.items():
        if abs(candidate - quantile) <= 1e-6:
            return value
    return None


def _parse_quantile_key(key: str) -> float | None:
    cleaned = key.strip().lower()

    try:
        numeric = float(cleaned)
        if 0.0 < numeric < 1.0:
            return numeric
    except ValueError:
        pass

    tokens = re.findall(r"\d*\.\d+|\d+", cleaned)
    for token in reversed(tokens):
        try:
            value = float(token)
        except ValueError:
            continue
        if 0.0 < value < 1.0:
            return value
        if 1.0 <= value <= 99.0 and ("q" in cleaned or "p" in cleaned or "lo" in cleaned or "hi" in cleaned):
            return value / 100.0
    return None


def _extract_prediction_value(row: dict[str, Any]) -> float | None:
    for key, value in row.items():
        if key in {"unique_id", "ds"}:
            continue
        if _parse_quantile_key(key) is not None:
            continue
        scalar = _coerce_scalar_float(value)
        if scalar is not None:
            return scalar
    return None


def _coerce_scalar_float(value: Any) -> float | None:
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


def _build_limitations_warnings(
    *,
    request: ForecastRequest,
    model_local_dir: str | None,
    quantile_support: str,
) -> list[str]:
    warnings: list[str] = []

    if request.quantiles and quantile_support != "full":
        warnings.append(
            "N-HiTS quantile extraction is backend-dependent; requested quantiles were unavailable, "
            "so mean-only forecasts were returned",
        )

    if any(
        series.past_covariates or series.future_covariates or series.static_covariates
        for series in request.series
    ):
        warnings.append(
            "N-HiTS currently supports target-only training/inference in this runner path; "
            "covariates and static features were ignored",
        )

    if model_local_dir:
        path = Path(model_local_dir.strip())
        if path.exists():
            warnings.append(
                "N-HiTS runner currently trains from request history at runtime; "
                "model_local_dir is ignored",
            )

    return warnings


def _future_start_timestamp(*, series: SeriesInput, horizon: int, pd_module: Any, resolved_freq: str) -> str:
    try:
        parsed = pd_module.to_datetime([series.timestamps[-1]], utc=True, errors="raise")
        generated = pd_module.date_range(start=parsed[0], periods=horizon + 1, freq=resolved_freq)
    except Exception as exc:  # noqa: BLE001
        raise AdapterInputError(f"invalid frequency {series.freq!r} for nhits forecast") from exc
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
                "N-HiTS currently requires one shared frequency across all series in a request",
            )
    if resolved is None:
        raise AdapterInputError("nhits forecast requires at least one input series")
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
        raise AdapterInputError(f"invalid frequency {series.freq!r} for nhits forecast") from exc
    return freq


def _coerce_finite_float(value: Any, *, field: str) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise AdapterInputError(f"{field} must be numeric") from exc
    if not math.isfinite(numeric):
        raise AdapterInputError(f"{field} must be finite")
    return numeric


def _format_quantile(value: float) -> str:
    return f"{value:.6f}".rstrip("0").rstrip(".")


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
    defaults = _NHITS_MODELS.get(model_name, {})
    repo_id = _dict_str(model_source, "repo_id") or _string_or_none(defaults.get("repo_id"))
    revision = _dict_str(model_source, "revision") or _string_or_none(defaults.get("revision")) or "main"
    implementation = (
        _dict_str(model_metadata, "implementation")
        or _string_or_none(defaults.get("implementation"))
        or "nhits"
    )

    if repo_id is None:
        raise UnsupportedModelError(f"unsupported nhits model {model_name!r}; missing repo_id")

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
