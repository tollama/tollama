"""N-BEATSx forecasting adapter used by the nbeatsx runner."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import NormalDist
from typing import Any

from tollama.core.schemas import ForecastRequest, ForecastResponse, SeriesForecast, SeriesInput

from .errors import AdapterInputError, DependencyMissingError, UnsupportedModelError

_NBEATSX_MODELS: dict[str, dict[str, Any]] = {
    "nbeatsx": {
        "repo_id": "tollama/nbeatsx-runner",
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


@dataclass(frozen=True)
class _CovariatePayload:
    train_df: Any
    futr_df: Any | None
    static_df: Any | None
    hist_exog: list[str]
    futr_exog: list[str]
    stat_exog: list[str]
    warnings: list[str]


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
        covariates_mode = request.parameters.covariates_mode
        covariate_payload = _build_covariate_frames(
            request=request,
            pd_module=deps.pd,
            resolved_freq=resolved_freq,
            covariates_mode=covariates_mode,
        )
        model = _build_model(
            request=request,
            runtime=runtime,
            nbeatsx_cls=deps.nbeatsx_cls,
            hist_exog=covariate_payload.hist_exog,
            futr_exog=covariate_payload.futr_exog,
            stat_exog=covariate_payload.stat_exog,
        )
        predictor = deps.neuralforecast_cls(models=[model], freq=resolved_freq)

        try:
            _fit_predictor(predictor=predictor, train_df=covariate_payload.train_df, static_df=covariate_payload.static_df)
            prediction = _predict(
                predictor=predictor,
                horizon=request.horizon,
                futr_df=covariate_payload.futr_df,
            )
        except Exception as exc:  # noqa: BLE001
            raise AdapterInputError(f"N-BEATSx forecast failed: {exc}") from exc

        mean_by_id, quantiles_by_id = _prediction_to_outputs(prediction, requested_quantiles=list(request.quantiles))

        forecasts: list[SeriesForecast] = []
        quantile_fallback = False
        for series in request.series:
            mean = mean_by_id.get(series.id)
            if mean is None:
                raise AdapterInputError(
                    f"N-BEATSx output missing forecast for series {series.id!r}",
                )

            quantiles = quantiles_by_id.get(series.id)
            if request.quantiles and not quantiles:
                quantiles = _calibrated_quantile_fallback(series=series, mean=mean, requested_quantiles=list(request.quantiles))
                quantile_fallback = True

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
                    quantiles=quantiles,
                ),
            )

        warnings = _build_limitations_warnings(
            request=request,
            model_local_dir=model_local_dir,
            quantile_fallback=quantile_fallback,
            covariate_warnings=covariate_payload.warnings,
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
                "covariates_mode": covariates_mode,
                "covariates_hist_exog": len(covariate_payload.hist_exog),
                "covariates_futr_exog": len(covariate_payload.futr_exog),
                "covariates_stat_exog": len(covariate_payload.stat_exog),
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


def _build_covariate_frames(
    *,
    request: ForecastRequest,
    pd_module: Any,
    resolved_freq: str,
    covariates_mode: str,
) -> _CovariatePayload:
    strict = covariates_mode == "strict"
    warnings: list[str] = []
    warning_seen: set[str] = set()

    hist_exog_candidates: set[str] = set()
    futr_exog_candidates: set[str] = set()
    stat_exog_candidates: set[str] = set()

    for series in request.series:
        hist_exog_candidates.update((series.past_covariates or {}).keys())
        futr_exog_candidates.update((series.future_covariates or {}).keys())
        stat_exog_candidates.update((series.static_covariates or {}).keys())

    hist_exog: list[str] = []
    futr_exog: list[str] = []
    stat_exog: list[str] = []

    rows: list[dict[str, Any]] = []
    future_rows: list[dict[str, Any]] = []
    static_rows: list[dict[str, Any]] = []

    for series in request.series:
        history_length = len(series.target)
        past_covariates = series.past_covariates or {}
        future_covariates = series.future_covariates or {}
        static_covariates = series.static_covariates or {}

        for index, (timestamp, value) in enumerate(zip(series.timestamps, series.target, strict=True)):
            row: dict[str, Any] = {
                "unique_id": series.id,
                "ds": timestamp,
                "y": _coerce_finite_float(value, field=f"series {series.id!r} target[{index}]"),
            }
            for name in sorted(hist_exog_candidates):
                series_values = past_covariates.get(name)
                if series_values is None:
                    row[name] = 0.0
                    _append_warning(
                        warnings=warnings,
                        seen=warning_seen,
                        message=(
                            f"N-BEATSx missing past covariate {name!r} for series {series.id!r}; "
                            "using zero-fill"
                        ),
                    )
                    continue
                try:
                    row[name] = _coerce_numeric_covariate(series_values[index], series_id=series.id, name=name)
                except AdapterInputError:
                    if strict:
                        raise
                    row[name] = 0.0
                    _append_warning(
                        warnings=warnings,
                        seen=warning_seen,
                        message=(
                            f"N-BEATSx dropped non-numeric past covariate {name!r} for series {series.id!r}; "
                            "using zero-fill in best_effort mode"
                        ),
                    )

            for name in sorted(futr_exog_candidates):
                series_values = past_covariates.get(name)
                source = "past"
                value_at_step: Any
                if series_values is not None:
                    value_at_step = series_values[index]
                else:
                    value_at_step = 0.0
                    source = "missing"
                    _append_warning(
                        warnings=warnings,
                        seen=warning_seen,
                        message=(
                            f"N-BEATSx missing historical values for future covariate {name!r} "
                            f"in series {series.id!r}; using zero-fill"
                        ),
                    )
                try:
                    row[name] = _coerce_numeric_covariate(value_at_step, series_id=series.id, name=name)
                except AdapterInputError:
                    if strict:
                        raise
                    row[name] = 0.0
                    _append_warning(
                        warnings=warnings,
                        seen=warning_seen,
                        message=(
                            f"N-BEATSx dropped non-numeric future covariate history {name!r} "
                            f"for series {series.id!r}; using zero-fill in best_effort mode"
                        ),
                    )
                if source == "past":
                    futr_exog.append(name)

            rows.append(row)

        future_timestamps = _future_timestamps(
            series=series,
            horizon=request.horizon,
            pd_module=pd_module,
            resolved_freq=resolved_freq,
        )
        for step, future_timestamp in enumerate(future_timestamps):
            futr_row: dict[str, Any] = {"unique_id": series.id, "ds": future_timestamp}
            for name in sorted(futr_exog_candidates):
                values = future_covariates.get(name)
                if values is None:
                    last = past_covariates.get(name, [0.0])[-1] if history_length else 0.0
                    value = last
                    _append_warning(
                        warnings=warnings,
                        seen=warning_seen,
                        message=(
                            f"N-BEATSx missing known-future covariate {name!r} for series {series.id!r}; "
                            "using last observed value"
                        ),
                    )
                else:
                    value = values[step]
                try:
                    futr_row[name] = _coerce_numeric_covariate(value, series_id=series.id, name=name)
                except AdapterInputError:
                    if strict:
                        raise
                    futr_row[name] = 0.0
                    _append_warning(
                        warnings=warnings,
                        seen=warning_seen,
                        message=(
                            f"N-BEATSx dropped non-numeric known-future covariate {name!r} "
                            f"for series {series.id!r}; using zero-fill in best_effort mode"
                        ),
                    )
            future_rows.append(futr_row)

        static_row: dict[str, Any] = {"unique_id": series.id}
        has_static = False
        for name in sorted(stat_exog_candidates):
            raw = static_covariates.get(name)
            if raw is None:
                continue
            has_static = True
            try:
                static_row[name] = _coerce_numeric_covariate(raw, series_id=series.id, name=name)
                stat_exog.append(name)
            except AdapterInputError:
                if strict:
                    raise
                has_static = False
                _append_warning(
                    warnings=warnings,
                    seen=warning_seen,
                    message=(
                        f"N-BEATSx dropped non-numeric static covariate {name!r} for series {series.id!r} "
                        "in best_effort mode"
                    ),
                )
        if has_static:
            static_rows.append(static_row)

    train_df = pd_module.DataFrame(rows)
    try:
        train_df["ds"] = pd_module.to_datetime(train_df["ds"], utc=True, errors="raise")
    except Exception as exc:  # noqa: BLE001
        raise AdapterInputError(f"invalid timestamps for nbeatsx request: {exc}") from exc

    futr_df = pd_module.DataFrame(future_rows) if future_rows and futr_exog_candidates else None
    if futr_df is not None:
        try:
            futr_df["ds"] = pd_module.to_datetime(futr_df["ds"], utc=True, errors="raise")
        except Exception as exc:  # noqa: BLE001
            raise AdapterInputError(f"invalid future timestamps for nbeatsx request: {exc}") from exc

    static_df = pd_module.DataFrame(static_rows) if static_rows else None

    hist_exog = sorted(set(hist_exog_candidates))
    futr_exog = sorted(set(futr_exog))
    stat_exog = sorted(set(stat_exog))

    return _CovariatePayload(
        train_df=train_df,
        futr_df=futr_df,
        static_df=static_df,
        hist_exog=hist_exog,
        futr_exog=futr_exog,
        stat_exog=stat_exog,
        warnings=warnings,
    )


def _build_model(
    *,
    request: ForecastRequest,
    runtime: _RuntimeConfig,
    nbeatsx_cls: Any,
    hist_exog: list[str],
    futr_exog: list[str],
    stat_exog: list[str],
) -> Any:
    input_size = max(2 * request.horizon, 4)
    kwargs: dict[str, Any] = {"h": request.horizon, "input_size": input_size}
    if hist_exog:
        kwargs["hist_exog_list"] = list(hist_exog)
    if futr_exog:
        kwargs["futr_exog_list"] = list(futr_exog)
    if stat_exog:
        kwargs["stat_exog_list"] = list(stat_exog)

    try:
        return nbeatsx_cls(**kwargs)
    except TypeError:
        fallback_kwargs = {"h": request.horizon, "input_size": input_size}
        return nbeatsx_cls(**fallback_kwargs)
    except Exception as exc:  # noqa: BLE001
        raise AdapterInputError(
            f"failed to initialize N-BEATSx model for {runtime.model_name!r}: {exc}",
        ) from exc


def _fit_predictor(*, predictor: Any, train_df: Any, static_df: Any | None) -> None:
    if static_df is None:
        predictor.fit(train_df)
        return

    try:
        predictor.fit(train_df, static_df=static_df)
    except TypeError:
        predictor.fit(train_df)


def _predict(*, predictor: Any, horizon: int, futr_df: Any | None) -> Any:
    if futr_df is None:
        return predictor.predict(h=horizon)

    try:
        return predictor.predict(futr_df=futr_df)
    except TypeError:
        try:
            return predictor.predict(h=horizon, futr_df=futr_df)
        except TypeError:
            return predictor.predict(h=horizon)


def _prediction_to_outputs(
    prediction: Any,
    *,
    requested_quantiles: list[float],
) -> tuple[dict[str, list[float]], dict[str, dict[str, list[float]]]]:
    rows = _prediction_rows(prediction)

    mean_by_id: dict[str, list[float]] = {}
    quantiles_by_id: dict[str, dict[str, list[float]]] = {}

    quantile_column_map: dict[str, float] = {}
    if rows:
        quantile_column_map = _resolve_quantile_columns(rows[0], requested_quantiles=requested_quantiles)

    for row in rows:
        if not isinstance(row, dict):
            continue
        series_id = row.get("unique_id")
        if not isinstance(series_id, str) or not series_id:
            continue

        value = _extract_prediction_value(row)
        if value is not None:
            mean_by_id.setdefault(series_id, []).append(value)

        for column, quantile in quantile_column_map.items():
            if column not in row:
                continue
            raw = row[column]
            try:
                numeric = _coerce_finite_float(raw, field=f"prediction column {column!r}")
            except AdapterInputError:
                continue
            q_key = format(float(quantile), "g")
            quantiles_by_id.setdefault(series_id, {}).setdefault(q_key, []).append(numeric)

    return mean_by_id, quantiles_by_id


def _prediction_rows(prediction: Any) -> list[dict[str, Any]]:
    if hasattr(prediction, "to_dict"):
        rows = prediction.to_dict(orient="records")
    elif isinstance(prediction, list):
        rows = prediction
    else:
        raise AdapterInputError("N-BEATSx prediction output has unexpected shape")
    return [row for row in rows if isinstance(row, dict)]


def _resolve_quantile_columns(row: dict[str, Any], *, requested_quantiles: list[float]) -> dict[str, float]:
    if not requested_quantiles:
        return {}

    extracted: list[tuple[str, float]] = []
    for key in row:
        quantile = _extract_quantile_from_column(key)
        if quantile is None:
            continue
        if 0.0 < quantile < 1.0:
            extracted.append((key, quantile))

    if not extracted:
        return {}

    resolved: dict[str, float] = {}
    for requested in requested_quantiles:
        nearest = min(extracted, key=lambda item: abs(item[1] - float(requested)))
        resolved[nearest[0]] = float(requested)
    return resolved


def _extract_quantile_from_column(name: str) -> float | None:
    lowered = name.strip().lower()
    if not lowered or lowered in {"unique_id", "ds"}:
        return None

    direct = re.search(r"(?:^|[^0-9])q(0?\.\d+|1\.0|0|1)(?:$|[^0-9])", lowered)
    if direct:
        try:
            return float(direct.group(1))
        except ValueError:
            return None

    generic = re.search(r"(0?\.\d+)", lowered)
    if generic:
        try:
            return float(generic.group(1))
        except ValueError:
            return None

    percent = re.search(r"(\d{1,2})(?:th)?(?:p|pct|percent|\b)", lowered)
    if percent:
        try:
            value = float(percent.group(1)) / 100.0
            if 0.0 < value < 1.0:
                return value
        except ValueError:
            return None

    return None


def _extract_prediction_value(row: dict[str, Any]) -> float | None:
    for key, value in row.items():
        if key in {"unique_id", "ds"}:
            continue
        if _extract_quantile_from_column(key) is not None:
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


def _calibrated_quantile_fallback(
    *,
    series: SeriesInput,
    mean: list[float],
    requested_quantiles: list[float],
) -> dict[str, list[float]]:
    scale = _series_scale(series.target)
    normal = NormalDist(mu=0.0, sigma=1.0)
    payload: dict[str, list[float]] = {}

    for quantile in requested_quantiles:
        z = normal.inv_cdf(float(quantile))
        values: list[float] = []
        for index, center in enumerate(mean, start=1):
            adjusted_scale = scale * math.sqrt(float(index))
            values.append(float(center) + z * adjusted_scale)
        payload[format(float(quantile), "g")] = values

    return payload


def _series_scale(values: list[int | float]) -> float:
    if not values:
        return 1e-6
    numeric = [float(value) for value in values]
    if len(numeric) < 2:
        return max(abs(numeric[0]) * 0.01, 1e-6)

    diffs = [numeric[index] - numeric[index - 1] for index in range(1, len(numeric))]
    median = sorted(diffs)[len(diffs) // 2]
    abs_dev = sorted(abs(value - median) for value in diffs)
    mad = abs_dev[len(abs_dev) // 2]
    robust_sigma = 1.4826 * mad
    if robust_sigma > 1e-9:
        return robust_sigma

    spread = max(numeric) - min(numeric)
    if spread > 0.0:
        return spread / 6.0

    return max(abs(numeric[-1]) * 0.01, 1e-6)


def _build_limitations_warnings(
    *,
    request: ForecastRequest,
    model_local_dir: str | None,
    quantile_fallback: bool,
    covariate_warnings: list[str],
) -> list[str]:
    warnings: list[str] = []

    if request.quantiles and quantile_fallback:
        warnings.append(
            "N-BEATSx used calibrated quantile fallback around mean forecasts because backend quantile outputs were unavailable",
        )

    warnings.extend(covariate_warnings)

    if model_local_dir:
        path = Path(model_local_dir.strip())
        if path.exists():
            warnings.append(
                "N-BEATSx baseline trains from request history at runtime; "
                "model_local_dir is currently ignored",
            )

    return warnings


def _future_timestamps(*, series: SeriesInput, horizon: int, pd_module: Any, resolved_freq: str) -> list[str]:
    try:
        parsed = pd_module.to_datetime([series.timestamps[-1]], utc=True, errors="raise")
        generated = pd_module.date_range(start=parsed[0], periods=horizon + 1, freq=resolved_freq)
    except Exception as exc:  # noqa: BLE001
        raise AdapterInputError(f"invalid frequency {series.freq!r} for nbeatsx forecast") from exc
    return [_to_iso_timestamp(value) for value in generated[1:]]


def _future_start_timestamp(*, series: SeriesInput, horizon: int, pd_module: Any, resolved_freq: str) -> str:
    return _future_timestamps(series=series, horizon=horizon, pd_module=pd_module, resolved_freq=resolved_freq)[0]


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


def _coerce_numeric_covariate(value: Any, *, series_id: str, name: str) -> float:
    if isinstance(value, bool):
        return float(int(value))
    if isinstance(value, (int, float)):
        numeric = float(value)
        if math.isfinite(numeric):
            return numeric
    raise AdapterInputError(
        f"covariate {name!r} for series {series_id!r} must be numeric for N-BEATSx",
    )


def _append_warning(*, warnings: list[str], seen: set[str], message: str) -> None:
    if message in seen:
        return
    seen.add(message)
    warnings.append(message)


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
