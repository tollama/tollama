"""TiDE forecasting adapter used by the tide runner."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from tollama.core.schemas import ForecastRequest, ForecastResponse, SeriesForecast, SeriesInput

from .errors import AdapterInputError, DependencyMissingError, UnsupportedModelError

_DEFAULT_INPUT_CHUNK_LENGTH = 32
_DEFAULT_HIDDEN_SIZE = 64
_DEFAULT_NUM_ENCODER_LAYERS = 1
_DEFAULT_NUM_DECODER_LAYERS = 1
_DEFAULT_DROPOUT = 0.1
_DEFAULT_BATCH_SIZE = 32
_DEFAULT_N_EPOCHS = 20

_TIDE_MODELS: dict[str, dict[str, Any]] = {
    "tide": {
        "implementation": "tide",
        "default_input_chunk_length": _DEFAULT_INPUT_CHUNK_LENGTH,
        "default_hidden_size": _DEFAULT_HIDDEN_SIZE,
        "default_num_encoder_layers": _DEFAULT_NUM_ENCODER_LAYERS,
        "default_num_decoder_layers": _DEFAULT_NUM_DECODER_LAYERS,
        "default_dropout": _DEFAULT_DROPOUT,
        "default_batch_size": _DEFAULT_BATCH_SIZE,
        "default_n_epochs": _DEFAULT_N_EPOCHS,
    },
}


@dataclass(frozen=True)
class _Dependencies:
    pandas: Any
    timeseries_cls: Any
    tide_model_cls: Any


@dataclass(frozen=True)
class _RuntimeConfig:
    model_name: str
    implementation: str
    input_chunk_length: int
    hidden_size: int
    num_encoder_layers: int
    num_decoder_layers: int
    dropout: float
    batch_size: int
    n_epochs: int


class TiDEAdapter:
    """Adapter that maps canonical request/response to Darts TiDEModel calls."""

    def __init__(self) -> None:
        self._dependencies: _Dependencies | None = None

    def unload(self, model_name: str | None = None) -> None:
        del model_name
        # TiDE path trains per-request; no persistent predictor cache yet.

    def forecast(
        self,
        request: ForecastRequest,
        *,
        model_local_dir: str | None = None,
        model_source: dict[str, Any] | None = None,
        model_metadata: dict[str, Any] | None = None,
    ) -> ForecastResponse:
        del model_local_dir, model_source
        runtime = _resolve_runtime_config(model_name=request.model, model_metadata=model_metadata)
        deps = self._resolve_dependencies()

        forecasts: list[SeriesForecast] = []
        warnings = build_covariate_warnings(request.series, requested_quantiles=request.quantiles)

        for series in request.series:
            target_ts = _build_target_timeseries(series=series, timeseries_cls=deps.timeseries_cls)
            past_cov_ts = _build_covariate_timeseries(
                timestamps=series.timestamps,
                covariates=series.past_covariates,
                timeseries_cls=deps.timeseries_cls,
                kind="past_covariates",
                series_id=series.id,
            )
            future_cov_timestamps = _future_covariate_timestamps(
                last_timestamp=series.timestamps[-1],
                freq=series.freq,
                horizon=request.horizon,
                pandas_module=deps.pandas,
            )
            future_cov_ts = _build_covariate_timeseries(
                timestamps=future_cov_timestamps,
                covariates=series.future_covariates,
                timeseries_cls=deps.timeseries_cls,
                kind="future_covariates",
                series_id=series.id,
            )

            model = _build_model(runtime=runtime, horizon=request.horizon, tide_model_cls=deps.tide_model_cls)
            fit_kwargs: dict[str, Any] = {"series": target_ts}
            if past_cov_ts is not None:
                fit_kwargs["past_covariates"] = past_cov_ts
            model.fit(**fit_kwargs)

            predict_kwargs: dict[str, Any] = {"n": request.horizon, "series": target_ts}
            if past_cov_ts is not None:
                predict_kwargs["past_covariates"] = past_cov_ts
            if future_cov_ts is not None:
                predict_kwargs["future_covariates"] = future_cov_ts
            predicted = model.predict(**predict_kwargs)

            mean = _extract_values(predicted, horizon=request.horizon)
            forecasts.append(
                SeriesForecast(
                    id=series.id,
                    freq=series.freq,
                    start_timestamp=_future_start_timestamp(
                        last_timestamp=series.timestamps[-1],
                        freq=series.freq,
                        pandas_module=deps.pandas,
                    ),
                    mean=mean,
                    quantiles=None,
                ),
            )

        return ForecastResponse(
            model=request.model,
            forecasts=forecasts,
            usage={
                "runner": "tollama-tide",
                "implementation": runtime.implementation,
                "series_count": len(forecasts),
                "horizon": request.horizon,
                "input_chunk_length": runtime.input_chunk_length,
                "n_epochs": runtime.n_epochs,
            },
            warnings=warnings or None,
        )

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

        timeseries_cls = None
        tide_model_cls = None
        try:
            from darts import TimeSeries

            timeseries_cls = TimeSeries
            try:
                from darts.models import TiDEModel

                tide_model_cls = TiDEModel
            except ImportError:
                from darts.models.forecasting.tide_model import TiDEModel  # type: ignore[attr-defined]

                tide_model_cls = TiDEModel
        except ModuleNotFoundError as exc:
            missing_packages.append(exc.name or "darts")
        except ImportError:
            missing_packages.append("darts")

        if timeseries_cls is None or tide_model_cls is None:
            missing_packages.append("u8darts")

        if missing_packages:
            joined = ", ".join(sorted(set(missing_packages)))
            raise DependencyMissingError(
                "missing optional TiDE runner dependencies "
                f"({joined}); install them with `pip install -e \".[dev,runner_tide]\"`",
            )

        assert pd is not None
        resolved = _Dependencies(
            pandas=pd,
            timeseries_cls=timeseries_cls,
            tide_model_cls=tide_model_cls,
        )
        self._dependencies = resolved
        return resolved


def _build_model(*, runtime: _RuntimeConfig, horizon: int, tide_model_cls: Any) -> Any:
    output_chunk_length = min(horizon, runtime.input_chunk_length)
    try:
        return tide_model_cls(
            input_chunk_length=runtime.input_chunk_length,
            output_chunk_length=output_chunk_length,
            hidden_size=runtime.hidden_size,
            num_encoder_layers=runtime.num_encoder_layers,
            num_decoder_layers=runtime.num_decoder_layers,
            dropout=runtime.dropout,
            batch_size=runtime.batch_size,
            n_epochs=runtime.n_epochs,
            random_state=0,
        )
    except TypeError as exc:
        raise AdapterInputError("TiDEModel constructor signature is incompatible with tollama runner") from exc


def _build_target_timeseries(*, series: SeriesInput, timeseries_cls: Any) -> Any:
    if len(series.timestamps) != len(series.target):
        raise AdapterInputError(f"series {series.id!r} timestamps and target lengths must match")
    if len(series.target) < 3:
        raise AdapterInputError("each input series must include at least three target points for TiDE")

    values = [float(value) for value in series.target]
    try:
        target = timeseries_cls.from_times_and_values(series.timestamps, values)
    except Exception as exc:  # noqa: BLE001
        raise AdapterInputError(f"failed to construct TiDE target series for {series.id!r}: {exc}") from exc

    static = series.static_covariates
    if static:
        with_static_covariates = getattr(target, "with_static_covariates", None)
        if callable(with_static_covariates):
            try:
                target = with_static_covariates(static)
            except Exception as exc:  # noqa: BLE001
                raise AdapterInputError(
                    f"failed to attach static_covariates for series {series.id!r}: {exc}",
                ) from exc
    return target


def _build_covariate_timeseries(
    *,
    timestamps: list[str],
    covariates: dict[str, list[Any]] | None,
    timeseries_cls: Any,
    kind: str,
    series_id: str,
) -> Any | None:
    if not covariates:
        return None

    names = sorted(covariates)
    expected = len(timestamps)
    matrix: list[list[float]] = []
    for index in range(expected):
        row: list[float] = []
        for name in names:
            values = covariates[name]
            if len(values) != expected:
                raise AdapterInputError(
                    f"{kind}[{name!r}] length must match timestamps length for series {series_id!r}",
                )
            value = values[index]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise AdapterInputError(
                    f"{kind}[{name!r}] must be numeric for TiDE; found {type(value).__name__}",
                )
            row.append(float(value))
        matrix.append(row)

    try:
        return timeseries_cls.from_times_and_values(timestamps, matrix, columns=names)
    except Exception as exc:  # noqa: BLE001
        raise AdapterInputError(
            f"failed to construct {kind} TimeSeries for series {series_id!r}: {exc}",
        ) from exc


def _extract_values(predicted: Any, *, horizon: int) -> list[float]:
    values = None
    extractor = getattr(predicted, "values", None)
    if callable(extractor):
        values = extractor()
    if values is None:
        values = getattr(predicted, "values", None)
    if values is None:
        values = predicted

    if hasattr(values, "tolist"):
        values = values.tolist()

    if not isinstance(values, list):
        raise AdapterInputError("TiDE prediction output is not list-like")

    while values and isinstance(values[0], list):
        values = values[0]

    normalized: list[float] = []
    for item in values:
        if isinstance(item, bool):
            normalized.append(float(int(item)))
        elif isinstance(item, (int, float)):
            normalized.append(float(item))
        elif hasattr(item, "item") and isinstance(item.item(), (int, float, bool)):
            scalar = item.item()
            normalized.append(float(int(scalar)) if isinstance(scalar, bool) else float(scalar))
        else:
            raise AdapterInputError(f"TiDE output contains non-numeric value: {item!r}")

    if len(normalized) < horizon:
        raise AdapterInputError(
            f"TiDE output is shorter than requested horizon ({len(normalized)} < {horizon})",
        )
    return normalized[:horizon]


def _future_start_timestamp(*, last_timestamp: str, freq: str, pandas_module: Any) -> str:
    timestamps = _future_covariate_timestamps(
        last_timestamp=last_timestamp,
        freq=freq,
        horizon=1,
        pandas_module=pandas_module,
    )
    return timestamps[0]


def _future_covariate_timestamps(
    *,
    last_timestamp: str,
    freq: str,
    horizon: int,
    pandas_module: Any,
) -> list[str]:
    try:
        parsed = pandas_module.to_datetime([last_timestamp], utc=True, errors="raise")
        generated = pandas_module.date_range(start=parsed[0], periods=horizon + 1, freq=freq)
    except ValueError as exc:
        raise AdapterInputError(f"invalid frequency {freq!r} for TiDE forecast") from exc
    return [_to_iso_timestamp(value) for value in generated[1:]]


def build_covariate_warnings(series_list: list[SeriesInput], *, requested_quantiles: list[float]) -> list[str]:
    warnings: list[str] = []
    if requested_quantiles:
        warnings.append(
            "TiDE runner currently returns deterministic mean forecasts only; requested quantiles were ignored",
        )

    for series in series_list:
        for covariates, kind in ((series.past_covariates, "past_covariates"), (series.future_covariates, "future_covariates")):
            if not covariates:
                continue
            for name, values in covariates.items():
                if values and isinstance(values[0], str):
                    warnings.append(
                        f"TiDE runner requires numeric covariates; categorical {kind}[{name!r}] is not supported",
                    )
                    return warnings
    return warnings


def _resolve_runtime_config(
    *,
    model_name: str,
    model_metadata: dict[str, Any] | None,
) -> _RuntimeConfig:
    defaults = _TIDE_MODELS.get(model_name)
    if defaults is None:
        raise UnsupportedModelError(f"unsupported TiDE model {model_name!r}")

    input_chunk_length = _dict_positive_int(model_metadata, "default_input_chunk_length") or int(
        defaults["default_input_chunk_length"],
    )
    hidden_size = _dict_positive_int(model_metadata, "default_hidden_size") or int(
        defaults["default_hidden_size"],
    )
    num_encoder_layers = _dict_positive_int(model_metadata, "default_num_encoder_layers") or int(
        defaults["default_num_encoder_layers"],
    )
    num_decoder_layers = _dict_positive_int(model_metadata, "default_num_decoder_layers") or int(
        defaults["default_num_decoder_layers"],
    )
    batch_size = _dict_positive_int(model_metadata, "default_batch_size") or int(
        defaults["default_batch_size"],
    )
    n_epochs = _dict_positive_int(model_metadata, "default_n_epochs") or int(
        defaults["default_n_epochs"],
    )
    dropout = _dict_float_in_range(model_metadata, "default_dropout", min_value=0.0, max_value=1.0)
    if dropout is None:
        dropout = float(defaults["default_dropout"])

    return _RuntimeConfig(
        model_name=model_name,
        implementation="tide",
        input_chunk_length=input_chunk_length,
        hidden_size=hidden_size,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        dropout=dropout,
        batch_size=batch_size,
        n_epochs=n_epochs,
    )


def _dict_positive_int(payload: dict[str, Any] | None, key: str) -> int | None:
    if not isinstance(payload, dict):
        return None
    value = payload.get(key)
    if not isinstance(value, int) or value <= 0:
        return None
    return value


def _dict_float_in_range(
    payload: dict[str, Any] | None,
    key: str,
    *,
    min_value: float,
    max_value: float,
) -> float | None:
    if not isinstance(payload, dict):
        return None
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    numeric = float(value)
    if numeric < min_value or numeric > max_value:
        return None
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
