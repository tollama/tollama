"""High-level Python SDK convenience facade for tollama."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import pandas as pd

from tollama.client import DEFAULT_BASE_URL, DEFAULT_TIMEOUT_SECONDS, TollamaClient
from tollama.core.ingest import (
    SERIES_ID_COLUMN_CANDIDATES as _SERIES_ID_COLUMN_CANDIDATES,
)
from tollama.core.ingest import (
    TIMESTAMP_COLUMN_CANDIDATES as _TIMESTAMP_COLUMN_CANDIDATES,
)
from tollama.core.ingest import (
    TabularFormat,
    load_series_inputs_from_path,
)
from tollama.core.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    AutoForecastRequest,
    AutoForecastResponse,
    ForecastRequest,
    ForecastResponse,
    PipelineRequest,
    PipelineResponse,
    SeriesForecast,
    WhatIfRequest,
    WhatIfResponse,
)

SeriesPayload = Mapping[str, Any] | Sequence[Mapping[str, Any]] | pd.Series | pd.DataFrame


class TollamaForecastResult:
    """Forecast result wrapper with convenience accessors for notebooks/scripts."""

    def __init__(self, response: ForecastResponse) -> None:
        self._response = response

    @property
    def response(self) -> ForecastResponse:
        """Return the canonical response object."""
        return self._response

    @property
    def model(self) -> str:
        """Return model name used for the forecast."""
        return self._response.model

    @property
    def forecasts(self) -> list[SeriesForecast]:
        """Return per-series forecast entries."""
        return list(self._response.forecasts)

    @property
    def mean(self) -> list[int | float]:
        """Return mean values for a single-series response."""
        forecast = self._single_series_forecast()
        return list(forecast.mean)

    @property
    def quantiles(self) -> dict[str, list[int | float]] | None:
        """Return quantiles for a single-series response."""
        forecast = self._single_series_forecast()
        if forecast.quantiles is None:
            return None
        return {key: list(values) for key, values in forecast.quantiles.items()}

    @property
    def warnings(self) -> list[str]:
        """Return warning messages from the response payload."""
        if self._response.warnings is None:
            return []
        return list(self._response.warnings)

    @property
    def usage(self) -> dict[str, Any]:
        """Return usage metadata from the response payload."""
        if self._response.usage is None:
            return {}
        return dict(self._response.usage)

    def to_df(self) -> pd.DataFrame:
        """Convert forecast payload into a flat pandas DataFrame."""
        rows: list[dict[str, Any]] = []
        for forecast in self._response.forecasts:
            timestamps = _forecast_timestamps(forecast)
            quantiles = forecast.quantiles or {}
            sorted_quantiles = sorted(quantiles.items(), key=lambda item: float(item[0]))

            for offset, mean_value in enumerate(forecast.mean):
                row: dict[str, Any] = {
                    "id": forecast.id,
                    "freq": forecast.freq,
                    "start_timestamp": forecast.start_timestamp,
                    "step": offset + 1,
                    "timestamp": timestamps[offset],
                    "mean": mean_value,
                }
                for quantile_key, quantile_values in sorted_quantiles:
                    row[f"q{quantile_key}"] = quantile_values[offset]
                rows.append(row)

        frame = pd.DataFrame(rows)
        if frame.empty:
            return frame
        return frame.sort_values(["id", "step"], kind="stable").reset_index(drop=True)

    def _single_series_forecast(self) -> SeriesForecast:
        forecasts = self._response.forecasts
        if len(forecasts) != 1:
            raise ValueError(
                "single-series accessor requested but response contains multiple series; "
                "use forecasts or to_df() instead",
            )
        return forecasts[0]


class Tollama:
    """High-level SDK facade over :class:`tollama.client.TollamaClient`."""

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        api_key: str | None = None,
        *,
        client: TollamaClient | None = None,
    ) -> None:
        self._client = client or TollamaClient(base_url=base_url, timeout=timeout, api_key=api_key)

    def health(self) -> dict[str, Any]:
        """Return daemon health and version payload."""
        return self._client.health()

    def models(self, mode: str = "installed") -> list[dict[str, Any]]:
        """List model entries for the requested mode."""
        return self._client.models(mode=mode)

    def pull(self, model: str, *, accept_license: bool = False) -> dict[str, Any]:
        """Pull a model in non-stream mode."""
        return self._client.pull(model, accept_license=accept_license)

    def show(self, model: str) -> dict[str, Any]:
        """Return model metadata payload."""
        return self._client.show(model)

    def analyze(
        self,
        *,
        series: SeriesPayload,
        parameters: Mapping[str, Any] | None = None,
    ) -> AnalyzeResponse:
        """Run a validated series analysis request."""
        payload: dict[str, Any] = {
            "series": _coerce_series_payload(series),
        }
        if parameters is not None:
            payload["parameters"] = dict(parameters)

        request = AnalyzeRequest.model_validate(payload)
        return self._client.analyze(request)

    def forecast(
        self,
        *,
        model: str,
        series: SeriesPayload,
        horizon: int,
        quantiles: Sequence[float] | None = None,
        options: Mapping[str, Any] | None = None,
        timeout: float | None = None,
        parameters: Mapping[str, Any] | None = None,
    ) -> TollamaForecastResult:
        """Run a validated non-streaming forecast request."""
        payload: dict[str, Any] = {
            "model": model,
            "horizon": horizon,
            "series": _coerce_series_payload(series),
            "options": dict(options or {}),
        }
        if quantiles is not None:
            payload["quantiles"] = list(quantiles)
        if timeout is not None:
            payload["timeout"] = timeout
        if parameters is not None:
            payload["parameters"] = dict(parameters)

        request = ForecastRequest.model_validate(payload)
        response = self._client.forecast_response(request)
        return TollamaForecastResult(response=response)

    def forecast_from_file(
        self,
        *,
        model: str,
        path: str | Path,
        horizon: int,
        format_hint: str | None = None,
        timestamp_column: str | None = None,
        series_id_column: str | None = None,
        target_column: str | None = None,
        freq_column: str | None = None,
        quantiles: Sequence[float] | None = None,
        options: Mapping[str, Any] | None = None,
        timeout: float | None = None,
        parameters: Mapping[str, Any] | None = None,
    ) -> TollamaForecastResult:
        """Load CSV/Parquet into canonical series payloads and run forecast."""
        series = load_series_inputs_from_path(
            path,
            format_hint=cast(TabularFormat | None, format_hint),
            timestamp_column=timestamp_column,
            series_id_column=series_id_column,
            target_column=target_column,
            freq_column=freq_column,
        )
        payload: dict[str, Any] = {
            "model": model,
            "horizon": horizon,
            "series": [item.model_dump(mode="python", exclude_none=True) for item in series],
            "options": dict(options or {}),
        }
        if quantiles is not None:
            payload["quantiles"] = list(quantiles)
        if timeout is not None:
            payload["timeout"] = timeout
        if parameters is not None:
            payload["parameters"] = dict(parameters)

        request = ForecastRequest.model_validate(payload)
        response = self._client.forecast_response(request)
        return TollamaForecastResult(response=response)

    def auto_forecast(
        self,
        *,
        series: SeriesPayload,
        horizon: int,
        strategy: str = "auto",
        model: str | None = None,
        allow_fallback: bool = False,
        ensemble_top_k: int = 3,
        ensemble_method: str = "mean",
        quantiles: Sequence[float] | None = None,
        options: Mapping[str, Any] | None = None,
        timeout: float | None = None,
        keep_alive: str | int | float | None = None,
        parameters: Mapping[str, Any] | None = None,
    ) -> AutoForecastResponse:
        """Run validated zero-config auto-forecast and return selection metadata."""
        payload: dict[str, Any] = {
            "horizon": horizon,
            "strategy": strategy,
            "series": _coerce_series_payload(series),
            "options": dict(options or {}),
            "allow_fallback": allow_fallback,
            "ensemble_top_k": ensemble_top_k,
            "ensemble_method": ensemble_method,
        }
        if model is not None:
            payload["model"] = model
        if quantiles is not None:
            payload["quantiles"] = list(quantiles)
        if timeout is not None:
            payload["timeout"] = timeout
        if keep_alive is not None:
            payload["keep_alive"] = keep_alive
        if parameters is not None:
            payload["parameters"] = dict(parameters)

        request = AutoForecastRequest.model_validate(payload)
        return self._client.auto_forecast(request)

    def what_if(
        self,
        *,
        model: str,
        series: SeriesPayload,
        horizon: int,
        scenarios: Sequence[Mapping[str, Any]],
        continue_on_error: bool = True,
        quantiles: Sequence[float] | None = None,
        options: Mapping[str, Any] | None = None,
        timeout: float | None = None,
        keep_alive: str | int | float | None = None,
        parameters: Mapping[str, Any] | None = None,
    ) -> WhatIfResponse:
        """Run validated what-if scenario analysis and return baseline + scenario outputs."""
        payload: dict[str, Any] = {
            "model": model,
            "horizon": horizon,
            "series": _coerce_series_payload(series),
            "scenarios": [dict(item) for item in scenarios],
            "continue_on_error": continue_on_error,
            "options": dict(options or {}),
        }
        if quantiles is not None:
            payload["quantiles"] = list(quantiles)
        if timeout is not None:
            payload["timeout"] = timeout
        if keep_alive is not None:
            payload["keep_alive"] = keep_alive
        if parameters is not None:
            payload["parameters"] = dict(parameters)

        request = WhatIfRequest.model_validate(payload)
        return self._client.what_if(request)

    def pipeline(
        self,
        *,
        series: SeriesPayload,
        horizon: int,
        strategy: str = "auto",
        model: str | None = None,
        allow_fallback: bool = False,
        ensemble_top_k: int = 3,
        ensemble_method: str = "mean",
        quantiles: Sequence[float] | None = None,
        options: Mapping[str, Any] | None = None,
        timeout: float | None = None,
        keep_alive: str | int | float | None = None,
        parameters: Mapping[str, Any] | None = None,
        analyze_parameters: Mapping[str, Any] | None = None,
        recommend_top_k: int = 3,
        allow_restricted_license: bool = False,
        pull_if_missing: bool = True,
        accept_license: bool = False,
    ) -> PipelineResponse:
        """Run full pipeline orchestration and return analysis + selection + forecast payload."""
        payload: dict[str, Any] = {
            "horizon": horizon,
            "strategy": strategy,
            "series": _coerce_series_payload(series),
            "options": dict(options or {}),
            "allow_fallback": allow_fallback,
            "ensemble_top_k": ensemble_top_k,
            "ensemble_method": ensemble_method,
            "recommend_top_k": recommend_top_k,
            "allow_restricted_license": allow_restricted_license,
            "pull_if_missing": pull_if_missing,
            "accept_license": accept_license,
        }
        if model is not None:
            payload["model"] = model
        if quantiles is not None:
            payload["quantiles"] = list(quantiles)
        if timeout is not None:
            payload["timeout"] = timeout
        if keep_alive is not None:
            payload["keep_alive"] = keep_alive
        if parameters is not None:
            payload["parameters"] = dict(parameters)
        if analyze_parameters is not None:
            payload["analyze_parameters"] = dict(analyze_parameters)

        request = PipelineRequest.model_validate(payload)
        return self._client.pipeline(request)


def _coerce_series_payload(series: SeriesPayload) -> list[dict[str, Any]]:
    if isinstance(series, pd.Series):
        return [_series_from_pandas_series(series)]
    if isinstance(series, pd.DataFrame):
        return _series_from_pandas_dataframe(series)
    if isinstance(series, Mapping):
        return [_series_from_mapping(series, default_id="series_0")]

    if isinstance(series, Sequence) and not isinstance(series, (str, bytes, bytearray)):
        if not series:
            raise ValueError("series must not be empty")
        payloads: list[dict[str, Any]] = []
        for index, item in enumerate(series):
            if not isinstance(item, Mapping):
                raise TypeError(f"series[{index}] must be a mapping")
            payloads.append(_series_from_mapping(item, default_id=f"series_{index}"))
        return payloads

    raise TypeError("series must be a mapping, list of mappings, pandas Series, or DataFrame")


def _series_from_mapping(payload: Mapping[str, Any], *, default_id: str) -> dict[str, Any]:
    if "target" not in payload:
        raise ValueError("series mapping must include target")

    target_values = payload["target"]
    if isinstance(target_values, (str, bytes, bytearray)) or not isinstance(
        target_values,
        Sequence,
    ):
        raise TypeError("series target must be a non-string sequence")
    target = list(target_values)

    timestamps_value = payload.get("timestamps")
    if timestamps_value is None:
        timestamps = [str(index) for index in range(len(target))]
    else:
        if isinstance(timestamps_value, (str, bytes, bytearray)) or not isinstance(
            timestamps_value,
            Sequence,
        ):
            raise TypeError("series timestamps must be a non-string sequence")
        timestamps = [_stringify_timestamp(value) for value in timestamps_value]

    normalized = dict(payload)
    normalized["id"] = _stringify_series_id(payload.get("id"), default_id=default_id)
    normalized["freq"] = _normalize_freq(payload.get("freq"))
    normalized["target"] = target
    normalized["timestamps"] = timestamps
    return normalized


def _series_from_pandas_series(series: pd.Series) -> dict[str, Any]:
    target = series.tolist()
    timestamps = [_stringify_timestamp(value) for value in series.index.tolist()]
    return {
        "id": _stringify_series_id(series.name, default_id="series_0"),
        "freq": _infer_freq(series.index.tolist()),
        "timestamps": timestamps,
        "target": target,
    }


def _series_from_pandas_dataframe(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if frame.empty:
        raise ValueError("series DataFrame must not be empty")
    if "target" in frame.columns:
        return _series_from_target_dataframe(frame)
    return _series_from_wide_dataframe(frame)


def _series_from_target_dataframe(frame: pd.DataFrame) -> list[dict[str, Any]]:
    timestamp_column = _first_existing_column(frame, _TIMESTAMP_COLUMN_CANDIDATES)
    series_id_column = _first_existing_column(frame, _SERIES_ID_COLUMN_CANDIDATES)

    if series_id_column is None:
        return [
            _single_series_from_target_frame(
                frame,
                series_id="series_0",
                timestamp_column=timestamp_column,
            ),
        ]

    payloads: list[dict[str, Any]] = []
    for raw_series_id, group in frame.groupby(series_id_column, sort=False, dropna=False):
        payloads.append(
            _single_series_from_target_frame(
                group,
                series_id=_stringify_series_id(raw_series_id, default_id="series_0"),
                timestamp_column=timestamp_column,
            ),
        )
    return payloads


def _single_series_from_target_frame(
    frame: pd.DataFrame,
    *,
    series_id: str,
    timestamp_column: str | None,
) -> dict[str, Any]:
    target_values = frame["target"].tolist()
    if timestamp_column is None:
        timestamp_values = frame.index.tolist()
    else:
        timestamp_values = frame[timestamp_column].tolist()
    return {
        "id": series_id,
        "freq": _infer_freq(timestamp_values),
        "timestamps": [_stringify_timestamp(value) for value in timestamp_values],
        "target": target_values,
    }


def _series_from_wide_dataframe(frame: pd.DataFrame) -> list[dict[str, Any]]:
    numeric_columns = [
        column for column in frame.columns if pd.api.types.is_numeric_dtype(frame[column])
    ]
    if not numeric_columns:
        raise ValueError(
            "series DataFrame must include target column or at least one numeric column",
        )

    timestamp_values = frame.index.tolist()
    timestamps = [_stringify_timestamp(value) for value in timestamp_values]
    freq = _infer_freq(timestamp_values)
    payloads: list[dict[str, Any]] = []
    for column in numeric_columns:
        payloads.append(
            {
                "id": _stringify_series_id(column, default_id="series_0"),
                "freq": freq,
                "timestamps": timestamps,
                "target": frame[column].tolist(),
            },
        )
    return payloads


def _first_existing_column(frame: pd.DataFrame, candidates: Sequence[str]) -> str | None:
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    return None


def _stringify_series_id(value: Any, *, default_id: str) -> str:
    if isinstance(value, str) and value:
        return value
    if value is None:
        return default_id
    return str(value)


def _normalize_freq(value: Any) -> str:
    if isinstance(value, str) and value:
        return value
    return "auto"


def _infer_freq(timestamps: Sequence[Any]) -> str:
    if len(timestamps) < 3:
        return "auto"
    try:
        normalized = pd.DatetimeIndex(pd.to_datetime(list(timestamps), errors="raise"))
    except Exception:  # noqa: BLE001
        return "auto"
    inferred = pd.infer_freq(normalized)
    if inferred is None:
        return "auto"
    return str(inferred)


def _stringify_timestamp(value: Any) -> str:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "isoformat"):
        try:
            serialized = value.isoformat()
        except Exception:  # noqa: BLE001
            serialized = None
        if isinstance(serialized, str) and serialized:
            return serialized
    if isinstance(value, str):
        return value
    return str(value)


def _forecast_timestamps(forecast: SeriesForecast) -> list[str | None]:
    horizon = len(forecast.mean)
    if horizon <= 0:
        return []

    try:
        start = pd.Timestamp(forecast.start_timestamp)
        index = pd.date_range(start=start, periods=horizon, freq=forecast.freq)
    except Exception:  # noqa: BLE001
        return [None for _ in range(horizon)]
    return [timestamp.isoformat() for timestamp in index]
