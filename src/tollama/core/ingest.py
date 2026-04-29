"""CSV/Parquet ingest helpers for canonical SeriesInput payloads."""

from __future__ import annotations

import csv
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from io import BytesIO, StringIO
from pathlib import Path
from typing import Any, Literal
from urllib.parse import unquote, urlparse

import numpy as np
import pandas as pd

from .schemas import (
    IngestPreprocessingOptions,
    MissingValuePreprocessingOptions,
    SeriesInput,
    SeriesPreprocessingDiagnostics,
)

TabularFormat = Literal["csv", "parquet"]
CSV_DELIMITERS = (",", ";", "\t", "|")
CSV_SAMPLE_BYTES = 64 * 1024

TIMESTAMP_COLUMN_CANDIDATES = (
    "timestamp",
    "timestamps",
    "ds",
    "time",
    "date",
    "datetime",
    "date_time",
    "observation_date",
    "utc_timestamp",
    "year",
    "fecha",
)
SERIES_ID_COLUMN_CANDIDATES = ("id", "series", "series_id", "unique_id", "entity", "country")
TARGET_COLUMN_CANDIDATES = (
    "target",
    "value",
    "y",
    "ot",
    "demand",
    "users",
    "number of flights",
    "total electricity",
    "gdp",
    "close",
    "actual",
    "pm2.5",
    "pm25",
    "pm10",
    "no2",
    "so2",
    "co",
)
TARGET_COLUMN_SUFFIX_CANDIDATES = ("_load_actual_entsoe_transparency",)
FREQ_COLUMN_CANDIDATES = ("freq", "frequency")


class IngestError(ValueError):
    """Raised when tabular input cannot be transformed into canonical series payloads."""


class IngestDependencyError(IngestError):
    """Raised when optional runtime dependencies for ingest are unavailable."""


@dataclass(frozen=True, slots=True)
class SeriesIngestResult:
    """Series plus optional preprocessing diagnostics from tabular ingest."""

    series: list[SeriesInput]
    preprocessing: list[SeriesPreprocessingDiagnostics]


def load_series_inputs_from_data_url(
    data_url: str,
    *,
    format_hint: TabularFormat | None = None,
    timestamp_column: str | None = None,
    series_id_column: str | None = None,
    target_column: str | None = None,
    freq: str | None = None,
    freq_column: str | None = None,
    allow_remote: bool = False,
    preprocessing: IngestPreprocessingOptions | None = None,
) -> list[SeriesInput]:
    """Load canonical series inputs from a local path/file:// URL (or remote URL when enabled)."""
    return load_series_inputs_result_from_data_url(
        data_url,
        format_hint=format_hint,
        timestamp_column=timestamp_column,
        series_id_column=series_id_column,
        target_column=target_column,
        freq=freq,
        freq_column=freq_column,
        allow_remote=allow_remote,
        preprocessing=preprocessing,
    ).series


def load_series_inputs_result_from_data_url(
    data_url: str,
    *,
    format_hint: TabularFormat | None = None,
    timestamp_column: str | None = None,
    series_id_column: str | None = None,
    target_column: str | None = None,
    freq: str | None = None,
    freq_column: str | None = None,
    allow_remote: bool = False,
    preprocessing: IngestPreprocessingOptions | None = None,
) -> SeriesIngestResult:
    """Load series inputs and preprocessing diagnostics from a data_url."""
    parsed = urlparse(data_url)
    scheme = parsed.scheme.lower()

    if scheme in {"", "file"}:
        raw_path = unquote(parsed.path) if scheme == "file" else data_url
        path = Path(raw_path).expanduser().resolve()
        if ".." in Path(raw_path).parts:
            raise IngestError(
                "data_url path must not contain '..' components",
            )
        return load_series_inputs_result_from_path(
            path,
            format_hint=format_hint,
            timestamp_column=timestamp_column,
            series_id_column=series_id_column,
            target_column=target_column,
            freq=freq,
            freq_column=freq_column,
            preprocessing=preprocessing,
        )

    if scheme in {"http", "https"}:
        if not allow_remote:
            raise IngestError(
                "remote data_url is disabled for security reasons; use a local path or file:// URL",
            )
        return _load_series_inputs_result_from_remote_url(
            data_url,
            format_hint=format_hint,
            timestamp_column=timestamp_column,
            series_id_column=series_id_column,
            target_column=target_column,
            freq=freq,
            freq_column=freq_column,
            preprocessing=preprocessing,
        )

    raise IngestError(
        f"unsupported data_url scheme {scheme!r}; expected local path, file://, or http(s)",
    )


def load_series_inputs_from_path(
    path: str | Path,
    *,
    format_hint: TabularFormat | None = None,
    timestamp_column: str | None = None,
    series_id_column: str | None = None,
    target_column: str | None = None,
    freq: str | None = None,
    freq_column: str | None = None,
    preprocessing: IngestPreprocessingOptions | None = None,
) -> list[SeriesInput]:
    """Load canonical series inputs from a local CSV or Parquet file."""
    return load_series_inputs_result_from_path(
        path,
        format_hint=format_hint,
        timestamp_column=timestamp_column,
        series_id_column=series_id_column,
        target_column=target_column,
        freq=freq,
        freq_column=freq_column,
        preprocessing=preprocessing,
    ).series


def load_series_inputs_result_from_path(
    path: str | Path,
    *,
    format_hint: TabularFormat | None = None,
    timestamp_column: str | None = None,
    series_id_column: str | None = None,
    target_column: str | None = None,
    freq: str | None = None,
    freq_column: str | None = None,
    preprocessing: IngestPreprocessingOptions | None = None,
) -> SeriesIngestResult:
    """Load canonical series inputs and diagnostics from a local CSV or Parquet file."""
    resolved_path = Path(path).expanduser().resolve()
    if not resolved_path.exists():
        raise IngestError(f"input file does not exist: {resolved_path}")
    if not resolved_path.is_file():
        raise IngestError(f"input path is not a file: {resolved_path}")

    tabular_format = _resolve_format(
        path=resolved_path,
        format_hint=format_hint,
    )
    frame = _read_frame_from_path(path=resolved_path, tabular_format=tabular_format)
    return series_inputs_result_from_frame(
        frame,
        timestamp_column=timestamp_column,
        series_id_column=series_id_column,
        target_column=target_column,
        freq=freq,
        freq_column=freq_column,
        preprocessing=preprocessing,
    )


def load_series_inputs_from_bytes(
    payload: bytes,
    *,
    filename: str,
    format_hint: TabularFormat | None = None,
    timestamp_column: str | None = None,
    series_id_column: str | None = None,
    target_column: str | None = None,
    freq: str | None = None,
    freq_column: str | None = None,
    preprocessing: IngestPreprocessingOptions | None = None,
) -> list[SeriesInput]:
    """Load canonical series inputs from uploaded CSV/Parquet bytes."""
    return load_series_inputs_result_from_bytes(
        payload,
        filename=filename,
        format_hint=format_hint,
        timestamp_column=timestamp_column,
        series_id_column=series_id_column,
        target_column=target_column,
        freq=freq,
        freq_column=freq_column,
        preprocessing=preprocessing,
    ).series


def load_series_inputs_result_from_bytes(
    payload: bytes,
    *,
    filename: str,
    format_hint: TabularFormat | None = None,
    timestamp_column: str | None = None,
    series_id_column: str | None = None,
    target_column: str | None = None,
    freq: str | None = None,
    freq_column: str | None = None,
    preprocessing: IngestPreprocessingOptions | None = None,
) -> SeriesIngestResult:
    """Load canonical series inputs and diagnostics from uploaded CSV/Parquet bytes."""
    if not payload:
        raise IngestError("uploaded file is empty")

    tabular_format = _resolve_format(path=Path(filename), format_hint=format_hint)
    frame = _read_frame_from_bytes(payload=payload, tabular_format=tabular_format)
    return series_inputs_result_from_frame(
        frame,
        timestamp_column=timestamp_column,
        series_id_column=series_id_column,
        target_column=target_column,
        freq=freq,
        freq_column=freq_column,
        preprocessing=preprocessing,
    )


def series_inputs_from_frame(
    frame: pd.DataFrame,
    *,
    timestamp_column: str | None = None,
    series_id_column: str | None = None,
    target_column: str | None = None,
    freq: str | None = None,
    freq_column: str | None = None,
) -> list[SeriesInput]:
    """Transform a tabular DataFrame into canonical SeriesInput payloads."""
    return series_inputs_result_from_frame(
        frame,
        timestamp_column=timestamp_column,
        series_id_column=series_id_column,
        target_column=target_column,
        freq=freq,
        freq_column=freq_column,
    ).series


def series_inputs_result_from_frame(
    frame: pd.DataFrame,
    *,
    timestamp_column: str | None = None,
    series_id_column: str | None = None,
    target_column: str | None = None,
    freq: str | None = None,
    freq_column: str | None = None,
    preprocessing: IngestPreprocessingOptions | None = None,
) -> SeriesIngestResult:
    """Transform a tabular DataFrame into canonical series plus diagnostics."""
    if frame.empty:
        raise IngestError("input table is empty")

    resolved_timestamp = timestamp_column or _first_existing_column(
        frame,
        TIMESTAMP_COLUMN_CANDIDATES,
    )
    resolved_series_id = series_id_column or _first_existing_column(
        frame,
        SERIES_ID_COLUMN_CANDIDATES,
    )
    resolved_freq = _normalize_optional_text(freq)
    resolved_freq_column = freq_column or _first_existing_column(frame, FREQ_COLUMN_CANDIDATES)
    resolved_target = target_column or _resolve_target_column(
        frame,
        timestamp_column=resolved_timestamp,
        series_id_column=resolved_series_id,
        freq_column=resolved_freq_column,
    )

    if resolved_timestamp is None and not isinstance(frame.index, pd.DatetimeIndex):
        raise IngestError(
            "timestamp column not found; provide timestamp_column or use one of "
            f"{TIMESTAMP_COLUMN_CANDIDATES!r}",
        )

    groups: list[tuple[str, pd.DataFrame]] = []
    if resolved_series_id is None:
        groups.append(("series_0", frame))
    else:
        for raw_series_id, group in frame.groupby(resolved_series_id, sort=False, dropna=False):
            series_id = _stringify_series_id(raw_series_id, default_id="series_0")
            groups.append((series_id, group))

    series_inputs: list[SeriesInput] = []
    preprocessing_diagnostics: list[SeriesPreprocessingDiagnostics] = []
    missing_options = _enabled_missing_options(preprocessing)
    for series_id, group in groups:
        sorted_group = _sort_group(
            group,
            timestamp_column=resolved_timestamp,
        )
        if missing_options is None:
            raw_timestamps = _extract_timestamps(
                sorted_group,
                timestamp_column=resolved_timestamp,
            )
            resolved_series_freq = _extract_freq(
                sorted_group,
                freq=resolved_freq,
                freq_column=resolved_freq_column,
                timestamps=raw_timestamps,
            )
            sorted_group = _drop_null_target_rows(sorted_group, target_column=resolved_target)
            timestamps = _extract_timestamps(
                sorted_group,
                timestamp_column=resolved_timestamp,
            )
            target = _extract_target_values(sorted_group, target_column=resolved_target)
            if resolved_series_freq == "auto" and timestamps != raw_timestamps:
                resolved_series_freq = _extract_freq(
                    sorted_group,
                    freq=resolved_freq,
                    freq_column=resolved_freq_column,
                    timestamps=timestamps,
                )
        else:
            timestamps, target, resolved_series_freq, diagnostics = _regularize_and_impute_group(
                sorted_group,
                series_id=series_id,
                timestamp_column=resolved_timestamp,
                target_column=resolved_target,
                freq=resolved_freq,
                freq_column=resolved_freq_column,
                options=missing_options,
            )
            preprocessing_diagnostics.append(diagnostics)

        series_inputs.append(
            SeriesInput.model_validate(
                {
                    "id": series_id,
                    "freq": resolved_series_freq,
                    "timestamps": timestamps,
                    "target": target,
                }
            )
        )

    if not series_inputs:
        raise IngestError("no series rows found in input")

    return SeriesIngestResult(series=series_inputs, preprocessing=preprocessing_diagnostics)


def _enabled_missing_options(
    preprocessing: IngestPreprocessingOptions | None,
) -> MissingValuePreprocessingOptions | None:
    if preprocessing is None:
        return None
    return preprocessing.missing if preprocessing.missing.enabled else None


def _regularize_and_impute_group(
    group: pd.DataFrame,
    *,
    series_id: str,
    timestamp_column: str | None,
    target_column: str,
    freq: str | None,
    freq_column: str | None,
    options: MissingValuePreprocessingOptions,
) -> tuple[list[str], list[float], str, SeriesPreprocessingDiagnostics]:
    original_row_count = len(group)
    timestamps = _extract_timestamps(group, timestamp_column=timestamp_column)
    target = _extract_target_values_allowing_nulls(group, target_column=target_column)
    raw_null_target_count = int(np.isnan(target).sum())
    if raw_null_target_count == len(target):
        raise IngestError("target column contains only null values")

    try:
        index = pd.DatetimeIndex(pd.to_datetime(timestamps, errors="raise"))
    except Exception as exc:  # noqa: BLE001
        raise IngestError(
            "missing preprocessing requires parseable datetime timestamps",
        ) from exc

    warnings: list[str] = []
    if index.has_duplicates:
        duplicate_timestamp_count = int(index.duplicated(keep="first").sum())
        collapsed = pd.Series(target, index=index).groupby(level=0, sort=True).mean()
        index = pd.DatetimeIndex(collapsed.index)
        target = collapsed.to_numpy(dtype=float)
        timestamps = [_stringify_timestamp(value) for value in index.tolist()]
        warnings.append(
            f"collapsed {duplicate_timestamp_count} duplicate timestamp rows by "
            "averaging target values"
        )

    resolved_series_freq = _extract_freq(
        group,
        freq=freq,
        freq_column=freq_column,
        timestamps=timestamps,
    )
    if resolved_series_freq == "auto":
        fallback_freq = _dominant_interval_frequency(index, min_share=0.5)
        if fallback_freq is None:
            raise IngestError(
                "missing preprocessing requires an explicit or inferred frequency; set ingest.freq",
            )
        resolved_series_freq = fallback_freq

    try:
        regular_index = pd.date_range(
            start=index.min(),
            end=index.max(),
            freq=resolved_series_freq,
        )
    except ValueError as exc:
        raise IngestError(
            f"missing preprocessing could not build regular grid for freq={resolved_series_freq!r}",
        ) from exc

    if len(regular_index) == 0:
        raise IngestError("missing preprocessing produced an empty timestamp grid")

    aligned = pd.Series(target, index=index).reindex(regular_index)
    values = aligned.to_numpy(dtype=float)
    missing_mask = np.isnan(values)
    missing_count = int(missing_mask.sum())
    missing_ratio = float(missing_count / len(values))
    max_gap = _max_consecutive_true(missing_mask)
    if missing_ratio > options.max_missing_ratio:
        raise IngestError(
            "target missing ratio "
            f"{missing_ratio:.4f} exceeds limit {options.max_missing_ratio:.4f}",
        )
    if options.max_gap is not None and max_gap > options.max_gap:
        raise IngestError(f"target max missing gap {max_gap} exceeds limit {options.max_gap}")

    imputed, used_method, imputation_warnings = _impute_missing_values(values, options=options)
    warnings.extend(imputation_warnings)
    missing_timestamp_count = int((~regular_index.isin(index)).sum())

    diagnostics = SeriesPreprocessingDiagnostics.model_validate(
        {
            "id": series_id,
            "original_row_count": original_row_count,
            "regularized_row_count": len(regular_index),
            "raw_null_target_count": raw_null_target_count,
            "missing_timestamp_count": missing_timestamp_count,
            "imputed_point_count": missing_count,
            "max_gap": max_gap,
            "missing_ratio": missing_ratio,
            "requested_method": options.method,
            "used_method": used_method,
            "warnings": warnings or None,
        }
    )
    return (
        [_stringify_timestamp(value) for value in regular_index.tolist()],
        [float(value) for value in imputed.tolist()],
        resolved_series_freq,
        diagnostics,
    )


def _extract_target_values_allowing_nulls(
    group: pd.DataFrame,
    *,
    target_column: str,
) -> np.ndarray:
    values = group[target_column].tolist()
    if not values:
        raise IngestError("series contains no target rows")

    normalized: list[float] = []
    for index, value in enumerate(values):
        if pd.isna(value):
            normalized.append(float("nan"))
            continue

        if hasattr(value, "item"):
            try:
                value = value.item()  # numpy scalar -> python scalar
            except Exception:  # noqa: BLE001
                pass

        if isinstance(value, bool):
            raise IngestError(f"target contains boolean value at row index {index}")
        if isinstance(value, (int, float)):
            normalized.append(float(value))
            continue

        raise IngestError(
            f"target contains non-numeric value {value!r} at row index {index}",
        )

    return np.asarray(normalized, dtype=float)


def _impute_missing_values(
    values: np.ndarray,
    *,
    options: MissingValuePreprocessingOptions,
) -> tuple[np.ndarray, str, list[str]]:
    result = np.asarray(values, dtype=float).copy()
    missing_mask = np.isnan(result)
    if not missing_mask.any():
        return result, "none", []

    valid_idx = np.where(~missing_mask)[0]
    if len(valid_idx) == 0:
        raise IngestError("target column contains only null values")

    warnings: list[str] = []
    first_valid = int(valid_idx[0])
    last_valid = int(valid_idx[-1])
    has_edge_missing = first_valid > 0 or last_valid < len(result) - 1
    if has_edge_missing:
        if options.edge_strategy == "reject":
            raise IngestError(
                "target contains leading or trailing missing values; "
                "set missing edge_strategy='nearest' to fill them",
            )
        result[:first_valid] = result[first_valid]
        result[last_valid + 1 :] = result[last_valid]
        warnings.append("edge missing values filled with nearest valid target")

    if not np.isnan(result).any():
        return result, "nearest", warnings

    requested = options.method
    if requested == "seasonal":
        return _seasonal_interpolate(result, period=options.seasonal_period), "seasonal", warnings
    if requested == "linear":
        return _linear_interpolate(result), "linear", warnings

    valid_count = int(np.count_nonzero(~np.isnan(result)))
    if requested == "bspline":
        if valid_count < 4:
            warnings.append("bspline interpolation requires at least 4 valid points; used linear")
            return _linear_interpolate(result), "linear", warnings
        return _bspline_interpolate(result), "bspline", warnings

    if valid_count < 4:
        warnings.append(
            "auto missing preprocessing used linear because fewer than 4 points are valid"
        )
        return _linear_interpolate(result), "linear", warnings

    try:
        return _bspline_interpolate(result), "bspline", warnings
    except IngestDependencyError:
        warnings.append("scipy unavailable for auto missing preprocessing; used linear")
        return _linear_interpolate(result), "linear", warnings


def _linear_interpolate(values: np.ndarray) -> np.ndarray:
    result = values.copy()
    mask = np.isnan(result)
    if not mask.any():
        return result

    valid_idx = np.where(~mask)[0]
    if len(valid_idx) == 0:
        return result
    if len(valid_idx) == 1:
        result[:] = result[valid_idx[0]]
        return result

    missing_idx = np.where(mask)[0]
    result[missing_idx] = np.interp(missing_idx, valid_idx, result[valid_idx])
    return result


def _bspline_interpolate(values: np.ndarray) -> np.ndarray:
    try:
        from tollama.preprocess.spline import SplinePreprocessor
    except ImportError as exc:
        raise IngestDependencyError(
            "bspline missing preprocessing requires optional dependency scipy; "
            'install with `pip install -e ".[preprocess]"`',
        ) from exc

    try:
        return SplinePreprocessor().interpolate_missing(values)
    except ImportError as exc:
        raise IngestDependencyError(str(exc)) from exc


def _seasonal_interpolate(values: np.ndarray, *, period: int | None) -> np.ndarray:
    result = values.copy()
    mask = np.isnan(result)
    if not mask.any():
        return result

    resolved_period = period or _detect_period(result[~mask])
    if resolved_period is None or resolved_period < 2:
        return _linear_interpolate(result)

    missing_idx = np.where(mask)[0]
    for idx in missing_idx:
        seasonal_values: list[float] = []
        previous = idx - resolved_period
        while previous >= 0:
            if not np.isnan(result[previous]):
                seasonal_values.append(float(result[previous]))
            previous -= resolved_period

        following = idx + resolved_period
        while following < len(result):
            if not np.isnan(result[following]):
                seasonal_values.append(float(result[following]))
            following += resolved_period

        if seasonal_values:
            result[idx] = float(np.mean(seasonal_values))

    if np.isnan(result).any():
        result = _linear_interpolate(result)
    return result


def _detect_period(values: np.ndarray) -> int | None:
    if len(values) < 6:
        return None
    centered = values - np.mean(values)
    fft_values = np.abs(np.fft.rfft(centered))
    if len(fft_values) < 3:
        return None
    fft_values[0] = 0
    dominant_freq_idx = int(np.argmax(fft_values[1:])) + 1
    if dominant_freq_idx == 0:
        return None
    period = len(values) // dominant_freq_idx
    if period < 2 or period > len(values) // 2:
        return None
    return period


def _max_consecutive_true(mask: np.ndarray) -> int:
    max_len = cur = 0
    for value in mask.astype(bool):
        if value:
            cur += 1
            max_len = max(max_len, cur)
        else:
            cur = 0
    return int(max_len)


def _resolve_format(*, path: Path, format_hint: TabularFormat | None) -> TabularFormat:
    if format_hint is not None:
        return format_hint

    suffix = path.suffix.lower()
    if suffix == ".csv":
        return "csv"
    if suffix in {".parquet", ".pq"}:
        return "parquet"

    raise IngestError(
        f"unable to detect file format for {path.name!r}; set format_hint to 'csv' or 'parquet'",
    )


def _read_frame_from_path(*, path: Path, tabular_format: TabularFormat) -> pd.DataFrame:
    if tabular_format == "csv":
        sample = _read_csv_sample_from_path(path)
        return _read_csv_with_fallback(
            lambda **kwargs: pd.read_csv(
                path,
                encoding="utf-8-sig",
                low_memory=False,
                **kwargs,
            ),
            sample=sample,
            source_label=str(path),
        )
    return _read_parquet(path)


def _read_frame_from_bytes(*, payload: bytes, tabular_format: TabularFormat) -> pd.DataFrame:
    if tabular_format == "csv":
        text = payload.decode("utf-8-sig")
        return _read_csv_with_fallback(
            lambda **kwargs: pd.read_csv(StringIO(text), **kwargs),
            sample=text[:CSV_SAMPLE_BYTES],
            source_label="uploaded CSV",
        )
    return _read_parquet(BytesIO(payload))


def _read_csv_sample_from_path(path: Path) -> str:
    with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
        return handle.read(CSV_SAMPLE_BYTES)


def _read_csv_with_fallback(
    read_csv: Any,
    *,
    sample: str,
    source_label: str,
) -> pd.DataFrame:
    errors: list[str] = []
    for sep, skiprows in _csv_parse_attempts(sample):
        try:
            frame = read_csv(sep=sep, skiprows=skiprows)
        except pd.errors.ParserError as exc:
            errors.append(str(exc))
            continue

        normalized = _normalize_csv_frame(frame)
        if _looks_like_unparsed_delimited_frame(normalized, sample=sample):
            errors.append("CSV appears to use a non-comma delimiter")
            continue
        return normalized

    detail = errors[-1] if errors else "no parse attempts produced a tabular frame"
    raise IngestError(f"unable to parse CSV {source_label}: {detail}")


def _csv_parse_attempts(sample: str) -> list[tuple[str, int]]:
    indexed_lines = [
        (index, line)
        for index, line in enumerate(sample.splitlines()[:25])
        if line.strip()
    ]
    if not indexed_lines:
        return [(",", 0)]

    delimiter_counts: dict[str, list[int]] = {
        delimiter: [_csv_field_count(line, delimiter=delimiter) for _, line in indexed_lines]
        for delimiter in CSV_DELIMITERS
    }
    delimiters = sorted(
        CSV_DELIMITERS,
        key=lambda delimiter: (
            max(delimiter_counts[delimiter]),
            1 if delimiter == "," else 0,
        ),
        reverse=True,
    )

    attempts: list[tuple[str, int]] = []
    for delimiter in delimiters:
        counts = delimiter_counts[delimiter]
        max_count = max(counts)
        header_candidates = [
            original_index
            for (original_index, _), count in zip(indexed_lines, counts, strict=True)
            if count == max_count
        ]
        skip_candidates = [0]
        if max_count > 1 and len(header_candidates) >= 2 and header_candidates[0] != 0:
            skip_candidates.append(header_candidates[0])
        for skiprows in skip_candidates:
            attempt = (delimiter, skiprows)
            if attempt not in attempts:
                attempts.append(attempt)
    return attempts


def _csv_field_count(line: str, *, delimiter: str) -> int:
    try:
        return len(next(csv.reader([line], delimiter=delimiter)))
    except csv.Error:
        return line.count(delimiter) + 1


def _looks_like_unparsed_delimited_frame(frame: pd.DataFrame, *, sample: str) -> bool:
    if len(frame.columns) != 1:
        return False
    column = str(frame.columns[0])
    if any(delimiter in column for delimiter in CSV_DELIMITERS):
        return True
    sample_lines = [line for line in sample.splitlines()[:5] if line.strip()]
    if not sample_lines:
        return False
    return any(
        max(_csv_field_count(line, delimiter=delimiter) for line in sample_lines) > 1
        for delimiter in CSV_DELIMITERS
    )


def _normalize_csv_frame(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.rename(columns={column: str(column).strip() for column in frame.columns})
    normalized = _normalize_unnamed_datetime_column(normalized)
    return _normalize_world_bank_frame(normalized)


def _normalize_unnamed_datetime_column(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or len(frame.columns) == 0:
        return frame
    first_column = str(frame.columns[0]).strip()
    if first_column and not first_column.lower().startswith("unnamed"):
        return frame
    parsed = pd.to_datetime(frame.iloc[:, 0], errors="coerce")
    if parsed.notna().mean() < 0.8:
        return frame
    return frame.rename(columns={frame.columns[0]: "timestamp"})


def _normalize_world_bank_frame(frame: pd.DataFrame) -> pd.DataFrame:
    id_columns = ("Country Name", "Country Code", "Indicator Name", "Indicator Code")
    if not all(column in frame.columns for column in id_columns):
        return frame

    year_columns = [
        column
        for column in frame.columns
        if len(str(column)) == 4 and str(column).isdigit()
    ]
    if len(year_columns) < 3:
        return frame

    long = frame.melt(
        id_vars=list(id_columns),
        value_vars=year_columns,
        var_name="year",
        value_name="target",
    )
    long["target"] = pd.to_numeric(long["target"], errors="coerce")
    country_code = long["Country Code"].astype(str).str.strip()
    country_name = long["Country Name"].astype(str).str.strip()
    long["series"] = country_code.where(country_code != "", country_name)
    long["timestamp"] = long["year"].astype(str) + "-01-01"

    has_target = long.groupby("series", dropna=False)["target"].transform(
        lambda values: values.notna().any()
    )
    long = long.loc[has_target, ["timestamp", "series", "target"]]
    if long.empty:
        raise IngestError("World Bank CSV contains no numeric yearly values")
    return long


def _load_series_inputs_from_remote_url(
    data_url: str,
    *,
    format_hint: TabularFormat | None,
    timestamp_column: str | None,
    series_id_column: str | None,
    target_column: str | None,
    freq: str | None,
    freq_column: str | None,
) -> list[SeriesInput]:
    return _load_series_inputs_result_from_remote_url(
        data_url,
        format_hint=format_hint,
        timestamp_column=timestamp_column,
        series_id_column=series_id_column,
        target_column=target_column,
        freq=freq,
        freq_column=freq_column,
        preprocessing=None,
    ).series


def _load_series_inputs_result_from_remote_url(
    data_url: str,
    *,
    format_hint: TabularFormat | None,
    timestamp_column: str | None,
    series_id_column: str | None,
    target_column: str | None,
    freq: str | None,
    freq_column: str | None,
    preprocessing: IngestPreprocessingOptions | None,
) -> SeriesIngestResult:
    tabular_format = _resolve_format(path=Path(urlparse(data_url).path), format_hint=format_hint)
    if tabular_format == "csv":
        try:
            frame = _normalize_csv_frame(
                pd.read_csv(data_url, encoding="utf-8-sig", low_memory=False)
            )
        except pd.errors.ParserError as exc:
            raise IngestError(f"unable to parse CSV {data_url}: {exc}") from exc
    else:
        frame = _read_parquet(data_url)
    return series_inputs_result_from_frame(
        frame,
        timestamp_column=timestamp_column,
        series_id_column=series_id_column,
        target_column=target_column,
        freq=freq,
        freq_column=freq_column,
        preprocessing=preprocessing,
    )


def _read_parquet(path_or_buffer: Any) -> pd.DataFrame:
    try:
        return pd.read_parquet(path_or_buffer)
    except ImportError as exc:
        raise IngestDependencyError(
            "parquet support requires optional dependency (pyarrow or fastparquet); "
            'install with `pip install -e ".[ingest]"`',
        ) from exc


def _resolve_target_column(
    frame: pd.DataFrame,
    *,
    timestamp_column: str | None,
    series_id_column: str | None,
    freq_column: str | None,
) -> str:
    candidate = _first_existing_column(frame, TARGET_COLUMN_CANDIDATES)
    if candidate is not None:
        return candidate

    candidate = _first_column_with_suffix(frame, TARGET_COLUMN_SUFFIX_CANDIDATES)
    if candidate is not None:
        return candidate

    excluded = {
        column for column in (timestamp_column, series_id_column, freq_column) if column is not None
    }
    numeric_candidates = [
        column
        for column in frame.columns
        if column not in excluded and pd.api.types.is_numeric_dtype(frame[column])
    ]
    if len(numeric_candidates) == 1:
        return numeric_candidates[0]
    if not numeric_candidates:
        raise IngestError(
            "target column not found; provide target_column or include one of "
            f"{TARGET_COLUMN_CANDIDATES!r}",
        )
    raise IngestError(
        "multiple numeric columns found; set target_column explicitly",
    )


def _first_existing_column(frame: pd.DataFrame, candidates: Sequence[str]) -> str | None:
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate

    normalized_columns: dict[str, str] = {}
    for column in frame.columns:
        if isinstance(column, str):
            normalized_columns.setdefault(_normalize_column_name(column), column)
    for candidate in candidates:
        match = normalized_columns.get(_normalize_column_name(candidate))
        if match is not None:
            return match
    return None


def _normalize_column_name(value: str) -> str:
    return value.strip().lstrip("\ufeff").casefold()


def _first_column_with_suffix(frame: pd.DataFrame, suffixes: Sequence[str]) -> str | None:
    normalized_suffixes = tuple(_normalize_column_name(suffix) for suffix in suffixes)
    for column in frame.columns:
        if not isinstance(column, str):
            continue
        normalized = _normalize_column_name(column)
        if normalized.endswith(normalized_suffixes):
            return column
    return None


def _sort_group(group: pd.DataFrame, *, timestamp_column: str | None) -> pd.DataFrame:
    if timestamp_column is not None:
        return group.sort_values(by=timestamp_column, kind="stable")
    return group.sort_index(kind="stable")


def _drop_null_target_rows(group: pd.DataFrame, *, target_column: str) -> pd.DataFrame:
    null_mask = group[target_column].isna()
    if not bool(null_mask.any()):
        return group

    filtered = group.loc[~null_mask]
    if filtered.empty:
        raise IngestError("target column contains only null values")
    return filtered


def _extract_timestamps(group: pd.DataFrame, *, timestamp_column: str | None) -> list[str]:
    if timestamp_column is not None:
        values = group[timestamp_column].tolist()
    else:
        values = group.index.tolist()
    if not values:
        raise IngestError("series contains no timestamp rows")
    return [_stringify_timestamp(value) for value in values]


def _extract_target_values(group: pd.DataFrame, *, target_column: str) -> list[int | float]:
    values = group[target_column].tolist()
    if not values:
        raise IngestError("series contains no target rows")

    normalized: list[int | float] = []
    for index, value in enumerate(values):
        if pd.isna(value):
            raise IngestError(f"target contains null at row index {index}")

        if hasattr(value, "item"):
            try:
                value = value.item()  # numpy scalar -> python scalar
            except Exception:  # noqa: BLE001
                pass

        if isinstance(value, bool):
            raise IngestError(f"target contains boolean value at row index {index}")

        if isinstance(value, int):
            normalized.append(value)
            continue
        if isinstance(value, float):
            normalized.append(value)
            continue

        raise IngestError(
            f"target contains non-numeric value {value!r} at row index {index}",
        )

    return normalized


def _extract_freq(
    group: pd.DataFrame,
    *,
    freq: str | None,
    freq_column: str | None,
    timestamps: Sequence[str],
) -> str:
    if freq is not None:
        return freq

    if freq_column is not None and freq_column in group.columns:
        for raw in group[freq_column].tolist():
            if isinstance(raw, str):
                normalized = raw.strip()
                if normalized:
                    return normalized

    if len(timestamps) < 3:
        return "auto"

    try:
        index = pd.DatetimeIndex(pd.to_datetime(list(timestamps), errors="raise"))
    except Exception:  # noqa: BLE001
        return "auto"

    inferred = pd.infer_freq(index)
    if inferred is None:
        fallback = _dominant_interval_frequency(index)
        if fallback is None:
            return "auto"
        return fallback
    return str(inferred)


def _dominant_interval_frequency(index: pd.DatetimeIndex, *, min_share: float = 0.8) -> str | None:
    if len(index) < 3:
        return None

    deltas: list[float] = []
    for previous, current in zip(index[:-1], index[1:], strict=True):
        delta = round((current - previous).total_seconds(), 6)
        if delta > 0.0:
            deltas.append(delta)
    if not deltas:
        return None

    dominant_seconds, dominant_count = Counter(deltas).most_common(1)[0]
    share = dominant_count / len(deltas)
    large_mostly_regular = len(deltas) >= 1_000 and share >= 0.5
    if share < min_share and not large_mostly_regular:
        return None

    return _offset_alias_from_seconds(dominant_seconds)


def _offset_alias_from_seconds(seconds: float) -> str | None:
    whole_seconds = round(seconds)
    if whole_seconds <= 0 or abs(seconds - whole_seconds) > 1e-6:
        return None

    units = (
        ("D", 86_400),
        ("h", 3_600),
        ("min", 60),
        ("s", 1),
    )
    for alias, unit_seconds in units:
        if whole_seconds % unit_seconds != 0:
            continue
        multiple = whole_seconds // unit_seconds
        return alias if multiple == 1 else f"{multiple}{alias}"
    return None


def _normalize_optional_text(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    if not normalized or normalized.lower() == "auto":
        return None
    return normalized


def _stringify_timestamp(value: Any) -> str:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    if hasattr(value, "isoformat"):
        try:
            encoded = value.isoformat()
        except Exception:  # noqa: BLE001
            encoded = None
        if isinstance(encoded, str) and encoded:
            return encoded

    if isinstance(value, str):
        return value
    return str(value)


def _stringify_series_id(value: Any, *, default_id: str) -> str:
    if isinstance(value, str):
        normalized = value.strip()
        if normalized:
            return normalized

    if value is None:
        return default_id
    return str(value)
