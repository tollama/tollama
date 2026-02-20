"""CSV/Parquet ingest helpers for canonical SeriesInput payloads."""

from __future__ import annotations

from collections.abc import Sequence
from io import BytesIO, StringIO
from pathlib import Path
from typing import Any, Literal
from urllib.parse import unquote, urlparse

import pandas as pd

from .schemas import SeriesInput

TabularFormat = Literal["csv", "parquet"]

TIMESTAMP_COLUMN_CANDIDATES = ("timestamp", "timestamps", "ds", "time")
SERIES_ID_COLUMN_CANDIDATES = ("id", "series_id", "unique_id")
TARGET_COLUMN_CANDIDATES = ("target", "value", "y")
FREQ_COLUMN_CANDIDATES = ("freq", "frequency")


class IngestError(ValueError):
    """Raised when tabular input cannot be transformed into canonical series payloads."""


class IngestDependencyError(IngestError):
    """Raised when optional runtime dependencies for ingest are unavailable."""


def load_series_inputs_from_data_url(
    data_url: str,
    *,
    format_hint: TabularFormat | None = None,
    timestamp_column: str | None = None,
    series_id_column: str | None = None,
    target_column: str | None = None,
    freq_column: str | None = None,
    allow_remote: bool = False,
) -> list[SeriesInput]:
    """Load canonical series inputs from a local path/file:// URL (or remote URL when enabled)."""
    parsed = urlparse(data_url)
    scheme = parsed.scheme.lower()

    if scheme in {"", "file"}:
        raw_path = unquote(parsed.path) if scheme == "file" else data_url
        path = Path(raw_path).expanduser()
        return load_series_inputs_from_path(
            path,
            format_hint=format_hint,
            timestamp_column=timestamp_column,
            series_id_column=series_id_column,
            target_column=target_column,
            freq_column=freq_column,
        )

    if scheme in {"http", "https"}:
        if not allow_remote:
            raise IngestError(
                "remote data_url is disabled for security reasons; "
                "use a local path or file:// URL",
            )
        return _load_series_inputs_from_remote_url(
            data_url,
            format_hint=format_hint,
            timestamp_column=timestamp_column,
            series_id_column=series_id_column,
            target_column=target_column,
            freq_column=freq_column,
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
    freq_column: str | None = None,
) -> list[SeriesInput]:
    """Load canonical series inputs from a local CSV or Parquet file."""
    resolved_path = Path(path).expanduser()
    if not resolved_path.exists():
        raise IngestError(f"input file does not exist: {resolved_path}")
    if not resolved_path.is_file():
        raise IngestError(f"input path is not a file: {resolved_path}")

    tabular_format = _resolve_format(
        path=resolved_path,
        format_hint=format_hint,
    )
    frame = _read_frame_from_path(path=resolved_path, tabular_format=tabular_format)
    return series_inputs_from_frame(
        frame,
        timestamp_column=timestamp_column,
        series_id_column=series_id_column,
        target_column=target_column,
        freq_column=freq_column,
    )


def load_series_inputs_from_bytes(
    payload: bytes,
    *,
    filename: str,
    format_hint: TabularFormat | None = None,
    timestamp_column: str | None = None,
    series_id_column: str | None = None,
    target_column: str | None = None,
    freq_column: str | None = None,
) -> list[SeriesInput]:
    """Load canonical series inputs from uploaded CSV/Parquet bytes."""
    if not payload:
        raise IngestError("uploaded file is empty")

    tabular_format = _resolve_format(path=Path(filename), format_hint=format_hint)
    frame = _read_frame_from_bytes(payload=payload, tabular_format=tabular_format)
    return series_inputs_from_frame(
        frame,
        timestamp_column=timestamp_column,
        series_id_column=series_id_column,
        target_column=target_column,
        freq_column=freq_column,
    )


def series_inputs_from_frame(
    frame: pd.DataFrame,
    *,
    timestamp_column: str | None = None,
    series_id_column: str | None = None,
    target_column: str | None = None,
    freq_column: str | None = None,
) -> list[SeriesInput]:
    """Transform a tabular DataFrame into canonical SeriesInput payloads."""
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
    resolved_freq = freq_column or _first_existing_column(frame, FREQ_COLUMN_CANDIDATES)
    resolved_target = target_column or _resolve_target_column(
        frame,
        timestamp_column=resolved_timestamp,
        series_id_column=resolved_series_id,
        freq_column=resolved_freq,
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
    for series_id, group in groups:
        sorted_group = _sort_group(
            group,
            timestamp_column=resolved_timestamp,
        )
        timestamps = _extract_timestamps(
            sorted_group,
            timestamp_column=resolved_timestamp,
        )
        target = _extract_target_values(sorted_group, target_column=resolved_target)
        resolved_series_freq = _extract_freq(
            sorted_group,
            freq_column=resolved_freq,
            timestamps=timestamps,
        )

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

    return series_inputs


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
        return pd.read_csv(path)
    return _read_parquet(path)


def _read_frame_from_bytes(*, payload: bytes, tabular_format: TabularFormat) -> pd.DataFrame:
    if tabular_format == "csv":
        text = payload.decode("utf-8-sig")
        return pd.read_csv(StringIO(text))
    return _read_parquet(BytesIO(payload))


def _load_series_inputs_from_remote_url(
    data_url: str,
    *,
    format_hint: TabularFormat | None,
    timestamp_column: str | None,
    series_id_column: str | None,
    target_column: str | None,
    freq_column: str | None,
) -> list[SeriesInput]:
    tabular_format = _resolve_format(path=Path(urlparse(data_url).path), format_hint=format_hint)
    if tabular_format == "csv":
        frame = pd.read_csv(data_url)
    else:
        frame = _read_parquet(data_url)
    return series_inputs_from_frame(
        frame,
        timestamp_column=timestamp_column,
        series_id_column=series_id_column,
        target_column=target_column,
        freq_column=freq_column,
    )


def _read_parquet(path_or_buffer: Any) -> pd.DataFrame:
    try:
        return pd.read_parquet(path_or_buffer)
    except ImportError as exc:
        raise IngestDependencyError(
            "parquet support requires optional dependency (pyarrow or fastparquet); "
            "install with `pip install -e \".[ingest]\"`",
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

    excluded = {
        column
        for column in (timestamp_column, series_id_column, freq_column)
        if column is not None
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
    return None


def _sort_group(group: pd.DataFrame, *, timestamp_column: str | None) -> pd.DataFrame:
    if timestamp_column is not None:
        return group.sort_values(by=timestamp_column, kind="stable")
    return group.sort_index(kind="stable")


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
    freq_column: str | None,
    timestamps: Sequence[str],
) -> str:
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
        return "auto"
    return str(inferred)


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
