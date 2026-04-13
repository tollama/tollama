"""Amazon S3 data connector for tollama (CSV/Parquet objects)."""

from __future__ import annotations

import csv
import io
import logging
from typing import Any

from .base import ConnectorConfig, DataConnector, SeriesChunk
from .registry import register_connector

logger = logging.getLogger(__name__)


class S3Connector(DataConnector):
    """Connector for reading time series from S3 CSV/Parquet objects."""

    name = "s3"

    def __init__(self) -> None:
        self._s3: Any = None
        self._bucket: str = ""
        self._prefix: str = ""
        self._ts_col: str = "timestamp"
        self._val_col: str = "value"
        self._id_col: str = "series_id"
        self._format: str = "csv"

    def connect(self, config: ConnectorConfig) -> None:
        try:
            import boto3
        except ImportError as exc:
            raise ImportError(
                "boto3 is required for S3 connector. Install with: pip install boto3"
            ) from exc

        session_kwargs: dict[str, Any] = {}
        if config.params.get("aws_access_key_id"):
            session_kwargs["aws_access_key_id"] = config.params["aws_access_key_id"]
        if config.params.get("aws_secret_access_key"):
            session_kwargs["aws_secret_access_key"] = config.params["aws_secret_access_key"]
        if config.params.get("region_name"):
            session_kwargs["region_name"] = config.params["region_name"]

        endpoint_url = config.params.get("endpoint_url")
        self._s3 = boto3.client("s3", endpoint_url=endpoint_url, **session_kwargs)
        self._bucket = config.params.get("bucket", "")
        self._prefix = config.params.get("prefix", "")
        self._ts_col = config.params.get("timestamp_column", self._ts_col)
        self._val_col = config.params.get("value_column", self._val_col)
        self._id_col = config.params.get("id_column", self._id_col)
        self._format = config.params.get("format", "csv")
        logger.info("connected to S3 bucket %s", self._bucket)

    def query_series(
        self,
        *,
        series_id: str | None = None,
        start: str | None = None,
        end: str | None = None,
        limit: int | None = None,
        **kwargs: Any,
    ) -> list[SeriesChunk]:
        if self._s3 is None:
            raise RuntimeError("not connected")

        key = kwargs.get("key", "")
        if not key:
            # List first file matching prefix
            resp = self._s3.list_objects_v2(
                Bucket=self._bucket,
                Prefix=self._prefix,
                MaxKeys=1,
            )
            contents = resp.get("Contents", [])
            if not contents:
                return []
            key = contents[0]["Key"]

        obj = self._s3.get_object(Bucket=self._bucket, Key=key)
        body = obj["Body"].read()

        if self._format == "parquet" or key.endswith(".parquet"):
            return self._parse_parquet(body, series_id=series_id, limit=limit)
        return self._parse_csv(body, series_id=series_id, limit=limit)

    def list_series(self) -> list[str]:
        if self._s3 is None:
            raise RuntimeError("not connected")

        resp = self._s3.list_objects_v2(Bucket=self._bucket, Prefix=self._prefix)
        return [obj["Key"] for obj in resp.get("Contents", [])]

    def close(self) -> None:
        self._s3 = None

    def _parse_csv(
        self,
        data: bytes,
        *,
        series_id: str | None = None,
        limit: int | None = None,
    ) -> list[SeriesChunk]:
        reader = csv.DictReader(io.StringIO(data.decode("utf-8")))
        by_id: dict[str, tuple[list[str], list[float]]] = {}
        count = 0
        for row in reader:
            sid = row.get(self._id_col, "default")
            if series_id is not None and sid != series_id:
                continue
            if sid not in by_id:
                by_id[sid] = ([], [])
            by_id[sid][0].append(row.get(self._ts_col, ""))
            by_id[sid][1].append(float(row.get(self._val_col, 0)))
            count += 1
            if limit is not None and count >= limit:
                break

        return [
            SeriesChunk(id=sid, timestamps=ts, values=vals) for sid, (ts, vals) in by_id.items()
        ]

    def _parse_parquet(
        self,
        data: bytes,
        *,
        series_id: str | None = None,
        limit: int | None = None,
    ) -> list[SeriesChunk]:
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise ImportError(
                "pyarrow is required for Parquet support. Install with: pip install pyarrow"
            ) from exc

        table = pq.read_table(io.BytesIO(data))
        df = table.to_pandas()

        if series_id is not None and self._id_col in df.columns:
            df = df[df[self._id_col] == series_id]
        if limit is not None:
            df = df.head(limit)

        by_id: dict[str, tuple[list[str], list[float]]] = {}
        for _, row in df.iterrows():
            sid = str(row.get(self._id_col, "default"))
            if sid not in by_id:
                by_id[sid] = ([], [])
            by_id[sid][0].append(str(row.get(self._ts_col, "")))
            by_id[sid][1].append(float(row.get(self._val_col, 0)))

        return [
            SeriesChunk(id=sid, timestamps=ts, values=vals) for sid, (ts, vals) in by_id.items()
        ]


register_connector("s3", S3Connector)
