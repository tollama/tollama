"""InfluxDB 2.x data connector for tollama."""

from __future__ import annotations

import logging
from typing import Any

from .base import ConnectorConfig, DataConnector, SeriesChunk
from .registry import register_connector

logger = logging.getLogger(__name__)


class InfluxDBConnector(DataConnector):
    """Connector for InfluxDB 2.x via the influxdb-client library."""

    name = "influxdb"

    def __init__(self) -> None:
        self._client: Any = None
        self._query_api: Any = None
        self._org: str = ""
        self._bucket: str = ""
        self._measurement: str = ""

    def connect(self, config: ConnectorConfig) -> None:
        try:
            from influxdb_client import InfluxDBClient
        except ImportError as exc:
            raise ImportError(
                "influxdb-client is required for InfluxDB connector. "
                "Install with: pip install influxdb-client"
            ) from exc

        url = config.params.get("url", "http://localhost:8086")
        token = config.params.get("token", "")
        self._org = config.params.get("org", "")
        self._bucket = config.params.get("bucket", "")
        self._measurement = config.params.get("measurement", "")

        if config.connection_string:
            url = config.connection_string

        self._client = InfluxDBClient(url=url, token=token, org=self._org)
        self._query_api = self._client.query_api()
        logger.info("connected to InfluxDB at %s", url)

    def query_series(
        self,
        *,
        series_id: str | None = None,
        start: str | None = None,
        end: str | None = None,
        limit: int | None = None,
        **kwargs: Any,
    ) -> list[SeriesChunk]:
        if self._query_api is None:
            raise RuntimeError("not connected")

        range_start = start or "-30d"
        range_stop = end or "now()"
        field_name = kwargs.get("field", "_value")

        flux = f'from(bucket: "{self._bucket}")'
        flux += f" |> range(start: {range_start}, stop: {range_stop})"

        if self._measurement:
            flux += f' |> filter(fn: (r) => r._measurement == "{self._measurement}")'
        if series_id is not None:
            flux += f' |> filter(fn: (r) => r.series_id == "{series_id}")'
        flux += f' |> filter(fn: (r) => r._field == "{field_name}")'

        if limit is not None:
            flux += f" |> limit(n: {int(limit)})"

        tables = self._query_api.query(flux, org=self._org)

        chunks: list[SeriesChunk] = []
        for table in tables:
            timestamps: list[str] = []
            values: list[float] = []
            sid = ""
            for record in table.records:
                timestamps.append(record.get_time().isoformat())
                values.append(float(record.get_value()))
                if not sid:
                    sid = str(record.values.get("series_id", record.get_measurement()))

            if timestamps:
                chunks.append(SeriesChunk(id=sid, timestamps=timestamps, values=values))

        return chunks

    def list_series(self) -> list[str]:
        if self._query_api is None:
            raise RuntimeError("not connected")

        flux = (
            f'import "influxdata/influxdb/schema"\n'
            f'schema.tagValues(bucket: "{self._bucket}", tag: "series_id")'
        )
        tables = self._query_api.query(flux, org=self._org)
        ids: list[str] = []
        for table in tables:
            for record in table.records:
                ids.append(str(record.get_value()))
        return ids

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None
            self._query_api = None


register_connector("influxdb", InfluxDBConnector)
