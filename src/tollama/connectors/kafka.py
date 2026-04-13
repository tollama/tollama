"""Kafka streaming data connector for tollama."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from typing import Any

from .base import ConnectorConfig, DataConnector, SeriesChunk
from .registry import register_connector

logger = logging.getLogger(__name__)


class KafkaConnector(DataConnector):
    """Connector for consuming time series from Kafka topics.

    Expects JSON messages with timestamp, value, and optional series_id fields.
    """

    name = "kafka"

    def __init__(self) -> None:
        self._consumer: Any = None
        self._topic: str = ""
        self._ts_key: str = "timestamp"
        self._val_key: str = "value"
        self._id_key: str = "series_id"

    def connect(self, config: ConnectorConfig) -> None:
        try:
            from kafka import KafkaConsumer
        except ImportError as exc:
            raise ImportError(
                "kafka-python is required for Kafka connector. "
                "Install with: pip install kafka-python"
            ) from exc

        bootstrap_servers = config.params.get("bootstrap_servers", "localhost:9092")
        self._topic = config.params.get("topic", "timeseries")
        group_id = config.params.get("group_id", "tollama-connector")
        self._ts_key = config.params.get("timestamp_key", self._ts_key)
        self._val_key = config.params.get("value_key", self._val_key)
        self._id_key = config.params.get("id_key", self._id_key)

        consumer_kwargs: dict[str, Any] = {
            "bootstrap_servers": bootstrap_servers,
            "group_id": group_id,
            "auto_offset_reset": config.params.get("auto_offset_reset", "earliest"),
            "enable_auto_commit": config.params.get("enable_auto_commit", True),
            "value_deserializer": lambda m: json.loads(m.decode("utf-8")),
            "consumer_timeout_ms": config.params.get("consumer_timeout_ms", 5000),
        }

        if config.connection_string:
            consumer_kwargs["bootstrap_servers"] = config.connection_string

        self._consumer = KafkaConsumer(self._topic, **consumer_kwargs)
        logger.info("connected to Kafka topic %s at %s", self._topic, bootstrap_servers)

    def query_series(
        self,
        *,
        series_id: str | None = None,
        start: str | None = None,
        end: str | None = None,
        limit: int | None = None,
        **kwargs: Any,
    ) -> list[SeriesChunk]:
        """Consume messages from the topic and return as series chunks."""
        if self._consumer is None:
            raise RuntimeError("not connected")

        by_id: dict[str, tuple[list[str], list[float]]] = {}
        count = 0

        for message in self._consumer:
            data = message.value
            if not isinstance(data, dict):
                continue

            sid = str(data.get(self._id_key, "default"))
            if series_id is not None and sid != series_id:
                continue

            ts = str(data.get(self._ts_key, ""))
            try:
                val = float(data.get(self._val_key, 0))
            except (TypeError, ValueError):
                continue

            if sid not in by_id:
                by_id[sid] = ([], [])
            by_id[sid][0].append(ts)
            by_id[sid][1].append(val)
            count += 1

            if limit is not None and count >= limit:
                break

        return [
            SeriesChunk(id=sid, timestamps=ts, values=vals) for sid, (ts, vals) in by_id.items()
        ]

    def stream_series(
        self,
        *,
        series_id: str | None = None,
        batch_size: int = 1000,
        **kwargs: Any,
    ) -> Iterator[SeriesChunk]:
        """Yield series chunks in batches from the Kafka stream."""
        if self._consumer is None:
            raise RuntimeError("not connected")

        buffer: dict[str, tuple[list[str], list[float]]] = {}
        count = 0

        for message in self._consumer:
            data = message.value
            if not isinstance(data, dict):
                continue

            sid = str(data.get(self._id_key, "default"))
            if series_id is not None and sid != series_id:
                continue

            ts = str(data.get(self._ts_key, ""))
            try:
                val = float(data.get(self._val_key, 0))
            except (TypeError, ValueError):
                continue

            if sid not in buffer:
                buffer[sid] = ([], [])
            buffer[sid][0].append(ts)
            buffer[sid][1].append(val)
            count += 1

            if count >= batch_size:
                for bid, (bts, bvals) in buffer.items():
                    yield SeriesChunk(id=bid, timestamps=bts, values=bvals)
                buffer.clear()
                count = 0

        # Flush remaining
        for bid, (bts, bvals) in buffer.items():
            yield SeriesChunk(id=bid, timestamps=bts, values=bvals)

    def list_series(self) -> list[str]:
        if self._consumer is None:
            raise RuntimeError("not connected")
        return [self._topic]

    def close(self) -> None:
        if self._consumer is not None:
            self._consumer.close()
            self._consumer = None


register_connector("kafka", KafkaConnector)
