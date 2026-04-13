"""Connector registry — discover and instantiate data connectors by name."""

from __future__ import annotations

from typing import Any

from .base import ConnectorConfig, DataConnector

_REGISTRY: dict[str, type[DataConnector]] = {}


def register_connector(name: str, cls: type[DataConnector]) -> None:
    """Register a connector class under the given name."""
    _REGISTRY[name] = cls


def get_connector(config: ConnectorConfig) -> DataConnector:
    """Instantiate and connect a data connector from config."""
    cls = _REGISTRY.get(config.backend)
    if cls is None:
        _try_lazy_import(config.backend)
        cls = _REGISTRY.get(config.backend)
    if cls is None:
        available = sorted(_REGISTRY) or ["(none registered)"]
        raise ValueError(
            f"unknown connector backend {config.backend!r}. Available: {', '.join(available)}"
        )
    connector = cls()
    connector.connect(config)
    return connector


def list_connectors() -> list[dict[str, Any]]:
    """List all registered connector backends."""
    # Ensure lazy imports are attempted for known backends
    for backend in ("postgresql", "influxdb", "s3", "kafka"):
        _try_lazy_import(backend)

    return [
        {"name": name, "class": cls.__name__, "module": cls.__module__}
        for name, cls in sorted(_REGISTRY.items())
    ]


def _try_lazy_import(backend: str) -> None:
    """Attempt to import a built-in connector module to trigger registration."""
    if backend in _REGISTRY:
        return
    module_map = {
        "postgresql": "tollama.connectors.postgresql",
        "timescaledb": "tollama.connectors.postgresql",
        "influxdb": "tollama.connectors.influxdb",
        "s3": "tollama.connectors.s3",
        "kafka": "tollama.connectors.kafka",
    }
    module_name = module_map.get(backend)
    if module_name is None:
        return
    try:
        __import__(module_name)
    except ImportError:
        pass
