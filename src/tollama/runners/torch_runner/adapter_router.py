"""Torch-runner adapter routing by implementation/model metadata."""

from __future__ import annotations

from typing import Any

from tollama.core.schemas import ForecastRequest, ForecastResponse

from .chronos_adapter import ChronosAdapter
from .errors import UnsupportedModelError
from .granite_ttm_adapter import GraniteTTMAdapter

_GRANITE_REPO_MARKER = "granite-timeseries-ttm"


class TorchAdapterRouter:
    """Route torch-family requests to concrete implementation adapters."""

    def __init__(
        self,
        *,
        chronos_adapter: ChronosAdapter | None = None,
        granite_adapter: GraniteTTMAdapter | None = None,
    ) -> None:
        self._adapters: dict[str, Any] = {
            "chronos2": chronos_adapter or ChronosAdapter(),
            "granite_ttm": granite_adapter or GraniteTTMAdapter(),
        }
        self._loaded_model_impl: dict[str, str] = {}

    def resolve_implementation(
        self,
        *,
        model_name: str,
        model_source: dict[str, Any] | None,
        model_metadata: dict[str, Any] | None,
    ) -> str:
        implementation = _dict_str(model_metadata, "implementation")
        if implementation is not None:
            return implementation

        repo_id = _dict_str(model_source, "repo_id")
        if repo_id is not None and _GRANITE_REPO_MARKER in repo_id:
            return "granite_ttm"

        if model_name == "chronos2":
            return "chronos2"

        return "chronos2"

    def load(
        self,
        model_name: str,
        *,
        model_local_dir: str | None,
        model_source: dict[str, Any] | None,
        model_metadata: dict[str, Any] | None,
    ) -> None:
        implementation = self.resolve_implementation(
            model_name=model_name,
            model_source=model_source,
            model_metadata=model_metadata,
        )
        adapter = self._adapter_for_implementation(implementation)
        adapter.load(
            model_name,
            model_local_dir=model_local_dir,
            model_source=model_source,
            model_metadata=model_metadata,
        )
        self._loaded_model_impl[model_name] = implementation

    def unload(self, model_name: str | None = None) -> None:
        if model_name is None:
            for adapter in self._adapters.values():
                adapter.unload(None)
            self._loaded_model_impl.clear()
            return

        implementation = self._loaded_model_impl.pop(model_name, None)
        if implementation is not None:
            adapter = self._adapters.get(implementation)
            if adapter is not None:
                adapter.unload(model_name)
                return

        for adapter in self._adapters.values():
            adapter.unload(model_name)

    def forecast(
        self,
        request: ForecastRequest,
        *,
        model_local_dir: str | None,
        model_source: dict[str, Any] | None,
        model_metadata: dict[str, Any] | None,
    ) -> ForecastResponse:
        implementation = self.resolve_implementation(
            model_name=request.model,
            model_source=model_source,
            model_metadata=model_metadata,
        )
        adapter = self._adapter_for_implementation(implementation)
        response = adapter.forecast(
            request,
            model_local_dir=model_local_dir,
            model_source=model_source,
            model_metadata=model_metadata,
        )
        self._loaded_model_impl[request.model] = implementation
        return response

    def _adapter_for_implementation(self, implementation: str) -> Any:
        adapter = self._adapters.get(implementation)
        if adapter is not None:
            return adapter
        supported = ", ".join(sorted(self._adapters))
        raise UnsupportedModelError(
            f"unsupported torch runner implementation {implementation!r}; "
            f"supported implementations: {supported}",
        )


def _dict_str(payload: dict[str, Any] | None, key: str) -> str | None:
    if not isinstance(payload, dict):
        return None
    value = payload.get(key)
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    if not normalized:
        return None
    return normalized
