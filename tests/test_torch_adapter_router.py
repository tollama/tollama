"""Tests for torch adapter routing by manifest metadata/source heuristics."""

from __future__ import annotations

from typing import Any

from tollama.core.schemas import ForecastRequest, ForecastResponse, SeriesForecast
from tollama.runners.torch_runner.adapter_router import TorchAdapterRouter


class _FakeAdapter:
    def __init__(self, *, label: str) -> None:
        self.label = label
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def load(
        self,
        model_name: str,
        *,
        model_local_dir: str | None,
        model_source: dict[str, Any] | None,
        model_metadata: dict[str, Any] | None,
    ) -> None:
        self.calls.append(
            (
                "load",
                {
                    "model_name": model_name,
                    "model_local_dir": model_local_dir,
                    "model_source": model_source,
                    "model_metadata": model_metadata,
                },
            ),
        )

    def unload(self, model_name: str | None = None) -> None:
        self.calls.append(("unload", {"model_name": model_name}))

    def forecast(
        self,
        request: ForecastRequest,
        *,
        model_local_dir: str | None,
        model_source: dict[str, Any] | None,
        model_metadata: dict[str, Any] | None,
    ) -> ForecastResponse:
        self.calls.append(
            (
                "forecast",
                {
                    "model_name": request.model,
                    "model_local_dir": model_local_dir,
                    "model_source": model_source,
                    "model_metadata": model_metadata,
                },
            ),
        )
        return ForecastResponse(
            model=request.model,
            forecasts=[
                SeriesForecast(
                    id=request.series[0].id,
                    freq=request.series[0].freq,
                    start_timestamp=request.series[0].timestamps[-1],
                    mean=[1.0] * request.horizon,
                ),
            ],
            usage={"adapter": self.label},
        )


def _request(model: str) -> ForecastRequest:
    return ForecastRequest.model_validate(
        {
            "model": model,
            "horizon": 2,
            "quantiles": [],
            "series": [
                {
                    "id": "s1",
                    "freq": "D",
                    "timestamps": ["2025-01-01", "2025-01-02"],
                    "target": [1.0, 2.0],
                }
            ],
            "options": {},
        },
    )


def test_router_uses_metadata_implementation_for_granite() -> None:
    chronos = _FakeAdapter(label="chronos")
    granite = _FakeAdapter(label="granite")
    router = TorchAdapterRouter(chronos_adapter=chronos, granite_adapter=granite)

    response = router.forecast(
        _request("granite-ttm-r2"),
        model_local_dir="/tmp/granite",
        model_source={
            "type": "huggingface",
            "repo_id": "ibm-granite/granite-timeseries-ttm-r2",
            "revision": "90-30-ft-l1-r2.1",
        },
        model_metadata={"implementation": "granite_ttm", "prediction_length": 30},
    )

    assert response.usage == {"adapter": "granite"}
    assert len(chronos.calls) == 0
    assert granite.calls[0][0] == "forecast"


def test_router_falls_back_to_repo_id_heuristic_for_granite() -> None:
    chronos = _FakeAdapter(label="chronos")
    granite = _FakeAdapter(label="granite")
    router = TorchAdapterRouter(chronos_adapter=chronos, granite_adapter=granite)

    response = router.forecast(
        _request("granite-ttm-r2"),
        model_local_dir=None,
        model_source={
            "type": "huggingface",
            "repo_id": "ibm-granite/granite-timeseries-ttm-r2",
            "revision": "90-30-ft-l1-r2.1",
        },
        model_metadata=None,
    )

    assert response.usage == {"adapter": "granite"}
    assert len(chronos.calls) == 0
    assert granite.calls[0][0] == "forecast"
