"""Unit tests for Chronos adapter covariate frame wiring without model downloads."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from tollama.core.schemas import ForecastRequest, SeriesForecast
from tollama.runners.torch_runner.chronos_adapter import ChronosAdapter, _ChronosDependencies


class _FakeDataFrame:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = [dict(row) for row in rows]
        self.columns: list[str] = []
        seen: set[str] = set()
        for row in self.rows:
            for key in row:
                if key not in seen:
                    self.columns.append(key)
                    seen.add(key)

    def __len__(self) -> int:
        return len(self.rows)


class _FakePandas:
    DataFrame = _FakeDataFrame

    @staticmethod
    def to_datetime(values: list[str], *, utc: bool, errors: str) -> list[datetime]:
        assert utc is True
        assert errors == "raise"
        return [datetime.fromisoformat(value).replace(tzinfo=UTC) for value in values]

    @staticmethod
    def date_range(*, start: datetime, periods: int, freq: str) -> list[datetime]:
        if freq != "D":
            raise ValueError("unsupported frequency")
        return [start + timedelta(days=offset) for offset in range(periods)]


@dataclass
class _CapturingPipeline:
    context_df: _FakeDataFrame | None = None
    future_df: _FakeDataFrame | None = None

    def predict_df(self, context_df: Any, **kwargs: Any) -> dict[str, Any]:
        self.context_df = context_df
        self.future_df = kwargs.get("future_df")
        return {"ok": True}


def _request() -> ForecastRequest:
    return ForecastRequest.model_validate(
        {
            "model": "chronos2",
            "horizon": 2,
            "quantiles": [],
            "series": [
                {
                    "id": "s1",
                    "freq": "D",
                    "timestamps": ["2025-01-01", "2025-01-02"],
                    "target": [100.0, 101.0],
                    "past_covariates": {
                        "promo": [0.0, 1.0],
                        "event": ["off", "on"],
                        "temperature": [22.0, 21.5],
                    },
                    "future_covariates": {
                        "promo": [1.0, 1.0],
                        "event": ["on", "off"],
                    },
                }
            ],
            "options": {},
        },
    )


def test_chronos_adapter_builds_context_and_future_frames_for_known_future_covariates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = ChronosAdapter()
    pipeline = _CapturingPipeline()

    monkeypatch.setattr(
        adapter,
        "_resolve_dependencies",
        lambda: _ChronosDependencies(chronos_pipeline=None, pandas=_FakePandas()),
    )
    monkeypatch.setattr(adapter, "load", lambda *args, **kwargs: None)
    adapter._pipelines["chronos2"] = pipeline  # noqa: SLF001
    monkeypatch.setattr(
        "tollama.runners.torch_runner.chronos_adapter._response_forecasts_from_pred_df",
        lambda **kwargs: [
            SeriesForecast(
                id="s1",
                freq="D",
                start_timestamp="2025-01-03T00:00:00Z",
                mean=[1.0, 1.0],
                quantiles=None,
            )
        ],
    )

    response = adapter.forecast(_request())

    assert response.model == "chronos2"
    assert len(response.forecasts) == 1
    assert pipeline.context_df is not None
    assert pipeline.future_df is not None
    assert len(pipeline.context_df.rows) == 2
    assert len(pipeline.future_df.rows) == 2

    context_keys = set(pipeline.context_df.rows[0])
    future_keys = set(pipeline.future_df.rows[0])
    assert {"id", "timestamp", "target", "promo", "event", "temperature"} <= context_keys
    assert {"id", "timestamp", "promo", "event"} <= future_keys
    assert "temperature" not in future_keys
