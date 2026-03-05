from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import pytest

from tollama.core.schemas import ForecastRequest
from tollama.runners.patchtst_runner.adapter import PatchTSTAdapter
from tollama.runners.patchtst_runner.errors import AdapterInputError


class _FakeTensor:
    def __init__(self, values: list[list[float]]) -> None:
        self.values = values


class _FakeTorch:
    float32 = "float32"

    @staticmethod
    def tensor(values: list[list[float]], dtype: Any = None) -> _FakeTensor:
        del dtype
        return _FakeTensor(values)

    @staticmethod
    def no_grad() -> Any:
        return nullcontext()


class _FakeDate:
    def __init__(self, iso: str) -> None:
        self._iso = iso

    def isoformat(self) -> str:
        return self._iso


class _FakeDatetime:
    def __init__(self, iso: str) -> None:
        self._iso = iso


class _FakePandas:
    @staticmethod
    def to_datetime(values: list[str], utc: bool = True, errors: str = "raise") -> list[_FakeDatetime]:
        del utc, errors
        return [_FakeDatetime(values[0])]

    @staticmethod
    def date_range(start: Any, periods: int, freq: str) -> list[_FakeDate]:
        del start
        if freq != "D":
            raise ValueError("unsupported")
        return [_FakeDate(f"2025-01-{index:02d}T00:00:00+00:00") for index in range(1, periods + 1)]


class _FakeModel:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def eval(self) -> None:
        return

    @classmethod
    def from_pretrained(cls, *args: Any, **kwargs: Any) -> _FakeModel:
        del args, kwargs
        return cls()

    def generate(self, **kwargs: Any) -> dict[str, list[float]]:
        self.calls.append(kwargs)
        return {"mean": [10.0, 11.0, 12.0], "quantiles": {"0.1": [9.0, 10.0, 11.0], "0.9": [11.0, 12.0, 13.0]}}


def _request() -> ForecastRequest:
    return ForecastRequest.model_validate(
        {
            "model": "patchtst",
            "horizon": 2,
            "quantiles": [0.1, 0.9],
            "series": [
                {
                    "id": "s1",
                    "freq": "D",
                    "timestamps": ["2025-01-01", "2025-01-02", "2025-01-03"],
                    "target": [1.0, 2.0, 3.0],
                },
                {
                    "id": "s2",
                    "freq": "D",
                    "timestamps": ["2025-01-01", "2025-01-02", "2025-01-03"],
                    "target": [5.0, 4.0, 3.0],
                    "past_covariates": {"promo": [0, 1, 1]},
                },
            ],
            "options": {"context_length": 2},
        },
    )


def test_patchtst_adapter_forecast_smoke_multi_series(monkeypatch) -> None:
    adapter = PatchTSTAdapter()
    monkeypatch.setattr(
        adapter,
        "_resolve_dependencies",
        lambda: type("D", (), {"torch": _FakeTorch(), "pandas": _FakePandas(), "model_loader_cls": _FakeModel})(),
    )

    response = adapter.forecast(_request())

    assert response.model == "patchtst"
    assert len(response.forecasts) == 2
    assert response.forecasts[0].mean == [10.0, 11.0]
    assert response.forecasts[0].quantiles == {"0.1": [9.0, 10.0], "0.9": [11.0, 12.0]}
    assert response.usage is not None
    assert response.usage["series_count"] == 2
    assert response.warnings is not None
    assert "ignores covariates" in response.warnings[0]

    model = next(iter(adapter._model_cache.values()))
    assert isinstance(model, _FakeModel)
    first_call = model.calls[0]
    assert first_call["past_values"].values == [[2.0, 3.0]]


def test_patchtst_adapter_rejects_invalid_frequency(monkeypatch) -> None:
    adapter = PatchTSTAdapter()
    monkeypatch.setattr(
        adapter,
        "_resolve_dependencies",
        lambda: type("D", (), {"torch": _FakeTorch(), "pandas": _FakePandas(), "model_loader_cls": _FakeModel})(),
    )
    req = _request()
    req.series[0].freq = "BAD"

    with pytest.raises(AdapterInputError, match="invalid frequency"):
        adapter.forecast(req)


def test_patchtst_adapter_allows_whitespace_frequency(monkeypatch) -> None:
    adapter = PatchTSTAdapter()
    monkeypatch.setattr(
        adapter,
        "_resolve_dependencies",
        lambda: type("D", (), {"torch": _FakeTorch(), "pandas": _FakePandas(), "model_loader_cls": _FakeModel})(),
    )
    req = _request()
    req.series[0].freq = " D "
    req.series[1].freq = "D"

    response = adapter.forecast(req)
    assert response.forecasts[0].start_timestamp.startswith("2025-01-")


def test_patchtst_adapter_rejects_non_finite_target(monkeypatch) -> None:
    adapter = PatchTSTAdapter()
    monkeypatch.setattr(
        adapter,
        "_resolve_dependencies",
        lambda: type("D", (), {"torch": _FakeTorch(), "pandas": _FakePandas(), "model_loader_cls": _FakeModel})(),
    )
    req = _request()
    req.series[0].target = [1.0, float("nan"), 3.0]

    with pytest.raises(AdapterInputError, match="non-finite"):
        adapter.forecast(req)
