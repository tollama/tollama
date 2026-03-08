from __future__ import annotations

import time
from typing import Any

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
    def to_datetime(
        values: list[str],
        utc: bool = True,
        errors: str = "raise",
    ) -> list[_FakeDatetime]:
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
        return {
            "mean": [10.0, 11.0, 12.0],
            "quantiles": {
                "0.1": [9.0, 10.0, 11.0],
                "0.9": [11.0, 12.0, 13.0],
            },
        }


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
        lambda: type(
            "D",
            (),
            {"torch": _FakeTorch(), "pandas": _FakePandas(), "model_loader_cls": _FakeModel},
        )(),
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


def test_patchtst_adapter_builds_3d_input_tensor(monkeypatch) -> None:
    adapter = PatchTSTAdapter()
    monkeypatch.setattr(
        adapter,
        "_resolve_dependencies",
        lambda: type(
            "D",
            (),
            {"torch": _FakeTorch(), "pandas": _FakePandas(), "model_loader_cls": _FakeModel},
        )(),
    )

    req = _request()
    req.series = [req.series[0]]
    req.options["context_length"] = 3
    adapter.forecast(req)

    model = adapter._model_cache[("patchtst", "ibm-granite/granite-timeseries-patchtst", "main")]
    payload = model.calls[0]["past_values"].values
    assert len(payload) == 1
    assert len(payload[0]) == 3
    assert len(payload[0][0]) == 1


def test_patchtst_adapter_rejects_invalid_frequency(monkeypatch) -> None:
    adapter = PatchTSTAdapter()
    monkeypatch.setattr(
        adapter,
        "_resolve_dependencies",
        lambda: type(
            "D",
            (),
            {"torch": _FakeTorch(), "pandas": _FakePandas(), "model_loader_cls": _FakeModel},
        )(),
    )
    req = _request()
    req.series[0].freq = "BAD"

    try:
        adapter.forecast(req)
        raise AssertionError("expected AdapterInputError")
    except AdapterInputError as exc:
        assert "invalid frequency" in str(exc)


def test_patchtst_adapter_reuses_cached_model_for_repeated_requests(monkeypatch) -> None:
    load_count = 0

    class _SlowFakeModel(_FakeModel):
        @classmethod
        def from_pretrained(cls, *args: Any, **kwargs: Any) -> _SlowFakeModel:
            nonlocal load_count
            del args, kwargs
            load_count += 1
            time.sleep(0.03)
            return cls()

    adapter = PatchTSTAdapter()
    monkeypatch.setattr(
        adapter,
        "_resolve_dependencies",
        lambda: type(
            "D",
            (),
            {"torch": _FakeTorch(), "pandas": _FakePandas(), "model_loader_cls": _SlowFakeModel},
        )(),
    )

    req = _request()
    started = time.perf_counter()
    adapter.forecast(req)
    first_elapsed = time.perf_counter() - started

    started = time.perf_counter()
    adapter.forecast(req)
    second_elapsed = time.perf_counter() - started

    assert load_count == 1
    assert second_elapsed < first_elapsed


def test_patchtst_adapter_allows_disabling_cache_per_request(monkeypatch) -> None:
    load_count = 0

    class _CountingModel(_FakeModel):
        @classmethod
        def from_pretrained(cls, *args: Any, **kwargs: Any) -> _CountingModel:
            nonlocal load_count
            del args, kwargs
            load_count += 1
            return cls()

    adapter = PatchTSTAdapter()
    monkeypatch.setattr(
        adapter,
        "_resolve_dependencies",
        lambda: type(
            "D",
            (),
            {"torch": _FakeTorch(), "pandas": _FakePandas(), "model_loader_cls": _CountingModel},
        )(),
    )

    req = _request()
    req.options["cache_reuse"] = False
    adapter.forecast(req)
    adapter.forecast(req)

    assert load_count == 2


def test_patchtst_adapter_enforces_context_length_guardrail(monkeypatch) -> None:
    monkeypatch.setenv("TOLLAMA_PATCHTST_MAX_CONTEXT_LENGTH", "8")
    adapter = PatchTSTAdapter()
    monkeypatch.setattr(
        adapter,
        "_resolve_dependencies",
        lambda: type(
            "D",
            (),
            {"torch": _FakeTorch(), "pandas": _FakePandas(), "model_loader_cls": _FakeModel},
        )(),
    )
    req = _request()
    req.options["context_length"] = 16

    try:
        adapter.forecast(req)
        raise AssertionError("expected AdapterInputError")
    except AdapterInputError as exc:
        assert "context_length exceeds patchtst guardrail" in str(exc)


def test_patchtst_adapter_enforces_series_count_guardrail(monkeypatch) -> None:
    monkeypatch.setenv("TOLLAMA_PATCHTST_MAX_SERIES_PER_REQUEST", "1")
    adapter = PatchTSTAdapter()
    monkeypatch.setattr(
        adapter,
        "_resolve_dependencies",
        lambda: type(
            "D",
            (),
            {"torch": _FakeTorch(), "pandas": _FakePandas(), "model_loader_cls": _FakeModel},
        )(),
    )

    try:
        adapter.forecast(_request())
        raise AssertionError("expected AdapterInputError")
    except AdapterInputError as exc:
        assert "series count exceeds patchtst guardrail" in str(exc)
