from __future__ import annotations

import sys
import types

import tollama.runners.timemixer_runner.adapter as timemixer_adapter
import tollama.runners.timer_runner.adapter as timer_adapter
from tollama.core.schemas import ForecastRequest, SeriesInput
from tollama.runners.timemixer_runner.adapter import TimeMixerAdapter
from tollama.runners.timemixer_runner.errors import UnsupportedModelError
from tollama.runners.timer_runner.adapter import TimerAdapter


class _NoGrad:
    def __enter__(self) -> None:
        return None

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


def _install_fake_torch(monkeypatch) -> None:
    fake_torch = types.ModuleType("torch")
    fake_torch.float32 = object()
    fake_torch.tensor = lambda data, dtype=None: data
    fake_torch.no_grad = lambda: _NoGrad()
    monkeypatch.setitem(sys.modules, "torch", fake_torch)


def _series_input() -> SeriesInput:
    return SeriesInput(
        id="series-1",
        freq="D",
        timestamps=["2025-01-01", "2025-01-02", "2025-01-03"],
        target=[1.0, 2.0, 3.0],
    )


def test_timemixer_uses_revision_for_remote_pretrained_load(monkeypatch) -> None:
    _install_fake_torch(monkeypatch)
    captured: dict[str, object] = {}

    class _FakeVector:
        def __init__(self, values: list[float]) -> None:
            self._values = values

        def tolist(self) -> list[float]:
            return self._values

    class _FakeModel:
        def eval(self) -> None:
            return None

        def __call__(self, input_tensor, prediction_length: int):
            return types.SimpleNamespace(predictions=[_FakeVector([1.0] * prediction_length)])

    class _AutoModel:
        @staticmethod
        def from_pretrained(repo_id: str, **kwargs):
            captured["repo_id"] = repo_id
            captured.update(kwargs)
            return _FakeModel()

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoModel = _AutoModel
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setattr(
        timemixer_adapter,
        "SeriesForecast",
        lambda **kwargs: types.SimpleNamespace(**kwargs),
    )
    monkeypatch.setattr(
        timemixer_adapter,
        "ForecastResponse",
        lambda **kwargs: types.SimpleNamespace(**kwargs),
    )

    adapter = TimeMixerAdapter()
    request = ForecastRequest(
        model="timemixer-base",
        horizon=2,
        series=[_series_input()],
    )

    response = adapter.forecast(
        request,
        model_source={"repo_id": "org/timemixer", "revision": "abc123"},
    )

    assert response.forecasts[0].mean == [1.0, 1.0]
    assert captured["repo_id"] == "org/timemixer"
    assert captured["revision"] == "abc123"
    assert captured["trust_remote_code"] is True


def test_timemixer_manifest_only_source_raises_clear_error() -> None:
    adapter = TimeMixerAdapter()
    request = ForecastRequest(
        model="timemixer-base",
        horizon=2,
        series=[_series_input()],
    )

    try:
        adapter.forecast(
            request,
            model_local_dir="/tmp/tollama-empty-timemixer",
            model_source={
                "type": "local",
                "repo_id": "tollama/timemixer-runner",
                "revision": "main",
            },
        )
    except UnsupportedModelError as exc:
        assert "manifest-only" in str(exc)
        assert "timer-base" in str(exc)
    else:
        raise AssertionError("expected TimeMixer manifest-only source to be unsupported")


def test_timer_uses_revision_for_remote_pretrained_load(monkeypatch) -> None:
    _install_fake_torch(monkeypatch)
    captured: dict[str, object] = {}

    class _FakeOutput:
        def __getitem__(self, key):
            if isinstance(key, tuple):
                return self
            if isinstance(key, slice):
                return self
            return self

        def tolist(self) -> list[float]:
            return [4.0, 5.0]

    class _FakeModel:
        def eval(self) -> None:
            return None

        def generate(self, input_tensor, max_new_tokens: int):
            captured["max_new_tokens"] = max_new_tokens
            return _FakeOutput()

    class _AutoModelForCausalLM:
        @staticmethod
        def from_pretrained(repo_id: str, **kwargs):
            captured["repo_id"] = repo_id
            captured.update(kwargs)
            return _FakeModel()

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoModelForCausalLM = _AutoModelForCausalLM
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setattr(
        timer_adapter,
        "SeriesForecast",
        lambda **kwargs: types.SimpleNamespace(**kwargs),
    )
    monkeypatch.setattr(
        timer_adapter,
        "ForecastResponse",
        lambda **kwargs: types.SimpleNamespace(**kwargs),
    )

    adapter = TimerAdapter()
    request = ForecastRequest(
        model="timer-base",
        horizon=2,
        series=[_series_input()],
    )

    response = adapter.forecast(
        request,
        model_source={"repo_id": "org/timer", "revision": "def456"},
    )

    assert response.forecasts[0].mean == [4.0, 5.0]
    assert captured["repo_id"] == "org/timer"
    assert captured["revision"] == "def456"
    assert captured["trust_remote_code"] is True
    assert captured["max_new_tokens"] == 2
