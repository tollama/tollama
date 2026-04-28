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
    fake_torch.ones = lambda shape: {"shape": shape}
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

    class _NewDynamicCache:
        def get_seq_length(self, layer_idx: int = 0) -> int:
            assert layer_idx == 0
            return 3

        def get_max_cache_shape(self, layer_idx: int = 0) -> int:
            assert layer_idx == 0
            return -1

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
        config = types.SimpleNamespace(input_token_len=1)

        def eval(self) -> None:
            return None

        def generate(self, input_tensor, max_new_tokens: int, **kwargs):
            captured["max_new_tokens"] = max_new_tokens
            captured["attention_mask"] = kwargs.get("attention_mask")
            cache = _NewDynamicCache()
            captured["seen_tokens"] = cache.seen_tokens
            captured["max_length"] = cache.get_max_length()
            captured["usable_length"] = cache.get_usable_length(2)
            captured["legacy_cache_type"] = type(cache).from_legacy_cache().__class__.__name__
            captured["legacy_cache"] = cache.to_legacy_cache()
            return _FakeOutput()

    class _AutoModelForCausalLM:
        @staticmethod
        def from_pretrained(repo_id: str, **kwargs):
            captured["repo_id"] = repo_id
            captured.update(kwargs)
            return _FakeModel()

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoModelForCausalLM = _AutoModelForCausalLM
    fake_cache_utils = types.ModuleType("transformers.cache_utils")
    fake_cache_utils.DynamicCache = _NewDynamicCache
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "transformers.cache_utils", fake_cache_utils)
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
    assert captured["attention_mask"] == {"shape": (1, 3)}
    assert captured["seen_tokens"] == 3
    assert captured["max_length"] is None
    assert captured["usable_length"] == 3
    assert captured["legacy_cache_type"] == "_NewDynamicCache"
    assert captured["legacy_cache"] == ()


def test_timer_prefers_direct_prediction_and_trims_to_token_boundary(monkeypatch) -> None:
    _install_fake_torch(monkeypatch)
    captured: dict[str, object] = {}

    class _FakeLogits:
        def tolist(self) -> list[list[float]]:
            return [[10.0, 11.0, 12.0]]

    class _FakeModel:
        config = types.SimpleNamespace(input_token_len=4)

        def eval(self) -> None:
            return None

        def __call__(self, input_tensor, max_output_length: int, revin: bool):
            captured["input_tensor"] = input_tensor
            captured["max_output_length"] = max_output_length
            captured["revin"] = revin
            return types.SimpleNamespace(logits=_FakeLogits())

        def generate(self, *args, **kwargs):  # pragma: no cover - should not be used
            raise AssertionError("direct Timer prediction should avoid generate()")

    class _AutoModelForCausalLM:
        @staticmethod
        def from_pretrained(repo_id: str, **kwargs):
            captured["repo_id"] = repo_id
            captured.update(kwargs)
            return _FakeModel()

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoModelForCausalLM = _AutoModelForCausalLM
    fake_cache_utils = types.ModuleType("transformers.cache_utils")
    fake_cache_utils.DynamicCache = type("DynamicCache", (), {})
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "transformers.cache_utils", fake_cache_utils)
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
        horizon=3,
        series=[
            SeriesInput(
                id="series-1",
                freq="D",
                timestamps=["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-04", "2025-01-05"],
                target=[1.0, 2.0, 3.0, 4.0, 5.0],
            )
        ],
    )

    response = adapter.forecast(
        request,
        model_source={"repo_id": "org/timer", "revision": "def456"},
    )

    assert response.forecasts[0].mean == [10.0, 11.0, 12.0]
    assert response.warnings == [
        "series 'series-1': truncated to last 4 points to match Timer input_token_len 4"
    ]
    assert captured["input_tensor"] == [[2.0, 3.0, 4.0, 5.0]]
    assert captured["max_output_length"] == 3
    assert captured["revin"] is True


def test_timer_handles_2880_context_with_30_step_horizon(monkeypatch) -> None:
    _install_fake_torch(monkeypatch)
    captured: dict[str, object] = {}

    class _FakeLogits:
        def tolist(self) -> list[list[float]]:
            return [[float(i) for i in range(30)]]

    class _FakeModel:
        config = types.SimpleNamespace(input_token_len=96)

        def eval(self) -> None:
            return None

        def __call__(self, input_tensor, max_output_length: int, revin: bool):
            captured["context_length"] = len(input_tensor[0])
            captured["max_output_length"] = max_output_length
            captured["revin"] = revin
            return types.SimpleNamespace(logits=_FakeLogits())

        def generate(self, *args, **kwargs):  # pragma: no cover - should not be used
            raise AssertionError("direct Timer prediction should avoid generate()")

    class _AutoModelForCausalLM:
        @staticmethod
        def from_pretrained(repo_id: str, **kwargs):
            del repo_id, kwargs
            return _FakeModel()

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoModelForCausalLM = _AutoModelForCausalLM
    fake_cache_utils = types.ModuleType("transformers.cache_utils")
    fake_cache_utils.DynamicCache = type("DynamicCache", (), {})
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "transformers.cache_utils", fake_cache_utils)

    adapter = TimerAdapter()
    request = ForecastRequest(
        model="timer-base",
        horizon=30,
        series=[
            SeriesInput(
                id="series_0",
                freq="h",
                timestamps=[f"2025-01-{(i % 28) + 1:02d}" for i in range(2880)],
                target=[float(i) for i in range(2880)],
            )
        ],
    )

    response = adapter.forecast(request)

    assert len(response.forecasts[0].mean) == 30
    assert response.warnings is None
    assert captured == {
        "context_length": 2880,
        "max_output_length": 30,
        "revin": True,
    }
