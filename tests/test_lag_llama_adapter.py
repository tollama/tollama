"""Unit tests for Lag-Llama adapter helper behavior without network access."""

from __future__ import annotations

import pytest

import tollama.runners.lag_llama_runner.adapter as lag_adapter
from tollama.runners.lag_llama_runner.adapter import (
    build_covariate_warnings,
    build_pandas_dataset,
    build_quantile_payload,
    create_lag_llama_estimator,
    resolve_positive_int,
)
from tollama.runners.lag_llama_runner.errors import AdapterInputError


class _Forecast:
    def quantile(self, value: float):
        if value == 0.1:
            return [1.0, 2.0]
        if value == 0.9:
            return [3.0, 4.0]
        raise KeyError(value)


class _NoQuantileForecast:
    mean = [1.0, 2.0]


def test_resolve_positive_int_validates_option() -> None:
    assert resolve_positive_int(option_value=None, default_value=7, field_name="x") == 7
    with pytest.raises(AdapterInputError, match="must be an integer"):
        resolve_positive_int(option_value="3", default_value=7, field_name="x")


def test_build_quantile_payload_maps_requested_quantiles() -> None:
    payload = build_quantile_payload(
        forecast=_Forecast(),
        requested_quantiles=[0.1, 0.9],
        horizon=2,
    )
    assert payload == {"0.1": [1.0, 2.0], "0.9": [3.0, 4.0]}


def test_build_quantile_payload_requires_quantile_method() -> None:
    with pytest.raises(AdapterInputError, match="no quantile method"):
        build_quantile_payload(
            forecast=_NoQuantileForecast(),
            requested_quantiles=[0.5],
            horizon=2,
        )


class _EstimatorV1:
    def __init__(
        self,
        ckpt_path: str,
        prediction_length: int,
        num_parallel_samples: int,
        **kwargs: object,
    ) -> None:
        self.args = {
            "ckpt_path": ckpt_path,
            "prediction_length": prediction_length,
            "num_parallel_samples": num_parallel_samples,
            "kwargs": kwargs,
        }


def test_create_lag_llama_estimator_uses_num_parallel_samples_fallback() -> None:
    estimator = create_lag_llama_estimator(
        estimator_cls=_EstimatorV1,
        ckpt_path="/tmp/model.ckpt",
        prediction_length=12,
        context_length=8,
        num_samples=100,
    )
    assert estimator.args["num_parallel_samples"] == 100
    assert estimator.args["prediction_length"] == 12
    assert estimator.args["ckpt_path"] == "/tmp/model.ckpt"


def test_create_lag_llama_estimator_applies_checkpoint_architecture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        lag_adapter,
        "load_checkpoint_hparams",
        lambda _: {
            "context_length": 32,
            "model_kwargs": {
                "context_length": 32,
                "max_context_length": 2048,
                "n_layer": 8,
                "n_embd_per_head": 16,
                "n_head": 9,
                "time_feat": True,
            },
        },
    )

    estimator = create_lag_llama_estimator(
        estimator_cls=_EstimatorV1,
        ckpt_path="/tmp/model.ckpt",
        prediction_length=12,
        context_length=256,
        num_samples=100,
    )

    assert estimator.args["kwargs"]["context_length"] == 32
    assert estimator.args["kwargs"]["max_context_length"] == 2048
    assert estimator.args["kwargs"]["n_layer"] == 8
    assert estimator.args["kwargs"]["n_embd_per_head"] == 16
    assert estimator.args["kwargs"]["n_head"] == 9
    assert estimator.args["kwargs"]["time_feat"] is True


class _EstimatorV0:
    def __init__(self, ckpt_path: str, **kwargs: object) -> None:
        raise TypeError("incompatible")


def test_create_lag_llama_estimator_reports_incompatible_signature() -> None:
    with pytest.raises(AdapterInputError, match="constructor signature is incompatible"):
        create_lag_llama_estimator(
            estimator_cls=_EstimatorV0,
            ckpt_path="/tmp/model.ckpt",
            prediction_length=12,
            context_length=8,
            num_samples=100,
        )


def test_build_covariate_warnings_when_covariates_supplied() -> None:
    class _Series:
        past_covariates = {"x": [1.0, 2.0]}
        future_covariates = None
        static_covariates = None

    warnings = build_covariate_warnings([_Series()])
    assert warnings
    assert "ignores covariates" in warnings[0]


def test_build_pandas_dataset_sets_freq() -> None:
    pd = pytest.importorskip("pandas")

    class _Dataset:
        def __init__(self, frames, target, freq):
            self.frames = frames
            self.target = target
            self.freq = freq

    class _Series:
        def __init__(self, series_id: str, freq: str):
            self.id = series_id
            self.freq = freq
            self.timestamps = ["2025-01-01", "2025-01-02", "2025-01-03"]
            self.target = [1.0, 2.0, 3.0]

    dataset, _ = build_pandas_dataset(
        series_list=[_Series("s1", "D")],
        pandas_module=pd,
        pandas_dataset_cls=_Dataset,
    )

    assert dataset.target == "target"
    assert dataset.freq == "D"


def test_build_pandas_dataset_rejects_mixed_freq() -> None:
    pd = pytest.importorskip("pandas")

    class _Dataset:
        def __init__(self, frames, target, freq):
            self.frames = frames
            self.target = target
            self.freq = freq

    class _Series:
        def __init__(self, series_id: str, freq: str):
            self.id = series_id
            self.freq = freq
            self.timestamps = ["2025-01-01", "2025-01-02", "2025-01-03"]
            self.target = [1.0, 2.0, 3.0]

    with pytest.raises(AdapterInputError, match="same frequency"):
        build_pandas_dataset(
            series_list=[_Series("s1", "D"), _Series("s2", "H")],
            pandas_module=pd,
            pandas_dataset_cls=_Dataset,
        )
