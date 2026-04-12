"""Tests for real-data run orchestrator helpers and artifact wiring."""

from __future__ import annotations

import json
import sys
import types
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest

_MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "e2e_realdata"
    / "run_tsfm_realdata.py"
)
_MODULE_SPEC = spec_from_file_location("scripts_e2e_realdata_run_tsfm_realdata", _MODULE_PATH)
assert _MODULE_SPEC is not None and _MODULE_SPEC.loader is not None
_MODULE = module_from_spec(_MODULE_SPEC)
sys.modules[_MODULE_SPEC.name] = _MODULE
_MODULE_SPEC.loader.exec_module(_MODULE)

_call_forecast_with_retry = _MODULE._call_forecast_with_retry
_resolve_models = _MODULE._resolve_models
_sanitize_artifact_filename = _MODULE._sanitize_artifact_filename
_scenarios_for_model = _MODULE._scenarios_for_model
_status_for_data_issue = _MODULE._status_for_data_issue
_validate_dataset_catalog = _MODULE._validate_dataset_catalog
main = _MODULE.main


def test_sanitize_artifact_filename_replaces_forbidden_characters() -> None:
    assert _sanitize_artifact_filename("m4_daily:D182") == "m4_daily_D182"
    assert _sanitize_artifact_filename('a"b<c>d|e*f?') == "a_b_c_d_e_f_"
    assert _sanitize_artifact_filename("\n\r") == "series"


def test_status_for_data_issue_skips_hf_only_in_optional_profile() -> None:
    assert _status_for_data_issue(
        gate_profile="hf_optional",
        dataset_kind="huggingface_dataset",
    ) == "skip"
    assert _status_for_data_issue(
        gate_profile="hf_optional",
        dataset_kind="open_m4_daily",
    ) == "fail"
    assert _status_for_data_issue(
        gate_profile="strict",
        dataset_kind="huggingface_dataset",
    ) == "fail"


def test_call_forecast_with_retry_retries_retryable_status(monkeypatch) -> None:
    calls = [
        (502, {"detail": "bad gateway"}, "HTTP 502"),
        (200, {"forecasts": [{"mean": [1.0]}]}, None),
    ]
    sleeps: list[float] = []

    def _fake_once(*, client, payload):  # noqa: ANN202
        del client, payload
        return calls.pop(0)

    monkeypatch.setattr(_MODULE, "_call_forecast_once", _fake_once)

    status, payload, detail, retry_count = _call_forecast_with_retry(
        client=None,
        payload={},
        sleep_fn=lambda seconds: sleeps.append(seconds),
    )

    assert status == 200
    assert isinstance(payload, dict)
    assert detail is None
    assert retry_count == 1
    assert sleeps == [2.0]


def test_call_forecast_with_retry_retries_timeout_like_errors(monkeypatch) -> None:
    calls = [
        (None, None, "request timed out after 900s"),
        (200, {"forecasts": [{"mean": [1.0]}]}, None),
    ]

    monkeypatch.setattr(_MODULE, "_call_forecast_once", lambda *, client, payload: calls.pop(0))

    status, _, _, retry_count = _call_forecast_with_retry(
        client=None,
        payload={},
        sleep_fn=lambda seconds: None,
    )

    assert status == 200
    assert retry_count == 1


def test_call_forecast_with_retry_stops_after_max_attempts(monkeypatch) -> None:
    calls = [
        (503, {"detail": "runner unavailable"}, "HTTP 503"),
        (503, {"detail": "runner unavailable"}, "HTTP 503"),
        (503, {"detail": "runner unavailable"}, "HTTP 503"),
    ]
    sleeps: list[float] = []

    monkeypatch.setattr(_MODULE, "_call_forecast_once", lambda *, client, payload: calls.pop(0))

    status, _, detail, retry_count = _call_forecast_with_retry(
        client=None,
        payload={},
        sleep_fn=lambda seconds: sleeps.append(seconds),
    )

    assert status == 503
    assert detail is not None
    assert retry_count == 2
    assert sleeps == [2.0, 5.0]


def test_resolve_models_preserves_strict_all_and_adds_hf_aliases() -> None:
    assert _resolve_models("all") == [
        "chronos2",
        "granite-ttm-r2",
        "timesfm-2.5-200m",
        "moirai-2.0-R-small",
        "sundial-base-128m",
        "toto-open-base-1.0",
    ]
    assert _resolve_models("hf_all") == [
        "chronos2",
        "granite-ttm-r2",
        "timesfm-2.5-200m",
        "moirai-2.0-R-small",
        "sundial-base-128m",
        "toto-open-base-1.0",
        "lag-llama",
        "patchtst",
        "tide",
        "nhits",
        "nbeatsx",
    ]
    assert _resolve_models("neural") == [
        "lag-llama",
        "patchtst",
        "tide",
        "nhits",
        "nbeatsx",
    ]


def test_scenarios_for_model_limit_benchmark_only_models() -> None:
    assert _scenarios_for_model("patchtst") == ["benchmark_target_only"]
    assert _scenarios_for_model("tide") == ["benchmark_target_only"]
    assert _scenarios_for_model("nhits") == [
        "benchmark_target_only",
        "contract_best_effort_covariates",
        "contract_strict_covariates",
    ]


def test_validate_dataset_catalog_rejects_duplicate_name() -> None:
    with pytest.raises(ValueError, match="duplicate dataset name"):
        _validate_dataset_catalog(
            {
                "datasets": [
                    {
                        "name": "ds",
                        "kind": "huggingface_dataset",
                        "horizon": 24,
                        "hf_id": "a/b",
                        "timestamp_column": "ts",
                        "target_column": "value",
                    },
                    {
                        "name": "ds",
                        "kind": "huggingface_dataset",
                        "horizon": 24,
                        "hf_id": "c/d",
                        "timestamp_column": "ts",
                        "target_column": "value",
                    },
                ]
            },
            require_unique_hf_ids=True,
        )


def test_validate_dataset_catalog_rejects_duplicate_hf_id_for_starter_lane() -> None:
    with pytest.raises(ValueError, match="duplicate hf_id"):
        _validate_dataset_catalog(
            {
                "datasets": [
                    {
                        "name": "ds1",
                        "kind": "huggingface_dataset",
                        "horizon": 24,
                        "hf_id": "a/b",
                        "timestamp_column": "ts",
                        "target_column": "value",
                    },
                    {
                        "name": "ds2",
                        "kind": "huggingface_dataset",
                        "horizon": 24,
                        "hf_id": "a/b",
                        "timestamp_column": "ts",
                        "target_column": "value",
                    },
                ]
            },
            require_unique_hf_ids=True,
        )


def test_validate_dataset_catalog_rejects_missing_fields_and_invalid_horizon() -> None:
    with pytest.raises(ValueError, match="invalid horizon"):
        _validate_dataset_catalog(
            {
                "datasets": [
                    {
                        "name": "ds1",
                        "kind": "huggingface_dataset",
                        "horizon": 0,
                        "hf_id": "a/b",
                        "timestamp_column": "ts",
                        "target_column": "value",
                    }
                ]
            },
            require_unique_hf_ids=True,
        )

    with pytest.raises(ValueError, match="missing target_column"):
        _validate_dataset_catalog(
            {
                "datasets": [
                    {
                        "name": "ds1",
                        "kind": "huggingface_dataset",
                        "horizon": 24,
                        "hf_id": "a/b",
                        "timestamp_column": "ts",
                    }
                ]
            },
            require_unique_hf_ids=True,
        )


def test_main_writes_summary_and_benchmark_artifacts_for_hf_all(
    monkeypatch,
    tmp_path: Path,
) -> None:
    prepare_data_module = types.SimpleNamespace()

    class _FakeClient:
        def __init__(self, *, base_url: str, timeout: float) -> None:
            self.base_url = base_url
            self.timeout = timeout

        def health(self) -> dict[str, object]:
            return {"status": "ok"}

    class _KagglePolicy:
        def __init__(
            self,
            include_kaggle: bool,
            hard_fail_on_missing: bool,
            message: str | None = None,
        ) -> None:
            self.include_kaggle = include_kaggle
            self.hard_fail_on_missing = hard_fail_on_missing
            self.message = message

    class _PreparedDataResult:
        def __init__(self, datasets: dict[str, object], messages: list[str]) -> None:
            self.datasets = datasets
            self.messages = messages

    catalog_path = tmp_path / "hf_dataset_catalog_starter.yaml"
    catalog = {
        "datasets": [
            {
                "name": f"ds{index}",
                "kind": "huggingface_dataset",
                "hf_id": f"org/ds{index}",
                "horizon": 24,
                "timestamp_column": "ts",
                "target_column": "value",
            }
            for index in range(10)
        ]
    }
    catalog_path.write_text(json.dumps(catalog), encoding="utf-8")

    prepared = _PreparedDataResult(
        datasets={
            f"ds{index}": [
                {
                    "id": f"series-{index}",
                    "freq": "H",
                    "timestamps": [f"2026-01-01T{hour:02d}:00:00" for hour in range(24)],
                    "target": [float(hour) for hour in range(24)],
                    "actuals": [1.0, 2.0, 3.0],
                }
            ]
            for index in range(10)
        },
        messages=[],
    )

    monkeypatch.setattr(
        _MODULE,
        "_create_client",
        lambda **kwargs: _FakeClient(**kwargs),
    )
    prepare_data_module.KagglePolicy = _KagglePolicy
    prepare_data_module.PreparedDataResult = _PreparedDataResult
    prepare_data_module.has_kaggle_credentials = lambda: False
    prepare_data_module.kaggle_policy_for_mode = lambda *args, **kwargs: _KagglePolicy(
        include_kaggle=False,
        hard_fail_on_missing=False,
        message="Kaggle credentials missing; local fallback enabled (open datasets only)",
    )
    prepare_data_module.load_dataset_catalog = lambda *args, **kwargs: catalog
    prepare_data_module.prepare_datasets = lambda **kwargs: prepared
    monkeypatch.setattr(_MODULE, "_load_prepare_data_module", lambda: prepare_data_module)
    monkeypatch.setattr(_MODULE, "_pull_model", lambda **kwargs: None)
    monkeypatch.setattr(
        _MODULE,
        "_call_forecast_with_retry",
        lambda **kwargs: (200, {"forecasts": [{"mean": [1.0, 2.0, 3.0]}]}, None, 0),
    )
    monkeypatch.setattr(
        _MODULE.validate_results,
        "evaluate_gate",
        lambda **kwargs: (
            True,
            None,
            [],
            {"mae": 1.0, "rmse": 2.0, "smape": 3.0, "mape": 4.0, "mase": 5.0},
        ),
    )

    output_dir = tmp_path / "artifacts"
    exit_code = main(
        [
            "--mode",
            "local",
            "--model",
            "hf_all",
            "--catalog-path",
            str(catalog_path),
            "--gate-profile",
            "hf_optional",
            "--allow-kaggle-fallback",
            "--skip-pull",
            "--max-series-per-dataset",
            "1",
            "--output-dir",
            str(output_dir),
        ]
    )

    assert exit_code == 0
    assert (output_dir / "result.json").exists()
    assert (output_dir / "summary.json").exists()
    assert (output_dir / "summary.md").exists()
    assert (output_dir / "benchmark_report.json").exists()
    assert (output_dir / "benchmark_report.md").exists()
    assert (output_dir / "raw").exists()

    result_payload = json.loads((output_dir / "result.json").read_text(encoding="utf-8"))
    summary_payload = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    benchmark_payload = json.loads(
        (output_dir / "benchmark_report.json").read_text(encoding="utf-8")
    )

    assert result_payload["max_series_per_dataset"] == 1
    assert len(result_payload["entries"]) == 270
    assert summary_payload["total_entries"] == 270
    assert len(benchmark_payload["benchmark_rows"]) == 110
    assert benchmark_payload["contract_summary"]["contract_best_effort_covariates"]["total"] == 80
    assert benchmark_payload["contract_summary"]["contract_strict_covariates"]["total"] == 80
