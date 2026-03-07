"""Tests for real-data run orchestrator retry/profile helpers."""

from __future__ import annotations

import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

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
_status_for_data_issue = _MODULE._status_for_data_issue
_sanitize_artifact_filename = _MODULE._sanitize_artifact_filename


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
