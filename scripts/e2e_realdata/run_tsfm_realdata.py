"""Run 6-model real-data E2E gate + benchmark matrix."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter, sleep
from typing import Any

from tollama.client import TollamaClient
from tollama.client.exceptions import DaemonHTTPError, TollamaClientError

if __package__ in {None, ""}:
    _THIS_DIR = Path(__file__).resolve().parent
    if str(_THIS_DIR) not in sys.path:
        sys.path.append(str(_THIS_DIR))
    import payload_builder as payload_builder  # noqa: PLC0414
    import prepare_data as prepare_data  # noqa: PLC0414
    import summarize_report as summarize_report  # noqa: PLC0414
    import validate_results as validate_results  # noqa: PLC0414
else:
    from . import payload_builder, prepare_data, summarize_report, validate_results

SUPPORTED_MODELS = [
    "chronos2",
    "granite-ttm-r2",
    "timesfm-2.5-200m",
    "moirai-2.0-R-small",
    "sundial-base-128m",
    "toto-open-base-1.0",
]

# Per-model context caps for accuracy optimisation.  Models that support
# longer context windows benefit from receiving more history than the
# harness-level --context-cap (which controls data *preparation*).  The
# _truncate_series_for_model helper trims each payload to the cap that
# matches the model's architecture.
MODEL_CONTEXT_CAPS: dict[str, int] = {
    "chronos2": 1024,
    "granite-ttm-r2": 512,  # adapter further truncates to context_length
    "moirai-2.0-R-small": 512,  # small variant works best with shorter context
    "timesfm-2.5-200m": 1024,  # model's max_context
    "sundial-base-128m": 2048,  # conservative subset of 2880
    "toto-open-base-1.0": 2048,  # conservative subset of 4096
}

SCENARIOS = [
    "benchmark_target_only",
    "contract_best_effort_covariates",
    "contract_strict_covariates",
]

RETRYABLE_STATUS_CODES = {502, 503}
RETRY_ATTEMPTS = 3
RETRY_BACKOFF_SECONDS = (2.0, 5.0)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run real-data E2E matrix for TSFM models.")
    parser.add_argument("--mode", choices=("pr", "nightly", "local"), required=True)
    parser.add_argument(
        "--gate-profile",
        choices=("strict", "hf_optional"),
        default="strict",
        help=(
            "Gate behavior profile. strict keeps all failures blocking; "
            "hf_optional downgrades HF data/payload build issues to skip."
        ),
    )
    parser.add_argument(
        "--model",
        default="all",
        help="Model name, comma-separated list, or 'all'.",
    )
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:11435",
        help="Daemon base URL.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for result.json, summary.md, and raw responses.",
    )
    parser.add_argument(
        "--catalog-path",
        default=str(Path(__file__).resolve().parent / "dataset_catalog.yaml"),
        help="Dataset catalog YAML path.",
    )
    parser.add_argument(
        "--cache-dir",
        default="/tmp/tollama-e2e-realdata-cache",
        help="Dataset cache directory.",
    )
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--context-cap", type=int, default=2048)
    parser.add_argument(
        "--allow-kaggle-fallback",
        action="store_true",
        help=(
            "In local mode, allow running with open datasets only when "
            "Kaggle credentials are missing."
        ),
    )
    parser.add_argument(
        "--skip-pull",
        action="store_true",
        help="Skip pull preflight if models are already installed.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    selected_models = _resolve_models(args.model)
    started_at = _now_iso()

    client = TollamaClient(base_url=args.base_url, timeout=float(args.timeout_seconds))

    try:
        _ = client.health()
    except TollamaClientError as exc:
        return _write_infra_failure(
            output_dir=output_dir,
            run_id=run_id,
            mode=args.mode,
            started_at=started_at,
            detail=f"daemon health check failed: {exc}",
        )

    creds_present = prepare_data.has_kaggle_credentials()
    policy = prepare_data.kaggle_policy_for_mode(
        args.mode,
        creds_present,
        allow_local_fallback=args.allow_kaggle_fallback,
    )
    if policy.hard_fail_on_missing and not policy.include_kaggle:
        return _write_infra_failure(
            output_dir=output_dir,
            run_id=run_id,
            mode=args.mode,
            started_at=started_at,
            detail=policy.message or "kaggle credentials missing",
        )

    try:
        catalog = prepare_data.load_dataset_catalog(Path(args.catalog_path))
        dataset_kinds = {
            str(item["name"]): str(item["kind"])
            for item in catalog["datasets"]
            if isinstance(item, dict)
        }
        prepared = prepare_data.prepare_datasets(
            mode=args.mode,
            catalog_path=Path(args.catalog_path),
            cache_dir=Path(args.cache_dir),
            include_kaggle=policy.include_kaggle,
            require_kaggle=policy.hard_fail_on_missing,
            seed=int(args.seed),
            context_cap=int(args.context_cap),
            timeout_seconds=max(int(args.timeout_seconds), 30),
        )
    except Exception as exc:  # noqa: BLE001
        return _write_infra_failure(
            output_dir=output_dir,
            run_id=run_id,
            mode=args.mode,
            started_at=started_at,
            detail=f"dataset preparation failed: {exc}",
        )

    entries: list[dict[str, Any]] = []

    for model in selected_models:
        if not args.skip_pull:
            pull_error = _pull_model(client=client, model=model)
            if pull_error is not None:
                entries.append(
                    _build_entry(
                        run_id=run_id,
                        mode=args.mode,
                        dataset="-",
                        scenario="preflight_pull",
                        model=model,
                        status="fail",
                        latency_ms=0.0,
                        metrics={},
                        warnings=[],
                        error=pull_error,
                        started_at=_now_iso(),
                        finished_at=_now_iso(),
                        http_status=None,
                        expected_status=None,
                        retry_count=0,
                    )
                )
                continue

        for dataset_name, series_list in prepared.datasets.items():
            dataset_kind = dataset_kinds.get(dataset_name, "")
            for series in series_list:
                horizon = len(series.get("actuals", []))
                if horizon <= 0:
                    status = _status_for_data_issue(
                        gate_profile=args.gate_profile,
                        dataset_kind=dataset_kind,
                    )
                    entries.append(
                        _build_entry(
                            run_id=run_id,
                            mode=args.mode,
                            dataset=dataset_name,
                            scenario="preflight_series",
                            model=model,
                            status=status,
                            latency_ms=0.0,
                            metrics={},
                            warnings=[],
                            error="series.actuals is required for real-data E2E",
                            started_at=_now_iso(),
                            finished_at=_now_iso(),
                            http_status=None,
                            expected_status=None,
                            retry_count=0,
                        )
                    )
                    continue

                for scenario in SCENARIOS:
                    try:
                        payload = _build_payload(
                            scenario=scenario,
                            model=model,
                            series=series,
                            horizon=horizon,
                            timeout_seconds=float(args.timeout_seconds),
                        )
                    except ValueError as exc:
                        status = _status_for_data_issue(
                            gate_profile=args.gate_profile,
                            dataset_kind=dataset_kind,
                        )
                        entries.append(
                            _build_entry(
                                run_id=run_id,
                                mode=args.mode,
                                dataset=dataset_name,
                                scenario=scenario,
                                model=model,
                                status=status,
                                latency_ms=0.0,
                                metrics={},
                                warnings=[],
                                error=f"payload_build_error: {exc}",
                                started_at=_now_iso(),
                                finished_at=_now_iso(),
                                http_status=None,
                                expected_status=None,
                                retry_count=0,
                            )
                        )
                        continue
                    expected_status = _expected_status(scenario=scenario, model=model)

                    started = _now_iso()
                    call_started = perf_counter()
                    (
                        http_status,
                        response_payload,
                        exception_detail,
                        retry_count,
                    ) = _call_forecast_with_retry(
                        client=client,
                        payload=payload,
                    )
                    latency_ms = round((perf_counter() - call_started) * 1000.0, 3)
                    finished = _now_iso()

                    passed, error_text, warnings, metrics = validate_results.evaluate_gate(
                        scenario=scenario,
                        model=model,
                        expected_status=expected_status,
                        http_status=http_status,
                        response_payload=response_payload,
                        horizon=horizon,
                        exception_detail=exception_detail,
                    )

                    status = "pass" if passed else "fail"
                    entry = _build_entry(
                        run_id=run_id,
                        mode=args.mode,
                        dataset=dataset_name,
                        scenario=scenario,
                        model=model,
                        status=status,
                        latency_ms=latency_ms,
                        metrics=metrics,
                        warnings=warnings,
                        error=error_text,
                        started_at=started,
                        finished_at=finished,
                        http_status=http_status,
                        expected_status=expected_status,
                        retry_count=retry_count,
                    )
                    entries.append(entry)

                    safe_series_id = _sanitize_artifact_filename(str(series["id"]))
                    _write_json(
                        raw_dir / model / dataset_name / scenario / f"{safe_series_id}.json",
                        {
                            "request": payload,
                            "response": response_payload,
                            "http_status": http_status,
                            "exception": exception_detail,
                            "retry_count": retry_count,
                            "entry": entry,
                            "series_id": series["id"],
                            "series_id_sanitized": safe_series_id,
                        },
                    )

    finished_at = _now_iso()
    result_payload = {
        "run_id": run_id,
        "mode": args.mode,
        "models": selected_models,
        "datasets": sorted(prepared.datasets.keys()),
        "messages": [message for message in prepared.messages if message],
        "started_at": started_at,
        "finished_at": finished_at,
        "entries": entries,
    }
    _write_json(output_dir / "result.json", result_payload)

    summary_payload = summarize_report.summarize_entries(entries)
    _write_json(output_dir / "summary.json", summary_payload)
    (output_dir / "summary.md").write_text(
        summarize_report.render_markdown(summary_payload),
        encoding="utf-8",
    )

    failed_count = sum(1 for item in entries if item.get("status") == "fail")
    print(f"run_id={run_id} entries={len(entries)} failed={failed_count}")
    print(f"artifacts={output_dir}")

    if failed_count > 0:
        return 1
    return 0


def _truncate_series_for_model(series: dict[str, Any], model: str) -> dict[str, Any]:
    """Return a copy of *series* with target/timestamps trimmed to the model's context cap."""
    cap = MODEL_CONTEXT_CAPS.get(model, 512)
    truncated = dict(series)
    truncated["target"] = list(series["target"][-cap:])
    truncated["timestamps"] = list(series["timestamps"][-cap:])
    return truncated


def _build_payload(
    *,
    scenario: str,
    model: str,
    series: dict[str, Any],
    horizon: int,
    timeout_seconds: float,
) -> dict[str, Any]:
    trimmed = _truncate_series_for_model(series, model)

    if scenario == "benchmark_target_only":
        return payload_builder.build_target_only_request(
            model=model,
            series=trimmed,
            horizon=horizon,
            timeout_seconds=timeout_seconds,
        )

    if scenario == "contract_best_effort_covariates":
        return payload_builder.build_covariate_request(
            model=model,
            series=trimmed,
            horizon=horizon,
            covariates_mode="best_effort",
            timeout_seconds=timeout_seconds,
        )

    if scenario == "contract_strict_covariates":
        return payload_builder.build_covariate_request(
            model=model,
            series=trimmed,
            horizon=horizon,
            covariates_mode="strict",
            timeout_seconds=timeout_seconds,
        )

    raise ValueError(f"unsupported scenario: {scenario!r}")


def _expected_status(*, scenario: str, model: str) -> int:
    if scenario == "contract_strict_covariates":
        return payload_builder.expected_status_for_strict_covariates(model)
    return 200


def _resolve_models(raw: str) -> list[str]:
    normalized = raw.strip().lower()
    if normalized == "all":
        return list(SUPPORTED_MODELS)

    selected = [item.strip() for item in raw.split(",") if item.strip()]
    if not selected:
        raise ValueError("--model must be a model name, comma-separated list, or 'all'")

    unknown = [item for item in selected if item not in SUPPORTED_MODELS]
    if unknown:
        raise ValueError(f"unsupported model(s): {', '.join(unknown)}")
    return selected


def _call_forecast_once(
    *,
    client: TollamaClient,
    payload: dict[str, Any],
) -> tuple[int | None, dict[str, Any] | None, str | None]:
    try:
        response_payload = client.forecast(payload, stream=False)
        if not isinstance(response_payload, dict):
            return None, None, "forecast response is not a JSON object"
        return 200, response_payload, None
    except DaemonHTTPError as exc:
        status = exc.status_code if isinstance(exc.status_code, int) else None
        return status, {"detail": exc.detail}, str(exc)
    except TollamaClientError as exc:
        return None, None, str(exc)


def _call_forecast_with_retry(
    *,
    client: TollamaClient,
    payload: dict[str, Any],
    max_attempts: int = RETRY_ATTEMPTS,
    backoff_seconds: tuple[float, ...] = RETRY_BACKOFF_SECONDS,
    sleep_fn: Callable[[float], None] = sleep,
) -> tuple[int | None, dict[str, Any] | None, str | None, int]:
    attempts = max(1, int(max_attempts))
    retry_count = 0

    http_status, response_payload, exception_detail = _call_forecast_once(
        client=client,
        payload=payload,
    )

    while retry_count < attempts - 1 and _is_retryable_failure(
        http_status=http_status,
        exception_detail=exception_detail,
    ):
        delay = 0.0
        if backoff_seconds:
            if retry_count < len(backoff_seconds):
                delay = float(backoff_seconds[retry_count])
            else:
                delay = float(backoff_seconds[-1])
        if delay > 0:
            sleep_fn(delay)

        retry_count += 1
        http_status, response_payload, exception_detail = _call_forecast_once(
            client=client,
            payload=payload,
        )

    return http_status, response_payload, exception_detail, retry_count


def _is_retryable_failure(*, http_status: int | None, exception_detail: str | None) -> bool:
    if http_status in RETRYABLE_STATUS_CODES:
        return True
    if not exception_detail:
        return False
    lowered = exception_detail.lower()
    return "timed out" in lowered or "timeout" in lowered


def _pull_model(*, client: TollamaClient, model: str) -> str | None:
    try:
        response = client.pull_model(name=model, stream=False, accept_license=True)
    except TollamaClientError as exc:
        return str(exc)

    if not isinstance(response, dict):
        return "pull returned non-object response"
    return None


def _build_entry(
    *,
    run_id: str,
    mode: str,
    dataset: str,
    scenario: str,
    model: str,
    status: str,
    latency_ms: float,
    metrics: dict[str, float],
    warnings: list[str],
    error: str | None,
    started_at: str,
    finished_at: str,
    http_status: int | None,
    expected_status: int | None,
    retry_count: int,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "mode": mode,
        "dataset": dataset,
        "scenario": scenario,
        "model": model,
        "status": status,
        "latency_ms": latency_ms,
        "metrics": metrics,
        "warnings": warnings,
        "error": error,
        "started_at": started_at,
        "finished_at": finished_at,
        "http_status": http_status,
        "expected_status": expected_status,
        "retry_count": int(retry_count),
    }


def _status_for_data_issue(*, gate_profile: str, dataset_kind: str) -> str:
    if gate_profile == "hf_optional" and dataset_kind == "huggingface_dataset":
        return "skip"
    return "fail"


def _write_infra_failure(
    *,
    output_dir: Path,
    run_id: str,
    mode: str,
    started_at: str,
    detail: str,
) -> int:
    finished_at = _now_iso()
    payload = {
        "run_id": run_id,
        "mode": mode,
        "models": [],
        "datasets": [],
        "messages": [detail],
        "started_at": started_at,
        "finished_at": finished_at,
        "entries": [],
    }
    _write_json(output_dir / "result.json", payload)
    _write_json(
        output_dir / "summary.json",
        {
            "gate_pass": False,
            "total_entries": 0,
            "total_failed": 1,
            "total_skipped": 0,
            "models": {},
            "infra_error": detail,
        },
    )
    (output_dir / "summary.md").write_text(
        "# Real-Data E2E Summary\n\n"
        f"- Gate pass: **False**\n"
        f"- Infra error: **{detail}**\n",
        encoding="utf-8",
    )
    print(detail)
    return 2


def _sanitize_artifact_filename(value: str) -> str:
    sanitized = value
    for char in ('"', ':', '<', '>', '|', '*', '?', '\\r', '\\n'):
        sanitized = sanitized.replace(char, "_")
    sanitized = sanitized.strip()
    return sanitized or "series"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
