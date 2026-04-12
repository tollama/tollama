"""Run real-data E2E gate + benchmark matrices."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter, sleep
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from tollama.client import TollamaClient

if __package__ in {None, ""}:
    _THIS_DIR = Path(__file__).resolve().parent
    if str(_THIS_DIR) not in sys.path:
        sys.path.append(str(_THIS_DIR))
    import benchmark_report as benchmark_report  # noqa: PLC0414
    import model_policy as model_policy  # noqa: PLC0414
    import payload_builder as payload_builder  # noqa: PLC0414
    import summarize_report as summarize_report  # noqa: PLC0414
    import validate_results as validate_results  # noqa: PLC0414
else:
    from . import (
        benchmark_report,
        model_policy,
        payload_builder,
        summarize_report,
        validate_results,
    )

RETRYABLE_STATUS_CODES = {502, 503}
RETRY_ATTEMPTS = 3
RETRY_BACKOFF_SECONDS = (2.0, 5.0)
STARTER_HF_CATALOG_NAME = "hf_dataset_catalog_starter.yaml"


def _load_client_runtime() -> tuple[type[Any], type[Exception], type[Exception]]:
    from tollama.client import TollamaClient
    from tollama.client.exceptions import DaemonHTTPError, TollamaClientError

    return TollamaClient, DaemonHTTPError, TollamaClientError


def _create_client(*, base_url: str, timeout: float) -> Any:
    client_cls, _, _ = _load_client_runtime()
    return client_cls(base_url=base_url, timeout=timeout)


def _load_prepare_data_module() -> Any:
    if __package__ in {None, ""}:
        if str(_THIS_DIR) not in sys.path:
            sys.path.append(str(_THIS_DIR))
        return importlib.import_module("prepare_data")
    return importlib.import_module(f"{__package__}.prepare_data")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run real-data E2E matrix for forecast models.")
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
        help="Model name, comma-separated list, or alias: all, hf_all, neural.",
    )
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:11435",
        help="Daemon base URL.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for result.json, summaries, reports, and raw responses.",
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
        "--max-series-per-dataset",
        type=int,
        default=None,
        help="Optional deterministic cap overriding the mode-based sample count.",
    )
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

    catalog_path = Path(args.catalog_path).resolve()
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    started_at = _now_iso()
    prepare_data = _load_prepare_data_module()

    try:
        selected_models = _resolve_models(args.model)
    except ValueError as exc:
        print(str(exc))
        return 2

    client = _create_client(base_url=args.base_url, timeout=float(args.timeout_seconds))

    try:
        _ = client.health()
    except _load_client_runtime()[2] as exc:
        payload = _build_result_payload(
            run_id=run_id,
            mode=args.mode,
            base_url=args.base_url,
            gate_profile=args.gate_profile,
            catalog_path=catalog_path,
            models=selected_models,
            datasets=[],
            messages=[f"daemon health check failed: {exc}"],
            started_at=started_at,
            finished_at=_now_iso(),
            entries=[],
            max_series_per_dataset=args.max_series_per_dataset,
            infra_error=f"daemon health check failed: {exc}",
        )
        _write_artifacts(output_dir=output_dir, result_payload=payload)
        print(f"daemon health check failed: {exc}")
        return 2

    creds_present = prepare_data.has_kaggle_credentials()
    policy = prepare_data.kaggle_policy_for_mode(
        args.mode,
        creds_present,
        allow_local_fallback=args.allow_kaggle_fallback,
    )
    if policy.hard_fail_on_missing and not policy.include_kaggle:
        detail = policy.message or "kaggle credentials missing"
        payload = _build_result_payload(
            run_id=run_id,
            mode=args.mode,
            base_url=args.base_url,
            gate_profile=args.gate_profile,
            catalog_path=catalog_path,
            models=selected_models,
            datasets=[],
            messages=[detail],
            started_at=started_at,
            finished_at=_now_iso(),
            entries=[],
            max_series_per_dataset=args.max_series_per_dataset,
            infra_error=detail,
        )
        _write_artifacts(output_dir=output_dir, result_payload=payload)
        print(detail)
        return 2

    try:
        catalog = prepare_data.load_dataset_catalog(
            catalog_path,
            require_unique_hf_ids=_require_unique_hf_ids(catalog_path),
        )
        _validate_dataset_catalog(
            catalog,
            require_unique_hf_ids=_require_unique_hf_ids(catalog_path),
        )
        dataset_kinds = {
            str(item["name"]): str(item["kind"])
            for item in catalog["datasets"]
            if isinstance(item, dict)
        }
        prepared = prepare_data.prepare_datasets(
            mode=args.mode,
            catalog_path=catalog_path,
            cache_dir=Path(args.cache_dir),
            include_kaggle=policy.include_kaggle,
            require_kaggle=policy.hard_fail_on_missing,
            seed=int(args.seed),
            context_cap=int(args.context_cap),
            timeout_seconds=max(int(args.timeout_seconds), 30),
            max_series_per_dataset=args.max_series_per_dataset,
        )
    except Exception as exc:  # noqa: BLE001
        detail = f"dataset preparation failed: {exc}"
        payload = _build_result_payload(
            run_id=run_id,
            mode=args.mode,
            base_url=args.base_url,
            gate_profile=args.gate_profile,
            catalog_path=catalog_path,
            models=selected_models,
            datasets=[],
            messages=[detail],
            started_at=started_at,
            finished_at=_now_iso(),
            entries=[],
            max_series_per_dataset=args.max_series_per_dataset,
            infra_error=detail,
        )
        _write_artifacts(output_dir=output_dir, result_payload=payload)
        print(detail)
        return 2

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
                        series_id=None,
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

        enabled_scenarios = _scenarios_for_model(model)
        for dataset_name, series_list in prepared.datasets.items():
            dataset_kind = dataset_kinds.get(dataset_name, "")
            for series in series_list:
                series_id = _series_id(series)
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
                            series_id=series_id,
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

                for scenario in enabled_scenarios:
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
                                series_id=series_id,
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
                        series_id=series_id,
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

                    safe_series_id = _sanitize_artifact_filename(series_id)
                    _write_json(
                        raw_dir / model / dataset_name / scenario / f"{safe_series_id}.json",
                        {
                            "request": payload,
                            "response": response_payload,
                            "http_status": http_status,
                            "exception": exception_detail,
                            "retry_count": retry_count,
                            "entry": entry,
                            "series_id": series_id,
                            "series_id_sanitized": safe_series_id,
                        },
                    )

    finished_at = _now_iso()
    messages = [message for message in prepared.messages if message]
    if policy.message:
        messages.append(policy.message)
    result_payload = _build_result_payload(
        run_id=run_id,
        mode=args.mode,
        base_url=args.base_url,
        gate_profile=args.gate_profile,
        catalog_path=catalog_path,
        models=selected_models,
        datasets=sorted(prepared.datasets.keys()),
        messages=messages,
        started_at=started_at,
        finished_at=finished_at,
        entries=entries,
        max_series_per_dataset=args.max_series_per_dataset,
        infra_error=None,
    )
    _write_artifacts(output_dir=output_dir, result_payload=result_payload)

    failed_count = sum(1 for item in entries if item.get("status") == "fail")
    print(f"run_id={run_id} entries={len(entries)} failed={failed_count}")
    print(f"artifacts={output_dir}")

    if failed_count > 0:
        return 1
    return 0


def _truncate_series_for_model(series: dict[str, Any], model: str) -> dict[str, Any]:
    """Return a copy of *series* trimmed to the model-specific context cap."""
    cap = model_policy.context_cap_for_model(model)
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

    if scenario == model_policy.BENCHMARK_TARGET_ONLY:
        return payload_builder.build_target_only_request(
            model=model,
            series=trimmed,
            horizon=horizon,
            timeout_seconds=timeout_seconds,
        )

    if scenario == model_policy.CONTRACT_BEST_EFFORT:
        return payload_builder.build_covariate_request(
            model=model,
            series=trimmed,
            horizon=horizon,
            covariates_mode="best_effort",
            timeout_seconds=timeout_seconds,
        )

    if scenario == model_policy.CONTRACT_STRICT:
        return payload_builder.build_covariate_request(
            model=model,
            series=trimmed,
            horizon=horizon,
            covariates_mode="strict",
            timeout_seconds=timeout_seconds,
        )

    raise ValueError(f"unsupported scenario: {scenario!r}")


def _expected_status(*, scenario: str, model: str) -> int:
    if scenario == model_policy.CONTRACT_STRICT:
        return payload_builder.expected_status_for_strict_covariates(model)
    return 200


def _resolve_models(raw: str) -> list[str]:
    return model_policy.resolve_models(raw)


def _scenarios_for_model(model: str) -> list[str]:
    return model_policy.scenarios_for_model(model)


def _require_unique_hf_ids(catalog_path: Path) -> bool:
    return catalog_path.name == STARTER_HF_CATALOG_NAME


def _validate_dataset_catalog(
    catalog: dict[str, Any],
    *,
    require_unique_hf_ids: bool,
) -> None:
    datasets = catalog.get("datasets")
    if not isinstance(datasets, list) or not datasets:
        raise ValueError("dataset catalog must contain a non-empty datasets list")

    seen_names: set[str] = set()
    seen_hf_ids: set[str] = set()
    for item in datasets:
        if not isinstance(item, dict):
            raise ValueError("each dataset catalog entry must be a mapping")

        name = item.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("dataset catalog entry missing name")
        if name in seen_names:
            raise ValueError(f"duplicate dataset name: {name}")
        seen_names.add(name)

        kind = item.get("kind")
        if not isinstance(kind, str) or not kind.strip():
            raise ValueError(f"dataset {name!r} missing kind")

        horizon = item.get("horizon")
        if not isinstance(horizon, int) or horizon <= 0:
            raise ValueError(f"dataset {name!r} has invalid horizon")

        if kind == "huggingface_dataset":
            for field in ("hf_id", "timestamp_column", "target_column"):
                value = item.get(field)
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(f"dataset {name!r} missing {field}")
            hf_id = str(item["hf_id"]).strip()
            if require_unique_hf_ids:
                if hf_id in seen_hf_ids:
                    raise ValueError(f"duplicate hf_id: {hf_id}")
                seen_hf_ids.add(hf_id)


def _call_forecast_once(
    *,
    client: Any,
    payload: dict[str, Any],
) -> tuple[int | None, dict[str, Any] | None, str | None]:
    _, daemon_http_error, tollama_client_error = _load_client_runtime()
    try:
        response_payload = client.forecast(payload, stream=False)
        if not isinstance(response_payload, dict):
            return None, None, "forecast response is not a JSON object"
        return 200, response_payload, None
    except daemon_http_error as exc:
        status = exc.status_code if isinstance(exc.status_code, int) else None
        return status, {"detail": exc.detail}, str(exc)
    except tollama_client_error as exc:
        return None, None, str(exc)


def _call_forecast_with_retry(
    *,
    client: Any,
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
    _, _, tollama_client_error = _load_client_runtime()
    try:
        response = client.pull_model(name=model, stream=False, accept_license=True)
    except tollama_client_error as exc:
        return str(exc)

    if not isinstance(response, dict):
        return "pull returned non-object response"
    return None


def _build_entry(
    *,
    run_id: str,
    mode: str,
    dataset: str,
    series_id: str | None,
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
        "series_id": series_id,
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


def _build_result_payload(
    *,
    run_id: str,
    mode: str,
    base_url: str,
    gate_profile: str,
    catalog_path: Path,
    models: list[str],
    datasets: list[str],
    messages: list[str],
    started_at: str,
    finished_at: str,
    entries: list[dict[str, Any]],
    max_series_per_dataset: int | None,
    infra_error: str | None,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "mode": mode,
        "base_url": base_url,
        "gate_profile": gate_profile,
        "catalog_path": str(catalog_path),
        "max_series_per_dataset": max_series_per_dataset,
        "scenario_policy": model_policy.scenario_policy_summary(),
        "models": models,
        "datasets": datasets,
        "messages": messages,
        "started_at": started_at,
        "finished_at": finished_at,
        "infra_error": infra_error,
        "entries": entries,
    }


def _status_for_data_issue(*, gate_profile: str, dataset_kind: str) -> str:
    if gate_profile == "hf_optional" and dataset_kind == "huggingface_dataset":
        return "skip"
    return "fail"


def _write_artifacts(*, output_dir: Path, result_payload: dict[str, Any]) -> None:
    infra_error = result_payload.get("infra_error")
    _write_json(output_dir / "result.json", result_payload)

    summary_payload = summarize_report.summarize_entries(
        [
            entry
            for entry in result_payload.get("entries", [])
            if isinstance(entry, dict)
        ]
    )
    if isinstance(infra_error, str) and infra_error.strip():
        summary_payload["infra_error"] = infra_error
        summary_payload["gate_pass"] = False
        summary_payload["total_failed"] = max(int(summary_payload.get("total_failed", 0)), 1)
    _write_json(output_dir / "summary.json", summary_payload)

    summary_markdown = summarize_report.render_markdown(summary_payload)
    if isinstance(infra_error, str) and infra_error.strip():
        summary_markdown += f"\n- Infra error: **{infra_error}**\n"
    (output_dir / "summary.md").write_text(summary_markdown, encoding="utf-8")

    benchmark_payload = benchmark_report.build_benchmark_report(result_payload)
    _write_json(output_dir / "benchmark_report.json", benchmark_payload)
    (output_dir / "benchmark_report.md").write_text(
        benchmark_report.render_markdown(benchmark_payload),
        encoding="utf-8",
    )


def _series_id(series: dict[str, Any]) -> str | None:
    value = series.get("id")
    if isinstance(value, str) and value:
        return value
    return None


def _sanitize_artifact_filename(value: str | None) -> str:
    sanitized = value or ""
    for char in ('"', ":", "<", ">", "|", "*", "?", "\r", "\n"):
        sanitized = sanitized.replace(char, "_")
    sanitized = sanitized.strip()
    if sanitized and set(sanitized) == {"_"}:
        sanitized = ""
    return sanitized or "series"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
