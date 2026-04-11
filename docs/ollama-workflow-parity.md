# Ollama-Workflow Parity

Tollama is not a full Ollama clone. The compatibility target is narrower and
intentional: an Ollama-style local workflow for time-series forecasting.

This document defines what "Ollama for Time Series Forecasting" means in
release-gated terms.

## Scope

Guaranteed parity target:

- local daemon + HTTP workflow
- model lifecycle: `pull`, `list`, `show`, `ps`, `rm`
- forecast execution via `tollama run`, `POST /api/forecast`, and `POST /v1/forecast`
- health/version checks
- stable error/status behavior for common workflow failures

Explicit non-goals:

- Ollama chat, generate, or embeddings APIs
- byte-for-byte payload equivalence with Ollama's LLM contracts
- Tollama-only advanced APIs such as `compare`, `analyze`, `pipeline`, and `xai`

## Workflow Matrix

| Workflow step | CLI | HTTP route | Expected result |
| --- | --- | --- | --- |
| Health | `tollama doctor` / `tollama info` | `GET /v1/health`, `GET /api/version` | Daemon reachable and versioned |
| Pull | `tollama pull <model>` | `POST /api/pull`, `POST /v1/models/pull` | Model installs or returns clear license/missing-model error |
| List installed | `tollama list` | `GET /api/tags` | Installed models visible in Ollama-style list |
| Show metadata | `tollama show <model>` | `POST /api/show` | Installed model metadata available |
| Run forecast | `tollama run <model>` | `POST /api/forecast`, `POST /v1/forecast` | Canonical forecast payload returned |
| List loaded | `tollama ps` | `GET /api/ps` | Loaded sessions reflect `keep_alive` behavior |
| Remove | `tollama rm <model>` | `DELETE /api/delete`, `DELETE /v1/models/{name}` | Model removed and loaded session cleared |

## Acceptance Rules

The release gate treats parity as satisfied when all of the following hold:

- lifecycle routes pass deterministic contract tests with `mock`
- `POST /api/forecast` with `stream=false` matches `POST /v1/forecast` on canonical forecast fields
- `pull -> show/list -> run -> ps -> rm` works end to end with `mock`
- deleting a loaded model removes it from `/api/ps`
- at least one local-source non-mock family passes pull + forecast smoke checks
- dependency-heavy TSFM families classify only as `pass` or `dependency-gated`, never as protocol regressions

## Release Gate

Primary gate:

```bash
bash scripts/verify_daemon_api.sh --output-dir artifacts/parity/daemon-api
```

Recommended supporting checks:

```bash
pytest -q tests/test_openapi_docs.py tests/test_daemon_api.py tests/test_client_http.py tests/test_cli.py tests/test_tsfm_parity_smoke.py
bash scripts/e2e_all_families.sh
```

When `--output-dir` is provided, `scripts/verify_daemon_api.sh` writes:

- `result.json`: flat list of categorized check results
- `summary.json`: aggregated pass/fail counts
- `summary.md`: short operator-facing report

## Dependency-Gated Results

Some model families require optional extras or heavyweight runtimes. For the
workflow parity gate, these are acceptable outcomes:

- `pass`: the family ran successfully
- `dependency-gated`: the family is unavailable only because its optional runtime is not installed

These are release blockers:

- protocol errors
- wrong HTTP status mapping
- missing route behavior
- loaded-model lifecycle drift
