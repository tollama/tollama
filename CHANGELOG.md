# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [0.1.0] - Unreleased

### Changed

- Re-centered tollama messaging around its core role as a unified TSFM platform
- Per-family runtime auto-bootstrap now pins `tollama[runner-extra]` installs to
  the current Tollama version when resolving from PyPI, preventing packaged
  child runtimes from drifting away from the parent app version
- macOS app runtime bootstrap now passes its bundled wheelhouse into family
  runtime installation, and the default app bundle includes the starter-model
  `runner-sundial` extra so `sundial-base-128m` forecasts work out of the box;
  the macOS packaging default no longer forces the optional `eval` extra
- Synced documentation with the current registry and runner surface, including
  Lag-Llama, PatchTST, TiDE, N-HiTS, N-BEATSx, Timer, TimeMixer, and ForecastPFN
- Synced documentation with the current API surface, including dashboard bootstrap
  and XAI/trust routes
- Upgraded development status from Alpha to Beta (pyproject.toml classifier)
- Synced TimesFM and Moirai registry capability metadata with their implemented covariate support
- Synced TiDE, N-HiTS, and N-BEATSx capability metadata/docs with current runner behavior and made daemon covariate preflight prefer current registry capabilities over stale installed manifest copies
- Synced raw HTTP API docs with current error body semantics (`detail`/`hint`, validation as HTTP `400`) and runtime telemetry semantics for `ForecastResponse.usage`
- Auto-forecast routing modes now honor benchmark-backed routing manifests before falling back to heuristics
- Added an explicit Ollama-workflow parity contract and release-gate documentation
- `scripts/verify_daemon_api.sh` can now emit `result.json`, `summary.json`, and `summary.md` artifacts for workflow parity verification
- Added a checked-in OpenAPI export workflow (`docs/openapi.json`) plus deterministic artifact normalization for CI drift checks
- Added phased Python 3.11 quality gates for scoped mypy, blocking coverage at the initial 64% baseline, pre-commit, medium+ severity Bandit, pip-audit, and compiled dev-lock freshness
- Extended daemon observability with additive `/health/live` and `/health/ready` probes and per-runner metrics derived from forecast timing/usage metadata
- Centralized daemon/core/cli-owned `TOLLAMA_*` environment access under `src/tollama/core/env.py`
- macOS app detail pane now uses native SwiftUI `Models`, `Data`, `Forecast`,
  and `Logs` tabs instead of the embedded WebKit dashboard, while keeping the
  existing daemon lifecycle/sidebar actions
- macOS app runtime bootstrap now verifies that `tollama.daemon.main` is
  importable before reusing a cached venv, and failed install commands no longer
  write a successful runtime marker
- macOS app managed daemon starts now reset the daemon log tail so stale
  bootstrap errors are not shown as current launch failures
- macOS app attached-daemon mode now refreshes and trims the log tail from the
  latest daemon run, avoiding stale bootstrap errors in the native Logs UI
- macOS native workspace navigation now uses an explicit segmented tab bar and
  fixed split-view sizing to avoid blank or collapsed detail layouts after
  repeated navigation
- macOS Data tab now uses a fixed-width native file browser pane instead of a
  nested split view, preventing the left pane from collapsing during tab changes
- macOS runtime asset builds now resolve and verify the bundled wheelhouse with
  the bundled Python runtime, preventing host-Python wheels from breaking first
  launch installs
- `GET /v1/models` now includes public registry `source`, `metadata`,
  `capabilities`, and license notice fields for native clients
- CSV ingest and the macOS CSV preview sniffer now auto-detect additional common
  timestamp, series identifier, and target aliases, including `date`, `Date`,
  `datetime`, `series`, `OT`, `demand`, `users`, air-quality targets, and OPSD
  load-actual columns
- CSV ingest now omits rows with null target values after resolving the target
  column, while still rejecting entirely null targets
- macOS per-family runtime bootstrap now creates symlinked virtualenv
  interpreters so bundled Python runtimes can locate `libpython` correctly
- Per-family runtime bootstrap now records an optional-extra dependency
  fingerprint and rebuilds stale family venvs when runner dependencies change
- Registry loading now falls back to editable-install source metadata if an
  in-place runtime hotfix removes packaged registry data while a daemon is live
- CSV/data-url forecasts for single-series model implementations now
  down-select to the first ingested series with a warning instead of failing in
  the runner; Granite TTM declares this one-series input limit in registry
  metadata
- TimeMixer and ForecastPFN registry pulls are now manifest-only local-source
  installs, and the macOS app surfaces streamed pull errors instead of treating
  them as successful progress events
- ForecastPFN now reports the built-in manifest-only model as `MODEL_UNSUPPORTED`
  instead of falling through to a generic missing-module inference error
- Manifest-only TimeMixer and ForecastPFN entries are now marked
  `forecast_ready=false`; recommend, auto-forecast, and the macOS Forecast
  picker skip them for executable forecasts, and `MODEL_UNSUPPORTED` maps to
  HTTP `400`; direct forecast calls are preflighted against the current registry
  before runner startup, even when an older installed manifest lacks the flag
- TiDE runner installs now explicitly include `pytorch-lightning`, matching the
  runtime import path used by Darts' `TiDEModel`
- Timer runner now restores legacy Transformers cache APIs expected by THUML's
  remote model code, fixing inference with newer `transformers` releases
- Granite TTM now left-pads histories shorter than its 512-point context window
  instead of rejecting app/demo forecasts with a context-length error
- Lag-Llama now normalizes dataset indexes before GluonTS ingestion and returns
  structured forecast errors for backend runtime failures instead of letting the
  runner process exit

### Added

- Daemon (`tollamad`) with FastAPI-based HTTP API surface
- macOS app distribution scaffold under `packaging/macos/`:
  - `build_dmg.sh` end-to-end DMG build/sign/notarize/checksum flow
  - `build_pkg.sh` installable PKG build/sign/notarize/checksum flow
  - `build_release_artifacts.sh` combined DMG + PKG build path
  - `prepare_runtime_assets.sh` bundled runtime/wheelhouse preparation
  - `Tollama.app` SwiftUI + `WKWebView` shell with first-launch runtime bootstrap,
    child daemon lifecycle management, demo/starter-model actions, and inline log-tail view
- Ollama-compatible model lifecycle: `pull`, `list`, `show`, `ps`, `rm`
- Unified forecasting endpoint (`POST /api/forecast`) with multi-family routing
- Auto-forecast with ensemble strategies (`POST /api/auto-forecast`)
- Series diagnostics (`POST /api/analyze`) and synthetic generation (`POST /api/generate`)
- Counterfactual, scenario-tree, what-if, pipeline, compare, and report endpoints
- CSV/Parquet ingest via `data_url` and upload endpoints
- TSModelfile profile management (YAML-based model presets)
- Runner families: Chronos-2, Granite TTM, TimesFM 2.5, Moirai, Sundial, Toto,
  Lag-Llama, PatchTST, TiDE, N-HiTS, N-BEATSx, Timer, TimeMixer, ForecastPFN
- Unified covariates contract (`past_covariates`, `future_covariates`, `static_covariates`)
- Python SDK (`Tollama` class) with workflow chaining and DataFrame support
- CLI with shell completion, interactive model selection, and progress display
- MCP server with 22 tool handlers for Claude Desktop integration
- Agent framework wrappers: LangChain, CrewAI, AutoGen, smolagents
- OpenClaw skill (`skills/tollama-forecast/`)
- Optional API-key authentication with HMAC constant-time comparison
- Per-key usage metering and SSE event streaming
- Rate limiting (token bucket algorithm)
- Prometheus metrics endpoint (`/metrics`)
- Bundled web dashboard and Textual TUI dashboard
- Security response headers middleware
- Path traversal protection in data ingest and model storage
- Docker and docker-compose support
- Tutorial notebooks (covariates, comparison, what-if, auto-forecast)
- Real-data TSFM E2E harness (`scripts/e2e_realdata/`) with PR smoke + nightly matrix workflow (`.github/workflows/e2e-realdata.yml`)
- HuggingFace optional local E2E hardening:
  - deterministic catalog gatherer with schema-quality thresholds and rejection report (`scripts/e2e_realdata/gather_hf_datasets.py`, `hf_dataset_rejections.json`)
  - normalized HF parser with contiguous-window enforcement and frequency inference (`prepare_data.py`)
  - orchestrator `--gate-profile {strict,hf_optional}` + transient retry/backoff with per-entry `retry_count`
  - HF local wrapper script (`scripts/e2e_realdata_hf.sh`)
- Lag-Llama + PatchTST merge-hardening pack:
  - E2E verification checklist (`docs/lag-llama-patchtst-e2e-checklist.md`)
  - helper script for targeted checks (`scripts/e2e_lag_llama_patchtst_check.sh`)
  - runner regression coverage for payload validation and validation-before-dependency-gating behavior
- TiDE phase-3 runner path:
  - stdio JSON-RPC inference runner with adapter-based error handling
  - deterministic mean forecasts with best-effort quantile extraction
  - explicit warning-based mean-only fallback when quantile outputs are unavailable
  - regression tests for quantile success/fallback and runner wiring
- Full-model local E2E helper: `scripts/run_all_models_e2e_local.sh`
  - executes pull + forecast smoke coverage across all currently registered models
  - auto-generates long-context payloads for long-history families
  - uses PatchTST-specific horizon payload for stable smoke coverage
- Shared runner protocol helpers in `src/tollama/runners/common_protocol.py` plus adoption across the scaffold and current runner entrypoints
- Shared error hierarchy in `src/tollama/core/errors.py` with backward-compatible runner error re-export shims
- Architecture, contract, and property-test scaffolding under `tests/architecture/`, `tests/contract/`, and `tests/property/`
- Shared pytest fixtures and markers in `tests/conftest.py`
- ADR scaffold and initial decision records under `docs/adr/`
- Compiled development lock artifact at `requirements-dev.lock`

### Fixed

- Deleting a loaded model now removes its loaded session from `/api/ps`
- Restored best-effort dynamic-covariate fallback for TimesFM and Moirai so
  runtime XReg/covariate-path failures degrade to target-only forecasts with
  warnings instead of hard-failing requests
- Restored per-family runtime bootstrap `uv venv` fallback when stdlib
  `venv`/`ensurepip` fails, including schema-version bump and regression tests
- Restored `scripts/e2e_realdata_tsfm.sh` interpreter preflight so the strict
  real-data wrapper now matches the HF wrapper's startup/runtime diagnostics
- PatchTST runner adapter:
  - aligned forecast tensor input shape to model expectations (`[batch, seq, channels]`)
  - added model-config channel detection to avoid input-channel mismatch errors
- TiDE runner adapter:
  - added robust model loading fallback when pretrained checkpoint API is unavailable
  - added runtime fit path fallback for local class-based TiDE execution
  - normalized TiDE input values to float32 to avoid MPS float64 conversion failures
- E2E suite stability:
  - `scripts/e2e_all_families.sh` now includes full-model smoke helper execution
- PostgreSQL connector query composition now validates identifiers, renders SQL identifiers safely, and supports pooled connections via `ThreadedConnectionPool`
- Runner supervisor stderr is drained continuously to prevent `PIPE` deadlocks while preserving recent stderr tails for crash diagnostics
- Daemon request IDs are now forwarded into supervisor calls so runner protocol request IDs stay correlated with the incoming HTTP request when available
