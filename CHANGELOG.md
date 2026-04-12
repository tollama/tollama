# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [0.1.0] - Unreleased

### Changed

- Re-centered tollama messaging around its core role as a unified TSFM platform
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

### Added

- Daemon (`tollamad`) with FastAPI-based HTTP API surface
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
