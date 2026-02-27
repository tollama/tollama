# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [0.1.0] - Unreleased

### Added

- Daemon (`tollamad`) with FastAPI-based HTTP API surface
- Ollama-compatible model lifecycle: `pull`, `list`, `show`, `ps`, `rm`
- Unified forecasting endpoint (`POST /api/forecast`) with multi-family routing
- Auto-forecast with ensemble strategies (`POST /api/auto-forecast`)
- Series diagnostics (`POST /api/analyze`) and synthetic generation (`POST /api/generate`)
- Counterfactual, scenario-tree, what-if, pipeline, compare, and report endpoints
- CSV/Parquet ingest via `data_url` and upload endpoints
- TSModelfile profile management (YAML-based model presets)
- Runner families: Chronos-2, Granite TTM, TimesFM 2.5, Moirai, Sundial, Toto
- Unified covariates contract (`past_covariates`, `future_covariates`, `static_covariates`)
- Python SDK (`Tollama` class) with workflow chaining and DataFrame support
- CLI with shell completion, interactive model selection, and progress display
- MCP server with 15 tool handlers for Claude Desktop integration
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
