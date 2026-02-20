# tollama roadmap (worker-per-model-family)

Last updated: 2026-02-20

Status legend:
- `[x]` implemented
- `[~]` partially implemented
- `[ ]` planned

This roadmap is implementation-aware for the repository under
`/Users/yongchoelchoi/Documents/GitHub/tollama/src/tollama/*` and still tracks
the optional future `packages/*` split as a migration phase.

## 0) Product goals [~]
### Current implementation status
- Unified forecasting endpoints are available at `POST /api/forecast` and `POST /v1/forecast`.
- Zero-config auto-forecast endpoint is available at `POST /api/auto-forecast`.
- Model-free series diagnostics endpoint is available at `POST /api/analyze`.
- Scenario analysis endpoint is available at `POST /api/what-if`.
- Ollama-style model lifecycle is available (`pull`, `list`, `show`, `ps`, `rm`) via HTTP and CLI.
- Forecast routing uses model-family worker selection from installed manifests.
- Multi-family adapters are shipped:
  - torch runner: Chronos-2 + Granite TTM
  - timesfm runner: TimesFM 2.5
  - uni2ts runner: Moirai
- Unified covariates contract is implemented with `past_covariates`, `future_covariates`,
  `parameters.covariates_mode`, compatibility preflight, and response `warnings`.
- v1 non-goals are still respected: no training/fine-tuning, no distributed scheduler, no multi-tenant auth.

### Planned work / TODO
- Strengthen VRAM reclaim policy with explicit idle strategy and crash recovery behavior.
- Keep local-first product posture for v1.

## 1) Top-level architecture [x]
### Current implementation status
- `tollamad` (`src/tollama/daemon/`) owns public HTTP API and process supervision.
- Runners (`src/tollama/runners/`) communicate over stdio JSON lines.
- Shared core (`src/tollama/core/`) provides schemas, protocol, registry/storage/config helpers.
- CLI (`src/tollama/cli/`) provides user commands and daemon HTTP client integration.
- Active runner implementations: `mock`, `torch`, `timesfm`, `uni2ts`, `sundial`, `toto`.

### Planned work / TODO
- Keep daemon free of heavy ML runtime imports and dependencies.
- Keep family boundaries strict so each runner evolves independently.

## 2) Repo layout [~]
### Current implementation status
- Current implemented layout is:

```text
/Users/yongchoelchoi/Documents/GitHub/tollama/
  src/tollama/
    cli/
    core/
    daemon/
    runners/
  tests/
  model-registry/
  examples/
  docs/
```

### Planned work / TODO
- Track this target product-grade split as an explicit migration phase:

```text
tollama/
  packages/
    tollama-core/
    tollamad/
    tollama-cli/
    tollama-runner-torch/
    tollama-runner-timesfm/
    tollama-runner-uni2ts/
  model-registry/
  docs/
  examples/
```

- Maintain one-way dependency rule: daemon must not import heavy ML dependencies.

## 3) Canonical forecasting contract [x]
### Current implementation status
- Canonical schemas are implemented in `ForecastRequest`, `ForecastResponse`, `SeriesInput`, and `SeriesForecast`.
- Auto-selection schemas are implemented in `AutoForecastRequest`, `AutoSelectionInfo`, and
  `AutoForecastResponse`.
- Canonical request shape in production includes:
  - `model`, `horizon`, `series[]`, optional `quantiles[]`, optional `options`, optional `parameters`.
  - `series[]` supports `id`, `freq` (defaults to `"auto"`), `timestamps[]`, `target[]`,
    optional `actuals`, `past_covariates`, `future_covariates`, `static_covariates`.
  - per-covariate values are validated as homogeneous numeric or homogeneous string arrays.
- Canonical response shape in production includes:
  - per-series `start_timestamp`, `mean[]`, optional `quantiles{q: []}`
  - optional top-level `metrics` (`aggregate` + per-series metric values)
  - optional top-level `timing` (`model_load_ms`, `inference_ms`, `total_ms`)
  - optional top-level `explanation` (per-series trend/confidence/pattern summary)
  - optional top-level `warnings[]`
  - optional `usage` (`runner`, `device`, `peak_memory_mb`, plus adapter-specific keys)
- Implemented request parameters include:
  - `covariates_mode = "best_effort" | "strict"` (default `best_effort`)
  - `timesfm` knobs: `xreg_mode`, `ridge`, `force_on_cpu`
  - optional `metrics`:
    - `names` supports `mape`, `mase`, `mae`, `rmse`, `smape`, `wape`, `rmsse`, `pinball`
    - `mase_seasonality` default `1`
    - requires `series.actuals` with length `horizon`

### Planned work / TODO
- Add explicit compatibility/versioning policy for canonical schema evolution.
- Enable runner-level static covariate support (daemon pass-through/filtering is in place).

## 4) Internal communication: daemon <-> runner protocol [~]
### Current implementation status
- JSON-over-stdio line protocol is implemented in `tollama.core.protocol`.
- Request/response primitives are implemented (`ProtocolRequest`, `ProtocolResponse`).
- Supported method set includes `load`, `unload`, `forecast`, `hello`.
- Active handlers today:
  - `mock`: `hello`, `forecast`
  - `torch`: `hello`, `load`, `unload`, `forecast`
  - `timesfm`: `hello`, `unload`, `forecast`
  - `uni2ts`: `hello`, `unload`, `forecast`
  - `sundial`: `hello`, `unload`, `forecast`
  - `toto`: `hello`, `unload`, `forecast`

### Planned work / TODO
- Standardize mandatory startup handshake semantics and health checks across all families.
- Standardize capabilities negotiation semantics end-to-end (daemon + runners).
- Publish stable RPC method and error-code compatibility contract.

## 5) Model registry + manifests [x]
### Current implementation status
- Registry exists at `model-registry/registry.yaml` with `name`, `family`, `source`, `license`, `defaults`.
- Registry model entries now include covariate capabilities metadata.
- Pull/install writes model manifest to `~/.tollama/models/<name>/manifest.json`.
- Pull flow resolves and persists:
  - `resolved.commit_sha`
  - `resolved.snapshot_path`
  - `size_bytes`
  - `pulled_at`
  - `installed_at`
  - `license.accepted`
  - `capabilities` (if present in registry)

### Planned work / TODO
- Add compatibility metadata (min runner version, feature flags, constraints).

## 6) Model store layout [~]
### Current implementation status
- Existing paths are defined around `~/.tollama/` via `TollamaPaths`.
- Implemented directories/files include:
  - `models/` manifests + snapshots
  - `config.json`
  - `runtimes/` path reserved in core paths

### Planned work / TODO
- Formalize and operationalize additional product layout:
  - `cache/`
  - `logs/`
  - `licenses/`
- Define cleanup, retention, and lifecycle rules for each directory.

## 7) Runner environments (dependency isolation) [x]
### Current implementation status
- Family routing supports separate command targets per family.
- Optional extras are available for each heavy family:
  - `runner_torch`
  - `runner_timesfm`
  - `runner_uni2ts`
  - `runner_sundial`
  - `runner_toto`
- Default `mock` and `torch` launches use interpreter-module commands.
- `timesfm` and `uni2ts` launch via console entrypoints intended for separate env/runtime installs.
- **Per-family venv auto-bootstrap** implemented under `~/.tollama/runtimes/<family>/venv/`.
  - `core/runtime_bootstrap.py` handles venv creation, pip install, staleness detection.
  - `RunnerManager` integrates auto-bootstrap: when `daemon.auto_bootstrap` is enabled,
    runner subprocesses use an isolated venv Python interpreter.
  - Staleness detection compares `tollama_version` in `installed.json` state file.
  - Manual override via `config.json` `daemon.runner_commands` takes precedence.
- CLI `tollama runtime` subcommand group:
  - `tollama runtime install <family>` / `--all` — eagerly create isolated venvs.
  - `tollama runtime remove <family>` / `--all` — delete isolated venvs.
  - `tollama runtime update <family>` / `--all` — reinstall to pick up version changes.
  - `tollama runtime list` — show per-family runtime status.

### Planned work / TODO
- Add progress reporting for long-running pip install during auto-bootstrap.
- Add support for specifying a custom Python interpreter per family in config.
- Make single-install UX product-grade while preserving dependency isolation.

## 8) Daemon supervisor lifecycle & VRAM reclaim [~]
### Current implementation status
- Supervisor supports lazy start, stop, restart-on-failure, timeout handling, and status inspection.
- Daemon tracks loaded models and supports keep-alive-driven unload behavior.
- Expired keep-alive entries can trigger runner stop and memory reclaim.

### Planned work / TODO
- Add explicit restart backoff policy with bounded retries.
- Add proactive handshake/health checks before serving forecast calls.
- Add default idle timeout policy and configurable thresholds.
- Add explicit GPU bad-state recovery policy and telemetry.

## 9) Request routing strategy [x]
### Current implementation status
- `POST /api/forecast` and `POST /v1/forecast` flow:
  - validate canonical request
  - resolve local manifest
  - map model to family
  - normalize covariates and classify known-future vs past-only
  - apply model capability preflight (`best_effort` warnings or `strict` 400)
  - dispatch to runner via manager/supervisor
  - merge runner warnings with daemon warnings
  - map failure classes to HTTP status
- Missing models are handled as `404`.
- Available and installed model views are exposed via `GET /v1/models`.

### Planned work / TODO
- Expose per-model runtime readiness and capabilities in model listing APIs.
- Consider dedicated `GET /v1/models/{name}/capabilities` endpoint for explicit API discovery.

## 10) Runner internals [~]
### Current implementation status
- `mock`, `torch`, `timesfm`, `uni2ts`, `sundial`, and `toto` runner loops are implemented with stdio RPC handling.
- Torch runner router supports:
  - `ChronosAdapter` using `predict_df` + history/future dataframes
  - `GraniteTTMAdapter` using `future_time_series`, `control_columns`, and `conditional_columns`
- TimesFM runner supports:
  - base `forecast(...)` path
  - covariate `forecast_with_covariates(...)` path
  - `return_backcast=True` compile requirement for covariates
- Uni2TS runner supports:
  - numeric dynamic covariates via `feat_dynamic_real`
  - numeric past covariates via `past_feat_dynamic_real`
- Sundial runner supports:
  - sample-based generation with canonical `mean`/`quantiles` projection
  - forecast/unload/hello RPC parity with other families
- Toto runner supports:
  - variate-building adapter path and canonical output shaping
  - forecast/unload/hello RPC parity with other families

### Planned work / TODO
- Add cache policy controls (LRU, max loaded models, memory thresholds).
- Add richer device and memory policy control per family.
- Expand runner-side capabilities reporting beyond static registry metadata.

## 11) Public API + CLI [x]
### Current implementation status
- Implemented HTTP endpoints:
  - `GET /metrics` (requires optional `prometheus-client`)
  - `GET /v1/health`
  - `GET /v1/models`
  - `POST /v1/models/pull`
  - `DELETE /v1/models/{name}`
  - `POST /v1/forecast`
  - `GET /api/version`
  - `GET /api/info`
  - `GET /api/tags`
  - `POST /api/show`
  - `POST /api/pull`
  - `DELETE /api/delete`
  - `GET /api/ps`
  - `POST /api/forecast`
  - `POST /api/auto-forecast`
  - `POST /api/analyze`
  - `POST /api/what-if`
  - `POST /api/compare`
- `GET /api/info` includes:
  - installed model capabilities
  - available model capabilities
  - runner statuses
- CLI command surface today:
  - `serve`, `quickstart`, `pull`, `list`, `ps`, `show`, `rm`, `run`, `info`, `config`
- CLI behavior includes:
  - warning output for forecast responses that include `warnings[]`
  - covariates capability summaries in `tollama info`

### Planned work / TODO
- Add CLI `forecast` UX alias (retain `run` for compatibility).
- Expand v1-first API parity docs around capabilities and compatibility contracts.

## 12) Licensing gates [~]
### Current implementation status
- Registry carries per-model license metadata and acceptance requirement.
- API/model install paths enforce acceptance and return `409` when required acceptance is missing.
- Manifest records acceptance state.
- CLI supports `--accept-license` on pull/install paths.
- Public-release checklist now documents license/compliance gates, including upstream registry license validation and third-party inventory generation (`docs/public-release-checklist.md`).

### Planned work / TODO
- Record dedicated license receipts under `~/.tollama/licenses/<model>.json`.
- Enforce acceptance checks consistently before runtime execution for all families.

## 13) Observability + robustness [~]
### Current implementation status
- `/api/info` provides diagnostics payload with daemon/model/runner/config/environment views.
- Sensitive values are redacted in diagnostics payloads.
- Runner status reports include install/running state, restart count, and last error.
- Forecast/pull paths map several failure classes to `400/404/409/502/503`.
- Forecast endpoints support optional accuracy metrics
  (`mape`, `mase`, `mae`, `rmse`, `smape`, `wape`, `rmsse`, `pinball`) in response payloads.
- Forecast responses include timing + enriched usage metadata and deterministic explainability payloads.

### Planned work / TODO
- Add structured logging for routing decisions, load/unload timings, and crash recovery.
- Add deeper telemetry for model load latency and inference latency distributions.

## 14) Testing plan [x]
### Current implementation status
- Unit and integration-style coverage exists for core schemas/protocol, daemon, CLI, supervisor, runner manager, and adapters.
- Added fast covariates-focused tests (no HF weight downloads):
  - schema/validation failure scenarios
  - strict vs best-effort compatibility behavior
  - Chronos/Granite/TimesFM/Uni2TS adapter wiring via mocks/helpers
- Baseline checks currently pass:
  - `ruff check .`
  - `pytest -q` (heavy integration tests stay opt-in)
- Optional real-model integration matrix was re-validated on `2026-02-17`:
  - pass: `chronos2`, `granite-ttm-r2`, `timesfm-2.5-200m`, `moirai-2.0-R-small`,
    `sundial-base-128m`
  - skipped: `toto-open-base-1.0` when `toto` dependency is not installed
- Per-family runtime isolation smoke (`tollama runtime install --all` + one
  forecast per family) was re-validated on `2026-02-17` after updating the
  TimesFM dependency pin to commit `2dcc66fbfe2155adba1af66aa4d564a0ee52f61e`:
  - pass: `chronos2`, `granite-ttm-r2`, `timesfm-2.5-200m`,
    `moirai-2.0-R-small`, `sundial-base-128m`, `toto-open-base-1.0`
- End-to-end regression was re-validated on `2026-02-20`:
  - pass: per-family forecasting suite (`bash scripts/e2e_all_families.sh`)
  - pass: OpenClaw skill suite (`bash scripts/e2e_skills_test.sh`)
  - pass: MCP SDK stdio smoke (`tollama-mcp` tool calls:
    `tollama_health`, `tollama_models`, `tollama_show`, `tollama_forecast`)
- Phase 4 feature set was re-validated on `2026-02-19`:
  - pass: OpenClaw skill regression (`bash scripts/e2e_skills_test.sh`)
  - pass: metrics expansion live daemon checks (`/api/forecast` non-stream):
    `mape`, `mase`, `mae`, `rmse`, `smape`, `wape`, `rmsse`, `pinball` aggregate +
    undefined-metric warning paths
  - pass: LangChain wrapper live invocation (`get_tollama_tools`):
    `tollama_health`, `tollama_models`, `tollama_forecast`, invalid-request mapping
  - pass: LangChain wrapper test file in LangChain-enabled venv
    (`PYTHONPATH=src python -m pytest -q tests/test_langchain_skill.py`)

### Planned work / TODO
- Add restart backoff and idle-timeout policy behavior tests.
- Add per-family runtime bootstrap/install tests.
- Add CLI license-receipt workflow tests once receipts are implemented.

## 15) Phased build checklist [~]
### Current implementation status
Phase A - Foundation:
- `[x]` `tollama-core` canonical schemas + protocol
- `[x]` `tollamad` with `/v1/forecast` routing to mock/real runners
- `[x]` CLI that calls daemon
- `[~]` supervisor hardening for idle/backoff behavior

Phase B - First real runner: Torch family:
- `[x]` `runner-torch` with Chronos adapter
- `[x]` `runner-torch` with Granite adapter
- `[x]` `tollama pull` downloads model + writes manifest metadata
- `[~]` model cache/LRU/device behavior

Phase C - Capability-aware covariates:
- `[x]` unified `past_covariates` + `future_covariates` contract
- `[x]` strict/best-effort compatibility preflight
- `[x]` capabilities in registry, manifests, `/api/info`, and CLI info summaries

Phase D - TimesFM runner:
- `[x]` separate runner implementation + adapter
- `[x]` canonical request/response parity
- `[x]` covariate xreg path with compile constraints

Phase E - Uni2TS/Moirai runner + license gating:
- `[x]` separate Uni2TS runner implementation
- `[x]` canonical request/response parity
- `[~]` license gating and acceptance UX (receipts still pending)

Phase F - Product hardening:
- `[ ]` crash recovery + backoff policy
- `[ ]` batching behavior improvements
- `[ ]` memory limit policies (max loaded models / VRAM strategy)
- `[~]` docs + examples for app developers

### Planned work / TODO
- Complete phase F hardening after multi-family baseline stability.
- Keep section 14 tests as acceptance gates for future milestone flips.

## 16) Practical defaults [~]
### Current implementation status
- Daemon defaults to local binding (`127.0.0.1`) and port `11435`.
- Forecast schemas enforce strict typing and deterministic response shaping.
- Keep-alive semantics are supported for model unload behavior.
- Pull defaults resolve from env/config/defaults with redaction in diagnostics.
- Covariates default behavior is `best_effort`, with warnings surfaced to API and CLI.
- `POST /api/validate` and `tollama run --dry-run` are implemented for no-inference request checks.
- `tollama doctor` is implemented (`pass/warn/fail` checks + JSON mode).
- `tollama list` / `tollama ps` default to table output with `--json` compatibility mode.

### Planned work / TODO
- Standardize default quantiles to `[0.1, 0.5, 0.9]`.
- Set product defaults for runner idle timeout and hot-model cache limits.

## 17) OpenClaw skill integration [x]
### Current implementation status
- OpenClaw skill package is added under `skills/tollama-forecast/` with:
  - `SKILL.md` (`requires.bins=["bash"]`, `requires.anyBins=["tollama","curl"]`)
  - `bin/_tollama_lib.sh` (shared helpers + structured stderr emitter)
  - `bin/tollama-health.sh`
  - `bin/tollama-models.sh`
  - `bin/tollama-forecast.sh`
  - `bin/tollama-pull.sh`
  - `bin/tollama-rm.sh`
  - `bin/tollama-info.sh`
  - `openai-tools.json` (OpenAI function definitions)
  - `examples/*.json`
- Integration is additive and keeps daemon/CLI HTTP contracts unchanged.
- Runtime guardrails for common failures are included:
  - base URL precedence and host mismatch hints
  - PATH/command availability handling
  - default timeout `300` with `TOLLAMA_FORECAST_TIMEOUT_SECONDS` override
  - no-auto-pull default (`--pull` required)
  - `/api/forecast` primary with `/v1/forecast` fallback only on `404`
  - daemon-only `available` resolution via `/api/info` or `tollama info --json --remote`
  - `tollama-health.sh --runtimes` enriches health output with runtime install/running state
- Skill implementation contracts are documented and aligned with code:
  - `_tollama_lib.sh` centralizes error classification + HTTP helper + JSON stderr emitter
  - `tollama-forecast.sh` handles metrics flags (`--metrics`, `--mase-seasonality`) and
    performs CLI-first forecast with HTTP fallback when CLI is unavailable
  - `tollama-models.sh` is lifecycle multiplexer; thin wrappers
    (`tollama-pull.sh`, `tollama-rm.sh`, `tollama-info.sh`) delegate to it
  - `tollama-health.sh` is curl-based health/version probe with optional runtimes
- Skill scripts share exit code contract v2:
  - `0` success
  - `2` invalid input/request
  - `3` daemon unreachable/health failure
  - `4` model not installed
  - `5` license/permission
  - `6` timeout
  - `10` unexpected internal error
- Optional structured stderr mode is available with `TOLLAMA_JSON_STDERR=1`.
- CI now runs `scripts/validate_openclaw_skill_tollama_forecast.sh`.
- Skill E2E re-validation passed on `2026-02-20` via `scripts/e2e_skills_test.sh`.

### Planned work / TODO
- Completed: end-to-end OpenClaw agent runbooks are documented:
  - `docs/openclaw-sandbox-runbook.md`
  - `docs/openclaw-gateway-runbook.md`

## 18) MCP integration for Claude Code [~]
### Current implementation status
- Shared HTTP client package added under `src/tollama/client/` and reused by CLI/MCP.
- HTTP client contracts (`src/tollama/client/http.py`):
  - default base URL `http://localhost:11435`, default timeout `10s`
  - endpoint coverage: health/version, tags/ps/info, show/pull/delete,
    forecast/auto-forecast/analyze/what-if/compare, validate
  - HTTP/status/request failures mapped into typed exceptions with category metadata
    (`INVALID_REQUEST`, `DAEMON_UNREACHABLE`, `MODEL_MISSING`, `LICENSE_REQUIRED`,
    `PERMISSION_DENIED`, `TIMEOUT`, `INTERNAL_ERROR`)
- MCP server scaffold added under `src/tollama/mcp/`:
  - `server.py`, `tools.py`, `schemas.py`, `__main__.py`
  - tool set:
    `tollama_health`, `tollama_models`, `tollama_forecast`, `tollama_auto_forecast`,
    `tollama_analyze`, `tollama_what_if`, `tollama_compare`, `tollama_recommend`,
    `tollama_pull`, `tollama_show`
  - each tool now includes rich MCP descriptions with required inputs, model-name examples,
    and invocation examples for agent discoverability
- MCP tool behavior/contracts:
  - strict input schemas (`extra="forbid"`, strict scalar typing, positive timeout)
  - `tollama_models(mode=installed|loaded|available)` mapped to `/api/tags|/api/ps|/api/info`
  - `tollama_forecast` is non-streaming and validates request via `ForecastRequest`
  - `tollama_auto_forecast` validates request via `AutoForecastRequest`
  - `tollama_analyze` validates request via `AnalyzeRequest`
  - `tollama_what_if` validates request via `WhatIfRequest`
  - tool failures are emitted as JSON payload with `{error:{category,exit_code,message}}`
- Optional dependency bundle added in `pyproject.toml`:
  - `.[mcp]` with `mcp>=1.0`
  - script entrypoint `tollama-mcp`
- Claude Desktop helper script added:
  - `scripts/install_mcp.sh` (upsert `mcpServers.<name>` + `env.TOLLAMA_BASE_URL`)
- Agent context doc added:
  - `CLAUDE.md`
- Optional LangChain SDK wrapper added under `src/tollama/skill/langchain.py`:
  - `TollamaForecastTool`, `TollamaAutoForecastTool`, `TollamaAnalyzeTool`,
    `TollamaCompareTool`, `TollamaRecommendTool`, `TollamaHealthTool`, `TollamaModelsTool`
  - `get_tollama_tools(base_url="http://127.0.0.1:11435", timeout=10.0)`
  - optional extra `.[langchain]` with `langchain-core`
  - tool descriptions now include schema guidance, model-name examples, and invocation examples
  - async tool paths now use real `_arun` implementations backed by
    `AsyncTollamaClient` (no sync `_run` delegation stubs)
- Additional agent framework wrappers added under `src/tollama/skill/`:
  - `crewai.py`: `get_crewai_tools(...)`
  - `autogen.py`: `get_autogen_tool_specs(...)`,
    `get_autogen_function_map(...)`, `register_autogen_tools(...)`
  - `smolagents.py`: `get_smolagents_tools(...)`
  - shared framework-neutral contracts in `framework_common.py`
- Benchmark script added for SDK-vs-raw comparison:
  - `benchmarks/tollama_vs_raw.py` reports effective LOC and time-to-first-forecast
- High-level Python SDK convenience facade added:
  - `src/tollama/sdk.py` with `Tollama` + `TollamaForecastResult`
  - package export `from tollama import Tollama`
  - supports dict/list/pandas Series/DataFrame series inputs and `result.to_df()`
- Onboarding quickstart command added:
  - `tollama quickstart` validates daemon reachability, pulls a model, runs demo forecast,
    and prints next-step commands
- Notebooks added for discoverability:
  - `examples/quickstart.ipynb`
  - `examples/agent_demo.ipynb`
- Container packaging baseline added:
  - `Dockerfile`
  - `.dockerignore`
- PyPI metadata/readiness updates:
  - `pyproject.toml` now includes `keywords`, `classifiers`, and `project.urls`
- Focused validation added:
  - `tests/test_client_http.py`
  - `tests/test_mcp_tools.py`
  - `tests/test_mcp_entrypoint.py`
  - `tests/test_langchain_skill.py`
  - `tests/test_agent_wrappers.py`
  - `tests/test_benchmark_tollama_vs_raw.py`
  - `tests/test_sdk.py`
- MCP end-to-end SDK smoke passed on `2026-02-20` with stdio transport and live daemon.

### Planned work / TODO
- Add tool-level auth/session policy guidance once deployment target is finalized.
- Add explicit client-facing MCP tool schema docs with sample request/response payloads.

## Prioritized TODO backlog
1. Implement supervisor restart backoff policy and startup handshake/health checks.
2. ~~Add per-family runtime bootstrap/install automation under `~/.tollama/runtimes/`.~~ ✓ Implemented.
3. Add runtime metrics endpoint and structured runtime telemetry.
4. Add cache/memory policy controls (LRU + limits + reclaim behavior).
5. Enable static covariates in runner adapters/capability flags (daemon-side pass-through done).
6. Add explicit license receipt files under `~/.tollama/licenses/`.
7. Add dedicated per-model capabilities endpoint for API consumers.
8. Expand developer docs and operational playbooks.

## Acceptance checklist for roadmap updates
- `[x]` Status changes in this update are tied to concrete code paths and tests.
- `[x]` `Granite`, `timesfm`, and `uni2ts` shipment markers now reflect implementation state.
- `[x]` Endpoint inventory in section 11 is aligned with current API routes.
- `[x]` Phased checklist in section 15 reflects current delivery status.
- `[x]` Baseline verification state (`ruff check .`, `pytest -q`) is reflected from latest passing run.
- `[x]` Optional real-model integration matrix status is reflected with date and outcomes.
- `[x]` Migration notes in section 2 remain aligned with active repository layout.
