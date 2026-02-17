# tollama roadmap (worker-per-model-family)

Last updated: 2026-02-17

Status legend:
- `[x]` implemented
- `[~]` partially implemented
- `[ ]` planned

This roadmap is implementation-aware for the repository under
`/Users/yongchoelchoi/Documents/GitHub/tollama/src/tollama/*` and still tracks
the optional future `packages/*` split as a migration phase.

## 0) Product goals [~]
### Current implementation status
- Unified forecasting endpoint is available at `POST /api/forecast` and `POST /v1/forecast`.
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
- Canonical request shape in production includes:
  - `model`, `horizon`, `series[]`, optional `quantiles[]`, optional `options`, optional `parameters`.
  - `series[]` supports `id`, `freq`, `timestamps[]`, `target[]`, `past_covariates`, `future_covariates`.
  - per-covariate values are validated as homogeneous numeric or homogeneous string arrays.
- Canonical response shape in production includes:
  - per-series `start_timestamp`, `mean[]`, optional `quantiles{q: []}`
  - optional top-level `warnings[]`
  - optional `usage`
- Implemented request parameters include:
  - `covariates_mode = "best_effort" | "strict"` (default `best_effort`)
  - `timesfm` knobs: `xreg_mode`, `ridge`, `force_on_cpu`

### Planned work / TODO
- Add explicit compatibility/versioning policy for canonical schema evolution.
- Add static covariates support in canonical adapters (currently planned).

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
- `GET /api/info` includes:
  - installed model capabilities
  - available model capabilities
  - runner statuses
- CLI command surface today:
  - `serve`, `pull`, `list`, `ps`, `show`, `rm`, `run`, `info`, `config`
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

### Planned work / TODO
- Record dedicated license receipts under `~/.tollama/licenses/<model>.json`.
- Enforce acceptance checks consistently before runtime execution for all families.

## 13) Observability + robustness [~]
### Current implementation status
- `/api/info` provides diagnostics payload with daemon/model/runner/config/environment views.
- Sensitive values are redacted in diagnostics payloads.
- Runner status reports include install/running state, restart count, and last error.
- Forecast/pull paths map several failure classes to `400/404/409/502/503`.

### Planned work / TODO
- Add `/metrics` endpoint with Prometheus-friendly counters/histograms.
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

### Planned work / TODO
- Standardize default quantiles to `[0.1, 0.5, 0.9]`.
- Always include usage timing payloads:
  - `usage.model_load_ms`
  - `usage.inference_ms`
  - `usage.runner`
  - `usage.device`
- Set product defaults for runner idle timeout and hot-model cache limits.

## Prioritized TODO backlog
1. Implement supervisor restart backoff policy and startup handshake/health checks.
2. ~~Add per-family runtime bootstrap/install automation under `~/.tollama/runtimes/`.~~ ✓ Implemented.
3. Add metrics endpoint and structured runtime telemetry.
4. Add cache/memory policy controls (LRU + limits + reclaim behavior).
5. Add static covariates support and capability flags.
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
