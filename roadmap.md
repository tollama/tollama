# tollama roadmap (worker-per-model-family)

Last updated: 2026-02-16

Status legend:
- `[x]` implemented
- `[~]` partially implemented
- `[ ]` planned

This roadmap is implementation-aware for the current repository state under
`/Users/yongchoelchoi/Documents/GitHub/tollama/src/tollama/*` and tracks the
future `packages/*` split as a migration phase.

## 0) Product goals [~]
### Current implementation status
- Unified forecasting endpoint is available at `POST /v1/forecast`.
- Ollama-style model lifecycle is available (`pull`, `list`, `rm`) via HTTP and CLI.
- Forecast routing uses model-family worker selection from installed manifests.
- v1 non-goals are currently respected: no training/fine-tuning, no distributed scheduler, no multi-tenant auth.

### Planned work / TODO
- Complete multi-family runner support (`timesfm`, `uni2ts`) with production adapters.
- Strengthen VRAM reclaim policy with explicit idle strategy and crash recovery behavior.
- Keep local-first product posture for v1.

## 1) Top-level architecture [~]
### Current implementation status
- `tollamad` (`src/tollama/daemon/`) owns public HTTP API and process supervision.
- Runners (`src/tollama/runners/`) communicate over stdio JSON lines.
- Shared core (`src/tollama/core/`) provides schemas, protocol, registry/storage/config helpers.
- CLI (`src/tollama/cli/`) provides user commands and daemon HTTP client integration.
- Active runner implementations: `mock` and `torch` (Chronos path).

### Planned work / TODO
- Add production runner implementations for `timesfm` and `uni2ts`.
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
```

### Planned work / TODO
- Track this target product-grade split as an explicit migration phase:

```text
tollama/
  packages/
    tollama-core/
      tollama_core/
        schemas.py
        protocol.py
        storage.py
        registry.py
        utils/
    tollamad/
      tollamad/
        api.py
        routing.py
        supervisor.py
        model_manager.py
        settings.py
    tollama-cli/
      tollama/
        cli.py
        client.py
    tollama-runner-torch/
      runner_torch/
        main.py
        chronos_adapter.py
        granite_adapter.py
        model_cache.py
    tollama-runner-timesfm/
      runner_timesfm/
        main.py
        timesfm_adapter.py
        model_cache.py
    tollama-runner-uni2ts/
      runner_uni2ts/
        main.py
        moirai_adapter.py
        model_cache.py
  model-registry/
    registry.yaml
  docs/
  examples/
  pyproject.toml
```

- Maintain one-way dependency rule: daemon must not import heavy ML dependencies.

## 3) Canonical forecasting contract [x]
### Current implementation status
- Canonical schemas are implemented in `ForecastRequest`, `ForecastResponse`, `SeriesInput`, and `SeriesForecast`.
- Strict validation is enforced (required fields, quantile ordering/uniqueness, length checks).
- Canonical request shape in production includes:
  - `model`, `horizon`, `series[]`, optional `quantiles[]`, optional `options`.
  - `series[]` supports `id`, `freq`, `timestamps[]`, `target[]`, and optional covariates.
- Canonical response shape in production includes:
  - per-series `start_timestamp`, `mean[]`, optional `quantiles{q: []}`, and optional `usage`.

### Planned work / TODO
- Add explicit compatibility/versioning policy for canonical schema evolution.
- Add contract-level compatibility notes for clients integrating across releases.

## 4) Internal communication: daemon <-> runner protocol [~]
### Current implementation status
- JSON-over-stdio line protocol is implemented in `tollama.core.protocol`.
- Request/response primitives are implemented (`ProtocolRequest`, `ProtocolResponse`).
- Supported method set currently includes `capabilities`, `load`, `unload`, `forecast`, `ping`, `hello`.
- Active handlers today:
  - `mock`: `hello`, `forecast`
  - `torch`: `hello`, `load`, `unload`, `forecast`

### Planned work / TODO
- Standardize mandatory startup handshake semantics (`HELLO`) and runtime health checks (`PING`).
- Standardize and enforce explicit `capabilities` usage end-to-end.
- Publish stable RPC method and error-code compatibility contract.

## 5) Model registry + manifests [x]
### Current implementation status
- Registry exists at `model-registry/registry.yaml` with `name`, `family`, `source`, `license`, and optional `defaults`.
- Pull/install writes model manifest to `~/.tollama/models/<name>/manifest.json`.
- Pull flow resolves and persists:
  - `resolved.commit_sha`
  - `resolved.snapshot_path`
  - `size_bytes`
  - `pulled_at`
  - `installed_at`
  - `license.accepted`

### Planned work / TODO
- Add richer runtime hints (context length, max horizon, family capabilities).
- Add compatibility metadata (min runner version, feature flags, constraints).

## 6) Model store layout [~]
### Current implementation status
- Existing paths are defined around `~/.tollama/` via `TollamaPaths`.
- Implemented directories/files include:
  - `models/` manifests + snapshots
  - `config.json`
  - `runtimes/` path reserved in core paths

### Planned work / TODO
- Formalize and operationalize the full product layout:
  - `cache/`
  - `logs/`
  - `licenses/`
- Define cleanup, retention, and lifecycle rules for each directory.

## 7) Runner environments (dependency isolation) [~]
### Current implementation status
- Optional torch dependency universe is available through `runner_torch` extras.
- Family routing supports separate command targets per family.
- Default `mock` and `torch` runner launches now use the daemon interpreter (`sys.executable -m ...`) instead of PATH-only console-script lookup, reducing environment mismatch failures (`src/tollama/daemon/runner_manager.py`).
- `timesfm` and `uni2ts` families are already modeled in routing/registry metadata but not implemented as runnable workers.

### Planned work / TODO
- Implement per-family runtime bootstrap under `~/.tollama/runtimes/`.
- Add automated install/repair flows for each runner family.
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
- `POST /v1/forecast` flow is implemented:
  - validate canonical request
  - resolve local manifest
  - map model to family
  - dispatch to runner via manager/supervisor
  - map runner failure classes to HTTP status
- Missing models are handled as `404`.
- Available and installed model views are exposed via `GET /v1/models`.

### Planned work / TODO
- Add capability-aware preflight validation before runner dispatch.
- Add richer user guidance for unsupported options/model-feature mismatches.
- Expose per-model runtime readiness and capabilities in model listing APIs.

## 10) Runner internals [~]
### Current implementation status
- `mock` and `torch` runner loops are implemented with stdio RPC handling.
- `ChronosAdapter` is implemented in torch runner.
- Basic in-memory model cache exists in torch adapter (dictionary cache of loaded pipelines).

### Planned work / TODO
- Add cache policy controls (LRU, max loaded models, memory thresholds).
- Add `GraniteAdapter` to torch runner.
- Implement `timesfm` and `uni2ts` runners with canonical conversion layers.
- Add first-class capabilities reporting and daemon exposure.

## 11) Public API + CLI [~]
### Current implementation status
- Implemented HTTP endpoints:
  - `/v1/health`
  - `/v1/models`
  - `/v1/models/pull`
  - `/v1/models/{name}`
  - `/v1/forecast`
  - `/api/version`
  - `/api/info`
  - `/api/tags`
  - `/api/show`
  - `/api/pull`
  - `/api/delete`
  - `/api/ps`
  - `/api/forecast`
- Canonical schema types in use:
  - `ForecastRequest`
  - `ForecastResponse`
  - `SeriesInput`
  - `SeriesForecast`
  - `keep_alive` is supported as an Ollama-compatible forecast extension.
- Internal RPC types in use:
  - `ProtocolRequest`
  - `ProtocolResponse`
- Model metadata contracts in use:
  - Registry fields: `name`, `family`, `source`, `license`, `defaults`
  - Manifest fields: `resolved.commit_sha`, `resolved.snapshot_path`, `size_bytes`, `pulled_at`, `installed_at`, `license.accepted`
- CLI command surface today:
  - `serve`
  - `pull`
  - `list`
  - `ps`
  - `show`
  - `rm`
  - `run`
  - `info`
  - `config`

### Planned work / TODO
- Add CLI `forecast` UX alias (retain `run` for compatibility).
- Expand v1-first API parity and docs around capabilities.
- Add `GET /v1/models/{name}/capabilities` and align routing preflight with it.

## 12) Licensing gates [~]
### Current implementation status
- Registry carries per-model license metadata and acceptance requirement.
- API/model install paths enforce acceptance and return `409` when required acceptance is missing.
- Manifest records acceptance state.

### Planned work / TODO
- Add explicit CLI `--accept-license` workflows across pull/install paths.
- Record dedicated license receipts under `~/.tollama/licenses/<model>.json`.
- Enforce acceptance checks consistently before runtime execution for all families.

## 13) Observability + robustness [~]
### Current implementation status
- `/api/info` provides diagnostics payload with runner/model/config/environment views.
- Sensitive values are redacted in diagnostics paths.
- Runner status reports include install/running state, restart count, and last error.
- Forecast/pull paths map several failure classes to `400/404/409/502/503`.

### Planned work / TODO
- Add `/metrics` endpoint with Prometheus-friendly counters/histograms.
- Add structured logging for routing decisions, load/unload timings, and crash recovery.
- Add deeper telemetry for model load latency and inference latency distributions.

## 14) Testing plan [x]
### Current implementation status
- Unit and integration-style coverage exists for core schemas/protocol, daemon, CLI, supervisor, and runner manager behavior.
- Baseline checks currently pass:
  - `ruff check .`
  - `pytest -q` (integration Chronos test skipped unless explicitly enabled)
- Validated behavior scenarios include:
  - `/v1/forecast` success and schema failure (`400`)
  - missing model (`404`)
  - license-not-accepted (`409`)
  - runner unavailable (`503`)
  - runner protocol/call failures (`502`)
  - pull streaming NDJSON and non-stream flow
  - config/env pull default resolution
  - diagnostics redaction via `/api/info`
  - runner-manager default command behavior for interpreter-module launches (`tests/test_runner_manager.py`)

### Planned work / TODO
- Add test coverage for future TimesFM and Uni2TS forecast paths.
- Add capabilities endpoint and family-feature preflight tests.
- Add restart backoff and idle-timeout policy behavior tests.
- Add CLI license-acceptance UX and receipt-file tests.

## 15) Phased build checklist [~]
### Current implementation status
Phase A — Foundation:
- `[x]` `tollama-core` canonical schemas + protocol
- `[x]` `tollamad` with `/v1/forecast` routing to a mock runner
- `[x]` CLI that calls daemon
- `[~]` supervisor hardening for idle/backoff behavior (core lifecycle exists, policy hardening pending)

Phase B — First real runner: Torch family:
- `[x]` `runner-torch` with Chronos adapter
- `[x]` `tollama pull` downloads model + writes manifest metadata
- `[x]` daemon default launch path for torch/mock runners is interpreter-module based (no PATH-only entrypoint dependency)
- `[~]` model cache/LRU/device behavior (basic cache exists, no LRU or explicit device policy)

Phase C — Add IBM Granite TSFM to torch runner:
- `[ ]` Granite adapter in torch runner
- `[ ]` capabilities reporting for Chronos vs Granite differences

Phase D — TimesFM runner:
- `[ ]` separate runner env + adapter
- `[ ]` canonical request/response parity
- `[ ]` clean horizon/context limit validation behavior

Phase E — Uni2TS/Moirai runner + license gating:
- `[ ]` separate Uni2TS runner implementation
- `[~]` license gating core behavior
- `[ ]` CLI acceptance flow + explicit receipt files

Phase F — Product hardening:
- `[ ]` crash recovery + backoff policy
- `[ ]` batching behavior improvements
- `[ ]` memory limit policies (max loaded models / VRAM strategy)
- `[~]` docs + examples for app developers

### Planned work / TODO
- Deliver phases C, D, and E in sequence to unlock true multi-family support.
- Complete phase F hardening after multi-family baseline is stable.
- Require section 14 test scenarios as acceptance gates before flipping `[~]` to `[x]`.

## 16) Practical defaults [~]
### Current implementation status
- Daemon defaults to local binding (`127.0.0.1`) and port `11435`.
- Forecast schemas enforce strict typing and deterministic response shaping.
- Keep-alive semantics are supported for model unload behavior.
- Pull defaults resolve from env/config/defaults with redaction support in diagnostics.

### Planned work / TODO
- Standardize default quantiles to `[0.1, 0.5, 0.9]`.
- Always include usage timing payloads:
  - `usage.model_load_ms`
  - `usage.inference_ms`
  - `usage.runner`
  - `usage.device`
- Set product defaults for runner idle timeout and hot-model cache limits.

## Prioritized TODO backlog
1. Implement `timesfm` runner package and adapter with canonical I/O parity.
2. Implement `uni2ts` runner package and adapter with canonical I/O parity.
3. Add capability negotiation and expose `GET /v1/models/{name}/capabilities`.
4. Implement supervisor restart backoff policy and startup handshake/health checks.
5. Add per-family runtime bootstrap/install automation under `~/.tollama/runtimes/`.
6. Implement Granite adapter in torch runner and capability matrix.
7. Add CLI license acceptance UX and explicit `~/.tollama/licenses/*.json` receipts.
8. Add metrics endpoint and structured runtime telemetry.
9. Add cache/memory policy controls (LRU + limits + reclaim behavior).
10. Expand developer docs/examples for app integration and operational playbooks.

## Acceptance checklist for roadmap updates
- `[ ]` Every status change links to concrete code paths and tests.
- `[ ]` No roadmap claim marks `timesfm`, `uni2ts`, or Granite as shipped until implementation + tests exist.
- `[ ]` Endpoint inventory in section 11 is updated for every public API change.
- `[ ]` Phased checklist in section 15 is updated whenever milestone scope changes.
- `[ ]` Baseline verification state (`ruff check .`, `pytest -q`) is refreshed when major status markers change.
- `[ ]` Migration notes in section 2 remain aligned with the active repository layout.
