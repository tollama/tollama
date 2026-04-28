# Troubleshooting

This is the canonical troubleshooting guide for Tollama runtime, CLI, daemon, Docker, and MCP setup issues.

## 1) Daemon won't start

Symptoms:
- `tollama serve` exits immediately.
- `address already in use`.
- Permission or bind errors.

Checks:
```bash
lsof -i :11435
```

Fix:
- Stop conflicting process or choose a new port.
- Start daemon on explicit host/port:
```bash
tollama serve --host 127.0.0.1 --port 11435
```
- On restricted environments, avoid privileged ports (<1024).

## 2) Model pull fails

Symptoms:
- Network/proxy errors.
- Auth failures for gated Hugging Face models.

Checks:
- Verify connectivity and proxy values.
- Confirm token exists and has repository access.

Fix:
```bash
export TOLLAMA_HF_TOKEN=hf_xxx
tollama pull <model>
```

Proxy overrides:
```bash
tollama pull <model> --http-proxy http://proxy:8080 --https-proxy http://proxy:8080
```

Offline mode only works after cache is already seeded:
```bash
tollama pull <model> --offline
```

## 3) Forecast times out

Symptoms:
- `failed: timed out`
- Slow first-run model load and inference.

Fix:
```bash
tollama run <model> --input examples/request.json --no-stream --timeout 600
```

Set daemon-wide default timeout:
```bash
export TOLLAMA_FORECAST_TIMEOUT_SECONDS=600
```

If hardware is constrained (CPU/RAM/VRAM), use smaller/faster models.

## 4) License required error

Symptoms:
- Pull returns license acceptance required.

Fix:
```bash
tollama pull <model> --accept-license
```

## 5) Invalid request

Symptoms:
- HTTP 400 validation errors.
- Missing fields or shape/type mismatches.

Fix:
- Validate before forecast:
```bash
tollama run <model> --input examples/request.json --dry-run
```
- Common mistakes:
  - Missing `horizon`.
  - Empty `series`.
  - `timestamps` and `target` length mismatch.
  - Non-positive `timeout`.

## 6) Runtime install fails

Symptoms:
- `tollama runtime install` fails with dependency or Python-version errors.
- macOS app logs show `ensurepip` aborts or a family runtime Python cannot load
  `@executable_path/../lib/libpython3.11.dylib`.

Fix:
```bash
tollama runtime list
tollama runtime install --all
```

For Uni2TS/Moirai stack, prefer Python 3.11 if wheel/build constraints fail on newer interpreters.
If stdlib `venv` / `ensurepip` is broken on the host interpreter, install `uv`
and retry; Tollama falls back to `uv venv` automatically during runtime bootstrap.
For bundled macOS runtimes, Tollama creates family virtualenvs with symlinked
interpreters so relocated Python builds can resolve their shared library.

## 7) Wrong frequency detected

Symptoms:
- Unexpected inferred frequency from SDK/ingest pipeline.

Fix:
- Override `freq` explicitly in request payload.
- Ensure timestamp granularity and ordering are consistent.

Example:
```json
{"series":[{"freq":"D","timestamps":["2025-01-01","2025-01-02"],"target":[1,2]}],"horizon":2,"model":"mock"}
```

## 8) Covariates rejected

Symptoms:
- Covariate compatibility errors in strict mode.
- Warnings in best-effort mode.

Fix:
- Decide behavior using `parameters.covariates_mode`:
  - `best_effort`: unsupported covariates are dropped with warnings.
  - `strict`: unsupported covariates fail with HTTP 400.
- Confirm model capabilities with:
```bash
tollama info
```
- See full contract: `docs/covariates.md`.

## 9) Docker won't start

Symptoms:
- Container starts but daemon unreachable.
- Volume permissions/cache not persisted.

Fix:
- Ensure host port mapping and daemon bind are aligned.
- Persist Tollama home volume for model cache.
- Validate from host:
```bash
curl -s http://127.0.0.1:11435/v1/health
```

## 10) MCP tools not found

Symptoms:
- Claude Desktop cannot find `tollama-mcp`.
- MCP server launches but tools unavailable.

Fix:
- Install MCP extra:
```bash
python -m pip install -e ".[mcp]"
```
- Ensure `tollama-mcp` is on PATH in the same environment used by Claude/Desktop.
- Check Claude Desktop config command path and environment:
```json
{
  "mcpServers": {
    "tollama": {
      "command": "tollama-mcp",
      "env": {"TOLLAMA_BASE_URL": "http://127.0.0.1:11435"}
    }
  }
}
```

## 11) GPU / memory issues

Symptoms:
- `CUDA out of memory` during inference.
- Runner process crashes immediately after load.
- No GPU detected even with CUDA installed.

Checks:
```bash
# Check GPU visibility and memory
nvidia-smi

# Confirm CUDA is available in the runner environment
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
```

Fix:
- Reduce `num_samples` in request `options` to lower VRAM usage:
  ```json
  {"options": {"num_samples": 10}}
  ```
- Force CPU inference if VRAM is insufficient:
  ```bash
  export CUDA_VISIBLE_DEVICES=""
  tollama serve
  ```
- Use a smaller model variant (e.g. `chronos2-tiny` instead of `chronos2-large`).
- If running multiple models simultaneously, unload idle ones with `tollama ps` to see what's loaded.

## 12) Reading daemon logs

Symptoms:
- Unexpected errors with no context.
- Need to trace a specific request through the daemon.

Where logs go:
- `tollama serve` writes to **stdout** (uvicorn access log) and **stderr** (app-level errors).
- Runner subprocess stderr is forwarded to daemon stderr.

Structured JSON logs:
```bash
# Enable JSON-formatted stderr for log aggregation
export TOLLAMA_JSON_STDERR=1
tollama serve 2>daemon.log
```

Useful grep filters:
```bash
# Filter for forecast errors
grep -i "error\|exception\|traceback" daemon.log

# Trace a specific request ID
grep '"id":"<uuid>"' daemon.log

# See runner process events
grep "runner\|subprocess\|stdio" daemon.log
```

## 13) First-run is slow

Symptoms:
- First forecast after `tollama pull` takes several minutes.
- Subsequent forecasts on the same model are much faster.

Explanation:
- **TimesFM** and **Sundial** JIT-compile model layers on first use. This is a one-time
  cost per environment; subsequent runs reuse the compiled artifact from cache.
- First run also loads model weights from disk into RAM/VRAM.

Fix:
- Extend the CLI timeout for first-run:
  ```bash
  tollama run timesfm-2.0-200m --input examples/request.json --no-stream --timeout 900
  ```
- Or increase the daemon-wide timeout environment variable:
  ```bash
  export TOLLAMA_FORECAST_TIMEOUT_SECONDS=900
  ```
- After the first successful run, normal timeout values (120–300 s) are safe.

## 14) Runtime re-installs once after upgrade or source change

Symptoms:
- Right after upgrading Tollama, the first run/pull for a model family is slower than usual.
- Logs mention runtime re-bootstrap due to `schema_version`, `dependency_fingerprint`,
  or `source_fingerprint` mismatch.
- It only happens once per family, then normal speed returns.

Explanation:
- Tollama stores per-family runtime metadata, including `schema_version` and
  dependency/source fingerprints, in runtime state.
- After an upgrade, dependency change, or local source checkout change, Tollama
  intentionally rebuilds the affected runtime once so the environment matches
  the current code.
- This is expected migration behavior, not runtime corruption.

Quick verification:
```bash
# Check runtime state, including schema_version and fingerprints
tollama runtime list --json

# Trigger one run for the affected family/model
# First run may be slower while rebuild happens
tollama run <model> --input examples/request.json --no-stream --timeout 900

# Re-check state; subsequent runs should be fast
tollama runtime list --json
```

Safe remediation (if rebuild did not complete cleanly):
```bash
# Rebuild only the affected runtime family
# Examples: torch, timesfm, uni2ts, sundial, toto, lag_llama, patchtst, tide,
# nhits, nbeatsx, timer, timemixer, forecastpfn
tollama runtime update <family>

# Or refresh all currently installed runtime families
tollama runtime update --all

# If needed, force a clean reinstall for one family
tollama runtime install <family> --reinstall
```

Notes:
- Prefer `runtime update`/`install --reinstall` over manual venv deletion.
- If rebuild keeps failing, capture stderr logs and run `tollama doctor` before opening an issue.

## 15) CI triage: dependency-gated vs regression failures

Symptoms:
- Real-data/runner checks fail in CI, but failures look environment-sensitive.
- You need to decide whether to block merge or classify as expected gating.

Classification guide:
- `DEPENDENCY_MISSING` → treat as **dependency-gated** (expected when optional deps are absent).
- `runner family '<family>' is not supported` → treat as **regression** (must be fixed before merge).
- Daemon readiness should probe `/v1/health` first, with `/api/health` only as compatibility fallback.

Quick checks:
```bash
# Health probe contract
curl -sf http://127.0.0.1:11435/v1/health >/dev/null || curl -sf http://127.0.0.1:11435/api/health >/dev/null

# Find dependency-gated outcomes in logs
grep -R "DEPENDENCY_MISSING" artifacts/ -n || true

# Find unsupported-family regressions in logs
grep -R "runner family .* is not supported" artifacts/ -n || true
```

---

## Quick Triage Commands

```bash
tollama info
tollama doctor
tollama runtime list
tollama list
tollama ps
```
