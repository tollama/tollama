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

Fix:
```bash
tollama runtime list
tollama runtime install --all
```

For Uni2TS/Moirai stack, prefer Python 3.11 if wheel/build constraints fail on newer interpreters.

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

---

## Quick Triage Commands

```bash
tollama info
tollama doctor
tollama runtime list
tollama list
tollama ps
```
