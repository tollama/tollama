---
name: tollama-forecast
description: Run time-series forecasts via a local Tollama daemon. Use bash scripts to validate daemon health, list models, and execute non-streaming forecast requests that return JSON.
homepage: https://github.com/tollama/tollama
user-invocable: true
# NOTE: Keep metadata as a single-line JSON object for maximum compatibility (and required by OpenClaw).
metadata: {"openclaw":{"emoji":"📈","os":["darwin","linux"],"requires":{"bins":["bash"],"anyBins":["tollama","curl"]}}}
---

# Tollama Forecast Skill

This skill runs **time-series forecasting** through a locally reachable **Tollama daemon**.
It is written to be **host-agnostic**: any AI agent that can run shell commands in a workspace can use it.

The skill ships **bash helper scripts** (under `bin/`) that:

- verify daemon health/version
- inspect installed/loaded models
- run forecasts in **non-streaming mode** and print a single **JSON** response to stdout

---

## What this skill is good for

Use this skill when you need to:

- produce forecast values (and optional intervals) from a time series
- run forecasts locally via Tollama without writing client code
- keep output machine-readable (single JSON object; no streaming NDJSON)

---

## Requirements

### Runtime

- **macOS or Linux**
- `bash`
- One of:
  - `tollama` CLI (recommended), **or**
  - `curl` (for HTTP fallback)
- `python3` (used by helper scripts for JSON normalization/extraction)

### A running Tollama daemon

- Default base URL: `http://127.0.0.1:11435`
- You can override via `TOLLAMA_BASE_URL` or `--base-url`.

> If your agent runs commands inside a container/sandbox, `127.0.0.1` might refer to *that container*, not your host machine.
> In that case, set `TOLLAMA_BASE_URL` to a daemon address reachable **from the execution environment**.

---

## Quickstart

From the skill directory:

```bash
# 1) (Optional) set daemon URL
export TOLLAMA_BASE_URL="http://127.0.0.1:11435"

# 2) Health check
bash ./bin/tollama-health.sh --base-url "$TOLLAMA_BASE_URL"

# 3) List installed models (optional)
bash ./bin/tollama-models.sh installed --base-url "$TOLLAMA_BASE_URL"

# 4) Run a forecast (example uses the mock model)
bash ./bin/tollama-forecast.sh \
  --model mock \
  --input ./examples/simple_forecast.json \
  --base-url "$TOLLAMA_BASE_URL" \
  --timeout 300
```

Expected: stdout prints **one JSON object** (the Tollama forecast response).

---

## Scripts and interfaces

All scripts:

- print **human-friendly errors** to stderr
- print **machine-readable outputs** to stdout (JSON where possible)
- return meaningful exit codes (see below)

### 1) Health check

```bash
bash ./bin/tollama-health.sh --base-url "$TOLLAMA_BASE_URL"
```

What it checks (implementation may use HTTP):

- `GET /v1/health`
- `GET /api/version`

### 2) Model inspection

```bash
bash ./bin/tollama-models.sh installed --base-url "$TOLLAMA_BASE_URL"
bash ./bin/tollama-models.sh loaded --base-url "$TOLLAMA_BASE_URL"
bash ./bin/tollama-models.sh show moirai-2.0-R-small --base-url "$TOLLAMA_BASE_URL"
bash ./bin/tollama-models.sh available --base-url "$TOLLAMA_BASE_URL"
```

Meaning of subcommands:

- `installed`: models available locally (e.g., `tollama list` / `/api/tags`)
- `loaded`: models currently loaded (e.g., `tollama ps` / `/api/ps`)
- `show <model>`: model details
- `available`: daemon “available models” metadata (from `tollama info --json --remote` or `/api/info`)

> Notes:
>
> - `available` is **daemon-only** in this skill. If daemon is unreachable, it fails (no local fallback).
> - Keep dependencies minimal: scripts do not require `jq`.

### 3) Forecast execution

```bash
bash ./bin/tollama-forecast.sh --model <MODEL> [--input <file>] [--base-url <url>] [--timeout <sec>] [--pull] [--accept-license]
```

**Inputs**

- `--model <MODEL>` (required): the model to use (e.g., `mock`, `timesfm-2.5-200m`, `moirai-2.0-R-small`)
- `--input <file>` (optional): request JSON file
  - If omitted, the script reads request JSON from **stdin**.

**Behavior**

- Prefers CLI (`tollama run`) when available; can fall back to HTTP (`curl`) when CLI is unavailable.
- Forces **non-streaming** output to avoid NDJSON parsing issues.
- Timeout is configurable; first run can take longer due to model downloads/runner bootstrap.

**Model pull policy (safe default)**

- By default, the script **does not auto-pull** missing models.
- If the model is missing, it exits with a “model not installed” error.
- To allow pull, pass `--pull`.
- If a model requires license acceptance, pass `--accept-license` (when pulling).

---

## Request format

This skill forwards a request JSON to the Tollama forecast interface.

Because request schemas can evolve with Tollama/model families, the recommended approach is:

- start from the included examples under `./examples/`
- keep `"stream": false` (or rely on script defaults)

Included examples:

- `examples/simple_forecast.json`: minimal single-series forecast (good sanity check)
- `examples/multi_series.json`: multiple series in one request
- `examples/covariates_forecast.json`: past/future covariates example (if supported by your model)

**Important**

- Do not enable streaming unless you explicitly want NDJSON and your host can parse it.
- If using covariates, align timestamps and lengths exactly as required by your model. Misalignment is a common error source.

---

## Output format

- **stdout:** one JSON object (the forecast response)
- **stderr:** diagnostics and failure hints
- **no streaming NDJSON** in the default path

---

## Exit codes

Scripts follow this shared contract:

- `0` success
- `2` invalid request/input (malformed JSON, missing required fields)
- `3` daemon unreachable / health check failed
- `4` model not installed locally
- `5` license required / not accepted / permission issue
- `6` timeout
- `10` unexpected internal error

---

## Troubleshooting

### `tollama: command not found`

- Ensure `tollama` is installed **in the execution environment** and on `PATH`.
- If your agent runs in a sandbox/container, install `tollama` there or call it via an absolute path.

### Health check passes locally, but fails in the agent

- The agent may execute commands in a different environment (container/remote runner).
- Set `TOLLAMA_BASE_URL` to an address reachable from that environment.

### Forecast is slow or times out on first run

- First run can include: model download, runner environment setup, JIT compilation.
- Increase `--timeout` (e.g., 600 seconds) and retry.

### Model requires license acceptance

- Pull with explicit acceptance:
  - `tollama pull <model> --accept-license`
- Or run the skill with `--pull --accept-license`.

---

# Host Notes

This section contains **host-specific tips**.
The core usage above remains the same: run the bash scripts from the skill directory.

---

## OpenClaw Notes

### Skill discovery / install

Common patterns:

- Managed skills directory: `~/.openclaw/skills/`
- Workspace skills directory: `<workspace>/skills/`

Symlink example:

```bash
mkdir -p ~/.openclaw/skills
ln -s "$(pwd)/tollama/skills/tollama-forecast" ~/.openclaw/skills/tollama-forecast
```

### `{baseDir}` placeholder

OpenClaw supports `{baseDir}` for resolving paths relative to the skill folder.
If you use it, keep it in OpenClaw-specific sections only.

### Exec environment (sandbox vs host)

If OpenClaw is executing commands inside a sandbox/container, `127.0.0.1` may not reach your host daemon.
Prefer running the scripts in a host-reachable execution context (e.g., gateway) or adjust `TOLLAMA_BASE_URL`.

### PATH / venv installs

If `tollama` is installed in a Python venv, OpenClaw may not inherit that venv’s PATH.
Use OpenClaw’s `tools.exec.pathPrepend` (or install `tollama` into a standard PATH location).

---

## OpenAI Codex / OpenAI API Notes

- Ensure your Codex environment mounts the repository/workspace that contains this skill folder.
- Use the agent’s shell/terminal execution capability to run the commands exactly as shown.
- Prefer non-streaming output to keep parsing deterministic.

---

## VS Code / GitHub Copilot Notes

- Keep the skill directory name **lowercase with hyphens** (e.g., `tollama-forecast`).
- Ensure `SKILL.md` is at the **root of the skill folder** (next to `bin/` and `examples/`).
- If your Copilot environment runs tools in containers, remember to set `TOLLAMA_BASE_URL` accordingly.

---

## Security considerations

This skill executes local shell scripts and may send request payloads to the configured Tollama daemon URL.

- Do not point `TOLLAMA_BASE_URL` to untrusted remote endpoints.
- Treat forecast requests and outputs as potentially sensitive data.
