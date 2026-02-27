# OpenClaw Sandbox Runbook

This runbook describes end-to-end operation of `skills/tollama-forecast/` when
OpenClaw executes commands in a sandbox runtime.

## 1) Topology assumptions

- OpenClaw tool execution host is a sandbox process.
- Tollama daemon may run on:
  - the same sandbox host, or
  - another reachable host (for example developer laptop or shared node).
- `127.0.0.1` resolves inside the sandbox, not always the user host.

## 2) Prerequisites

- `bash` is available.
- One of `tollama` or `curl` is available in the sandbox PATH.
- `python3` is available (used by skill request normalization).
- Tollama daemon is reachable from sandbox:
  - default URL: `http://127.0.0.1:11435`
  - override: `TOLLAMA_BASE_URL` or `--base-url`.

## 3) Install and register the skill

```bash
mkdir -p ~/.openclaw/skills
ln -s "$(pwd)/skills/tollama-forecast" ~/.openclaw/skills/tollama-forecast
openclaw skills list --eligible | rg tollama-forecast
```

Validate package structure:

```bash
bash scripts/validate_openclaw_skill_tollama_forecast.sh
```

## 4) Reachability checks

Set base URL explicitly when sandbox and daemon are not co-located.

```bash
export TOLLAMA_BASE_URL="http://127.0.0.1:11435"
bash skills/tollama-forecast/bin/tollama-health.sh --base-url "$TOLLAMA_BASE_URL"
bash skills/tollama-forecast/bin/tollama-models.sh installed --base-url "$TOLLAMA_BASE_URL"
```

Expected health output contains `healthy: true` and daemon version.

## 5) E2E command flow

1. Pull (or ensure) test model:

```bash
bash skills/tollama-forecast/bin/tollama-pull.sh \
  --model mock \
  --base-url "$TOLLAMA_BASE_URL"
```

2. Run forecast:

```bash
bash skills/tollama-forecast/bin/tollama-forecast.sh \
  --model mock \
  --input skills/tollama-forecast/examples/simple_forecast.json \
  --base-url "$TOLLAMA_BASE_URL" \
  --timeout 300
```

3. Run metrics-enabled forecast:

```bash
bash skills/tollama-forecast/bin/tollama-forecast.sh \
  --model mock \
  --input skills/tollama-forecast/examples/metrics_forecast.json \
  --base-url "$TOLLAMA_BASE_URL" \
  --metrics mape,mase,mae,rmse,smape \
  --mase-seasonality 1 \
  --timeout 300
```

## 6) Failure triage by exit code

| Exit code | Meaning | Primary action |
|---|---|---|
| `2` | invalid input/request | Validate input JSON shape and option values (`--metrics`, `--timeout`, `--mase-seasonality`). |
| `3` | daemon unreachable/health failure | Re-check `TOLLAMA_BASE_URL`; avoid `127.0.0.1` if daemon is outside sandbox. |
| `4` | model not installed | Re-run with `--pull` or pull model explicitly first. |
| `5` | license/permission | Pull with `--accept-license` for gated models. |
| `6` | timeout | Increase `--timeout` or `TOLLAMA_FORECAST_TIMEOUT_SECONDS`. |
| `10` | internal error | Inspect stderr (set `TOLLAMA_JSON_STDERR=1` for structured error). |

## 7) Sandbox-specific debugging checklist

- Confirm command path resolution:
  - `which tollama`
  - `which curl`
  - `which python3`
- Confirm daemon endpoints from sandbox:
  - `curl -s "$TOLLAMA_BASE_URL/api/version"`
  - `curl -s "$TOLLAMA_BASE_URL/v1/health"`
- If local terminal works but sandbox fails:
  - treat as host mismatch first,
  - move from loopback to a reachable host/IP in `TOLLAMA_BASE_URL`.

## 8) Sandbox quirks

**`127.0.0.1` points inside the sandbox, not the user host.**
If the tollama daemon is running on your local machine, the sandbox cannot reach it via
`127.0.0.1` — that loopback resolves to the sandbox container itself. Use the host's
LAN IP or a forwarded port instead, and set `TOLLAMA_BASE_URL` accordingly.

**`python3` vs `python` availability.**
Some sandbox images only have `python3` in PATH; others only `python`. The skill scripts
use `python3`. If `python3` is absent, create a symlink or set `TOLLAMA_PYTHON_CMD`:
```bash
export TOLLAMA_PYTHON_CMD=python
```

**Snapshot persistence across sessions.**
Sandbox sessions are typically ephemeral. Model snapshots downloaded inside a sandbox
session (under `~/.tollama/models/`) are lost when the sandbox exits unless the directory
is explicitly mounted or backed by a persistent volume. Pre-pull models on your host and
mount `$TOLLAMA_HOME` into the sandbox, or point `TOLLAMA_BASE_URL` at a daemon that
already has models installed.

**First-run bootstrap delay.**
If the daemon auto-bootstraps a runner venv inside the sandbox (on first forecast request),
expect a 1–5 minute delay for `pip install`. Use `TOLLAMA_FORECAST_TIMEOUT_SECONDS=600`
or pre-install runtimes on the host before starting the sandbox.

**PATH isolation.**
The sandbox PATH may not include the virtualenv where `tollama` is installed. Verify with
`which tollama` and use an absolute path if needed (e.g. `/home/user/.venv/bin/tollama`).
