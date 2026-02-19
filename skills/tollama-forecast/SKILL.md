---
name: tollama-forecast
description: Run time-series forecasting via tollama daemon (CLI-first with HTTP fallback when CLI is unavailable).
homepage: https://github.com/tollama/tollama
user-invocable: true
metadata: {"openclaw":{"emoji":"📈","os":["darwin","linux"],"requires":{"bins":["bash","tollama","curl"]}}}
---

# tollama-forecast

Run forecasting requests against a tollama daemon with operational safeguards for path, host, and timeout issues.

## What This Skill Does

- checks daemon health and version (`tollama-health.sh`)
- inspects installed/loaded/available models (`tollama-models.sh`)
- runs forecast requests with non-stream default and no-auto-pull policy (`tollama-forecast.sh`)

## Install / Load (choose one)

### Option A: Managed skills (recommended)

```bash
mkdir -p ~/.openclaw/skills
ln -s "$(pwd)/skills/tollama-forecast" ~/.openclaw/skills/tollama-forecast
openclaw skills list --eligible | rg tollama-forecast
```

### Option B: Workspace skills

If OpenClaw gateway workspace is this repository root, `skills/tollama-forecast` can be auto-discovered.

### Option C: extraDirs in `~/.openclaw/openclaw.json`

```json5
{
  "skills": {
    "load": {
      "extraDirs": [
        "/ABSOLUTE/PATH/TO/tollama/skills"
      ]
    }
  }
}
```

## Runtime Defaults

- base URL: `--base-url` > `TOLLAMA_BASE_URL` > `http://localhost:11435`
- timeout: `--timeout` > `TOLLAMA_FORECAST_TIMEOUT_SECONDS` > `300`
- forecast mode: non-stream by default
- auto-pull: disabled by default, enabled only with `--pull`

## Common Troubleshooting

### `tollama: command not found`

`tollama` is often installed in a venv and missing from OpenClaw exec PATH.

Fix options:
- put `tollama` on system PATH
- prepend venv path in OpenClaw config:

```json5
{
  "tools": {
    "exec": {
      "pathPrepend": [
        "/ABSOLUTE/PATH/TO/tollama/.venv/bin"
      ]
    }
  }
}
```

### Health check works in terminal but fails in OpenClaw

`127.0.0.1` means different hosts depending on exec location.
- gateway host exec: points to host daemon
- sandbox/container exec: points to container localhost

If mismatch happens, set `--base-url` to a daemon address reachable from the exec host.

## Manual Smoke Commands

```bash
bash skills/tollama-forecast/bin/tollama-health.sh --base-url "${TOLLAMA_BASE_URL:-http://localhost:11435}"
bash skills/tollama-forecast/bin/tollama-models.sh installed --base-url "${TOLLAMA_BASE_URL:-http://localhost:11435}"
cat skills/tollama-forecast/examples/simple_forecast.json | \
  bash skills/tollama-forecast/bin/tollama-forecast.sh --model mock --base-url "${TOLLAMA_BASE_URL:-http://localhost:11435}"
```
