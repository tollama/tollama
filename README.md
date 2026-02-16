# tollama

Lightweight Python project skeleton for the `tollama` codebase.

## Overview

This repository provides a minimal but structured forecasting stack:
- PEP 621 packaging via `pyproject.toml`
- `src/` project layout
- FastAPI daemon (`tollamad`) exposing the public HTTP API
- Runner processes (currently mock) connected over stdio JSON lines
- Typer CLI (`tollama`) for Ollama-style model lifecycle and forecasting
- Ruff + Pytest + CI checks

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m pip install -e ".[dev,runner_torch]"  # optional: torch runner (Chronos + Granite TTM)
python -m pip install -e ".[dev,runner_timesfm]"  # optional: TimesFM runner
python -m pip install -e ".[dev,runner_uni2ts]"  # optional: Uni2TS/Moirai runner

# Terminal 1: run daemon (default http://127.0.0.1:11435)
tollama serve

# Terminal 2: Ollama-style lifecycle
tollama list
tollama pull mock
tollama show mock
tollama run mock --input examples/request.json --no-stream
tollama ps
tollama rm mock

# Check daemon version endpoint
curl http://localhost:11435/api/version

ruff check .
pytest -q
```

## Daemon Base URL and Host Config

- Default CLI base URL is `http://localhost:11435`.
- Ollama-style API base URL is `http://localhost:11435/api`.
- Default daemon port remains `11435`.
- Configure bind host and port with `TOLLAMA_HOST` in `host:port` format.
  - Example: `TOLLAMA_HOST=0.0.0.0:11435 tollamad`
- Local tollama state lives under `~/.tollama` by default.
  - Override with `TOLLAMA_HOME=/custom/path` (config, models, runtimes all use this base).
- Primary lifecycle endpoints are under `/api/*`.
  - `GET /api/info`
  - `GET /api/tags`
  - `GET /api/ps`
  - `POST /api/show`
  - `POST /api/pull`
  - `DELETE /api/delete`
  - `POST /api/forecast`
- Existing forecast and health endpoints remain under `/v1/*` for backward compatibility.
  - `GET /v1/health`
  - `POST /v1/forecast`

## Chronos2 Forecasting

```bash
# install optional torch runner dependencies
python -m pip install -e ".[dev,runner_torch]"

# run daemon
tollama serve

# pull Chronos snapshot + manifest metadata
tollama pull chronos2

# run forecast through CLI
tollama run chronos2 --input examples/chronos2_request.json --no-stream
```

```bash
curl -s http://localhost:11435/api/forecast \
  -H 'content-type: application/json' \
  -d @examples/chronos2_request.json
```

`/api/pull` now performs a real Hugging Face snapshot pull for registry models with `source.type=huggingface`,
streams NDJSON progress by default, and writes resolved pull metadata into the local manifest:
- `resolved.commit_sha`
- `resolved.snapshot_path`
- `size_bytes`
- `pulled_at`

### Pull examples

```bash
# basic pull
tollama pull chronos2

# proxy override
tollama pull chronos2 --https-proxy http://proxy:3128

# offline after first pull
tollama pull chronos2
tollama pull chronos2 --offline

# override Hugging Face cache home
tollama pull chronos2 --hf-home /mnt/fastcache/hf

# token via environment
export TOLLAMA_HF_TOKEN=hf_xxx
tollama pull <private-model>
```

## Granite TTM Forecasting (torch runner)

`runner_torch` now includes optional Chronos and Granite TTM dependencies.

```bash
# install optional torch runner dependencies
python -m pip install -e ".[dev,runner_torch]"

# run daemon (default http://127.0.0.1:11435)
tollama serve

# pull one Granite TTM revision pinned in the registry
tollama pull granite-ttm-r2

# run forecast
tollama run granite-ttm-r2 --input examples/granite_ttm_request.json --no-stream
```

```bash
curl -s http://localhost:11435/api/forecast \
  -H 'content-type: application/json' \
  -d @examples/granite_ttm_request.json
```

## TimesFM 2.5 Forecasting (separate timesfm runner family)

```bash
# install optional TimesFM runner dependencies
python -m pip install -e ".[dev,runner_timesfm]"
# note: for GPU builds, install a compatible torch build first if needed

# run daemon (default http://127.0.0.1:11435)
tollama serve

# pull TimesFM 2.5 model snapshot from Hugging Face
tollama pull timesfm-2.5-200m

# run forecast
tollama run timesfm-2.5-200m --input examples/timesfm_2p5_request.json --no-stream
```

```bash
curl -s http://localhost:11435/api/forecast \
  -H 'content-type: application/json' \
  -d @examples/timesfm_2p5_request.json
```

## Uni2TS/Moirai Forecasting (separate uni2ts runner family)

```bash
# install optional Uni2TS runner dependencies
python -m pip install -e ".[dev,runner_uni2ts]"
# note: install a compatible torch build first for your platform when needed

# run daemon (default http://127.0.0.1:11435)
tollama serve

# Moirai model requires explicit license acceptance
tollama pull moirai-1.1-R-base --accept-license

# run forecast
tollama run moirai-1.1-R-base --input examples/moirai_request.json --no-stream
```

```bash
curl -s http://localhost:11435/api/forecast \
  -H 'content-type: application/json' \
  -d @examples/moirai_request.json
```

## Persistent Pull Defaults

`tollama config` stores pull defaults in `~/.tollama/config.json` (or `$TOLLAMA_HOME/config.json`).
These defaults are applied by the daemon on `/api/pull`, so any API client benefits.

```bash
# inspect current defaults
tollama config list

# set persistent defaults
tollama config set pull.https_proxy http://proxy:3128
tollama config set pull.hf_home /mnt/fastcache/hf
tollama config set pull.offline true

# no pull flags needed; daemon applies config defaults
tollama pull chronos2
```

Tokens are intentionally not persisted in config. Use `TOLLAMA_HF_TOKEN` or `--token`.

## Diagnostics Endpoint: `/api/info`

`GET /api/info` returns one fast diagnostics payload for server-side integrations.
It includes:
- daemon metadata (`version`, `started_at`, `uptime_seconds`, `host_binding`)
- local paths and config presence (`tollama_home`, `config_path`, `config_exists`)
- redacted config and safe env subset
- effective pull defaults with value source (`env`/`config`/`default`)
- installed + loaded models
- runner statuses for `mock`, `torch`, `timesfm`, and `uni2ts`

Example shape:

```json
{
  "daemon": {"version": "0.1.0", "started_at": "...", "uptime_seconds": 42},
  "paths": {"tollama_home": "...", "config_path": "...", "config_exists": true},
  "config": {"pull": {"https_proxy": "http://***:***@proxy:3128"}},
  "env": {"HTTP_PROXY": "http://***:***@proxy:3128", "TOLLAMA_HF_TOKEN_present": false},
  "pull_defaults": {"offline": {"value": false, "source": "default"}},
  "models": {"installed": [], "loaded": []},
  "runners": []
}
```

## Debugging: `tollama info`

Use `tollama info` for one diagnostics view of daemon reachability, local config, effective pull
defaults, and installed/loaded models.

```bash
tollama info
tollama info --json
tollama info --local   # force local-only
tollama info --remote  # require daemon /api/info
```

Example output excerpt:

```text
Tollama
  Home: /Users/you/.tollama
  Config: /Users/you/.tollama/config.json (exists)
  Daemon: http://localhost:11435 (reachable) version=0.1.0

Pull defaults (effective)
  offline: false (source=default)
  https_proxy: http://proxy:3128 (source=config)
```

`tollama info` prefers `GET /api/info` when the daemon is reachable, and automatically falls back
to local collection when the daemon is down (unless `--remote` is set).

## Architecture

- `tollama.daemon`: Public API layer (`/api/*`, `/v1/health`, `/v1/forecast`) and runner supervision.
- `tollama.runners`: Runner implementations that speak newline-delimited JSON over stdio.
- `tollama.core`: Canonical data schemas (`ForecastRequest`, `ForecastResponse`) and protocol helpers.
- `tollama.cli`: User CLI commands for serving and sending forecast requests.

### Daemon <-> Runner Protocol

- Transport: newline-delimited JSON over stdio.
- Request shape: `{"id":"...","method":"...","params":{...}}`
- Response shape: `{"id":"...","result":{...}}` or `{"id":"...","error":{"code":...,"message":"..."}}`
- Current mock runner methods: `hello`, `forecast`.
- Current torch runner methods: `hello`, `load`, `unload`, `forecast`.

## Adding A New Runner Family

1. Add a new package under `src/tollama/runners/<family>/`.
2. Implement a `main.py` stdio loop using `tollama.core.protocol` encode/decode helpers.
3. Validate incoming forecast params with `tollama.core.schemas.ForecastRequest`.
4. Return canonical outputs compatible with `tollama.core.schemas.ForecastResponse`.
5. Register a console script in `pyproject.toml` if the runner should be executable directly.
6. Add tests for protocol behavior and end-to-end daemon routing.

## Project Structure

```text
.
├── src/
│   └── tollama/
│       ├── cli/
│       ├── core/
│       ├── daemon/
│       └── runners/
├── examples/
├── tests/
└── .github/workflows/
```
