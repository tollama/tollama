# tollama

Local-first forecasting daemon + CLI with pluggable model-family runners.

## Overview

This repository provides a structured forecasting stack:
- PEP 621 packaging via `pyproject.toml`
- `src/` project layout
- FastAPI daemon (`tollamad`) exposing the public HTTP API
- Runner processes (`mock`, `torch`, `timesfm`, `uni2ts`, `sundial`, `toto`) connected over stdio JSON lines
- Typer CLI (`tollama`) for Ollama-style model lifecycle and forecasting
- Unified covariates contract (`past_covariates` + `future_covariates`) with
  `best_effort`/`strict` handling and warnings
- Optional forecast accuracy metrics (`MAPE`, `MASE`) via `series[].actuals`
  and `parameters.metrics`
- Ruff + Pytest + CI checks

## Quickstart

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
# optional single-environment mode (not default):
# python -m pip install -e ".[dev,runner_torch,runner_timesfm,runner_uni2ts,runner_sundial,runner_toto]"

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

## Runner Runtime Mode (Default)

Default daemon behavior is **family-level isolated runtimes** (not one venv per model):

- `daemon.auto_bootstrap` defaults to `true`
- first run for each family bootstraps `~/.tollama/runtimes/<family>/venv/`
- active families: `torch`, `timesfm`, `uni2ts`, `sundial`, `toto`

This allows different dependency sets per family without conflicts.

If you prefer one shared `.venv` for all families, disable auto-bootstrap in
`~/.tollama/config.json` and preinstall runner extras into that environment:

```json
{
  "version": 1,
  "daemon": {
    "auto_bootstrap": false
  }
}
```

## E2E Integration Snapshot (2026-02-17)

Optional real-model integration tests were re-run with:

```bash
TOLLAMA_RUN_INTEGRATION_TESTS=1 TOLLAMA_TOTO_INTEGRATION_CPU=1 pytest -q -rs \
  tests/test_chronos_integration.py \
  tests/test_granite_integration.py \
  tests/test_timesfm_integration.py \
  tests/test_uni2ts_integration.py \
  tests/test_sundial_integration.py \
  tests/test_toto_integration.py
```

| Model | Test | Result |
|---|---|---|
| `chronos2` | `tests/test_chronos_integration.py` | pass |
| `granite-ttm-r2` | `tests/test_granite_integration.py` | pass |
| `timesfm-2.5-200m` | `tests/test_timesfm_integration.py` | pass |
| `moirai-2.0-R-small` | `tests/test_uni2ts_integration.py` | pass |
| `sundial-base-128m` | `tests/test_sundial_integration.py` | pass |
| `toto-open-base-1.0` | `tests/test_toto_integration.py` | skipped (`toto` package missing) |

Per-family runtime isolation smoke was also re-verified on `2026-02-17`
after updating the TimesFM dependency pin to
`git+https://github.com/google-research/timesfm.git@2dcc66fbfe2155adba1af66aa4d564a0ee52f61e`.

| Model | Runtime Isolation Smoke |
|---|---|
| `chronos2` | pass |
| `granite-ttm-r2` | pass |
| `timesfm-2.5-200m` | pass |
| `moirai-2.0-R-small` | pass |
| `sundial-base-128m` | pass |
| `toto-open-base-1.0` | pass |

### OpenClaw Skills E2E (2026-02-19)

We also verified the OpenClaw skill integration using `scripts/e2e_skills_test.sh`:

| Skill | Result | Notes |
|---|---|---|
| `tollama-forecast` | pass | Validated skill structure + execution with `mock` model; verified Exit Code Contract v2 compliance |

All runner commands were confirmed from `/api/info` to use
`~/.tollama/runtimes/<family>/venv/bin/python`.

Optional: one shared environment for all families (single-venv mode).
Use this only when you intentionally disable auto-bootstrap as shown above.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[dev,runner_torch,runner_timesfm,runner_uni2ts,runner_sundial,runner_toto]"

which python
which tollama
which tollama-runner-uni2ts
# all paths should resolve under .../tollama/.venv/bin/

./.venv/bin/tollama serve
```

For a full setup/run walkthrough with dependency breakdown and commands to install every TSFM model in the registry, see `docs/how-to-run.md`.

Before publishing a public release, follow `docs/public-release-checklist.md` for licensing/compliance and release-readiness gates.

## Daemon Base URL and Host Config

- Default CLI base URL is `http://localhost:11435`.
- Ollama-style API base URL is `http://localhost:11435/api`.
- Default daemon port remains `11435`.
- Configure bind host and port with `TOLLAMA_HOST` in `host:port` format.
  - Example: `TOLLAMA_HOST=0.0.0.0:11435 tollamad`
- Override daemon runner forecast timeout with `TOLLAMA_FORECAST_TIMEOUT_SECONDS` (default `300`).
- Local tollama state lives under `~/.tollama` by default.
  - Override with `TOLLAMA_HOME=/custom/path` (config, models, runtimes all use this base).
- Runner runtime default is per-family isolated venv under
  `~/.tollama/runtimes/<family>/venv/` (`daemon.auto_bootstrap=true`).
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

### Covariates (Past + Known-Future)

Unified covariates contract:

- `past_covariates[name]` length must equal `len(target)`
- `future_covariates[name]` length must equal `horizon`
- every `future_covariates[name]` key must also exist in `past_covariates`
- each covariate array must be all-numeric or all-string (no mixed arrays)
- known-future covariates are keys present in both past and future maps
- set `parameters.covariates_mode` to `best_effort` (default) or `strict`
- in `best_effort`, unsupported covariates are ignored with response `warnings[]`
- in `strict`, unsupported covariates return HTTP `400`

```json
{
  "model": "timesfm-2.5-200m",
  "horizon": 2,
  "series": [
    {
      "id": "s1",
      "freq": "D",
      "timestamps": ["2025-01-01", "2025-01-02", "2025-01-03"],
      "target": [10.0, 11.0, 12.0],
      "past_covariates": {"promo": [0.0, 1.0, 0.0]},
      "future_covariates": {"promo": [1.0, 1.0]}
    }
  ],
  "parameters": {"covariates_mode": "best_effort"},
  "options": {}
}
```

### Forecast Accuracy Metrics (MAPE + MASE)

`/api/forecast` and `/v1/forecast` can optionally calculate forecast accuracy
metrics against provided actuals:

- set `series[].actuals` (length must equal `horizon`)
- set `parameters.metrics.names` to any of `["mape", "mase"]`
- optional `parameters.metrics.mase_seasonality` (default `1`)
- undefined cases are best-effort with response `warnings[]`:
  - MAPE skips when all actual values are `0`
  - MASE skips when `len(target) <= mase_seasonality` or naive denominator is `0`

```json
{
  "model": "mock",
  "horizon": 2,
  "quantiles": [],
  "series": [
    {
      "id": "s1",
      "freq": "D",
      "timestamps": ["2025-01-01", "2025-01-02"],
      "target": [1.0, 3.0],
      "actuals": [2.0, 4.0]
    }
  ],
  "parameters": {
    "metrics": {
      "names": ["mape", "mase"],
      "mase_seasonality": 1
    }
  },
  "options": {}
}
```

Example response:

```json
{
  "model": "mock",
  "forecasts": [
    {
      "id": "s1",
      "freq": "D",
      "start_timestamp": "2025-01-02",
      "mean": [3.0, 3.0]
    }
  ],
  "metrics": {
    "aggregate": {"mape": 37.5, "mase": 0.5},
    "series": [
      {"id": "s1", "values": {"mape": 37.5, "mase": 0.5}}
    ]
  }
}
```

See `docs/covariates.md` for the full compatibility matrix and model-family mappings.

Compatibility snapshot:

| Family | Past Numeric | Past Categorical | Known-Future Numeric | Known-Future Categorical |
|---|---|---|---|---|
| Chronos-2 | Yes | Yes | Yes | Yes |
| Granite TTM | Yes | No | Yes | No |
| TimesFM 2.5 | Yes | No | Yes | No |
| Uni2TS / Moirai | Yes | No | Yes | No |
| Sundial | No | No | No | No |
| Toto Open Base 1.0 | Yes | No | No | No |

TimesFM XReg knobs are available at `parameters.timesfm`:
`xreg_mode`, `ridge`, `force_on_cpu`.

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
# note: Python 3.12+ may fail to build Uni2TS dependencies; prefer Python 3.11

# run daemon (default http://127.0.0.1:11435)
tollama serve

# Moirai model requires explicit license acceptance
tollama pull moirai-2.0-R-small --accept-license

# run forecast
tollama run moirai-2.0-R-small --input examples/moirai_2p0_request.json --no-stream
# if you hit timeout on first run, increase CLI timeout
tollama run moirai-2.0-R-small --input examples/moirai_2p0_request.json --no-stream --timeout 120
```

```bash
curl -s http://localhost:11435/api/forecast \
  -H 'content-type: application/json' \
  -d @examples/moirai_2p0_request.json
```

## Sundial Forecasting (separate sundial runner family)

Sundial (`thuml/sundial-base-128m`) is frequency-agnostic and target-only in this runner.
Covariates are ignored in `best_effort` mode and rejected in `strict` mode by daemon compatibility checks.

```bash
# install optional Sundial runner dependencies
python -m pip install -e ".[dev,runner_sundial]"
# note: install a compatible torch build first for your platform when needed

# run daemon (default http://127.0.0.1:11435)
tollama serve

# pull Sundial model snapshot from Hugging Face
tollama pull sundial-base-128m

# run forecast
tollama run sundial-base-128m --input examples/sundial_request.json --no-stream
```

If you upgraded dependencies or switched Python environments, restart `tollama serve`
before re-running Sundial so the daemon does not reuse stale runner subprocesses.

```bash
curl -s http://localhost:11435/api/forecast \
  -H 'content-type: application/json' \
  -d @examples/sundial_request.json
```

## Toto Forecasting (separate toto runner family)

Toto (`Datadog/Toto-Open-Base-1.0`) supports target + past numeric covariates in this runner.
Known-future, categorical, and static covariates are unsupported in the first Toto integration.

```bash
# install optional Toto runner dependencies
python -m pip install -e ".[dev,runner_toto]"
# note: install a compatible torch build first for your platform when needed

# run daemon (default http://127.0.0.1:11435)
tollama serve

# pull Toto model snapshot from Hugging Face
tollama pull toto-open-base-1.0

# run forecast
tollama run toto-open-base-1.0 --input examples/toto_request.json --no-stream
```

```bash
curl -s http://localhost:11435/api/forecast \
  -H 'content-type: application/json' \
  -d @examples/toto_request.json
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
- installed + loaded models (including covariate capabilities when available)
- registry-available models with covariate capability metadata
- runner statuses for `mock`, `torch`, `timesfm`, `uni2ts`, `sundial`, and `toto`

`tollama info` renders the same payload and prints a short per-model covariates summary.

Example shape:

```json
{
  "daemon": {"version": "0.1.0", "started_at": "...", "uptime_seconds": 42},
  "paths": {"tollama_home": "...", "config_path": "...", "config_exists": true},
  "config": {"pull": {"https_proxy": "http://***:***@proxy:3128"}},
  "env": {"HTTP_PROXY": "http://***:***@proxy:3128", "TOLLAMA_HF_TOKEN_present": false},
  "pull_defaults": {"offline": {"value": false, "source": "default"}},
  "models": {
    "installed": [{"name": "chronos2", "capabilities": {"past_covariates_numeric": true}}],
    "loaded": [],
    "available": [{"name": "timesfm-2.5-200m", "capabilities": {"future_covariates_numeric": true}}]
  },
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

## OpenClaw Integration (Skill: `tollama-forecast`)

OpenClaw integration is provided by the skill package under
`skills/tollama-forecast/`:

- `SKILL.md`
- `bin/tollama-health.sh`
- `bin/tollama-models.sh`
- `bin/tollama-forecast.sh`
- `examples/*.json`

This integration is OpenClaw-first and does not require any daemon/core/plugin
changes.

### Install (Managed skills, recommended)

```bash
mkdir -p ~/.openclaw/skills
ln -s "$(pwd)/skills/tollama-forecast" ~/.openclaw/skills/tollama-forecast
openclaw skills list --eligible | rg tollama-forecast
```

### Environment checks

```bash
tollama serve
tollama info --json
curl -s http://localhost:11435/api/version
openclaw skills list --eligible | rg tollama-forecast
```

### Runtime defaults and policy

- Base URL: `--base-url` > `TOLLAMA_BASE_URL` > `http://127.0.0.1:11435`
- Timeout: `--timeout` > `TOLLAMA_FORECAST_TIMEOUT_SECONDS` > `300`
- Forecast requests are non-stream by default.
- `tollama-forecast.sh` does not auto-pull by default; model install happens
  only when `--pull` is provided.
- CLI -> HTTP fallback in `tollama-forecast.sh` is enabled only when `tollama`
  is unavailable in PATH.
- HTTP forecast endpoint order is `POST /api/forecast` first, then
  `POST /v1/forecast` only when `/api/forecast` returns `404`.
- `tollama-models.sh available` is daemon-only (`tollama info --json --remote`
  or `GET /api/info`) and does not use local fallback.
- Skill metadata eligibility is `bins=["bash"]` and `anyBins=["tollama","curl"]`.

### Breaking change: skill exit code contract v2

OpenClaw `tollama-forecast` scripts now share this exit code contract:
- `0`: success
- `2`: invalid input/request
- `3`: daemon unreachable/health failure
- `4`: model not installed
- `5`: license required/not accepted/permission
- `6`: timeout
- `10`: unexpected internal error

### Troubleshooting

1. `tollama: command not found`
   - Ensure `tollama` is on system PATH, or prepend venv bin path in OpenClaw:

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

2. Daemon connect failure in OpenClaw but not in local terminal
   - This is usually an exec host mismatch (`sandbox` vs `gateway`) with
     `127.0.0.1`.
   - Set `--base-url` to a daemon address reachable from the current exec host.

3. First-run timeout
   - Increase `--timeout`, or set `TOLLAMA_FORECAST_TIMEOUT_SECONDS`.

4. License-gated models
   - Pull with license acceptance:
     `tollama pull moirai-2.0-R-small --accept-license`

### Skill smoke checks

```bash
bash skills/tollama-forecast/bin/tollama-health.sh --base-url "$TOLLAMA_BASE_URL"
bash skills/tollama-forecast/bin/tollama-models.sh installed --base-url "$TOLLAMA_BASE_URL"
cat skills/tollama-forecast/examples/simple_forecast.json | \
  bash skills/tollama-forecast/bin/tollama-forecast.sh --model mock --base-url "$TOLLAMA_BASE_URL"
```

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
