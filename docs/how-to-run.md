# Run Guide

This guide explains how to run `tollama`, install dependencies, and install all TSFM models currently listed in the repository registry.

## Scope

This guide targets the current TSFM-capable registry entries in `model-registry/registry.yaml` (excluding local `mock`).

## TSFM Model Matrix

| Model name | Family | Hugging Face repo | Revision | License | `--accept-license` required | Runner extra |
|---|---|---|---|---|---|---|
| `chronos2` | `torch` | `amazon/chronos-2` | `main` | `apache-2.0` | No | `runner_torch` |
| `granite-ttm-r2` | `torch` | `ibm-granite/granite-timeseries-ttm-r2` | `90-30-ft-l1-r2.1` | `apache-2.0` | No | `runner_torch` |
| `timesfm-2.5-200m` | `timesfm` | `google/timesfm-2.5-200m-pytorch` | `main` | `apache-2.0` | No | `runner_timesfm` |
| `moirai-2.0-R-small` | `uni2ts` | `Salesforce/moirai-2.0-R-small` | `main` | `cc-by-nc-4.0` | Yes | `runner_uni2ts` |
| `sundial-base-128m` | `sundial` | `thuml/sundial-base-128m` | `main` | `apache-2.0` | No | `runner_sundial` |
| `toto-open-base-1.0` | `toto` | `Datadog/Toto-Open-Base-1.0` | `main` | `apache-2.0` | No | `runner_toto` |

> [!NOTE]
> `timesfm` models may take several minutes to compile on the first run. The default timeout has been increased to 5 minutes to accommodate this, but slower machines may require even more time.

Sundial is target-only in the current runner; do not include covariates in Sundial requests.
Toto supports target + past numeric covariates; known-future/static/categorical covariates are unsupported.

## Requirements

Minimum requirements:

- Python `3.11+`
- `pip` (latest recommended)
- `git`
- Linux/macOS (or compatible shell environment)
- Internet access to pull model snapshots from Hugging Face
- Enough disk space for model snapshots under `~/.tollama/models` (or `$TOLLAMA_HOME/models`)

Recommended:

- A dedicated virtual environment
- A Hugging Face token (`TOLLAMA_HF_TOKEN`) for gated/private models
- Preinstalled PyTorch wheel for your platform if you need GPU acceleration
- Python `3.11` for one environment covering all model families
- Python `3.12+` may fail to resolve/install `gluonts` required by `runner_uni2ts` + `runner_toto`

## Dependency Matrix

`tollama` base dependencies from `pyproject.toml`:

- `fastapi`, `huggingface_hub`, `httpx`, `pydantic`, `pyyaml`, `tqdm`, `typer`, `uvicorn`

Optional extras:

| Extra | Purpose | Packages |
|---|---|---|
| `dev` | Local quality gates | `pytest`, `ruff` |
| `runner_torch` | Chronos + Granite runner dependencies | `chronos-forecasting`, `granite-tsfm`, `pandas`, `numpy` |
| `runner_timesfm` | TimesFM runner dependencies | `numpy`, `pandas`, `huggingface_hub`, `timesfm[torch]` from `git+https://github.com/google-research/timesfm.git@v2.0.0` |
| `runner_uni2ts` | Uni2TS/Moirai runner dependencies | `uni2ts`, `numpy`, `pandas`, `huggingface_hub`, `gluonts` |
| `runner_sundial` | Sundial runner dependencies | `transformers`, `torch`, `numpy`, `pandas`, `huggingface_hub` |
| `runner_toto` | Toto runner dependencies | `toto-ts`, `torch`, `numpy`, `pandas` |

## One-Time Environment Setup

From repository root:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

Install everything needed for development plus all runner families:

```bash
python -m pip install -e ".[dev,runner_torch,runner_timesfm,runner_uni2ts,runner_sundial,runner_toto]"
```

If you only need runtime (no lint/test tooling):

```bash
python -m pip install -e ".[runner_torch,runner_timesfm,runner_uni2ts,runner_sundial,runner_toto]"
```

## Single-Environment Checklist (Legacy)

> [!NOTE]
> **Prefer per-family isolated runtimes** (see section below) to avoid
> dependency conflicts between runner families.  The single-environment approach
> still works but is harder to maintain when different runners require
> incompatible library versions.

Use this checklist to avoid mixed Conda/system/venv runners:

1. Create `.venv` with Python `3.11` and activate it.
2. Install all required extras in that same environment.
3. Verify all command paths come from `.venv`.
4. Start daemon and CLI from `.venv` binaries.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[dev,runner_torch,runner_timesfm,runner_uni2ts,runner_sundial,runner_toto]"

which python
which tollama
which tollama-runner-uni2ts
# each should be under .../tollama/.venv/bin/

./.venv/bin/tollama serve
./.venv/bin/tollama run moirai-2.0-R-small --accept-license --input examples/moirai_2p0_request.json --no-stream --timeout 600

# optional: verify daemon reports runner commands using the same interpreter
curl -s http://127.0.0.1:11435/api/info | rg '"command"|uni2ts|timesfm|sundial|toto'
```

## Per-Family Runtime Isolation (Recommended)

Each runner family can be installed into its own virtualenv under
`~/.tollama/runtimes/<family>/venv/`.  This avoids dependency conflicts
(e.g. different `torch`, `transformers`, `gluonts` versions) and lets each
model run independently.

### Automatic bootstrap (lazy)

When `daemon.auto_bootstrap` is `true` (the default), the daemon will
automatically create the per-family venv the first time a forecast request
needs that runner.  No manual setup is needed — just start the daemon and
send requests:

```bash
tollama serve
tollama run chronos2 --input examples/request.json
# → daemon creates ~/.tollama/runtimes/torch/venv/ on first use
```

### Manual install (eager)

Use the `tollama runtime` CLI to pre-create venvs before they are needed:

```bash
# Install one family
tollama runtime install torch

# Install all families at once
tollama runtime install --all

# List status of all runtimes
tollama runtime list

# Update runtimes after upgrading tollama
tollama runtime update --all

# Remove a runtime
tollama runtime remove torch
tollama runtime remove --all
```

### Configuration

Control bootstrap behavior via `~/.tollama/config.json`:

```json
{
  "version": 1,
  "daemon": {
    "auto_bootstrap": true
  }
}
```

Set `auto_bootstrap` to `false` to disable automatic venv creation.

You can also override runner commands for specific families:

```json
{
  "version": 1,
  "daemon": {
    "auto_bootstrap": true,
    "runner_commands": {
      "torch": ["/custom/venv/bin/python", "-m", "tollama.runners.torch_runner.main"]
    }
  }
}
```

Families with explicit `runner_commands` entries skip auto-bootstrap.

### Runtime directory layout

```text
~/.tollama/runtimes/
├── torch/
│   ├── venv/bin/python
│   └── installed.json
├── timesfm/
│   ├── venv/bin/python
│   └── installed.json
├── uni2ts/
│   └── ...
├── sundial/
│   └── ...
└── toto/
    └── ...
```

Each `installed.json` records the tollama version, extra name, Python version,
and install timestamp.  When tollama is upgraded, runtimes are automatically
re-bootstrapped on next use (or can be updated manually with
`tollama runtime update --all`).

## Environment Variables and Paths

Important variables:

- `TOLLAMA_HOME`: overrides default state root (`~/.tollama`)
- `TOLLAMA_HOST`: bind host and port in `host:port` format for daemon process
- `TOLLAMA_FORECAST_TIMEOUT_SECONDS`: daemon runner-call timeout for forecast/unload (default `300`)
- `TOLLAMA_HF_TOKEN`: Hugging Face token used by pull operations
- `HTTP_PROXY`, `HTTPS_PROXY`, `NO_PROXY`: optional proxy settings for pulls

Default state layout:

```text
~/.tollama/
├── config.json
├── models/
│   └── <model-name>/
│       ├── manifest.json
│       └── snapshot/
└── runtimes/
```

## Start the Daemon

Terminal 1:

```bash
tollama serve
```

Custom bind example:

```bash
TOLLAMA_HOST=0.0.0.0:11435 tollamad
```

Terminal 2 quick checks:

```bash
curl http://localhost:11435/api/version
curl http://localhost:11435/v1/health
```

## Install All TSFM Models

If any model is gated/private in your environment, set a token first:

```bash
export TOLLAMA_HF_TOKEN=hf_xxx
```

Pull models that do not require explicit acceptance:

```bash
for model in chronos2 granite-ttm-r2 timesfm-2.5-200m sundial-base-128m toto-open-base-1.0; do
  tollama pull "$model" --no-stream
done
```

Pull models that require explicit `--accept-license`:

```bash
tollama pull moirai-2.0-R-small --accept-license --no-stream
```

Single command block version:

```bash
set -euo pipefail

for model in chronos2 granite-ttm-r2 timesfm-2.5-200m sundial-base-128m toto-open-base-1.0; do
  tollama pull "$model" --no-stream
done

tollama pull moirai-2.0-R-small --accept-license --no-stream
```

## Verify Installation

List installed models:

```bash
tollama list
```

Inspect one model manifest:

```bash
tollama show timesfm-2.5-200m
```

Verify all expected model directories exist:

```bash
BASE="${TOLLAMA_HOME:-$HOME/.tollama}/models"
for model in chronos2 granite-ttm-r2 timesfm-2.5-200m moirai-2.0-R-small sundial-base-128m toto-open-base-1.0; do
  test -f "$BASE/$model/manifest.json" && echo "ok: $model" || echo "missing: $model"
done
```

## Smoke Forecast Commands

Run one forecast per family:

```bash
tollama run chronos2 --input examples/chronos2_request.json --no-stream
tollama run granite-ttm-r2 --input examples/granite_ttm_request.json --no-stream
tollama run timesfm-2.5-200m --input examples/timesfm_2p5_request.json --no-stream
tollama run moirai-2.0-R-small --input examples/moirai_2p0_request.json --no-stream
tollama run sundial-base-128m --input examples/sundial_request.json --no-stream
tollama run toto-open-base-1.0 --input examples/toto_request.json --no-stream
```

## Useful Pull Controls

Common options for `tollama pull`:

- `--offline` / `--no-offline`
- `--local-files-only` / `--no-local-files-only`
- `--http-proxy`, `--https-proxy`, `--no-proxy`
- `--hf-home`
- `--token`
- `--insecure` (debug only; disables SSL verification)
- `--no-config` (ignore daemon pull defaults for this request)

Persistent defaults:

```bash
tollama config set pull.https_proxy http://proxy:3128
tollama config set pull.hf_home /mnt/fastcache/hf
tollama config set pull.offline true
tollama config list
```

## Troubleshooting

License acceptance error (`409`):

- If pull output says license acceptance is required, re-run with `--accept-license`.

Missing runner dependency errors:

- Install the matching extras:
  - `python -m pip install -e ".[runner_torch]"`
  - `python -m pip install -e ".[runner_timesfm]"`
  - `python -m pip install -e ".[runner_uni2ts]"`
  - `python -m pip install -e ".[runner_sundial]"`
  - `python -m pip install -e ".[runner_toto]"`
- If `runner_uni2ts` install fails on Python 3.12+ (build backend/dependency issues),
  recreate the venv with Python 3.11 and reinstall.

Dependency resolver says `gluonts` has no matching distribution / install conflict:

- This usually means the environment is Python `3.12+`.
- Recreate the virtual environment with Python `3.11` and reinstall extras.


Offline/local-files-only pull failures:

- Pull at least once without `--offline`/`--local-files-only` to seed cache.

Authentication errors from Hugging Face:

- Export a valid token: `export TOLLAMA_HF_TOKEN=hf_xxx`
- Confirm your account has access to the model repository.

Daemon unreachable:

- Ensure `tollama serve` is running.
- Confirm host/port (`http://localhost:11435` by default).
- Use `tollama info --remote` to force daemon diagnostics.

Runner unavailable after restart:

- Most commonly caused by mixed environments (for example daemon in one Python, runner scripts in another).
- Recreate one `.venv` with Python 3.11, reinstall all runner extras, and launch with `./.venv/bin/tollama`.
- Also restart the daemon after dependency upgrades so stale runner subprocesses are not reused.

Sundial `shape '[-1, 2]' is invalid for input of size 1`:

- Use latest `tollama` code and restart `tollama serve` (this was caused by legacy generation-path compatibility).
- Verify the running daemon is from the same `.venv` as your CLI: `which python`, `which tollama`.

Forecast timeout (`failed: timed out` / runner timeout):

- Increase CLI timeout for slower first-run inference:
  - `tollama run moirai-2.0-R-small --input examples/moirai_2p0_request.json --no-stream --timeout 600`
- The `--timeout` flag now propagates to the daemon, overriding the default.
- You can also set a higher default for the daemon:
  - `export TOLLAMA_FORECAST_TIMEOUT_SECONDS=600`

## Development Checks

```bash
ruff check .
pytest -q
```
