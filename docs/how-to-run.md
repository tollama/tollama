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

Sundial is target-only in the current runner; do not include covariates in Sundial requests.

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

## Dependency Matrix

`tollama` base dependencies from `pyproject.toml`:

- `fastapi`, `huggingface_hub`, `httpx`, `pydantic`, `pyyaml`, `tqdm`, `typer`, `uvicorn`

Optional extras:

| Extra | Purpose | Packages |
|---|---|---|
| `dev` | Local quality gates | `pytest`, `ruff` |
| `runner_torch` | Chronos + Granite runner dependencies | `chronos-forecasting`, `granite-tsfm`, `pandas`, `numpy` |
| `runner_timesfm` | TimesFM runner dependencies | `numpy`, `pandas`, `huggingface_hub`, `timesfm[torch]` from `git+https://github.com/google-research/timesfm.git` |
| `runner_uni2ts` | Uni2TS/Moirai runner dependencies | `uni2ts`, `numpy`, `pandas`, `huggingface_hub`, `gluonts` |
| `runner_sundial` | Sundial runner dependencies | `transformers`, `torch`, `numpy`, `pandas`, `huggingface_hub` |

## One-Time Environment Setup

From repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

Install everything needed for development plus all runner families:

```bash
python -m pip install -e ".[dev,runner_torch,runner_timesfm,runner_uni2ts,runner_sundial]"
```

If you only need runtime (no lint/test tooling):

```bash
python -m pip install -e ".[runner_torch,runner_timesfm,runner_uni2ts,runner_sundial]"
```

## Environment Variables and Paths

Important variables:

- `TOLLAMA_HOME`: overrides default state root (`~/.tollama`)
- `TOLLAMA_HOST`: bind host and port in `host:port` format for daemon process
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
for model in chronos2 granite-ttm-r2 timesfm-2.5-200m sundial-base-128m; do
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

for model in chronos2 granite-ttm-r2 timesfm-2.5-200m sundial-base-128m; do
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
for model in chronos2 granite-ttm-r2 timesfm-2.5-200m moirai-2.0-R-small sundial-base-128m; do
  test -f "$BASE/$model/manifest.json" && echo "ok: $model" || echo "missing: $model"
done
```

## Smoke Forecast Commands

Run one forecast per family:

```bash
tollama run chronos2 --input examples/chronos2_request.json --no-stream
tollama run granite-ttm-r2 --input examples/granite_ttm_request.json --no-stream
tollama run timesfm-2.5-200m --input examples/timesfm_2p5_request.json --no-stream
tollama run moirai-2.0-R-small --input examples/moirai_request.json --no-stream
tollama run sundial-base-128m --input examples/sundial_request.json --no-stream
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

Offline/local-files-only pull failures:

- Pull at least once without `--offline`/`--local-files-only` to seed cache.

Authentication errors from Hugging Face:

- Export a valid token: `export TOLLAMA_HF_TOKEN=hf_xxx`
- Confirm your account has access to the model repository.

Daemon unreachable:

- Ensure `tollama serve` is running.
- Confirm host/port (`http://localhost:11435` by default).
- Use `tollama info --remote` to force daemon diagnostics.

## Development Checks

```bash
ruff check .
pytest -q
```
