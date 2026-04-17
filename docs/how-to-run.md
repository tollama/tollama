# Run Guide

This guide explains how to run `tollama`, install dependencies, and install all TSFM models currently listed in the repository registry.

## Scope

This guide targets the current TSFM-capable registry entries in `model-registry/registry.yaml` (excluding local `mock`).

> [!IMPORTANT]
> **Prerequisites before you start**
> - Python **3.11** recommended (3.12/3.13 work for most families; Uni2TS/Moirai may have issues on 3.12+)
> - ~**5 GB** free disk space per model family under `~/.tollama/models/`
> - Internet access for pulling model snapshots from Hugging Face
> - A [Hugging Face token](https://huggingface.co/settings/tokens) (`TOLLAMA_HF_TOKEN`) for gated models such as Moirai

## Notebooks

- SDK quickstart notebook: `examples/quickstart.ipynb`
- Agent-oriented notebook: `examples/agent_demo.ipynb`
- Tutorial notebooks:
  - `examples/tutorial_covariates.ipynb`
  - `examples/tutorial_comparison.ipynb`
  - `examples/tutorial_what_if.ipynb`
  - `examples/tutorial_auto_forecast.ipynb`

Additional docs:

- Troubleshooting (canonical): `docs/troubleshooting.md`
- CLI cheat sheet: `docs/cli-cheatsheet.md`
- API reference: `docs/api-reference.md`
- Dashboard user guide (Web GUI + TUI): `docs/dashboard-user-guide.md`
- macOS app guide: `docs/macos-app.md`

## Forecast Model Matrix

| Model name | Family | Hugging Face repo | Revision | License | `--accept-license` required | Runner extra |
|---|---|---|---|---|---|---|
| `chronos2` | `torch` | `amazon/chronos-2` | `main` | `apache-2.0` | No | `runner_torch` |
| `granite-ttm-r2` | `torch` | `ibm-granite/granite-timeseries-ttm-r2` | `512-96-ft-l1-r2.1` | `apache-2.0` | No | `runner_torch` |
| `timesfm-2.5-200m` | `timesfm` | `google/timesfm-2.5-200m-pytorch` | `main` | `apache-2.0` | No | `runner_timesfm` |
| `moirai-2.0-R-small` | `uni2ts` | `Salesforce/moirai-2.0-R-small` | `main` | `cc-by-nc-4.0` | Yes | `runner_uni2ts` |
| `sundial-base-128m` | `sundial` | `thuml/sundial-base-128m` | `main` | `apache-2.0` | No | `runner_sundial` |
| `toto-open-base-1.0` | `toto` | `Datadog/Toto-Open-Base-1.0` | `main` | `apache-2.0` | No | `runner_toto` |
| `lag-llama` | `lag_llama` | `time-series-foundation-models/Lag-Llama` | `main` | `apache-2.0` | No | `runner_lag_llama` |
| `patchtst` | `patchtst` | `ibm-granite/granite-timeseries-patchtst` | `main` | `apache-2.0` | No | `runner_patchtst` |
| `tide` | `tide` | `tollama/tide-runner` (local source manifest) | `main` | `apache-2.0` | No | `runner_tide` |
| `nhits` | `nhits` | `tollama/nhits-runner` (local source manifest) | `main` | `apache-2.0` | No | `runner_nhits` |
| `nbeatsx` | `nbeatsx` | `tollama/nbeatsx-runner` (local source manifest) | `main` | `apache-2.0` | No | `runner_nbeatsx` |
| `timer-base` | `timer` | `thuml/timer-base-84m` | `main` | `apache-2.0` | No | `runner_timer` |
| `timemixer-base` | `timemixer` | `thuml/timemixer` | `main` | `apache-2.0` | No | `runner_timemixer` |
| `forecastpfn` | `forecastpfn` | `abacusai/ForecastPFN` | `main` | `apache-2.0` | No | `runner_forecastpfn` |

> [!NOTE]
> `timesfm` models may take several minutes to compile on the first run. The default timeout has been increased to 5 minutes to accommodate this, but slower machines may require even more time.

Sundial and TiDE are target-only in the current runners; do not include covariates or static features in those requests.
Toto supports target + past numeric covariates; known-future/static/categorical covariates are unsupported.
Timer, TimeMixer, and ForecastPFN are currently target-only runner integrations and return
canonical mean forecasts without quantiles.

> [!IMPORTANT]
> `patchtst` is a **Phase-2 baseline integration**: it is discoverable/pullable and now executes canonical target-only forecasts via the dedicated runner family. Quantiles are returned when the backend exposes them; otherwise the runner returns mean-only forecasts with a warning. If dependencies are missing, the runner returns `DEPENDENCY_MISSING` with the install command `python -m pip install -e ".[dev,runner_patchtst]"`.
>
> `nhits` is a **Phase-4 quality integration**: it is discoverable/pullable and executes canonical single/multi-series forecasts via the dedicated runner family with stricter input/frequency validation. Quantiles use backend outputs when exposed; otherwise the runner returns calibrated residual-based fallback quantiles with explicit warnings. Numeric covariates/static features are supported in practical best-effort mode, with strict-mode validation (`parameters.covariates_mode=strict`) for hard failures on unsupported/non-numeric values. If dependencies are missing, the runner returns `DEPENDENCY_MISSING` with the install command `python -m pip install -e ".[dev,runner_nhits]"`.
>
> `nbeatsx` is a **Phase-4 quality integration**: it is discoverable/pullable and executes canonical single/multi-series forecasts via the dedicated runner family with stricter input/frequency validation. Quantiles use backend outputs when exposed; otherwise the runner returns calibrated residual-based fallback quantiles with explicit warnings. Numeric covariates/static features are supported in practical best-effort mode, with strict-mode validation (`parameters.covariates_mode=strict`) for hard failures on unsupported/non-numeric values. If dependencies are missing, the runner returns `DEPENDENCY_MISSING` with the install command `python -m pip install -e ".[dev,runner_nbeatsx]"`.

## Expected Failure Signatures vs Regressions

Use these signatures when triaging smoke failures:

- **Expected dependency-gated** (environment/setup issue):
  - HTTP `503` from `/v1/forecast`
  - detail contains `DEPENDENCY_MISSING`
  - install hint includes the family extra (for example `runner_patchtst`, `runner_tide`, `runner_nhits`, `runner_nbeatsx`)
- **Regression** (runtime registration/config issue):
  - detail contains `runner family '<family>' is not supported`
  - or pull fails for local-source models (`tide`, `nhits`, `nbeatsx`) that should be manifest-only

For `tide`, `nhits`, and `nbeatsx`, `tollama pull` is manifest-only (`source.type=local`), so pull should succeed without Hugging Face auth/network snapshot fetches.

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
- Python `3.11` for stable per-family runtime bootstrap
- Python `3.12+` may fail to resolve/install `gluonts` required by `runner_uni2ts` + `runner_toto`

## macOS App (DMG)

If you prefer a drag-and-drop macOS app instead of a Python-first setup:

1. download the Apple Silicon DMG from GitHub Releases
2. drag `Tollama.app` into `Applications`
3. open the app and let it bootstrap its private runtime

For installer-flow testing, you can use the Apple Silicon PKG instead. The PKG
installs the same `Tollama.app` bundle into `/Applications`.

The app stores its own runtime and state under:

- `~/Library/Application Support/Tollama/runtime`
- `~/Library/Application Support/Tollama/state`
- `~/Library/Logs/Tollama/daemon.log`

The macOS app bundles the Tollama core plus the default starter-model runner
extra (`runner_sundial`) so the built-in starter flow can forecast immediately
after pull. Other runner extras and model weights are still installed on
demand after launch.

## Dependency Matrix

`tollama` base dependencies from `pyproject.toml`:

- `fastapi`, `huggingface_hub`, `httpx`, `pydantic`, `pyyaml`, `tqdm`, `typer`, `uvicorn`

Optional extras:

| Extra | Purpose | Packages |
|---|---|---|
| `dev` | Local quality gates | `pytest`, `ruff` |
| `mcp` | MCP server integration | `mcp`, `httpx` |
| `langchain` | Python SDK LangChain tool wrappers | `langchain-core` |
| `runner_torch` | Chronos + Granite runner dependencies | `chronos-forecasting`, `granite-tsfm`, `pandas`, `numpy` |
| `runner_timesfm` | TimesFM runner dependencies | `numpy`, `pandas`, `huggingface_hub`, `timesfm[torch,xreg]` from `https://github.com/google-research/timesfm/archive/2dcc66fbfe2155adba1af66aa4d564a0ee52f61e.tar.gz` |
| `runner_uni2ts` | Uni2TS/Moirai runner dependencies | `uni2ts`, `numpy`, `pandas`, `huggingface_hub`, `gluonts` |
| `runner_sundial` | Sundial runner dependencies | `transformers`, `torch`, `numpy`, `pandas`, `huggingface_hub` |
| `runner_toto` | Toto runner dependencies | `toto-ts`, `torch`, `numpy`, `pandas` |
| `runner_lag_llama` | Lag-Llama runner dependencies | `lag-llama`, `gluonts`, `lightning`, `huggingface_hub`, `numpy`, `pandas`, `wandb` |
| `runner_patchtst` | PatchTST runner dependencies | `transformers`, `torch` |
| `runner_tide` | TiDE runner dependencies | `u8darts`, `torch` |
| `runner_nhits` | N-HiTS runner dependencies | `neuralforecast`, `pytorch-lightning`, `torch` |
| `runner_nbeatsx` | N-BEATSx runner dependencies | `neuralforecast`, `pytorch-lightning`, `torch` |
| `runner_timer` | Timer runner dependencies | `transformers`, `torch`, `numpy`, `pandas`, `huggingface_hub` |
| `runner_timemixer` | TimeMixer runner dependencies | `transformers`, `torch`, `numpy`, `pandas`, `huggingface_hub` |
| `runner_forecastpfn` | ForecastPFN runner dependencies | `ForecastPFN`, `torch`, `numpy` |

## One-Time Environment Setup (Default: Per-Family Runtime Isolation)

From repository root:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

Install everything needed for development plus all runner families:

```bash
python -m pip install -e ".[dev]"
```

With the default `daemon.auto_bootstrap=true`, family runtimes are created lazily under
`~/.tollama/runtimes/<family>/venv/` on first use.

If you prefer single-environment mode (legacy), install all runner extras in the same `.venv`:

```bash
python -m pip install -e ".[dev,runner_torch,runner_timesfm,runner_uni2ts,runner_sundial,runner_toto,runner_lag_llama,runner_patchtst,runner_tide,runner_nhits,runner_nbeatsx,runner_timer,runner_timemixer,runner_forecastpfn]"
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
python -m pip install -e ".[dev,runner_torch,runner_timesfm,runner_uni2ts,runner_sundial,runner_toto,runner_lag_llama,runner_patchtst,runner_tide,runner_nhits,runner_nbeatsx,runner_timer,runner_timemixer,runner_forecastpfn]"

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
(e.g. different `torch`, `transformers`, `gluonts` versions) and isolates
runner families from each other.

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

If stdlib `venv` / `ensurepip` fails on the host interpreter, Tollama now
falls back to `uv venv --seed --clear ... --python <current-interpreter>` when
`uv` is available.

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

### Full-model E2E smoke

Run the repository helper to execute pull+forecast smoke tests across all
currently registered models with model-appropriate payloads:

```bash
scripts/run_all_models_e2e_local.sh
```

This helper auto-generates long-context payloads for families that require
substantial history windows (for example, Granite TTM / PatchTST / TiDE /
N-HiTS / N-BEATSx) and uses a PatchTST-compatible horizon in the request.

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
├── toto/
│   └── ...
├── lag_llama/
│   └── ...
├── patchtst/
│   └── ...
├── tide/
│   └── ...
├── nhits/
│   └── ...
├── nbeatsx/
│   └── ...
├── timer/
│   └── ...
├── timemixer/
│   └── ...
└── forecastpfn/
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

## Docker One-Liner

Build image:

```bash
docker build -t tollama/tollama .
```

Run daemon container:

```bash
docker run -p 11435:11435 tollama/tollama
```

## 5-Minute Quickstart

With daemon running in one terminal:

```bash
tollama quickstart
```

This command:

- verifies daemon reachability
- pulls `mock` (or `--model <name>`)
- runs a demo non-streaming forecast
- prints next-step commands

Useful options:

- `--model` to switch model for the demo
- `--horizon` to change forecast horizon
- `--accept-license` for gated model pulls
- `--base-url` and `--timeout` for remote daemon setups

## Python SDK Convenience API

The base `tollama` package includes a high-level SDK facade:

```python
from tollama import Tollama

t = Tollama()
result = t.forecast(
    model="chronos2",
    series={"target": [10, 11, 12, 13, 14], "freq": "D"},
    horizon=3,
)
print(result.mean)
print(result.quantiles)
print(result.to_df())
```

`Tollama.forecast(...)` accepts:

- `dict` series payload (`target` required)
- list of series payload dicts
- `pandas.Series`
- `pandas.DataFrame` (wide or long-format with `target`)

## Install All Forecast Models

If any model is gated/private in your environment, set a token first:

```bash
export TOLLAMA_HF_TOKEN=hf_xxx
```

Pull models that do not require explicit acceptance:

```bash
for model in chronos2 granite-ttm-r2 timesfm-2.5-200m sundial-base-128m toto-open-base-1.0 lag-llama patchtst tide nhits nbeatsx timer-base timemixer-base forecastpfn; do
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

for model in chronos2 granite-ttm-r2 timesfm-2.5-200m sundial-base-128m toto-open-base-1.0 lag-llama patchtst tide nhits nbeatsx timer-base timemixer-base forecastpfn; do
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
for model in chronos2 granite-ttm-r2 timesfm-2.5-200m moirai-2.0-R-small sundial-base-128m toto-open-base-1.0 lag-llama patchtst tide nhits nbeatsx timer-base timemixer-base forecastpfn; do
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
tollama run lag-llama --input examples/lag_llama_request.json --no-stream
tollama run patchtst --input examples/request.json --no-stream
tollama run tide --input examples/request.json --no-stream
tollama run nhits --input examples/request.json --no-stream
tollama run nbeatsx --input examples/request.json --no-stream
tollama run timer-base --input examples/request.json --no-stream
tollama run timemixer-base --input examples/request.json --no-stream
tollama run forecastpfn --input examples/request.json --no-stream
```

## Real-Data E2E Matrix (Kaggle + Open Data)

Use the real-data harness to run contract-gate + benchmark checks for the 6-model TSFM matrix.

```bash
# PR smoke mode: 1 sample per dataset, falls back to open-only if Kaggle creds are missing
python scripts/e2e_realdata/run_tsfm_realdata.py \
  --mode pr \
  --model all \
  --gate-profile strict \
  --base-url http://127.0.0.1:11435 \
  --output-dir artifacts/realdata/pr

# Nightly mode: 4 samples per dataset, requires KAGGLE_USERNAME/KAGGLE_KEY
python scripts/e2e_realdata/run_tsfm_realdata.py \
  --mode nightly \
  --model all \
  --gate-profile strict \
  --base-url http://127.0.0.1:11435 \
  --output-dir artifacts/realdata/nightly

# Local mode: requires KAGGLE_USERNAME/KAGGLE_KEY by default
# Add --allow-kaggle-fallback to run open datasets only when credentials are missing.
python scripts/e2e_realdata/run_tsfm_realdata.py \
  --mode local \
  --model all \
  --gate-profile strict \
  --allow-kaggle-fallback \
  --base-url http://127.0.0.1:11435 \
  --output-dir artifacts/realdata/local-open-fallback
```

Alternative wrapper:

```bash
bash scripts/e2e_realdata_tsfm.sh pr all http://127.0.0.1:11435 artifacts/realdata/wrapper false
# Set PYTHON_BIN=/path/to/python if you need to force a specific interpreter
# Set PYTHON_SKIP_PROBE=1 only if you intentionally want to bypass startup/runtime preflight
```

Each run writes:

- `result.json`: raw scenario/model run entries
- `summary.json`: aggregated pass/fail + latency/metric summary
- `summary.md`: human-readable leaderboard
- `benchmark_report.json`: detailed benchmark-only report with dataset/model breakdowns
- `benchmark_report.md`: human-readable benchmark report
- `raw/`: per-request payload/response captures

`--gate-profile strict` is the CI default. For optional HuggingFace local runs use
`--gate-profile hf_optional` to downgrade HF data/payload issues to `skip`.

## HuggingFace Optional Local Profile

Generate a quality-gated catalog plus rejection report:

```bash
python scripts/e2e_realdata/gather_hf_datasets.py \
  --output scripts/e2e_realdata/hf_dataset_catalog.yaml \
  --rejections-output scripts/e2e_realdata/hf_dataset_rejections.json
```

Run the local optional HF starter profile. This lane covers 11 models:
the 6 TSFMs in the strict matrix plus `lag-llama`, `patchtst`, `tide`, `nhits`,
and `nbeatsx`. It uses the curated 10-dataset starter catalog, prefers cached
HF snapshot files when present, samples 1 series per dataset, and the wrapper
defaults dataset preparation to `HF_STARTER_CONTEXT_CAP=256`. When a starter
dataset declares explicit `series_id_columns`, the parser now scans deeper into
interleaved panel snapshots so datasets such as `kashif/App_Flow` can still
produce deterministic windows.

```bash
python scripts/e2e_realdata/run_tsfm_realdata.py \
  --mode local \
  --model hf_all \
  --catalog-path scripts/e2e_realdata/hf_dataset_catalog_starter.yaml \
  --gate-profile hf_optional \
  --max-series-per-dataset 1 \
  --allow-kaggle-fallback \
  --output-dir artifacts/realdata/hf-local
```

Wrapper script:

```bash
bash scripts/e2e_realdata_hf.sh all http://127.0.0.1:11435 artifacts/realdata/hf-local
# Set PYTHON_BIN=/path/to/python if you need to force a specific interpreter
# Set HF_STARTER_CONTEXT_CAP=<n> if you need a different dataset-prep window
# Set PYTHON_SKIP_PROBE=1 only if you intentionally want to bypass startup/runtime preflight
```

HF starter runs emit the standard `result.json`, `summary.json`, `summary.md`,
and `raw/` artifacts plus mandatory `benchmark_report.json` and
`benchmark_report.md` outputs containing benchmark rows, model leaderboards,
dataset breakdowns, failure classifications, and separate contract summaries.

If the wrapper hangs during interpreter startup or imports, run:

```bash
bash scripts/e2e_realdata_runtime_diag.sh
# Or force an interpreter:
PYTHON_BIN=/path/to/python bash scripts/e2e_realdata_runtime_diag.sh
# To capture a stack sample for timed out imports:
CAPTURE_STACK_SAMPLE_ON_TIMEOUT=1 bash scripts/e2e_realdata_runtime_diag.sh
```

The real-data wrappers preflight the chosen interpreter with `python -V` and
`import ssl, yaml, httpx` before starting the harness. If no candidate passes,
the wrapper fails fast and points to the diagnostics helper instead of hanging.

## Shell Completion

Install or inspect completion scripts:

```bash
tollama --install-completion
tollama --show-completion
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
- `--progress auto|on|off` (auto uses terminal detection)

Persistent defaults:

```bash
tollama config init
tollama config keys
tollama config set pull.https_proxy http://proxy:3128
tollama config set pull.hf_home /mnt/fastcache/hf
tollama config set pull.offline true
tollama config list
```

If you pass an unknown config key, `tollama config set/get/unset` suggests the closest
supported key (for example `offline` -> `pull.offline`).

`tollama run` also supports:

- `--progress auto|on|off`
- omitting `MODEL` in an interactive terminal to choose from installed models
- `--interactive` to select an example request file when `--input` is omitted

## Troubleshooting

The canonical troubleshooting guide is now:

- `docs/troubleshooting.md`

Quick triage commands:

```bash
tollama info
tollama doctor
tollama runtime list
tollama list
tollama ps
```

## Integration Test Status

Weight-backed integration tests currently cover the primary pulled families below. To run
them locally against real model weights (requires models pulled and internet access):

```bash
TOLLAMA_RUN_INTEGRATION_TESTS=1 TOLLAMA_TOTO_INTEGRATION_CPU=1 pytest -q -rs \
  tests/test_chronos_integration.py \
  tests/test_granite_integration.py \
  tests/test_timesfm_integration.py \
  tests/test_uni2ts_integration.py \
  tests/test_sundial_integration.py \
  tests/test_toto_integration.py
```

Additional runner families (`lag-llama`, `patchtst`, `tide`, `nhits`, `nbeatsx`,
`timer`, `timemixer`, `forecastpfn`) currently rely on dedicated runner/adapter tests plus
`scripts/run_all_models_e2e_local.sh` for cross-family smoke coverage.

See `docs/releases/v0.1.0.md` for per-family compatibility notes and known limitations.

## LangChain Support

Install optional LangChain support:

```bash
python -m pip install -e ".[langchain]"
```

Available wrappers (in `src/tollama/skill/langchain.py`):

- `TollamaHealthTool`
- `TollamaModelsTool`
- `TollamaShowTool`
- `TollamaPullTool`
- `TollamaForecastTool`
- `TollamaAutoForecastTool`
- `TollamaAnalyzeTool`
- `TollamaGenerateTool`
- `TollamaCounterfactualTool`
- `TollamaScenarioTreeTool`
- `TollamaReportTool`
- `TollamaWhatIfTool`
- `TollamaPipelineTool`
- `TollamaCompareTool`
- `TollamaRecommendTool`
- `get_tollama_tools(base_url="http://127.0.0.1:11435", timeout=10.0)`

Canonical cross-framework inventory:

- `docs/agent-tools.md`

Quick smoke:

```python
from tollama.skill import get_tollama_tools

tools = {tool.name: tool for tool in get_tollama_tools()}
print(tools["tollama_health"].invoke({}))
print(tools["tollama_models"].invoke({"mode": "installed"}))
print(
    tools["tollama_recommend"].invoke(
        {
            "horizon": 12,
            "freq": "D",
            "has_future_covariates": True,
            "covariates_type": "numeric",
            "top_k": 3,
        }
    )
)
print(
    tools["tollama_forecast"].invoke(
        {
            "request": {
                "model": "mock",
                "horizon": 2,
                "series": [
                    {
                        "id": "s1",
                        "freq": "D",
                        "timestamps": ["2025-01-01", "2025-01-02"],
                        "target": [1.0, 2.0],
                    }
                ],
                "options": {},
            }
        }
    )
)
print(
    tools["tollama_analyze"].invoke(
        {
            "request": {
                "series": [
                    {
                        "id": "s1",
                        "freq": "D",
                        "timestamps": ["2025-01-01", "2025-01-02", "2025-01-03"],
                        "target": [1.0, 2.0, 1.5],
                    }
                ]
            }
        }
    )
)
print(
    tools["tollama_compare"].invoke(
        {
            "request": {
                "models": ["mock", "chronos2"],
                "horizon": 2,
                "series": [
                    {
                        "id": "s1",
                        "freq": "D",
                        "timestamps": ["2025-01-01", "2025-01-02"],
                        "target": [1.0, 2.0],
                    }
                ],
                "options": {},
            }
        }
    )
)
```

Tool contract summary:

- `tollama_health`: no args, returns daemon `health` + `version`
- `tollama_models`: args `{"mode":"installed|loaded|available"}`
- `tollama_show`: args `{"model": "<model-name>"}`
- `tollama_pull`: args `{"model": "<model-name>", "accept_license"?: bool}`
- `tollama_forecast`: args `{"request": <ForecastRequest dict>}`
- `tollama_auto_forecast`: args `{"request": <AutoForecastRequest dict>}`
- `tollama_analyze`: args `{"request": <AnalyzeRequest dict>}`
- `tollama_generate`: args `{"request": <GenerateRequest dict>}`
- `tollama_counterfactual`: args `{"request": <CounterfactualRequest dict>}`
- `tollama_scenario_tree`: args `{"request": <ScenarioTreeRequest dict>}`
- `tollama_report`: args `{"request": <ReportRequest dict>}`
- `tollama_what_if`: args `{"request": <WhatIfRequest dict>}`
- `tollama_pipeline`: args `{"request": <PipelineRequest dict>}`
- `tollama_compare`: args `{"request": <CompareRequest dict>}`
- `tollama_recommend`: args `{"horizon": int, ...}` for ranked model hints
- all wrappers return structured `dict` payloads; errors use
  `{"error":{"category","exit_code","message"}}`
- async invocations are fully implemented via `_arun` + `AsyncTollamaClient`
- missing optional dependency hint:
  `pip install "tollama[langchain]"`
- MCP-only trust/XAI tools are intentionally left out of the default LangChain
  and shared-wrapper surface; see `docs/agent-tools.md`

LangChain wrapper validation command:

```bash
PYTHONPATH=src python -m pytest -q tests/test_langchain_skill.py
```

## Additional Agent Wrappers

`tollama` also provides wrapper factories for CrewAI, AutoGen, and smolagents.
These wrappers reuse the shared `framework_common.py` subset, whose current
coverage is documented in `docs/agent-tools.md`.

```python
from tollama.skill import (
    get_autogen_function_map,
    get_autogen_tool_specs,
    get_crewai_tools,
    get_smolagents_tools,
)

crewai_tools = get_crewai_tools()
smolagents_tools = get_smolagents_tools()
autogen_tools = get_autogen_tool_specs()
autogen_function_map = get_autogen_function_map()
```

Validation test:

```bash
PYTHONPATH=src python -m pytest -q tests/test_agent_wrappers.py
```

## Benchmark

Compare time-to-first-forecast and request ergonomics (`LOC`) between the SDK
and raw client calls:

```bash
PYTHONPATH=src python benchmarks/tollama_vs_raw.py --model mock --iterations 3 --warmup 1
```

Cross-model TSFM benchmark + routing-default recommendation (Lag-Llama, PatchTST,
TiDE, N-HiTS, N-BEATSx):

```bash
# Template-only protocol/report artifact (works without a running daemon)
PYTHONPATH=src python benchmarks/cross_model_tsfm.py \
  --template-only \
  --output-dir benchmarks/reports/cross_model_baseline

# Full benchmark run against a running daemon
PYTHONPATH=src python benchmarks/cross_model_tsfm.py \
  --base-url http://127.0.0.1:11435 \
  --output-dir artifacts/benchmarks/cross_model
```

Artifacts:

- `result.json` / `result.md`: per-model per-dataset quality + latency results and
  a routing recommendation (`default`, `fast_path`, `high_accuracy`)
- `report_template.json` / `report_template.md`: protocol/report template when
  execution environment is unavailable

See `docs/tsfm-routing-defaults.md` for protocol details and policy interpretation.

## Development Checks

```bash
ruff check .
pytest -q
```
