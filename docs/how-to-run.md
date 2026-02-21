# Run Guide

This guide explains how to run `tollama`, install dependencies, and install all TSFM models currently listed in the repository registry.

## Scope

This guide targets the current TSFM-capable registry entries in `model-registry/registry.yaml` (excluding local `mock`).

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
- Python `3.11` for stable per-family runtime bootstrap
- Python `3.12+` may fail to resolve/install `gluonts` required by `runner_uni2ts` + `runner_toto`

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
| `runner_timesfm` | TimesFM runner dependencies | `numpy`, `pandas`, `huggingface_hub`, `timesfm[torch]` from `git+https://github.com/google-research/timesfm.git@2dcc66fbfe2155adba1af66aa4d564a0ee52f61e` |
| `runner_uni2ts` | Uni2TS/Moirai runner dependencies | `uni2ts`, `numpy`, `pandas`, `huggingface_hub`, `gluonts` |
| `runner_sundial` | Sundial runner dependencies | `transformers`, `torch`, `numpy`, `pandas`, `huggingface_hub` |
| `runner_toto` | Toto runner dependencies | `toto-ts`, `torch`, `numpy`, `pandas` |

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
python -m pip install -e ".[dev,runner_torch,runner_timesfm,runner_uni2ts,runner_sundial,runner_toto]"
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

## Integration Matrix Snapshot (2026-02-17)

Latest optional integration run (`TOLLAMA_RUN_INTEGRATION_TESTS=1`) result:

| Model | Result | Notes |
|---|---|---|
| `chronos2` | pass | `tests/test_chronos_integration.py` |
| `granite-ttm-r2` | pass | `tests/test_granite_integration.py` |
| `timesfm-2.5-200m` | pass | `tests/test_timesfm_integration.py` |
| `moirai-2.0-R-small` | pass | `tests/test_uni2ts_integration.py` |
| `sundial-base-128m` | pass | `tests/test_sundial_integration.py` |
| `toto-open-base-1.0` | skipped | `tests/test_toto_integration.py` (`No module named 'toto'`) |

To reproduce:

```bash
TOLLAMA_RUN_INTEGRATION_TESTS=1 TOLLAMA_TOTO_INTEGRATION_CPU=1 pytest -q -rs \
  tests/test_chronos_integration.py \
  tests/test_granite_integration.py \
  tests/test_timesfm_integration.py \
  tests/test_uni2ts_integration.py \
  tests/test_sundial_integration.py \
  tests/test_toto_integration.py
```

Per-family runtime isolation smoke re-run (`tollama runtime install --all` +
one forecast per family) on `2026-02-17` after the TimesFM pin update:

| Model | Result | Notes |
|---|---|---|
| `chronos2` | pass | `tollama run chronos2 --input examples/chronos2_request.json --no-stream` |
| `granite-ttm-r2` | pass | `tollama run granite-ttm-r2 --input examples/granite_ttm_request.json --no-stream` |
| `timesfm-2.5-200m` | pass | `tollama run timesfm-2.5-200m --input examples/timesfm_2p5_request.json --no-stream` |
| `moirai-2.0-R-small` | pass | `tollama run moirai-2.0-R-small --input examples/moirai_2p0_request.json --no-stream --timeout 600` |
| `sundial-base-128m` | pass | `tollama run sundial-base-128m --input examples/sundial_request.json --no-stream` |
| `toto-open-base-1.0` | pass | `tollama run toto-open-base-1.0 --input examples/toto_request.json --no-stream` |

`/api/info` confirmed all families used isolated runtime commands under
`~/.tollama/runtimes/<family>/venv/bin/python`.

## LangChain Support

Install optional LangChain support:

```bash
python -m pip install -e ".[langchain]"
```

Available wrappers (in `src/tollama/skill/langchain.py`):

- `TollamaForecastTool`
- `TollamaAnalyzeTool`
- `TollamaCompareTool`
- `TollamaRecommendTool`
- `TollamaHealthTool`
- `TollamaModelsTool`
- `get_tollama_tools(base_url="http://127.0.0.1:11435", timeout=10.0)`

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
- `tollama_forecast`: args `{"request": <ForecastRequest dict>}`
- `tollama_analyze`: args `{"request": <AnalyzeRequest dict>}`
- `tollama_compare`: args `{"request": <CompareRequest dict>}`
- `tollama_recommend`: args `{"horizon": int, ...}` for ranked model hints
- all wrappers return structured `dict` payloads; errors use
  `{"error":{"category","exit_code","message"}}`
- async invocations are fully implemented via `_arun` + `AsyncTollamaClient`
- missing optional dependency hint:
  `pip install "tollama[langchain]"`

LangChain wrapper validation command:

```bash
PYTHONPATH=src python -m pytest -q tests/test_langchain_skill.py
```

## Additional Agent Wrappers

`tollama` also provides wrapper factories for CrewAI, AutoGen, and smolagents.
All wrappers reuse the same tool contracts
(`health/models/forecast/analyze/compare/recommend`).

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

## Development Checks

```bash
ruff check .
pytest -q
```
