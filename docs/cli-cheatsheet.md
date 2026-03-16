# CLI Cheat Sheet

## Install / Setup

Install tollama, start the daemon, then verify with a demo forecast:
```bash
python -m pip install tollama
# dev
python -m pip install -e ".[dev]"

# terminal 1 — start the daemon
tollama serve

# terminal 2 — verify with a demo forecast
tollama quickstart
```

Shell completion (bash/zsh/fish):
```bash
tollama --install-completion
tollama --show-completion
```

---

## Model Management

List installed models with name, family, size, and modification time:
```bash
tollama list
tollama list --json
```

Example output:
```
NAME             FAMILY   SIZE      MODIFIED
---------------  -------  --------  --------------------
mock             mock     -         2025-01-15T10:23:00Z
chronos2-small   torch    1.23 GB   2025-01-14T08:00:00Z
```

Show raw metadata for one installed model:
```bash
tollama show chronos2
```

Explain model capabilities, license, limits, and recommended use cases:
```bash
tollama explain chronos2
tollama explain chronos2 --json
```

Example output (`tollama explain mock`):
```
mock
family: mock
installed: yes
source: - @ -

license
  type: MIT
  acceptance required: no

limits
  max_horizon: -
  max_context: -

capabilities
  past_covariates_numeric: no
  ...

recommended use cases
  - Fast local smoke tests and integration checks.
```

Pull a model from the registry into local storage:
```bash
tollama pull chronos2
tollama pull chronos2 --accept-license
tollama pull chronos2 --progress on
```

Delete a model from local storage:
```bash
tollama rm chronos2
```

List currently loaded (in-memory) models:
```bash
tollama ps
tollama ps --json
```

---

## Forecasting

Run a forecast with a request file:
```bash
tollama run chronos2 --input examples/chronos2_request.json --no-stream
```

Run forecast from stdin:
```bash
cat examples/request.json | tollama run mock --no-stream
```

Interactive model and payload selection (prompts when omitted):
```bash
tollama run --interactive --no-stream
```

Dry-run — validate the request payload without running inference:
```bash
tollama run mock --input examples/request.json --dry-run
```

Extend timeout for large models or slow first-run loads:
```bash
tollama run moirai-2.0-R-small --input examples/moirai_2p0_request.json --timeout 600 --no-stream
```

---

## Dashboard

Open the bundled web dashboard in the default browser:
```bash
tollama open
```

Launch the Textual terminal dashboard (TUI):
```bash
tollama dashboard
```

---

## Diagnostics

Run health checks with pass/warn/fail summaries:
```bash
tollama doctor
tollama doctor --json
```

Show local and daemon diagnostics (Python version, installed models, runners, env):
```bash
tollama info
tollama info --json
tollama info --verbose       # includes raw model details
tollama info --local         # skip daemon call
tollama info --remote        # require daemon call
```

---

## Runtime Environments

Each model family runs in an isolated virtualenv. These commands manage those environments.

Show install status for all runner families:
```bash
tollama runtime list
tollama runtime list --json
```

Install (or update) a runner environment:
```bash
tollama runtime install torch
tollama runtime install --all --progress on
tollama runtime install uni2ts --reinstall
```

Update already-installed runtimes to match the current tollama version:
```bash
tollama runtime update torch
tollama runtime update --all
```

Remove a runner environment:
```bash
tollama runtime remove torch
tollama runtime remove --all
```

---

## TSModelfile Profiles

TSModelfiles are named preset profiles that bundle model, horizon, and quantiles into a reusable name.

List stored profiles:
```bash
tollama modelfile list
tollama modelfile list --json
```

Show one profile:
```bash
tollama modelfile show my-profile
```

Create or update a profile inline:
```bash
tollama modelfile create my-profile --model chronos2 --horizon 14 --quantiles 0.1,0.5,0.9
```

Create a profile from a YAML file:
```bash
tollama modelfile create my-profile --file profiles/my-profile.yaml
```

Delete a profile:
```bash
tollama modelfile rm my-profile
```

---

## Configuration

View all config values with descriptions and current settings:
```bash
tollama config list
tollama config keys
```

Read, write, or clear one config key:
```bash
tollama config set pull.offline true
tollama config get pull.offline
tollama config unset pull.offline
```

Write a default `~/.tollama/config.json` template:
```bash
tollama config init
tollama config init --force
```

---

## Benchmarking

Run all models against a dataset and compare accuracy/latency:
```bash
tollama benchmark examples/benchmark_data.json
tollama benchmark examples/benchmark_data.json --horizon 96 --models chronos2,mock --folds 3
tollama benchmark examples/benchmark_data.json --output results/
```

---

## Export & Quantize

Export a model to ONNX or TorchScript for edge deployment:
```bash
tollama export chronos2
tollama export chronos2 --format onnx --context-length 512 --prediction-length 96
tollama export chronos2 --format torchscript --output exports/
```

Quantize a pulled model for reduced memory usage:
```bash
tollama quantize chronos2
tollama quantize chronos2 --precision int8
tollama quantize chronos2 --precision int4
```

---

## Common Flags

Progress control:
- `--progress auto|on|off` (`pull`, `run`, `quickstart`, `runtime install`)

Output mode:
- `--json` on list/info-style commands
- `--stream/--no-stream` on forecast/pull flows

Connection:
- `--base-url http://127.0.0.1:11435`
- `--timeout <seconds>`

---

## Troubleshooting

See canonical guide: `docs/troubleshooting.md`.
