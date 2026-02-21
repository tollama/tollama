# CLI Cheat Sheet

## Install / Setup

```bash
python -m pip install tollama
# dev
python -m pip install -e ".[dev]"

# terminal 1
tollama serve

# terminal 2
tollama quickstart
```

Shell completion:
```bash
tollama --install-completion
tollama --show-completion
```

## Model Management

List models:
```bash
tollama list
tollama list --json
```

Show one model:
```bash
tollama show chronos2
```

Pull model:
```bash
tollama pull chronos2
tollama pull chronos2 --accept-license
tollama pull chronos2 --progress on
```

Delete model:
```bash
tollama rm chronos2
```

Loaded models:
```bash
tollama ps
tollama ps --json
```

## Forecasting

Run forecast from file:
```bash
tollama run chronos2 --input examples/chronos2_request.json --no-stream
```

Run forecast from stdin:
```bash
cat examples/request.json | tollama run mock --no-stream
```

Interactive model/payload selection:
```bash
tollama run --interactive --no-stream
```

Dry-run validation only:
```bash
tollama run mock --input examples/request.json --dry-run
```

Adjust timeout:
```bash
tollama run moirai-2.0-R-small --input examples/moirai_2p0_request.json --timeout 600 --no-stream
```

## Diagnostics

Health and diagnostics:
```bash
tollama doctor
tollama info
tollama info --json
```

Runtime environments:
```bash
tollama runtime list
tollama runtime install torch
tollama runtime install --all --progress on
tollama runtime remove torch
```

## Configuration

View all config:
```bash
tollama config list
tollama config keys
```

Set/unset values:
```bash
tollama config set pull.offline true
tollama config get pull.offline
tollama config unset pull.offline
```

Initialize config template:
```bash
tollama config init
tollama config init --force
```

## Common Flags

Progress control:
- `--progress auto|on|off` (`pull`, `run`, `quickstart`, `runtime install`)

Output mode:
- `--json` on list/info-style commands
- `--stream/--no-stream` on forecast/pull flows

Connection:
- `--base-url http://127.0.0.1:11435`
- `--timeout <seconds>`

## Troubleshooting

See canonical guide: `docs/troubleshooting.md`.
