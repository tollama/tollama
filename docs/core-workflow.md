# Core Workflow

This is the canonical Tollama Core path:

`preprocess -> forecast -> benchmark -> route`

Use this walkthrough when you want one runnable story that matches the Core-first
product direction.

## 1. Install the Core path

```bash
python -m pip install "tollama[eval,preprocess]"
```

## 2. Start the daemon

```bash
tollama serve
```

In a second terminal, verify the daemon and pull a demo model:

```bash
tollama quickstart
```

## 3. Preprocess irregular data

`preprocess` is local and does not require the daemon.

```python
import numpy as np
from tollama.preprocess import PreprocessConfig, run_pipeline

x = np.arange(48, dtype=float)
y = np.sin(x * 0.15) * 10
y[[7, 19, 33]] = np.nan

result = run_pipeline(x, y, config=PreprocessConfig(lookback=12, horizon=4))
print(result.X.shape, result.y.shape)
```

## 4. Run a forecast

```bash
tollama run mock --input examples/request.json --no-stream
```

Or from Python:

```python
from tollama import Tollama

sdk = Tollama()
forecast = sdk.forecast(
    model="mock",
    series={"target": [10, 11, 12, 13, 14], "freq": "D"},
    horizon=3,
)
print(forecast.to_df())
```

## 5. Benchmark and create Core artifacts

```bash
tollama benchmark examples/benchmark_data.json --models mock --horizon 4 --folds 1 --output artifacts/benchmarks/demo
```

That output directory contains:

- `result.json`
- `routing.json`
- `leaderboard.csv`
- `benchmark_<fingerprint>.json`

## 6. Apply routing evidence

Apply the generated artifact as the active local routing manifest:

```bash
tollama routing apply artifacts/benchmarks/demo/result.json
```

Inspect the applied policy any time:

```bash
tollama routing show
```

## 7. Route through `auto_forecast`

Once the routing manifest is present, use `mode` instead of hard-coding a model.

```python
from tollama import Tollama

sdk = Tollama()
response = sdk.auto_forecast(
    series={"target": [10, 11, 12, 13, 14], "freq": "D"},
    horizon=3,
    mode="high_accuracy",
)
print(response.selection.chosen_model)
print(response.response.forecasts[0].mean)
```

Recommended `mode` values:

- `default` for general workloads
- `fast_path` for latency-sensitive paths
- `high_accuracy` for accuracy-prioritized forecasts

## 8. What this demo proves

At the end of this flow, you have shown all four Core actions:

1. preprocess irregular series
2. forecast with one canonical interface
3. benchmark models and write reusable artifacts
4. route future requests from benchmark-backed evidence
