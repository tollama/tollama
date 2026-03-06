# TSFM Cross-Model Benchmark Protocol & Routing Defaults

This document defines a reproducible benchmark protocol for five TSFM models and
how to derive routing defaults from benchmark outcomes.

## Scope

Compared models:

- `lag-llama`
- `patchtst`
- `tide`
- `nhits`
- `nbeatsx`

## Standard protocol

- **Datasets** (deterministic synthetic):
  - `seasonal_daily` (`freq=D`)
  - `trend_weekly` (`freq=W`)
  - `intermittent_daily` (`freq=D`)
- **Split**: fixed holdout = last `horizon` points, training context = preceding
  `context_length` points.
- **Default split params**:
  - `context_length=96`
  - `horizon=24`
- **Latency sampling**: repeated non-stream calls (`repeats=3` by default).
- **Metrics**:
  - Accuracy: `MAE`, `RMSE`, `MAPE`, `sMAPE`, `MASE`
  - Performance: latency `p50`, `mean`, `p95` (ms)
- **Reporting format**:
  - machine-readable: `result.json`
  - human-readable: `result.md`
  - environment-limited fallback: `report_template.json`, `report_template.md`

## Run commands

```bash
# 1) Template-only (no daemon needed)
PYTHONPATH=src python benchmarks/cross_model_tsfm.py \
  --template-only \
  --output-dir benchmarks/reports/cross_model_baseline

# 2) Full benchmark run (daemon required)
PYTHONPATH=src python benchmarks/cross_model_tsfm.py \
  --base-url http://127.0.0.1:11435 \
  --output-dir artifacts/benchmarks/cross_model
```

## Routing default policy

The benchmark derives a policy with three routing targets:

- **default**: balanced quality/latency objective
- **fast_path**: minimum latency objective
- **high_accuracy**: minimum error objective

Current scoring is intentionally simple and transparent:

- quality score = `sMAPE + 10 * MASE` (lower is better)
- latency score = `p50_ms / 1000`
- balanced ranking = `0.7 * quality_score + 0.3 * latency_score`

Use policy like this:

- standard requests → `default`
- interactive/low-latency requests → `fast_path`
- backtesting/critical forecasting requests → `high_accuracy`

## Caveats

- Synthetic data is reproducible and useful for regressions, but final routing
  defaults should be validated on production-like datasets.
- First-run latency can be inflated by model/runtime bootstrap and cache misses.
- If model dependencies are missing, use template artifacts and mark recommendation
  as provisional.

## Known blocker matrix (live benchmark runs)

If `result.md` shows `DEPENDENCY_MISSING`, install the corresponding optional extras
before retrying the full benchmark.

| model | blocker shown by daemon | install command |
|---|---|---|
| `lag-llama` | missing `gluonts`, `lag-llama`, `lag_llama` | `python -m pip install -e ".[dev,runner_lag_llama]"` |
| `patchtst` | missing `torch`, `transformers` | `python -m pip install -e ".[dev,runner_patchtst]"` |
| `tide` | missing `darts` | `python -m pip install -e ".[dev,runner_tide]"` |
| `nhits` | missing `neuralforecast` | `python -m pip install -e ".[dev,runner_nhits]"` |
| `nbeatsx` | missing `neuralforecast` | `python -m pip install -e ".[dev,runner_nbeatsx]"` |

### Runnable fallback command matrix

```bash
# 0) Start daemon
tollama serve

# 1) Environment-limited fallback (always works; no daemon/model execution)
PYTHONPATH=src python benchmarks/cross_model_tsfm.py \
  --template-only \
  --output-dir benchmarks/reports/cross_model_baseline

# 2) Live benchmark (best effort; writes result.json/result.md)
PYTHONPATH=src python benchmarks/cross_model_tsfm.py \
  --base-url http://127.0.0.1:11435 \
  --output-dir artifacts/benchmarks/cross_model

# 3) After installing missing extras, rerun quickly without pull preflight
PYTHONPATH=src python benchmarks/cross_model_tsfm.py \
  --skip-pull \
  --base-url http://127.0.0.1:11435 \
  --output-dir artifacts/benchmarks/cross_model
```

## Baseline artifact in repository

A baseline template artifact is committed at:

- `benchmarks/reports/cross_model_baseline/report_template.json`
- `benchmarks/reports/cross_model_baseline/report_template.md`

These artifacts document protocol and reporting schema even when benchmark
execution is unavailable in CI or local environments.
