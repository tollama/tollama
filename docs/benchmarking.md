# Benchmarking Tooling

Tollama ships with built-in scripts to compare standard SDK latency or run comprehensive cross-model benchmark reporting.

## Compare SDK Ergonomics

Compare SDK ergonomics and time-to-first-forecast versus raw HTTP client calls:

```bash
PYTHONPATH=src python benchmarks/tollama_vs_raw.py --model mock --iterations 3 --warmup 1
```

## Cross-Model TSFM Benchmark

Run cross-model TSFM benchmark + routing default recommendation
(Lag-Llama, PatchTST, TiDE, N-HiTS, N-BEATSx):

```bash
# protocol/report template (no daemon required)
PYTHONPATH=src python benchmarks/cross_model_tsfm.py \
  --template-only \
  --output-dir benchmarks/reports/cross_model_baseline

# full benchmark run
PYTHONPATH=src python benchmarks/cross_model_tsfm.py \
  --base-url http://127.0.0.1:11435 \
  --output-dir artifacts/benchmarks/cross_model
```

See `docs/tsfm-routing-defaults.md` for benchmark protocol and routing-policy interpretations.
