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

## Real-Data Benchmark Reports

The real-data E2E harness also emits benchmark-focused reports alongside the
gate summary:

- `benchmark_report.json`
- `benchmark_report.md`

For the curated HuggingFace starter lane:

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

These reports include benchmark rows, model leaderboards, dataset breakdowns,
failure classifications, and separate contract summaries.
