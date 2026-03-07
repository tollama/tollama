# Cross-Model TSFM Benchmark Report

- run_id: `<template-run-id>`
- generated_at: `<template-generated-at>`
- base_url: `http://127.0.0.1:11435`
- models: lag-llama, patchtst, tide, nhits, nbeatsx
- datasets: seasonal_daily, trend_weekly, intermittent_daily
- split: context=96, horizon=24
- repeats per run: 3

## Per-run results

| model | dataset | status | error class | sMAPE | MASE | p50 latency (ms) | p95 latency (ms) | error |
|---|---|---|---|---:|---:|---:|---:|---|

## Routing recommendation

- default: `<to-be-filled>`
- fast_path: `<to-be-filled>`
- high_accuracy: `<to-be-filled>`
- policy: Populate after benchmark execution.

### Caveats
- Template-only run. No daemon/model execution performed.
- Fill with actual results after running with a live tollama daemon.
