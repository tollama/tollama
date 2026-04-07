# Operator Summary

- Default lane: `chronos2`
  - MASE=0.8400
  - latency=220.0ms
  - Best balanced benchmark profile for general workloads.
- Fast path: `timesfm-2.5-200m`
  - MASE=0.9500
  - latency=72.0ms
  - Lowest observed latency among successful benchmark runs.
- High accuracy: `chronos2`
  - MASE=0.8400
  - latency=220.0ms
  - Best MASE among successful benchmark runs.

## Policy

Use default for general workloads; route latency-sensitive requests to fast_path; route accuracy-critical requests to high_accuracy.

## Caveats

- Routing recommendation is only as good as the benchmark dataset and fold design.
- Latency can shift with hardware, daemon warm-up, and runner cache state.
