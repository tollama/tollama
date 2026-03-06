# Cross-Model TSFM Benchmark Report

- run_id: `20260306T163304Z`
- generated_at: `2026-03-06T16:33:10+00:00`
- base_url: `http://127.0.0.1:11435`
- models: lag-llama, patchtst, tide, nhits, nbeatsx
- datasets: seasonal_daily, trend_weekly, intermittent_daily
- split: context=96, horizon=24
- repeats per run: 3

## Per-run results

| model | dataset | status | sMAPE | MASE | p50 latency (ms) | p95 latency (ms) | error |
|---|---|---|---:|---:|---:|---:|---|
| lag-llama | seasonal_daily | fail | - | - | - | - | forecast with model 'lag-llama' failed with HTTP 503: missing optional lag-llama runner dependencies (gluonts, lag-llama, lag_llama); install them with `pip install -e ".[dev,runner_lag_llama]"` (code=DEPENDENCY_MISSING) |
| lag-llama | trend_weekly | fail | - | - | - | - | forecast with model 'lag-llama' failed with HTTP 503: missing optional lag-llama runner dependencies (gluonts, lag-llama, lag_llama); install them with `pip install -e ".[dev,runner_lag_llama]"` (code=DEPENDENCY_MISSING) |
| lag-llama | intermittent_daily | fail | - | - | - | - | forecast with model 'lag-llama' failed with HTTP 503: missing optional lag-llama runner dependencies (gluonts, lag-llama, lag_llama); install them with `pip install -e ".[dev,runner_lag_llama]"` (code=DEPENDENCY_MISSING) |
| patchtst | seasonal_daily | fail | - | - | - | - | forecast with model 'patchtst' failed with HTTP 503: missing optional patchtst runner dependencies (torch, transformers); install them with `pip install -e ".[dev,runner_patchtst]"` (code=DEPENDENCY_MISSING) |
| patchtst | trend_weekly | fail | - | - | - | - | forecast with model 'patchtst' failed with HTTP 503: missing optional patchtst runner dependencies (torch, transformers); install them with `pip install -e ".[dev,runner_patchtst]"` (code=DEPENDENCY_MISSING) |
| patchtst | intermittent_daily | fail | - | - | - | - | forecast with model 'patchtst' failed with HTTP 503: missing optional patchtst runner dependencies (torch, transformers); install them with `pip install -e ".[dev,runner_patchtst]"` (code=DEPENDENCY_MISSING) |
| tide | seasonal_daily | fail | - | - | - | - | forecast with model 'tide' failed with HTTP 503: missing optional tide runner dependencies (darts); install them with `pip install -e ".[dev,runner_tide]"` (code=DEPENDENCY_MISSING) |
| tide | trend_weekly | fail | - | - | - | - | forecast with model 'tide' failed with HTTP 503: missing optional tide runner dependencies (darts); install them with `pip install -e ".[dev,runner_tide]"` (code=DEPENDENCY_MISSING) |
| tide | intermittent_daily | fail | - | - | - | - | forecast with model 'tide' failed with HTTP 503: missing optional tide runner dependencies (darts); install them with `pip install -e ".[dev,runner_tide]"` (code=DEPENDENCY_MISSING) |
| nhits | seasonal_daily | fail | - | - | - | - | forecast with model 'nhits' failed with HTTP 503: missing optional nhits runner dependencies (neuralforecast); install them with `pip install -e ".[dev,runner_nhits]"` (code=DEPENDENCY_MISSING) |
| nhits | trend_weekly | fail | - | - | - | - | forecast with model 'nhits' failed with HTTP 503: missing optional nhits runner dependencies (neuralforecast); install them with `pip install -e ".[dev,runner_nhits]"` (code=DEPENDENCY_MISSING) |
| nhits | intermittent_daily | fail | - | - | - | - | forecast with model 'nhits' failed with HTTP 503: missing optional nhits runner dependencies (neuralforecast); install them with `pip install -e ".[dev,runner_nhits]"` (code=DEPENDENCY_MISSING) |
| nbeatsx | seasonal_daily | fail | - | - | - | - | forecast with model 'nbeatsx' failed with HTTP 503: missing optional nbeatsx runner dependencies (neuralforecast); install them with `pip install -e ".[dev,runner_nbeatsx]"` (code=DEPENDENCY_MISSING) |
| nbeatsx | trend_weekly | fail | - | - | - | - | forecast with model 'nbeatsx' failed with HTTP 503: missing optional nbeatsx runner dependencies (neuralforecast); install them with `pip install -e ".[dev,runner_nbeatsx]"` (code=DEPENDENCY_MISSING) |
| nbeatsx | intermittent_daily | fail | - | - | - | - | forecast with model 'nbeatsx' failed with HTTP 503: missing optional nbeatsx runner dependencies (neuralforecast); install them with `pip install -e ".[dev,runner_nbeatsx]"` (code=DEPENDENCY_MISSING) |

## Routing recommendation

- default: `None`
- fast_path: `None`
- high_accuracy: `None`
- policy: no successful benchmark runs

### Caveats
- all model runs failed; keep existing routing unchanged
