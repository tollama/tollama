# Model Guides

This page is the canonical human-facing model and family guide. The
machine-readable source of truth is `model-registry/registry.yaml`, and overview
docs should link here instead of repeating model counts or long support tables.

`GET /v1/models` exposes the curated registry inventory for clients, including
public source, metadata, capability, and license fields used by the macOS native
model browser and forecast controls.

## Registry-Backed Model Inventory

| Model | Family | Source | Covariates |
|---|---|---|---|
| `mock` | `mock` | `tollama/mock-runner` | Target only |
| `chronos2` | `torch` | `amazon/chronos-2` | Past + Future |
| `granite-ttm-r2` | `torch` | `ibm-granite/granite-timeseries-ttm-r2` | Past + Future |
| `timesfm-2.5-200m` | `timesfm` | `google/timesfm-2.5-200m-pytorch` | Past + Future |
| `moirai-2.0-R-small` | `uni2ts` | `Salesforce/moirai-2.0-R-small` | Past + Future |
| `sundial-base-128m` | `sundial` | `thuml/sundial-base-128m` | Target only |
| `toto-open-base-1.0` | `toto` | `Datadog/Toto-Open-Base-1.0` | Past only |
| `lag-llama` | `lag_llama` | `time-series-foundation-models/Lag-Llama` | Target only |
| `patchtst` | `patchtst` | `ibm-granite/granite-timeseries-patchtst` | Target only |
| `tide` | `tide` | `tollama/tide-runner` | Target only |
| `nhits` | `nhits` | `tollama/nhits-runner` | Past + Future + Static |
| `nbeatsx` | `nbeatsx` | `tollama/nbeatsx-runner` | Past + Future + Static |
| `timer-base` | `timer` | `thuml/Timer` | Target only |
| `timemixer-base` | `timemixer` | `thuml/timemixer` | Target only |
| `forecastpfn` | `forecastpfn` | `abacusai/ForecastPFN` | Target only |

## Chronos2 Forecasting

```bash
# install optional torch runner dependencies
python -m pip install -e ".[dev,runner_torch]"

# run daemon
tollama serve

# pull Chronos snapshot + manifest metadata
tollama pull chronos2

# run forecast through CLI
tollama run chronos2 --input examples/chronos2_request.json --no-stream
```

```bash
curl -s http://localhost:11435/api/forecast \
  -H 'content-type: application/json' \
  -d @examples/chronos2_request.json
```

## Covariates (Past + Known-Future)

Unified covariates contract:

- `past_covariates[name]` length must equal `len(target)`
- `future_covariates[name]` length must equal `horizon`
- every `future_covariates[name]` key must also exist in `past_covariates`
- each covariate array must be all-numeric or all-string (no mixed arrays)
- known-future covariates are keys present in both past and future maps
- set `parameters.covariates_mode` to `best_effort` (default) or `strict`
- in `best_effort`, unsupported covariates are ignored with response `warnings[]`
- in `strict`, unsupported covariates return HTTP `400`
- `series[].freq` defaults to `"auto"` when omitted and is inferred from timestamps

```json
{
  "model": "timesfm-2.5-200m",
  "horizon": 2,
  "series": [
    {
      "id": "s1",
      "freq": "D",
      "timestamps": ["2025-01-01", "2025-01-02", "2025-01-03"],
      "target": [10.0, 11.0, 12.0],
      "past_covariates": {"promo": [0.0, 1.0, 0.0]},
      "future_covariates": {"promo": [1.0, 1.0]}
    }
  ],
  "parameters": {"covariates_mode": "best_effort"},
  "options": {}
}
```

## Forecast Accuracy Metrics (MAPE, MASE, MAE, RMSE, SMAPE, WAPE, RMSSE, Pinball)

`/api/forecast` and `/v1/forecast` can optionally calculate forecast accuracy
metrics against provided actuals:

- set `series[].actuals` (length must equal `horizon`)
- set `parameters.metrics.names` to any of
  `["mape", "mase", "mae", "rmse", "smape", "wape", "rmsse", "pinball"]`
- optional `parameters.metrics.mase_seasonality` (default `1`)
- `metrics.aggregate` is a macro average across series where that metric is defined
- undefined cases are best-effort with response `warnings[]`:
  - MAPE skips when all actual values are `0`
  - MASE skips when `len(target) <= mase_seasonality` or naive denominator is `0`
  - SMAPE skips when all `|actual|+|prediction|` denominators are `0`
  - WAPE skips when `sum(|actual|) == 0`
  - RMSSE skips when `len(target) <= mase_seasonality` or squared naive denominator is `0`
  - Pinball skips when quantile forecasts are missing or malformed

### Metric Ownership

- Shared primitive formulas for `mae`, `mase`, `smape`, and `rmsse` follow
  `ts_autopilot.evaluation.metrics` in `tollama-eval`.
- `tollama` owns request-time adaptation: overlap trimming, warning-driven
  best-effort behavior, and response aggregation.
- `mape`, `rmse`, `wape`, and `pinball` are currently Core-side runtime metrics
  defined in `tollama.core.forecast_metrics`.

```json
{
  "model": "mock",
  "horizon": 2,
  "quantiles": [],
  "series": [
    {
      "id": "s1",
      "freq": "D",
      "timestamps": ["2025-01-01", "2025-01-02"],
      "target": [1.0, 3.0],
      "actuals": [2.0, 4.0]
    }
  ],
  "parameters": {
    "metrics": {
      "names": ["mape", "mase", "mae", "rmse", "smape"],
      "mase_seasonality": 1
    }
  },
  "options": {}
}
```

Example response:

```json
{
  "model": "mock",
  "forecasts": [
    {
      "id": "s1",
      "freq": "D",
      "start_timestamp": "2025-01-02",
      "mean": [3.0, 3.0]
    }
  ],
  "metrics": {
    "aggregate": {
      "mape": 37.5,
      "mase": 0.5,
      "mae": 1.0,
      "rmse": 1.0,
      "smape": 34.2857142857
    },
    "series": [
      {
        "id": "s1",
        "values": {
          "mape": 37.5,
          "mase": 0.5,
          "mae": 1.0,
          "rmse": 1.0,
          "smape": 34.2857142857
        }
      }
    ]
  }
}
```

See `docs/covariates.md` for the full compatibility matrix and model-family mappings.

Compatibility snapshot:

| Family | Past Numeric | Past Categorical | Known-Future Numeric | Known-Future Categorical | Static |
|---|---|---|---|---|---|
| Chronos-2 | Yes | Yes | Yes | Yes | No |
| Granite TTM | Yes | No | Yes | No | No |
| TimesFM 2.5 | Yes | No | Yes | No | No |
| Uni2TS / Moirai | Yes | No | Yes | No | No |
| Sundial | No | No | No | No | No |
| Toto Open Base 1.0 | Yes | No | No | No | No |
| Lag-Llama | No | No | No | No | No |
| PatchTST | No | No | No | No | No |
| TiDE | No | No | No | No | No |
| N-HiTS | Yes | No | Yes | No | Yes |
| N-BEATSx | Yes | No | Yes | No | Yes |
| Timer | No | No | No | No | No |
| TimeMixer | No | No | No | No | No |
| ForecastPFN | No | No | No | No | No |

TimesFM XReg knobs are available at `parameters.timesfm`:
`xreg_mode`, `ridge`, `force_on_cpu`.

`/api/pull` performs a real Hugging Face snapshot pull for registry models with
`source.type=huggingface`, streams NDJSON progress by default, and writes
resolved pull metadata into the local manifest:

- `resolved.commit_sha`
- `resolved.snapshot_path`
- `size_bytes`
- `pulled_at`

## Pull Examples

```bash
# basic pull
tollama pull chronos2

# proxy override
tollama pull chronos2 --https-proxy http://proxy:3128

# offline after first pull
tollama pull chronos2
tollama pull chronos2 --offline

# override Hugging Face cache home
tollama pull chronos2 --hf-home /mnt/fastcache/hf

# token via environment
export TOLLAMA_HF_TOKEN=hf_xxx
tollama pull <private-model>
```

## Granite TTM Forecasting (torch runner)

`runner_torch` includes optional Chronos and Granite TTM dependencies.

```bash
# install optional torch runner dependencies
python -m pip install -e ".[dev,runner_torch]"

# run daemon (default http://127.0.0.1:11435)
tollama serve

# pull one Granite TTM revision pinned in the registry
tollama pull granite-ttm-r2

# run forecast
tollama run granite-ttm-r2 --input examples/granite_ttm_request.json --no-stream
```

```bash
curl -s http://localhost:11435/api/forecast \
  -H 'content-type: application/json' \
  -d @examples/granite_ttm_request.json
```

## TimesFM 2.5 Forecasting (separate timesfm runner family)

```bash
# install optional TimesFM runner dependencies
python -m pip install -e ".[dev,runner_timesfm]"
# note: for GPU builds, install a compatible torch build first if needed

# run daemon (default http://127.0.0.1:11435)
tollama serve

# pull TimesFM 2.5 model snapshot from Hugging Face
tollama pull timesfm-2.5-200m

# run forecast
tollama run timesfm-2.5-200m --input examples/timesfm_2p5_request.json --no-stream
```

```bash
curl -s http://localhost:11435/api/forecast \
  -H 'content-type: application/json' \
  -d @examples/timesfm_2p5_request.json
```

In `parameters.covariates_mode="best_effort"`, runtime XReg/covariate-path
failures degrade to a target-only TimesFM forecast with a warning instead of
failing the request.

## Uni2TS/Moirai Forecasting (separate uni2ts runner family)

```bash
# install optional Uni2TS runner dependencies
python -m pip install -e ".[dev,runner_uni2ts]"
# note: install a compatible torch build first for your platform when needed
# note: Python 3.12+ may fail to build Uni2TS dependencies; prefer Python 3.11

# run daemon (default http://127.0.0.1:11435)
tollama serve

# Moirai model requires explicit license acceptance
tollama pull moirai-2.0-R-small --accept-license

# run forecast
tollama run moirai-2.0-R-small --input examples/moirai_2p0_request.json --no-stream
# if you hit timeout on first run, increase CLI timeout
tollama run moirai-2.0-R-small --input examples/moirai_2p0_request.json --no-stream --timeout 120
```

```bash
curl -s http://localhost:11435/api/forecast \
  -H 'content-type: application/json' \
  -d @examples/moirai_2p0_request.json
```

In `parameters.covariates_mode="best_effort"`, runtime dynamic-covariate
failures degrade to a target-only Moirai forecast with a warning instead of
failing the request.

## Sundial Forecasting (separate sundial runner family)

Sundial (`thuml/sundial-base-128m`) is frequency-agnostic and target-only in this
runner. Covariates are ignored in `best_effort` mode and rejected in `strict`
mode by daemon compatibility checks.

```bash
# install optional Sundial runner dependencies
python -m pip install -e ".[dev,runner_sundial]"
# note: install a compatible torch build first for your platform when needed

# run daemon (default http://127.0.0.1:11435)
tollama serve

# pull Sundial model snapshot from Hugging Face
tollama pull sundial-base-128m

# run forecast
tollama run sundial-base-128m --input examples/sundial_request.json --no-stream
```

If you upgraded dependencies or switched Python environments, restart `tollama serve`
before re-running Sundial so the daemon does not reuse stale runner subprocesses.

```bash
curl -s http://localhost:11435/api/forecast \
  -H 'content-type: application/json' \
  -d @examples/sundial_request.json
```

## Toto Forecasting (separate toto runner family)

Toto (`Datadog/Toto-Open-Base-1.0`) supports target + past numeric covariates in
this runner. Known-future, categorical, and static covariates are unsupported in
the first Toto integration.

```bash
# install optional Toto runner dependencies
python -m pip install -e ".[dev,runner_toto]"
# note: install a compatible torch build first for your platform when needed

# run daemon (default http://127.0.0.1:11435)
tollama serve

# pull Toto model snapshot from Hugging Face
tollama pull toto-open-base-1.0

# run forecast
tollama run toto-open-base-1.0 --input examples/toto_request.json --no-stream
```

```bash
curl -s http://localhost:11435/api/forecast \
  -H 'content-type: application/json' \
  -d @examples/toto_request.json
```

## Lag-Llama Forecasting (separate lag_llama runner family)

Lag-Llama is integrated as a target-only runner in Tollama.

Runtime notes:

- install optional runner dependencies: `runner_lag_llama`
- `lag-llama` is installed from GitHub (git must be available in your environment)
- first run downloads checkpoint artifacts from Hugging Face during `tollama pull`
- covariates/static features are unsupported (ignored in `best_effort`, rejected in `strict`)

```bash
# install optional Lag-Llama runner dependencies
python -m pip install -e ".[dev,runner_lag_llama]"
# note: install a compatible torch build first for your platform when needed

# run daemon (default http://127.0.0.1:11435)
tollama serve

# pull Lag-Llama snapshot + checkpoint metadata
tollama pull lag-llama

# run forecast
tollama run lag-llama --input examples/lag_llama_request.json --no-stream
```

```bash
curl -s http://localhost:11435/api/forecast \
  -H 'content-type: application/json' \
  -d @examples/lag_llama_request.json
```


## PatchTST Forecasting (Phase-2 baseline)

PatchTST is integrated for **real forecast execution** via the dedicated `patchtst` runner family.

- model name: `patchtst`
- runner family: `patchtst`
- install extra: `runner_patchtst`
- current runtime behavior:
  - returns `DEPENDENCY_MISSING` when optional dependencies are absent
  - caches loaded PatchTST model instances in-process (LRU) to reduce repeat-request latency
  - supports canonical point forecasts for single/multi-series requests
  - supports quantiles when exposed by the runtime backend
  - currently ignores covariates/static features (target-only history)

Runtime tuning knobs (PatchTST):

- request `options.context_length` — history window length used for inference
- request `options.cache_reuse` (default: `true`) — set `false` for one-off/no-reuse loads
- env `TOLLAMA_PATCHTST_CACHE_MAX_MODELS` (default: `4`) — max distinct cached model entries
- env `TOLLAMA_PATCHTST_DISABLE_CACHE` (default: `false`) — disable shared in-process cache
- env `TOLLAMA_PATCHTST_MAX_SERIES_PER_REQUEST` (default: `64`) — guardrail for batch size
- env `TOLLAMA_PATCHTST_MAX_CONTEXT_LENGTH` (default: `4096`) — upper bound for context length

Timeout note:

- request-level timeout is controlled by API/CLI `timeout` (runner supervisor watchdog).
- lower timeout improves stuck-runner recovery but may cut off large cold-start runs.

```bash
# install PatchTST runner dependencies
python -m pip install -e ".[dev,runner_patchtst]"

# pull registry entry
tollama pull patchtst

# run forecast
tollama run patchtst --input examples/request.json --no-stream
```

## TiDE Forecasting (Phase-3 probabilistic)

TiDE is integrated for inference via the dedicated `tide` runner family.

- model name: `tide`
- runner family: `tide`
- install extra: `runner_tide`
- pull behavior: registry pull is manifest-only (local source), so `tollama pull tide` does not require Hugging Face auth/snapshot download
- current runtime behavior:
  - returns `DEPENDENCY_MISSING` when optional dependencies are absent
  - returns deterministic mean forecasts for valid requests
  - attempts to produce requested quantiles using probabilistic sampling
  - explicitly falls back to mean-only responses (with warning) when quantiles are unavailable in the active runtime/backend
  - treats requests as target-only history; `past_covariates`, `future_covariates`, and `static_covariates` are unsupported in the current adapter path

Runtime tuning knobs (TiDE):

- request `options.quantile_samples` (default: `200`) — number of probabilistic samples used for quantile estimation when quantiles are requested

Calibration & limitations guidance:

- quantile quality depends on runtime sampling behavior and underlying TiDE checkpoint calibration; treat intervals as best-effort uncertainty estimates, not guaranteed calibrated prediction intervals.
- if you need tighter/steadier quantile estimates, increase `options.quantile_samples` (at the cost of latency/compute).
- if runtime/model combination does not expose quantile extraction, response warnings will state that quantiles were omitted and mean-only output was returned.

## N-HiTS Forecasting (Phase-4 quality)

N-HiTS is integrated for real inference via the dedicated `nhits` runner family.

- model name: `nhits`
- runner family: `nhits`
- install extra: `runner_nhits`
- pull behavior: registry pull is manifest-only (local source), so `tollama pull nhits` does not require Hugging Face auth/snapshot download
- current runtime behavior:
  - returns `DEPENDENCY_MISSING` with install guidance when optional dependencies are absent
  - performs runtime NeuralForecast inference for canonical single/multi-series forecast requests
  - validates edge cases more strictly (finite numeric targets, timestamp parsing, and per-series frequency sanity)
  - uses backend quantile outputs when available; otherwise applies calibrated residual-based quantile fallback with explicit warnings
  - supports numeric past/future/static covariates in the current runner path (with strict-mode validation via `parameters.covariates_mode`)

Runtime tuning & limitations (N-HiTS):

- requests with multiple series must currently use one shared resolved frequency
- `series[].freq: "auto"` is accepted, but frequency inference can fail for irregular timestamps (provide explicit `freq` when possible)
- if backend quantile columns are unavailable, fallback quantiles are generated from robust residual scale; warnings will identify this path
- covariates/static features are numeric-only for this runner path; in `best_effort` mode non-numeric values are dropped/zero-filled with warnings, and in `strict` mode they raise `BAD_REQUEST`
- `model_local_dir` is currently metadata-only for this runner path (runtime trains from request history) and is ignored with warning when present

```bash
# install N-HiTS runner dependencies
python -m pip install -e ".[dev,runner_nhits]"

# pull registry entry
tollama pull nhits

# run forecast
tollama run nhits --input examples/request.json --no-stream
```


## N-BEATSx Forecasting (Phase-4 quality)

N-BEATSx is integrated for real inference via the dedicated `nbeatsx` runner family.

- model name: `nbeatsx`
- runner family: `nbeatsx`
- install extra: `runner_nbeatsx`
- pull behavior: registry pull is manifest-only (local source), so `tollama pull nbeatsx` does not require Hugging Face auth/snapshot download
- current runtime behavior:
  - returns `DEPENDENCY_MISSING` with install guidance when optional dependencies are absent
  - performs runtime NeuralForecast inference for canonical single/multi-series forecast requests
  - validates edge cases more strictly (finite numeric targets, timestamp parsing, and per-series frequency sanity)
  - uses backend quantile outputs when available; otherwise applies calibrated residual-based quantile fallback with explicit warnings
  - supports numeric past/future/static covariates in the current runner path (with strict-mode validation via `parameters.covariates_mode`)

Runtime tuning & limitations (N-BEATSx):

- requests with multiple series must currently use one shared resolved frequency
- `series[].freq: "auto"` is accepted, but frequency inference can fail for irregular timestamps (provide explicit `freq` when possible)
- if backend quantile columns are unavailable, fallback quantiles are generated from robust residual scale; warnings will identify this path
- covariates/static features are numeric-only for this runner path; in `best_effort` mode non-numeric values are dropped/zero-filled with warnings, and in `strict` mode they raise `BAD_REQUEST`
- `model_local_dir` is currently metadata-only for this runner path (runtime trains from request history) and is ignored with warning when present

```bash
# install N-BEATSx runner dependencies
python -m pip install -e ".[dev,runner_nbeatsx]"

# pull registry entry
tollama pull nbeatsx

# run forecast
tollama run nbeatsx --input examples/request.json --no-stream
```

## Timer Forecasting

Timer is integrated for inference via the dedicated `timer` runner family.

- model name: `timer-base`
- runner family: `timer`
- install extra: `runner_timer`
- pull behavior: pulls a Hugging Face snapshot from `thuml/timer-base-84m`
- current runner behavior:
  - returns `DEPENDENCY_MISSING` when optional dependencies are absent
  - performs target-only forecasting in the current adapter path
  - truncates long histories to the declared `max_context`
  - returns canonical mean forecasts; quantiles are currently omitted

```bash
# install Timer runner dependencies
python -m pip install -e ".[dev,runner_timer]"

# pull model snapshot
tollama pull timer-base

# run forecast
tollama run timer-base --input examples/request.json --no-stream
```

## TimeMixer Forecasting

TimeMixer is integrated for inference via the dedicated `timemixer` runner family.

- model name: `timemixer-base`
- runner family: `timemixer`
- install extra: `runner_timemixer`
- pull behavior: pulls a Hugging Face snapshot from `thuml/timemixer`
- current runner behavior:
  - returns `DEPENDENCY_MISSING` when optional dependencies are absent
  - runs the current target-only adapter path
  - truncates long histories to the declared `max_context`
  - returns canonical mean forecasts; quantiles are currently omitted

```bash
# install TimeMixer runner dependencies
python -m pip install -e ".[dev,runner_timemixer]"

# pull model snapshot
tollama pull timemixer-base

# run forecast
tollama run timemixer-base --input examples/request.json --no-stream
```

## ForecastPFN Forecasting

ForecastPFN is integrated for inference via the dedicated `forecastpfn` runner family.

- model name: `forecastpfn`
- runner family: `forecastpfn`
- install extra: `runner_forecastpfn`
- pull behavior: pulls a Hugging Face snapshot from `abacusai/ForecastPFN`
- current runner behavior:
  - returns `DEPENDENCY_MISSING` when optional dependencies are absent
  - runs the current target-only adapter path
  - truncates long histories to the declared `max_context`
  - returns canonical mean forecasts; quantiles are currently omitted

```bash
# install ForecastPFN runner dependencies
python -m pip install -e ".[dev,runner_forecastpfn]"

# pull model snapshot
tollama pull forecastpfn

# run forecast
tollama run forecastpfn --input examples/request.json --no-stream
```
