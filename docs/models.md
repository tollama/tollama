# Model Guides

This page collects model-family setup and forecasting examples that were previously
in `README.md`.

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

## Forecast Accuracy Metrics (MAPE, MASE, MAE, RMSE, SMAPE)

`/api/forecast` and `/v1/forecast` can optionally calculate forecast accuracy
metrics against provided actuals:

- set `series[].actuals` (length must equal `horizon`)
- set `parameters.metrics.names` to any of
  `["mape", "mase", "mae", "rmse", "smape"]`
- optional `parameters.metrics.mase_seasonality` (default `1`)
- undefined cases are best-effort with response `warnings[]`:
  - MAPE skips when all actual values are `0`
  - MASE skips when `len(target) <= mase_seasonality` or naive denominator is `0`
  - SMAPE skips when all `|actual|+|prediction|` denominators are `0`

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

| Family | Past Numeric | Past Categorical | Known-Future Numeric | Known-Future Categorical |
|---|---|---|---|---|
| Chronos-2 | Yes | Yes | Yes | Yes |
| Granite TTM | Yes | No | Yes | No |
| TimesFM 2.5 | Yes | No | Yes | No |
| Uni2TS / Moirai | Yes | No | Yes | No |
| Sundial | No | No | No | No |
| Toto Open Base 1.0 | Yes | No | No | No |

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
