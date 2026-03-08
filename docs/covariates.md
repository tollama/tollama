# Covariates Contract

## Primer

Covariates are optional side-information you can pass alongside your target time series to help
the model make better forecasts. **Past covariates** are features you observed historically — they
have one value per timestamp in your target window (e.g. a promotion flag for each past day).
**Future covariates** are features whose values you already know for the forecast horizon — for
example, a holiday calendar or a scheduled promotion. Future covariates must also appear as past
covariates so the model sees the full sequence. **Static covariates** are time-invariant metadata
about the series itself (e.g. `{"store_id": "A42", "category": "electronics"}`); they have no
time axis. Not all models support all covariate types — check the compatibility matrix below or
run `tollama explain <model>` to see what a specific model accepts.

---

`tollama` accepts two dynamic covariate groups and one static covariate map per input series:

- `past_covariates`: aligned to history (`len(target)`).
- `future_covariates`: aligned to forecast horizon (`prediction_length` / `horizon`).
- `static_covariates`: key/value metadata per series (no time axis).

## Request Rules

For each `series[i]`:

- `past_covariates[name]` length must equal `len(target)`.
- `future_covariates[name]` length must equal `horizon`.
- `future_covariates[name]` keys must also exist in `past_covariates`.
- Each covariate array must be homogeneous:
  - all numeric (`int`/`float`), or
  - all string (categorical).
- Mixed numeric + string in one covariate array is rejected.
- Known-future covariates are keys present in both `past_covariates` and `future_covariates`.
- Past-only covariates are keys present only in `past_covariates`.
- Static covariates are normalized/filtered in daemon compatibility checks.
  TiDE advertises `static_covariates=true`; other runners do not yet support static covariates.

## Covariates Mode

Use `parameters.covariates_mode`:

- `best_effort` (default): unsupported covariates are dropped, and response `warnings` explains what was ignored.
- `strict`: unsupported covariates fail early with HTTP 400.

Warnings may come from daemon-side compatibility filtering and runner-side adapter behavior;
the response merges both into `warnings[]`.

For static covariates specifically:

- `best_effort`: static covariates are dropped with warnings for current runners.
- `strict`: static covariates are rejected with HTTP 400 for current runners.

## Family Mapping

### Chronos-2 (torch runner)

- History `df`: `id`, `timestamp`, `target`, plus all `past_covariates`.
- `future_df`: `id`, `timestamp`, plus only known-future covariates.
- Known-future columns appear in both frames; past-only appears only in history.

### Granite TSFM (torch runner)

- Known-future numeric covariates map to `control_columns`.
- Past-only numeric covariates map to `conditional_columns`.
- `future_time_series` contains future timestamps + known-future columns.
- Current implementation is numeric-only.

### TimesFM 2.5 (timesfm runner)

- No covariates: uses `forecast(...)`.
- With covariates: uses `forecast_with_covariates(...)`.
- Numeric dynamic covariates are converted to TimesFM format:
  - `dynamic_numerical_covariates[name] -> list[series_sequence]`
  - sequence format: `past + future`.
- Past-only numeric covariates are carried forward with the last observed value over horizon.
- Categorical covariates are ignored in `best_effort` with warnings; `strict` rejects.
- Covariate path compiles with `ForecastConfig(return_backcast=True, ...)`.

TimesFM knobs live under `parameters.timesfm`:

- `xreg_mode` (default: `"xreg + timesfm"`)
- `ridge` (default: `0.0`)
- `force_on_cpu` (default: `false`)

### Uni2TS / Moirai (uni2ts runner)

- Known-future numeric covariates map to `feat_dynamic_real`.
- Past-only numeric covariates map to `past_feat_dynamic_real`.
- Dataset includes horizon rows with `target=NaN` and known-future feature values.
- Categorical covariates are unsupported.

### Sundial (sundial runner)

- Sundial is target-only in this runner implementation.
- `past_covariates`, `future_covariates`, and `static_covariates` are unsupported.
- In `best_effort`, unsupported covariates are ignored and surfaced as warnings.
- In `strict`, covariate usage is rejected with HTTP 400.

### Toto (toto runner)

- Toto uses target + past numeric covariates as multivariate variates.
- `past_covariates` numeric values are supported and aligned to target timestamps.
- `future_covariates`, categorical covariates, and `static_covariates` are unsupported.
- In `best_effort`, unsupported covariates are ignored and surfaced as warnings.
- In `strict`, unsupported covariate usage is rejected with HTTP 400.

### Lag-Llama (lag_llama runner)

- Lag-Llama is target-only in this runner implementation.
- `past_covariates`, `future_covariates`, and `static_covariates` are unsupported.
- In `best_effort`, unsupported covariates are ignored and surfaced as warnings.
- In `strict`, covariate usage is rejected with HTTP 400.

### PatchTST (patchtst runner)

- PatchTST is target-only in this runner implementation.
- `past_covariates`, `future_covariates`, and `static_covariates` are unsupported.
- In `best_effort`, unsupported covariates are ignored and surfaced as warnings.
- In `strict`, covariate usage is rejected with HTTP 400.

### TiDE (tide runner)

- TiDE supports past numeric and future numeric covariates, as well as static covariates.
- Categorical covariates are unsupported.
- In `best_effort`, unsupported covariates (categorical) are ignored and surfaced as warnings.
- In `strict`, unsupported covariate usage is rejected with HTTP 400.

### N-HiTS (nhits runner)

- N-HiTS is target-only per registry capabilities.
- Numeric covariates/static features are handled in practical best-effort mode.
- In `best_effort`, non-numeric values are dropped/zero-filled with warnings.
- In `strict`, non-numeric values raise `BAD_REQUEST`.

### N-BEATSx (nbeatsx runner)

- N-BEATSx is target-only per registry capabilities.
- Numeric covariates/static features are handled in practical best-effort mode.
- In `best_effort`, non-numeric values are dropped/zero-filled with warnings.
- In `strict`, non-numeric values raise `BAD_REQUEST`.

## Capability Visibility

Covariate compatibility is exposed in:

- `GET /api/info` under `models.installed[].capabilities` and `models.available[].capabilities`
- `tollama info` human-readable covariates summaries

## Compatibility Matrix

| Model | Past Numeric | Past Categorical | Known-Future Numeric | Known-Future Categorical | Static |
|---|---|---|---|---|---|
| Chronos-2 | Yes | Yes | Yes | Yes | Planned |
| Granite TTM | Yes | No | Yes | No | Planned |
| TimesFM 2.5 | Yes | No | Yes | No | Planned |
| Uni2TS / Moirai | Yes | No | Yes | No | Planned |
| Sundial | No | No | No | No | Planned |
| Toto Open Base 1.0 | Yes | No | No | No | Planned |
| Lag-Llama | No | No | No | No | Planned |
| PatchTST | No | No | No | No | No |
| TiDE | Yes | No | Yes | No | Yes |
| N-HiTS | No | No | No | No | No |
| N-BEATSx | No | No | No | No | No |

## API Example (curl)

```bash
curl -s http://localhost:11435/api/forecast \
  -H 'content-type: application/json' \
  -d '{
    "model":"timesfm-2.5-200m",
    "horizon":2,
    "series":[
      {
        "id":"s1",
        "freq":"D",
        "timestamps":["2025-01-01","2025-01-02","2025-01-03"],
        "target":[10.0,11.0,12.0],
        "past_covariates":{"promo":[0.0,1.0,0.0]},
        "future_covariates":{"promo":[1.0,1.0]}
      }
    ],
    "parameters":{"covariates_mode":"best_effort"},
    "options":{}
  }'
```

## Common Mistakes

**1. Future covariate key not in `past_covariates`**

A key in `future_covariates` must also appear in `past_covariates`. The daemon rejects requests
where the past sequence is missing.

```json
// Wrong — "promo" in future_covariates but missing from past_covariates
{
  "past_covariates": {},
  "future_covariates": {"promo": [1.0, 1.0]}
}

// Correct
{
  "past_covariates": {"promo": [0.0, 1.0, 0.0]},
  "future_covariates": {"promo": [1.0, 1.0]}
}
```

**2. Mixed types in one covariate array**

Each covariate array must be uniformly numeric (`int`/`float`) or uniformly string (categorical).
Mixing causes a 400 validation error.

```json
// Wrong — mixing int and string in one array
{"temperature": [20, "cold", 18]}

// Correct — numeric only
{"temperature": [20.0, 19.5, 18.0]}
```

**3. Passing covariates to Sundial (target-only model)**

Sundial does not use any covariates. In `best_effort` mode they are silently dropped (you will
see a `warnings[]` entry). In `strict` mode the request is rejected with HTTP 400. Use a model
that supports covariates, or omit the covariate fields entirely for Sundial.

**4. Categorical covariates on numeric-only families**

Granite TTM, TimesFM, Uni2TS, and Toto only accept **numeric** covariates. Passing string
arrays (categorical) for these families generates a warning in `best_effort` mode or fails in
`strict` mode. Chronos-2 is currently the only family that accepts categorical covariates.

---

## API Example (Python)

```python
import httpx

payload = {
    "model": "chronos2",
    "horizon": 3,
    "series": [
        {
            "id": "daily-sales",
            "freq": "D",
            "timestamps": ["2025-01-01", "2025-01-02", "2025-01-03"],
            "target": [100.0, 101.0, 103.0],
            "past_covariates": {
                "promo": [0.0, 1.0, 0.0],
                "region": ["us", "us", "us"],
            },
            "future_covariates": {
                "promo": [1.0, 1.0, 0.0],
                "region": ["us", "us", "us"],
            },
        }
    ],
    "parameters": {"covariates_mode": "best_effort"},
    "options": {},
}

with httpx.Client(base_url="http://localhost:11435", timeout=30.0) as client:
    res = client.post("/api/forecast", json=payload)
    res.raise_for_status()
    print(res.json())
```
