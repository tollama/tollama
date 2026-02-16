# Covariates Contract

`tollama` accepts two dynamic covariate groups per input series:

- `past_covariates`: aligned to history (`len(target)`).
- `future_covariates`: aligned to forecast horizon (`prediction_length` / `horizon`).

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

## Covariates Mode

Use `parameters.covariates_mode`:

- `best_effort` (default): unsupported covariates are dropped, and response `warnings` explains what was ignored.
- `strict`: unsupported covariates fail early with HTTP 400.

Warnings may come from daemon-side compatibility filtering and runner-side adapter behavior;
the response merges both into `warnings[]`.

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
