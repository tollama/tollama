# API Reference

Tollama daemon exposes Ollama-style and v1-style HTTP endpoints.

Base URL (default): `http://127.0.0.1:11435`

## Interactive API Docs

- Swagger UI: `/docs`
- ReDoc: `/redoc`
- OpenAPI JSON: `/openapi.json`

Auth behavior:
- If `auth.api_keys` is configured, these docs endpoints require `Authorization: Bearer <key>`.
- Override for local dev only: `TOLLAMA_DOCS_PUBLIC=1`.

## Auth

When API keys are configured in `~/.tollama/config.json`, include:

```http
Authorization: Bearer <api-key>
```

## Endpoint Inventory

### System

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/version` | Daemon version |
| GET | `/api/info` | Full diagnostics payload |
| GET | `/v1/health` | Health/liveness probe |

### Runtime / Observability

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/usage` | Per-key usage snapshot |
| GET | `/api/events` | SSE event stream |
| GET | `/metrics` | Prometheus metrics (optional dependency) |

### Modelfiles

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/modelfiles` | List TSModelfile profiles |
| GET | `/api/modelfiles/{name}` | Show one profile |
| POST | `/api/modelfiles` | Create/update profile |
| DELETE | `/api/modelfiles/{name}` | Delete profile |

### Ingest + Upload

| Method | Path | Purpose |
|---|---|---|
| POST | `/api/ingest/upload` | Upload CSV/parquet and normalize into series |
| POST | `/api/forecast/upload` | Upload CSV/parquet and forecast in one call |

### Model Lifecycle

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/tags` | List installed models (Ollama format) |
| POST | `/api/show` | Show one model metadata |
| POST | `/api/pull` | Pull model snapshot (JSON or NDJSON stream) |
| DELETE | `/api/delete` | Delete installed model |
| GET | `/api/ps` | List loaded models |
| GET | `/v1/models` | List available + installed models |
| POST | `/v1/models/pull` | Pull model (v1 route) |
| DELETE | `/v1/models/{name}` | Delete model (v1 route) |

### Forecasting

| Method | Path | Purpose |
|---|---|---|
| POST | `/api/validate` | Validate forecast payload |
| POST | `/api/forecast` | Forecast (JSON or NDJSON stream) |
| POST | `/api/forecast/progressive` | Progressive SSE forecast stream |
| POST | `/v1/forecast` | Forecast (stable JSON response) |
| POST | `/api/auto-forecast` | Zero-config auto model-selection forecast |

### Structured Analysis

| Method | Path | Purpose |
|---|---|---|
| POST | `/api/compare` | Multi-model forecast comparison |
| POST | `/api/analyze` | Time-series descriptive analysis |
| POST | `/api/generate` | Synthetic series generation |
| POST | `/api/counterfactual` | Intervention/counterfactual analysis |
| POST | `/api/scenario-tree` | Probabilistic scenario tree |
| POST | `/api/what-if` | Scenario transform + forecast |
| POST | `/api/report` | Composite report (analysis + recommendation + forecast) |
| POST | `/api/pipeline` | End-to-end autonomous pipeline |

### A2A

| Method | Path | Purpose |
|---|---|---|
| GET | `/.well-known/agent-card.json` | A2A discovery card |
| GET | `/.well-known/agent.json` | Legacy alias (not in OpenAPI schema) |
| POST | `/a2a` | A2A JSON-RPC endpoint |

---

## Forecast Request Schema

### `POST /api/forecast` — Request fields

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `model` | string | yes | — | Model name (e.g. `chronos2`, `mock`) |
| `horizon` | integer > 0 | yes | — | Number of future steps to forecast |
| `series` | array of `SeriesInput` | yes* | `[]` | Input time series. Required unless `data_url` is set |
| `data_url` | string | yes* | `null` | Local file path or `file://` URL for CSV/Parquet ingest. Mutually exclusive with `series` |
| `quantiles` | array of float | no | `[]` | Sorted unique quantiles in (0, 1). E.g. `[0.1, 0.5, 0.9]` |
| `modelfile` | string | no | `null` | Named TSModelfile profile to apply as defaults |
| `options` | object | no | `{}` | Model-family-specific options (e.g. `{"num_samples": 20}`) |
| `timeout` | float > 0 | no | `null` | Per-request timeout in seconds |
| `ingest` | `IngestOptions` | no | `null` | Column mapping hints when using `data_url` |
| `parameters` | `ForecastParameters` | no | `{}` | Shared forecast control parameters |
| `response_options` | `ResponseOptions` | no | `{}` | Optional response enrichments |

\* Exactly one of `series` or `data_url` must be provided.

#### `SeriesInput`

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `id` | string | yes | — | Series identifier, unique within the request |
| `freq` | string | no | `"auto"` | Pandas offset alias (e.g. `"D"`, `"h"`, `"ME"`) or `"auto"` to infer from timestamps |
| `timestamps` | array of string | yes | — | ISO-8601 timestamps, minimum length 1. Must match `target` length |
| `target` | array of number | yes | — | Numeric values to forecast from. Must match `timestamps` length |
| `past_covariates` | object | no | `null` | Map of feature name → array of numbers/strings, length equal to `len(timestamps)` |
| `future_covariates` | object | no | `null` | Map of feature name → array of length `horizon`. Every key must also appear in `past_covariates` |
| `static_covariates` | object | no | `null` | Key/value metadata with no time axis (e.g. `{"store_id": "A42"}`) |
| `actuals` | array of number | no | `null` | Known future ground truth, length `horizon`. Required when `parameters.metrics` is set |

#### `ForecastParameters`

| Field | Type | Default | Description |
|---|---|---|---|
| `covariates_mode` | `"best_effort"` \| `"strict"` | `"best_effort"` | How unsupported covariates are handled. `best_effort` drops them with a warning; `strict` returns HTTP 400 |
| `metrics` | `MetricsParameters` | `null` | When set, computes accuracy metrics against `series.actuals` |
| `timesfm` | `TimesFMParameters` | `null` | TimesFM-specific xreg knobs |

#### `MetricsParameters`

| Field | Type | Default | Description |
|---|---|---|---|
| `names` | array of string | — | Metrics to compute. Options: `"mape"`, `"mase"`, `"mae"`, `"rmse"`, `"smape"`, `"wape"`, `"rmsse"`, `"pinball"` |
| `mase_seasonality` | integer ≥ 1 | `1` | Seasonality lag used for MASE normalization |

#### `ResponseOptions`

| Field | Type | Default | Description |
|---|---|---|---|
| `narrative` | boolean | `false` | When `true`, includes a natural-language forecast narrative in the response |

#### `IngestOptions`

| Field | Type | Default | Description |
|---|---|---|---|
| `format` | `"csv"` \| `"parquet"` | `null` | Override auto-detected file format |
| `timestamp_column` | string | `null` | Column name to use as timestamps |
| `series_id_column` | string | `null` | Column name to use as series identifier |
| `target_column` | string | `null` | Column name to use as forecast target |
| `freq_column` | string | `null` | Column name that contains the frequency alias |

---

### `POST /api/forecast` — Response fields

| Field | Type | Always present | Description |
|---|---|---|---|
| `model` | string | yes | Model name used for the forecast |
| `forecasts` | array of `SeriesForecast` | yes | One entry per input series |
| `warnings` | array of string | no | Covariate drops, compatibility notices |
| `metrics` | object | no | Accuracy metrics per series (only when `parameters.metrics` is set) |
| `timing` | object | no | `model_load_ms`, `inference_ms`, `total_ms` |
| `explanation` | object | no | Explainability summaries (model-dependent) |
| `narrative` | object | no | Natural-language summary (only when `response_options.narrative=true`) |
| `usage` | object | no | Per-key usage snapshot at time of response |

#### `SeriesForecast`

| Field | Type | Description |
|---|---|---|
| `id` | string | Matches the input `series.id` |
| `mean` | array of float | Point forecast (conditional mean), length `horizon` |
| `quantiles` | object | Map of quantile string → array of float. E.g. `{"0.1": [...], "0.9": [...]}` |

---

### `POST /api/auto-forecast` — Additional fields

Accepts all `ForecastRequest` fields (with `model` optional) plus:

| Field | Type | Default | Description |
|---|---|---|---|
| `strategy` | `"auto"` \| `"fastest"` \| `"best_accuracy"` \| `"ensemble"` | `"auto"` | Model selection strategy |
| `ensemble_top_k` | integer 2–8 | `3` | Number of models to include in ensemble strategies |
| `ensemble_method` | `"mean"` \| `"median"` | `"mean"` | Aggregation method for ensemble forecasts |
| `allow_fallback` | boolean | `false` | Fall back to best available model if the selected one fails |

Response is a `AutoForecastResponse` wrapping a standard `ForecastResponse` plus a `selection` block that includes `strategy`, `chosen_model`, `candidates`, and `rationale`.

---

## Minimal End-to-End Example

Request:
```bash
curl -s http://127.0.0.1:11435/api/forecast \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "mock",
    "horizon": 3,
    "series": [{
      "id": "sales",
      "freq": "D",
      "timestamps": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"],
      "target": [10.0, 11.5, 9.8, 12.0, 10.5]
    }],
    "quantiles": [0.1, 0.9]
  }'
```

Response (abbreviated):
```json
{
  "model": "mock",
  "forecasts": [{
    "id": "sales",
    "mean": [10.76, 10.76, 10.76],
    "quantiles": {
      "0.1": [8.61, 8.61, 8.61],
      "0.9": [12.91, 12.91, 12.91]
    }
  }],
  "warnings": null
}
```

---

## HTTP Status Codes

| Status | Meaning | Example cause |
|---|---|---|
| `200` | Success | Forecast completed |
| `400` | Bad request | Missing `horizon`, covariate shape mismatch, invalid `quantiles` order, `series` + `data_url` both set |
| `401` | Unauthorized | Missing or invalid `Authorization: Bearer` header |
| `403` | Forbidden | Model requires license acceptance (`accept_license` not provided) |
| `404` | Not found | Model not installed — run `tollama pull <model>` first |
| `409` | Conflict | Runner process conflict during concurrent operation |
| `422` | Validation error | Pydantic schema rejection — field type mismatch, extra fields |
| `500` | Internal error | Unhandled exception in runner process |
| `503` | Service unavailable | Runner process crashed or daemon unreachable |

---

## Structured Error Envelope

All error responses use a consistent JSON envelope:

```json
{
  "error": {
    "code": "MODEL_MISSING",
    "message": "model 'chronos2' is not installed; run: tollama pull chronos2",
    "exit_code": 4
  }
}
```

| `code` | `exit_code` | HTTP status | Meaning |
|---|---|---|---|
| `INVALID_REQUEST` | `2` | 400 | Schema or validation failure |
| `DAEMON_UNREACHABLE` | `3` | 503 | Daemon or runner not reachable |
| `MODEL_MISSING` | `4` | 404 | Model not installed |
| `LICENSE_REQUIRED` | `5` | 403 | Model requires license acceptance |
| `PERMISSION_DENIED` | `5` | 401 | Authentication failure |
| `TIMEOUT` | `6` | 504 | Request exceeded timeout |
| `INTERNAL_ERROR` | `10` | 500 | Unexpected internal failure |

The `hint` field may also appear alongside `message` with actionable next steps.

---

## Curl Examples

Version:
```bash
curl -s http://127.0.0.1:11435/api/version
```

Validate request:
```bash
curl -s http://127.0.0.1:11435/api/validate \
  -H 'Content-Type: application/json' \
  -d @examples/request.json
```

Forecast (non-stream):
```bash
curl -s http://127.0.0.1:11435/api/forecast \
  -H 'Content-Type: application/json' \
  -d @examples/request.json
```

Pull model with license acceptance:
```bash
curl -s http://127.0.0.1:11435/api/pull \
  -H 'Content-Type: application/json' \
  -d '{"model":"moirai-2.0-R-small","accept_license":true,"stream":false}'
```

Authenticated docs/openapi:
```bash
curl -s http://127.0.0.1:11435/openapi.json \
  -H 'Authorization: Bearer dev-key-1'
```
