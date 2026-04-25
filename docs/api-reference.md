# API Reference

Tollama daemon exposes Ollama-style and v1-style HTTP endpoints.

For the release-gated Ollama-style workflow surface, see
`docs/ollama-workflow-parity.md`. This page remains the canonical endpoint
inventory.

Base URL (default): `http://127.0.0.1:11435`

This page is the canonical HTTP endpoint inventory. Overview documents should
link here instead of repeating hardcoded endpoint counts.

## Interactive API Docs

- Swagger UI: `/docs`
- ReDoc: `/redoc`
- OpenAPI JSON: `/openapi.json`
- Checked-in artifact: `docs/openapi.json`

Auth behavior:
- If `auth.api_keys` is configured, these docs endpoints require `Authorization: Bearer <key>`.
- Override for local dev only: `TOLLAMA_DOCS_PUBLIC=1`.
- Refresh the checked-in artifact with `python scripts/export_openapi.py --output docs/openapi.json`.

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
| GET | `/health/live` | Additive liveness alias for `/v1/health` |
| GET | `/health/ready` | Structured readiness probe |

### Runtime / Observability

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/usage` | Per-key usage snapshot |
| GET | `/api/events` | SSE event stream |
| GET | `/metrics` | Prometheus metrics (optional dependency), including per-runner inference/load/memory series |

### Request correlation

- Tollama accepts and emits `X-Request-ID` on daemon responses.
- When present, the daemon forwards that request ID into supervisor-managed runner calls so
  runner protocol IDs can be correlated with the originating HTTP request.

### Readiness semantics

- `GET /v1/health` remains the simple compatibility probe and returns `{"status": "ok"}`.
- `GET /health/live` is an additive alias for liveness checks.
- `GET /health/ready` returns a structured payload with:
  - runner manager readiness
  - local disk-space readiness
  - optional live XAI connector summary only when `TOLLAMA_USE_LIVE_CONNECTORS=1`
- `GET /health/ready` does not probe generic ingest/data connectors under `tollama.connectors.*`.

### Dashboard

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/dashboard/state` | Aggregated dashboard bootstrap payload with partial-failure warnings |

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
| GET | `/v1/models` | List available + installed models with public source, metadata, capabilities, and license details |
| POST | `/v1/models/pull` | Pull model (v1 route) |
| DELETE | `/v1/models/{name}` | Delete model (v1 route) |

### Forecasting

| Method | Path | Purpose |
|---|---|---|
| POST | `/api/validate` | Validate forecast payload |
| POST | `/api/forecast` | Forecast (JSON or NDJSON stream) |
| POST | `/api/forecast/stream` | SSE forecast stream with granular progress events |
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
| POST | `/api/explain-decision` | Decision explanation facade (trust-layer v3.8) |

### Advanced

| Method | Path | Purpose |
|---|---|---|
| POST | `/api/reconcile` | Reconcile hierarchical forecasts (bottom-up, top-down, OLS, MinT) |
| POST | `/api/conformal` | Apply conformal prediction intervals with guaranteed coverage |

### XAI (Explainability & Trust)

| Method | Path | Purpose |
|---|---|---|
| POST | `/api/xai/explain-decision` | End-to-end decision explanation (trust-aware) |
| POST | `/api/xai/trust-breakdown` | Trust Score component breakdown |
| POST | `/api/xai/forecast-decompose` | Trend/seasonal/residual decomposition |
| POST | `/api/xai/model-card` | Generate EU AI Act model card |
| POST | `/api/xai/decision-report` | Build structured decision or explanation report |
| POST | `/api/xai/connectors/health` | Check connector availability |
| POST | `/api/xai/record-outcome` | Record prediction outcome for calibration |
| POST | `/api/xai/gate-decision` | Trust-gated auto-execution check |
| POST | `/api/xai/trust-attribution` | SHAP-like trust component attribution |
| POST | `/api/xai/batch-analyze` | Batch trust analysis (concurrent, max 100) |
| POST | `/api/xai/alerts/configure` | Configure trust alert thresholds |
| GET | `/api/xai/alerts/config` | Get current alert configuration |
| POST | `/api/xai/alerts/check` | Check trust against alert thresholds |
| GET | `/api/xai/cache/stats` | Trust cache hit/miss statistics |
| PUT | `/api/xai/cache/ttl` | Configure cache TTL |
| DELETE | `/api/xai/cache/invalidate` | Clear trust cache |
| GET | `/api/xai/dashboard/agents` | List registered trust agents |
| POST | `/api/xai/dashboard/trust` | Trust dashboard aggregation |
| POST | `/api/xai/dashboard/history` | Trust history and trend snapshots |

### A2A

| Method | Path | Purpose |
|---|---|---|
| GET | `/.well-known/agent-card.json` | A2A discovery card |
| GET | `/.well-known/agent.json` | Legacy alias (not in OpenAPI schema) |
| POST | `/a2a` | A2A JSON-RPC endpoint |

---

## XAI Endpoint Schemas

### `POST /api/xai/explain-decision`

Assembles evidence from eval, calibration, policy, and trust-intelligence layers
into a unified Decision Explanation.

| Field | Type | Required | Description |
|---|---|---|---|
| `forecast_result` | object | yes | Output from tollama forecast endpoint |
| `eval_result` | object | no | Output from tollama-eval |
| `calibration_result` | object | no | Output from Market Calibration Agent |
| `trust_result` | object | no | Normalized output from a domain trust agent |
| `policy_config` | object | no | Decision policy configuration |
| `time_series_data` | array of float | no | Raw time series for decomposition |
| `explain_options` | object | no | Control explanation depth (`decompose`, `attribution`) |
| `trust_features` | object | no | Features for SHAP attribution in trust pipeline |
| `trust_context` | object | no | Trust agent routing context (`domain`, `source_type`, `mode`) |
| `trust_payload` | object | no | Payload routed through the trust-agent registry |

**Supported `trust_context.domain` values:** `prediction_market`, `financial_market`, `supply_chain`, `news`, `geopolitical`, `regulatory`

**Domain-specific `trust_payload` requirements:**
- `financial_market`: requires `instrument_id`
- `news`: requires `story_id`
- `supply_chain`: requires `network_id`
- `geopolitical`: requires `region_id`
- `regulatory`: requires `jurisdiction_id`

### `POST /api/xai/trust-breakdown`

| Field | Type | Required | Description |
|---|---|---|---|
| `trust_score` | float | yes | Overall trust score |
| `metrics` | object | yes | Calibration metrics (brier_score, log_loss, ece, ...) |
| `source` | string | no | Signal source name (default: `polymarket`) |
| `signals` | array of object | no | Multiple signals |

### `POST /api/xai/forecast-decompose`

| Field | Type | Required | Description |
|---|---|---|---|
| `data` | array of float | yes | Time series values |
| `period` | integer | no | Seasonal period (auto-detect if null) |
| `method` | string | no | Decomposition method (default: `stl`) |

### `POST /api/xai/model-card`

| Field | Type | Required | Description |
|---|---|---|---|
| `model_info` | object | yes | Model identity information |
| `eval_result` | object | no | Evaluation results |
| `explanation_result` | object | no | Explanation results |
| `governance_info` | object | no | Governance metadata |
| `format` | string | no | Output format: `json` or `markdown` (default: `json`) |

### `POST /api/xai/decision-report`

| Field | Type | Required | Description |
|---|---|---|---|
| `explanation` | object | yes | Output from `/api/xai/explain-decision` |
| `forecast_result` | object | no | Original forecast result |
| `report_config` | object | no | Report customization (title, audience, format) |
| `report_type` | string | no | `decision` or `explanation` (default: `decision`) |
| `format` | string | no | Output format: `json` or `markdown` (default: `json`) |

### `POST /api/xai/dashboard/history`

| Field | Type | Required | Description |
|---|---|---|---|
| `domains` | array of string | no | Filter to specific trust domains |
| `limit` | integer | no | Maximum history rows per domain (default: `50`, max: `500`) |
| `include_stats` | boolean | no | Include aggregated stats and trend summary |

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

When `data_url` or `/api/forecast/upload` ingestion produces multiple series
for a model that declares a one-series input limit, the daemon forecasts the
first ingested series and returns a warning. Explicit JSON `series` requests are
left unchanged.

When `freq` is omitted and no frequency column is found, `freq="auto"` first uses
strict pandas frequency inference and then falls back to a dominant timestamp
interval for mostly regular data with gaps. Truly irregular timestamps still
require an explicit `freq` alias.

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

Shared primitive formulas for `mae`, `mase`, `smape`, and `rmsse` follow
`ts_autopilot.evaluation.metrics` in `tollama-eval`. The forecast API wraps
those formulas with best-effort request-time behavior: overlap trimming,
warning emission, and macro aggregation across series with defined values.

#### `ResponseOptions`

| Field | Type | Default | Description |
|---|---|---|---|
| `narrative` | boolean | `false` | When `true`, includes a natural-language forecast narrative in the response |
| `explain` | boolean | `false` | When `true`, runs the XAI explanation engine and includes trust-aware decision explanation in the response |

#### `IngestOptions`

| Field | Type | Default | Description |
|---|---|---|---|
| `format` | `"csv"` \| `"parquet"` | `null` | Override auto-detected file format |
| `timestamp_column` | string | `null` | Column name to use as timestamps |
| `series_id_column` | string | `null` | Column name to use as series identifier |
| `target_column` | string | `null` | Column name to use as forecast target |
| `freq` | string | `null` | Explicit pandas offset alias to apply to every ingested series |
| `freq_column` | string | `null` | Column name that contains the frequency alias |

When `timestamp_column` or `target_column` is omitted, ingest auto-detects common
time-series column names case-insensitively. Timestamp aliases include `timestamp`,
`ds`, `time`, `date`, `datetime`, `date_time`, `observation_date`,
`utc_timestamp`, `year`, and `fecha`. Series identifier aliases include `id`,
`series`, `series_id`, `unique_id`, `entity`, and `country`. Target aliases
include `target`, `value`, `y`, `OT`, `demand`, `users`, `number of flights`,
`total electricity`, `GDP`, `close`, `actual`, `pm2.5`, `pm10`, `no2`, `so2`,
and `co`; OPSD-style `*_load_actual_entsoe_transparency` columns are also
detected.
Rows where the resolved target value is null are omitted during ingest; files
whose resolved target column is entirely null are rejected.

---

### `POST /api/forecast` — Response fields

| Field | Type | Always present | Description |
|---|---|---|---|
| `model` | string | yes | Model name used for the forecast |
| `forecasts` | array of `SeriesForecast` | yes | One entry per input series |
| `warnings` | array of string | no | Covariate drops, compatibility notices |
| `metrics` | object | no | Accuracy metrics per series plus `aggregate` macro averages over defined series (only when `parameters.metrics` is set) |
| `timing` | object | no | `model_load_ms`, `inference_ms`, `total_ms` |
| `explanation` | object | no | Explainability summaries (model-dependent) |
| `narrative` | object | no | Natural-language summary (only when `response_options.narrative=true`) |
| `usage` | object | no | Runtime telemetry snapshot such as `runner`, `device`, `peak_memory_mb`, plus adapter-specific keys |

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
| `400` | Bad request | Missing `horizon`, field type mismatch, extra fields, covariate shape mismatch, invalid `quantiles` order, `series` + `data_url` both set |
| `401` | Unauthorized | Missing or invalid `Authorization: Bearer` header |
| `403` | Forbidden | Model requires license acceptance (`accept_license` not provided) |
| `404` | Not found | Model not installed — run `tollama pull <model>` first |
| `409` | Conflict | Runner process conflict during concurrent operation |
| `502` | Bad gateway | Runner returned an execution error or invalid upstream response |
| `500` | Internal error | Unhandled exception in runner process |
| `503` | Service unavailable | Runner process crashed or daemon unreachable |

---

Request payload/schema validation is intentionally normalized to HTTP `400`
rather than `422`.

## HTTP Error Body

Raw HTTP endpoints return a compact JSON body built around `detail`, with an
optional `hint` when the daemon can suggest the next step:

```json
{
  "detail": "model 'chronos2' is not installed; run: tollama pull chronos2",
  "hint": "Run `tollama pull <model>` to install. Use `tollama info --json` to inspect available models."
}
```

| Field | Type | Always present | Description |
|---|---|---|---|
| `detail` | string \| array \| object | yes | Human-readable or structured error detail |
| `hint` | string | no | Actionable retry guidance when the daemon can provide one |

Typed clients and agent integrations may map these raw HTTP failures into
higher-level categories or exit codes, but the HTTP response body itself does
not include `code` or `exit_code`.

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
