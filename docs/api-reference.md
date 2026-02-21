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
