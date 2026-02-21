# tollama

Local-first forecasting daemon + CLI for TSFM models, with SDK and agent integrations.

## Install

```bash
python -m pip install tollama
```

For local development:

```bash
python -m pip install -e ".[dev]"
```

## 5-Minute Quickstart

```bash
# Terminal 1: run daemon (default http://127.0.0.1:11435)
tollama serve

# Terminal 2: pull + forecast demo + next steps
tollama quickstart
```

Human-friendly progress is enabled automatically on interactive terminals.
You can override with `--progress on` or `--progress off` on `pull`, `run`,
`quickstart`, and `runtime install`.

Useful CLI additions:

```bash
# explain model limits/capabilities/license from registry + local manifest
tollama explain chronos2

# scaffold a new runner family skeleton (files only)
tollama dev scaffold acme_family

# scaffold + register script/module-map/registry template entry
tollama dev scaffold acme_family --register
```

## Shell Completion

Typer completion is available out of the box:

```bash
# install completion for your current shell
tollama --install-completion

# print completion script for custom setup
tollama --show-completion
```

## Python SDK

```python
from tollama import Tollama

t = Tollama()
result = t.forecast(
    model="chronos2",
    series={"target": [10, 11, 12, 13, 14], "freq": "D"},
    horizon=3,
)
print(result.mean)
print(result.to_df())

auto = t.auto_forecast(
    series={"target": [10, 11, 12, 13, 14], "freq": "D"},
    horizon=3,
    strategy="auto",
    ensemble_method="mean",  # or "median" when strategy="ensemble"
)
print(auto.selection.chosen_model)
print(auto.response.forecasts[0].mean)

from_file = t.forecast_from_file(
    model="chronos2",
    path="examples/history.csv",
    horizon=3,
    format_hint="csv",
)
print(from_file.to_df())

what_if = t.what_if(
    model="chronos2",
    series={"target": [10, 11, 12, 13, 14], "freq": "D"},
    horizon=3,
    scenarios=[
        {"name": "high_demand", "transforms": [{"operation": "multiply", "field": "target", "value": 1.2}]}
    ],
)
print(what_if.summary)

pipeline = t.pipeline(
    series={"target": [10, 11, 12, 13, 14], "freq": "D"},
    horizon=3,
    strategy="auto",
    pull_if_missing=True,
)
print(pipeline.auto_forecast.response.model)

synthetic = t.generate(
    series={"target": [10, 11, 12, 13, 14], "freq": "D"},
    count=2,
    length=7,
    seed=42,
)
print(synthetic.generated[0].id)

# additive chainable workflow (keeps existing SDK method contracts unchanged)
with Tollama() as sdk:
    flow = (
        sdk.workflow(series={"target": [10, 11, 12, 13, 14], "freq": "D"})
        .analyze()
        .auto_forecast(horizon=3)
    )
print(flow.auto_forecast_result.selection.chosen_model)

# reuse one forecast request for compare/what-if
baseline = t.forecast(
    model="chronos2",
    series={"target": [10, 11, 12, 13, 14], "freq": "D"},
    horizon=3,
)
comparison = baseline.then_compare(models=["timesfm-2.5-200m"])
print(comparison.summary)
```

## Data Ingest (CSV/Parquet)

- `POST /v1/forecast` and `POST /api/forecast` now accept `data_url` + optional `ingest` options.
- `POST /api/forecast/upload` accepts multipart upload (`payload` JSON + `file`) and runs forecast.
- `POST /api/ingest/upload` accepts multipart upload and returns normalized `series` payloads.

## Real-Time Streams

- `GET /api/events` exposes per-key SSE event streams.
  - Event filters: `event=<name>` (repeatable) or `events=a,b,c`
  - Optional controls: `heartbeat=<seconds>`, `max_events=<count>`
- `POST /api/forecast/progressive` streams staged forecast refinement events over SSE:
  - `model.selected`
  - `forecast.progress`
  - `forecast.complete`

## Dashboards (Web + TUI)

- Web dashboard ships with the Python package and is served by daemon routes:
  - `GET /dashboard`
  - `GET /dashboard/{path:path}` (SPA-style deep-link fallback)
  - `GET /dashboard/static/*` (bundled assets)
  - `GET /dashboard/partials/{name}` (HTML partial templates)
- Aggregated bootstrap endpoint: `GET /api/dashboard/state`
  - combines `/api/info`, `/api/ps`, and `/api/usage`
  - follows existing API-key auth policy (same as other `/api/*` routes)
  - returns partial data plus `warnings[]` when one source is unavailable
- Dashboard environment flags:
  - `TOLLAMA_DASHBOARD` (`1/0`, default enabled)
  - `TOLLAMA_DASHBOARD_REQUIRE_AUTH` (`1/0`, default disabled)
  - `TOLLAMA_CORS_ORIGINS` (comma-separated explicit allowlist; no wildcard default)
- CLI helpers:
  - `tollama open` opens the web dashboard in a browser
  - `tollama dashboard` launches the Textual TUI (install with `python -m pip install -e \".[tui]\"`)

## Structured Intelligence + Generative Planning

- `POST /api/report` returns composite structured intelligence in one call:
  analyze + recommend + auto-forecast (+ optional baseline/narrative).
- `POST /api/counterfactual` estimates post-intervention counterfactual trajectories and
  divergence vs observed values.
- `POST /api/scenario-tree` builds branching probabilistic futures as flattened tree nodes.

Example `data_url` request:

```json
{
  "model": "mock",
  "horizon": 3,
  "data_url": "file:///tmp/history.csv",
  "ingest": {"format": "csv"},
  "options": {}
}
```

Parquet requires optional dependency:

```bash
python -m pip install -e ".[ingest]"
```

## TSModelfile

Create/list/show/remove forecast profiles:

```bash
tollama modelfile create baseline --file examples/modelfile.yaml
tollama modelfile list
tollama modelfile show baseline
tollama modelfile rm baseline
```

Daemon APIs:

- `GET /api/modelfiles`
- `GET /api/modelfiles/{name}`
- `POST /api/modelfiles`
- `DELETE /api/modelfiles/{name}`

## Agent Integrations

- MCP server: `tollama-mcp`
- LangChain tools: `tollama.skill.langchain.get_tollama_tools(...)`
- CrewAI tools: `tollama.skill.get_crewai_tools(...)`
- AutoGen specs/map: `tollama.skill.get_autogen_tool_specs(...)`, `tollama.skill.get_autogen_function_map(...)`
- smolagents tools: `tollama.skill.get_smolagents_tools(...)`
- OpenClaw skill: `skills/tollama-forecast/`

## A2A Integration (Latest Spec)

- Discovery: `GET /.well-known/agent-card.json`
- JSON-RPC endpoint: `POST /a2a`
- Implemented methods:
  - `message/send`
  - `message/stream` (SSE)
  - `tasks/get`
  - `tasks/query`
  - `tasks/cancel`
- Current capability flags in Agent Card:
  - `streaming=true`
  - `pushNotifications=false`
- When API keys are configured, discovery and `/a2a` calls require
  `Authorization: Bearer <key>` (authenticated discovery default).

Minimal outbound A2A client is available:

```python
from tollama.a2a import A2AClient

client = A2AClient()
card = client.discover(base_url="http://127.0.0.1:11435")
print(card["name"])
```

## Model Guides

- Model-family setup and per-model examples: `docs/models.md`
- Covariates contract and compatibility matrix: `docs/covariates.md`
- Full setup and dependency guide: `docs/how-to-run.md`
- Troubleshooting (canonical): `docs/troubleshooting.md`
- CLI command reference: `docs/cli-cheatsheet.md`
- API reference: `docs/api-reference.md`
- Notebooks: `examples/quickstart.ipynb`, `examples/agent_demo.ipynb`, `examples/tutorial_covariates.ipynb`, `examples/tutorial_comparison.ipynb`, `examples/tutorial_what_if.ipynb`, `examples/tutorial_auto_forecast.ipynb`

## Local Checks

```bash
ruff check .
pytest -q
```

## Prometheus Metrics

Install optional metrics dependency:

```bash
python -m pip install -e ".[metrics]"
```

Daemon endpoint:

```bash
curl -s http://127.0.0.1:11435/metrics
```

## API Key Auth + Usage Metering

Optional API-key auth is configured in `~/.tollama/config.json`:

```json
{
  "version": 1,
  "auth": {
    "api_keys": ["dev-key-1", "dev-key-2"]
  }
}
```

When API keys are configured, interactive docs endpoints (`/docs`, `/redoc`, `/openapi.json`)
also require bearer auth by default. For local-only public docs, set `TOLLAMA_DOCS_PUBLIC=1`.

When keys are set, daemon endpoints require `Authorization: Bearer <key>`.
CLI and SDK clients can use `TOLLAMA_API_KEY` / `api_key=...`.

Usage metering is available at:

```bash
curl -s http://127.0.0.1:11435/api/usage
```

## Benchmark Script

Compare SDK ergonomics and time-to-first-forecast versus raw client calls:

```bash
PYTHONPATH=src python benchmarks/tollama_vs_raw.py --model mock --iterations 3 --warmup 1
```

## Docker

Build and run:

```bash
docker build -t tollama/tollama .
docker run -p 11435:11435 tollama/tollama
```

Compose (CPU + optional GPU profile):

```bash
docker compose up --build tollama
docker compose --profile gpu up --build tollama-gpu
```

## Persistent Pull Defaults

`tollama config` stores pull defaults in `~/.tollama/config.json` (or `$TOLLAMA_HOME/config.json`).
These defaults are applied by the daemon on `/api/pull`, so any API client benefits.

```bash
# write a default config file
tollama config init

# inspect current defaults
tollama config list
tollama config keys

# set persistent defaults
tollama config set pull.https_proxy http://proxy:3128
tollama config set pull.hf_home /mnt/fastcache/hf
tollama config set pull.offline true

# no pull flags needed; daemon applies config defaults
tollama pull chronos2
```

Tokens are intentionally not persisted in config. Use `TOLLAMA_HF_TOKEN` or `--token`.
Unknown config keys return suggestions (for example `offline` -> `pull.offline`).

For `tollama run`, omitting `MODEL` in an interactive terminal prompts you to pick
from installed models. You can also add `--interactive` to choose from discovered
`examples/*_request.json` files when `--input` is omitted.

## Diagnostics Endpoint: `/api/info`

`GET /api/info` returns one fast diagnostics payload for server-side integrations.
It includes:
- daemon metadata (`version`, `started_at`, `uptime_seconds`, `host_binding`)
- local paths and config presence (`tollama_home`, `config_path`, `config_exists`)
- redacted config and safe env subset
- effective pull defaults with value source (`env`/`config`/`default`)
- installed + loaded models (including covariate capabilities when available)
- registry-available models with covariate capability metadata
- runner statuses for `mock`, `torch`, `timesfm`, `uni2ts`, `sundial`, and `toto`

Prometheus-compatible metrics are also exposed at `GET /metrics` when the
optional `prometheus-client` dependency is installed.
Per-key usage aggregates are exposed at `GET /api/usage`.

`tollama info` renders the same payload and prints a short per-model covariates summary.

Example shape:

```json
{
  "daemon": {"version": "0.1.0", "started_at": "...", "uptime_seconds": 42},
  "paths": {"tollama_home": "...", "config_path": "...", "config_exists": true},
  "config": {"pull": {"https_proxy": "http://***:***@proxy:3128"}},
  "env": {"HTTP_PROXY": "http://***:***@proxy:3128", "TOLLAMA_HF_TOKEN_present": false},
  "pull_defaults": {"offline": {"value": false, "source": "default"}},
  "models": {
    "installed": [{"name": "chronos2", "capabilities": {"past_covariates_numeric": true}}],
    "loaded": [],
    "available": [{"name": "timesfm-2.5-200m", "capabilities": {"future_covariates_numeric": true}}]
  },
  "runners": []
}
```

## Debugging: `tollama info`

Use `tollama info` for one diagnostics view of daemon reachability, local config, effective pull
defaults, and installed/loaded models.

```bash
tollama info
tollama info --json
tollama info --local   # force local-only
tollama info --remote  # require daemon /api/info
```

Example output excerpt:

```text
Tollama
  Home: /Users/you/.tollama
  Config: /Users/you/.tollama/config.json (exists)
  Daemon: http://localhost:11435 (reachable) version=0.1.0

Pull defaults (effective)
  offline: false (source=default)
  https_proxy: http://proxy:3128 (source=config)
```

`tollama info` prefers `GET /api/info` when the daemon is reachable, and automatically falls back
to local collection when the daemon is down (unless `--remote` is set).

## LangChain Integration (Python SDK)

Install optional LangChain dependencies:

```bash
python -m pip install -e ".[langchain]"
```

Create preconfigured tools:

```python
from tollama.skill import get_tollama_tools

tools = get_tollama_tools(base_url="http://127.0.0.1:11435", timeout=10.0)
```

Provided `BaseTool` wrappers:

- `TollamaForecastTool`
- `TollamaAutoForecastTool`
- `TollamaAnalyzeTool`
- `TollamaGenerateTool`
- `TollamaWhatIfTool`
- `TollamaPipelineTool`
- `TollamaCompareTool`
- `TollamaRecommendTool`
- `TollamaHealthTool`
- `TollamaModelsTool`
- `get_tollama_tools(base_url="http://127.0.0.1:11435", timeout=10.0)`

Tool input contracts:

- `tollama_health`: no runtime args (`{}`)
- `tollama_models`: `{"mode": "installed"|"loaded"|"available"}`
- `tollama_forecast`: `{"request": <ForecastRequest-compatible dict>}`
- `tollama_auto_forecast`: `{"request": <AutoForecastRequest-compatible dict>}`
- `tollama_analyze`: `{"request": <AnalyzeRequest-compatible dict>}`
- `tollama_generate`: `{"request": <GenerateRequest-compatible dict>}`
- `tollama_what_if`: `{"request": <WhatIfRequest-compatible dict>}`
- `tollama_pipeline`: `{"request": <PipelineRequest-compatible dict>}`
- `tollama_compare`: `{"request": <CompareRequest-compatible dict>}`
- `tollama_recommend`:
  `{"horizon": int, "freq"?: str, "has_past_covariates"?: bool, ...}`

Tool output/error behavior:

- all wrappers return structured `dict` payloads (not JSON strings)
- client/runtime failures return
  `{"error":{"category","exit_code","message"}}`
- invalid local input validation maps to
  `{"error":{"category":"INVALID_REQUEST","exit_code":2,...}}`
- async calls use real non-blocking `_arun` implementations backed by
  `AsyncTollamaClient` (no sync fallback stubs)
- if `langchain_core` is missing, import raises:
  `pip install "tollama[langchain]"`

Validation test:

```bash
PYTHONPATH=src python -m pytest -q tests/test_langchain_skill.py
```

## Additional Agent Wrappers (Python)

These wrappers reuse the same tool contracts via
`src/tollama/skill/framework_common.py`.

```python
from tollama.skill import (
    get_autogen_function_map,
    get_autogen_tool_specs,
    get_crewai_tools,
    get_smolagents_tools,
)

crewai_tools = get_crewai_tools()
smolagents_tools = get_smolagents_tools()
autogen_tools = get_autogen_tool_specs()
autogen_function_map = get_autogen_function_map()
```

## OpenClaw Integration (Skill: `tollama-forecast`)

OpenClaw integration is provided by the skill package under
`skills/tollama-forecast/`:

- `SKILL.md`
- `bin/tollama-health.sh`
- `bin/tollama-models.sh`
- `bin/tollama-forecast.sh`
- `bin/tollama-pull.sh`
- `bin/tollama-rm.sh`
- `bin/tollama-info.sh`
- `bin/_tollama_lib.sh`
- `openai-tools.json`
- `examples/*.json`

This integration is OpenClaw-first and does not require any daemon/core/plugin
changes.

Runbooks for end-to-end operator workflows:

- `docs/openclaw-sandbox-runbook.md`
- `docs/openclaw-gateway-runbook.md`

### Skill implementation layout

- `skills/tollama-forecast/bin/_tollama_lib.sh`:
  shared HTTP helpers (`http_request`), error classifiers, and `emit_error`
  (plain stderr / JSON stderr dual-mode).
- `skills/tollama-forecast/bin/tollama-forecast.sh`:
  forecast entrypoint (payload normalization + metrics flag injection +
  model check/pull + forecast execution).
- `skills/tollama-forecast/bin/tollama-models.sh`:
  model lifecycle multiplexer (`installed|loaded|show|available|pull|rm|info`).
- `skills/tollama-forecast/bin/tollama-health.sh`:
  daemon health/version probe with optional runtime summary (`--runtimes`).
- `skills/tollama-forecast/bin/tollama-pull.sh` / `tollama-rm.sh` /
  `tollama-info.sh`:
  thin wrappers delegating to `tollama-models.sh`.

### Script contracts (current)

| Script | Primary behavior | CLI path | HTTP path / endpoints |
|---|---|---|---|
| `tollama-forecast.sh` | non-stream forecast JSON output | `tollama show` + optional `tollama pull --no-stream` + `tollama run --no-stream` | `POST /api/show`, optional `POST /api/pull`, `POST /api/forecast`, fallback `POST /v1/forecast` on `/api/forecast` `404` |
| `tollama-models.sh installed` | installed models list | `tollama list --json` | `GET /api/tags` |
| `tollama-models.sh loaded` | loaded models list | `tollama ps --json` | `GET /api/ps` |
| `tollama-models.sh show` | model detail | `tollama show` | `POST /api/show` |
| `tollama-models.sh pull` | pull/install model | `tollama pull --no-stream` | `POST /api/pull` |
| `tollama-models.sh rm` | remove model | `tollama rm` | `DELETE /api/delete` |
| `tollama-models.sh available` | available model list | `tollama info --json --remote` + local extract | `GET /api/info` + local extract |
| `tollama-models.sh info` | daemon info section output | `tollama info --json --remote` + section extract | `GET /api/info` + section extract |
| `tollama-health.sh` | daemon health payload | n/a (curl path only) | `GET /v1/health`, `GET /api/version`, optional `GET /api/info` (`--runtimes`) |

### Install (Managed skills, recommended)

```bash
mkdir -p ~/.openclaw/skills
ln -s "$(pwd)/skills/tollama-forecast" ~/.openclaw/skills/tollama-forecast
openclaw skills list --eligible | rg tollama-forecast
```

### Environment checks

```bash
tollama serve
tollama info --json
curl -s http://localhost:11435/api/version
openclaw skills list --eligible | rg tollama-forecast
```

### Runtime defaults and policy

- Base URL: `--base-url` > `TOLLAMA_BASE_URL` > `http://127.0.0.1:11435`
- Timeout: `--timeout` > `TOLLAMA_FORECAST_TIMEOUT_SECONDS` > `300`
- Forecast requests are non-stream by default.
- `tollama-forecast.sh` does not auto-pull by default; model install happens
  only when `--pull` is provided.
- `tollama-forecast.sh` supports metrics-aware requests via input JSON
  (`series[].actuals`, `parameters.metrics`) and convenience flags:
  `--metrics <csv>`, `--mase-seasonality <int>`.
- CLI -> HTTP fallback in `tollama-forecast.sh` is enabled only when `tollama`
  is unavailable in PATH.
- HTTP forecast endpoint order is `POST /api/forecast` first, then
  `POST /v1/forecast` only when `/api/forecast` returns `404`.
- `tollama-models.sh available` is daemon-only (`tollama info --json --remote`
  or `GET /api/info`) and does not use local fallback.
- `tollama-health.sh` output includes `healthy`, `version_name`, and optional
  `runtimes` (`--runtimes`).
- Structured error output is available via `TOLLAMA_JSON_STDERR=1`.
- Skill metadata eligibility is `bins=["bash"]` and `anyBins=["tollama","curl"]`.
- Skill-side validation remains minimal by design:
  - validates option shape (`--timeout` positive number, `--metrics` non-empty CSV,
    `--mase-seasonality` positive integer)
  - full schema/covariate/metrics validation is delegated to daemon (HTTP `400` path)

### Breaking change: skill exit code contract v2

OpenClaw `tollama-forecast` scripts now share this exit code contract:
- `0`: success
- `2`: invalid input/request
- `3`: daemon unreachable/health failure
- `4`: model not installed
- `5`: license required/not accepted/permission
- `6`: timeout
- `10`: unexpected internal error

When `TOLLAMA_JSON_STDERR=1`, stderr is emitted as structured JSON:

```json
{"error":{"code":"MODEL_MISSING","subcode":"LICENSE_REQUIRED","exit_code":5,"message":"...","hint":"..."}}
```

### Troubleshooting

- Canonical troubleshooting guide: `docs/troubleshooting.md`
- OpenClaw-specific path issue: ensure `tollama` is on PATH in the OpenClaw exec environment.
- OpenClaw-specific connectivity issue: verify `--base-url` is reachable from the OpenClaw exec host.

### Skill smoke checks

```bash
bash skills/tollama-forecast/bin/tollama-health.sh --base-url "$TOLLAMA_BASE_URL"
bash skills/tollama-forecast/bin/tollama-models.sh installed --base-url "$TOLLAMA_BASE_URL"
bash skills/tollama-forecast/bin/tollama-pull.sh --model mock --base-url "$TOLLAMA_BASE_URL"
bash skills/tollama-forecast/bin/tollama-info.sh --section runners --base-url "$TOLLAMA_BASE_URL"
cat skills/tollama-forecast/examples/simple_forecast.json | \
  bash skills/tollama-forecast/bin/tollama-forecast.sh --model mock --base-url "$TOLLAMA_BASE_URL"
# metrics-aware request example (input payload includes actuals + parameters.metrics)
bash skills/tollama-forecast/bin/tollama-forecast.sh \
  --model mock \
  --input skills/tollama-forecast/examples/metrics_forecast.json \
  --base-url "$TOLLAMA_BASE_URL"
# optional convenience override flags
bash skills/tollama-forecast/bin/tollama-forecast.sh \
  --model mock \
  --input skills/tollama-forecast/examples/simple_forecast.json \
  --metrics mape,mase,mae,rmse,smape \
  --mase-seasonality 1 \
  --base-url "$TOLLAMA_BASE_URL"
```

## MCP Integration (Claude Code)

Tollama now includes an MCP server package for native Claude Code integration.

Install optional MCP dependencies:

```bash
python -m pip install -e ".[mcp]"
```

Run MCP server:

```bash
tollama-mcp
```

Claude Desktop registration example:

```json
{
  "mcpServers": {
    "tollama": {
      "command": "tollama-mcp",
      "env": {
        "TOLLAMA_BASE_URL": "http://127.0.0.1:11435"
      }
    }
  }
}
```

Automated installer (macOS/Linux):

```bash
bash scripts/install_mcp.sh --base-url "http://127.0.0.1:11435"
```

### Implementation layout

- `src/tollama/client/http.py`:
  shared HTTP client used by CLI and MCP (`TollamaClient`).
- `src/tollama/client/exceptions.py`:
  typed error hierarchy with category + exit code metadata.
- `src/tollama/mcp/schemas.py`:
  strict tool input schemas (`extra="forbid"`, strict types).
- `src/tollama/mcp/tools.py`:
  tool handlers, input validation, client calls, response normalization.
- `src/tollama/mcp/server.py`:
  FastMCP registration + MCP error envelope mapping.
- `src/tollama/mcp/__main__.py`:
  `tollama-mcp` entrypoint.

### Tool contracts (current)

| Tool | Backend endpoint(s) | Key args | Return shape |
|---|---|---|---|
| `tollama_health` | `GET /v1/health`, `GET /api/version` | `base_url?`, `timeout?` | `{healthy, health, version}` |
| `tollama_models` | `GET /api/tags` or `/api/ps` or `/api/info` | `mode=installed\|loaded\|available`, `base_url?`, `timeout?` | `{mode, items}` |
| `tollama_forecast` | `POST /api/forecast` (non-stream) | `request`, `base_url?`, `timeout?` | canonical `ForecastResponse` JSON |
| `tollama_auto_forecast` | `POST /api/auto-forecast` | `request`, `base_url?`, `timeout?` | canonical `AutoForecastResponse` JSON |
| `tollama_analyze` | `POST /api/analyze` | `request`, `base_url?`, `timeout?` | canonical `AnalyzeResponse` JSON |
| `tollama_generate` | `POST /api/generate` | `request`, `base_url?`, `timeout?` | canonical `GenerateResponse` JSON |
| `tollama_counterfactual` | `POST /api/counterfactual` | `request`, `base_url?`, `timeout?` | canonical `CounterfactualResponse` JSON |
| `tollama_scenario_tree` | `POST /api/scenario-tree` | `request`, `base_url?`, `timeout?` | canonical `ScenarioTreeResponse` JSON |
| `tollama_report` | `POST /api/report` | `request`, `base_url?`, `timeout?` | canonical `ForecastReport` JSON |
| `tollama_what_if` | `POST /api/what-if` | `request`, `base_url?`, `timeout?` | canonical `WhatIfResponse` JSON |
| `tollama_pipeline` | `POST /api/pipeline` | `request`, `base_url?`, `timeout?` | canonical `PipelineResponse` JSON |
| `tollama_compare` | `POST /api/compare` | `request`, `base_url?`, `timeout?` | canonical `CompareResponse` JSON |
| `tollama_recommend` | registry metadata + capabilities | `horizon`, covariate flags, `top_k`, `allow_restricted_license` | ranked recommendation payload |
| `tollama_pull` | `POST /api/pull` (non-stream) | `model`, `accept_license?`, `base_url?`, `timeout?` | daemon pull result JSON |
| `tollama_show` | `POST /api/show` | `model`, `base_url?`, `timeout?` | daemon show payload JSON |

Notes:
- `tollama_forecast` validates `request` with `ForecastRequest` before HTTP call.
- `tollama_auto_forecast` validates `request` with `AutoForecastRequest` before HTTP call.
- `tollama_analyze` validates `request` with `AnalyzeRequest` before HTTP call.
- `tollama_generate` validates `request` with `GenerateRequest` before HTTP call.
- `tollama_counterfactual` validates `request` with `CounterfactualRequest` before HTTP call.
- `tollama_scenario_tree` validates `request` with `ScenarioTreeRequest` before HTTP call.
- `tollama_report` validates `request` with `ReportRequest` before HTTP call.
- `tollama_what_if` validates `request` with `WhatIfRequest` before HTTP call.
- `tollama_pipeline` validates `request` with `PipelineRequest` before HTTP call.
- `response_options.narrative=true` enables deterministic structured narrative blocks
  in `forecast`/`analyze`/`compare`/`pipeline` responses.
- MCP tool input schemas require `timeout > 0` when provided.
- MCP integration is intentionally non-streaming for deterministic tool responses.

### Error mapping contract

MCP tools map client/daemon failures to a stable category + exit-code-aligned payload:

| Category | Exit code |
|---|---|
| `INVALID_REQUEST` | `2` |
| `DAEMON_UNREACHABLE` | `3` |
| `MODEL_MISSING` | `4` |
| `LICENSE_REQUIRED` / `PERMISSION_DENIED` | `5` |
| `TIMEOUT` | `6` |
| `INTERNAL_ERROR` | `10` |

MCP server emits tool failures as:

```json
{"error":{"category":"...","exit_code":3,"message":"...","hint":"..."}}
```

### Defaults and overrides

- Default MCP client base URL: `http://localhost:11435`.
- Default MCP client timeout: `10` seconds.
- Every MCP tool accepts optional `base_url` and `timeout` overrides.
- `scripts/install_mcp.sh` upserts Claude Desktop `mcpServers.<name>` and sets
  `env.TOLLAMA_BASE_URL`.

Quick checks:

```bash
tollama-mcp
bash scripts/install_mcp.sh --dry-run --base-url "http://127.0.0.1:11435"
```

## Architecture

- `tollama.daemon`: Public API layer (`/api/*`, `/v1/health`, `/v1/forecast`) and runner supervision.
- `tollama.runners`: Runner implementations that speak newline-delimited JSON over stdio.
- `tollama.core`: Canonical schemas (`Forecast*`, `AutoForecast*`, `Analyze*`, `WhatIf*`,
  `Pipeline*`, `Compare*`, `Counterfactual*`, `ScenarioTree*`, `Report*`) and protocol helpers.
- `tollama.cli`: User CLI commands for serving and sending forecast requests.
- `tollama.sdk`: High-level Python convenience facade (`Tollama`, `TollamaForecastResult`).
- `tollama.client`: Shared HTTP client abstraction for CLI/MCP integrations.
- `tollama.mcp`: MCP tool server for Claude Code/native MCP clients.

### Daemon <-> Runner Protocol

- Transport: newline-delimited JSON over stdio.
- Request shape: `{"id":"...","method":"...","params":{...}}`
- Response shape: `{"id":"...","result":{...}}` or `{"id":"...","error":{"code":...,"message":"..."}}`
- Current mock runner methods: `hello`, `forecast`.
- Current torch runner methods: `hello`, `load`, `unload`, `forecast`.

## Adding A New Runner Family

1. Add a new package under `src/tollama/runners/<family>/`.
2. Implement a `main.py` stdio loop using `tollama.core.protocol` encode/decode helpers.
3. Validate incoming forecast params with `tollama.core.schemas.ForecastRequest`.
4. Return canonical outputs compatible with `tollama.core.schemas.ForecastResponse`.
5. Register a console script in `pyproject.toml` if the runner should be executable directly.
6. Add tests for protocol behavior and end-to-end daemon routing.

## Project Structure

```text
.
├── src/
│   └── tollama/
│       ├── client/
│       ├── cli/
│       ├── core/
│       ├── daemon/
│       ├── mcp/
│       ├── runners/
│       └── sdk.py
├── examples/
├── scripts/
├── tests/
└── .github/workflows/
```
