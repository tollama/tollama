# Tollama — Ollama for Time Series Foundation Models

> A unified TSFM platform to run, compare, and serve time series foundation models from a single interface

---

## Why?

TSFMs are emerging as general-purpose models — just like LLMs — enabling time series forecasting without domain-specific custom models. But **actually using them is still painful.**

| Problem | Reality |
|---------|--------|
| Fragmented installation | Chronos requires a specific PyTorch version, TimesFM has its own package, Uni2TS requires Python 3.11 |
| Non-unified APIs | Each model has different input formats, prediction methods, and covariate handling |
| Dependency conflicts | Installing two models simultaneously causes package version collisions |
| Integration friction | Using TSFMs as tools in AI agents or services requires writing per-model wrappers |

## Goal

**Unify fragmented TSFMs under a single interface so that both developers and AI agents can easily leverage time series forecasting through one TSFM platform.**

---

## Value for Developers — API, SDK, Dashboard

Build forecast-based services **quickly** and operate them **with minimal maintenance overhead.**

```python
from tollama import Tollama

t = Tollama()

# Forecast in 3 lines
result = t.forecast(model="chronos2", series=my_data, horizon=30)
df = result.to_df()

# Auto model selection + ensemble
best = t.auto_forecast(series=my_data, horizon=30, strategy="ensemble")

# Chained workflow: analyze → forecast → what-if scenario
flow = t.workflow(my_data).analyze().auto_forecast(horizon=30).what_if(scenarios)
```

| Interface | Description |
|-----------|------------|
| **HTTP API** | One daemon for forecasting, comparison, analysis, reporting, dashboard, and XAI |
| **Python SDK** | `Tollama` class with DataFrame conversion and chained workflow helpers |
| **CLI** | `tollama pull` → `tollama run` — Ollama-style workflow |
| **Dashboard** | Web (Chart.js) + TUI (Textual) — model monitoring, forecast visualization |

## Value for AI Agents — MCP, A2A, Skills

AI agents can **invoke TSFMs as tools the moment they need a forecast.**

| Integration | Description |
|------------|------------|
| **MCP Server** | 22 tools — 15 forecast/orchestration tools + 7 XAI/trust tools |
| **A2A Protocol** | JSON-RPC based agent-to-agent communication with task queue |
| **LangChain** | 13 natively integrated tools |
| **CrewAI / AutoGen / Smolagents** | Per-framework adapters |
| **OpenClaw Skill** | OpenAI tool schema + shell scripts |

## TSFM Platform Dashboard

Manage the entire platform at a glance through web and terminal TUI dashboards.

- Model status monitoring (installed / loaded / running)
- Forecast result visualization and cross-model comparison
- Real-time event streaming (SSE)
- Usage metrics (Prometheus integration)

---

## Architecture

```
┌────────────────────────────────────────────────────────┐
│  Developers: CLI / SDK / HTTP API / Dashboard          │
├────────────────────────────────────────────────────────┤
│  AI Agents: MCP (22 tools) / A2A / LangChain / ...    │
├────────────────────────────────────────────────────────┤
│  TSFM Platform Daemon (tollamad)                       │
│  Forecast · Analysis · Compare · What-if · Pipeline    │
│  Auth · Rate Limiting · Prometheus · SSE               │
├────────────────────────────────────────────────────────┤
│  stdio JSON-lines protocol                             │
│  torch / timesfm / uni2ts / sundial / toto / lag_llama│
│  patchtst / tide / nhits / nbeatsx / timer            │
│  timemixer / forecastpfn / mock                       │
│   Independent venv per family — zero dependency clash   │
└────────────────────────────────────────────────────────┘
```

## Supported Models

| Model | Provider | Covariates |
|-------|----------|:----------:|
| Chronos-2 | Amazon | Past + Future |
| Granite TTM R2 | IBM | Past + Future |
| TimesFM 2.5-200M | Google | Past + Future |
| Moirai 2.0-R Small | Salesforce | Past + Future |
| Sundial Base 128M | THUML | Target only |
| Toto Open Base 1.0 | Datadog | Past only |
| Lag-Llama | Time-Series-Foundation-Models | Target only |
| PatchTST | IBM Granite | Target only |
| TiDE | Tollama local runner | Past + Future |
| N-HiTS | Tollama local runner | Target only |
| N-BEATSx | Tollama local runner | Target only |
| Timer Base | THUML | Target only |
| TimeMixer Base | THUML | Target only |
| ForecastPFN | Abacus.AI | Target only |

---

## To-Do

| Item | Current Status | Goal |
|------|:--------------:|------|
| **Auto model comparison / selection** | ✅ Basic impl (`/api/compare`, `/api/auto-forecast`) | Advanced best-model routing based on data characteristics |
| **Auto data preprocessing** | ⚠️ Basic pipeline exists (`preprocess/pipeline.py`, ingest normalization) | Harden defaults for interpolation, resampling, and outlier handling |
| **Fine-tuning / ensemble** | ⚠️ Ensemble only (`ensemble.py`) | Add domain-adaptation fine-tuning workflow |
| **Local + cloud execution** | ⚠️ Dockerfile exists, local-centric | K8s manifests, docker-compose, cloud deployment guide |
