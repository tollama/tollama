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
| **HTTP API** | Forecasting, analysis, comparison, report, ingest, and XAI routes. Canonical inventory lives in `docs/api-reference.md` |
| **Python SDK** | `Tollama` class with forecast / analyze / report / workflow helpers |
| **CLI** | `tollama pull` → `tollama run` — Ollama-style workflow plus diagnostics and runtime tooling |
| **Dashboard** | Web (Chart.js) + TUI (Textual) — model monitoring, forecast visualization |

## Value for AI Agents — MCP, A2A, Skills

AI agents can **invoke TSFMs as tools the moment they need a forecast.**

| Integration | Description |
|------------|------------|
| **MCP Server** | Forecast / analyze / compare / what-if / report oriented tool surface |
| **A2A Protocol** | JSON-RPC based agent-to-agent communication with task queue |
| **LangChain** | Native wrappers for forecasting and structured analysis workflows |
| **CrewAI / AutoGen / Smolagents** | Per-framework adapters built from shared specs |
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
│  AI Agents: MCP / A2A / LangChain / ...               │
├────────────────────────────────────────────────────────┤
│  TSFM Platform Daemon (tollamad)                       │
│  Forecast · Analysis · Compare · What-if · Pipeline    │
│  Auth · Rate Limiting · Prometheus · SSE               │
├──────┬──────┬──────┬──────┬──────┬──────┬──────────────┤
│      │ stdio JSON-lines protocol      │              │
│      ▼      ▼      ▼      ▼      ▼      ▼              │
│   family-specific runner processes (torch, timesfm, ...)│
│   Independent venv per family — zero dependency clash   │
└────────────────────────────────────────────────────────┘
```

## Supported Models

TSFMs and neural baselines sit behind the same forecast contract. The
human-facing guide lives in `docs/models.md`, the machine-readable registry
lives in `model-registry/registry.yaml`, and covariate compatibility lives in
`docs/covariates.md`.

---

## Canonical References

- HTTP endpoint inventory: `docs/api-reference.md`
- Agent tool inventory: `docs/agent-tools.md`
- Model / family guide: `docs/models.md`
- Covariates contract: `docs/covariates.md`
- Machine-readable registry: `model-registry/registry.yaml`

---

## To-Do

| Item | Current Status | Goal |
|------|:--------------:|------|
| **Auto model comparison / selection** | ✅ Basic impl (`/api/compare`, `/api/auto-forecast`) | Advanced best-model routing based on data characteristics |
| **Auto data preprocessing** | ⚠️ Basic pipeline exists (`preprocess/pipeline.py`, ingest normalization) | Harden defaults for interpolation, resampling, and outlier handling |
| **Fine-tuning / ensemble** | ⚠️ Ensemble only (`ensemble.py`) | Add domain-adaptation fine-tuning workflow |
| **Local + cloud execution** | ⚠️ Dockerfile exists, local-centric | K8s manifests, docker-compose, cloud deployment guide |
