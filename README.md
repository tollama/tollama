# Tollama — Ollama for Time Series Forecasting

[![CI](https://github.com/yongchoelchoi/tollama/actions/workflows/ci.yml/badge.svg)](https://github.com/yongchoelchoi/tollama/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)

> Local-first time series forecasting core for preprocessing irregular series, benchmarking models, and routing forecast workloads through one API, CLI, and SDK

---

## Why Tollama?

TSFMs are becoming practical building blocks for forecasting, but **the path from raw series to an operational decision is still fragmented.**

| Problem                  | Reality                                                                                               |
| ------------------------ | ----------------------------------------------------------------------------------------------------- |
| Fragmented installation  | Chronos requires a specific PyTorch version, TimesFM has its own package, Uni2TS requires Python 3.11 |
| Non-unified APIs         | Each model has different input formats, prediction methods, and covariate handling                    |
| Irregular-series cleanup | Missing timestamps, gaps, and smoothing are often solved outside the forecast stack                   |
| No benchmark evidence    | Teams still guess which model to trust in production without comparable accuracy and latency results  |

## Goal

**Turn fragmented TSFMs and neural baselines into one core workflow for forecast-driven time-series work: preprocess -> forecast -> benchmark -> route.**

---

## Quickstart (5 min)

```bash
# Terminal 1: Install & run daemon
python -m pip install "tollama[eval,preprocess]"
tollama serve

# Terminal 2: pull + forecast demo + next steps
tollama quickstart

# Terminal 2: save a first benchmark artifact
tollama benchmark examples/benchmark_data.json --models mock --horizon 4 --folds 1 --output artifacts/benchmarks/demo
```

For more, see [Core Workflow](docs/core-workflow.md) and [TSFM Routing Defaults](docs/tsfm-routing-defaults.md).

The precise meaning of "Ollama for Time Series Forecasting" is documented in
[Ollama-Workflow Parity](docs/ollama-workflow-parity.md).

---

## Supported Models

| Model              | Provider                | Type            | Covariates    |
| ------------------ | ----------------------- |:---------------:|:-------------:|
| Chronos-2          | Amazon                  | TSFM            | Past + Future |
| Granite TTM R2     | IBM                     | TSFM            | Past + Future |
| TimesFM 2.5-200M   | Google                  | TSFM            | Past + Future |
| Moirai 2.0-R Small | Salesforce              | TSFM            | Past + Future |
| Sundial Base 128M  | THUML                   | TSFM            | Target only   |
| Toto Open Base 1.0 | Datadog                 | TSFM            | Past only     |
| Lag-Llama          | TSFM Community          | TSFM            | Target only   |
| PatchTST           | IBM Granite             | Neural Baseline | Target only   |
| TiDE               | Unit8 / Darts           | Neural Baseline | Target only   |
| N-HiTS             | Nixtla / NeuralForecast | Neural Baseline | Past + Future + Static |
| N-BEATSx           | Nixtla / NeuralForecast | Neural Baseline | Past + Future + Static |
| TimeMixer          | Tsinghua / THUML        | Neural Baseline | Target only   |
| Timer              | Tsinghua / THUML        | Neural Baseline | Target only   |
| ForecastPFN        | PFN Research            | Neural Baseline | Target only   |

TimesFM 2.5 and Moirai keep dynamic covariates in practical best-effort mode:
when `parameters.covariates_mode` remains at the default `best_effort`, runtime
covariate-path failures degrade to target-only forecasts with warnings.

> **TSFM vs Neural Baseline — what's the difference?**
>
> - **TSFM (Time Series Foundation Model)**: Pre-trained on large, diverse time series corpora (billions of data points across multiple domains). Supports **zero-shot forecasting** — produces predictions on completely unseen data without per-dataset fine-tuning.
> - **Neural Baseline**: Deep learning architectures that are **not** pre-trained on large diverse corpora. Tollama includes them as reference baselines for model comparison and routing benchmarks. Behind the scenes, PatchTST and TiDE load pre-trained weights for direct inference, while N-HiTS and N-BEATSx auto-fit on the provided input data then predict — all transparently within a single forecast call.

---

## Architecture

```
┌────────────────────────────────────────────────────────┐
│  Core users: CLI / SDK / HTTP API                      │
├────────────────────────────────────────────────────────┤
│  AI Agents: MCP / A2A / LangChain / CrewAI ...         │
│  Core workflow: Preprocess -> Forecast -> Benchmark -> Route │
├────────────────────────────────────────────────────────┤
│  Tollama daemon (tollamad)                             │
│  Forecast · Benchmark · Routing · Pipeline             │
├──────┬──────┬──────┬──────┬──────┬──────┬──────────────┤
│      │ stdio JSON-lines protocol      │              │
│      ▼      ▼      ▼      ▼      ▼      ▼              │
│  14 isolated model runners (torch, timesfm, etc.)      │
│  Independent venv per family — zero dependency clash   │
└────────────────────────────────────────────────────────┘
```

Family runtimes auto-bootstrap on first use, and fall back to `uv venv` when
stdlib `venv` / `ensurepip` fails on the host interpreter.

---

## Python SDK

```python
from tollama import Tollama

t = Tollama()
result = t.forecast(
    model="chronos2",
    series={"target": [10, 11, 12, 13, 14], "freq": "D"},
    horizon=3,
)
print(result.to_df())

auto = t.auto_forecast(
    series={"target": [10, 11, 12, 13, 14], "freq": "D"},
    horizon=3,
    strategy="auto",
    mode="high_accuracy",
)
print(auto.selection.chosen_model)
```

---

## Interfaces at a Glance

| Interface      | Description                                                               |
| -------------- | ------------------------------------------------------------------------- |
| **HTTP API**   | Forecasting, routing, analysis, and ingestion. See [docs/api-reference.md](docs/api-reference.md) |
| **Python SDK** | Chainable Python workflows via the `Tollama` class.                       |
| **CLI**        | Fast execution with `tollama pull`, `tollama serve`, `tollama run`, `tollama benchmark`. |
| **Dashboard**  | Web / TUI for visualizing forecasts and model statuses (`tollama open` / `tollama dashboard`). |

---

## AI Agent Integrations

Tollama can **invoke TSFMs as tools when forecasting is part of a broader workflow.**

| Integration                       | Guide                                                   |
| --------------------------------- | ------------------------------------------------------- |
| **LangChain**                     | [docs/agent-tools.md](docs/agent-tools.md)              |
| **MCP Server**                    | [docs/agent-tools.md](docs/agent-tools.md)              |
| **CrewAI / AutoGen / Smolagents** | [docs/agent-tools.md](docs/agent-tools.md)              |
| **A2A Protocol**                  | [docs/a2a-integration.md](docs/a2a-integration.md)      |
| **OpenClaw Skill**                | [skills/tollama-forecast/SKILL.md](skills/tollama-forecast/SKILL.md) |

---

## Documentation

- **Getting Started & Workflows**
  - [Core Workflow](docs/core-workflow.md)
  - [Concrete Solution Path](docs/concrete-solution.md)
  - [Dashboard User Guide](docs/dashboard-user-guide.md)
- **References & Configurations**
  - [API Reference](docs/api-reference.md)
  - [Ollama-Workflow Parity](docs/ollama-workflow-parity.md)
  - [CLI Cheatsheet](docs/cli-cheatsheet.md)
  - [Supported Models & Setup](docs/models.md)
  - [Covariates Contract](docs/covariates.md)
  - [Preprocessing Pipeline](docs/preprocessing.md)
  - [Benchmarking Tooling](docs/benchmarking.md)
- **Agent Integrations**
  - [Agent Tool Inventory](docs/agent-tools.md)
  - [A2A Integration](docs/a2a-integration.md)
- **Troubleshooting & Development**
  - [Troubleshooting](docs/troubleshooting.md)
  - [End-to-End Testing](docs/testing.md)
  - [Contributing](CONTRIBUTING.md)

---

## Docker

Build and run:

```bash
docker compose up --build tollama
docker compose --profile gpu up --build tollama-gpu
```

---

## Roadmap

| Item                                  | Current Status                      | Goal                                                                                   |
| ------------------------------------- |:-----------------------------------:| -------------------------------------------------------------------------------------- |
| **Auto model comparison / selection** | ✅ Basic impl                        | Advanced best-model routing based on data characteristics                              |
| **Auto data preprocessing**           | ✅ Basic impl                        | Spline interpolation, smoothing, train-fit scaling, windowing via `tollama.preprocess` |
| **Fine-tuning / ensemble**            | ⚠️ Ensemble only                    | Add domain-adaptation fine-tuning workflow                                             |
| **Local + cloud execution**           | ⚠️ Dockerfile exists, local-centric | K8s manifests, docker-compose, cloud deployment guide                                  |

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for how to set up your dev environment and run local tests.

## License

[MIT](LICENSE)
