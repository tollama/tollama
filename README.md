# tollama

Lightweight Python project skeleton for the `tollama` codebase.

## Overview

This repository provides a minimal but structured forecasting stack:
- PEP 621 packaging via `pyproject.toml`
- `src/` project layout
- FastAPI daemon (`tollamad`) exposing the public HTTP API
- Runner processes (currently mock) connected over stdio JSON lines
- Typer CLI (`tollama`) for Ollama-style model lifecycle and forecasting
- Ruff + Pytest + CI checks

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"

# Terminal 1: run daemon (default http://localhost:11434)
tollama serve

# Terminal 2: Ollama-style lifecycle
tollama list
tollama pull mock
tollama show mock
tollama run mock --input examples/request.json --no-stream
tollama ps
tollama rm mock

# Check daemon version endpoint
curl http://localhost:11434/api/version

ruff check .
pytest -q
```

## Daemon Base URL and Host Config

- Default CLI base URL is `http://localhost:11434`.
- Ollama-style API base URL is `http://localhost:11434/api`.
- Configure bind host and port with `TOLLAMA_HOST` in `host:port` format.
  - Example: `TOLLAMA_HOST=0.0.0.0:11434 tollamad`
- Primary lifecycle endpoints are under `/api/*`.
  - `GET /api/tags`
  - `GET /api/ps`
  - `POST /api/show`
  - `POST /api/pull`
  - `DELETE /api/delete`
  - `POST /api/forecast`
- Existing forecast and health endpoints remain under `/v1/*` for backward compatibility.
  - `GET /v1/health`
  - `POST /v1/forecast`

## Architecture

- `tollama.daemon`: Public API layer (`/api/*`, `/v1/health`, `/v1/forecast`) and runner supervision.
- `tollama.runners`: Runner implementations that speak newline-delimited JSON over stdio.
- `tollama.core`: Canonical data schemas (`ForecastRequest`, `ForecastResponse`) and protocol helpers.
- `tollama.cli`: User CLI commands for serving and sending forecast requests.

### Daemon <-> Runner Protocol

- Transport: newline-delimited JSON over stdio.
- Request shape: `{"id":"...","method":"...","params":{...}}`
- Response shape: `{"id":"...","result":{...}}` or `{"id":"...","error":{"code":...,"message":"..."}}`
- Current mock runner methods: `hello`, `forecast`.

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
│       ├── cli/
│       ├── core/
│       ├── daemon/
│       └── runners/
├── examples/
├── tests/
└── .github/workflows/
```
