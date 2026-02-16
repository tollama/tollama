# tollama

Lightweight Python project skeleton for the `tollama` codebase.

## Overview

This repository is intentionally minimal. It provides:
- PEP 621 packaging via `pyproject.toml`
- `src/` project layout
- Fast linting with Ruff
- Testing with Pytest
- CI checks on push and pull request

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"

# Terminal 1: run daemon (default http://127.0.0.1:11435)
tollama serve

# Terminal 2: request a forecast from example payload
tollama forecast --model mock --input examples/request.json

ruff check .
pytest -q
```

## Project Structure

```text
.
├── src/
│   └── tollama/
├── tests/
└── .github/workflows/
```

## Next Steps

- Add core package modules under `src/tollama/`.
- Replace placeholders with concrete product and architecture docs.
