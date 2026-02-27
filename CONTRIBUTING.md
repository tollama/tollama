# Contributing to tollama

Thank you for your interest in contributing to tollama. This guide covers
development setup, testing, and the pull request process.

## Development Setup

```bash
# Clone and install in editable mode with dev dependencies
git clone https://github.com/yongchoelchoi/tollama.git
cd tollama
python -m pip install -e ".[dev]"
```

Verify the setup:

```bash
ruff check .
pytest -q
```

## Architecture Overview

```
src/tollama/
  daemon/    # HTTP API surface and runner process supervision
  runners/   # Per-family inference adapters (torch, timesfm, uni2ts, sundial, toto, mock)
  core/      # Shared schemas, protocol, registry, storage, config
  cli/       # User-facing CLI commands
  client/    # Shared HTTP client used by CLI, MCP, and SDK
  mcp/       # MCP server and tool handlers
  sdk.py     # High-level Python SDK
  skill/     # Agent framework wrappers (LangChain, CrewAI, AutoGen, smolagents)
  dashboard/ # Bundled web dashboard static assets
  tui/       # Textual TUI dashboard
```

**Key boundaries:**

- `daemon/` routes requests and manages runner processes but does not import
  heavy ML runtimes.
- `runners/` communicate with the daemon over stdio JSON lines only; they do
  not expose HTTP endpoints.
- `core/` provides shared contracts reused across all layers.

## Running Tests

Baseline tests (no optional runner dependencies required):

```bash
pytest -q
```

Focused MCP/client tests:

```bash
pytest -q tests/test_client_http.py tests/test_mcp_tools.py tests/test_mcp_entrypoint.py
```

Per-runner-family tests (install the corresponding extra first):

```bash
# torch family (Chronos-2 + Granite TTM)
python -m pip install -e ".[dev,runner_torch]"
pytest -q tests/test_torch_runner.py tests/test_chronos_adapter.py tests/test_granite_ttm_adapter.py

# timesfm family
python -m pip install -e ".[dev,runner_timesfm]"
pytest -q tests/test_timesfm_runner.py tests/test_timesfm_adapter.py

# uni2ts/moirai family
python -m pip install -e ".[dev,runner_uni2ts]"
pytest -q tests/test_uni2ts_runner.py tests/test_uni2ts_adapter.py

# sundial family
python -m pip install -e ".[dev,runner_sundial]"
pytest -q tests/test_sundial_runner.py tests/test_sundial_adapter.py

# toto family
python -m pip install -e ".[dev,runner_toto]"
pytest -q tests/test_toto_runner.py tests/test_toto_adapter.py
```

OpenClaw skill validation:

```bash
bash scripts/validate_openclaw_skill_tollama_forecast.sh
```

## Adding a New Runner Family

Use the built-in scaffolding tool:

```bash
tollama dev scaffold <family_name>          # create runner skeleton
tollama dev scaffold <family_name> --register  # create and register in pyproject.toml
```

This generates the boilerplate runner directory under `src/tollama/runners/`.
See existing runners for reference on the stdio protocol contract.

## Pull Request Guidelines

- Keep PRs focused with one coherent purpose per change.
- Include tests for behavior changes and bug fixes.
- Run `ruff check .` and `pytest -q` locally before submitting.
- Update relevant documentation when behavior, APIs, or defaults change
  (README.md, docs/, roadmap status markers).
- Note any API, CLI, protocol, dependency, or config compatibility impact.

## Branching Strategy

Use short-lived feature branches and open a PR against `main`:

| Prefix | When to use |
|---|---|
| `feature/` | New features or significant additions |
| `fix/` | Bug fixes |
| `docs/` | Documentation-only changes |
| `chore/` | Maintenance tasks (deps, CI, tooling) |

Examples: `feature/add-toto-runner`, `fix/path-traversal-ingest`, `docs/cli-cheatsheet`.

## CI Pipeline

The CI workflow (`.github/workflows/ci.yml`) runs on every push and PR:

1. **Ruff** — linting and import ordering (`ruff check .`)
2. **Validate OpenClaw skill** — `scripts/validate_openclaw_skill_tollama_forecast.sh`
3. **Pytest** — baseline test suite with remote registry validation (`TOLLAMA_VALIDATE_REGISTRY_REMOTE=1 pytest -q`)
4. **Notebook execution** (Python 3.11 only) — runs tutorial notebooks with a 15-minute timeout

All four steps must pass before a PR can be merged. The matrix runs on Python 3.11, 3.12, and 3.13.

## Code Style

- Python 3.11+ target.
- [Ruff](https://docs.astral.sh/ruff/) for linting (rules: E, F, I, UP; line
  length 100).
- Type hints on all public functions.
- Pydantic models with strict validation for API schemas.
- Lowercase `snake_case` for modules and functions; explicit names for
  model/exception classes (e.g., `ForecastRequest`, `RunnerUnavailableError`).

## Reporting Security Issues

Please do **not** open public issues for security vulnerabilities. See
[SECURITY.md](SECURITY.md) for responsible disclosure instructions.

## License

By contributing, you agree that your contributions will be licensed under the
MIT License that covers this project.
