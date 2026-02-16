# AGENTS

## Project Guidance

- Keep the package under `src/tollama/` and prefer small, focused modules.
- Favor lightweight dependencies; avoid heavy ML stacks unless explicitly approved.
- Keep behavior deterministic and testable. New behavior should include tests.
- Prefer clear type hints and straightforward interfaces over premature abstraction.

## Architecture

- `src/tollama/daemon/` owns the public HTTP API surface and runner supervision.
- `src/tollama/runners/` owns runner implementations and stdio protocol handling.
- `src/tollama/core/` owns shared request/response schemas and protocol primitives.
- `src/tollama/cli/` owns user-facing command entrypoints and daemon HTTP client code.
- `tests/` contains unit and integration coverage for all layers.
- Tooling and quality gates are defined in `pyproject.toml` and `.github/workflows/`.

## Repository Conventions

- Naming:
  - Use lowercase snake_case for modules and functions.
  - Keep model/exception class names explicit (e.g., `ForecastRequest`, `RunnerUnavailableError`).
  - Keep CLI command names short and stable (`tollama`, `tollamad`, `tollama-runner-mock`).
- Boundaries:
  - Daemon code should not implement model logic; it routes requests and maps errors.
  - Runner code should not expose HTTP endpoints; communication is stdio JSON lines only.
  - Shared schemas/protocol types should live in `core` and be reused across daemon/runners/tests.
  - Avoid cross-layer shortcuts that bypass `core` contracts.
- Change scope:
  - Prefer additive, focused changes over broad refactors.
  - Keep protocol/API compatibility in mind; document intentional breaking changes.

## Conventions

- Python target is 3.11+.
- Use Ruff for linting (`ruff check .`).
- Use Pytest for tests (`pytest -q`).
- Keep PRs minimal, with one coherent purpose per change.

## Review Guidelines (Codex GitHub Review)

- Prioritize correctness, regressions, and edge-case behavior over style-only feedback.
- Verify daemon <-> runner contracts (`core.schemas`, `core.protocol`) remain consistent.
- Confirm lint and test commands pass locally before approving.
- Require tests for behavior changes and bug fixes.
- Check HTTP status mapping for failure paths (400/502/503) where relevant.
- Check CLI behavior for clear errors and stable defaults.
- Flag unnecessary dependency growth, especially heavyweight runtime packages.
- Call out unclear ownership boundaries or architectural drift early.

## How To Run Checks Locally

```bash
python -m pip install -e ".[dev]"
ruff check .
pytest -q
```
