# AGENTS

## Project Guidance

- Keep the package under `src/tollama/` and prefer small, focused modules.
- Favor lightweight dependencies; avoid heavy ML stacks unless explicitly approved.
- Keep behavior deterministic and testable. New behavior should include tests.
- Prefer clear type hints and straightforward interfaces over premature abstraction.

## Architecture

- `src/tollama/` is the application/package root.
- `tests/` contains test coverage for package behavior.
- Tooling and quality gates are defined in `pyproject.toml` and CI workflow files.

## Conventions

- Python target is 3.11+.
- Use Ruff for linting (`ruff check .`).
- Use Pytest for tests (`pytest -q`).
- Keep PRs minimal, with one coherent purpose per change.

## Review Guidelines

- Prioritize correctness, regressions, and edge-case behavior over style-only feedback.
- Confirm lint and test commands pass locally before approving.
- Require tests for behavior changes and bug fixes.
- Flag unnecessary dependency growth, especially heavyweight runtime packages.
- Call out unclear ownership boundaries or architectural drift early.
