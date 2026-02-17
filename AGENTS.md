# AGENTS

## Project Guidance

- Keep the package under `src/tollama/` and prefer small, focused modules.
- Keep `daemon/core/cli` dependencies lightweight; avoid importing heavy ML runtimes there.
- Heavy model dependencies are allowed in runner families via optional extras:
  `runner_torch`, `runner_timesfm`, `runner_uni2ts`, `runner_sundial`, `runner_toto`.
- Do not leak runner-only dependencies into `daemon/core/cli`.
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
  - Keep CLI command names short and stable (`tollama`, `tollamad`,
    `tollama-runner-mock`, `tollama-runner-torch`, `tollama-runner-timesfm`,
    `tollama-runner-uni2ts`, `tollama-runner-sundial`, `tollama-runner-toto`).
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
- Verify covariates contract correctness (`past_covariates`, `future_covariates`,
  length/type rules, known-future key consistency).
- Verify `covariates_mode` parity:
  - `best_effort` drops unsupported covariates with `warnings`
  - `strict` rejects unsupported covariates with HTTP `400`
- Verify keep-alive and loaded-model lifecycle behavior (`keep_alive`, unload, `/api/ps`).
- Verify `/api/info` diagnostics consistency (runner/model capability visibility and redaction).
- Confirm lint and test commands pass locally before approving.
- Require tests for behavior changes and bug fixes.
- Check HTTP status mapping for failure paths (400/502/503) where relevant.
- Check CLI behavior for clear errors and stable defaults.
- Flag unnecessary dependency growth, especially heavyweight runtime packages.
- Call out unclear ownership boundaries or architectural drift early.
- Require roadmap/TODO status sync when behavior or capabilities change.

## How To Run Checks Locally

Baseline checks:

```bash
python -m pip install -e ".[dev]"
ruff check .
pytest -q
```

If touching runner family code, install the corresponding extra and run focused tests:

```bash
# torch family (Chronos + Granite TTM)
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

## Documentation Sync

- When behavior, APIs/contracts, runner capabilities, or defaults change, update the
  relevant docs in the same PR:
  - `README.md`
  - `docs/covariates.md`
  - `roadmap.md`
  - `todo-list.md`
- Minimum expectation:
  - status markers stay consistent with implementation (`[x]`, `[~]`, `[ ]`)
  - capability matrices and endpoint/command inventories match current behavior
