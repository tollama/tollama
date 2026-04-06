# Core Solution Expected Output

This directory contains an illustrative `Tollama Core` output bundle for the
checked-in hourly demand concrete-solution path.

Use it when you want to understand what a successful run should look like
before running:

```bash
USE_CHECKED_IN_INPUT=1 MODELS=mock bash examples/core_concrete_solution_demo.sh
```

The bundle mirrors the same front-door contract used by `tollama benchmark`:

- `result.json`
- `routing.json`
- `leaderboard.csv`
- `summary.md`

The values here are example values for documentation and onboarding. They are
not meant to be treated as a recorded production benchmark run.
