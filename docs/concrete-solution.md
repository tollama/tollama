# Concrete Solution

This is the concrete `Tollama Core` solution path for the current phase:

`benchmark-backed hourly demand forecasting for operations teams`

The point is not to show every possible feature. The point is to prove one
operator workflow end to end:

1. prepare a real operational time-series workload
2. benchmark a curated set of strong forecast models
3. generate reusable evidence artifacts
4. promote that evidence into runtime routing defaults

## Why This Is The Right Core Story

This solution uses the strongest parts of `Tollama Core` together:

- preprocessing for imperfect series
- unified inference across multiple model families
- benchmark artifacts for model comparison
- routing defaults derived from benchmark evidence

It also stays cleanly inside the `Core` boundary.
No Trust packaging, policy review layer, or domain-agent setup is required to
understand the value.

## Target User

Use this solution when the user looks like:

- an operations team
- a planning team
- a quantitative team with recurring forecast cycles

The question they need answered is:

> Which forecast model should be the default for this workload, and what is the
> evidence for that choice?

## Workload Choice

For this phase, the hero workload is:

- `next-24h hourly demand forecasting`

Dataset policy:

- preferred hero dataset: `pjm_hourly_energy`
- public fallback dataset: `m4_daily`

The repo already contains the real-data harness and dataset catalog under
`scripts/e2e_realdata/`.

## Curated Model Set

Do not start with all 14 models.

Start with this benchmark set:

- `chronos2`
- `granite-ttm-r2`
- `timesfm-2.5-200m`
- `moirai-2.0-R-small`

Use `mock` only for smoke tests or zero-download local validation.

## Install

```bash
python -m pip install "tollama[eval,preprocess]"
```

## Start The Daemon

```bash
tollama serve
```

## Pull The Curated Models

```bash
tollama pull chronos2 --accept-license --no-stream
tollama pull granite-ttm-r2 --accept-license --no-stream
tollama pull timesfm-2.5-200m --accept-license --no-stream
tollama pull moirai-2.0-R-small --accept-license --no-stream
```

## Run The Real-Data Core Solution Path

Use the existing real-data harness with the curated model set.

```bash
python scripts/e2e_realdata/run_tsfm_realdata.py \
  --mode local \
  --model chronos2,granite-ttm-r2,timesfm-2.5-200m,moirai-2.0-R-small \
  --gate-profile strict \
  --allow-kaggle-fallback \
  --base-url http://127.0.0.1:11435 \
  --output-dir artifacts/realdata/core-solution
```

This command keeps the story narrow:

- hourly-demand-style real data when available
- open fallback when credentials are missing
- one curated benchmark set instead of the full model matrix

## What To Read In The Output

The real-data harness writes:

- `result.json`
- `summary.json`
- `summary.md`
- `raw/`

The standard Core benchmark flow writes:

- `result.json`
- `routing.json`
- `leaderboard.csv`

The concrete solution is complete only when the user can move from benchmark
evidence into routing action.

## Required Operator Answer

At the end of the workflow, the operator should be able to answer:

- which model is the `default`
- which model is the `fast_path`
- which model is the `high_accuracy` path
- what benchmark evidence justified those assignments
- what caveats still apply to the dataset or benchmark design

If the output does not answer those questions, the workflow is still a toolkit,
not yet a concrete solution.

## Relationship To Other Docs

- Use `docs/core-workflow.md` for the canonical Core product path.
- Use this document for the opinionated real-data solution path.
- Use `docs/tsfm-routing-defaults.md` for routing-manifest semantics.
- Use `docs/how-to-run.md` for broader installation and real-data harness details.
