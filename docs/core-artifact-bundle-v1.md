# Core Artifact Bundle V1

This document freezes the minimum contract for the `Tollama Core` benchmark
bundle written by:

```bash
tollama benchmark ... --output <dir>
```

The canonical bundle is:

- `result.json`
- `routing.json`
- `leaderboard.csv`

The operator-facing companion file `summary.md` may also be written.
The legacy compatibility file `benchmark_<fingerprint>.json` may still be
written, but it is not the primary contract.

## Stability Rule

For `v1`, these artifact names and minimum required fields should be treated as
stable.

Changes should be:

- additive when possible
- backward-compatible for readers of `v1`
- explicitly versioned if a breaking change becomes unavoidable

## `result.json`

Purpose:

- canonical Core benchmark payload
- routing recommendation source
- artifact index for the bundle

Minimum required top-level fields:

- `artifact_kind`
- `schema_version`
- `generated_at`
- `run_id`
- `source`
- `dataset_fingerprint`
- `horizon`
- `num_folds`
- `metric_names`
- `quality_metric_priority`
- `models`
- `leaderboard`
- `learned_weights`
- `routing_recommendation`
- `artifact_mapping`

Required values for `v1`:

- `artifact_kind = "tollama_core_benchmark"`
- `schema_version = 1`
- `source = "tollama.core.benchmark"`

Each entry in `models` must include:

- `model`
- `metrics`
- `latency_ms`
- `folds_evaluated`
- `warnings`
- `learned_weight`

`routing_recommendation` must include:

- `default`
- `fast_path`
- `high_accuracy`
- `policy`
- `ranking`
- `caveats`

`artifact_mapping` must include:

- `result_json`
- `routing_manifest`
- `leaderboard_csv`
- `operator_summary_md`
- `legacy_summary_json`
- `rich_eval_artifacts`

## `routing.json`

Purpose:

- runtime routing manifest
- stable handoff for `tollama routing apply`

Minimum required top-level fields:

- `version`
- `generated_at`
- `run_id`
- `source`
- `routing`
- `policy`
- `caveats`

Required values for `v1`:

- `version = 1`
- `source = "tollama.core.benchmark"`

`routing` must include:

- `default`
- `fast_path`
- `high_accuracy`

## `leaderboard.csv`

Purpose:

- lightweight human-readable export
- quick sharing surface for model comparison

Minimum required columns:

- `rank`
- `model`
- each metric listed in `metric_names`
- `latency_ms`
- `folds_evaluated`
- `learned_weight`

Rows should be sorted quality-first according to the benchmark leaderboard logic.

## `summary.md`

Purpose:

- operator-facing answer layer
- quick explanation of which model should run in each routing lane

This file is a human-readable companion, not the primary machine contract.

Minimum contents:

- recommended `default`
- recommended `fast_path`
- recommended `high_accuracy`
- reason for each lane
- policy summary
- caveats

## Relationship To `tollama-eval`

`Tollama Core` owns the thin bundle above.

Use `tollama-eval` when you need:

- richer reports
- deeper benchmark details
- multi-artifact evaluation output beyond the Core bundle

The Core bundle should remain thin and operator-friendly.
It is the front-door artifact set, not the full evaluation warehouse.

## Concrete Solution Requirement

The concrete-solution path should rely on this bundle directly.

That means:

- the hero demo should end in `result.json` and `routing.json`
- the runtime routing step should read `result.json` or `routing.json` without adapters
- future `Core + Trust` attachment should reference this bundle rather than invent a second front-door contract
