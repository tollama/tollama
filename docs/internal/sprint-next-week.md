# Sprint Plan: Next Week (2026-03-09)

## WIP policy (guardrail)

- Keep **1-2 active PRs max** at any time.
- Do not open a third PR until one active PR is merged or closed.
- Keep each PR scoped to a single issue goal.

## Prioritized mini-backlog (max 3)

### P1 — Benchmark refresh + routing calibration
Issue: #49

- Refresh cross-model benchmark outputs.
- Classify failures (`DEPENDENCY_GATED` vs regression classes).
- Update routing recommendation based on measured results.

### P2 — CI reliability hardening
Issue: #50

- Harden daemon readiness probe checks.
- Guard unsupported-family error contract in tests.
- Keep triage guidance explicit in troubleshooting docs.

### P3 — Post-release quality pass
Issue: #51

- Run targeted smoke and regression checks for TSFM runners.
- Update release-readiness notes with command + outcome evidence.

## Execution order

1. #49
2. #50
3. #51

## Definition of done (week close)

- Backlog remains capped at 3 priority items.
- No stale open PRs older than 3 days without update.
- CI green on merged PRs tied to this sprint.
