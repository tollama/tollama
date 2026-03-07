# Dashboard Improvement Spec (v1)

## Goal
Reduce context bloat and improve operational visibility for app stability, CI triage, and agent orchestration.

## Primary KPIs
- Time-to-detect failure (TTD)
- Time-to-next-action (TTNA)
- Compact pressure reduction (chat/session token churn)

## Layout

### A. Top Status Strip (sticky)
1. Active Tasks: running / queued / failed
2. PR CI Health: green / yellow / red summary
3. Token Pressure:
   - compact ratio
   - msgs/hour
4. Last Successful Stability Run timestamp

### B. Workload Board (Main + Sub-agents)
Columns:
- Planned
- Running
- Blocked
- Done

Card schema:
- owner: main | subagent:<id>
- title
- repo/branch
- issue/pr
- status
- eta
- blocker (single line)
- next_action (single line)

### C. Stability Panel
Pinned suites:
- tests/test_daemon_api.py
- tests/test_runner_manager.py
- tests/test_runtime_bootstrap.py

For each suite:
- last_result (pass/fail)
- pass_rate_24h
- duration
- last_failure_reason (1 line)
- rerun button

### D. CI Triage Panel
- Collapse duplicate/stale runs by SHA
- Show only latest relevant failed run per workflow
- Quick actions:
  - Open logs
  - Rerun failed jobs
  - Create fix task

### E. Compact-safe Activity Feed
Event format:
`[time] [scope] [result] [next]`

Examples:
- `23:34 PR#64 checks fail(py3.12) -> adjust deps`
- `23:36 stability tests pass -> merge candidate`

Rules:
- max 120 chars/event
- no raw log dump
- dedupe repeated failures

### F. “What should I do now?” Assistant Box
Return exactly 3 ranked actions:
1) Highest-impact blocker
2) Fastest merge path
3) Long-running/idle task intervention

Each recommendation includes:
- reason (<= 1 line)
- expected gain
- one-click action

---

## Data Contract (minimal JSON)
```json
{
  "top_strip": {
    "active": {"running": 0, "queued": 0, "failed": 0},
    "ci_health": "green|yellow|red",
    "token_pressure": {"compact_ratio": 0.0, "messages_per_hour": 0},
    "last_successful_stability_run": "ISO8601"
  },
  "workload": [
    {
      "id": "task-1",
      "owner": "main",
      "title": "Issue #61 calibration",
      "repo": "tollama/tollama",
      "branch": "feat/issue-61-calibration-unblock",
      "issue": 61,
      "pr": 64,
      "status": "Running",
      "eta": "15m",
      "blocker": "py3.12 CI fail",
      "next_action": "extract failed test and patch"
    }
  ],
  "stability": [
    {
      "suite": "tests/test_daemon_api.py",
      "last_result": "pass",
      "pass_rate_24h": 1.0,
      "duration_sec": 42,
      "last_failure_reason": null
    }
  ],
  "ci_triage": [
    {
      "workflow": "checks (3.12)",
      "sha": "<sha>",
      "status": "failed",
      "run_url": "https://...",
      "stale": false
    }
  ],
  "activity": [
    "23:34 PR#64 checks fail(py3.12) -> adjust deps"
  ],
  "next_actions": [
    {
      "rank": 1,
      "title": "Fix py3.12 check failure",
      "reason": "blocks merge",
      "expected_gain": "PR mergeability",
      "action": "open_latest_failed_log"
    }
  ]
}
```

## Implementation Phases

### Phase 1 (1 day)
- Add top strip + compact-safe feed
- Add stability panel with last result only

### Phase 2 (1–2 days)
- Workload board with main/sub-agent cards
- CI dedupe by SHA and latest-run selection

### Phase 3 (1 day)
- “What should I do now?” ranking logic
- one-click actions

## Acceptance Criteria
- Dashboard shows active tasks in <2s refresh
- Duplicate CI failures reduced by >=50% in view
- No verbose logs in feed
- Operator can identify next action in <=10s
