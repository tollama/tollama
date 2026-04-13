# ADR 0001: NDJSON Stdio Runner Protocol

## Status
Accepted

## Context
Tollama needs a lightweight, deterministic boundary between the daemon and runner families.
That boundary must avoid importing heavyweight ML runtimes into `daemon/core/cli`, remain easy
to test, and support long-lived subprocess supervision.

## Decision
Daemon-to-runner communication uses newline-delimited JSON messages over stdio, with
`ProtocolRequest` and `ProtocolResponse` as the canonical contract in `src/tollama/core/protocol.py`.
Runners stay process-local and do not expose HTTP endpoints.

## Consequences
- Keeps daemon and runner ownership boundaries explicit.
- Enables focused contract tests and protocol round-trip property tests.
- Requires careful stdout/stderr separation and supervision logic, including stderr draining.
