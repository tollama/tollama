# ADR 0003: Connector Plugin Architecture

## Status
Accepted

## Context
Tollama integrates optional data connectors with very different runtime dependencies and failure
profiles. The daemon should not hard-code every backend or assume all optional dependencies are
installed.

## Decision
Connectors are registered through the connector registry and loaded by backend name. Optional
dependencies stay isolated behind connector-specific extras, and connector implementations provide
small, testable interfaces rooted in `DataConnector`.

## Consequences
- Optional backends remain additive and can fail independently.
- Security and contract tests can target connector-specific behavior like SQL identifier handling.
- Readiness stays focused on daemon-owned checks instead of pretending every connector is globally
  managed by the daemon.
