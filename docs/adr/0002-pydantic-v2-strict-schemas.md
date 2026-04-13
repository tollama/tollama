# ADR 0002: Pydantic v2 Strict Schemas

## Status
Accepted

## Context
Tollama has multiple trust-sensitive boundaries: HTTP requests, daemon/runner protocol payloads,
and reusable domain schemas shared across tests and SDK code. Loose coercion increases the chance
of silent contract drift.

## Decision
Core request/response and protocol models use Pydantic v2 with strict validation enabled. Schema
validation is treated as the canonical source of truth for API and protocol compatibility.

## Consequences
- Contracts fail fast instead of silently coercing ambiguous input.
- OpenAPI generation and contract tests share the same schema source.
- Type-checking and property-based tests have clearer invariants to build on.
