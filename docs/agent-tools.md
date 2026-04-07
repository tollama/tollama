# Agent Tool Inventory

This page is the canonical inventory for agent-facing tool surfaces in
Tollama.

Source of truth:

- MCP registration: `src/tollama/mcp/server.py`
- LangChain tool set: `src/tollama/skill/langchain.py`
- CrewAI / AutoGen / smolagents shared specs:
  `src/tollama/skill/framework_common.py`

This page focuses on tool surfaces. A2A is an operation-oriented protocol, not
a tool registry; its HTTP routes remain documented in `docs/api-reference.md`.

## Surface Summary

| Surface | Entry point | Current scope | Notes |
|---|---|---|---|
| MCP | `tollama-mcp` / `python -m tollama.mcp` | 22 registered tools | Widest surface; includes operational and trust/XAI tools |
| LangChain | `get_tollama_tools(...)` | 15 `BaseTool` wrappers | Dedicated LangChain classes with async `_arun` paths |
| CrewAI / AutoGen / smolagents | `get_crewai_tools(...)`, `get_autogen_tool_specs(...)`, `get_smolagents_tools(...)` | 15 shared specs | All reuse `get_agent_tool_specs(...)` from `framework_common.py` |

## Cross-Surface Matrix

| Tool | MCP | LangChain | CrewAI / AutoGen / smolagents |
|---|---|---|---|
| `tollama_health` | Yes | Yes | Yes |
| `tollama_models` | Yes | Yes | Yes |
| `tollama_show` | Yes | Yes | Yes |
| `tollama_pull` | Yes | Yes | Yes |
| `tollama_forecast` | Yes | Yes | Yes |
| `tollama_auto_forecast` | Yes | Yes | Yes |
| `tollama_analyze` | Yes | Yes | Yes |
| `tollama_generate` | Yes | Yes | Yes |
| `tollama_counterfactual` | Yes | Yes | Yes |
| `tollama_scenario_tree` | Yes | Yes | Yes |
| `tollama_report` | Yes | Yes | Yes |
| `tollama_what_if` | Yes | Yes | Yes |
| `tollama_pipeline` | Yes | Yes | Yes |
| `tollama_compare` | Yes | Yes | Yes |
| `tollama_recommend` | Yes | Yes | Yes |
| `tollama_explain` | Yes | - | - |
| `tollama_trust_score` | Yes | - | - |
| `tollama_model_card` | Yes | - | - |
| `tollama_gate_decision` | Yes | - | - |
| `tollama_batch_analyze` | Yes | - | - |
| `tollama_alerts_configure` | Yes | - | - |
| `tollama_alerts_check` | Yes | - | - |

## Why The Surfaces Differ

- MCP is the richest surface. It is the only agent integration that currently
  exposes `tollama_explain` plus trust/XAI tools
  (`trust_score`, `model_card`, `gate_decision`, `batch_analyze`,
  `alerts_configure`, `alerts_check`).
- LangChain and the shared CrewAI / AutoGen / smolagents subset now expose the
  same 15-tool forecasting, structured-analysis, and basic lifecycle layer.
- CrewAI, AutoGen, and smolagents stay in lockstep because their wrappers are
  all generated from `framework_common.py`.

## Remaining Differences

- `tollama_explain` stays MCP-only for now because it is not a model metadata
  helper. It is an XAI decision-explanation tool backed by
  `/api/xai/explain-decision` and expects a composite payload
  (`forecast_result`, optional `eval_result`, `trust_context`, `trust_payload`,
  and explanation options).
- Trust/XAI agent tools stay MCP-only for now because they rely on a separate
  trust-intelligence contract under `src/tollama/xai/` rather than the shared
  forecasting/analysis request models used by LangChain and
  `framework_common.py`.
- `tollama_alerts_configure` and `tollama_alerts_check` are also operationally
  stateful: they manage or evaluate alert thresholds, and `alerts_configure`
  may trigger webhook-oriented workflows. Keeping them MCP-only avoids
  broadening the default wrapper surface before a framework-neutral trust/XAI
  contract exists.

## Promotion Criteria

- Promote `tollama_explain` only if Tollama adopts a framework-neutral XAI
  request/response contract under `src/tollama/core/` or a dedicated shared
  `skill_xai` layer.
- Promote trust/XAI tools only as an intentional bundle, not one by one, so
  LangChain and `framework_common.py` do not drift into a partial trust surface.
- Keep `docs/agent-tools.md` as the canonical place to record whether a tool is
  forecasting/shared-surface or MCP-only-by-design.

## Update Rules

- When adding or removing an MCP tool, update this page together with
  `src/tollama/mcp/server.py`.
- When changing LangChain wrapper coverage, update this page together with
  `src/tollama/skill/langchain.py`.
- When changing CrewAI / AutoGen / smolagents coverage, update this page
  together with `src/tollama/skill/framework_common.py`.
