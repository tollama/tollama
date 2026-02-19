# CLAUDE

## Project Context

Tollama is a local-first time-series forecasting daemon with Ollama-compatible APIs.

Core layers:
- `src/tollama/daemon`: HTTP API surface and runner lifecycle
- `src/tollama/core`: shared schemas/contracts
- `src/tollama/runners`: per-family inference adapters
- `src/tollama/cli`: user-facing command entrypoints
- `src/tollama/client`: shared HTTP client used by CLI/MCP/tooling
- `src/tollama/mcp`: MCP server and tool handlers

## Common Commands

```bash
ruff check .
pytest -q
```

Focused checks:

```bash
pytest -q tests/test_client_http.py tests/test_mcp_tools.py tests/test_mcp_entrypoint.py
bash scripts/validate_openclaw_skill_tollama_forecast.sh
```

## Skills and Integration Paths

OpenClaw skill path:
- `skills/tollama-forecast/`

MCP entrypoint:
- `tollama-mcp` (requires `pip install "tollama[mcp]"`)

Claude Desktop helper:
- `scripts/install_mcp.sh`
