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

## MCP Implementation Details

Tool modules:
- `src/tollama/mcp/schemas.py`: strict tool input schemas
- `src/tollama/mcp/tools.py`: tool handlers + daemon client calls
- `src/tollama/mcp/server.py`: FastMCP registration + error envelope mapping

Exposed tools:
- `tollama_health`
- `tollama_models`
- `tollama_forecast` (non-streaming)
- `tollama_pull`
- `tollama_show`

Error categories (exit-code aligned):
- `INVALID_REQUEST` (`2`)
- `DAEMON_UNREACHABLE` (`3`)
- `MODEL_MISSING` (`4`)
- `LICENSE_REQUIRED` / `PERMISSION_DENIED` (`5`)
- `TIMEOUT` (`6`)
- `INTERNAL_ERROR` (`10`)

Defaults:
- base URL: `http://localhost:11435`
- timeout: `10` seconds
- per-tool override args: `base_url`, `timeout`
