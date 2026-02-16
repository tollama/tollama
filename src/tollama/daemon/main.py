"""CLI entrypoint for running the tollama daemon."""

from __future__ import annotations

import os

import uvicorn


def main() -> int:
    """Run the HTTP daemon with uvicorn."""
    host = os.environ.get("TOLLAMA_HOST", "127.0.0.1")
    port_value = os.environ.get("TOLLAMA_PORT", "11435")
    log_level = os.environ.get("TOLLAMA_LOG_LEVEL", "info")

    try:
        port = int(port_value)
    except ValueError as exc:
        raise ValueError(f"invalid TOLLAMA_PORT: {port_value!r}") from exc

    uvicorn.run(
        "tollama.daemon.app:app",
        host=host,
        port=port,
        log_level=log_level,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
