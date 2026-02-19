"""Module entrypoint for the tollama MCP server."""

from __future__ import annotations

import sys


def _run_server() -> None:
    from .server import run_server

    run_server()


def main() -> None:
    try:
        _run_server()
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":  # pragma: no cover
    main()
