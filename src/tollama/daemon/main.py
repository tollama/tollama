"""CLI entrypoint for running the tollama daemon."""

from __future__ import annotations

import os

import uvicorn

DEFAULT_DAEMON_HOST = "127.0.0.1"
DEFAULT_DAEMON_PORT = 11435


def _parse_host_port(value: str) -> tuple[str, int]:
    raw = value.strip()
    host, separator, port_value = raw.rpartition(":")
    host = host.strip()
    port_value = port_value.strip()
    if not separator or not host or not port_value:
        raise ValueError(f"invalid TOLLAMA_HOST: {value!r}")
    return host, _parse_port(port_value, env_name="TOLLAMA_HOST", raw_value=value)


def _parse_port(value: str, *, env_name: str, raw_value: str) -> int:
    try:
        port = int(value)
    except ValueError as exc:
        raise ValueError(f"invalid {env_name}: {raw_value!r}") from exc
    if not 1 <= port <= 65535:
        raise ValueError(f"invalid {env_name}: {raw_value!r}")
    return port


def _resolve_bind() -> tuple[str, int]:
    host_port = os.environ.get("TOLLAMA_HOST")
    if host_port is not None:
        return _parse_host_port(host_port)

    port_value = os.environ.get("TOLLAMA_PORT")
    if port_value is None:
        return DEFAULT_DAEMON_HOST, DEFAULT_DAEMON_PORT
    port = _parse_port(port_value, env_name="TOLLAMA_PORT", raw_value=port_value)
    return DEFAULT_DAEMON_HOST, port


def main() -> int:
    """Run the HTTP daemon with uvicorn."""
    host, port = _resolve_bind()
    log_level = os.environ.get("TOLLAMA_LOG_LEVEL", "info")
    previous_binding = os.environ.get("TOLLAMA_EFFECTIVE_HOST_BINDING")
    os.environ["TOLLAMA_EFFECTIVE_HOST_BINDING"] = f"{host}:{port}"

    try:
        uvicorn.run(
            "tollama.daemon.app:app",
            host=host,
            port=port,
            log_level=log_level,
        )
    finally:
        if previous_binding is None:
            os.environ.pop("TOLLAMA_EFFECTIVE_HOST_BINDING", None)
        else:
            os.environ["TOLLAMA_EFFECTIVE_HOST_BINDING"] = previous_binding
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
