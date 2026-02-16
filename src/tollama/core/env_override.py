"""Environment override helpers for pull-time process settings."""

from __future__ import annotations

import os
import threading
from collections.abc import Iterator, Mapping
from contextlib import contextmanager

_ENV_OVERRIDE_LOCK = threading.Lock()


@contextmanager
def set_env_temporarily(mapping: Mapping[str, str | None]) -> Iterator[None]:
    """Temporarily set environment variables and restore them afterwards."""
    previous_values: dict[str, str | None] = {}
    with _ENV_OVERRIDE_LOCK:
        for key, value in mapping.items():
            previous_values[key] = os.environ.get(key)
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

        try:
            yield
        finally:
            for key, previous in previous_values.items():
                if previous is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = previous
