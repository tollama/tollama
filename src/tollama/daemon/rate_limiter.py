"""Token-bucket rate limiter utilities for daemon endpoints."""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass
from threading import Lock

_RATE_PER_MINUTE_ENV = "TOLLAMA_RATE_LIMIT_PER_MINUTE"
_BURST_ENV = "TOLLAMA_RATE_LIMIT_BURST"


@dataclass(slots=True)
class _TokenBucket:
    tokens: float
    last_refill_s: float


class TokenBucketRateLimiter:
    """In-memory token bucket keyed by API key id."""

    def __init__(
        self,
        *,
        tokens_per_second: float,
        burst: int,
    ) -> None:
        if tokens_per_second <= 0:
            raise ValueError("tokens_per_second must be > 0")
        if burst <= 0:
            raise ValueError("burst must be > 0")

        self._tokens_per_second = tokens_per_second
        self._burst = burst
        self._lock = Lock()
        self._buckets: dict[str, _TokenBucket] = {}

    def allow(self, *, key_id: str, now_s: float | None = None) -> bool:
        """Return whether a request is allowed for the provided key."""
        normalized_key = key_id.strip() or "anonymous"
        current_s = time.monotonic() if now_s is None else now_s

        with self._lock:
            bucket = self._buckets.get(normalized_key)
            if bucket is None:
                bucket = _TokenBucket(tokens=float(self._burst), last_refill_s=current_s)
                self._buckets[normalized_key] = bucket

            elapsed = max(current_s - bucket.last_refill_s, 0.0)
            replenished = bucket.tokens + elapsed * self._tokens_per_second
            bucket.tokens = min(float(self._burst), replenished)
            bucket.last_refill_s = current_s

            if bucket.tokens < 1.0:
                return False

            bucket.tokens -= 1.0
            return True


def create_rate_limiter_from_env(
    *,
    env: Mapping[str, str] | None = None,
) -> TokenBucketRateLimiter | None:
    """Create a rate limiter when environment configuration is present."""
    source = env if env is not None else {}
    rate_raw = source.get(_RATE_PER_MINUTE_ENV)
    if rate_raw is None:
        return None

    normalized_rate = rate_raw.strip()
    if not normalized_rate:
        return None

    try:
        rate_per_minute = float(normalized_rate)
    except ValueError:
        return None
    if rate_per_minute <= 0:
        return None

    burst_default = max(int(rate_per_minute), 1)
    burst_raw = source.get(_BURST_ENV, str(burst_default)).strip()
    try:
        burst = int(burst_raw)
    except ValueError:
        burst = burst_default
    burst = max(burst, 1)

    return TokenBucketRateLimiter(
        tokens_per_second=rate_per_minute / 60.0,
        burst=burst,
    )
