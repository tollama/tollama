"""Token-bucket rate limiter utilities for daemon endpoints.

Supports two backends:
- In-memory (default): process-local, suitable for single-replica deployments
- Redis: distributed rate limiting for horizontal scaling

Configure via TOLLAMA_RATE_LIMIT_BACKEND=redis and TOLLAMA_REDIS_URL.
"""

from __future__ import annotations

import logging
import os
import time
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from threading import Lock

logger = logging.getLogger(__name__)

_RATE_PER_MINUTE_ENV = "TOLLAMA_RATE_LIMIT_PER_MINUTE"
_BURST_ENV = "TOLLAMA_RATE_LIMIT_BURST"
_BACKEND_ENV = "TOLLAMA_RATE_LIMIT_BACKEND"
_REDIS_URL_ENV = "TOLLAMA_REDIS_URL"


class RateLimiter(ABC):
    """Abstract rate limiter interface."""

    @abstractmethod
    def allow(self, *, key_id: str, now_s: float | None = None) -> bool:
        """Return whether a request is allowed for the provided key."""


@dataclass(slots=True)
class _TokenBucket:
    tokens: float
    last_refill_s: float


class TokenBucketRateLimiter(RateLimiter):
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


class RedisRateLimiter(RateLimiter):
    """Redis-backed sliding window rate limiter for multi-replica deployments.

    Uses a Lua script for atomic check-and-decrement to prevent race
    conditions across replicas.

    Parameters
    ----------
    redis_url : str
        Redis connection URL (e.g. ``redis://localhost:6379/0``).
    tokens_per_second : float
        Refill rate.
    burst : int
        Maximum tokens (burst capacity).
    key_prefix : str
        Redis key prefix for namespacing.
    """

    _LUA_SCRIPT = """
    local key = KEYS[1]
    local burst = tonumber(ARGV[1])
    local rate = tonumber(ARGV[2])
    local now = tonumber(ARGV[3])
    local ttl = math.ceil(burst / rate) + 1

    local data = redis.call('HMGET', key, 'tokens', 'last_refill')
    local tokens = tonumber(data[1])
    local last_refill = tonumber(data[2])

    if tokens == nil then
        tokens = burst
        last_refill = now
    end

    local elapsed = math.max(now - last_refill, 0)
    tokens = math.min(burst, tokens + elapsed * rate)

    local allowed = 0
    if tokens >= 1 then
        tokens = tokens - 1
        allowed = 1
    end

    redis.call('HMSET', key, 'tokens', tokens, 'last_refill', now)
    redis.call('EXPIRE', key, ttl)
    return allowed
    """

    def __init__(
        self,
        *,
        redis_url: str,
        tokens_per_second: float,
        burst: int,
        key_prefix: str = "tollama:ratelimit:",
    ) -> None:
        try:
            import redis as redis_lib
        except ImportError:
            raise ImportError(
                "redis package required for Redis rate limiting. Install with: pip install redis"
            )

        self._client = redis_lib.Redis.from_url(redis_url, decode_responses=True)
        self._tokens_per_second = tokens_per_second
        self._burst = burst
        self._key_prefix = key_prefix
        self._script = self._client.register_script(self._LUA_SCRIPT)

    def allow(self, *, key_id: str, now_s: float | None = None) -> bool:
        """Check rate limit via Redis Lua script (atomic)."""
        normalized_key = key_id.strip() or "anonymous"
        current_s = now_s if now_s is not None else time.time()
        redis_key = f"{self._key_prefix}{normalized_key}"

        try:
            result = self._script(
                keys=[redis_key],
                args=[self._burst, self._tokens_per_second, current_s],
            )
            return int(result) == 1
        except Exception as exc:
            logger.warning("Redis rate limiter failed, allowing request: %s", exc)
            return True  # Fail open to avoid blocking on Redis outage


def create_rate_limiter_from_env(
    *,
    env: Mapping[str, str] | None = None,
) -> RateLimiter | None:
    """Create a rate limiter when environment configuration is present.

    Supports ``TOLLAMA_RATE_LIMIT_BACKEND=redis`` for distributed limiting.
    """
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

    tokens_per_second = rate_per_minute / 60.0

    # Check for Redis backend
    backend = source.get(_BACKEND_ENV, os.environ.get(_BACKEND_ENV, "memory")).strip().lower()
    if backend == "redis":
        redis_url = source.get(
            _REDIS_URL_ENV,
            os.environ.get(_REDIS_URL_ENV, "redis://localhost:6379/0"),
        )
        try:
            return RedisRateLimiter(
                redis_url=redis_url,
                tokens_per_second=tokens_per_second,
                burst=burst,
            )
        except ImportError:
            logger.warning("Redis not available, falling back to in-memory rate limiter")

    return TokenBucketRateLimiter(
        tokens_per_second=tokens_per_second,
        burst=burst,
    )
