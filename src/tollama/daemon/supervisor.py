"""Runner subprocess supervision and stdio protocol calls."""

from __future__ import annotations

import enum
import logging
import os
import select
import shutil
import subprocess
import sys
import threading
import time
from collections import deque
from collections.abc import Sequence
from contextlib import suppress
from datetime import UTC, datetime
from typing import Any

from tollama.core.protocol import (
    ProtocolDecodeError,
    ProtocolRequest,
    decode_response_line,
    encode_line,
    generate_message_id,
)

logger = logging.getLogger(__name__)

DEFAULT_CALL_TIMEOUT_SECONDS = 10.0

# Circuit breaker defaults
CB_FAILURE_THRESHOLD = 3
CB_FAILURE_WINDOW_SECONDS = 30.0
CB_RECOVERY_TIMEOUT_SECONDS = 15.0

# Retry backoff defaults
RETRY_BASE_DELAY_SECONDS = 0.1
RETRY_MAX_DELAY_SECONDS = 5.0
RETRY_MAX_ATTEMPTS = 3


class RunnerError(RuntimeError):
    """Base runner supervision error."""


class RunnerUnavailableError(RunnerError):
    """Raised when the runner process is unavailable."""


class RunnerCircuitOpenError(RunnerUnavailableError):
    """Raised when the circuit breaker is open and rejecting calls."""


class RunnerProtocolError(RunnerError):
    """Raised when a runner response violates protocol expectations."""


class RunnerCallError(RunnerError):
    """Raised when the runner returns a structured error response."""

    def __init__(self, code: int | str, message: str, data: Any | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.data = data


class CircuitState(enum.Enum):
    """Circuit breaker states."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """Circuit breaker for runner failure protection.

    State transitions:
    - CLOSED: Normal operation. Track failures within a sliding window.
    - OPEN: Reject all calls. Transition to HALF_OPEN after recovery timeout.
    - HALF_OPEN: Allow one probe call. Success → CLOSED, failure → OPEN.
    """

    def __init__(
        self,
        *,
        failure_threshold: int = CB_FAILURE_THRESHOLD,
        failure_window_seconds: float = CB_FAILURE_WINDOW_SECONDS,
        recovery_timeout_seconds: float = CB_RECOVERY_TIMEOUT_SECONDS,
    ) -> None:
        self._failure_threshold = failure_threshold
        self._failure_window_seconds = failure_window_seconds
        self._recovery_timeout_seconds = recovery_timeout_seconds
        self._state = CircuitState.CLOSED
        self._failure_timestamps: list[float] = []
        self._opened_at: float | None = None

    @property
    def state(self) -> CircuitState:
        """Return the current circuit state, auto-transitioning OPEN → HALF_OPEN."""
        if (
            self._state == CircuitState.OPEN
            and self._opened_at is not None
            and time.monotonic() - self._opened_at >= self._recovery_timeout_seconds
        ):
            self._state = CircuitState.HALF_OPEN
        return self._state

    def check(self) -> None:
        """Check if a call is allowed. Raises RunnerCircuitOpenError if not."""
        current = self.state
        if current == CircuitState.OPEN:
            raise RunnerCircuitOpenError(
                "circuit breaker is open — runner has failed repeatedly; "
                f"will retry after {self._recovery_timeout_seconds:.0f}s recovery period"
            )

    def record_success(self) -> None:
        """Record a successful call."""
        self._failure_timestamps.clear()
        if self._state in (CircuitState.HALF_OPEN, CircuitState.OPEN):
            logger.info("circuit breaker closed after successful probe")
        self._state = CircuitState.CLOSED
        self._opened_at = None

    def record_failure(self) -> None:
        """Record a failed call. May trip the breaker to OPEN."""
        now = time.monotonic()
        self._failure_timestamps.append(now)

        if self._state == CircuitState.HALF_OPEN:
            self._trip(now)
            return

        # Prune failures outside the sliding window
        cutoff = now - self._failure_window_seconds
        self._failure_timestamps = [t for t in self._failure_timestamps if t > cutoff]

        if len(self._failure_timestamps) >= self._failure_threshold:
            self._trip(now)

    def _trip(self, now: float) -> None:
        self._state = CircuitState.OPEN
        self._opened_at = now
        logger.warning(
            "circuit breaker tripped to OPEN after %d failures in %.0fs",
            len(self._failure_timestamps),
            self._failure_window_seconds,
        )

    def get_status(self) -> dict[str, Any]:
        """Return circuit breaker status for diagnostics."""
        now = time.monotonic()
        cutoff = now - self._failure_window_seconds
        recent_failures = sum(1 for t in self._failure_timestamps if t > cutoff)
        return {
            "state": self.state.value,
            "recent_failures": recent_failures,
            "failure_threshold": self._failure_threshold,
        }


class RunnerSupervisor:
    """Manage one long-lived runner subprocess and synchronous method calls."""

    def __init__(
        self,
        runner_command: Sequence[str] | None = None,
        *,
        circuit_breaker: CircuitBreaker | None = None,
        max_retry_attempts: int = RETRY_MAX_ATTEMPTS,
        retry_base_delay: float = RETRY_BASE_DELAY_SECONDS,
        retry_max_delay: float = RETRY_MAX_DELAY_SECONDS,
    ) -> None:
        default_command = (sys.executable, "-m", "tollama.runners.mock.main")
        self._runner_command = list(runner_command or default_command)
        self._process: subprocess.Popen[bytes] | None = None
        self._lock = threading.Lock()
        self._started_at: datetime | None = None
        self._last_used_at: datetime | None = None
        self._restarts = 0
        self._last_error_message: str | None = None
        self._last_error_at: datetime | None = None
        self._stdout_buffer = bytearray()
        self._stderr_buffer = bytearray()
        self._stderr_lines: deque[str] = deque(maxlen=100)
        self._stderr_thread: threading.Thread | None = None
        self._stderr_stop = threading.Event()
        self._stderr_lock = threading.Lock()
        self._circuit_breaker = circuit_breaker or CircuitBreaker()
        self._max_retry_attempts = max(max_retry_attempts, 1)
        self._retry_base_delay = retry_base_delay
        self._retry_max_delay = retry_max_delay

    def start(self) -> None:
        """Start the runner process if it is not currently running."""
        with self._lock:
            self._start_locked()

    def stop(self) -> None:
        """Stop the runner process if it is running."""
        with self._lock:
            self._stop_locked()

    def restart(self) -> None:
        """Restart the runner process."""
        with self._lock:
            self._restarts += 1
            self._stop_locked()
            self._start_locked()

    def call(
        self,
        method: str,
        params: dict[str, Any],
        timeout: float = DEFAULT_CALL_TIMEOUT_SECONDS,
        request_id: str | None = None,
    ) -> dict[str, Any]:
        """Call a runner method and return the result payload.

        Uses a circuit breaker to avoid hammering a persistently failing runner
        and exponential backoff between retry attempts.
        """
        if timeout <= 0:
            raise ValueError("timeout must be greater than zero")

        # Circuit breaker check (outside lock to avoid holding lock while sleeping)
        self._circuit_breaker.check()

        request = ProtocolRequest(
            id=self._new_request_id(request_id=request_id),
            method=method,
            params=params,
        )

        last_unavailable: RunnerUnavailableError | None = None
        for attempt in range(self._max_retry_attempts):
            if attempt > 0:
                # Re-check circuit breaker before each retry
                self._circuit_breaker.check()
                # Exponential backoff: base * 2^(attempt-1), capped
                delay = min(
                    self._retry_base_delay * (2 ** (attempt - 1)),
                    self._retry_max_delay,
                )
                time.sleep(delay)

            with self._lock:
                try:
                    process = self._ensure_running_locked()
                    self._last_used_at = datetime.now(UTC)
                    result = self._call_once_locked(process, request, timeout)
                    self._circuit_breaker.record_success()
                    return result
                except RunnerUnavailableError as exc:
                    self._record_error_locked(str(exc))
                    self._circuit_breaker.record_failure()
                    last_unavailable = exc
                    self._restart_locked_for_failure()
                except (RunnerCallError, RunnerProtocolError) as exc:
                    self._record_error_locked(str(exc))
                    # Structured errors are not retry-able (bad request, etc.)
                    raise

        final_message = (
            str(last_unavailable)
            if last_unavailable is not None
            else "runner unavailable after retries"
        )
        with self._lock:
            self._record_error_locked(final_message)
        raise RunnerUnavailableError(final_message) from last_unavailable

    def get_status(self, *, family: str | None = None) -> dict[str, Any]:
        """Return current runner process status without starting the process."""
        with self._lock:
            process = self._process
            if process is not None and process.poll() is not None:
                self._clear_process_locked()
                process = None

            running = process is not None and process.poll() is None
            pid = process.pid if running else None
            started_at = _to_utc_iso(self._started_at) if running else None
            last_used_at = _to_utc_iso(self._last_used_at)
            command = list(self._runner_command)
            installed = bool(command) and shutil.which(command[0]) is not None
            if self._last_error_message is not None and self._last_error_at is not None:
                last_error: dict[str, Any] | None = {
                    "message": self._last_error_message,
                    "at": _to_utc_iso(self._last_error_at),
                }
            else:
                last_error = None

            return {
                "family": family,
                "command": command,
                "installed": installed,
                "running": running,
                "pid": pid,
                "started_at": started_at,
                "last_used_at": last_used_at,
                "restarts": self._restarts,
                "last_error": last_error,
                "circuit_breaker": self._circuit_breaker.get_status(),
            }

    def _call_once_locked(
        self,
        process: subprocess.Popen[bytes],
        request: ProtocolRequest,
        timeout: float,
    ) -> dict[str, Any]:
        stdin = process.stdin
        stdout = process.stdout
        if stdin is None or stdout is None:
            raise RunnerUnavailableError("runner stdio pipes are not available")

        if process.poll() is not None:
            raise RunnerUnavailableError(self._dead_process_message(process))

        try:
            stdin.write(encode_line(request).encode("utf-8"))
            stdin.flush()
        except OSError as exc:
            raise RunnerUnavailableError(f"failed to write to runner: {exc}") from exc

        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RunnerUnavailableError(f"runner call timed out after {timeout:.2f}s")

            line = self._readline_with_timeout(process, remaining)
            try:
                response = decode_response_line(line)
            except ProtocolDecodeError:
                # Some model libraries emit stdout logs that can appear interleaved with the
                # protocol channel. Ignore non-protocol lines and keep waiting.
                continue

            if response.id != request.id:
                # Ignore unrelated responses (for example stale responses that arrived late).
                continue

            if response.error is not None:
                raise RunnerCallError(
                    code=response.error.code,
                    message=response.error.message,
                    data=response.error.data,
                )

            result = response.result
            if result is None:
                raise RunnerProtocolError("runner response is missing result")

            return result

    def _readline_with_timeout(self, process: subprocess.Popen[bytes], timeout: float) -> str:
        stdout = process.stdout
        if stdout is None:
            raise RunnerUnavailableError("runner stdout is not available")

        deadline = time.monotonic() + timeout
        while True:
            line = self._consume_buffered_line()
            if line is not None:
                return line

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                if process.poll() is not None:
                    raise RunnerUnavailableError(self._dead_process_message(process))
                raise RunnerUnavailableError(f"runner call timed out after {timeout:.2f}s")

            ready, _, _ = select.select([stdout], [], [], remaining)
            if not ready:
                if process.poll() is not None:
                    raise RunnerUnavailableError(self._dead_process_message(process))
                raise RunnerUnavailableError(f"runner call timed out after {timeout:.2f}s")

            chunk = os.read(stdout.fileno(), 4096)
            if not chunk:
                raise RunnerUnavailableError(self._dead_process_message(process))
            self._stdout_buffer.extend(chunk)

    def _consume_buffered_line(self) -> str | None:
        newline_index = self._stdout_buffer.find(b"\n")
        if newline_index < 0:
            return None
        line = bytes(self._stdout_buffer[: newline_index + 1])
        del self._stdout_buffer[: newline_index + 1]
        return line.decode("utf-8", errors="replace")

    def _ensure_running_locked(self) -> subprocess.Popen[bytes]:
        process = self._process
        if process is not None and process.poll() is None:
            return process

        if process is not None and process.poll() is not None:
            self._clear_process_locked()

        self._start_locked()
        if self._process is None:
            raise RunnerUnavailableError("runner failed to start")
        return self._process

    def _start_locked(self) -> None:
        process = self._process
        if process is not None and process.poll() is None:
            return

        if process is not None:
            self._clear_process_locked()

        try:
            self._process = subprocess.Popen(  # noqa: S603 - configured local entrypoints only
                self._runner_command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=False,
                bufsize=0,
            )
            self._started_at = datetime.now(UTC)
            self._stdout_buffer.clear()
            self._stderr_buffer.clear()
            self._stderr_lines.clear()
            self._start_stderr_drain_locked(self._process)
        except OSError as exc:
            self._process = None
            self._started_at = None
            self._record_error_locked(f"failed to start runner: {exc}")
            raise RunnerUnavailableError(f"failed to start runner: {exc}") from exc

    def _stop_locked(self) -> None:
        process = self._process
        if process is None:
            return

        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=3)

        self._clear_process_locked()

    def _restart_locked_for_failure(self) -> None:
        self._restarts += 1
        self._stop_locked()
        self._start_locked()

    def _clear_process_locked(self) -> None:
        process = self._process
        if process is None:
            return

        self._stop_stderr_drain_locked(process.stderr)

        for stream in (process.stdin, process.stdout, process.stderr):
            if stream is not None and not stream.closed:
                stream.close()

        self._process = None
        self._started_at = None
        self._stdout_buffer.clear()
        self._stderr_buffer.clear()

    def _dead_process_message(self, process: subprocess.Popen[bytes]) -> str:
        returncode = process.poll()
        stderr_tail = self._stderr_tail()

        if stderr_tail:
            return f"runner exited with code {returncode}: {stderr_tail}"
        return f"runner exited with code {returncode}"

    def _new_request_id(self, *, request_id: str | None = None) -> str:
        if request_id is not None:
            normalized = request_id.strip()
            if normalized:
                return normalized
        return generate_message_id()

    def _record_error_locked(self, message: str) -> None:
        normalized = message.strip()
        self._last_error_message = normalized if normalized else "runner error"
        self._last_error_at = datetime.now(UTC)

    def _start_stderr_drain_locked(self, process: subprocess.Popen[bytes]) -> None:
        stderr = process.stderr
        if stderr is None:
            return

        self._stderr_stop = threading.Event()
        self._stderr_thread = threading.Thread(
            target=self._drain_stderr,
            args=(stderr, self._stderr_stop),
            name=f"tollama-stderr-{process.pid}",
            daemon=True,
        )
        self._stderr_thread.start()

    def _stop_stderr_drain_locked(self, stderr: Any) -> None:
        self._stderr_stop.set()
        if stderr is not None and not stderr.closed:
            with suppress(OSError):
                stderr.close()

        thread = self._stderr_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.0)
        self._stderr_thread = None

    def _drain_stderr(self, stderr: Any, stop_event: threading.Event) -> None:
        fileno = getattr(stderr, "fileno", None)
        if callable(fileno):
            try:
                fd = fileno()
            except (AttributeError, OSError, ValueError):
                fd = None
        else:
            fd = None

        if fd is None:
            self._drain_stderr_fallback(stderr, stop_event)
            return

        while not stop_event.is_set():
            try:
                chunk = os.read(fd, 4096)
            except OSError:
                break
            if not chunk:
                break
            self._append_stderr_chunk(chunk)
        self._flush_stderr_buffer()

    def _drain_stderr_fallback(self, stderr: Any, stop_event: threading.Event) -> None:
        while not stop_event.is_set():
            try:
                chunk = stderr.readline()
            except (AttributeError, OSError, ValueError):
                break
            if not chunk:
                break
            if isinstance(chunk, str):
                payload = chunk.encode("utf-8", errors="replace")
            else:
                payload = bytes(chunk)
            self._append_stderr_chunk(payload)
        self._flush_stderr_buffer()

    def _append_stderr_chunk(self, chunk: bytes) -> None:
        with self._stderr_lock:
            self._stderr_buffer.extend(chunk)
            while True:
                newline_index = self._stderr_buffer.find(b"\n")
                if newline_index < 0:
                    break
                line = bytes(self._stderr_buffer[:newline_index]).decode("utf-8", errors="replace")
                del self._stderr_buffer[: newline_index + 1]
                self._stderr_lines.append(line.strip())

    def _flush_stderr_buffer(self) -> None:
        with self._stderr_lock:
            if not self._stderr_buffer:
                return
            line = bytes(self._stderr_buffer).decode("utf-8", errors="replace").strip()
            self._stderr_buffer.clear()
            if line:
                self._stderr_lines.append(line)

    def _stderr_tail(self) -> str:
        with self._stderr_lock:
            tail = [line for line in self._stderr_lines if line]
        return "\n".join(tail).strip()


def _to_utc_iso(value: datetime | None) -> str | None:
    if value is None:
        return None
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")
