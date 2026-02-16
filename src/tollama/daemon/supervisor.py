"""Runner subprocess supervision and stdio protocol calls."""

from __future__ import annotations

import select
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

from tollama.core.protocol import (
    ProtocolDecodeError,
    ProtocolRequest,
    decode_response_line,
    encode_line,
    generate_message_id,
)

DEFAULT_CALL_TIMEOUT_SECONDS = 10.0


class RunnerError(RuntimeError):
    """Base runner supervision error."""


class RunnerUnavailableError(RunnerError):
    """Raised when the runner process is unavailable."""


class RunnerProtocolError(RunnerError):
    """Raised when a runner response violates protocol expectations."""


class RunnerCallError(RunnerError):
    """Raised when the runner returns a structured error response."""

    def __init__(self, code: int | str, message: str, data: Any | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.data = data


class RunnerSupervisor:
    """Manage one long-lived runner subprocess and synchronous method calls."""

    def __init__(self, runner_command: Sequence[str] | None = None) -> None:
        default_command = (sys.executable, "-m", "tollama.runners.mock.main")
        self._runner_command = list(runner_command or default_command)
        self._process: subprocess.Popen[str] | None = None
        self._lock = threading.Lock()
        self._started_at: datetime | None = None
        self._last_used_at: datetime | None = None
        self._restarts = 0
        self._last_error_message: str | None = None
        self._last_error_at: datetime | None = None

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
    ) -> dict[str, Any]:
        """Call a runner method and return the result payload."""
        if timeout <= 0:
            raise ValueError("timeout must be greater than zero")

        request = ProtocolRequest(id=self._new_request_id(), method=method, params=params)

        with self._lock:
            last_unavailable: RunnerUnavailableError | None = None
            for _attempt in range(2):
                process = self._ensure_running_locked()
                self._last_used_at = datetime.now(UTC)
                try:
                    return self._call_once_locked(process, request, timeout)
                except RunnerUnavailableError as exc:
                    self._record_error_locked(str(exc))
                    last_unavailable = exc
                    self._restart_locked_for_failure()
                except (RunnerCallError, RunnerProtocolError) as exc:
                    self._record_error_locked(str(exc))
                    raise

            final_message = (
                str(last_unavailable)
                if last_unavailable is not None
                else "runner unavailable after restart"
            )
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
            }

    def _call_once_locked(
        self,
        process: subprocess.Popen[str],
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
            stdin.write(encode_line(request))
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

    def _readline_with_timeout(self, process: subprocess.Popen[str], timeout: float) -> str:
        stdout = process.stdout
        if stdout is None:
            raise RunnerUnavailableError("runner stdout is not available")

        ready, _, _ = select.select([stdout], [], [], timeout)
        if not ready:
            if process.poll() is not None:
                raise RunnerUnavailableError(self._dead_process_message(process))
            raise RunnerUnavailableError(f"runner call timed out after {timeout:.2f}s")

        line = stdout.readline()
        if line:
            return line

        raise RunnerUnavailableError(self._dead_process_message(process))

    def _ensure_running_locked(self) -> subprocess.Popen[str]:
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
            self._process = subprocess.Popen(
                self._runner_command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
            self._started_at = datetime.now(UTC)
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

        for stream in (process.stdin, process.stdout, process.stderr):
            if stream is not None and not stream.closed:
                stream.close()

        self._process = None
        self._started_at = None

    def _dead_process_message(self, process: subprocess.Popen[str]) -> str:
        returncode = process.poll()
        stderr = process.stderr
        stderr_tail = ""
        if stderr is not None:
            try:
                stderr_tail = stderr.read().strip()
            except OSError:
                stderr_tail = ""

        if stderr_tail:
            return f"runner exited with code {returncode}: {stderr_tail}"
        return f"runner exited with code {returncode}"

    def _new_request_id(self) -> str:
        return generate_message_id()

    def _record_error_locked(self, message: str) -> None:
        normalized = message.strip()
        self._last_error_message = normalized if normalized else "runner error"
        self._last_error_at = datetime.now(UTC)


def _to_utc_iso(value: datetime | None) -> str | None:
    if value is None:
        return None
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")
