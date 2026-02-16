"""Runner subprocess supervision and stdio protocol calls."""

from __future__ import annotations

import select
import subprocess
import sys
import threading
from collections.abc import Sequence
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
                try:
                    return self._call_once_locked(process, request, timeout)
                except RunnerUnavailableError as exc:
                    last_unavailable = exc
                    self._restart_locked_for_failure()

            raise RunnerUnavailableError("runner unavailable after restart") from last_unavailable

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

        line = self._readline_with_timeout(process, timeout)
        try:
            response = decode_response_line(line)
        except ProtocolDecodeError as exc:
            raise RunnerProtocolError(f"invalid response from runner: {exc}") from exc

        if response.id != request.id:
            raise RunnerProtocolError(
                f"response id mismatch: expected {request.id!r}, got {response.id!r}",
            )

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
        except OSError as exc:
            self._process = None
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
