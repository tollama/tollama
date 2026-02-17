"""Tests for runner supervisor status metadata."""

from __future__ import annotations

import pytest

from tollama.core.protocol import ProtocolRequest, ProtocolResponse, encode_line
from tollama.daemon.supervisor import RunnerSupervisor, RunnerUnavailableError


class _FakeStream:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True

    def readline(self) -> str:
        return ""

    def read(self) -> str:
        return ""

    def write(self, value: str | bytes) -> int:
        return len(value)

    def flush(self) -> None:
        return


class _FakeProcess:
    def __init__(self) -> None:
        self.stdin = _FakeStream()
        self.stdout = _FakeStream()
        self.stderr = _FakeStream()
        self.pid = 4321
        self._returncode: int | None = None

    def poll(self) -> int | None:
        return self._returncode

    def wait(self, timeout: float | None = None) -> int:
        self._returncode = 0
        return 0

    def terminate(self) -> None:
        self._returncode = 0

    def kill(self) -> None:
        self._returncode = -9


class _FakeBinaryStdout:
    def __init__(self, fd: int = 42) -> None:
        self.closed = False
        self._fd = fd

    def close(self) -> None:
        self.closed = True

    def fileno(self) -> int:
        return self._fd


class _FakeBinaryStderr:
    def __init__(self, payload: bytes) -> None:
        self.closed = False
        self._payload = payload

    def close(self) -> None:
        self.closed = True

    def read(self) -> bytes:
        return self._payload


class _FakeBinaryProcess:
    def __init__(self) -> None:
        self.stdin = _FakeStream()
        self.stdout = _FakeBinaryStdout()
        self.stderr = _FakeBinaryStderr(b"")
        self.pid = 1234
        self._returncode: int | None = None

    def poll(self) -> int | None:
        return self._returncode


def test_runner_supervisor_get_status_without_start_reports_not_running(monkeypatch) -> None:
    monkeypatch.setattr("tollama.daemon.supervisor.shutil.which", lambda _cmd: None)

    supervisor = RunnerSupervisor(runner_command=("missing-runner",))
    status = supervisor.get_status(family="torch")

    assert status["family"] == "torch"
    assert status["command"] == ["missing-runner"]
    assert status["installed"] is False
    assert status["running"] is False
    assert status["pid"] is None
    assert status["started_at"] is None
    assert status["last_used_at"] is None
    assert status["restarts"] == 0
    assert status["last_error"] is None


def test_runner_supervisor_get_status_reports_started_process(monkeypatch) -> None:
    fake_process = _FakeProcess()

    def _fake_popen(*args, **kwargs):  # noqa: ANN002, ANN003
        return fake_process

    monkeypatch.setattr("tollama.daemon.supervisor.subprocess.Popen", _fake_popen)
    monkeypatch.setattr("tollama.daemon.supervisor.shutil.which", lambda _cmd: "/usr/bin/runner")

    supervisor = RunnerSupervisor(runner_command=("runner-cmd",))
    supervisor.start()
    status = supervisor.get_status(family="mock")

    assert status["running"] is True
    assert status["pid"] == 4321
    assert isinstance(status["started_at"], str)
    assert status["installed"] is True

    supervisor.stop()
    stopped = supervisor.get_status(family="mock")
    assert stopped["running"] is False
    assert stopped["started_at"] is None


def test_runner_supervisor_records_last_error_and_restarts(monkeypatch) -> None:
    supervisor = RunnerSupervisor(runner_command=("runner-cmd",))
    monkeypatch.setattr(supervisor, "_ensure_running_locked", lambda: object())
    monkeypatch.setattr(
        supervisor,
        "_call_once_locked",
        lambda process, request, timeout: (_ for _ in ()).throw(RunnerUnavailableError("boom")),
    )

    def _fake_restart() -> None:
        supervisor._restarts += 1

    monkeypatch.setattr(supervisor, "_restart_locked_for_failure", _fake_restart)

    with pytest.raises(RunnerUnavailableError):
        supervisor.call(method="forecast", params={"x": 1}, timeout=1.0)

    status = supervisor.get_status(family="mock")
    assert status["restarts"] == 2
    assert status["last_used_at"] is not None
    assert status["last_error"] == {
        "message": "boom",
        "at": status["last_error"]["at"],
    }
    assert isinstance(status["last_error"]["at"], str)


def test_call_once_ignores_non_protocol_lines(monkeypatch) -> None:
    supervisor = RunnerSupervisor(runner_command=("runner-cmd",))
    process = _FakeProcess()
    request = ProtocolRequest(id="req-1", method="forecast", params={"x": 1})

    response = ProtocolResponse(id="req-1", result={"ok": True})
    lines = iter(["this is a log line\n", encode_line(response)])

    monkeypatch.setattr(
        supervisor,
        "_readline_with_timeout",
        lambda _process, _timeout: next(lines),
    )

    result = supervisor._call_once_locked(process, request, timeout=1.0)
    assert result == {"ok": True}


def test_call_once_ignores_mismatched_response_id(monkeypatch) -> None:
    supervisor = RunnerSupervisor(runner_command=("runner-cmd",))
    process = _FakeProcess()
    request = ProtocolRequest(id="req-2", method="forecast", params={"x": 1})

    stale = ProtocolResponse(id="stale-id", result={"ok": False})
    expected = ProtocolResponse(id="req-2", result={"ok": True})
    lines = iter([encode_line(stale), encode_line(expected)])

    monkeypatch.setattr(
        supervisor,
        "_readline_with_timeout",
        lambda _process, _timeout: next(lines),
    )

    result = supervisor._call_once_locked(process, request, timeout=1.0)
    assert result == {"ok": True}


def test_readline_with_timeout_uses_internal_buffer_before_select(monkeypatch) -> None:
    supervisor = RunnerSupervisor(runner_command=("runner-cmd",))
    process = _FakeBinaryProcess()

    select_calls = 0

    def _fake_select(readers, _writers, _errors, _timeout):  # noqa: ANN001, ANN202
        nonlocal select_calls
        select_calls += 1
        assert readers == [process.stdout]
        return ([process.stdout], [], [])

    monkeypatch.setattr("tollama.daemon.supervisor.select.select", _fake_select)
    monkeypatch.setattr(
        "tollama.daemon.supervisor.os.read",
        lambda fd, _size: b"log line\n{\"id\":\"req\"}\n" if fd == 42 else b"",
    )

    first = supervisor._readline_with_timeout(process, timeout=1.0)
    assert first == "log line\n"
    assert select_calls == 1

    def _unexpected_select(*_args, **_kwargs):  # noqa: ANN002, ANN003, ANN202
        raise AssertionError("select should not be called when a full line is already buffered")

    monkeypatch.setattr("tollama.daemon.supervisor.select.select", _unexpected_select)
    second = supervisor._readline_with_timeout(process, timeout=1.0)
    assert second == '{"id":"req"}\n'


def test_dead_process_message_decodes_binary_stderr() -> None:
    supervisor = RunnerSupervisor(runner_command=("runner-cmd",))
    process = _FakeBinaryProcess()
    process._returncode = 1  # noqa: SLF001
    process.stderr = _FakeBinaryStderr(b"fatal: boom\n")

    message = supervisor._dead_process_message(process)  # noqa: SLF001
    assert "runner exited with code 1" in message
    assert "fatal: boom" in message
