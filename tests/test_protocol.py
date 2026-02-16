"""Tests for internal daemon/runner JSON line protocol."""

import pytest

from tollama.core.protocol import (
    ProtocolDecodeError,
    ProtocolErrorMessage,
    ProtocolRequest,
    ProtocolResponse,
    decode_line,
    decode_request_line,
    decode_response_line,
    encode_line,
    generate_message_id,
    validate_message,
)


def test_encode_decode_request_line_roundtrip() -> None:
    request = ProtocolRequest(
        id=generate_message_id(),
        method="ping",
        params={"echo": "ok"},
    )

    line = encode_line(request)
    assert line.endswith("\n")

    decoded = decode_request_line(line)
    assert decoded == request


def test_encode_decode_response_success_roundtrip() -> None:
    response = ProtocolResponse(
        id=generate_message_id(),
        result={"loaded": True},
    )

    line = encode_line(response)
    decoded = decode_response_line(line)
    assert decoded == response


def test_encode_decode_response_error_roundtrip() -> None:
    response = ProtocolResponse(
        id=generate_message_id(),
        error=ProtocolErrorMessage(code=400, message="invalid request", data={"field": "horizon"}),
    )

    line = encode_line(response)
    decoded = decode_response_line(line)
    assert decoded == response


def test_encode_decode_response_string_error_code_roundtrip() -> None:
    response = ProtocolResponse(
        id=generate_message_id(),
        error=ProtocolErrorMessage(code="DEPENDENCY_MISSING", message="install extras"),
    )

    line = encode_line(response)
    decoded = decode_response_line(line)
    assert decoded == response


def test_validate_message_detects_request_and_response() -> None:
    request_data = decode_line('{"id":"1","method":"capabilities","params":{}}')
    request = validate_message(request_data)
    assert isinstance(request, ProtocolRequest)

    response_data = decode_line('{"id":"1","result":{"ok":true}}')
    response = validate_message(response_data)
    assert isinstance(response, ProtocolResponse)


def test_decode_line_rejects_invalid_json() -> None:
    with pytest.raises(ProtocolDecodeError):
        decode_line("{not-json}")

    with pytest.raises(ProtocolDecodeError):
        decode_line("[]")

    with pytest.raises(ProtocolDecodeError):
        decode_line("  \n")


def test_response_requires_exactly_one_of_result_or_error() -> None:
    invalid = '{"id":"1","result":{"ok":true},"error":{"code":1,"message":"boom"}}'
    with pytest.raises(ProtocolDecodeError):
        decode_response_line(invalid)

    invalid_missing = '{"id":"1"}'
    with pytest.raises(ProtocolDecodeError):
        decode_response_line(invalid_missing)


def test_request_requires_object_params() -> None:
    with pytest.raises(ProtocolDecodeError):
        decode_request_line('{"id":"1","method":"ping","params":[]}')
