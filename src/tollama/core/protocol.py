"""Internal daemon<->runner line protocol over stdio."""

from __future__ import annotations

import json
import uuid
from collections.abc import Mapping
from typing import Annotated, Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    StrictInt,
    StrictStr,
    ValidationError,
    model_validator,
)

NonEmptyStr = Annotated[StrictStr, Field(min_length=1)]

SUPPORTED_METHODS = frozenset({"capabilities", "load", "forecast", "ping", "hello"})


class ProtocolDecodeError(ValueError):
    """Raised when decoding or validating a protocol message fails."""


class ProtocolBaseModel(BaseModel):
    """Base model with strict validation for protocol messages."""

    model_config = ConfigDict(extra="forbid", strict=True, ser_json_inf_nan="null")


class ProtocolRequest(ProtocolBaseModel):
    """Request: {id, method, params}."""

    id: NonEmptyStr
    method: NonEmptyStr
    params: dict[NonEmptyStr, JsonValue] = Field(default_factory=dict)


class ProtocolErrorMessage(ProtocolBaseModel):
    """Error payload for failed responses."""

    code: StrictInt
    message: NonEmptyStr
    data: JsonValue | None = None


class ProtocolResponse(ProtocolBaseModel):
    """Response: {id, result} or {id, error}."""

    id: NonEmptyStr
    result: dict[NonEmptyStr, JsonValue] | None = None
    error: ProtocolErrorMessage | None = None

    @model_validator(mode="after")
    def validate_result_or_error(self) -> ProtocolResponse:
        has_result = self.result is not None
        has_error = self.error is not None
        if has_result == has_error:
            raise ValueError("response must include exactly one of result or error")
        return self


Message = ProtocolRequest | ProtocolResponse


def generate_message_id() -> str:
    """Generate a unique message identifier."""
    return uuid.uuid4().hex


def _to_payload(message: Message | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(message, BaseModel):
        return message.model_dump(mode="json", exclude_none=True)
    if isinstance(message, Mapping):
        return dict(message)
    raise TypeError("message must be a protocol model or mapping")


def encode_line(message: Message | Mapping[str, Any]) -> str:
    """Encode a message to one canonical JSON line."""
    payload = _to_payload(message)
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return f"{encoded}\n"


def decode_line(line: str | bytes) -> dict[str, Any]:
    """Decode one line into a JSON object without applying request/response schema."""
    if isinstance(line, bytes):
        try:
            text = line.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ProtocolDecodeError("line is not valid UTF-8") from exc
    else:
        text = line

    text = text.strip()
    if not text:
        raise ProtocolDecodeError("line is empty")

    try:
        decoded = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ProtocolDecodeError("line is not valid JSON") from exc

    if not isinstance(decoded, dict):
        raise ProtocolDecodeError("message must be a JSON object")

    return decoded


def validate_request(message: Mapping[str, Any]) -> ProtocolRequest:
    """Validate a request message object."""
    try:
        return ProtocolRequest.model_validate(dict(message))
    except ValidationError as exc:
        raise ProtocolDecodeError("invalid request message") from exc


def validate_response(message: Mapping[str, Any]) -> ProtocolResponse:
    """Validate a response message object."""
    try:
        return ProtocolResponse.model_validate(dict(message))
    except ValidationError as exc:
        raise ProtocolDecodeError("invalid response message") from exc


def validate_message(message: Mapping[str, Any]) -> Message:
    """Validate either request or response message based on discriminating keys."""
    is_request = "method" in message
    is_response = "result" in message or "error" in message

    if is_request and is_response:
        raise ProtocolDecodeError("message cannot contain both request and response fields")
    if is_request:
        return validate_request(message)
    if is_response:
        return validate_response(message)
    raise ProtocolDecodeError("message is neither a request nor a response")


def decode_request_line(line: str | bytes) -> ProtocolRequest:
    """Decode and validate a request line."""
    return validate_request(decode_line(line))


def decode_response_line(line: str | bytes) -> ProtocolResponse:
    """Decode and validate a response line."""
    return validate_response(decode_line(line))
