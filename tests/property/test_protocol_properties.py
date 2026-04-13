"""Property-based tests for protocol and SQL identifier validation."""

from __future__ import annotations

import pytest

try:
    from hypothesis import given
    from hypothesis import strategies as st
except ImportError:  # pragma: no cover - optional dev dependency
    pytest.skip("hypothesis is not installed", allow_module_level=True)

from tollama.connectors.postgresql import _validate_column_identifier
from tollama.core.protocol import ProtocolRequest, decode_request_line, encode_line

_REQUEST_ID_STRATEGY = st.text(
    alphabet=st.characters(whitelist_categories=("Ll", "Lu", "Nd"), whitelist_characters="_"),
    min_size=1,
    max_size=12,
)
_ASCII_IDENTIFIER_STRATEGY = st.text(
    alphabet=st.sampled_from(
        list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_")
    ),
    min_size=1,
    max_size=12,
)


@given(
    request_id=_REQUEST_ID_STRATEGY,
    horizon=st.integers(min_value=1, max_value=16),
)
def test_protocol_request_encoding_round_trip(request_id: str, horizon: int) -> None:
    request = ProtocolRequest(
        id=request_id,
        method="forecast",
        params={"model": "mock", "horizon": horizon},
    )

    assert decode_request_line(encode_line(request)) == request


@given(candidate=_ASCII_IDENTIFIER_STRATEGY)
def test_column_identifier_validation_accepts_only_single_part_identifiers(candidate: str) -> None:
    if candidate[0].isdigit():
        with pytest.raises(ValueError):
            _validate_column_identifier(candidate, field_name="id_column")
        return

    _validate_column_identifier(candidate, field_name="id_column")
