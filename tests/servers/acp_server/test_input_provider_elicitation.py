"""Tests for ACPInputProvider elicitation response mapping."""

from __future__ import annotations

from typing import Any

from mcp.types import ElicitResult
import pytest

from wolfharness_server.acp_server.input_provider import ACPInputProvider


class _FakeElicitationResponse:
    """Fake ACP ElicitationCreateResponse for testing."""

    def __init__(self, action: str = "accept", content: dict[str, Any] | None = None) -> None:
        self.action = action
        self.content = content


pytestmark = pytest.mark.unit


@pytest.mark.unit
def test_map_elicitation_url_mode_drops_content() -> None:
    """URL mode must omit content per the MCP elicitation spec."""
    response = _FakeElicitationResponse(action="accept", content={"content": {}})
    result = ACPInputProvider._map_elicitation_create_response(response, mode="url")
    assert result.action == "accept"
    assert result.content is None


@pytest.mark.unit
def test_map_elicitation_form_mode_normalizes_nested_content() -> None:
    """Form mode must normalize nested content to MCP primitive types."""
    response = _FakeElicitationResponse(
        action="accept",
        content={"content": {"annotations": ["note"]}, "score": 0.9},
    )
    result = ACPInputProvider._map_elicitation_create_response(response, mode="form")
    assert result.action == "accept"
    assert result.content is not None
    assert result.content["content"] == '{"annotations": ["note"]}'
    assert result.content["score"] == 0.9
    # Validate against the strict MCP SDK model
    validated = ElicitResult.model_validate(result.model_dump(by_alias=True))
    assert validated.action == "accept"


@pytest.mark.unit
def test_map_elicitation_form_mode_empty_content_defaults_to_empty_dict() -> None:
    """Form mode with None content defaults to an empty dict."""
    response = _FakeElicitationResponse(action="accept", content=None)
    result = ACPInputProvider._map_elicitation_create_response(response, mode="form")
    assert result.action == "accept"
    assert result.content == {}


@pytest.mark.unit
def test_map_elicitation_decline_and_cancel() -> None:
    """Decline and cancel actions map without content."""
    decline_response = _FakeElicitationResponse(action="decline")
    result = ACPInputProvider._map_elicitation_create_response(decline_response, mode="form")
    assert result.action == "decline"
    assert result.content is None

    cancel_response = _FakeElicitationResponse(action="cancel")
    result = ACPInputProvider._map_elicitation_create_response(cancel_response, mode="form")
    assert result.action == "cancel"
    assert result.content is None
