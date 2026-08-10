"""Tests for safe_args_as_dict helper.

PydanticAI's ``args_as_dict()`` returns ``{"INVALID_JSON": partial_string}``
for malformed JSON instead of raising ``ValueError``.  These tests verify
that ``safe_args_as_dict`` detects this pattern and returns the fallback.
"""

from __future__ import annotations

from pydantic_ai.messages import ToolCallPart
import pytest

from wolfharness.utils.pydantic_ai_helpers import safe_args_as_dict


pytestmark = pytest.mark.unit


@pytest.mark.unit
class TestSafeArgsAsDict:
    """Tests for safe_args_as_dict with various arg formats."""

    def test_valid_json_args(self) -> None:
        """Valid JSON args are parsed into a dict."""
        part = ToolCallPart(
            tool_name="test",
            args='{"path": "/tmp"}',
            tool_call_id="call_1",
        )
        result = safe_args_as_dict(part, default={})
        assert result == {"path": "/tmp"}

    def test_empty_string_args(self) -> None:
        """Empty string args return the default."""
        part = ToolCallPart(
            tool_name="test",
            args="",
            tool_call_id="call_2",
        )
        result = safe_args_as_dict(part, default={})
        assert result == {}

    def test_none_args(self) -> None:
        """None args return the default."""
        part = ToolCallPart(
            tool_name="test",
            args=None,
            tool_call_id="call_3",
        )
        result = safe_args_as_dict(part, default={})
        assert result == {}

    def test_partial_json_args_with_default(self) -> None:
        """Partial JSON args return the default, not INVALID_JSON dict."""
        part = ToolCallPart(
            tool_name="test",
            args='{"path": "',
            tool_call_id="call_4",
        )
        result = safe_args_as_dict(part, default={})
        assert result == {}
        assert "INVALID_JSON" not in result

    def test_partial_json_args_without_default(self) -> None:
        """Partial JSON args without default return _raw_args."""
        part = ToolCallPart(
            tool_name="test",
            args='{"path": "',
            tool_call_id="call_5",
        )
        result = safe_args_as_dict(part)
        assert "_raw_args" in result
        assert result["_raw_args"] == '{"path": "'

    def test_partial_json_with_partial_value(self) -> None:
        """Partial JSON with a partially complete value is handled."""
        part = ToolCallPart(
            tool_name="test",
            args='{"path": "scratch',
            tool_call_id="call_6",
        )
        result = safe_args_as_dict(part, default={})
        assert result == {}
        assert "INVALID_JSON" not in result

    def test_nested_partial_json(self) -> None:
        """Nested partial JSON is handled."""
        part = ToolCallPart(
            tool_name="test",
            args='{"agent": "lib',
            tool_call_id="call_7",
        )
        result = safe_args_as_dict(part, default={})
        assert result == {}
        assert "INVALID_JSON" not in result
