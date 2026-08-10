"""Tests for sanitize_tool_call_args_in_messages and _extract_first_json_object.

Verifies that concatenated JSON arguments (caused by vLLM/SGLang streaming bugs)
are detected and repaired to the first valid JSON object.
"""

from __future__ import annotations

from typing import Any

from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
import pytest

from wolfharness.orchestrator.event_mapper import (
    _extract_first_json_object,
    sanitize_tool_call_args_in_messages,
)


pytestmark = pytest.mark.unit


class TestExtractFirstJsonObject:
    """Unit tests for the _extract_first_json_object brace-depth scanner."""

    def test_already_valid_json_returns_none(self):
        """A single valid JSON object needs no repair."""
        assert _extract_first_json_object('{"path": "/foo"}') is None

    def test_empty_object_valid_returns_none(self):
        assert _extract_first_json_object("{}") is None

    def test_duplicated_simple_object(self):
        """Two identical JSON objects concatenated — extract the first."""
        result = _extract_first_json_object('{"path": "/foo"}{"path": "/foo"}')
        assert result == '{"path": "/foo"}'

    def test_duplicated_different_objects(self):
        """Two different JSON objects concatenated."""
        result = _extract_first_json_object('{"a": 1}{"b": 2}')
        assert result == '{"a": 1}'

    def test_triple_concatenation(self):
        """Three JSON objects concatenated (vLLM #47504 pattern)."""
        s = '{"city": "NYC"}{"city": "NYC"}{"city": "NYC"}'
        result = _extract_first_json_object(s)
        assert result == '{"city": "NYC"}'

    def test_nested_objects(self):
        """Nested braces inside the first object are handled correctly."""
        s = '{"outer": {"inner": "val"}}{"outer": {"inner": "val"}}'
        result = _extract_first_json_object(s)
        assert result == '{"outer": {"inner": "val"}}'

    def test_string_with_braces(self):
        """Braces inside JSON strings don't confuse the scanner."""
        s = '{"text": "a{b}c"}{"text": "a{b}c"}'
        result = _extract_first_json_object(s)
        assert result == '{"text": "a{b}c"}'

    def test_escaped_quotes_in_string(self):
        """Escaped double-quotes inside strings are handled."""
        s = r'{"text": "say \"hi\""}{"text": "say \"hi\""}'
        result = _extract_first_json_object(s)
        assert result == r'{"text": "say \"hi\""}'

    def test_backslash_at_end_of_string(self):
        """A backslash at the very end of input doesn't crash."""
        s = '{"a": "b\\'
        result = _extract_first_json_object(s)
        assert result is None

    def test_not_starting_with_brace_returns_none(self):
        """Non-object JSON (e.g. arrays) is not handled."""
        assert _extract_first_json_object("[1, 2, 3]") is None
        assert _extract_first_json_object("just text") is None
        assert _extract_first_json_object("") is None

    def test_whitespace_padded(self):
        """Leading/trailing whitespace is stripped before scanning."""
        result = _extract_first_json_object('  {"a": 1}{"a": 1}  ')
        assert result == '{"a": 1}'

    def test_first_object_invalid_returns_none(self):
        """If the first brace-depth segment isn't valid JSON, return None."""
        result = _extract_first_json_object('{not valid}{"a": 1}')
        assert result is None

    def test_real_world_vllm_bug_pattern(self):
        """Exact pattern from the vLLM glm47 streaming bug report."""
        path = "scratchpad:///fta-workspace/excavator-engine-overheat-workspace"
        s = f'{{"path": "{path}"}}{{"path": "{path}"}}'
        result = _extract_first_json_object(s)
        assert result == f'{{"path": "{path}"}}'


class TestSanitizeToolCallArgsInMessages:
    """Tests for sanitize_tool_call_args_in_messages on full message lists."""

    def test_repairs_duplicated_string_args(self):
        """ToolCallPart with concatenated JSON args is repaired in-place."""
        call = ToolCallPart(
            tool_name="workspace_list_dir",
            args='{"path": "/foo"}{"path": "/foo"}',
            tool_call_id="chatcmpl-tool-abc",
        )
        response = ModelResponse(parts=[call], model_name="svc/kimi-k2")
        messages: list[Any] = [response]

        sanitize_tool_call_args_in_messages(messages)

        assert messages[0].parts[0].args == '{"path": "/foo"}'

    def test_preserves_valid_string_args(self):
        """Valid JSON string args are left unchanged."""
        call = ToolCallPart(
            tool_name="bash",
            args='{"command": "ls"}',
            tool_call_id="call_123",
        )
        response = ModelResponse(parts=[call], model_name="svc/kimi-k2")
        messages: list[Any] = [response]

        sanitize_tool_call_args_in_messages(messages)

        assert messages[0].parts[0].args == '{"command": "ls"}'

    def test_preserves_dict_args(self):
        """Dict args are never touched (only str args can be concatenated)."""
        call = ToolCallPart(
            tool_name="bash",
            args={"command": "ls"},
            tool_call_id="call_123",
        )
        response = ModelResponse(parts=[call], model_name="svc/kimi-k2")
        messages: list[Any] = [response]

        sanitize_tool_call_args_in_messages(messages)

        assert messages[0].parts[0].args == {"command": "ls"}

    def test_preserves_none_args(self):
        """None args are left unchanged."""
        call = ToolCallPart(
            tool_name="bash",
            args=None,
            tool_call_id="call_123",
        )
        response = ModelResponse(parts=[call], model_name="svc/kimi-k2")
        messages: list[Any] = [response]

        sanitize_tool_call_args_in_messages(messages)

        assert messages[0].parts[0].args is None

    def test_skips_non_model_response_messages(self):
        """ModelRequest and other message types are skipped."""
        from pydantic_ai.messages import ModelRequest, UserPromptPart

        request = ModelRequest(parts=[UserPromptPart(content="hello")])
        messages: list[Any] = [request]

        sanitize_tool_call_args_in_messages(messages)

        # Unchanged
        assert isinstance(messages[0], ModelRequest)

    def test_skips_non_tool_call_parts(self):
        """TextPart and other non-tool parts are skipped."""
        text = TextPart(content="hello world")
        response = ModelResponse(parts=[text], model_name="svc/kimi-k2")
        messages: list[Any] = [response]

        sanitize_tool_call_args_in_messages(messages)

        assert messages[0].parts[0].content == "hello world"

    def test_repairs_multiple_tool_calls_in_same_response(self):
        """Multiple corrupted tool calls in one ModelResponse are all repaired."""
        call1 = ToolCallPart(
            tool_name="read",
            args='{"path": "/a"}{"path": "/a"}',
            tool_call_id="call_1",
        )
        call2 = ToolCallPart(
            tool_name="write",
            args='{"path": "/b"}{"path": "/b"}',
            tool_call_id="call_2",
        )
        response = ModelResponse(parts=[call1, call2], model_name="svc/kimi-k2")
        messages: list[Any] = [response]

        sanitize_tool_call_args_in_messages(messages)

        assert messages[0].parts[0].args == '{"path": "/a"}'
        assert messages[0].parts[1].args == '{"path": "/b"}'

    def test_mixed_valid_and_corrupted_in_same_response(self):
        """Only corrupted args are repaired; valid ones are untouched."""
        good = ToolCallPart(
            tool_name="read",
            args='{"path": "/ok"}',
            tool_call_id="call_good",
        )
        bad = ToolCallPart(
            tool_name="write",
            args='{"path": "/bad"}{"path": "/bad"}',
            tool_call_id="call_bad",
        )
        response = ModelResponse(parts=[good, bad], model_name="svc/kimi-k2")
        messages: list[Any] = [response]

        sanitize_tool_call_args_in_messages(messages)

        assert messages[0].parts[0].args == '{"path": "/ok"}'
        assert messages[0].parts[1].args == '{"path": "/bad"}'

    def test_empty_message_list_no_error(self):
        """An empty list is a no-op."""
        messages: list[Any] = []
        sanitize_tool_call_args_in_messages(messages)
        assert messages == []

    def test_idempotent(self):
        """Running sanitize twice doesn't corrupt already-repaired args."""
        call = ToolCallPart(
            tool_name="bash",
            args='{"cmd": "ls"}{"cmd": "ls"}',
            tool_call_id="call_1",
        )
        response = ModelResponse(parts=[call], model_name="svc/kimi-k2")
        messages: list[Any] = [response]

        sanitize_tool_call_args_in_messages(messages)
        assert messages[0].parts[0].args == '{"cmd": "ls"}'

        sanitize_tool_call_args_in_messages(messages)
        assert messages[0].parts[0].args == '{"cmd": "ls"}'
