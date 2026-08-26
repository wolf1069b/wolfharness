"""Unit tests for ToolArgSanitizeCapability.

Covers the ``before_model_request`` sanitization behavior: invalid-JSON
tool call arguments get replaced with ``{}`` while valid arguments are
left untouched, and unchanged histories return the original context.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast
from unittest.mock import MagicMock

from pydantic_ai import RunContext, RunUsage
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import ModelRequestContext
from pydantic_ai.models.test import TestModel
import pytest

from wolfharness.capabilities.tool_arg_sanitize import ToolArgSanitizeCapability


if TYPE_CHECKING:
    from pydantic_ai.messages import ModelMessage

pytestmark = pytest.mark.unit


def _make_run_context() -> RunContext[Any]:
    """Create a minimal ``RunContext`` for capability hook testing."""
    return RunContext(
        deps=MagicMock(),
        model=TestModel(),
        usage=RunUsage(),
        messages=[],
    )


def _make_request_context(messages: list[ModelMessage]) -> ModelRequestContext:
    """Create a ``ModelRequestContext`` wrapping the given messages."""
    return ModelRequestContext(
        model=TestModel(),
        messages=messages,
        model_settings=None,
        model_request_parameters=MagicMock(),
    )


async def test_before_model_request_empty_messages_no_crash() -> None:
    """Empty message history passes through unchanged."""
    cap = ToolArgSanitizeCapability()
    req_ctx = _make_request_context([])
    result = await cap.before_model_request(_make_run_context(), req_ctx)
    assert result is req_ctx


async def test_invalid_json_args_replaced_with_empty_dict() -> None:
    """Invalid-JSON string args are replaced with ``{}``."""
    bad_call = ToolCallPart(
        tool_name="search_kb",
        args='{"query": "missing brace',
        tool_call_id="call_1",
    )
    response = ModelResponse(parts=[bad_call])
    req_ctx = _make_request_context([response])

    cap = ToolArgSanitizeCapability()
    result = await cap.before_model_request(_make_run_context(), req_ctx)

    assert result is not req_ctx
    sanitized_call = cast(ToolCallPart, result.messages[0].parts[0])
    assert isinstance(sanitized_call, ToolCallPart)
    assert sanitized_call.args == {}
    assert sanitized_call.tool_call_id == "call_1"
    assert sanitized_call.tool_name == "search_kb"


async def test_valid_json_args_left_untouched() -> None:
    """Valid JSON string args, dict args, None args, and empty strings pass through."""
    valid_call = ToolCallPart(
        tool_name="search_kb",
        args='{"query": "hydraulic"}',
        tool_call_id="call_ok",
    )
    dict_call = ToolCallPart(
        tool_name="viking_read",
        args={"uri": "viking://resources/poc/hsjg/1.md"},
        tool_call_id="call_dict",
    )
    none_call = ToolCallPart(tool_name="list_tools", args=None, tool_call_id="call_none")
    empty_call = ToolCallPart(tool_name="ping", args="", tool_call_id="call_empty")
    response = ModelResponse(parts=[valid_call, dict_call, none_call, empty_call])
    req_ctx = _make_request_context([response])

    cap = ToolArgSanitizeCapability()
    result = await cap.before_model_request(_make_run_context(), req_ctx)

    assert result is req_ctx  # unchanged -> same identity


async def test_only_invalid_call_sanitized_others_preserved() -> None:
    """Mixed history: only the invalid call is sanitized, the rest preserved."""
    bad_call = ToolCallPart(
        tool_name="search_kb",
        args="{not json",
        tool_call_id="call_bad",
    )
    good_call = ToolCallPart(
        tool_name="viking_read",
        args='{"uri": "x"}',
        tool_call_id="call_good",
    )
    response = ModelResponse(parts=[bad_call, good_call])
    req_ctx = _make_request_context([response])

    cap = ToolArgSanitizeCapability()
    result = await cap.before_model_request(_make_run_context(), req_ctx)

    parts: list[Any] = list(result.messages[0].parts)
    assert parts[0].args == {}
    assert parts[1] is good_call  # untouched instance


async def test_multiple_responses_sanitized_independently() -> None:
    """Invalid args in multiple ModelResponse messages are all sanitized."""
    bad1 = ToolCallPart(tool_name="a", args="{", tool_call_id="c1")
    bad2 = ToolCallPart(tool_name="b", args="{broken", tool_call_id="c2")
    resp1 = ModelResponse(parts=[bad1])
    resp2 = ModelResponse(parts=[bad2])
    req_ctx = _make_request_context([resp1, resp2])

    cap = ToolArgSanitizeCapability()
    result = await cap.before_model_request(_make_run_context(), req_ctx)

    parts1: list[Any] = list(result.messages[0].parts)
    parts2: list[Any] = list(result.messages[1].parts)
    assert parts1[0].args == {}
    assert parts2[0].args == {}


async def test_non_response_messages_pass_through() -> None:
    """ModelRequest / other message types are not touched."""
    bad_call = ToolCallPart(tool_name="a", args="{", tool_call_id="c1")
    resp = ModelResponse(parts=[bad_call])
    req = ModelRequest(parts=[ToolReturnPart(tool_name="a", content="ok", tool_call_id="c1")])
    user = ModelRequest(parts=[UserPromptPart(content="hi")])
    req_ctx = _make_request_context([resp, req, user])

    cap = ToolArgSanitizeCapability()
    result = await cap.before_model_request(_make_run_context(), req_ctx)

    parts: list[Any] = list(result.messages[0].parts)
    assert parts[0].args == {}
    assert result.messages[1] is req
    assert result.messages[2] is user


async def test_config_discriminated_union_resolves() -> None:
    """The ``tool_arg_sanitize`` config type resolves to the capability."""
    from pydantic import TypeAdapter

    from wolfharness_config.capabilities import (
        CapabilityConfig,
        build_capability,
    )

    raw = {"type": "tool_arg_sanitize"}
    config = TypeAdapter(CapabilityConfig).validate_python(raw)
    cap = build_capability(config)
    assert isinstance(cap, ToolArgSanitizeCapability)
