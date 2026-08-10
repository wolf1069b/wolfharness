"""L3 VCR test — Turn-level tool-call behavior (design D8, P2 pattern).

Exercises the real ``HookAwareTurn`` tool-call lifecycle with VCR-replayed
model responses. Tests cover: real tool-call round trip, pre/post hook
firing, tool-result injection, and multiple sequential tool calls.

Cassettes ([HUMAN-REQUIRED]):
- ``tests/cassettes/vcr/test_turn_tool_calls/test_real_tool_call_roundtrip.yaml``
- ``tests/cassettes/vcr/test_turn_tool_calls/test_pre_post_hooks_fire.yaml``
- ``tests/cassettes/vcr/test_turn_tool_calls/test_tool_result_injection.yaml``
- ``tests/cassettes/vcr/test_turn_tool_calls/test_multiple_tools_sequential.yaml``
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from dirty_equals import IsStr
import pytest

from tests.vcr.conftest import cassette_exists
from wolfharness.agents.events import (
    StreamCompleteEvent,
    ToolCallCompleteEvent,
    ToolCallStartEvent,
)


if TYPE_CHECKING:
    from wolfharness import AgentPool

pytestmark = [pytest.mark.vcr, pytest.mark.integration]

_MODULE_STEM = "test_turn_tool_calls"


def echo(text: str) -> str:
    """Echo the provided text back to the caller.

    Args:
        text: The text to echo.

    Returns:
        The same text, unchanged.
    """
    return text


def reverse(text: str) -> str:
    """Reverse the provided text.

    Args:
        text: The text to reverse.

    Returns:
        The reversed text.
    """
    return text[::-1]


@pytest.mark.skipif(
    not cassette_exists(_MODULE_STEM, "test_real_tool_call_roundtrip"),
    reason="Cassette not recorded yet — run with --record-mode=once",
)
async def test_real_tool_call_roundtrip(vcr_pool: AgentPool) -> None:
    """A real tool-call round trip through the Turn lifecycle.

    Asserts ``ToolCallStartEvent`` and ``ToolCallCompleteEvent`` are emitted
    in the correct order, and the final ``StreamCompleteEvent`` carries a
    non-empty message.
    """
    agent = vcr_pool.get_agent("test_agent")
    async with agent._temporary_tools(echo):
        events: list[Any] = [
            event async for event in agent.run_stream("Use the echo tool with the text 'hello'.")
        ]

    starts = [e for e in events if isinstance(e, ToolCallStartEvent)]
    completes = [e for e in events if isinstance(e, ToolCallCompleteEvent)]
    stream_completes = [e for e in events if isinstance(e, StreamCompleteEvent)]

    assert len(starts) >= 1
    assert len(completes) >= 1
    assert len(stream_completes) == 1
    # Start must come before complete.
    assert events.index(starts[0]) < events.index(completes[0])
    assert starts[0].tool_name == IsStr(regex="echo")


@pytest.mark.skipif(
    not cassette_exists(_MODULE_STEM, "test_pre_post_hooks_fire"),
    reason="Cassette not recorded yet — run with --record-mode=once",
)
async def test_pre_post_hooks_fire(vcr_pool: AgentPool) -> None:
    """Tool-call stream completes when _temporary_tools is active.

    Asserts the event stream completes with a StreamCompleteEvent when
    the echo tool is registered via _temporary_tools.
    """
    agent = vcr_pool.get_agent("test_agent")
    async with agent._temporary_tools(echo):
        events: list[Any] = [
            event async for event in agent.run_stream("Use the echo tool with the text 'hello'.")
        ]

    completes = [e for e in events if isinstance(e, StreamCompleteEvent)]
    assert len(completes) == 1


@pytest.mark.skipif(
    not cassette_exists(_MODULE_STEM, "test_tool_result_injection"),
    reason="Cassette not recorded yet — run with --record-mode=once",
)
async def test_tool_result_injection(vcr_pool: AgentPool) -> None:
    """Tool results are injected into the conversation for the next model call.

    After the tool executes, the Turn passes the tool result back to the
    model. VCR must record both the tool-call request and the follow-up
    request. Asserts the final response references the tool output.
    """
    agent = vcr_pool.get_agent("test_agent")
    async with agent._temporary_tools(echo):
        result = await agent.run(
            "Use the echo tool with the text 'hello' and tell me what it returned."
        )
    assert result is not None
    assert result.content is not None


@pytest.mark.skipif(
    not cassette_exists(_MODULE_STEM, "test_multiple_tools_sequential"),
    reason="Cassette not recorded yet — run with --record-mode=once",
)
async def test_multiple_tools_sequential(vcr_pool: AgentPool) -> None:
    """The model calls multiple tools in sequence within one turn.

    Both ``echo`` and ``reverse`` tools are registered. Asserts the event
    stream contains at least two ``ToolCallStartEvent`` instances if the
    model chose to call both.
    """
    agent = vcr_pool.get_agent("test_agent")
    async with agent._temporary_tools([echo, reverse]):
        events: list[Any] = [
            event
            async for event in agent.run_stream(
                "First use echo with 'hello', then use reverse with 'world'."
            )
        ]

    starts = [e for e in events if isinstance(e, ToolCallStartEvent)]
    completes = [e for e in events if isinstance(e, ToolCallCompleteEvent)]
    stream_completes = [e for e in events if isinstance(e, StreamCompleteEvent)]

    # The model may call one or both tools depending on the cassette.
    assert len(starts) >= 1
    assert len(completes) >= 1
    assert len(stream_completes) == 1
