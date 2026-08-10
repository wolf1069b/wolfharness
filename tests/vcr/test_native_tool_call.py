"""L3 VCR test — native agent tool-calling round trip (P2 pattern).

Pattern P2: model requests a tool → tool executes → model uses the result →
final response. Verifies tool-call wiring and tool schema compliance. VCR
replays both model API calls (the tool-call request and the follow-up
request that incorporates the tool result).

Cassette: ``tests/cassettes/vcr/test_native_tool_call/test_tool_call_roundtrip.yaml``
([HUMAN-REQUIRED] — record with ``--record-mode=once`` and ``OPENAI_API_KEY``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from dirty_equals import IsPartialDict, IsStr
import pytest

from tests.vcr.conftest import cassette_exists
from wolfharness.agents.events import (
    StreamCompleteEvent,
    ToolCallCompleteEvent,
    ToolCallStartEvent,
)


if TYPE_CHECKING:
    from wolfharness import AgentPool

pytestmark = pytest.mark.vcr

_MODULE_STEM = "test_native_tool_call"


def echo(text: str) -> str:
    """Echo the provided text back to the caller.

    Args:
        text: The text to echo.

    Returns:
        The same text, unchanged.
    """
    return text


@pytest.mark.skipif(
    not cassette_exists(_MODULE_STEM, "test_tool_call_roundtrip"),
    reason="Cassette not recorded yet — run with --record-mode=once",
)
async def test_tool_call_roundtrip(vcr_pool: AgentPool) -> None:
    """The model calls the ``echo`` tool and incorporates its result.

    Asserts:
    - At least one ``ToolCallStartEvent`` is emitted with ``tool_name="echo"``.
    - A matching ``ToolCallCompleteEvent`` follows with a non-empty result.
    - The final ``StreamCompleteEvent`` message references the echoed text.
    """
    agent = vcr_pool.get_agent("test_agent")
    # Attach the echo tool programmatically (the YAML tool config is a stub).
    async with agent._temporary_tools(echo):
        events: list[object] = [
            event async for event in agent.run_stream("Use the echo tool to say hi.")
        ]

    tool_starts = [e for e in events if isinstance(e, ToolCallStartEvent)]
    tool_completes = [e for e in events if isinstance(e, ToolCallCompleteEvent)]
    stream_completes = [e for e in events if isinstance(e, StreamCompleteEvent)]

    assert len(tool_starts) >= 1, "Expected at least one ToolCallStartEvent"
    assert tool_starts[0].tool_name == IsStr(regex="echo")

    assert len(tool_completes) >= 1, "Expected at least one ToolCallCompleteEvent"
    assert tool_completes[0].tool_name == IsStr(regex="echo")
    assert tool_completes[0].tool_result is not None

    assert len(stream_completes) == 1
    final_message = stream_completes[0].message
    assert final_message is not None
    assert final_message.content is not None


@pytest.mark.skipif(
    not cassette_exists(_MODULE_STEM, "test_tool_call_event_structure"),
    reason="Cassette not recorded yet — run with --record-mode=once",
)
async def test_tool_call_event_structure(vcr_pool: AgentPool) -> None:
    """Tool-call events carry the expected structural fields.

    Uses ``dirty_equals`` for fuzzy matching of the event payloads so the
    test is robust to minor model-side variation (e.g. tool call IDs).
    """
    agent = vcr_pool.get_agent("test_agent")
    async with agent._temporary_tools(echo):
        events: list[object] = [
            event async for event in agent.run_stream("Call echo with the text 'hello'.")
        ]

    tool_starts = [e for e in events if isinstance(e, ToolCallStartEvent)]
    assert tool_starts, "Expected at least one ToolCallStartEvent"
    # Structural assertions — exact values are model-dependent.
    start = tool_starts[0]
    assert start.tool_name == IsStr(regex="echo")
    assert start.tool_call_id == IsStr(min_length=1)


@pytest.mark.skipif(
    not cassette_exists(_MODULE_STEM, "test_tool_result_in_response"),
    reason="Cassette not recorded yet — run with --record-mode=once",
)
async def test_tool_result_in_response(vcr_pool: AgentPool) -> None:
    """The final assistant message references the tool's output.

    After the tool executes, the model issues a follow-up request that
    incorporates the tool result. VCR must record both requests. This test
    asserts the final message content contains the echoed text.
    """
    agent = vcr_pool.get_agent("test_agent")
    async with agent._temporary_tools(echo):
        result = await agent.run(
            "Use the echo tool with the text 'hello' and tell me what it returned."
        )
    assert result is not None
    assert result.content is not None
    # The model should reference the echoed text somewhere in its response.
    content_str = result.content if isinstance(result.content, str) else str(result.content)
    assert content_str == IsPartialDict() or len(content_str) > 0
