"""Unit tests for ACPTurn."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock

import pytest

from acp.schema import (
    AgentMessageChunk,
    TextContentBlock,
)
from wolfharness.agents.acp_agent.turn import ACPTurn
from wolfharness.agents.context import AgentRunContext
from wolfharness.agents.events import (
    PartDeltaEvent,
    RunErrorEvent,
    StreamCompleteEvent,
)


if TYPE_CHECKING:
    from collections.abc import AsyncIterator


class MockACPClient:
    """Mock ACP client for testing ACPTurn."""

    def __init__(
        self,
        *,
        updates: list[Any] | None = None,
        messages: list[Any] | None = None,
        prompt_error: Exception | None = None,
        stream_error: Exception | None = None,
        get_messages_error: Exception | None = None,
    ) -> None:
        self._updates = updates or []
        self._messages = messages or []
        self._prompt_error = prompt_error
        self._stream_error = stream_error
        self._get_messages_error = get_messages_error
        self.prompt_calls: list[tuple[str, list[Any]]] = []

    async def prompt(self, session_id: str, content: list[Any]) -> Any:
        self.prompt_calls.append((session_id, content))
        if self._prompt_error:
            raise self._prompt_error
        return MagicMock(name="PromptResponse")

    async def stream_events(self, response: Any) -> AsyncIterator[Any]:
        for update in self._updates:
            yield update
        if self._stream_error:
            raise self._stream_error

    async def get_messages(self, session_id: str) -> list[Any]:
        if self._get_messages_error:
            raise self._get_messages_error
        return list(self._messages)


def _make_run_ctx() -> AgentRunContext:
    return AgentRunContext(session_id="test-session")


def _text_update(text: str) -> AgentMessageChunk:
    return AgentMessageChunk(content=TextContentBlock(text=text))


@pytest.mark.unit
async def test_acp_turn_prompt_stream_complete_cycle() -> None:
    """Given a mock client, ACPTurn yields PartDelta, StreamComplete.

    RunStartedEvent is published by RunHandle.start(), not by turn.execute().
    """
    updates = [_text_update("Hello"), _text_update(" world")]
    messages = [_text_update("Hello world")]
    client = MockACPClient(updates=updates, messages=messages)
    run_ctx = _make_run_ctx()

    turn = ACPTurn(
        acp_client=client,
        prompts=["Say hello"],
        run_ctx=run_ctx,
        session_id="test-session",
    )

    events = [event async for event in turn.execute()]

    # Verify prompt was called with correct session and content
    assert len(client.prompt_calls) == 1
    assert client.prompt_calls[0][0] == "test-session"

    # Verify event sequence: PartDelta, PartDelta, StreamComplete
    # (RunStartedEvent is published by RunHandle.start(), not turn.execute())
    assert len(events) == 3

    assert isinstance(events[0], PartDeltaEvent)
    assert isinstance(events[1], PartDeltaEvent)

    assert isinstance(events[2], StreamCompleteEvent)
    assert events[2].message.content == "Hello world"


@pytest.mark.unit
async def test_acp_turn_prompt_error_yields_run_error_event() -> None:
    """Given a prompt error, ACPTurn yields RunError.

    RunStartedEvent is published by RunHandle.start(), not turn.execute().
    """
    client = MockACPClient(prompt_error=RuntimeError("Connection refused"))
    run_ctx = _make_run_ctx()

    turn = ACPTurn(
        acp_client=client,
        prompts=["Hello"],
        run_ctx=run_ctx,
        session_id="test-session",
    )

    events = [event async for event in turn.execute()]

    # RunErrorEvent only (RunStartedEvent is published by RunHandle.start())
    assert len(events) == 1
    assert isinstance(events[0], RunErrorEvent)
    assert "Connection refused" in events[0].message


@pytest.mark.unit
async def test_acp_turn_stream_error_yields_run_error_event() -> None:
    """Given a stream error, ACPTurn yields partial events, then RunError.

    RunStartedEvent is published by RunHandle.start(), not turn.execute().
    """
    client = MockACPClient(
        updates=[_text_update("partial")],
        stream_error=RuntimeError("Stream broken"),
    )
    run_ctx = _make_run_ctx()

    turn = ACPTurn(
        acp_client=client,
        prompts=["Hello"],
        run_ctx=run_ctx,
        session_id="test-session",
    )

    events = [event async for event in turn.execute()]

    # PartDelta (from partial update), RunError
    # (RunStartedEvent is published by RunHandle.start(), not turn.execute())
    assert len(events) == 2
    assert isinstance(events[0], PartDeltaEvent)
    assert isinstance(events[1], RunErrorEvent)
    assert "Stream broken" in events[1].message


@pytest.mark.unit
async def test_acp_turn_message_history_and_final_message_after_execute() -> None:
    """Given completed execute, message_history and final_message are populated."""
    updates = [_text_update("Response text")]
    messages = [_text_update("Response text")]
    client = MockACPClient(updates=updates, messages=messages)
    run_ctx = _make_run_ctx()

    turn = ACPTurn(
        acp_client=client,
        prompts=["Hello"],
        run_ctx=run_ctx,
        session_id="test-session",
    )

    events = [event async for event in turn.execute()]

    # message_history is populated
    history = turn.message_history
    assert len(history) > 0

    # final_message is populated
    final = turn.final_message
    assert final.role == "assistant"
    assert final.content == "Response text"

    # StreamCompleteEvent carries the same message
    complete_event = next(e for e in events if isinstance(e, StreamCompleteEvent))
    assert complete_event.message is final


@pytest.mark.unit
async def test_acp_turn_properties_raise_before_execute() -> None:
    """Given execute not called, message_history and final_message raise RuntimeError."""
    client = MockACPClient()
    run_ctx = _make_run_ctx()

    turn = ACPTurn(
        acp_client=client,
        prompts=["Hello"],
        run_ctx=run_ctx,
        session_id="test-session",
    )

    with pytest.raises(RuntimeError, match="message_history is not available"):
        _ = turn.message_history

    with pytest.raises(RuntimeError, match="final_message is not available"):
        _ = turn.final_message


@pytest.mark.unit
async def test_acp_turn_cancelled_error_propagates() -> None:
    """Given CancelledError during prompt, ACPTurn re-raises without yielding RunError."""
    client = MockACPClient(prompt_error=asyncio.CancelledError())
    run_ctx = _make_run_ctx()

    turn = ACPTurn(
        acp_client=client,
        prompts=["Hello"],
        run_ctx=run_ctx,
        session_id="test-session",
    )

    with pytest.raises(asyncio.CancelledError):
        async for _event in turn.execute():
            pass


@pytest.mark.unit
async def test_acp_turn_cancelled_error_during_stream_propagates() -> None:
    """Given CancelledError during stream, ACPTurn re-raises without yielding RunError."""
    client = MockACPClient(
        updates=[_text_update("partial")],
        stream_error=asyncio.CancelledError(),
    )
    run_ctx = _make_run_ctx()

    turn = ACPTurn(
        acp_client=client,
        prompts=["Hello"],
        run_ctx=run_ctx,
        session_id="test-session",
    )

    with pytest.raises(asyncio.CancelledError):
        async for _event in turn.execute():
            pass


@pytest.mark.unit
async def test_acp_turn_empty_prompts_uses_empty_string() -> None:
    """Given empty prompts list, ACPTurn sends empty string as content."""
    client = MockACPClient(
        updates=[],
        messages=[],
    )
    run_ctx = _make_run_ctx()

    turn = ACPTurn(
        acp_client=client,
        prompts=[],
        run_ctx=run_ctx,
        session_id="test-session",
    )

    events = [event async for event in turn.execute()]

    # Should still yield StreamComplete
    # (RunStartedEvent is published by RunHandle.start(), not turn.execute())
    assert len(events) == 1
    assert isinstance(events[0], StreamCompleteEvent)

    # Prompt was called with empty content
    assert len(client.prompt_calls) == 1
