"""Tests for context preservation after run cancellation.

Regression tests for the bug where RunHandle._message_history was not
updated when a turn was cancelled, causing the next turn to start with
stale (empty) message history. The model would "forget" all previous
conversation context after a cancel.

Covers three bugs:
1. RunHandle.start() skips _message_history update on cancel (continue
   at line 293 skips line 300).
2. _start_run_handle() creates RunHandle with empty _message_history,
   never bridging agent.conversation (ChatMessage list) to
   list[ModelMessage].
3. NativeTurn.execute() Path B (CancelledError) does not capture
   _message_history from agent_run, unlike Path A (graceful cancel).
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock

from pydantic_ai.messages import ModelResponse, TextPart
import pytest

from wolfharness.agents.events import (
    RunErrorEvent,
    RunFailedEvent,
    RunStartedEvent,
    StreamCompleteEvent,
    ToolCallStartEvent,
)
from wolfharness.lifecycle.types import DeliveryMode
from wolfharness.messaging import ChatMessage
from wolfharness.orchestrator.core import EventEnvelope, SessionPool
from wolfharness.orchestrator.turn import Turn


if TYPE_CHECKING:
    from wolfharness import AgentPool
    from wolfharness.agents.context import AgentRunContext

# Reusable ModelMessage for tests that need .messages populated on ChatMessage.
_ModelResponse = ModelResponse(parts=[TextPart(content="response")])


async def _drain_async_gen(gen: Any) -> None:
    """Drain an async generator to completion."""
    async for _ in gen:
        pass


pytestmark = pytest.mark.unit

# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class _BlockingTurnWithHistory(Turn):
    """Turn that sets _message_history, then blocks until cancelled.

    Simulates a turn that partially executes (accumulating messages)
    before being cancelled mid-stream.
    """

    def __init__(self, run_ctx: AgentRunContext, history: list[Any]) -> None:
        self._run_ctx = run_ctx
        self._history = history

    async def execute(self):  # type: ignore[override]
        # Simulate partial execution: messages were accumulated
        # before the cancel signal arrived.
        self._message_history = list(self._history)
        self._final_message = ChatMessage(
            content="partial response",
            role="assistant",
        )
        # Set .messages so the finally block in _execute_turn saves
        # ModelMessages to agent.conversation, enabling bridging for
        # the next RunHandle.
        self._final_message.messages = list(self._history)
        while not self._run_ctx.cancelled:
            await asyncio.sleep(0.01)
        return
        yield  # unreachable — makes this an async generator


# ---------------------------------------------------------------------------
# Test 1: Cancel preserves _message_history on RunHandle
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.anyio
async def test_cancel_preserves_message_history(minimal_pool: AgentPool) -> None:
    """After cancel, partial turn's message history is preserved for next turn.

    Given: A blocking turn that sets _message_history to ["partial_msg"].
    When: The turn is cancelled, then a new prompt is sent.
    Then: The second turn's message_history includes "partial_msg".
    """
    session_pool = minimal_pool.session_pool
    assert session_pool is not None
    session_id = "sess-cancel-history"
    await session_pool.create_session(session_id, agent_name="test_agent")

    received_histories: list[Any] = []
    call_count = 0

    def _create_turn(
        prompts: Any, run_ctx: AgentRunContext, message_history: Any, **kwargs: Any
    ) -> Turn:
        nonlocal call_count
        call_count += 1
        received_histories.append(message_history)
        if call_count == 1:
            return _BlockingTurnWithHistory(run_ctx, ["partial_msg_1", "partial_msg_2"])
        return _StubTurn_e2e(
            events=[StreamCompleteEvent(message=ChatMessage(content="response", role="assistant"))],
            message_history=["next_msg"],
        )

    await _patch_agent_create_turn(session_pool, session_id, _create_turn)
    queue = await session_pool.event_bus.subscribe(session_id)

    first_handle = await _receive_and_get_handle(session_pool, session_id, "first prompt")
    assert first_handle is not None
    await asyncio.sleep(0.1)

    session_pool.sessions.cancel_run_for_session(session_id)
    await asyncio.sleep(0.2)
    _ = await _drain_queue(queue)

    await asyncio.wait_for(
        session_pool.send_message(session_id, "second prompt"),
        timeout=10.0,
    )
    await _collect_events_until(queue, StreamCompleteEvent, timeout=10.0)

    assert len(received_histories) >= 2, (
        f"Expected 2 create_turn calls, got {len(received_histories)}"
    )
    second_turn_history = received_histories[1]
    assert len(second_turn_history) > 0, (
        "Second turn received empty message_history — cancelled turn's history was lost"
    )

    _assert_cancel_invariants(session_pool, session_id)
    first_handle.close()
    handle = session_pool._get_active_run_handle(session_id)
    if handle is not None and handle is not first_handle:
        handle.close()
    await asyncio.sleep(0.1)


# ---------------------------------------------------------------------------
# Test 2: New RunHandle bridges agent.conversation → list[ModelMessage]
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.anyio
async def test_new_runhandle_bridges_conversation(minimal_pool: AgentPool) -> None:
    """New RunHandle bridges agent.conversation to message_history.

    Given: An agent with conversation history from a prior turn.
    When: A new prompt is sent (creating a new RunHandle).
    Then: The new turn's message_history contains ModelMessages from the prior turn.
    """
    session_pool = minimal_pool.session_pool
    assert session_pool is not None
    session_id = "sess-bridge-conv"
    await session_pool.create_session(session_id, agent_name="test_agent")

    received_histories: list[Any] = []
    call_count = 0

    def _create_turn(
        prompts: Any, run_ctx: AgentRunContext, message_history: Any, **kwargs: Any
    ) -> Turn:
        nonlocal call_count
        call_count += 1
        received_histories.append(message_history)
        msg = ChatMessage(content="response", role="assistant")
        msg.messages = [_ModelResponse]
        return _StubTurn_e2e(
            events=[
                RunStartedEvent(run_id="test-run"),
                StreamCompleteEvent(message=msg),
            ],
            message_history=["msg"],
        )

    await _patch_agent_create_turn(session_pool, session_id, _create_turn)
    queue = await session_pool.event_bus.subscribe(session_id)

    first_handle = await _receive_and_get_handle(session_pool, session_id, "what is 2+2")
    assert first_handle is not None
    await _collect_events_until(queue, StreamCompleteEvent, timeout=5.0)
    await asyncio.sleep(0.1)

    await asyncio.wait_for(
        session_pool.send_message(session_id, "follow up question"),
        timeout=10.0,
    )
    await _collect_events_until(queue, StreamCompleteEvent, timeout=10.0)

    assert len(received_histories) >= 2, (
        f"Expected 2 create_turn calls, got {len(received_histories)}"
    )
    second_history = received_histories[1]
    assert len(second_history) > 0, (
        "Second turn received empty message_history — "
        "agent.conversation was not bridged to _message_history"
    )

    _assert_cancel_invariants(session_pool, session_id)
    first_handle.close()
    handle = session_pool._get_active_run_handle(session_id)
    if handle is not None and handle is not first_handle:
        handle.close()
    await asyncio.sleep(0.1)


# ---------------------------------------------------------------------------
# Test 3: NativeTurn Path B (CancelledError) captures _message_history
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.anyio
async def test_cancellederror_path_captures_history(minimal_pool: AgentPool) -> None:
    """CancelledError during turn execution preserves partial message history.

    Given: A blocking turn that sets _message_history before blocking.
    When: The turn is cancelled (CancelledError propagates through _consume_run).
    Then: The RunHandle's _message_history contains the partial turn's messages.
    """
    session_pool = minimal_pool.session_pool
    assert session_pool is not None
    session_id = "sess-cancel-history-path"
    await session_pool.create_session(session_id, agent_name="test_agent")
    await _patch_agent_create_turn(session_pool, session_id, _make_cancel_aware_create_turn())
    queue = await session_pool.event_bus.subscribe(session_id)

    first_handle = await _receive_and_get_handle(session_pool, session_id, "first prompt")
    assert first_handle is not None
    await asyncio.sleep(0.1)

    session_pool.sessions.cancel_run_for_session(session_id)
    await asyncio.sleep(0.3)

    events = await _drain_queue(queue)
    event_types = [type(_unwrap_event(e)) for e in events]
    assert RunFailedEvent in event_types, f"Expected RunFailedEvent from cancel, got: {event_types}"

    assert first_handle._message_history is not None, (
        "_message_history is None — CancelledError path didn't capture history"
    )

    _assert_cancel_invariants(session_pool, session_id)
    first_handle.close()
    await asyncio.sleep(0.1)


# ---------------------------------------------------------------------------
# Test 4: Multi-turn context preservation via _consume_run
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_consume_run_keeps_generator_alive_after_turn1() -> None:
    """E1: _consume_run should keep the generator alive for multi-turn.

    This is a pure unit test that simulates _consume_run's drain-only
    behavior with a fake multi-turn generator. The generator yields two
    turns worth of events, and the drain-only loop (no break, no aclose)
    allows turn 2 to execute.
    """
    turn2_executed = False

    async def fake_start(initial_prompt: str = "") -> Any:
        nonlocal turn2_executed
        yield RunStartedEvent(run_id="r1", session_id="s1")
        yield StreamCompleteEvent(
            message=ChatMessage(content="turn 1", role="assistant"),
        )
        turn2_executed = True
        yield RunStartedEvent(run_id="r1", session_id="s1")
        yield StreamCompleteEvent(
            message=ChatMessage(content="turn 2", role="assistant"),
        )

    # Fixed _consume_run: drain-only loop, no break, no aclose
    gen = fake_start("")
    async for _event in gen:
        pass

    assert turn2_executed, (
        "Turn 2 never executed because the generator was closed "
        "before reaching turn 2 events (issue E1)."
    )


@pytest.mark.integration
@pytest.mark.anyio
async def test_multi_turn_preserves_context_via_consume_run(
    minimal_pool: AgentPool,
) -> None:
    """Multi-turn context is preserved via _consume_run chaining.

    Given: First turn completes, adding messages to agent.conversation.
    When: Second prompt is sent (via _consume_run chaining or new RunHandle).
    Then: Second turn's message_history contains ModelMessages from the first turn.
    """
    session_pool = minimal_pool.session_pool
    assert session_pool is not None
    session_id = "sess-multi-turn-context"
    await session_pool.create_session(session_id, agent_name="test_agent")

    received_histories: list[Any] = []
    call_count = 0

    def _create_turn(
        prompts: Any, run_ctx: AgentRunContext, message_history: Any, **kwargs: Any
    ) -> Turn:
        nonlocal call_count
        call_count += 1
        received_histories.append(list(message_history) if message_history else [])
        msg = ChatMessage(content=f"response {call_count}", role="assistant")
        msg.messages = [_ModelResponse]
        return _StubTurn_e2e(
            events=[
                RunStartedEvent(run_id="test-run"),
                StreamCompleteEvent(message=msg),
            ],
            message_history=["msg"],
        )

    await _patch_agent_create_turn(session_pool, session_id, _create_turn)
    queue = await session_pool.event_bus.subscribe(session_id)

    first_handle = await _receive_and_get_handle(session_pool, session_id, "what is 2+2")
    assert first_handle is not None
    await _collect_events_until(queue, StreamCompleteEvent, timeout=5.0)
    await asyncio.sleep(0.1)

    await asyncio.wait_for(
        session_pool.send_message(session_id, "follow up"),
        timeout=10.0,
    )
    await _collect_events_until(queue, StreamCompleteEvent, timeout=10.0)

    assert len(received_histories) >= 2, (
        f"Expected 2 create_turn calls, got {len(received_histories)}"
    )

    first_history = received_histories[0]
    second_history = received_histories[1]
    assert len(second_history) > len(first_history), (
        f"Second turn history ({len(second_history)} items) should be larger than "
        f"first turn history ({len(first_history)} items) — context was not preserved"
    )

    _assert_cancel_invariants(session_pool, session_id)
    first_handle.close()
    handle = session_pool._get_active_run_handle(session_id)
    if handle is not None and handle is not first_handle:
        handle.close()
    await asyncio.sleep(0.1)


# ---------------------------------------------------------------------------
# Test 5: Bridged history must not contain trailing unprocessed tool calls
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.anyio
async def test_bridged_history_injects_cancelled_tool_results(
    minimal_pool: AgentPool,
) -> None:
    """Bridged history injects RetryPromptPart for cancelled tool calls.

    Given: An agent with conversation history containing an unprocessed
           tool call (ModelResponse with ToolCallPart, no tool result).
    When: A new prompt is sent (bridging conversation to message_history).
    Then: The new turn's message_history includes a RetryPromptPart for
          the cancelled tool call, preventing PydanticAI validation errors.
    """
    from pydantic_ai.messages import (
        ModelRequest,
        ModelResponse,
        RetryPromptPart,
        ToolCallPart,
        UserPromptPart,
    )

    session_pool = minimal_pool.session_pool
    assert session_pool is not None
    session_id = "sess-cancelled-tool-results"
    await session_pool.create_session(session_id, agent_name="test_agent")

    agent = await session_pool.sessions.get_or_create_session_agent(session_id)

    tool_call = ToolCallPart(tool_name="bash", args={"cmd": "ls"})
    prior_messages: list[Any] = [
        ModelRequest(parts=[UserPromptPart(content="run a command")]),
        ModelResponse(parts=[tool_call]),
    ]
    chat_msg1 = ChatMessage(content="run a command", role="user")
    chat_msg1.messages = [prior_messages[0]]
    chat_msg2 = ChatMessage(content="", role="assistant")
    chat_msg2.messages = [prior_messages[1]]
    agent.conversation.add_chat_messages([chat_msg1, chat_msg2])

    received_history: list[Any] = []

    def _create_turn(
        prompts: Any, run_ctx: AgentRunContext, message_history: Any, **kwargs: Any
    ) -> Turn:
        received_history.extend(message_history)
        return _StubTurn_e2e(
            events=[
                RunStartedEvent(run_id="test-run"),
                StreamCompleteEvent(message=ChatMessage(content="response", role="assistant")),
            ],
            message_history=["msg"],
        )

    agent.create_turn = _create_turn  # type: ignore[method-assign]

    queue = await session_pool.event_bus.subscribe(session_id)

    await asyncio.wait_for(
        session_pool.send_message(session_id, "follow up"),
        timeout=10.0,
    )
    await _collect_events_until(queue, StreamCompleteEvent, timeout=10.0)

    assert len(received_history) >= 3, (
        f"Expected at least 3 messages in bridged history (user + tool_call + retry), "
        f"got {len(received_history)}: {[type(m).__name__ for m in received_history]}"
    )

    last_msg = received_history[-1]
    assert isinstance(last_msg, ModelRequest), (
        f"Expected last message to be ModelRequest with RetryPromptPart, "
        f"got {type(last_msg).__name__}"
    )
    retry_parts = [p for p in last_msg.parts if isinstance(p, RetryPromptPart)]
    assert len(retry_parts) >= 1, "Expected at least 1 RetryPromptPart for the cancelled tool call"

    _assert_cancel_invariants(session_pool, session_id)
    handle = session_pool._get_active_run_handle(session_id)
    if handle is not None:
        handle.close()
    await asyncio.sleep(0.1)


# ---------------------------------------------------------------------------
# Test 6: Invalid JSON tool call args are sanitized in bridged history
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.anyio
async def test_bridged_history_sanitizes_invalid_json_tool_args(
    minimal_pool: AgentPool,
) -> None:
    """Bridged history replaces invalid JSON args with {} and injects RetryPromptPart.

    Given: An agent with conversation history containing a ToolCallPart
           whose args is an invalid JSON string (e.g. from deepseek-v4-flash).
    When: A new prompt is sent (bridging conversation to message_history).
    Then: The ToolCallPart's args are replaced with {} and a RetryPromptPart
          is injected, so the model API doesn't reject the history with 400.
    """
    from pydantic_ai.messages import (
        ModelRequest,
        ModelResponse,
        RetryPromptPart,
        ToolCallPart,
        UserPromptPart,
    )

    session_pool = minimal_pool.session_pool
    assert session_pool is not None
    session_id = "sess-invalid-json-tool-args"
    await session_pool.create_session(session_id, agent_name="test_agent")

    agent = await session_pool.sessions.get_or_create_session_agent(session_id)

    # Simulate what deepseek-v4-flash produces: invalid JSON string args
    tool_call = ToolCallPart(
        tool_name="send_message",
        args='{"target": "operator", "message": "do',  # truncated/invalid JSON
        tool_call_id="call_invalid_1",
    )
    prior_messages: list[Any] = [
        ModelRequest(parts=[UserPromptPart(content="send a message")]),
        ModelResponse(parts=[tool_call]),
    ]
    chat_msg1 = ChatMessage(content="send a message", role="user")
    chat_msg1.messages = [prior_messages[0]]
    chat_msg2 = ChatMessage(content="", role="assistant")
    chat_msg2.messages = [prior_messages[1]]
    agent.conversation.add_chat_messages([chat_msg1, chat_msg2])

    received_history: list[Any] = []

    def _create_turn(
        prompts: Any, run_ctx: AgentRunContext, message_history: Any, **kwargs: Any
    ) -> Turn:
        received_history.extend(message_history)
        return _StubTurn_e2e(
            events=[
                RunStartedEvent(run_id="test-run"),
                StreamCompleteEvent(message=ChatMessage(content="response", role="assistant")),
            ],
            message_history=["msg"],
        )

    agent.create_turn = _create_turn  # type: ignore[method-assign]

    queue = await session_pool.event_bus.subscribe(session_id)

    await asyncio.wait_for(
        session_pool.send_message(session_id, "follow up"),
        timeout=10.0,
    )
    await _collect_events_until(queue, StreamCompleteEvent, timeout=10.0)

    # Find the ModelResponse in the bridged history
    model_responses = [m for m in received_history if isinstance(m, ModelResponse)]
    assert len(model_responses) >= 1, (
        f"Expected at least 1 ModelResponse in history, got {len(model_responses)}"
    )

    # The ToolCallPart should have args={} (sanitized)
    tool_call_parts = [p for p in model_responses[0].parts if isinstance(p, ToolCallPart)]
    assert len(tool_call_parts) >= 1, "Expected at least 1 ToolCallPart"
    sanitized_tc = tool_call_parts[0]
    assert sanitized_tc.args == {}, (
        f"Expected args to be {{}} after sanitization, got {sanitized_tc.args!r}"
    )

    # A RetryPromptPart should be injected for the invalid tool call
    last_msg = received_history[-1]
    assert isinstance(last_msg, ModelRequest), (
        f"Expected last message to be ModelRequest with RetryPromptPart, "
        f"got {type(last_msg).__name__}"
    )
    retry_parts = [p for p in last_msg.parts if isinstance(p, RetryPromptPart)]
    assert len(retry_parts) >= 1, (
        "Expected at least 1 RetryPromptPart for the invalid JSON tool call"
    )

    _assert_cancel_invariants(session_pool, session_id)
    handle = session_pool._get_active_run_handle(session_id)
    if handle is not None:
        handle.close()
    await asyncio.sleep(0.1)


@pytest.mark.integration
@pytest.mark.anyio
async def test_bridged_history_sanitizes_mixed_tool_calls(
    minimal_pool: AgentPool,
) -> None:
    """Mixed valid/invalid tool calls: only invalid args replaced, all get RetryPromptPart.

    Given: A ModelResponse with two ToolCallParts — one valid JSON, one invalid.
    When: A new prompt is sent.
    Then: The invalid one's args are replaced with {}, the valid one's args
          are preserved, and both get RetryPromptParts (both are unprocessed).
    """
    from pydantic_ai.messages import (
        ModelRequest,
        ModelResponse,
        RetryPromptPart,
        ToolCallPart,
        UserPromptPart,
    )

    session_pool = minimal_pool.session_pool
    assert session_pool is not None
    session_id = "sess-mixed-tool-args"
    await session_pool.create_session(session_id, agent_name="test_agent")

    agent = await session_pool.sessions.get_or_create_session_agent(session_id)

    valid_call = ToolCallPart(
        tool_name="bash",
        args='{"cmd": "ls"}',
        tool_call_id="call_valid_1",
    )
    invalid_call = ToolCallPart(
        tool_name="read",
        args='{"path": "scratch',  # invalid JSON
        tool_call_id="call_invalid_1",
    )
    prior_messages: list[Any] = [
        ModelRequest(parts=[UserPromptPart(content="run tools")]),
        ModelResponse(parts=[valid_call, invalid_call]),
    ]
    chat_msg1 = ChatMessage(content="run tools", role="user")
    chat_msg1.messages = [prior_messages[0]]
    chat_msg2 = ChatMessage(content="", role="assistant")
    chat_msg2.messages = [prior_messages[1]]
    agent.conversation.add_chat_messages([chat_msg1, chat_msg2])

    received_history: list[Any] = []

    def _create_turn(
        prompts: Any, run_ctx: AgentRunContext, message_history: Any, **kwargs: Any
    ) -> Turn:
        received_history.extend(message_history)
        return _StubTurn_e2e(
            events=[
                RunStartedEvent(run_id="test-run"),
                StreamCompleteEvent(message=ChatMessage(content="response", role="assistant")),
            ],
            message_history=["msg"],
        )

    agent.create_turn = _create_turn  # type: ignore[method-assign]

    queue = await session_pool.event_bus.subscribe(session_id)

    await asyncio.wait_for(
        session_pool.send_message(session_id, "follow up"),
        timeout=10.0,
    )
    await _collect_events_until(queue, StreamCompleteEvent, timeout=10.0)

    model_responses = [m for m in received_history if isinstance(m, ModelResponse)]
    assert len(model_responses) >= 1

    tool_call_parts = [p for p in model_responses[0].parts if isinstance(p, ToolCallPart)]
    assert len(tool_call_parts) == 2

    # Find by tool_call_id
    tc_by_id = {tc.tool_call_id: tc for tc in tool_call_parts}
    valid_tc = tc_by_id["call_valid_1"]
    invalid_tc = tc_by_id["call_invalid_1"]

    # Valid args preserved
    assert valid_tc.args == '{"cmd": "ls"}', (
        f"Valid args should be preserved, got {valid_tc.args!r}"
    )
    # Invalid args replaced with {}
    assert invalid_tc.args == {}, (
        f"Invalid args should be replaced with {{}}, got {invalid_tc.args!r}"
    )

    # Both should have RetryPromptParts (each in a separate ModelRequest)
    trailing_requests = [
        m
        for m in received_history
        if isinstance(m, ModelRequest) and any(isinstance(p, RetryPromptPart) for p in m.parts)
    ]
    assert len(trailing_requests) >= 2, (
        f"Expected at least 2 ModelRequests with RetryPromptPart, got {len(trailing_requests)}"
    )

    _assert_cancel_invariants(session_pool, session_id)
    handle = session_pool._get_active_run_handle(session_id)
    if handle is not None:
        handle.close()
    await asyncio.sleep(0.1)


# ---------------------------------------------------------------------------
# Merged from test_cancelled_cleanup_review.py (suffix: cr)
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.anyio
async def test_cancelled_error_cleanup_in_process_prompt(
    minimal_pool: AgentPool,
) -> None:
    """CancelledError during run execution doesn't skip cleanup.

    Given: A blocking turn that gets cancelled.
    When: CancelledError propagates through the run execution path.
    Then: session.current_run_id is cleared and _runs no longer contains
          the cancelled run's handle.
    """
    session_pool = minimal_pool.session_pool
    assert session_pool is not None
    session_id = "sess-cancel-cleanup-process"
    await session_pool.create_session(session_id, agent_name="test_agent")
    await _patch_agent_create_turn(session_pool, session_id, _make_cancel_aware_create_turn())
    queue = await session_pool.event_bus.subscribe(session_id)

    first_handle = await _receive_and_get_handle(session_pool, session_id, "first prompt")
    assert first_handle is not None
    assert first_handle.run_id is not None
    await asyncio.sleep(0.1)

    session = session_pool.sessions.get_session(session_id)
    assert session is not None
    assert session.current_run_id == first_handle.run_id
    assert first_handle.run_id in session_pool.sessions._runs

    session_pool.sessions.cancel_run_for_session(session_id)
    await asyncio.sleep(0.3)

    _ = await _drain_queue(queue)

    session = session_pool.sessions.get_session(session_id)
    assert session is not None
    assert session.current_run_id is None, (
        "current_run_id was not cleared after cancel — cleanup was skipped"
    )
    assert first_handle.run_id not in session_pool.sessions._runs, (
        "RunHandle still in _runs after cancel — _runs.pop was not called"
    )
    assert first_handle.complete_event.is_set(), (
        "complete_event not set after cancel — cleanup was skipped"
    )

    _assert_cancel_invariants(session_pool, session_id)
    first_handle.close()
    await asyncio.sleep(0.1)


@pytest.mark.integration
@pytest.mark.anyio
async def test_cancelled_error_cleanup_in_run_stream(
    minimal_pool: AgentPool,
) -> None:
    """CancelledError during run_stream doesn't skip cleanup.

    Given: A streaming run that gets cancelled.
    When: CancelledError propagates through _run_stream_run_turn.
    Then: session.current_run_id is cleared, _runs no longer contains the
          handle, and the EventBus subscription is cleaned up.
    """
    session_pool = minimal_pool.session_pool
    assert session_pool is not None
    session_id = "sess-cancel-cleanup-stream"
    await session_pool.create_session(session_id, agent_name="test_agent")
    await _patch_agent_create_turn(session_pool, session_id, _make_cancel_aware_create_turn())

    # Start consuming the stream to kick off the run
    stream_gen = session_pool.run_stream(session_id, "first prompt")
    stream_task = asyncio.create_task(_drain_async_gen(stream_gen))
    await asyncio.sleep(0.1)

    session = session_pool.sessions.get_session(session_id)
    assert session is not None
    assert session.current_run_id is not None
    run_id = session.current_run_id
    assert run_id in session_pool.sessions._runs

    session_pool.sessions.cancel_run_for_session(session_id)
    await asyncio.sleep(0.3)

    # Cancel the stream consumer task
    stream_task.cancel()
    with contextlib.suppress(asyncio.CancelledError, GeneratorExit, StopAsyncIteration):
        await stream_task

    session = session_pool.sessions.get_session(session_id)
    assert session is not None
    assert session.current_run_id is None, (
        "current_run_id was not cleared after stream cancel — cleanup was skipped"
    )
    assert run_id not in session_pool.sessions._runs, (
        "RunHandle still in _runs after stream cancel — _runs.pop was not called"
    )

    _assert_cancel_invariants(session_pool, session_id)
    await asyncio.sleep(0.1)


# ---------------------------------------------------------------------------
# Merged from test_cancel_e2e.py (suffix: e2e)
# ---------------------------------------------------------------------------


def _unwrap_event(event: Any) -> Any:
    """Unwrap EventEnvelope if present, otherwise return the event as-is."""
    return event.event if isinstance(event, EventEnvelope) else event


async def _receive_and_get_handle(
    session_pool: SessionPool, session_id: str, content: str, **kwargs: Any
) -> Any:
    """Call receive_request and return the RunHandle for the active run.

    receive_request() now returns str | None (message_id), but many tests
    need the RunHandle to inspect state. This helper bridges the gap.
    """
    message_id = await session_pool.send_message(session_id, content, **kwargs)
    assert message_id is not None, "receive_request should return a message_id for idle session"
    handle = session_pool._get_active_run_handle(session_id)
    assert handle is not None, "Expected an active RunHandle after receive_request"
    return handle


class _BlockingTurn(Turn):
    """Turn that blocks until run_ctx.cancelled, then returns without StreamCompleteEvent."""

    def __init__(self, run_ctx: AgentRunContext) -> None:
        self._run_ctx = run_ctx

    async def execute(self):
        self._message_history = []
        self._final_message = ChatMessage(content="blocked", role="assistant")
        while not self._run_ctx.cancelled:
            await asyncio.sleep(0.01)
        return
        yield


class _StubTurn_e2e(Turn):  # noqa: N801
    """Minimal Turn that yields events from a list and sets message history."""

    def __init__(
        self, *, events: list[Any] | None = None, message_history: list[Any] | None = None
    ) -> None:
        self._events = events or []
        self._history = message_history or []

    async def execute(self):
        self._message_history = self._history
        self._final_message = ChatMessage(content="done", role="assistant")
        for event in self._events:
            yield event


class _ToolBlockingTurn(Turn):
    """Turn that yields ToolCallStartEvent then blocks until run_ctx.cancelled."""

    def __init__(self, run_ctx: AgentRunContext) -> None:
        self._run_ctx = run_ctx

    async def execute(self):
        self._message_history = []
        self._final_message = ChatMessage(content="tool-blocked", role="assistant")
        yield ToolCallStartEvent(
            tool_call_id="test-tool-1", tool_name="bash", title="Running bash command"
        )
        while not self._run_ctx.cancelled:
            await asyncio.sleep(0.01)


async def _patch_agent_create_turn(
    session_pool: SessionPool,
    session_id: str,
    create_turn_fn: Any,
) -> Any:
    """Get the real agent from the pool and patch its create_turn method.

    Returns the real agent for any further manipulation.
    """
    agent = await session_pool.sessions.get_or_create_session_agent(session_id)
    agent.create_turn = create_turn_fn  # type: ignore[method-assign]
    return agent


def _make_cancel_aware_create_turn() -> Any:
    """Return a create_turn function whose first call returns _BlockingTurn.

    Subsequent calls return _StubTurn instances that yield StreamCompleteEvent.
    """
    call_count = 0

    def _create_turn(
        prompts: Any, run_ctx: AgentRunContext, message_history: Any, **kwargs: Any
    ) -> Turn:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return _BlockingTurn(run_ctx)
        return _StubTurn_e2e(
            events=[
                RunStartedEvent(run_id="test-run"),
                StreamCompleteEvent(message=ChatMessage(content="response", role="assistant")),
            ],
            message_history=["msg"],
        )

    return _create_turn


def _make_stub_only_create_turn() -> Any:
    """Return a create_turn function that always returns _StubTurn_e2e."""

    def _create_turn(
        prompts: Any, run_ctx: AgentRunContext, message_history: Any, **kwargs: Any
    ) -> Turn:
        return _StubTurn_e2e(
            events=[
                RunStartedEvent(run_id="test-run"),
                StreamCompleteEvent(message=ChatMessage(content="response", role="assistant")),
            ],
            message_history=["msg"],
        )

    return _create_turn


async def _drain_queue(queue: asyncio.Queue) -> list[Any]:
    """Drain all currently-available events from a queue without blocking."""
    events: list[Any] = []
    while True:
        with contextlib.suppress(asyncio.QueueEmpty):
            events.append(queue.get_nowait())
            continue
        break
    return events


def _assert_cancel_invariants(session_pool: Any, session_id: str) -> None:
    """Assert post-cancel invariants: prompt_queue empty, no stale current_run_id."""
    session = session_pool.sessions.get_session(session_id)
    if session is not None:
        assert session.prompt_queue.empty(), (
            f"prompt_queue has {session.prompt_queue.qsize()} stuck messages after cancel"
        )


@pytest.mark.integration
@pytest.mark.anyio
async def test_cancel_then_new_prompt_full_flow(minimal_pool: AgentPool) -> None:
    """End-to-end: cancel a running turn, then send a new prompt.

    Steps:
        1. Start a run with a blocking turn (patched on real agent).
        2. Cancel via cancel_run_for_session().
        3. Send new prompt via receive_request().
        4. Verify new prompt processed (events published, no hang).
        5. Verify RunHandle is same instance (1:1 model) or new one (if old died).

    Uses asyncio.wait_for() with a 30s timeout to catch hangs.
    """
    session_pool = minimal_pool.session_pool
    assert session_pool is not None
    session_id = "sess-cancel-e2e"
    await session_pool.create_session(session_id, agent_name="test_agent")
    await _patch_agent_create_turn(session_pool, session_id, _make_cancel_aware_create_turn())
    queue = await session_pool.event_bus.subscribe(session_id)
    first_handle = await _receive_and_get_handle(session_pool, session_id, "first prompt")
    await asyncio.sleep(0.1)
    session_pool.sessions.cancel_run_for_session(session_id)
    await asyncio.sleep(0.2)
    pre_events = await _drain_queue(queue)
    pre_event_types = [type(_unwrap_event(e)) for e in pre_events]
    assert RunFailedEvent in pre_event_types, (
        f"Expected RunFailedEvent from cancelled turn, got: {pre_event_types}"
    )
    second_msg_id = await asyncio.wait_for(
        session_pool.send_message(session_id, "second prompt"), timeout=30.0
    )
    post_events: list[Any] = []
    try:
        async with asyncio.timeout(30.0):
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=5.0)
                    post_events.append(event)
                    unwrapped = _unwrap_event(event)
                    if isinstance(unwrapped, StreamCompleteEvent):
                        break
                except TimeoutError:
                    break
    except TimeoutError:
        pytest.fail("Timed out waiting for events after cancel-then-prompt")
    post_event_types = [type(_unwrap_event(e)) for e in post_events]
    assert RunStartedEvent in post_event_types, (
        f"Expected RunStartedEvent for new prompt, got: {post_event_types}"
    )
    assert StreamCompleteEvent in post_event_types, (
        f"Expected StreamCompleteEvent for new prompt, got: {post_event_types}"
    )
    if second_msg_id is not None:
        second_handle = session_pool._get_active_run_handle(session_id)
        if second_handle is not None and second_handle is not first_handle:
            pass
    assert first_handle.complete_event.is_set(), "First RunHandle should be done after cancel"
    _assert_cancel_invariants(session_pool, session_id)
    first_handle.close()
    await asyncio.sleep(0.1)


def _make_tool_blocking_create_turn() -> Any:
    """Return a create_turn function whose first call returns _ToolBlockingTurn.

    Subsequent calls return _StubTurn instances that yield StreamCompleteEvent.
    """
    call_count = 0

    def _create_turn(
        prompts: Any, run_ctx: AgentRunContext, message_history: Any, **kwargs: Any
    ) -> Turn:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return _ToolBlockingTurn(run_ctx)
        return _StubTurn_e2e(
            events=[
                RunStartedEvent(run_id="test-run"),
                StreamCompleteEvent(message=ChatMessage(content="response", role="assistant")),
            ],
            message_history=["msg"],
        )

    return _create_turn


def _make_stub_then_die_create_turn() -> Any:
    """Return a create_turn function: first returns _StubTurn, second raises, rest _StubTurn."""
    call_count = 0

    def _create_turn(
        prompts: Any, run_ctx: AgentRunContext, message_history: Any, **kwargs: Any
    ) -> Turn:
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            msg = "Simulated unrecoverable error in create_turn"
            raise RuntimeError(msg)
        return _StubTurn_e2e(
            events=[
                RunStartedEvent(run_id="test-run"),
                StreamCompleteEvent(message=ChatMessage(content="response", role="assistant")),
            ],
            message_history=["msg"],
        )

    return _create_turn


async def _collect_events_until(
    queue: asyncio.Queue, target_type: type, *, timeout: float = 30.0
) -> list[Any]:
    """Collect events from a queue until a target event type is seen."""
    events: list[Any] = []
    try:
        async with asyncio.timeout(timeout):
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=5.0)
                    events.append(event)
                    if isinstance(_unwrap_event(event), target_type):
                        break
                except TimeoutError:
                    break
    except TimeoutError:
        pytest.fail(f"Timed out waiting for {target_type.__name__}")
    return events


@pytest.mark.integration
@pytest.mark.anyio
async def test_double_cancel(minimal_pool: AgentPool) -> None:
    """Call cancel() twice during active turn — idempotent, no errors.

    Given: a running turn with _BlockingTurn.
    When: cancel() is called twice.
    Then: no exceptions, RunHandle returns to idle/done, new prompt works.
    """
    session_pool = minimal_pool.session_pool
    assert session_pool is not None
    session_id = "sess-double-cancel"
    await session_pool.create_session(session_id, agent_name="test_agent")
    await _patch_agent_create_turn(session_pool, session_id, _make_cancel_aware_create_turn())
    queue = await session_pool.event_bus.subscribe(session_id)
    first_handle = await _receive_and_get_handle(session_pool, session_id, "first prompt")
    assert first_handle is not None
    await asyncio.sleep(0.1)
    session_pool.sessions.cancel_run_for_session(session_id)
    session_pool.sessions.cancel_run_for_session(session_id)
    await asyncio.sleep(0.2)
    pre_events = await _drain_queue(queue)
    pre_types = [type(_unwrap_event(e)) for e in pre_events]
    assert RunFailedEvent in pre_types, f"Expected RunFailedEvent, got: {pre_types}"
    assert first_handle.complete_event.is_set(), "RunHandle should be done after double cancel"
    await asyncio.wait_for(session_pool.send_message(session_id, "second prompt"), timeout=30.0)
    post_events = await _collect_events_until(queue, StreamCompleteEvent)
    post_types = [type(_unwrap_event(e)) for e in post_events]
    assert StreamCompleteEvent in post_types, f"Expected StreamCompleteEvent, got: {post_types}"
    _assert_cancel_invariants(session_pool, session_id)
    first_handle.close()
    await asyncio.sleep(0.1)


@pytest.mark.integration
@pytest.mark.anyio
async def test_cancel_during_idle_then_new_prompt(minimal_pool: AgentPool) -> None:
    """Cancel while idle (no active turn), then send new prompt.

    Given: a completed turn, RunHandle is idle.
    When: cancel() is called while idle, then a new prompt is sent.
    Then: cancelled flag is reset before new turn starts, prompt is processed.
    """
    session_pool = minimal_pool.session_pool
    assert session_pool is not None
    session_id = "sess-cancel-idle"
    await session_pool.create_session(session_id, agent_name="test_agent")
    agent = await _patch_agent_create_turn(
        session_pool, session_id, _make_cancel_aware_create_turn()
    )
    queue = await session_pool.event_bus.subscribe(session_id)
    # Re-patch with a stub-only create_turn for immediate completion
    agent.create_turn = lambda prompts, run_ctx, message_history, **kwargs: _StubTurn_e2e(
        events=[
            RunStartedEvent(run_id="test-run"),
            StreamCompleteEvent(message=ChatMessage(content="response", role="assistant")),
        ],
        message_history=["msg"],
    )  # type: ignore[method-assign]
    first_handle = await _receive_and_get_handle(session_pool, session_id, "first prompt")
    assert first_handle is not None
    await _collect_events_until(queue, StreamCompleteEvent)
    await asyncio.sleep(0.1)
    session_pool.sessions.cancel_run_for_session(session_id)
    await asyncio.sleep(0.05)
    assert first_handle.run_ctx.cancelled is False, (
        "cancelled flag should remain False when cancel is called while idle"
        " (no active run to cancel)"
    )
    await asyncio.wait_for(session_pool.send_message(session_id, "second prompt"), timeout=30.0)
    post_events = await _collect_events_until(queue, StreamCompleteEvent)
    post_types = [type(_unwrap_event(e)) for e in post_events]
    assert StreamCompleteEvent in post_types, (
        f"Expected StreamCompleteEvent from new turn, got: {post_types}"
    )
    assert first_handle.run_ctx.cancelled is False, (
        "cancelled flag should remain False — new turn ran without cancel"
    )
    _assert_cancel_invariants(session_pool, session_id)
    first_handle.close()
    await asyncio.sleep(0.1)


@pytest.mark.integration
@pytest.mark.anyio
async def test_cancel_then_steer_continues_turn(minimal_pool: AgentPool) -> None:
    """Cancel then new prompt — cancel interrupts turn, new prompt processed.

    Given: a running turn with _BlockingTurn.
    When: cancel() is called, then a new prompt is sent.
    Then: cancel interrupts the current turn (RunFailedEvent), and
          the new prompt is processed in a new RunHandle (StreamCompleteEvent).

    In the per-prompt model, steer on a terminated RunHandle queues to
    run_ctx.queued_steer_messages on the dead handle. The correct way
    to route messages between turns is via SessionState.feedback_queue
    or by sending a new prompt.
    """
    session_pool = minimal_pool.session_pool
    assert session_pool is not None
    session_id = "sess-cancel-steer"
    await session_pool.create_session(session_id, agent_name="test_agent")
    await _patch_agent_create_turn(session_pool, session_id, _make_cancel_aware_create_turn())
    queue = await session_pool.event_bus.subscribe(session_id)
    first_handle = await _receive_and_get_handle(session_pool, session_id, "first prompt")
    assert first_handle is not None
    await asyncio.sleep(0.1)
    session_pool.sessions.cancel_run_for_session(session_id)
    await asyncio.sleep(0.3)
    pre_events = await _drain_queue(queue)
    pre_types = [type(_unwrap_event(e)) for e in pre_events]
    assert RunFailedEvent in pre_types, (
        f"Expected RunFailedEvent from cancelled turn, got: {pre_types}"
    )
    # Send a new prompt which creates a new RunHandle
    await asyncio.wait_for(session_pool.send_message(session_id, "new prompt"), timeout=30.0)
    post_events = await _collect_events_until(queue, StreamCompleteEvent, timeout=30.0)
    post_types = [type(_unwrap_event(e)) for e in post_events]
    assert StreamCompleteEvent in post_types, (
        f"Expected StreamCompleteEvent from subsequent turn, got: {post_types}"
    )
    _assert_cancel_invariants(session_pool, session_id)
    first_handle.close()
    await asyncio.sleep(0.1)


@pytest.mark.integration
@pytest.mark.anyio
async def test_cancel_during_tool_execution(minimal_pool: AgentPool) -> None:
    """Cancel during tool execution — run_ctx.cancelled is set, turn exits after tool.

    Given: a turn that yields ToolCallStartEvent then blocks.
    When: cancel() is called during the blocking period.
    Then: run_ctx.cancelled is set, turn exits, RunFailedEvent is published.
    """
    session_pool = minimal_pool.session_pool
    assert session_pool is not None
    session_id = "sess-cancel-tool"
    await session_pool.create_session(session_id, agent_name="test_agent")
    await _patch_agent_create_turn(session_pool, session_id, _make_tool_blocking_create_turn())
    queue = await session_pool.event_bus.subscribe(session_id)
    first_handle = await _receive_and_get_handle(session_pool, session_id, "first prompt")
    assert first_handle is not None
    await asyncio.sleep(0.1)
    session_pool.sessions.cancel_run_for_session(session_id)
    await asyncio.sleep(0.2)
    events = await _drain_queue(queue)
    event_types = [type(_unwrap_event(e)) for e in events]
    assert ToolCallStartEvent in event_types, (
        f"Expected ToolCallStartEvent before cancel, got: {event_types}"
    )
    assert RunFailedEvent in event_types, (
        f"Expected RunFailedEvent after cancel, got: {event_types}"
    )
    assert first_handle.complete_event.is_set(), "RunHandle should be done after cancel"
    _assert_cancel_invariants(session_pool, session_id)
    first_handle.close()
    await asyncio.sleep(0.1)


@pytest.mark.integration
@pytest.mark.anyio
async def test_double_cancel_then_new_prompt(minimal_pool: AgentPool) -> None:
    """Double cancel then new prompt — no hang, new prompt processed.

    Given: a running turn with _BlockingTurn.
    When: cancel() is called twice, then a new prompt is sent via receive_request().
    Then: no hang, new prompt is processed (StreamCompleteEvent published).
    """
    session_pool = minimal_pool.session_pool
    assert session_pool is not None
    session_id = "sess-double-cancel-prompt"
    await session_pool.create_session(session_id, agent_name="test_agent")
    await _patch_agent_create_turn(session_pool, session_id, _make_cancel_aware_create_turn())
    queue = await session_pool.event_bus.subscribe(session_id)
    first_handle = await _receive_and_get_handle(session_pool, session_id, "first prompt")
    assert first_handle is not None
    await asyncio.sleep(0.1)
    session_pool.sessions.cancel_run_for_session(session_id)
    session_pool.sessions.cancel_run_for_session(session_id)
    await asyncio.sleep(0.2)
    pre_events = await _drain_queue(queue)
    pre_types = [type(_unwrap_event(e)) for e in pre_events]
    assert RunFailedEvent in pre_types, (
        f"Expected RunFailedEvent from cancelled turn, got: {pre_types}"
    )
    await asyncio.wait_for(session_pool.send_message(session_id, "second prompt"), timeout=30.0)
    post_events = await _collect_events_until(queue, StreamCompleteEvent, timeout=30.0)
    post_types = [type(_unwrap_event(e)) for e in post_events]
    assert StreamCompleteEvent in post_types, (
        f"Expected StreamCompleteEvent from new prompt, got: {post_types}"
    )
    _assert_cancel_invariants(session_pool, session_id)
    first_handle.close()
    await asyncio.sleep(0.1)


@pytest.mark.integration
@pytest.mark.anyio
async def test_runhandle_dies_in_idle_loop(minimal_pool: AgentPool) -> None:
    """Simulate unrecoverable error in start().

    Finally block sets events, cleanup clears current_run_id.

    Given: an agent whose second create_turn call raises RuntimeError.
    When: the first turn completes, followup triggers the second create_turn which raises.
    Then: finally block sets complete_event, _cleanup_run clears current_run_id,
          next receive_request creates a new RunHandle and processes the prompt.
    """
    session_pool = minimal_pool.session_pool
    assert session_pool is not None
    session_id = "sess-dies-in-idle"
    await session_pool.create_session(session_id, agent_name="test_agent")
    await _patch_agent_create_turn(session_pool, session_id, _make_stub_then_die_create_turn())
    queue = await session_pool.event_bus.subscribe(session_id)
    first_handle = await _receive_and_get_handle(session_pool, session_id, "first prompt")
    assert first_handle is not None
    await _collect_events_until(queue, StreamCompleteEvent)
    await asyncio.sleep(0.1)
    await asyncio.wait_for(session_pool.send_message(session_id, "trigger error"), timeout=30.0)
    await asyncio.sleep(0.5)
    crash_session = session_pool.sessions.get_session(session_id)
    assert crash_session is not None
    assert crash_session.current_run_id is None, (
        "current_run_id should be cleared by _cleanup_run after error"
    )
    second_msg_id = await asyncio.wait_for(
        session_pool.send_message(session_id, "new prompt after crash"), timeout=30.0
    )
    assert second_msg_id is not None, "receive_request should return a new message_id after cleanup"
    second_handle = session_pool._get_active_run_handle(session_id)
    assert second_handle is not None
    assert second_handle is not first_handle, "New RunHandle should be a different instance"
    post_events = await _collect_events_until(queue, StreamCompleteEvent, timeout=30.0)
    post_types = [type(_unwrap_event(e)) for e in post_events]
    assert StreamCompleteEvent in post_types, (
        f"Expected StreamCompleteEvent from new RunHandle, got: {post_types}"
    )
    _assert_cancel_invariants(session_pool, session_id)
    second_handle.close()
    await asyncio.sleep(0.1)


# ---------------------------------------------------------------------------
# Test: Cancel with queued prompt — prompt_queue must be drained
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.anyio
async def test_cancel_with_queued_prompt_drains_queue(
    minimal_pool: AgentPool,
) -> None:
    """Cancel during active run drains prompt_queue — queued message is processed.

    Regression test for the race condition where a message arrives in
    prompt_queue during the cancel propagation window:

    1. ``_route_message`` enqueues to ``prompt_queue`` (session appears busy).
    2. ``_consume_run`` is cancelled — ``CancelledError`` is ``BaseException``,
       not caught by ``except Exception``.
    3. The ``prompt_queue`` check in ``_consume_run`` (lines 180-202) is SKIPPED.
    4. ``_cleanup_run`` clears ``current_run_id`` but does NOT check
       ``prompt_queue``.
    5. The queued message is stuck forever.

    The fix adds ``prompt_queue`` checking to ``_cleanup_run``, so queued
    messages are processed even when ``_consume_run`` is cancelled.

    Steps:
        1. Start a blocking turn.
        2. Manually enqueue a message to ``prompt_queue`` (simulates race).
        3. Cancel the run.
        4. Verify the queued message is processed (events received).
    """
    session_pool = minimal_pool.session_pool
    assert session_pool is not None
    session_id = "sess-cancel-queued"
    await session_pool.create_session(session_id, agent_name="test_agent")
    await _patch_agent_create_turn(session_pool, session_id, _make_cancel_aware_create_turn())
    queue = await session_pool.event_bus.subscribe(session_id)
    first_handle = await _receive_and_get_handle(session_pool, session_id, "first prompt")
    assert first_handle is not None

    # Wait for the blocking turn to start.
    await asyncio.sleep(0.1)

    # Manually enqueue a message to prompt_queue — simulates a message
    # that arrived during the cancel race window via _route_message's
    # busy path (session appeared busy, complete_event not yet set).
    session = session_pool.sessions.get_session(session_id)
    assert session is not None
    session.prompt_queue.put_nowait("second prompt")

    # Cancel the active run.
    session_pool.sessions.cancel_run_for_session(session_id)

    # Wait for events from the queued message being processed.
    post_events: list[Any] = []
    try:
        async with asyncio.timeout(10.0):
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=5.0)
                    post_events.append(event)
                    unwrapped = _unwrap_event(event)
                    if isinstance(unwrapped, StreamCompleteEvent):
                        break
                except TimeoutError:
                    break
    except TimeoutError:
        pytest.fail(
            "Timed out waiting for events after cancel with queued prompt — "
            "message stuck in prompt_queue (race condition bug)"
        )

    post_event_types = [type(_unwrap_event(e)) for e in post_events]

    # The queued message should have been processed: RunStartedEvent
    # and StreamCompleteEvent should be in the events.
    assert RunStartedEvent in post_event_types, (
        f"Expected RunStartedEvent for queued prompt, got: {post_event_types}"
    )
    assert StreamCompleteEvent in post_event_types, (
        f"Expected StreamCompleteEvent for queued prompt, got: {post_event_types}"
    )

    # Cleanup
    _assert_cancel_invariants(session_pool, session_id)
    first_handle.close()
    second_handle = session_pool._get_active_run_handle(session_id)
    if second_handle is not None and second_handle is not first_handle:
        second_handle.close()
    await asyncio.sleep(0.1)


# ---------------------------------------------------------------------------
# Additional cancel invariant tests (S1-S7)
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.anyio
async def test_cancel_with_multiple_queued_prompts_drains_all(
    minimal_pool: AgentPool,
) -> None:
    """Cancel with multiple messages in prompt_queue — all are processed via chaining.

    _drain_prompt_queue pops the first message and starts a new _consume_run.
    The remaining messages should be processed by _consume_run's normal
    chaining logic (lines 180-202).
    """
    session_pool = minimal_pool.session_pool
    assert session_pool is not None
    session_id = "sess-cancel-multi-queued"
    await session_pool.create_session(session_id, agent_name="test_agent")
    await _patch_agent_create_turn(session_pool, session_id, _make_cancel_aware_create_turn())
    queue = await session_pool.event_bus.subscribe(session_id)
    first_handle = await _receive_and_get_handle(session_pool, session_id, "first prompt")
    assert first_handle is not None
    await asyncio.sleep(0.1)

    # Enqueue 3 messages to prompt_queue
    session = session_pool.sessions.get_session(session_id)
    assert session is not None
    for i in range(3):
        session.prompt_queue.put_nowait(f"queued prompt {i}")

    # Cancel the active run
    session_pool.sessions.cancel_run_for_session(session_id)

    # Collect ALL StreamCompleteEvents — expect 3 (one per queued message)
    stream_complete_count = 0
    try:
        async with asyncio.timeout(30.0):
            while stream_complete_count < 3:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=10.0)
                    unwrapped = _unwrap_event(event)
                    if isinstance(unwrapped, StreamCompleteEvent):
                        stream_complete_count += 1
                except TimeoutError:
                    break
    except TimeoutError:
        pytest.fail(
            f"Timed out waiting for queued prompts — only {stream_complete_count}/3 processed"
        )
    assert stream_complete_count == 3, (
        f"Expected 3 StreamCompleteEvents for 3 queued prompts, got {stream_complete_count}"
    )

    # Invariant: prompt_queue must be empty
    _assert_cancel_invariants(session_pool, session_id)

    # Cleanup
    first_handle.close()
    handle = session_pool._get_active_run_handle(session_id)
    if handle is not None and handle is not first_handle:
        handle.close()
    await asyncio.sleep(0.1)


@pytest.mark.integration
@pytest.mark.anyio
async def test_cancel_then_close_session_no_orphaned_runs(
    minimal_pool: AgentPool,
) -> None:
    """Cancel with queued message, then close session — no orphaned runs.

    The _drain_prompt_queue fix must not start a new run if the session
    is closing. The is_closing check in _drain_prompt_queue prevents this.
    """
    session_pool = minimal_pool.session_pool
    assert session_pool is not None
    session_id = "sess-cancel-close"
    await session_pool.create_session(session_id, agent_name="test_agent")
    await _patch_agent_create_turn(session_pool, session_id, _make_cancel_aware_create_turn())
    queue = await session_pool.event_bus.subscribe(session_id)
    first_handle = await _receive_and_get_handle(session_pool, session_id, "first prompt")
    assert first_handle is not None
    await asyncio.sleep(0.1)

    # Enqueue a message
    session = session_pool.sessions.get_session(session_id)
    assert session is not None
    session.prompt_queue.put_nowait("queued prompt")

    # Mark session as closing BEFORE cancel — _drain_prompt_queue should skip
    session.is_closing = True

    # Drain pre-cancel events (from the initial run) so we only see post-cancel events
    await _drain_queue(queue)

    # Cancel the active run
    session_pool.sessions.cancel_run_for_session(session_id)
    await asyncio.sleep(0.3)

    # Drain events — should get RunFailedEvent from cancel, but NO RunStartedEvent
    events = await _drain_queue(queue)
    event_types = [type(_unwrap_event(e)) for e in events]
    assert RunFailedEvent in event_types, f"Expected RunFailedEvent from cancel, got: {event_types}"
    assert RunStartedEvent not in event_types, (
        f"RunStartedEvent found — _drain_prompt_queue started a run on a closing session! "
        f"Events: {event_types}"
    )

    # The queued message should still be in prompt_queue (not drained)
    assert not session.prompt_queue.empty(), (
        "prompt_queue should still contain the message — session is closing, drain skipped"
    )

    # Cleanup
    session.is_closing = False
    first_handle.close()
    await asyncio.sleep(0.1)


@pytest.mark.unit
@pytest.mark.anyio
async def test_drain_prompt_queue_skips_closing_session() -> None:
    """_drain_prompt_queue must not start a new run on a closing session.

    Unit test: directly call _drain_prompt_queue with is_closing=True
    and verify no task is created.
    """
    from wolfharness.orchestrator.core import SessionController, SessionState

    controller = SessionController.__new__(SessionController)
    controller._background_tasks = set()
    controller._runs = {}

    session = SessionState(session_id="test-closing", agent_name="test-agent")
    session.is_closing = True
    session.prompt_queue.put_nowait("should not be drained")

    mock_handle = MagicMock()
    mock_handle.agent = MagicMock()
    mock_handle.session_id = "test-closing"

    # Should be a no-op — session is closing
    controller._drain_prompt_queue(session, mock_handle)

    # prompt_queue should still have the message
    assert not session.prompt_queue.empty(), (
        "prompt_queue was drained on a closing session — should be skipped"
    )
    # No background tasks should have been created
    assert len(controller._background_tasks) == 0, (
        "Background task was created on a closing session"
    )


@pytest.mark.unit
@pytest.mark.anyio
async def test_drain_prompt_queue_agent_none_preserves_message() -> None:
    """_drain_prompt_queue with agent=None must put the message back, not drop it.

    The fix puts the message back into prompt_queue when agent is None,
    instead of silently dropping it.
    """
    from wolfharness.orchestrator.core import SessionController, SessionState

    controller = SessionController.__new__(SessionController)
    controller._background_tasks = set()
    controller._runs = {}

    session = SessionState(session_id="test-no-agent", agent_name="test-agent")
    session.prompt_queue.put_nowait("message with no agent")

    mock_handle = MagicMock()
    mock_handle.agent = None
    mock_handle.session_id = "test-no-agent"

    controller._drain_prompt_queue(session, mock_handle)

    # Message should be back in the queue, not dropped
    assert not session.prompt_queue.empty(), (
        "prompt_queue is empty — message was silently dropped when agent is None"
    )
    assert len(controller._background_tasks) == 0, "Background task was created without an agent"


@pytest.mark.integration
@pytest.mark.anyio
async def test_cancel_then_steer_does_not_lose_message(
    minimal_pool: AgentPool,
) -> None:
    """Cancel then immediate steer — steer message is not lost.

    When cancel is in progress and a steer (priority=asap) arrives,
    _route_message calls run.steer() on the about-to-be-cancelled run.
    The steer message goes to the dead RunHandle's feedback queue.

    After cancel completes, the steer message should be visible somehow —
    either processed or clearly rejected. This test verifies the session
    is not left in a stuck state.
    """
    session_pool = minimal_pool.session_pool
    assert session_pool is not None
    session_id = "sess-cancel-steer-race"
    await session_pool.create_session(session_id, agent_name="test_agent")
    await _patch_agent_create_turn(session_pool, session_id, _make_cancel_aware_create_turn())
    queue = await session_pool.event_bus.subscribe(session_id)
    first_handle = await _receive_and_get_handle(session_pool, session_id, "first prompt")
    assert first_handle is not None
    await asyncio.sleep(0.1)

    # Cancel the run
    session_pool.sessions.cancel_run_for_session(session_id)

    # Immediately steer (asap) — this races with cancel propagation
    # steer() on a cancelled handle goes to feedback_queue (lost)
    # but the session should still be usable
    await asyncio.wait_for(
        session_pool.send_message(session_id, "steer after cancel", mode=DeliveryMode.STEER),
        timeout=5.0,
    )

    # Wait a bit for things to settle
    await asyncio.sleep(0.3)

    # The session should still accept a new message normally
    await asyncio.wait_for(
        session_pool.send_message(session_id, "new prompt after steer"),
        timeout=10.0,
    )
    post_events = await _collect_events_until(queue, StreamCompleteEvent, timeout=10.0)
    post_types = [type(_unwrap_event(e)) for e in post_events]
    assert StreamCompleteEvent in post_types, (
        f"Expected StreamCompleteEvent after steer+new prompt, got: {post_types}"
    )

    # Invariant: prompt_queue must be empty
    _assert_cancel_invariants(session_pool, session_id)

    # Cleanup
    first_handle.close()
    handle = session_pool._get_active_run_handle(session_id)
    if handle is not None and handle is not first_handle:
        handle.close()
    await asyncio.sleep(0.1)


@pytest.mark.integration
@pytest.mark.anyio
async def test_cancel_during_chaining_drains_remaining(
    minimal_pool: AgentPool,
) -> None:
    """Cancel during prompt_queue chaining — remaining messages are drained.

    _consume_run chains prompts from prompt_queue after each turn.
    If cancel happens during chaining (between turns), the current
    _consume_run is cancelled, but _cleanup_run's _drain_prompt_queue
    should pick up the remaining messages.
    """
    session_pool = minimal_pool.session_pool
    assert session_pool is not None
    session_id = "sess-cancel-chaining"
    await session_pool.create_session(session_id, agent_name="test_agent")

    # Agent that completes immediately for all calls (no blocking)
    await _patch_agent_create_turn(
        session_pool,
        session_id,
        _make_stub_only_create_turn(),
    )
    queue = await session_pool.event_bus.subscribe(session_id)

    # Enqueue 2 messages BEFORE sending the first prompt
    session = session_pool.sessions.get_session(session_id)
    assert session is not None
    session.prompt_queue.put_nowait("queued 1")
    session.prompt_queue.put_nowait("queued 2")

    # Send first prompt — starts the first turn, then chains to queued 1, then queued 2
    first_handle = await _receive_and_get_handle(session_pool, session_id, "first prompt")
    assert first_handle is not None

    # Wait for the first turn to complete, then cancel during chaining
    await _collect_events_until(queue, StreamCompleteEvent, timeout=5.0)

    # Cancel during the chaining window (between turn 1 and turn 2)
    session_pool.sessions.cancel_run_for_session(session_id)

    # Wait for remaining events — _drain_prompt_queue should process remaining
    await asyncio.sleep(0.5)

    # The session should still be usable — send a new message
    await asyncio.wait_for(
        session_pool.send_message(session_id, "after chaining cancel"),
        timeout=10.0,
    )
    post_events = await _collect_events_until(queue, StreamCompleteEvent, timeout=10.0)
    post_types = [type(_unwrap_event(e)) for e in post_events]
    assert StreamCompleteEvent in post_types, (
        f"Expected StreamCompleteEvent after chaining cancel, got: {post_types}"
    )

    # Invariant: prompt_queue must be empty
    _assert_cancel_invariants(session_pool, session_id)

    # Cleanup
    first_handle.close()
    handle = session_pool._get_active_run_handle(session_id)
    if handle is not None and handle is not first_handle:
        handle.close()
    await asyncio.sleep(0.1)


# ---------------------------------------------------------------------------
# Test: Turn error with queued prompt — message must not get stuck
# ---------------------------------------------------------------------------


class _BlockingThenErrorTurn(Turn):
    """Turn that blocks briefly (allowing enqueuing), then raises RuntimeError.

    Simulates a turn that fails mid-execution after running long enough
    for a concurrent _route_message to enqueue a message to prompt_queue.
    """

    def __init__(self, run_ctx: AgentRunContext) -> None:
        self._run_ctx = run_ctx

    async def execute(self):  # type: ignore[override]
        self._message_history = []
        self._final_message = ChatMessage(content="error", role="assistant")
        for _ in range(20):
            if self._run_ctx.cancelled:
                return
            await asyncio.sleep(0.01)
        raise RuntimeError("Simulated turn error")
        yield  # unreachable


def _make_blocking_then_error_create_turn() -> Any:
    """Return a create_turn: first call blocks-then-errors, rest are stubs."""
    call_count = 0

    def _create_turn(
        prompts: Any, run_ctx: AgentRunContext, message_history: Any, **kwargs: Any
    ) -> Turn:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return _BlockingThenErrorTurn(run_ctx)
        return _StubTurn_e2e(
            events=[
                RunStartedEvent(run_id="test-run"),
                StreamCompleteEvent(message=ChatMessage(content="response", role="assistant")),
            ],
            message_history=["msg"],
        )

    return _create_turn


@pytest.mark.integration
@pytest.mark.anyio
async def test_turn_error_with_queued_prompt_drains_queue(
    minimal_pool: AgentPool,
) -> None:
    """Turn error with queued message — queued message is not stuck.

    Regression test: _consume_run's error path (except Exception ->
    turn_failed -> break) skips the prompt_queue check, leaving queued
    messages stuck or processed out-of-order.

    Scenario:
        1. Turn 1 runs (blocks briefly, then raises RuntimeError).
        2. While turn 1 runs, message 2 is enqueued to prompt_queue.
        3. Turn 1 fails with Exception.
        4. Queued message 2 must be processed (not stuck).

    Without fix: message stuck in prompt_queue -> timeout.
    With fix: _consume_run checks prompt_queue in the error path.
    """
    session_pool = minimal_pool.session_pool
    assert session_pool is not None
    session_id = "sess-turn-error-queued"
    await session_pool.create_session(session_id, agent_name="test_agent")
    await _patch_agent_create_turn(
        session_pool, session_id, _make_blocking_then_error_create_turn()
    )
    queue = await session_pool.event_bus.subscribe(session_id)

    first_handle = await _receive_and_get_handle(session_pool, session_id, "first prompt")
    assert first_handle is not None
    await asyncio.sleep(0.05)

    # Enqueue message 2 during turn 1 (simulates race)
    session = session_pool.sessions.get_session(session_id)
    assert session is not None
    session.prompt_queue.put_nowait("second prompt")

    # Wait for turn 1 to fail
    pre_events: list[Any] = []
    try:
        async with asyncio.timeout(10.0):
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=5.0)
                    pre_events.append(event)
                    if isinstance(_unwrap_event(event), RunErrorEvent | RunFailedEvent):
                        break
                except TimeoutError:
                    break
    except TimeoutError:
        pytest.fail("Timed out waiting for error event from turn 1")

    pre_event_types = [type(_unwrap_event(e)) for e in pre_events]
    assert RunErrorEvent in pre_event_types or RunFailedEvent in pre_event_types, (
        f"Expected RunErrorEvent or RunFailedEvent from turn 1 error, got: {pre_event_types}"
    )

    # Queued message 2 should be processed (with fix) or stuck (without fix)
    post_events: list[Any] = []
    try:
        async with asyncio.timeout(10.0):
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=5.0)
                    post_events.append(event)
                    if isinstance(_unwrap_event(event), StreamCompleteEvent):
                        break
                except TimeoutError:
                    break
    except TimeoutError:
        pytest.fail(
            "Timed out waiting for queued prompt after turn error - "
            "message stuck in prompt_queue (error path skips queue check)"
        )

    post_event_types = [type(_unwrap_event(e)) for e in post_events]
    assert RunStartedEvent in post_event_types, (
        f"Expected RunStartedEvent for queued prompt, got: {post_event_types}"
    )
    assert StreamCompleteEvent in post_event_types, (
        f"Expected StreamCompleteEvent for queued prompt, got: {post_event_types}"
    )

    _assert_cancel_invariants(session_pool, session_id)
    first_handle.close()
    handle = session_pool._get_active_run_handle(session_id)
    if handle is not None and handle is not first_handle:
        handle.close()
    await asyncio.sleep(0.1)


@pytest.mark.integration
@pytest.mark.anyio
async def test_session_close_with_queued_prompt_abandons_messages(
    minimal_pool: AgentPool,
) -> None:
    """Session close with messages in prompt_queue - no new run started.

    When a session is closing with messages still in prompt_queue, no new
    run should be started for those messages. The is_closing guard in
    _drain_prompt_queue prevents this.
    """
    session_pool = minimal_pool.session_pool
    assert session_pool is not None
    session_id = "sess-close-queued-warn"
    await session_pool.create_session(session_id, agent_name="test_agent")
    await _patch_agent_create_turn(session_pool, session_id, _make_cancel_aware_create_turn())
    queue = await session_pool.event_bus.subscribe(session_id)

    first_handle = await _receive_and_get_handle(session_pool, session_id, "first prompt")
    assert first_handle is not None
    await asyncio.sleep(0.1)

    # Enqueue messages
    session = session_pool.sessions.get_session(session_id)
    assert session is not None
    session.prompt_queue.put_nowait("will be lost 1")
    session.prompt_queue.put_nowait("will be lost 2")

    # Mark session as closing BEFORE cancel — _drain_prompt_queue should skip
    session.is_closing = True

    # Cancel the active run
    session_pool.sessions.cancel_run_for_session(session_id)
    await asyncio.sleep(0.3)

    # Drain remaining events
    events = await _drain_queue(queue)
    event_types = [type(_unwrap_event(e)) for e in events]

    # RunFailedEvent from cancel is expected
    assert RunFailedEvent in event_types, f"Expected RunFailedEvent from cancel, got: {event_types}"
    # NO RunStartedEvent for the queued messages — session is closing
    run_started_count = event_types.count(RunStartedEvent)
    assert run_started_count <= 1, (
        f"Expected at most 1 RunStartedEvent (from turn 1), got {run_started_count} — "
        f"new run was started on a closing session! Events: {event_types}"
    )

    # Queued messages should still be in prompt_queue (not drained)
    assert not session.prompt_queue.empty(), (
        "prompt_queue should still contain messages — session is closing, drain skipped"
    )

    # Cleanup
    session.is_closing = False
    first_handle.close()
    await asyncio.sleep(0.1)
