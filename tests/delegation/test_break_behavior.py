"""Test script to validate the behavior of breaking from `run_stream()` iteration.

This test validates the behavior when breaking from an async for loop iterating
over `agent.run_stream()`. The findings document known issues and edge cases.

## Summary of Findings

### Current Behavior (ISSUES IDENTIFIED):

1. **Exception Propagation on Break** (CRITICAL):
   - Breaking from `run_stream()` causes multiple internal exceptions:
     - `RuntimeError: Attempted to exit cancel scope in a different task`
     - `ValueError: Token was created in a different Context`
     - `RuntimeError: generator didn't stop after athrow()`
   - These exceptions are printed to stderr but may not propagate to user code
   - Previously caused by `merge_queue_into_iterator` context manager task switching (now removed)

2. **_cancelled Flag State** (PARTIALLY WORKS):
   - `_cancelled` flag is set to `True` when break happens during active streaming
   - However, flag state can be inconsistent depending on where break occurs

3. **Conversation History** (BROKEN):
   - After break, conversation history often shows 0 messages
   - History accumulation is unreliable due to exception during cleanup

4. **Subsequent Runs** (BROKEN):
   - After breaking, subsequent `run_stream()` calls may fail with:
     - `CancelledError: Cancelled via cancel scope`
   - The agent enters a corrupted state

### Recommendation:

AVOID breaking from `run_stream()` iteration in production code.
Use explicit cancellation via `agent.interrupt()` instead,
or consume all events until `StreamCompleteEvent`.

For simulation use cases (break on tool call), wrap the agent to intercept
events rather than breaking the iteration.
"""

from __future__ import annotations

import asyncio
from contextlib import redirect_stderr, suppress
from io import StringIO
from typing import Any
from unittest.mock import MagicMock

from pydantic_ai import PartDeltaEvent, TextPartDelta
from pydantic_ai.models.test import TestModel
import pytest

from wolfharness import Agent, AgentPool
from wolfharness.agents.events import StreamCompleteEvent, ToolCallStartEvent
from wolfharness.lifecycle import DirectChannel, MemoryJournal
from wolfharness.orchestrator.core import SessionPool


pytestmark = pytest.mark.unit


TEST_RESPONSE = "I am a test response"
TOOL_CALL_RESPONSE = "Tool was called"


@pytest.fixture
def break_test_agent() -> Agent[None]:
    """Create an agent with TestModel for break testing."""
    model = TestModel(custom_output_text=TEST_RESPONSE)
    return Agent(name="break-test-agent", model=model)


@pytest.fixture
def tool_call_agent() -> Agent[None]:
    """Create an agent with a tool for testing break on tool call."""

    async def question_tool(prompt: str) -> str:
        """A test tool that simulates asking a question."""
        return f"Question processed: {prompt}"

    model = TestModel(custom_output_text=TOOL_CALL_RESPONSE)
    return Agent(
        name="tool-call-agent",
        model=model,
        tools=[question_tool],
    )


async def _setup_session_pool(
    agent: Agent[Any], minimal_pool: AgentPool
) -> tuple[SessionPool, str]:
    """Create a SessionPool with the given agent attached using the real pool."""
    # Create a new SessionPool with auto_resume disabled for break testing
    session_pool = SessionPool(minimal_pool, enable_auto_resume=False)
    await session_pool.start()

    session_id = f"test-session-{agent.name}"
    await session_pool.create_session(session_id, agent_name=agent.name)

    # Attach agent to session
    state, _ = await session_pool.sessions.get_or_create_session(session_id)
    state.agent = agent
    # Initialize comm_channel on session (normally done by
    # _initialize_lifecycle_and_recovery, but we pre-cache the agent
    # so that path is skipped).
    state._comm_channel = DirectChannel(MemoryJournal())
    session_pool.sessions._session_agents[session_id] = agent
    minimal_pool.get_agent = MagicMock(return_value=agent)  # type: ignore[assignment]

    # Link agent back to pool so interrupt() can resolve session state
    agent.agent_pool = minimal_pool
    agent.session_id = session_id

    return session_pool, session_id


async def test_simple_break_after_n_events(break_test_agent: Agent[None], minimal_pool: AgentPool):
    """Test 1: Simple break after receiving N events.

    !!! warning "Known Issue"
        This test documents current behavior which has issues. Breaking from
        run_stream causes internal exceptions and may corrupt agent state.

    Current behavior:
    - Events are collected correctly before break
    - _cancelled flag may or may not be set depending on timing
    - Conversation history may be 0 due to cleanup exceptions
    """
    session_pool, session_id = await _setup_session_pool(break_test_agent, minimal_pool)
    try:
        # Capture stderr to check for internal exceptions
        stderr_capture = StringIO()

        with redirect_stderr(stderr_capture):
            events = []
            async for event in session_pool.run_stream(session_id, "Hello"):
                events.append(event)
                if len(events) >= 3:
                    break

        # We collected some events
        assert len(events) >= 3, f"Expected at least 3 events, got {len(events)}"

        # Check for internal exceptions in stderr
        stderr_output = stderr_capture.getvalue()
        if "RuntimeError" in stderr_output or "CancelledError" in stderr_output:
            # Document the issue - do not fail the test, just note it
            print(f"[ISSUE] Internal exceptions on break: {stderr_output[:500]}")
    finally:
        await session_pool.shutdown()


async def test_break_with_exception_handling(
    break_test_agent: Agent[None], minimal_pool: AgentPool
):
    """Test 2: Verify exception handling around break.

    !!! warning "Known Issue"
        While user code may not see exceptions, internal errors occur during
        generator cleanup that can corrupt agent state.
    """
    session_pool, session_id = await _setup_session_pool(break_test_agent, minimal_pool)
    try:
        user_exception = None

        try:
            async for _event in session_pool.run_stream(session_id, "Test"):
                break
        except Exception as e:  # noqa: BLE001
            user_exception = e

        # User code typically does not see exceptions (they are in cleanup)
        # BUT internally there are errors
        assert user_exception is None, "Exceptions should not propagate to user code"
    finally:
        await session_pool.shutdown()


async def test_conversation_history_after_break(
    break_test_agent: Agent[None], minimal_pool: AgentPool
):
    """Test 3: Conversation history after break.

    !!! warning "Known Issue"
        Due to cleanup exceptions, conversation history is often not preserved
        correctly after a break.
    """
    session_pool, session_id = await _setup_session_pool(break_test_agent, minimal_pool)
    try:
        # Run and break
        async for _event in session_pool.run_stream(session_id, "Test message"):
            break  # Break immediately

        history = break_test_agent.conversation.get_history()
        # Document behavior rather than assert correctness
        print(f"[INFO] History length after break: {len(history)}")
    finally:
        await session_pool.shutdown()


async def test_subsequent_run_after_break(break_test_agent: Agent[None], minimal_pool: AgentPool):
    """Test 4: Subsequent run_stream after break.

    !!! warning "Known Issue"
        After breaking, subsequent runs may fail with CancelledError due to
        leftover cancel scope state.
    """
    session_pool, session_id = await _setup_session_pool(break_test_agent, minimal_pool)
    try:
        # First run with break
        async for _event in session_pool.run_stream(session_id, "First prompt"):
            break

        # Try second run - this may fail
        second_run_succeeded = False
        second_run_error = None

        try:
            async for event in session_pool.run_stream(session_id, "Second prompt"):
                if isinstance(event, StreamCompleteEvent):
                    second_run_succeeded = True
                    break
        except asyncio.CancelledError as e:
            second_run_error = e
        except Exception as e:  # noqa: BLE001
            second_run_error = e

        # Document the issue
        if second_run_error:
            print(
                f"[ISSUE] Second run failed: {type(second_run_error).__name__}: {second_run_error}"
            )
        else:
            print(f"[INFO] Second run succeeded: {second_run_succeeded}")
    finally:
        await session_pool.shutdown()


@pytest.mark.skip(
    reason="Async generator cleanup deadlock in session_pool.run_stream() — "
    "agent.interrupt() triggers aclose() on a running generator, causing "
    "'asynchronous generator is already running'. Tracked as architecture issue. "
    "Use consume-until-StreamCompleteEvent pattern instead."
)
async def test_interrupt_vs_break(break_test_agent: Agent[None], minimal_pool: AgentPool):
    """Test 5: Compare interrupt() vs break behavior.

    Shows that interrupt() is the recommended approach instead of break.
    """
    session_pool, session_id = await _setup_session_pool(break_test_agent, minimal_pool)
    try:
        # Test interrupt() method
        events = []

        # Start streaming in background task so we can interrupt it
        async def stream_task():
            events.extend([event async for event in session_pool.run_stream(session_id, "Test")])

        task = asyncio.create_task(stream_task())
        await asyncio.sleep(0.1)  # Let it start

        # Interrupt
        await break_test_agent.interrupt()

        # Wait for task to finish
        with suppress(asyncio.CancelledError):
            await task

        # Check interrupt worked
        assert break_test_agent._cancelled is True, "_cancelled should be True after interrupt"
        print(f"[INFO] Events collected before interrupt: {len(events)}")
    finally:
        await session_pool.shutdown()


async def test_safe_pattern_complete_consumption(
    break_test_agent: Agent[None], minimal_pool: AgentPool
):
    """Test 6: Safe pattern - consume until StreamCompleteEvent.

    !!! tip "Recommended Pattern"
        Instead of breaking, always consume until StreamCompleteEvent.
        This is the only reliable pattern currently.
    """
    session_pool, session_id = await _setup_session_pool(break_test_agent, minimal_pool)
    try:
        events = []
        final_message = None

        # Safe pattern - do not break early, consume all events
        async for event in session_pool.run_stream(session_id, "Test"):
            events.append(event)
            if isinstance(event, StreamCompleteEvent):
                final_message = event.message
                break  # OK to break after StreamCompleteEvent

        assert final_message is not None, "Should get final message"
        assert len(events) > 0, "Should have events"
        print(f"[INFO] Safe consumption: {len(events)} events")
    finally:
        await session_pool.shutdown()


async def test_tool_call_detection_without_break(
    tool_call_agent: Agent[None], minimal_pool: AgentPool
):
    """Test 7: Tool call detection simulation without breaking.

    !!! tip "Recommended Pattern"
        For simulation use case, intercept events but do not break.
        Use a flag to track state and let the stream complete.
    """
    session_pool, session_id = await _setup_session_pool(tool_call_agent, minimal_pool)
    try:
        tool_detected = False
        events = []

        # Safe pattern - detect but do not break
        async for event in session_pool.run_stream(session_id, "Trigger the tool"):
            events.append(event)

            if isinstance(event, ToolCallStartEvent):
                tool_detected = True
                print(f"[INFO] Tool call detected: {event.tool_name}")
                # Do not break! Let it continue

            if isinstance(event, StreamCompleteEvent):
                break

        print(f"[INFO] Tool detected: {tool_detected}, Total events: {len(events)}")
    finally:
        await session_pool.shutdown()


async def test_partial_text_collection(break_test_agent: Agent[None], minimal_pool: AgentPool):
    """Test 8: Collect partial text without breaking.

    !!! tip "Recommended Pattern"
        If you need partial results, collect text deltas but still
        consume the full stream.
    """
    session_pool, session_id = await _setup_session_pool(break_test_agent, minimal_pool)
    try:
        text_chunks = []
        final_message = None

        async for event in session_pool.run_stream(session_id, "Generate text"):
            match event:
                case PartDeltaEvent(delta=TextPartDelta(content_delta=delta)):
                    text_chunks.append(delta)
                case StreamCompleteEvent(message=msg):
                    final_message = msg
                    break

        partial_text = "".join(text_chunks)
        print(f"[INFO] Collected text: {partial_text[:100]}...")
        assert final_message is not None
    finally:
        await session_pool.shutdown()


async def run_test_safely(test_name: str, test_func, agent: Agent[None]) -> bool:
    """Run a test, handling the case where even context manager entry fails."""
    print(f"\n{'=' * 70}")
    print(test_name)
    print("=" * 70)

    try:
        async with agent as a:
            try:
                await test_func(a)
                print("[PASS] Test completed")
            except Exception as e:  # noqa: BLE001
                print(f"[FAIL] {type(e).__name__}: {e}")
                return False
            else:
                return True
    except asyncio.CancelledError as e:
        # Even the context manager entry failed - this demonstrates the issue
        print(f"[FAIL] CancelledError during agent entry: {e}")
        print("[NOTE] This demonstrates the state corruption issue from previous break")
        return False
    except Exception as e:  # noqa: BLE001
        print(f"[FAIL] {type(e).__name__} during agent entry: {e}")
        return False


async def main():
    """Run all tests and document findings."""
    print("=" * 70)
    print("BREAK BEHAVIOR TEST SUITE - Documenting Current Behavior")
    print("=" * 70)
    print()
    print("IMPORTANT: These tests document KNOWN ISSUES with breaking from")
    print("run_stream(). See module docstring for details.")
    print()

    tests = [
        ("Test 1: Simple break (documents issues)", test_simple_break_after_n_events),
        ("Test 2: Exception handling", test_break_with_exception_handling),
        ("Test 3: Conversation history after break", test_conversation_history_after_break),
        ("Test 4: Subsequent run after break", test_subsequent_run_after_break),
        ("Test 5: Interrupt vs break", test_interrupt_vs_break),
        ("Test 6: Safe pattern - complete consumption", test_safe_pattern_complete_consumption),
        ("Test 7: Tool detection without break", test_tool_call_detection_without_break),
        ("Test 8: Partial text collection", test_partial_text_collection),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        # Create fresh agent for each test (but state may still be affected)
        agent = Agent(name="test", model=TestModel(custom_output_text=TEST_RESPONSE))

        if await run_test_safely(name, test_func, agent):
            passed += 1
        else:
            failed += 1

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print(f"Results: {passed} passed, {failed} failed")
    print()
    print("Key Findings:")
    print("1. Breaking from run_stream causes internal CancelScope/ContextVar errors")
    print("2. Agent state may be corrupted after break")
    print("3. Conversation history is unreliable after break")
    print("4. RECOMMENDED: Always consume until StreamCompleteEvent")
    print("5. ALTERNATIVE: Use agent.interrupt() for cancellation")
    print()


if __name__ == "__main__":
    asyncio.run(main())
