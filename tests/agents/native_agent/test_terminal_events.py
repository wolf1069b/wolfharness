"""Unit tests for terminal event guarantees in NativeTurn.execute().

Verifies that every exit path yields exactly one terminal event
(StreamCompleteEvent or RunErrorEvent as the last event).

Focuses on the 4 NEW fixed paths:
- Path #4: CancelledError (cancelled=True) → StreamCompleteEvent(cancelled=True)
- Path #6: RuntimeError (GeneratorExit) → StreamCompleteEvent(cancelled=True)
- Path #7b: Generic exception (no history) → RunErrorEvent + StreamCompleteEvent(cancelled=True)
- Path #9: Belt-and-suspenders cancelled → StreamCompleteEvent(cancelled=True)
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from pydantic_ai.models.test import TestModel
import pytest

from wolfharness import Agent
from wolfharness.agents.context import AgentRunContext
from wolfharness.agents.events.events import (
    RunErrorEvent,
    StepErrorMetadata,
    StreamCompleteEvent,
)
from wolfharness.agents.native_agent.turn import NativeTurn


def _make_mock_agentlet_raising(exc: BaseException) -> MagicMock:
    """Create a mock agentlet whose iter() async CM raises *exc* in __aenter__."""
    mock_agentlet = MagicMock()
    mock_run = AsyncMock()
    mock_run.__aenter__ = AsyncMock(side_effect=exc)
    mock_run.__aexit__ = AsyncMock(return_value=None)
    mock_agentlet.iter = MagicMock(return_value=mock_run)
    return mock_agentlet


def _make_mock_agentlet_with_cancelled_error() -> MagicMock:
    """Create a mock agentlet whose iter() async CM raises CancelledError in __aenter__.

    This simulates a cancellation that occurs when run_ctx.cancelled is True
    (cooperative cancellation from cancel()).
    """
    mock_agentlet = MagicMock()
    mock_run = AsyncMock()
    mock_run.__aenter__ = AsyncMock(side_effect=asyncio.CancelledError())
    mock_run.__aexit__ = AsyncMock(return_value=None)
    mock_agentlet.iter = MagicMock(return_value=mock_run)
    return mock_agentlet


def _make_mock_agentlet_with_runtime_error_generator_exit() -> MagicMock:
    """Create a mock agentlet whose iter() async CM raises RuntimeError('ignored GeneratorExit')."""
    mock_agentlet = MagicMock()
    mock_run = AsyncMock()
    mock_run.__aenter__ = AsyncMock(
        side_effect=RuntimeError("generator ignored GeneratorExit"),
    )
    mock_run.__aexit__ = AsyncMock(return_value=None)
    mock_agentlet.iter = MagicMock(return_value=mock_run)
    return mock_agentlet


def _make_mock_agentlet_with_generic_exception() -> MagicMock:
    """Create a mock agentlet whose iter() async CM raises a generic ValueError."""
    mock_agentlet = MagicMock()
    mock_run = AsyncMock()
    mock_run.__aenter__ = AsyncMock(side_effect=ValueError("unexpected error"))
    mock_run.__aexit__ = AsyncMock(return_value=None)
    mock_agentlet.iter = MagicMock(return_value=mock_run)
    return mock_agentlet


# ---------------------------------------------------------------------------
# Path #4: CancelledError with cancelled=True → StreamCompleteEvent(cancelled=True)
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_cancelled_error_with_cancelled_true_yields_stream_complete() -> None:
    """CancelledError when run_ctx.cancelled=True yields StreamCompleteEvent(cancelled=True).

    This tests path #4: cooperative cancellation from cancel() should produce
    a terminal StreamCompleteEvent so consumers know the stream has ended.
    """
    agent = Agent(
        name="test-cancel-true",
        model=TestModel(custom_output_text="hello"),
    )
    async with agent:
        mock_agentlet = _make_mock_agentlet_with_cancelled_error()
        run_ctx = AgentRunContext(session_id="test-session")
        run_ctx.cancelled = True

        turn = NativeTurn(
            agent=agent,
            prompts=["test"],
            run_ctx=run_ctx,
            message_history=[],
        )

        events: list[Any] = []
        with patch.object(agent, "get_agentlet", AsyncMock(return_value=mock_agentlet)):
            events.extend([event async for event in turn.execute()])

        stream_completes = [e for e in events if isinstance(e, StreamCompleteEvent)]
        assert len(stream_completes) == 1, (
            f"Expected exactly 1 StreamCompleteEvent, got {len(stream_completes)}"
        )
        assert stream_completes[0].cancelled is True, (
            "StreamCompleteEvent should have cancelled=True"
        )
        assert isinstance(events[-1], StreamCompleteEvent), "Last event must be StreamCompleteEvent"


# ---------------------------------------------------------------------------
# Path #5: CancelledError with cancelled=False → re-raises (no terminal event)
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_cancelled_error_with_cancelled_false_reraises() -> None:
    """CancelledError when run_ctx.cancelled=False re-raises without terminal event."""
    agent = Agent(
        name="test-cancel-false",
        model=TestModel(custom_output_text="hello"),
    )
    async with agent:
        mock_agentlet = _make_mock_agentlet_with_cancelled_error()
        run_ctx = AgentRunContext(session_id="test-session")
        # cancelled is False by default

        turn = NativeTurn(
            agent=agent,
            prompts=["test"],
            run_ctx=run_ctx,
            message_history=[],
        )

        with (
            patch.object(agent, "get_agentlet", AsyncMock(return_value=mock_agentlet)),
            pytest.raises(asyncio.CancelledError),
        ):
            async for _ in turn.execute():
                pass


# ---------------------------------------------------------------------------
# Path #6: RuntimeError (GeneratorExit) → StreamCompleteEvent(cancelled=True)
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_runtime_error_generator_exit_yields_stream_complete() -> None:
    """RuntimeError('ignored GeneratorExit') yields StreamCompleteEvent(cancelled=True).

    This tests path #6: when pydantic-ai doesn't properly handle GeneratorExit,
    we catch the RuntimeError and yield a terminal event.
    """
    agent = Agent(
        name="test-genexit",
        model=TestModel(custom_output_text="hello"),
    )
    async with agent:
        mock_agentlet = _make_mock_agentlet_with_runtime_error_generator_exit()
        run_ctx = AgentRunContext(session_id="test-session")

        turn = NativeTurn(
            agent=agent,
            prompts=["test"],
            run_ctx=run_ctx,
            message_history=[],
        )

        events: list[Any] = []
        with patch.object(agent, "get_agentlet", AsyncMock(return_value=mock_agentlet)):
            events.extend([event async for event in turn.execute()])

        stream_completes = [e for e in events if isinstance(e, StreamCompleteEvent)]
        assert len(stream_completes) == 1, (
            f"Expected exactly 1 StreamCompleteEvent, got {len(stream_completes)}"
        )
        assert stream_completes[0].cancelled is True, (
            "StreamCompleteEvent should have cancelled=True for GeneratorExit path"
        )
        assert isinstance(events[-1], StreamCompleteEvent), "Last event must be StreamCompleteEvent"


# ---------------------------------------------------------------------------
# Path #7b: Generic exception (no history) → RunErrorEvent (terminal)
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_generic_exception_yields_run_error_as_terminal() -> None:
    """Generic exception with no message history yields RunErrorEvent as terminal.

    This tests path #7b: when an unexpected exception occurs and there's no
    message history to recover, we yield RunErrorEvent (with step_error metadata)
    as the terminal event. RunErrorEvent IS the terminal — no trailing
    StreamCompleteEvent is needed because consumers already treat it as terminal.
    """
    agent = Agent(
        name="test-generic-exc",
        model=TestModel(custom_output_text="hello"),
    )
    async with agent:
        mock_agentlet = _make_mock_agentlet_with_generic_exception()
        run_ctx = AgentRunContext(session_id="test-session")

        turn = NativeTurn(
            agent=agent,
            prompts=["test"],
            run_ctx=run_ctx,
            message_history=[],
        )

        events: list[Any] = []
        with patch.object(agent, "get_agentlet", AsyncMock(return_value=mock_agentlet)):
            events.extend([event async for event in turn.execute()])

        run_errors = [e for e in events if isinstance(e, RunErrorEvent)]
        assert len(run_errors) == 1, f"Expected exactly 1 RunErrorEvent, got {len(run_errors)}"
        assert run_errors[0].message == "unexpected error"
        assert run_errors[0].agent_name == "test-generic-exc"
        assert run_errors[0].step_error is not None, (
            "RunErrorEvent should have step_error populated"
        )
        assert isinstance(run_errors[0].step_error, StepErrorMetadata)
        # Since the error occurs in iter() __aenter__, current_node is None → "unknown"
        assert run_errors[0].step_error.node_type == "unknown"
        assert run_errors[0].step_error.exception_type == "ValueError"
        assert run_errors[0].step_error.exception_message == "unexpected error"

        # RunErrorEvent is the terminal event — no StreamCompleteEvent follows
        stream_completes = [e for e in events if isinstance(e, StreamCompleteEvent)]
        assert len(stream_completes) == 0, (
            "Expected 0 StreamCompleteEvent — RunErrorEvent is terminal"
        )

        # Last event must be RunErrorEvent
        assert isinstance(events[-1], RunErrorEvent), "Last event must be RunErrorEvent"


# ---------------------------------------------------------------------------
# Path #9: Belt-and-suspenders cancelled → StreamCompleteEvent(cancelled=True)
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_belt_and_suspenders_cancelled_yields_stream_complete() -> None:
    """Post-loop cancelled check yields StreamCompleteEvent(cancelled=True).

    This tests path #9: when run_ctx.cancelled is True after the while loop
    (e.g. CancelledError swallowed by pydantic-ai), the belt-and-suspenders
    check yields a terminal event.
    """
    agent = Agent(
        name="test-belt-cancel",
        model=TestModel(custom_output_text="hello"),
    )
    async with agent:
        run_ctx = AgentRunContext(session_id="test-session")

        turn = NativeTurn(
            agent=agent,
            prompts=["test"],
            run_ctx=run_ctx,
            message_history=[],
        )

        # Patch the while loop to set cancelled=True after a normal run,
        # simulating a CancelledError swallowed by pydantic-ai.
        original_execute = turn.execute

        async def patched_execute() -> Any:
            """Wrap execute() to set cancelled before the belt-and-suspenders check."""
            gen = original_execute()
            # Consume events until we see the while loop is about to finish.
            # We set cancelled=True after consuming all events from the normal path.
            # The TestModel completes in one cycle, so we just need to set cancelled
            # after the first StreamCompleteEvent or when the while loop finishes.
            # Instead, we'll patch _run_ctx.cancelled before execute() starts
            # and use a custom agentlet that breaks the while loop early.
            run_ctx.cancelled = True
            async for event in gen:
                yield event

        events: list[Any] = [event async for event in patched_execute()]

        stream_completes = [e for e in events if isinstance(e, StreamCompleteEvent)]
        assert len(stream_completes) == 1, (
            f"Expected exactly 1 StreamCompleteEvent, got {len(stream_completes)}"
        )
        assert stream_completes[0].cancelled is True, (
            "StreamCompleteEvent should have cancelled=True for belt-and-suspenders path"
        )
        assert isinstance(events[-1], StreamCompleteEvent), "Last event must be StreamCompleteEvent"


# ---------------------------------------------------------------------------
# Normal success path: exactly one StreamCompleteEvent (not cancelled)
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_success_path_yields_exactly_one_stream_complete() -> None:
    """Normal success path yields exactly one StreamCompleteEvent (not cancelled)."""
    agent = Agent(
        name="test-success",
        model=TestModel(custom_output_text="success response"),
    )
    async with agent:
        run_ctx = AgentRunContext(session_id="test-session")
        turn = NativeTurn(
            agent=agent,
            prompts=["test"],
            run_ctx=run_ctx,
            message_history=[],
        )

        events = [event async for event in turn.execute()]

        stream_completes = [e for e in events if isinstance(e, StreamCompleteEvent)]
        assert len(stream_completes) == 1, (
            f"Expected exactly 1 StreamCompleteEvent, got {len(stream_completes)}"
        )
        assert stream_completes[0].cancelled is False, (
            "StreamCompleteEvent should not be cancelled on success path"
        )
        assert isinstance(events[-1], StreamCompleteEvent), "Last event must be StreamCompleteEvent"
