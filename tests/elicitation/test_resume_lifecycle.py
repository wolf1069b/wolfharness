"""Red-flag test: StreamCompleteEvent after in-process elicitation resume.

Reproduces the bug where after future resolution (in-process elicitation
resume), the agent run reaches ``End`` node but never yields
``StreamCompleteEvent``. The stream just dies.

Lifecycle under test:

1. Agent with TestModel calls a local tool on first LLM response.
2. Local tool calls ``ctx.handle_elicitation()`` which (with
   ``supports_durable_elicitation=True`` and ``in_mcp_callback=False``)
   checkpoints, emits event, registers a future, and awaits it.
3. Test resolves the future with an ``ElicitationResumePayload``.
4. ``handle_elicitation()`` returns the ``ElicitResult``.
5. Tool returns a result to the agent.
6. Second LLM response: final text (no more tool calls).
7. Agent run reaches ``End`` node.
8. ``NativeTurn.execute()`` should yield ``StreamCompleteEvent``.

Bug: Step 8 does not happen — ``StreamCompleteEvent`` is never yielded
after future resolution.

Two test levels:

- **Level 1** (``test_stream_complete_via_native_turn``): Directly drives
  ``NativeTurn.execute()``. If this passes, the core turn generator is
  healthy and the bug is in the integration layer.

- **Level 2** (``test_stream_complete_via_run_handle_event_bus``): Drives
  ``RunHandle.start()`` → ``EventBus`` → consumer. If this fails while
  Level 1 passes, the bug is in ``RunHandle.start()`` event forwarding
  or ``EventBus`` delivery.

Refs: https://github.com/Leoyzen/wolfharness/issues/107
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import TYPE_CHECKING, Any

import anyio
from mcp.types import ElicitRequestFormParams, ElicitResult
from pydantic_ai.models.test import TestModel
import pytest

from wolfharness import Agent
from wolfharness.agents.context import AgentContext, AgentRunContext, ConfirmationResult
from wolfharness.agents.events.events import (
    RichAgentStreamEvent,
    StreamCompleteEvent,
)
from wolfharness.agents.native_agent.turn import NativeTurn
from wolfharness.lifecycle import MemoryJournal, ProtocolChannel
from wolfharness.orchestrator.core import EventBus, EventEnvelope, SessionState
from wolfharness.orchestrator.run import RunHandle
from wolfharness.sessions.models import ElicitationResumePayload
from wolfharness.ui.base import InputProvider


if TYPE_CHECKING:
    from wolfharness.agents.native_agent.elicitation_bridge import (
        ElicitationFutureRegistry,
    )


class DurableElicitationProvider(InputProvider):
    """InputProvider that advertises durable elicitation support.

    ``supports_durable_elicitation = True`` causes ``handle_elicitation()``
    to take the local-tool path (Path 3): checkpoint + emit event +
    register future + await future, instead of raising ``CallDeferred``
    or calling ``get_elicitation()`` synchronously.
    """

    @property
    def supports_durable_elicitation(self) -> bool:
        return True

    async def get_text_input(self, context: Any, prompt: str) -> str:
        raise NotImplementedError

    async def get_structured_input(
        self,
        context: Any,
        prompt: str,
        output_type: type[Any],
    ) -> Any:
        raise NotImplementedError

    async def get_tool_confirmation(
        self,
        context: AgentContext[Any],
        tool_description: str = "",
    ) -> ConfirmationResult:
        return "allow"

    async def get_elicitation(
        self,
        params: Any,
    ) -> ElicitResult:
        # Should not be called when supports_durable_elicitation is True.
        return ElicitResult(action="accept", content={"q0": "yes"})


def _make_elicit_tool() -> Any:
    """Create a local tool function that calls handle_elicitation().

    The tool suspends at ``await future`` inside ``handle_elicitation()``
    when ``supports_durable_elicitation`` is True and the tool is a local
    (non-MCP) tool.
    """

    async def elicit_tool(ctx: AgentContext[None]) -> str:
        params = ElicitRequestFormParams(
            message="Do you agree?",
            requestedSchema={
                "type": "object",
                "properties": {
                    "q0": {"type": "string", "title": "Answer"},
                },
                "required": ["q0"],
            },
        )
        result = await ctx.handle_elicitation(params)
        # If we get here, the future was resolved.
        # handle_elicitation() returns ElicitResult | ErrorData.
        # With action="accept", it returns ElicitResult.
        match result:
            case ElicitResult(action=action):
                return f"Elicitation action: {action}"
            case _:
                return f"Elicitation result: {result}"

    return elicit_tool


def _make_agent() -> Agent[None, str]:
    """Create an Agent with TestModel + elicitation tool + durable provider."""
    provider = DurableElicitationProvider()
    model = TestModel(
        call_tools=["elicit_tool"],
        custom_output_text="All done!",
    )
    return Agent(
        name="test-elicit-agent",
        model=model,
        tools=[_make_elicit_tool()],
        input_provider=provider,
    )


async def _wait_for_future_and_resolve(
    run_ctx: AgentRunContext,
    collector: asyncio.Task[Any],
) -> None:
    """Poll elicitation_registry until a future is registered, then resolve.

    If the collector task finishes before a future appears, this function
    returns without resolving (the test assertions will catch the issue).
    """
    registry: ElicitationFutureRegistry | None = None
    while True:
        if collector.done():
            return
        registry = run_ctx.elicitation_registry
        if registry is not None and len(registry) > 0:
            break
        await asyncio.sleep(0.01)

    assert registry is not None, (
        "ElicitationFutureRegistry was never created — get_agentlet() may not have been called"
    )
    assert len(registry) > 0, (
        "No elicitation future was registered — "
        "handle_elicitation() may not have been called or "
        "took a different path (e.g. synchronous get_elicitation)"
    )

    # Resolve the future — simulates user responding to elicitation.
    handles = list(registry._futures.keys())
    assert len(handles) == 1, f"Expected 1 handle, got {len(handles)}"
    handle = handles[0]
    registry.resolve(
        handle,
        ElicitationResumePayload(
            deferred_handle=handle,
            action="accept",
            content={"q0": "yes"},
        ),
    )


# ---------------------------------------------------------------------------
# Level 1: Direct NativeTurn.execute() — core turn generator
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
async def test_stream_complete_via_native_turn() -> None:
    """StreamCompleteEvent must be yielded after in-process elicitation resume.

    Directly drives ``NativeTurn.execute()``. If this passes, the core
    turn generator is healthy — the bug (if present) is in the
    integration layer (RunHandle/EventBus).
    """
    agent = _make_agent()

    async with agent:
        run_ctx = AgentRunContext(session_id="test-native-turn-resume")

        turn = NativeTurn(
            agent=agent,
            prompts=["Call the elicit tool"],
            run_ctx=run_ctx,
            message_history=[],
        )

        events: list[RichAgentStreamEvent[Any]] = []

        async def collect_events() -> None:
            async for event in turn.execute():
                events.append(event)  # noqa: PERF401

        collector = asyncio.create_task(collect_events())

        # Wait for the elicitation future to be registered, then resolve it.
        await _wait_for_future_and_resolve(run_ctx, collector)

        # Wait for turn.execute() to complete after future resolution.
        try:
            await asyncio.wait_for(collector, timeout=30.0)
        except TimeoutError:
            collector.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await collector
            pytest.fail(
                "turn.execute() did not complete within 30s after future "
                "resolution — the generator may be stuck. "
                f"Events so far: {[type(e).__name__ for e in events]}"
            )

        stream_complete_events = [e for e in events if isinstance(e, StreamCompleteEvent)]
        assert len(stream_complete_events) > 0, (
            "StreamCompleteEvent was NOT yielded after future resolution. "
            "The agent run reached End node but NativeTurn.execute() "
            "did not yield StreamCompleteEvent. "
            f"Events collected: {[type(e).__name__ for e in events]}"
        )


# ---------------------------------------------------------------------------
# Level 2: RunHandle.start() → EventBus → consumer — integration layer
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
async def test_stream_complete_via_run_handle_event_bus() -> None:
    """StreamCompleteEvent must reach the EventBus consumer after resume.

    Drives ``RunHandle.start()`` which creates a ``NativeTurn``,
    iterates its events, publishes them to ``EventBus``, and yields
    them. A consumer subscribed to the ``EventBus`` must receive
    ``StreamCompleteEvent``.

    If Level 1 passes but this test fails, the bug is in
    ``RunHandle.start()`` event forwarding or ``EventBus`` delivery
    after future resolution.
    """
    agent = _make_agent()

    async with agent:
        event_bus = EventBus()
        session_id = "test-runhandle-resume"

        session = SessionState(
            session_id=session_id,
            agent_name="test-elicit-agent",
        )
        # Initialize comm_channel so RunHandle._execute_turn can publish events.
        session._comm_channel = ProtocolChannel(
            journal=MemoryJournal(),
            event_bus=event_bus,
            session_id=session_id,
        )

        run_ctx = AgentRunContext(
            session_id=session_id,
            event_bus=event_bus,
        )

        run_handle = RunHandle(
            run_id="test-run-resume",
            session_id=session_id,
            agent_type="test-elicit-agent",
            agent=agent,
            event_bus=event_bus,
            session=session,
            run_ctx=run_ctx,
        )

        # Subscribe to EventBus BEFORE starting the run.
        receive_stream = await event_bus.subscribe(session_id, scope="session")

        # Start the run in a background task.
        async def _drive_run() -> None:
            async for _ in run_handle.start("Call the elicit tool"):
                pass

        drive_task = asyncio.create_task(_drive_run())

        # Wait for the elicitation future to be registered, then resolve it.
        await _wait_for_future_and_resolve(run_ctx, drive_task)

        # Consume events from EventBus, waiting for StreamCompleteEvent.
        received_events: list[RichAgentStreamEvent[Any]] = []
        stream_complete_received = False

        try:
            async with asyncio.timeout(30):
                while True:
                    try:
                        envelope: EventEnvelope = await receive_stream.get()
                    except anyio.EndOfStream:
                        break

                    event: RichAgentStreamEvent[Any] = envelope.event
                    received_events.append(event)

                    if isinstance(event, StreamCompleteEvent):
                        stream_complete_received = True
                        break
        except TimeoutError:
            pytest.fail(
                "Timed out waiting for StreamCompleteEvent on EventBus "
                "after future resolution. "
                f"Received {len(received_events)} events: "
                f"{[type(e).__name__ for e in received_events]}"
            )
        finally:
            run_handle.close()
            drive_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await drive_task

        assert stream_complete_received, (
            "Consumer never received StreamCompleteEvent from EventBus "
            "after future resolution. "
            f"Events received: {[type(e).__name__ for e in received_events]}"
        )


# ---------------------------------------------------------------------------
# Bug 11: Elicitation timeout should yield StreamCompleteEvent(cancelled=True)
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
async def test_elicitation_timeout_yields_cancelled_stream_complete() -> None:
    """RunAbortedError from elicitation timeout yields StreamCompleteEvent(cancelled=True).

    When the elicitation timeout fires, handle_elicitation() raises
    RunAbortedError. NativeTurn catches it and yields
    StreamCompleteEvent(cancelled=True) so that:

    1. The ACP event converter emits TurnCompleteUpdate(stop_reason="cancelled")
       — clients know the turn was cancelled, not completed normally.
    2. _execute_turn saves the final message to agent.conversation
       (the StreamCompleteEvent branch handles this), preserving history.
    3. _consume_run breaks on StreamCompleteEvent and closes the generator,
       setting _turn_complete_event — unblocking legacy clients.

    Previously this yielded RunErrorEvent(stop_reason="refusal"), which
    caused history loss because _execute_turn's StreamCompleteEvent branch
    was never taken, leaving agent.conversation without the assistant message.
    """
    agent = _make_agent()

    async with agent:
        run_ctx = AgentRunContext(session_id="test-timeout-error")
        run_ctx.elicitation_timeout = 0.1  # 100ms timeout

        turn = NativeTurn(
            agent=agent,
            prompts=["Call the elicit tool"],
            run_ctx=run_ctx,
            message_history=[],
        )

        events: list[RichAgentStreamEvent[Any]] = []

        async def collect_events() -> None:
            async for event in turn.execute():
                events.append(event)  # noqa: PERF401

        collector = asyncio.create_task(collect_events())

        # Wait for the elicitation future to be registered (registry is
        # set by get_agentlet() inside NativeTurn.execute()).
        while not collector.done():
            registry = run_ctx.elicitation_registry
            if registry is not None and len(registry) > 0:
                break
            await asyncio.sleep(0.01)

        # Wait for the turn to complete (timeout should fire quickly).
        try:
            await asyncio.wait_for(collector, timeout=10.0)
        except TimeoutError:
            collector.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await collector
            pytest.fail(
                f"Turn did not complete within 10s. Events: {[type(e).__name__ for e in events]}"
            )

        # Assert: StreamCompleteEvent(cancelled=True) should be yielded.
        from wolfharness.agents.events.events import RunErrorEvent

        stream_complete_events = [e for e in events if isinstance(e, StreamCompleteEvent)]
        run_error_events = [e for e in events if isinstance(e, RunErrorEvent)]

        assert len(run_error_events) == 0, (
            "RunErrorEvent should NOT be yielded on elicitation timeout. "
            "StreamCompleteEvent(cancelled=True) is the correct event. "
            f"Events: {[type(e).__name__ for e in events]}"
        )
        assert len(stream_complete_events) >= 1, (
            "StreamCompleteEvent should be yielded on elicitation timeout. "
            f"Events: {[type(e).__name__ for e in events]}"
        )
        assert stream_complete_events[0].cancelled is True, (
            "StreamCompleteEvent should have cancelled=True on elicitation timeout. "
            f"Got cancelled={stream_complete_events[0].cancelled}"
        )
