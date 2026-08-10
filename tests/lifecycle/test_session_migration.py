"""Tests for SessionController & SessionPool migration to per-prompt RunHandle.

In the per-prompt model, lifecycle dimensions (TriggerSource, Journal,
SnapshotStore, CommChannel, EventTransport) live on SessionState, not
RunHandle. These are initialized by
``SessionController._initialize_lifecycle_and_recovery()`` during
``get_or_create_session_agent()``.

The following test categories were REMOVED because they tested RunHandle
features that no longer exist:

- _trigger_source on RunHandle — moved to SessionState
- _comm_channel on RunHandle — moved to SessionState
- followup() — removed (routing moves to SessionState.prompt_queue)
- steer() via ProtocolChannel.deliver_feedback — steer() now uses
  agent_run.enqueue() or run_ctx.queued_steer_messages
- _closing / _closed flags — replaced by complete_event.is_set()
- steer() after close() raises RuntimeError — steer() after close()
  no longer raises; it just queues to run_ctx.queued_steer_messages
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import ANY, AsyncMock, MagicMock

import anyio
import pytest

from tests._controller_helpers import send_via_controller
from wolfharness.agents.events import (
    RunStartedEvent,
    StreamCompleteEvent,
)
from wolfharness.lifecycle import (
    DirectChannel,
    MemoryJournal,
    ProtocolChannel,
    ProtocolTrigger,
)
from wolfharness.messaging import ChatMessage
from wolfharness.orchestrator.event_bus import EventBus
from wolfharness.orchestrator.session_controller import (
    SessionController,
    SessionState,
)
from wolfharness.orchestrator.session_pool import SessionPool
from wolfharness.orchestrator.turn import Turn


pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _StubTurn(Turn):
    """Minimal Turn implementation for testing."""

    def __init__(
        self,
        *,
        events: list[Any] | None = None,
        message_history: list[Any] | None = None,
        raise_exc: BaseException | None = None,
    ) -> None:
        self._events = events or []
        self._history = message_history or []
        self._raise = raise_exc
        self._final_message = ChatMessage(content="done", role="assistant")

    async def execute(self):  # type: ignore[override]
        if self._raise is not None:
            raise self._raise
        self._message_history = self._history
        for event in self._events:
            yield event


def _stream_complete_event() -> StreamCompleteEvent[Any]:
    return StreamCompleteEvent(message=ChatMessage(content="done", role="assistant"))


def _make_mock_agent() -> MagicMock:
    """Create a mock agent with a stub turn."""
    agent = MagicMock()
    agent.AGENT_TYPE = "native"
    agent.name = "test_agent"
    agent.conversation = MagicMock()
    agent.conversation.get_history.return_value = []
    agent.conversation.add_chat_messages = MagicMock()
    agent.create_turn = MagicMock(return_value=_StubTurn(events=[_stream_complete_event()]))
    agent._interrupt = MagicMock(return_value=None)
    return agent


def _make_mock_pool(agent: MagicMock | None = None) -> MagicMock:
    """Create a mock AgentPool for SessionController."""
    pool = MagicMock()
    pool.main_agent_name = "test_agent"
    pool.manifest = MagicMock()
    pool.manifest.agents = {}
    pool.get_context = MagicMock(return_value=None)
    pool._factory = MagicMock()
    if agent is not None:
        pool._factory.create_session_agent = AsyncMock(return_value=agent)
    return pool


# ---------------------------------------------------------------------------
# _initialize_lifecycle_and_recovery: dimensions on SessionState
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_start_run_handle_creates_protocol_trigger() -> None:
    """_initialize_lifecycle_and_recovery creates ProtocolTrigger on SessionState."""
    agent = _make_mock_agent()
    pool = _make_mock_pool(agent)
    controller = SessionController(pool)
    event_bus = EventBus()
    controller._event_bus = event_bus

    session = SessionState(session_id="s1", agent_name="test_agent")
    controller._sessions["s1"] = session
    controller._session_agents["s1"] = agent

    # Initialize lifecycle dimensions on the session (normally called
    # by get_or_create_session_agent).
    await controller._initialize_lifecycle_and_recovery(session, agent)

    assert isinstance(session._trigger_source, ProtocolTrigger)

    # Cleanup
    if session._comm_channel is not None:
        session._comm_channel.close()


@pytest.mark.unit
async def test_start_run_handle_creates_protocol_channel() -> None:
    """_initialize_lifecycle_and_recovery creates ProtocolChannel on SessionState."""
    agent = _make_mock_agent()
    pool = _make_mock_pool(agent)
    controller = SessionController(pool)
    event_bus = EventBus()
    controller._event_bus = event_bus

    session = SessionState(session_id="s1", agent_name="test_agent")
    controller._sessions["s1"] = session
    controller._session_agents["s1"] = agent

    await controller._initialize_lifecycle_and_recovery(session, agent)

    assert isinstance(session._comm_channel, ProtocolChannel)
    assert session._comm_channel is not None
    assert session._comm_channel._session_id == "s1"
    assert session._comm_channel._event_bus is event_bus

    # Cleanup
    session._comm_channel.close()


@pytest.mark.unit
async def test_start_run_handle_protocol_channel_has_journal() -> None:
    """The ProtocolChannel has a MemoryJournal injected."""
    agent = _make_mock_agent()
    pool = _make_mock_pool(agent)
    controller = SessionController(pool)
    event_bus = EventBus()
    controller._event_bus = event_bus

    session = SessionState(session_id="s1", agent_name="test_agent")
    controller._sessions["s1"] = session
    controller._session_agents["s1"] = agent

    await controller._initialize_lifecycle_and_recovery(session, agent)

    assert isinstance(session._comm_channel, ProtocolChannel)
    assert isinstance(session._comm_channel._journal, MemoryJournal)

    # Cleanup
    session._comm_channel.close()


# ---------------------------------------------------------------------------
# close_session calls RunHandle.close()
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_close_session_calls_run_handle_close() -> None:
    """close_session() signals RunHandle.close() via the run-turn path."""
    agent = _make_mock_agent()
    pool = _make_mock_pool(agent)
    controller = SessionController(pool)
    event_bus = EventBus()
    controller._event_bus = event_bus

    session = SessionState(session_id="s1", agent_name="test_agent")
    controller._sessions["s1"] = session
    controller._session_agents["s1"] = agent
    controller._session_scopes["s1"] = anyio.CancelScope()

    # Create a run handle but don't start its background task.
    controller._start_run_handle(session, agent, "s1", "hello")
    run_handle = controller._runs[session.current_run_id]

    # Give the background task a moment to start.
    await asyncio.sleep(0.01)

    # Close the session.
    await controller.close_session("s1")

    # The run handle should have complete_event set (either by close()
    # or by the turn completing naturally).
    assert run_handle.complete_event.is_set() is True


# ---------------------------------------------------------------------------
# Event delivery: ProtocolChannel → EventBus
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_protocol_channel_publishes_to_event_bus() -> None:
    """ProtocolChannel.publish() delivers events to EventBus."""
    event_bus = EventBus()
    journal = MemoryJournal()
    channel = ProtocolChannel(journal=journal, event_bus=event_bus, session_id="s1")

    # Subscribe to the EventBus.
    queue = await event_bus.subscribe("s1", scope="session")

    # Publish an event via ProtocolChannel.
    event = RunStartedEvent(
        run_id="r1",
        session_id="s1",
        agent_name="test_agent",
    )
    await channel.publish(event)

    # The event should arrive on the EventBus subscription.
    envelope = await asyncio.wait_for(queue.get(), timeout=1.0)
    assert envelope.event is event

    await event_bus.unsubscribe("s1", queue)
    channel.close()


# ---------------------------------------------------------------------------
# No double-publish when ProtocolChannel is used
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_no_double_publish_with_protocol_channel() -> None:
    """ProtocolChannel.publishes_to_event_bus returns True.

    When ProtocolChannel is the comm_channel, events are published to
    EventBus by the channel itself, so RunHandle.start() skips the
    direct event_bus.publish() call.
    """
    event_bus = EventBus()
    journal = MemoryJournal()
    channel = ProtocolChannel(journal=journal, event_bus=event_bus, session_id="s1")

    # The property should return True.
    assert channel.publishes_to_event_bus is True

    # Cleanup
    channel.close()


@pytest.mark.unit
async def test_double_publish_with_direct_channel() -> None:
    """DirectChannel.publishes_to_event_bus returns False.

    DirectChannel does not publish to EventBus, so RunHandle.start()
    must call event_bus.publish() directly.
    """
    journal = MemoryJournal()
    channel = DirectChannel(journal=journal)

    # The property should return False.
    assert channel.publishes_to_event_bus is False


# ---------------------------------------------------------------------------
# receive_request on closed session
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_receive_request_closed_session_returns_none() -> None:
    """receive_request on a closing session returns None."""
    agent = _make_mock_agent()
    pool = _make_mock_pool(agent)
    controller = SessionController(pool)
    event_bus = EventBus()
    controller._event_bus = event_bus

    session = SessionState(session_id="s1", agent_name="test_agent")
    session.is_closing = True
    controller._sessions["s1"] = session
    controller._session_agents["s1"] = agent

    result = await send_via_controller(controller, "s1", "hello")
    assert result is None


# ---------------------------------------------------------------------------
# SessionPool creates ProtocolTrigger/ProtocolChannel on SessionState
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_session_pool_create_run_handle_creates_protocol_dimensions() -> None:
    """_initialize_lifecycle_and_recovery creates dimensions on SessionState."""
    agent = _make_mock_agent()
    pool = _make_mock_pool(agent)

    # Create SessionPool with minimal setup.
    session_pool = SessionPool(pool, enable_event_bus=True)
    session_pool.sessions._event_bus = session_pool._event_bus

    session = SessionState(session_id="s1", agent_name="test_agent")
    session_pool.sessions._sessions["s1"] = session
    session_pool.sessions._session_agents["s1"] = agent

    # Initialize lifecycle dimensions on the session.
    await session_pool.sessions._initialize_lifecycle_and_recovery(session, agent)

    assert isinstance(session._trigger_source, ProtocolTrigger)
    assert isinstance(session._comm_channel, ProtocolChannel)
    assert session._comm_channel._session_id == "s1"
    assert session._comm_channel._event_bus is session_pool._event_bus

    # Cleanup
    session._comm_channel.close()


# ---------------------------------------------------------------------------
# Event delivery end-to-end through ProtocolChannel → EventBus
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_event_delivery_through_protocol_channel_to_event_bus() -> None:
    """Events published via ProtocolChannel arrive on EventBus subscriptions.

    This verifies the event delivery path that protocol servers rely on:
    ProtocolChannel.publish() → EventBus → ProtocolEventConsumerMixin.
    """
    event_bus = EventBus()

    # Subscribe BEFORE publishing to avoid race conditions.
    queue = await event_bus.subscribe("s1", scope="session")

    journal = MemoryJournal()
    channel = ProtocolChannel(journal=journal, event_bus=event_bus, session_id="s1")

    # Publish multiple events. StateUpdate is intentionally excluded
    # because ProtocolChannel filters StateUpdate from EventBus
    # (they are internal lifecycle signals, not turn events).
    events = [
        RunStartedEvent(run_id="r1", session_id="s1", agent_name="test"),
        StreamCompleteEvent(message=ChatMessage(content="done", role="assistant")),
    ]

    for event in events:
        await channel.publish(event)

    # Verify all events arrive in order.
    received: list[Any] = []
    for _ in range(len(events)):
        envelope = await asyncio.wait_for(queue.get(), timeout=1.0)
        received.append(envelope.event)

    assert len(received) == len(events)
    assert isinstance(received[0], RunStartedEvent)
    assert isinstance(received[1], StreamCompleteEvent)

    await event_bus.unsubscribe("s1", queue)
    channel.close()


# ---------------------------------------------------------------------------
# close_session with active run waits then completes
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_close_session_with_active_run_completes() -> None:
    """close_session with an active run handle completes gracefully."""
    agent = _make_mock_agent()
    pool = _make_mock_pool(agent)
    controller = SessionController(pool)
    event_bus = EventBus()
    controller._event_bus = event_bus

    session = SessionState(session_id="s1", agent_name="test_agent")
    controller._sessions["s1"] = session
    controller._session_agents["s1"] = agent
    controller._session_scopes["s1"] = anyio.CancelScope()

    # Create run handle.
    controller._start_run_handle(session, agent, "s1", "hello")

    # Give background task a moment.
    await asyncio.sleep(0.01)

    # Close session — should not hang.
    await asyncio.wait_for(
        controller.close_session("s1"),
        timeout=15.0,
    )

    # Session should be removed.
    assert controller.get_session("s1") is None


# ---------------------------------------------------------------------------
# SessionState.set_current_run_id() publishes StateUpdate
# (replaces old test_state_transition_publishes_state_update)
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_set_current_run_id_publishes_state_update_running() -> None:
    """set_current_run_id(run_id) publishes StateUpdate with RunState.RUNNING.

    When ``current_run_id`` transitions from None to a non-None value,
    ``SessionState.set_current_run_id()`` calls
    ``comm.on_state_change(RunState.RUNNING)`` and schedules a
    ``StateUpdate`` event publish.
    """
    from wolfharness.agents.events import StateUpdate
    from wolfharness.lifecycle.types import RunState

    session = SessionState(session_id="s1", agent_name="test_agent")
    comm = MagicMock()
    comm.publishes_to_event_bus = True
    comm.publish = AsyncMock()
    session._comm_channel = comm
    session._lifecycle_session_id = "s1"

    assert session.current_run_id is None
    session.set_current_run_id("run-1")

    assert session.current_run_id == "run-1"
    comm.on_state_change.assert_called_once_with(RunState.RUNNING)
    # publish is scheduled via create_task; let it run.
    await asyncio.sleep(0.01)
    comm.publish.assert_called_once()
    published_event = comm.publish.call_args.args[0]
    assert isinstance(published_event, StateUpdate)
    assert published_event.state == RunState.RUNNING
    assert published_event.session_id == "s1"


@pytest.mark.unit
async def test_set_current_run_id_publishes_state_update_idle() -> None:
    """set_current_run_id(None) publishes StateUpdate with RunState.IDLE.

    When ``current_run_id`` transitions from a non-None value to None,
    ``SessionState.set_current_run_id()`` calls
    ``comm.on_state_change(RunState.IDLE)`` and schedules a
    ``StateUpdate`` event publish.
    """
    from wolfharness.agents.events import StateUpdate
    from wolfharness.lifecycle.types import RunState

    session = SessionState(session_id="s1", agent_name="test_agent")
    comm = MagicMock()
    comm.publishes_to_event_bus = True
    comm.publish = AsyncMock()
    session._comm_channel = comm
    session._lifecycle_session_id = "s1"
    session.current_run_id = "run-1"

    session.set_current_run_id(None)

    assert session.current_run_id is None
    comm.on_state_change.assert_called_once_with(RunState.IDLE)
    await asyncio.sleep(0.01)
    comm.publish.assert_called_once()
    published_event = comm.publish.call_args.args[0]
    assert isinstance(published_event, StateUpdate)
    assert published_event.state == RunState.IDLE


@pytest.mark.unit
async def test_set_current_run_id_no_transition_when_same_value() -> None:
    """set_current_run_id with same value does not publish StateUpdate.

    When the new value equals the old value, no transition occurs and
    no StateUpdate is published.
    """
    session = SessionState(session_id="s1", agent_name="test_agent")
    comm = MagicMock()
    comm.publish = AsyncMock()
    session._comm_channel = comm
    session.current_run_id = "run-1"

    session.set_current_run_id("run-1")

    assert session.current_run_id == "run-1"
    comm.on_state_change.assert_not_called()
    comm.publish.assert_not_called()


@pytest.mark.unit
async def test_set_current_run_id_no_comm_channel_no_crash() -> None:
    """set_current_run_id with no comm_channel sets field without crashing.

    When ``_comm_channel`` is None (standalone execution), the method
    only sets ``current_run_id`` without publishing events.
    """
    session = SessionState(session_id="s1", agent_name="test_agent")
    assert session._comm_channel is None

    session.set_current_run_id("run-1")
    assert session.current_run_id == "run-1"

    session.set_current_run_id(None)
    assert session.current_run_id is None


# ---------------------------------------------------------------------------
# _route_message followup (when_idle) enqueues to prompt_queue
# (replaces old test_busy_when_idle_calls_followup which tested followup())
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_route_message_when_idle_enqueues_to_prompt_queue() -> None:
    """_route_message with priority='when_idle' enqueues to prompt_queue.

    When the session has an active run and a message arrives with
    ``priority="when_idle"``, the message is enqueued to
    ``SessionState.prompt_queue`` for the next RunHandle.
    """
    agent = _make_mock_agent()
    pool = _make_mock_pool(agent)
    controller = SessionController(pool)
    event_bus = EventBus()
    controller._event_bus = event_bus

    session = SessionState(session_id="s1", agent_name="test_agent")
    session._comm_channel = DirectChannel(MemoryJournal())
    controller._sessions["s1"] = session
    controller._session_agents["s1"] = agent

    # Create an active run handle so the session is "busy".
    controller._start_run_handle(session, agent, "s1", "first")
    assert session.current_run_id is not None
    assert session.prompt_queue.empty()  # queue is empty initially

    # Send a followup message with when_idle priority.
    result = await controller._route_message(
        session,
        agent,
        "s1",
        "followup msg",
        priority="when_idle",
    )

    # Should return None (message is queued, not processed immediately).
    # This was changed in issue #253: followup messages return None instead
    # of fb.message_id to avoid duplicate session_created broadcasts.
    assert result is None
    # The message should be in prompt_queue.
    assert not session.prompt_queue.empty()
    queued = session.prompt_queue.get_nowait()
    assert queued == "followup msg"

    # Cleanup: cancel the background run task.
    run_id = session.current_run_id
    if run_id is not None:
        run_handle = controller._runs.pop(run_id, None)
        if run_handle is not None:
            run_handle.close()
    session.set_current_run_id(None)


@pytest.mark.unit
async def test_route_message_asap_calls_steer() -> None:
    """_route_message with priority='asap' calls RunHandle.steer().

    When the session has an active run and a message arrives with
    ``priority="asap"``, the message is injected via ``steer()``.
    """
    agent = _make_mock_agent()
    pool = _make_mock_pool(agent)
    controller = SessionController(pool)
    event_bus = EventBus()
    controller._event_bus = event_bus

    session = SessionState(session_id="s1", agent_name="test_agent")
    session._comm_channel = DirectChannel(MemoryJournal())
    controller._sessions["s1"] = session
    controller._session_agents["s1"] = agent

    # Create an active run handle.
    controller._start_run_handle(session, agent, "s1", "first")
    run_id = session.current_run_id
    assert run_id is not None
    run_handle = controller._runs[run_id]

    # Mock steer to capture the call.
    run_handle.steer = MagicMock(return_value="steer-msg-id")

    result = await controller._route_message(
        session,
        agent,
        "s1",
        "steer msg",
        priority="asap",
    )

    assert result == "steer-msg-id"
    run_handle.steer.assert_called_once_with("steer msg", message_id=ANY, emit_user_message=False)

    # Cleanup.
    controller._runs.pop(run_id, None)
    run_handle.close()
    session.set_current_run_id(None)


# ---------------------------------------------------------------------------
# _consume_run chains RunHandles via prompt_queue
# (replaces old test_followup_triggers_new_turn, test_multi_turn_with_protocol_trigger)
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_consume_run_chains_via_prompt_queue() -> None:
    """_consume_run creates a new RunHandle when prompt_queue is non-empty.

    After the first turn terminates, if ``prompt_queue`` has a queued
    prompt, ``_consume_run()`` creates a new RunHandle and executes
    the next turn with that prompt.
    """
    agent = _make_mock_agent()
    pool = _make_mock_pool(agent)
    controller = SessionController(pool)
    event_bus = EventBus()
    controller._event_bus = event_bus

    session = SessionState(session_id="s1", agent_name="test_agent")
    session._comm_channel = DirectChannel(MemoryJournal())
    controller._sessions["s1"] = session
    controller._session_agents["s1"] = agent

    # Track prompts passed to create_turn.
    captured_prompts: list[list[Any]] = []
    original_create_turn = agent.create_turn

    def _capturing_create_turn(*, prompts: list[Any], **kwargs: Any) -> Any:
        captured_prompts.append(list(prompts))
        return original_create_turn(prompts=prompts, **kwargs)

    agent.create_turn = _capturing_create_turn

    # Enqueue a followup prompt BEFORE starting the run.
    # _consume_run will find it after the first turn completes.
    session.prompt_queue.put_nowait("second prompt")

    # Start the run with the first prompt.
    controller._start_run_handle(session, agent, "s1", "first prompt")

    # Wait for both turns to complete.
    await asyncio.sleep(0.3)

    # Two turns should have executed: first prompt + second prompt.
    assert len(captured_prompts) == 2, (
        f"Expected 2 turns, got {len(captured_prompts)}: {captured_prompts}"
    )
    assert captured_prompts[0] == ["first prompt"]
    assert captured_prompts[1] == ["second prompt"]

    # prompt_queue should be drained.
    assert session.prompt_queue.empty()
    # Session should be idle.
    assert session.current_run_id is None


@pytest.mark.unit
async def test_consume_run_no_chain_when_prompt_queue_empty() -> None:
    """_consume_run does not chain when prompt_queue is empty.

    After the single turn completes, if ``prompt_queue`` is empty,
    the session goes idle and no new RunHandle is created.
    """
    agent = _make_mock_agent()
    pool = _make_mock_pool(agent)
    controller = SessionController(pool)
    event_bus = EventBus()
    controller._event_bus = event_bus

    session = SessionState(session_id="s1", agent_name="test_agent")
    session._comm_channel = DirectChannel(MemoryJournal())
    controller._sessions["s1"] = session
    controller._session_agents["s1"] = agent

    captured_prompts: list[list[Any]] = []
    original_create_turn = agent.create_turn

    def _capturing_create_turn(*, prompts: list[Any], **kwargs: Any) -> Any:
        captured_prompts.append(list(prompts))
        return original_create_turn(prompts=prompts, **kwargs)

    agent.create_turn = _capturing_create_turn

    # No prompt in prompt_queue — only the initial prompt.
    controller._start_run_handle(session, agent, "s1", "only prompt")

    await asyncio.sleep(0.2)

    # Only one turn.
    assert len(captured_prompts) == 1
    assert captured_prompts[0] == ["only prompt"]
    assert session.prompt_queue.empty()
    assert session.current_run_id is None


@pytest.mark.unit
async def test_consume_run_fifo_ordering_of_queued_prompts() -> None:
    """_consume_run processes queued prompts in FIFO order.

    Multiple prompts enqueued to ``prompt_queue`` are executed in
    the order they were enqueued.
    """
    agent = _make_mock_agent()
    pool = _make_mock_pool(agent)
    controller = SessionController(pool)
    event_bus = EventBus()
    controller._event_bus = event_bus

    session = SessionState(session_id="s1", agent_name="test_agent")
    session._comm_channel = DirectChannel(MemoryJournal())
    controller._sessions["s1"] = session
    controller._session_agents["s1"] = agent

    captured_prompts: list[list[Any]] = []
    original_create_turn = agent.create_turn

    def _capturing_create_turn(*, prompts: list[Any], **kwargs: Any) -> Any:
        captured_prompts.append(list(prompts))
        return original_create_turn(prompts=prompts, **kwargs)

    agent.create_turn = _capturing_create_turn

    # Enqueue two followup prompts in FIFO order.
    session.prompt_queue.put_nowait("second")
    session.prompt_queue.put_nowait("third")

    # Start with the first prompt.
    controller._start_run_handle(session, agent, "s1", "first")

    await asyncio.sleep(0.5)

    # Three turns in order: first, second, third.
    assert len(captured_prompts) == 3
    assert captured_prompts[0] == ["first"]
    assert captured_prompts[1] == ["second"]
    assert captured_prompts[2] == ["third"]
    assert session.prompt_queue.empty()
    assert session.current_run_id is None
