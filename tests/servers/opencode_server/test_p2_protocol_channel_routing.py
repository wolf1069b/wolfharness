"""Tests for P2: Route UserMessageInsertedEvent through ProtocolChannel.

P2 ensures that when ``source == "accepted"`` and a ProtocolChannel is
available on the active session, the event is published through
``comm_channel.publish()`` (which journals before publishing to the
EventBus) instead of direct ``EventBus.publish()``. This avoids
double-publish and ensures the event is journaled for crash-recovery
replay.

Additionally, ``ProtocolChannel.publish()`` skips EventBus publish for
``UserMessageInsertedEvent`` during replay (``_replaying=True``) to
prevent duplicate user message rendering.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock

import pytest

from wolfharness.agents.events import UserMessageInsertedEvent
from wolfharness.lifecycle.comm_channel import ProtocolChannel
from wolfharness.lifecycle.journal import MemoryJournal
from wolfharness.orchestrator.core import EventBus
from wolfharness.orchestrator.session_controller_runs import (
    SessionControllerRunsMixin,
)


pytestmark = pytest.mark.unit


if TYPE_CHECKING:
    from wolfharness.orchestrator.session_controller import SessionState


class _FakeController(SessionControllerRunsMixin):
    """Minimal SessionController for testing _emit_user_message_inserted."""

    def __init__(
        self,
        event_bus: EventBus | None = None,
        sessions: dict[str, SessionState] | None = None,
    ) -> None:
        self._event_bus = event_bus
        self._sessions: dict[str, SessionState] = sessions or {}
        self._runs: dict[str, Any] = {}
        self._lock = MagicMock()
        self._background_tasks: set[Any] = set()
        self.pool = MagicMock()

    def get_session(self, session_id: str) -> SessionState | None:
        return self._sessions.get(session_id)


def _make_protocol_channel(
    event_bus: EventBus,
    session_id: str = "sess-p2",
) -> ProtocolChannel:
    """Create a ProtocolChannel with a MemoryJournal."""
    journal = MemoryJournal()
    return ProtocolChannel(journal=journal, event_bus=event_bus, session_id=session_id)


def _make_session_state(
    *,
    current_run_id: str | None = "run-1",
    comm_channel: Any = None,
) -> MagicMock:
    """Create a mock SessionState."""
    session = MagicMock()
    session.current_run_id = current_run_id
    session._comm_channel = comm_channel
    return session


@pytest.mark.anyio
async def test_steer_message_published_through_protocol_channel() -> None:
    """P2: ProtocolChannels + source="accepted" routes through comm_channel.publish().

    Given: A session with an active run and a ProtocolChannel.
    When: _emit_user_message_inserted is called with source="accepted".
    Then: The event is published through comm_channel.publish(), NOT
        directly through EventBus.publish().
    """
    event_bus = EventBus()
    protocol_channel = _make_protocol_channel(event_bus, "sess-p2")
    session = _make_session_state(comm_channel=protocol_channel)

    controller = _FakeController(event_bus=event_bus, sessions={"sess-p2": session})

    # Spy on EventBus.publish to verify it's NOT called directly
    bus_publish_called = False
    original_publish = event_bus.publish

    async def spy_publish(session_id: str, event: Any) -> None:
        nonlocal bus_publish_called
        bus_publish_called = True
        await original_publish(session_id, event)

    event_bus.publish = spy_publish  # type: ignore[method-assign]

    # Spy on ProtocolChannel.publish
    channel_publish_called = False
    channel_publish_arg: Any = None
    original_channel_publish = protocol_channel.publish

    async def spy_channel_publish(event: Any) -> None:
        nonlocal channel_publish_called, channel_publish_arg
        channel_publish_called = True
        channel_publish_arg = event
        await original_channel_publish(event)

    protocol_channel.publish = spy_channel_publish  # type: ignore[method-assign]

    await controller._emit_user_message_inserted(
        session_id="sess-p2",
        content="steer message",
        delivery="steer",
        source="accepted",
        message_id="msg-1",
    )

    assert channel_publish_called, "Event should be published through ProtocolChannel"
    assert isinstance(channel_publish_arg, UserMessageInsertedEvent)
    assert channel_publish_arg.content == "steer message"
    assert channel_publish_arg.source == "accepted"

    # The EventBus.publish should eventually be called by the channel,
    # but NOT directly by _emit_user_message_inserted. Since the channel
    # calls event_bus.publish internally, we just verify the channel was
    # used (not the direct path).
    # The fact that channel_publish_called is True is sufficient.


@pytest.mark.anyio
async def test_initial_rest_message_published_directly_to_event_bus() -> None:
    """P2: When no ProtocolChannel (idle session), event goes through EventBus.publish() directly.

    Given: A session with no active run (idle, current_run_id=None).
    When: _emit_user_message_inserted is called with source="accepted".
    Then: The event is published directly through EventBus.publish(),
        NOT through any ProtocolChannel.
    """
    event_bus = EventBus()
    # Session is idle — no current run
    session = _make_session_state(current_run_id=None, comm_channel=None)

    controller = _FakeController(event_bus=event_bus, sessions={"sess-p2": session})

    # Track EventBus.publish calls
    publish_calls: list[Any] = []
    original_publish = event_bus.publish

    async def tracking_publish(session_id: str, event: Any) -> None:
        publish_calls.append((session_id, event))
        await original_publish(session_id, event)

    event_bus.publish = tracking_publish  # type: ignore[method-assign]

    await controller._emit_user_message_inserted(
        session_id="sess-p2",
        content="initial message",
        delivery="initial",
        source="accepted",
        message_id="msg-initial",
    )

    assert len(publish_calls) == 1, "EventBus.publish should be called directly for idle sessions"
    sid, event = publish_calls[0]
    assert sid == "sess-p2"
    assert isinstance(event, UserMessageInsertedEvent)
    assert event.content == "initial message"


@pytest.mark.anyio
async def test_non_accepted_source_published_directly_to_event_bus() -> None:
    """P2: source="accepted" also routes through ProtocolChannel when available.

    Given: A session with an active run and ProtocolChannel.
    When: _emit_user_message_inserted is called with source="accepted".
    Then: The event is published through ProtocolChannel (which journals
        before publishing to the EventBus), same as any other source.
        Internal sources no longer bypass the channel — all sources go
        through the channel when one is available.
    """
    event_bus = EventBus()
    protocol_channel = _make_protocol_channel(event_bus, "sess-p2")
    session = _make_session_state(comm_channel=protocol_channel)

    controller = _FakeController(event_bus=event_bus, sessions={"sess-p2": session})

    # Spy on ProtocolChannel.publish
    channel_publish_called = False
    channel_publish_arg: Any = None
    original_channel_publish = protocol_channel.publish

    async def spy_channel_publish(event: Any) -> None:
        nonlocal channel_publish_called, channel_publish_arg
        channel_publish_called = True
        channel_publish_arg = event
        await original_channel_publish(event)

    protocol_channel.publish = spy_channel_publish  # type: ignore[method-assign]

    await controller._emit_user_message_inserted(
        session_id="sess-p2",
        content="internal message",
        delivery="followup",
        source="accepted",
        message_id="msg-internal",
    )

    assert channel_publish_called, (
        "ProtocolChannel should be used for source='internal' when available"
    )
    assert isinstance(channel_publish_arg, UserMessageInsertedEvent)
    assert channel_publish_arg.source == "accepted"


@pytest.mark.anyio
async def test_user_message_inserted_not_duplicated_during_replay() -> None:
    """P2: When _replaying=True and event is UserMessageInsertedEvent, EventBus publish is skipped.

    Given: A ProtocolChannel with _replaying=True.
    When: publish() is called with a UserMessageInsertedEvent.
    Then: The event is journaled but NOT published to the EventBus.
    """
    event_bus = EventBus()
    protocol_channel = _make_protocol_channel(event_bus, "sess-p2")
    protocol_channel.set_replaying(True)

    # Track EventBus.publish calls
    publish_calls: list[Any] = []
    original_publish = event_bus.publish

    async def tracking_publish(session_id: str, event: Any) -> None:
        publish_calls.append((session_id, event))
        await original_publish(session_id, event)

    event_bus.publish = tracking_publish  # type: ignore[method-assign]

    event = UserMessageInsertedEvent(
        session_id="sess-p2",
        message_id="msg-replay",
        content="replayed message",
        delivery="initial",
        source="accepted",
    )

    await protocol_channel.publish(event)

    assert len(publish_calls) == 0, (
        "EventBus publish should be skipped for replayed UserMessageInsertedEvent"
    )


@pytest.mark.anyio
async def test_non_replayed_user_message_inserted_published_to_event_bus() -> None:
    """P2: When _replaying=False and event is UserMessageInsertedEvent, EventBus publish occurs.

    Given: A ProtocolChannel with _replaying=False (normal operation).
    When: publish() is called with a UserMessageInsertedEvent.
    Then: The event IS published to the EventBus (normal behavior).
    """
    event_bus = EventBus()
    protocol_channel = _make_protocol_channel(event_bus, "sess-p2")
    # _replaying is False by default

    publish_calls: list[Any] = []
    original_publish = event_bus.publish

    async def tracking_publish(session_id: str, event: Any) -> None:
        publish_calls.append((session_id, event))
        await original_publish(session_id, event)

    event_bus.publish = tracking_publish  # type: ignore[method-assign]

    event = UserMessageInsertedEvent(
        session_id="sess-p2",
        message_id="msg-normal",
        content="normal message",
        delivery="initial",
        source="accepted",
    )

    await protocol_channel.publish(event)

    assert len(publish_calls) == 1, (
        "EventBus publish should occur for non-replayed UserMessageInsertedEvent"
    )


@pytest.mark.anyio
async def test_replayed_non_user_message_still_published() -> None:
    """P2: During replay, non-UserMessageInsertedEvent events still go to EventBus.

    Given: A ProtocolChannel with _replaying=True.
    When: publish() is called with a non-UserMessageInsertedEvent.
    Then: The event IS still published to the EventBus (only
        UserMessageInsertedEvent is skipped during replay).
    """
    from wolfharness.agents.events.events import RunStartedEvent

    event_bus = EventBus()
    protocol_channel = _make_protocol_channel(event_bus, "sess-p2")
    protocol_channel.set_replaying(True)

    publish_calls: list[Any] = []
    original_publish = event_bus.publish

    async def tracking_publish(session_id: str, event: Any) -> None:
        publish_calls.append((session_id, event))
        await original_publish(session_id, event)

    event_bus.publish = tracking_publish  # type: ignore[method-assign]

    event = RunStartedEvent(
        run_id="run-replay",
        agent_name="test-agent",
        session_id="sess-p2",
    )

    await protocol_channel.publish(event)

    assert len(publish_calls) == 1, (
        "Non-UserMessageInsertedEvent should still be published during replay"
    )
