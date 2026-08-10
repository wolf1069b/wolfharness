"""Tests for P4: _message_registered lifecycle in the turn state machine.

P4's original approach (reset _message_registered at StreamCompleteEvent)
was WRONG because it prevented D1 from firing on the next turn, causing
all turns to merge into one assistant message.

The correct state machine:
- StreamCompleteEvent SHALL NOT reset _message_registered.
- RunStartedEvent's D1 block SHALL reset _message_registered after
  creating a new assistant message.
- D1 SHALL fire on turns 2+ (when _message_registered is True from
  the previous turn).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest

from wolfharness.agents.events.events import (
    RunFailedEvent,
    RunStartedEvent,
    StreamCompleteEvent,
)
from wolfharness.messaging import ChatMessage
from wolfharness.orchestrator.core import EventEnvelope
from wolfharness_server.opencode_server.event_processor_context import (
    EventProcessorContext,
)
from wolfharness_server.opencode_server.models import (
    MessagePath,
    MessageTime,
    MessageWithParts,
)
from wolfharness_server.opencode_server.opencode_event_bridge import (
    OpenCodeEventBridgeMixin,
)


pytestmark = pytest.mark.unit


if TYPE_CHECKING:
    from collections.abc import AsyncIterator


async def _async_iter(items: list[Any]) -> AsyncIterator[Any]:
    for item in items:
        yield item


class _FakeBridge(OpenCodeEventBridgeMixin):
    def __init__(self) -> None:
        self.session_pool = MagicMock()
        self.server_state = MagicMock()
        self._contexts: dict[str, Any] = {}
        self._adapters: dict[str, Any] = {}
        self._message_registered: dict[str, bool] = {}
        self._child_to_parent: dict[str, str] = {}
        self._child_spawns: dict[str, Any] = {}
        self._children_of: dict[str, set[str]] = {}
        self._resume_contexts: dict[str, dict[str, Any]] = {}
        self._pending_message_ids: dict[str, str] = {}
        self._pending_message_metadata: dict[str, dict[str, str | None]] = {}
        self._steer_split_ids: dict[str, set[str]] = {}
        self.set_session_context_data = self._resume_contexts.__setitem__
        self.get_session_context_data = lambda sid: self._resume_contexts.pop(sid, None)


def _make_ctx(
    session_id: str = "sess-p4",
    msg_id: str = "msg-assistant-1",
) -> EventProcessorContext:
    assistant_msg = MessageWithParts.assistant(
        message_id=msg_id,
        session_id=session_id,
        time=MessageTime(created=1000),
        agent_name="test-agent",
        model_id="test-model",
        parent_id=session_id,
        provider_id="test-provider",
        path=MessagePath(cwd="/tmp", root="/tmp"),
        mode="test-agent",
    )
    return EventProcessorContext(
        session_id=session_id,
        assistant_msg_id=msg_id,
        assistant_msg=assistant_msg,
        state=MagicMock(),
        working_dir="/tmp",
    )


def _setup_bridge(
    session_id: str = "sess-p4",
    *,
    message_registered: bool = False,
    pending_msg_id: str | None = None,
) -> tuple[_FakeBridge, EventProcessorContext, list[Any]]:
    initial_msg_id = pending_msg_id if pending_msg_id is not None else "msg-initial"
    ctx = _make_ctx(session_id, msg_id=initial_msg_id)

    bridge = _FakeBridge()
    bridge._contexts[session_id] = ctx
    bridge._message_registered[session_id] = message_registered
    if pending_msg_id is not None:
        bridge._pending_message_ids.pop(session_id, None)

    adapter_mock = MagicMock()
    adapter_mock.convert_event = lambda _e: _async_iter([])
    bridge._adapters[session_id] = adapter_mock

    broadcast_calls: list[Any] = []

    async def fake_broadcast(event: Any) -> None:
        broadcast_calls.append(event)

    bridge.server_state.broadcast_event = fake_broadcast  # type: ignore[method-assign]
    bridge.server_state.working_dir = "/tmp"
    bridge.server_state.resolve_default_model_info = Mock(
        return_value=("test-model", "test-provider")
    )
    bridge.session_pool.sessions.get_session = Mock(return_value=None)

    return bridge, ctx, broadcast_calls


def _make_envelope(session_id: str, event: Any) -> EventEnvelope:
    return EventEnvelope(source_session_id=session_id, event=event)


def _patch_mocks():
    from contextlib import contextmanager

    @contextmanager
    def _ctx():
        with (
            __import__("unittest.mock").mock.patch(
                "wolfharness_server.opencode_server.opencode_event_bridge.set_session_status",
                new_callable=AsyncMock,
            ),
            __import__("unittest.mock").mock.patch(
                "wolfharness_server.opencode_server.opencode_event_bridge.append_message_to_session",
                new_callable=AsyncMock,
            ) as mock_append,
        ):
            yield mock_append

    return _ctx()


# =============================================================================
# State machine: StreamCompleteEvent SHALL NOT reset _message_registered
# =============================================================================


@pytest.mark.anyio
async def test_stream_complete_does_not_reset_message_registered() -> None:
    """StreamCompleteEvent SHALL keep _message_registered=True.

    This is critical: if _message_registered is reset to False here,
    the next RunStartedEvent's D1 block won't fire, and no new assistant
    message will be created for turn 2. All subsequent turns would merge
    into turn 1's assistant message.
    """
    session_id = "sess-p4-sc"
    bridge, _ctx, _broadcast = _setup_bridge(
        session_id, message_registered=True, pending_msg_id="msg-1"
    )

    event = StreamCompleteEvent(
        message=ChatMessage(content="done", role="assistant"),
        session_id=session_id,
    )

    with _patch_mocks():
        await bridge._handle_event(session_id, _make_envelope(session_id, event))

    assert bridge._message_registered.get(session_id) is True, (
        "_message_registered must stay True after StreamCompleteEvent "
        "so D1 fires on the next RunStartedEvent"
    )


@pytest.mark.anyio
async def test_run_failed_does_not_reset_message_registered() -> None:
    """RunFailedEvent SHALL keep _message_registered=True (same reasoning)."""
    session_id = "sess-p4-rf"
    bridge, _ctx, _broadcast = _setup_bridge(
        session_id, message_registered=True, pending_msg_id="msg-1"
    )

    event = RunFailedEvent(
        run_id="run-fail-1",
        exception=RuntimeError("test failure"),
        session_id=session_id,
    )

    with _patch_mocks():
        await bridge._handle_event(session_id, _make_envelope(session_id, event))

    assert bridge._message_registered.get(session_id) is True, (
        "_message_registered must stay True after RunFailedEvent "
        "so D1 fires on the next RunStartedEvent"
    )


# =============================================================================
# State machine: D1 SHALL fire on turn 2 after StreamCompleteEvent
# =============================================================================


@pytest.mark.anyio
async def test_d1_fires_on_turn2_after_stream_complete() -> None:
    """D1 block fires on turn 2 after turn 1 completed normally.

    This is the KEY test that the original P4 implementation broke.
    With the correct state machine:
    - Turn 1: RunStarted → register → _message_registered=True
    - Turn 1: StreamComplete → finalize (keep _message_registered=True)
    - Turn 2: RunStarted → D1 fires (True) → new msg → _message_registered=False
    - Turn 2: register → _message_registered=True
    """
    import wolfharness_server.opencode_server.opencode_event_bridge as bridge_module

    session_id = "sess-p4-d1"
    bridge, ctx, _broadcast = _setup_bridge(
        session_id, message_registered=False, pending_msg_id="msg-t1"
    )

    with _patch_mocks():
        # Turn 1: RunStarted → register
        await bridge._handle_event(
            session_id,
            _make_envelope(
                session_id,
                RunStartedEvent(run_id="r1", agent_name="a", session_id=session_id),
            ),
        )
        assert bridge._message_registered[session_id] is True

        # Turn 1: StreamComplete → finalize (keep True)
        await bridge._handle_event(
            session_id,
            _make_envelope(
                session_id,
                StreamCompleteEvent(
                    message=ChatMessage(content="done", role="assistant"),
                    session_id=session_id,
                ),
            ),
        )
        assert bridge._message_registered[session_id] is True

        # Turn 2: set pending ID
        bridge._pending_message_ids[session_id] = "msg-t2"

        # Turn 2: RunStarted → D1 should fire
        with (
            __import__("unittest.mock").mock.patch.object(
                bridge_module.logger, "warning"
            ) as mock_warning,
        ):
            await bridge._handle_event(
                session_id,
                _make_envelope(
                    session_id,
                    RunStartedEvent(run_id="r2", agent_name="a", session_id=session_id),
                ),
            )

        # D1 should NOT log a warning (time.completed was already set
        # by StreamCompleteEvent, so _finalize_assistant_time is a no-op)
        d1_warnings = [
            call
            for call in mock_warning.call_args_list
            if call.args
            and ("incomplete turn" in call.args[0].lower() or "StreamCompleteEvent" in call.args[0])
        ]
        assert len(d1_warnings) == 0, "D1 should not warn when previous turn completed normally"

        # D1 should have created a new assistant message
        assert ctx.assistant_msg_id == "msg-t2", (
            f"D1 should have popped pending ID 'msg-t2', got {ctx.assistant_msg_id}"
        )
        assert bridge._message_registered[session_id] is True
