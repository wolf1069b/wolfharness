"""State-machine-based tests for the event bridge turn lifecycle.

These tests model the correct state machine:

    Turn:  IDLE → RUNNING → COMPLETE → IDLE
    Msg:   UNREGISTERED → REGISTERED → FINALIZED

Key invariants:
- RunStartedEvent SHALL always create a new assistant message (pop
  _pending_message_ids, reset per-turn state) for turns 2+.
- D1 block SHALL fire on RunStartedEvent when the previous turn's
  assistant message is still registered (i.e., StreamCompleteEvent
  did NOT reset _message_registered).
- StreamCompleteEvent SHALL finalize the assistant message but SHALL
  NOT reset _message_registered (that happens at the next RunStartedEvent).
- Each turn SHALL get a distinct assistant_msg_id from _pending_message_ids.
- append_message_to_session SHALL be called with different message IDs
  for each turn.

Correct event timing diagram (two-turn flow):

    ┌──────────────────────────────────────────────────────────────────────┐
    │                       Turn 1                                        │
    │                                                                      │
    │  REST handler          EventBridge              TUI (SSE)           │
    │  ────────────          ───────────              ──────────           │
    │                                                                      │
    │  route_message()                                                    │
    │  ├─ set _pending_msg_ids["msg-t1"]                                 │
    │  ├─ _emit_user_message_inserted()                                  │
    │  │  └─ EventBus → UserMessageInsertedEvent                        │
    │  │                       │                                          │
    │  │                       ▼                                          │
    │  │              _handle_event()                                    │
    │  │              ├─ is_user_msg=True → skip registration            │
    │  │              └─ adapter.convert_event()                         │
    │  │                       │                                          │
    │  │                       ▼  (SSE: message.updated + part.updated)  │
    │  │                                                                  │
    │  └─ _start_run_handle()                                            │
    │     └─ agent starts → RunStartedEvent                              │
    │                       │                                              │
    │                       ▼                                              │
    │              _handle_event()                                        │
    │              ├─ D1: _message_registered=False → SKIP D1            │
    │              ├─ registration block: _message_registered=False      │
    │              │  ├─ append_message_to_session(ctx.assistant_msg)    │
    │              │  │  (msg_id = "msg-t1" from _before_consumer_loop)  │
    │              │  ├─ broadcast MessageUpdatedEvent                   │
    │              │  ├─ broadcast PartUpdatedEvent(StepStart)           │
    │              │  └─ _message_registered = True                     │
    │              └─ adapter.convert_event() → SSE events               │
    │                                                                      │
    │  ┌─ streaming events (PartStart, PartDelta, ToolCall, ...) ─┐     │
    │  │  Each event → adapter.convert_event() → SSE                │ │     │
    │  └────────────────────────────────────────────────────────────┘     │
    │                                                                      │
    │  StreamCompleteEvent                                                 │
    │  ├─ _finalize_assistant_time()  (set time.completed)               │
    │  ├─ _persist_assistant_message()  (write to DB)                    │
    │  ├─ _persist_context_for_resume()  (P3: serialize ctx)             │
    │  └─ _message_registered stays True  ← KEY: do NOT reset here       │
    │                                                                      │
    └──────────────────────────────────────────────────────────────────────┘
                              │
                              ▼  (user sends next message)
    ┌──────────────────────────────────────────────────────────────────────┐
    │                       Turn 2                                        │
    │                                                                      │
    │  REST handler          EventBridge              TUI (SSE)           │
    │  ────────────          ───────────              ──────────           │
    │                                                                      │
    │  route_message()                                                    │
    │  ├─ set _pending_msg_ids["msg-t2"]                                 │
    │  ├─ _emit_user_message_inserted() → SSE (user msg)                 │
    │  └─ _start_run_handle() → RunStartedEvent                          │
    │                       │                                              │
    │                       ▼                                              │
    │              _handle_event()                                        │
    │              ├─ D1: _message_registered=True → FIRE D1 ✓           │
    │              │  ├─ _finalize_assistant_time(warn=False)             │
    │              │  │  (time.completed already set by StreamComplete)   │
    │              │  ├─ _create_assistant_message()                      │
    │              │  │  ├─ pop _pending_msg_ids → "msg-t2"              │
    │              │  │  └─ create new MessageWithParts(id="msg-t2")     │
    │              │  ├─ reset per-turn state (response_text, tools, ...)│
    │              │  └─ _message_registered = False                     │
    │              ├─ registration block: _message_registered=False      │
    │              │  ├─ append_message_to_session(ctx.assistant_msg)    │
    │              │  │  (msg_id = "msg-t2" — NEW, distinct from t1)    │
    │              │  ├─ broadcast MessageUpdatedEvent                   │
    │              │  ├─ broadcast PartUpdatedEvent(StepStart)           │
    │              │  └─ _message_registered = True                     │
    │              └─ adapter.convert_event() → SSE events               │
    │                                                                      │
    │  ┌─ streaming events ─────────────────────────────────────┐       │
    │  │  All content goes to "msg-t2" (NOT "msg-t1")           │       │
    │  └────────────────────────────────────────────────────────┘       │
    │                                                                      │
    │  StreamCompleteEvent                                                 │
    │  └─ _message_registered stays True  (for turn 3's D1)              │
    │                                                                      │
    └──────────────────────────────────────────────────────────────────────┘

P4 BUG (original implementation): StreamCompleteEvent reset
_message_registered=False → D1 skipped on turn 2 → _pending_msg_ids
not popped → ctx.assistant_msg_id stayed "msg-t1" → all turn 2
content merged into turn 1's assistant message → user messages
appeared at bottom with wrong timestamps.

D1 = the "finalize incomplete turn" block in _handle_event() that
fires when RunStartedEvent arrives and _message_registered is True.
It creates a new assistant message with a fresh ID for the new turn.
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
    """Minimal concrete subclass for testing the mixin."""

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
    session_id: str = "sess-sm",
    msg_id: str = "msg-assistant-1",
) -> EventProcessorContext:
    """Create an EventProcessorContext with an AssistantMessage."""
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
    session_id: str = "sess-sm",
    *,
    message_registered: bool = False,
    pending_msg_id: str | None = None,
) -> tuple[_FakeBridge, EventProcessorContext, list[Any]]:
    """Set up a _FakeBridge ready for multi-turn event simulation.

    Simulates _before_consumer_loop() by popping the first pending_msg_id
    and using it to create the initial EventProcessorContext (just like
    the real _before_consumer_loop does on consumer start).

    Returns:
        (bridge, ctx, broadcast_calls) where broadcast_calls accumulates
        all events passed to broadcast_event.
    """
    bridge = _FakeBridge()

    # Simulate _before_consumer_loop: pop first pending ID for initial context
    initial_msg_id = pending_msg_id if pending_msg_id is not None else "msg-initial"
    ctx = _make_ctx(session_id, msg_id=initial_msg_id)

    bridge._contexts[session_id] = ctx
    bridge._message_registered[session_id] = message_registered
    # Note: _pending_message_ids is consumed by _before_consumer_loop in real
    # code. For tests, we pop it here to simulate that. Subsequent turns
    # set new pending IDs before sending RunStartedEvent.
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
    """Context manager that patches all external calls in _handle_event."""
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
# State Machine: Turn lifecycle tests
# =============================================================================


@pytest.mark.anyio
async def test_turn1_creates_new_assistant_message() -> None:
    """Turn 1: RunStartedEvent → registration block fires → new assistant msg.

    Given: Fresh session, _message_registered=False, _pending_message_ids has
           the REST handler's message ID.
    When: RunStartedEvent arrives.
    Then: D1 block does NOT fire (first turn, no previous message to finalize).
    And: Registration block fires: append_message_to_session called with
         ctx.assistant_msg, _message_registered set to True.
    """
    session_id = "sess-t1"
    bridge, _ctx, _broadcast_calls = _setup_bridge(
        session_id, message_registered=False, pending_msg_id="msg-turn-1"
    )

    event = RunStartedEvent(run_id="run-1", agent_name="test-agent", session_id=session_id)

    with _patch_mocks() as mock_append:
        await bridge._handle_event(session_id, _make_envelope(session_id, event))

    # D1 should NOT have fired (first turn, _message_registered was False)
    # Registration block should have fired
    assert mock_append.called, "append_message_to_session should be called"
    appended_msg = mock_append.call_args.args[2]
    assert appended_msg.info.id == "msg-turn-1", (
        "Should use the pending message ID from REST handler"
    )
    assert bridge._message_registered[session_id] is True


@pytest.mark.anyio
async def test_full_two_turn_flow_assigns_distinct_msg_ids() -> None:
    """Two complete turns: each gets a distinct assistant_msg_id.

    This is the KEY test that the existing tests missed. It verifies:
    - Turn 1: RunStarted → registration → StreamComplete
    - Turn 2: RunStarted → D1 fires → new assistant msg → registration
    - assistant_msg_id differs between turns
    - append_message_to_session called with different message IDs
    """
    session_id = "sess-2turn"
    bridge, ctx, _broadcast = _setup_bridge(
        session_id, message_registered=False, pending_msg_id="msg-turn-1"
    )

    with _patch_mocks() as mock_append:
        # === Turn 1 ===
        await bridge._handle_event(
            session_id,
            _make_envelope(
                session_id,
                RunStartedEvent(run_id="run-1", agent_name="agent", session_id=session_id),
            ),
        )
        turn1_msg_id = ctx.assistant_msg_id
        assert turn1_msg_id == "msg-turn-1"
        assert bridge._message_registered[session_id] is True
        assert mock_append.call_count == 1

        # StreamCompleteEvent finalizes turn 1
        await bridge._handle_event(
            session_id,
            _make_envelope(
                session_id,
                StreamCompleteEvent(
                    message=ChatMessage(content="turn 1 done", role="assistant"),
                    session_id=session_id,
                ),
            ),
        )

        # === Turn 2 ===
        # Set up pending message ID for turn 2 (REST handler would do this)
        bridge._pending_message_ids[session_id] = "msg-turn-2"

        await bridge._handle_event(
            session_id,
            _make_envelope(
                session_id,
                RunStartedEvent(run_id="run-2", agent_name="agent", session_id=session_id),
            ),
        )

        turn2_msg_id = ctx.assistant_msg_id
        assert turn2_msg_id == "msg-turn-2", (
            f"Turn 2 should have a new assistant_msg_id, got {turn2_msg_id}"
        )
        assert turn2_msg_id != turn1_msg_id, "Turn 2 assistant_msg_id must differ from turn 1"
        assert bridge._message_registered[session_id] is True

        # StreamCompleteEvent finalizes turn 2
        await bridge._handle_event(
            session_id,
            _make_envelope(
                session_id,
                StreamCompleteEvent(
                    message=ChatMessage(content="turn 2 done", role="assistant"),
                    session_id=session_id,
                ),
            ),
        )

        # append_message_to_session calls:
        # Turn 1: register(1) + StreamComplete finalize(1) = 2
        # Turn 2: D1 finalize skip (already set) + register(1) + finalize(1) = 2
        # Total: 4 calls
        assert mock_append.call_count == 4, (
            f"Expected 4 append calls (2 per turn), got {mock_append.call_count}"
        )
        # Call 0: turn 1 registration (msg-turn-1)
        # Call 1: turn 1 finalization (msg-turn-1, from StreamComplete)
        # Call 2: turn 2 registration (msg-turn-2, from D1 + register)
        # Call 3: turn 2 finalization (msg-turn-2, from StreamComplete)
        turn1_register = mock_append.call_args_list[0].args[2]
        turn1_finalize = mock_append.call_args_list[1].args[2]
        turn2_register = mock_append.call_args_list[2].args[2]
        turn2_finalize = mock_append.call_args_list[3].args[2]
        assert turn1_register.info.id == "msg-turn-1"
        assert turn1_finalize.info.id == "msg-turn-1"
        assert turn2_register.info.id == "msg-turn-2", (
            f"Turn 2 registration should use msg-turn-2, got {turn2_register.info.id}"
        )
        assert turn2_finalize.info.id == "msg-turn-2", (
            f"Turn 2 finalization should use msg-turn-2, got {turn2_finalize.info.id}"
        )


@pytest.mark.anyio
async def test_d1_fires_on_turn2_after_stream_complete() -> None:
    """D1 block SHALL fire on turn 2 even after StreamCompleteEvent.

    This test verifies the CORRECT state machine behavior:
    - StreamCompleteEvent finalizes the message but does NOT reset
      _message_registered.
    - RunStartedEvent on turn 2 finds _message_registered=True → D1 fires.
    - D1 finalizes previous turn, creates new assistant msg, resets state.

    This is the test that EXPOSES the P4 bug: if P4 resets
    _message_registered=False at StreamCompleteEvent, D1 does NOT fire
    on turn 2, and no new assistant message is created.
    """
    session_id = "sess-d1"
    bridge, _ctx, _broadcast = _setup_bridge(
        session_id, message_registered=False, pending_msg_id="msg-a"
    )

    with _patch_mocks():
        # Turn 1: register assistant message
        await bridge._handle_event(
            session_id,
            _make_envelope(
                session_id,
                RunStartedEvent(run_id="r1", agent_name="a", session_id=session_id),
            ),
        )
        assert bridge._message_registered[session_id] is True

        # Turn 1: StreamComplete
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

        # After StreamComplete, _message_registered SHOULD still be True
        # so that D1 fires on the next RunStartedEvent.
        # If P4 resets it to False, D1 won't fire → BUG.
        assert bridge._message_registered[session_id] is True, (
            "_message_registered should remain True after StreamCompleteEvent "
            "so D1 fires on the next turn. P4's reset breaks this."
        )


@pytest.mark.anyio
async def test_d1_finalizes_previous_turn_on_run_started() -> None:
    """D1 block finalizes the previous turn's assistant message time.

    Given: Turn 1 completed (StreamCompleteEvent set time.completed).
    When: RunStartedEvent arrives for turn 2.
    Then: D1 fires, calls _finalize_assistant_time (no warning since
          time.completed is already set), creates new assistant msg.
    """
    session_id = "sess-d1-fin"
    bridge, ctx, _broadcast = _setup_bridge(
        session_id, message_registered=False, pending_msg_id="msg-fin-1"
    )

    with _patch_mocks():
        # Turn 1: RunStarted + StreamComplete
        await bridge._handle_event(
            session_id,
            _make_envelope(
                session_id,
                RunStartedEvent(run_id="r1", agent_name="a", session_id=session_id),
            ),
        )
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

        # Verify time.completed was set by StreamCompleteEvent
        assert ctx.assistant_msg.info.time.completed is not None

        # Turn 2: set pending ID
        bridge._pending_message_ids[session_id] = "msg-fin-2"

        await bridge._handle_event(
            session_id,
            _make_envelope(
                session_id,
                RunStartedEvent(run_id="r2", agent_name="a", session_id=session_id),
            ),
        )

        # D1 should have created a new assistant message
        assert ctx.assistant_msg_id == "msg-fin-2"
        # New message should have time.completed = None (fresh message)
        assert ctx.assistant_msg.info.time.completed is None, (
            "New assistant message should not have time.completed set"
        )


@pytest.mark.anyio
async def test_d1_warns_on_incomplete_turn() -> None:
    """D1 logs warning when previous turn's StreamCompleteEvent was missed.

    Given: Turn 1 started but StreamCompleteEvent never arrived (e.g., crash).
    When: RunStartedEvent arrives for turn 2.
    Then: D1 fires, _finalize_assistant_time logs warning because
          time.completed is None.
    """
    import wolfharness_server.opencode_server.opencode_event_bridge as bridge_module

    session_id = "sess-d1-warn"
    bridge, _ctx, _broadcast = _setup_bridge(
        session_id, message_registered=False, pending_msg_id="msg-warn-1"
    )

    with _patch_mocks():
        # Turn 1: RunStarted (register message) — no StreamCompleteEvent
        await bridge._handle_event(
            session_id,
            _make_envelope(
                session_id,
                RunStartedEvent(run_id="r1", agent_name="a", session_id=session_id),
            ),
        )
        # _message_registered is now True, time.completed is None

        # Turn 2: set pending ID
        bridge._pending_message_ids[session_id] = "msg-warn-2"

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

        # D1 should have logged a warning about incomplete turn
        d1_warnings = [
            call
            for call in mock_warning.call_args_list
            if call.args
            and ("incomplete turn" in call.args[0].lower() or "StreamCompleteEvent" in call.args[0])
        ]
        assert len(d1_warnings) > 0, (
            "D1 should warn when previous turn's StreamCompleteEvent was missed"
        )


@pytest.mark.anyio
async def test_run_failed_does_not_break_turn2() -> None:
    """Turn 2 works correctly after RunFailedEvent on turn 1.

    Given: Turn 1 failed with RunFailedEvent.
    When: RunStartedEvent arrives for turn 2.
    Then: D1 fires, creates new assistant msg with turn 2's pending ID.
    """
    session_id = "sess-rf"
    bridge, ctx, _broadcast = _setup_bridge(
        session_id, message_registered=False, pending_msg_id="msg-rf-1"
    )

    with _patch_mocks():
        # Turn 1: RunStarted
        await bridge._handle_event(
            session_id,
            _make_envelope(
                session_id,
                RunStartedEvent(run_id="r1", agent_name="a", session_id=session_id),
            ),
        )
        assert bridge._message_registered[session_id] is True

        # Turn 1: RunFailed
        await bridge._handle_event(
            session_id,
            _make_envelope(
                session_id,
                RunFailedEvent(
                    run_id="r1",
                    exception=RuntimeError("test error"),
                    session_id=session_id,
                ),
            ),
        )

        # Turn 2: set pending ID
        bridge._pending_message_ids[session_id] = "msg-rf-2"

        await bridge._handle_event(
            session_id,
            _make_envelope(
                session_id,
                RunStartedEvent(run_id="r2", agent_name="a", session_id=session_id),
            ),
        )

        assert ctx.assistant_msg_id == "msg-rf-2", (
            f"Turn 2 should get new msg ID, got {ctx.assistant_msg_id}"
        )
        assert bridge._message_registered[session_id] is True


@pytest.mark.anyio
async def test_three_turns_each_get_distinct_msg_ids() -> None:
    """Three complete turns: each gets a distinct assistant_msg_id.

    Regression test: ensures the fix works for 3+ turns, not just 2.
    """
    session_id = "sess-3turn"
    bridge, ctx, _broadcast = _setup_bridge(
        session_id, message_registered=False, pending_msg_id="msg-t1"
    )

    with _patch_mocks() as mock_append:
        for turn in range(1, 4):
            pending_id = f"msg-t{turn}"
            bridge._pending_message_ids[session_id] = pending_id

            await bridge._handle_event(
                session_id,
                _make_envelope(
                    session_id,
                    RunStartedEvent(run_id=f"r{turn}", agent_name="a", session_id=session_id),
                ),
            )
            assert ctx.assistant_msg_id == pending_id, (
                f"Turn {turn}: expected {pending_id}, got {ctx.assistant_msg_id}"
            )

            await bridge._handle_event(
                session_id,
                _make_envelope(
                    session_id,
                    StreamCompleteEvent(
                        message=ChatMessage(content=f"turn {turn} done", role="assistant"),
                        session_id=session_id,
                    ),
                ),
            )

        # 2 calls per turn (register + finalize), 3 turns = 6 total
        assert mock_append.call_count == 6, (
            f"Expected 6 append calls (2 per turn), got {mock_append.call_count}"
        )
        for i, call in enumerate(mock_append.call_args_list):
            msg = call.args[2]
            expected_turn = (i // 2) + 1
            expected_id = f"msg-t{expected_turn}"
            assert msg.info.id == expected_id, (
                f"Call {i}: expected {expected_id}, got {msg.info.id}"
            )
