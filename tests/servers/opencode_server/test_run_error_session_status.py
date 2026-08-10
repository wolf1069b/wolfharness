"""Tests for RunErrorEvent handling in the session status match block.

When ``RunErrorEvent`` is yielded as a terminal event (no trailing
``StreamCompleteEvent``), the OpenCode TUI session stays "busy" forever
because the session status match block in ``_handle_event`` did not handle
``RunErrorEvent``.

These tests verify that the ``RunErrorEvent`` case:
1. Sets session status to "idle"
2. Finalizes assistant time (``_finalize_assistant_time``)
3. Sets aborted error on the assistant message
4. Persists the assistant message (``_persist_assistant_message``)
5. Persists context for resume (``_persist_context_for_resume``)
6. Does NOT reset ``_message_registered``
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest

from wolfharness.agents.events.events import RunErrorEvent
from wolfharness.orchestrator.core import EventEnvelope
from wolfharness_server.opencode_server.event_processor_context import (
    EventProcessorContext,
)
from wolfharness_server.opencode_server.models import (
    AssistantMessage,
    MessageAbortedError,
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
        self.set_session_context_data = self._resume_contexts.__setitem__
        self.get_session_context_data = lambda sid: self._resume_contexts.pop(sid, None)


def _make_ctx(
    session_id: str = "sess-err",
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
    session_id: str = "sess-err",
    *,
    message_registered: bool = True,
) -> tuple[_FakeBridge, EventProcessorContext, list[Any]]:
    """Create a bridge with mocked dependencies for RunErrorEvent testing.

    Returns (bridge, ctx, broadcast_calls).
    """
    ctx = _make_ctx(session_id)
    bridge = _FakeBridge()
    bridge._contexts[session_id] = ctx
    bridge._message_registered[session_id] = message_registered

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

    # Mock cleanup methods on the bridge
    bridge._finalize_assistant_time = AsyncMock()  # type: ignore[method-assign]
    bridge._persist_assistant_message = AsyncMock()  # type: ignore[method-assign]
    bridge._persist_context_for_resume = AsyncMock()  # type: ignore[method-assign]

    return bridge, ctx, broadcast_calls


def _make_envelope(session_id: str, event: Any) -> EventEnvelope:
    return EventEnvelope(source_session_id=session_id, event=event)


def _patch_set_session_status() -> AsyncMock:
    """Patch set_session_status in the bridge module."""
    from unittest.mock import patch

    return patch(
        "wolfharness_server.opencode_server.opencode_event_bridge.set_session_status",
        new_callable=AsyncMock,
    )


def _patch_append_message() -> AsyncMock:
    """Patch append_message_to_session in the bridge module."""
    from unittest.mock import patch

    return patch(
        "wolfharness_server.opencode_server.opencode_event_bridge.append_message_to_session",
        new_callable=AsyncMock,
    )


# =============================================================================
# RunErrorEvent sets session status to "idle"
# =============================================================================


@pytest.mark.anyio
async def test_run_error_sets_session_idle() -> None:
    """RunErrorEvent SHALL set session status to 'idle'."""
    session_id = "sess-err-idle"
    bridge, _ctx, _broadcast = _setup_bridge(session_id)

    event = RunErrorEvent(message="Something went wrong", run_id="r1", agent_name="a")

    with _patch_set_session_status() as mock_set_status:
        await bridge._handle_event(session_id, _make_envelope(session_id, event))

    # Verify set_session_status was called with type="idle"
    mock_set_status.assert_called_once()
    _args, kwargs = mock_set_status.call_args
    status = kwargs.get("status") or (_args[2] if len(_args) > 2 else None)
    assert status is not None
    assert status.type == "idle", f"Expected session status 'idle', got '{status.type}'"


# =============================================================================
# RunErrorEvent finalizes assistant time
# =============================================================================


@pytest.mark.anyio
async def test_run_error_finalizes_assistant_time() -> None:
    """RunErrorEvent SHALL call _finalize_assistant_time."""
    session_id = "sess-err-finalize"
    bridge, _ctx, _broadcast = _setup_bridge(session_id)

    event = RunErrorEvent(message="Error occurred", run_id="r1", agent_name="a")

    with _patch_set_session_status():
        await bridge._handle_event(session_id, _make_envelope(session_id, event))

    bridge._finalize_assistant_time.assert_called_once_with(session_id)  # type: ignore[attr-defined]


# =============================================================================
# RunErrorEvent sets aborted error on assistant message
# =============================================================================


@pytest.mark.anyio
async def test_run_error_sets_aborted_error() -> None:
    """RunErrorEvent SHALL set MessageAbortedError on assistant message.

    The error message SHALL come from RunErrorEvent.message.
    """
    session_id = "sess-err-abort"
    bridge, ctx, broadcast_calls = _setup_bridge(session_id)

    error_message = "Model API returned 500"
    event = RunErrorEvent(message=error_message, run_id="r1", agent_name="a")

    with _patch_set_session_status():
        await bridge._handle_event(session_id, _make_envelope(session_id, event))

    # Verify the assistant message has an error set
    info = ctx.assistant_msg.info
    assert isinstance(info, AssistantMessage)
    assert info.error is not None, "assistant_msg.info.error must be set"
    assert isinstance(info.error, MessageAbortedError)
    assert info.error.data.message == error_message, (
        f"Expected error message '{error_message}', got '{info.error.data.message}'"
    )

    # Verify a MessageUpdatedEvent was broadcast with the error
    updated_events = [e for e in broadcast_calls if e.__class__.__name__ == "MessageUpdatedEvent"]
    assert len(updated_events) >= 1, (
        "MessageUpdatedEvent must be broadcast after setting aborted error"
    )


# =============================================================================
# RunErrorEvent persists assistant message
# =============================================================================


@pytest.mark.anyio
async def test_run_error_persists_assistant_message() -> None:
    """RunErrorEvent SHALL call _persist_assistant_message."""
    session_id = "sess-err-persist"
    bridge, _ctx, _broadcast = _setup_bridge(session_id)

    event = RunErrorEvent(message="Persist me", run_id="r1", agent_name="a")

    with _patch_set_session_status():
        await bridge._handle_event(session_id, _make_envelope(session_id, event))

    bridge._persist_assistant_message.assert_called_once_with(session_id)  # type: ignore[attr-defined]


# =============================================================================
# RunErrorEvent persists context for resume
# =============================================================================


@pytest.mark.anyio
async def test_run_error_persists_context_for_resume() -> None:
    """RunErrorEvent SHALL call _persist_context_for_resume."""
    session_id = "sess-err-resume"
    bridge, _ctx, _broadcast = _setup_bridge(session_id)

    event = RunErrorEvent(message="Resume context", run_id="r1", agent_name="a")

    with _patch_set_session_status():
        await bridge._handle_event(session_id, _make_envelope(session_id, event))

    bridge._persist_context_for_resume.assert_called_once_with(session_id)  # type: ignore[attr-defined]


# =============================================================================
# RunErrorEvent does NOT reset _message_registered
# =============================================================================


@pytest.mark.anyio
async def test_run_error_does_not_reset_message_registered() -> None:
    """RunErrorEvent SHALL keep _message_registered=True (same as RunFailedEvent)."""
    session_id = "sess-err-noreg"
    bridge, _ctx, _broadcast = _setup_bridge(session_id, message_registered=True)

    event = RunErrorEvent(message="No reset", run_id="r1", agent_name="a")

    with _patch_set_session_status():
        await bridge._handle_event(session_id, _make_envelope(session_id, event))

    assert bridge._message_registered.get(session_id) is True, (
        "_message_registered must stay True after RunErrorEvent "
        "so D1 fires on the next RunStartedEvent"
    )


# =============================================================================
# RunErrorEvent with unregistered message triggers C3 fallback
# =============================================================================


@pytest.mark.anyio
async def test_run_error_registers_unregistered_message() -> None:
    """RunErrorEvent SHALL register assistant message if C3 was not triggered."""
    session_id = "sess-err-c3"
    bridge, _ctx, _broadcast = _setup_bridge(session_id, message_registered=False)

    event = RunErrorEvent(message="C3 fallback", run_id="r1", agent_name="a")

    with _patch_set_session_status(), _patch_append_message() as mock_append:
        await bridge._handle_event(session_id, _make_envelope(session_id, event))

    mock_append.assert_called_once()
    assert bridge._message_registered.get(session_id) is True


# =============================================================================
# RunErrorEvent does NOT broadcast SessionErrorEvent (EventProcessor handles it)
# =============================================================================


@pytest.mark.anyio
async def test_run_error_does_not_broadcast_session_error() -> None:
    """RunErrorEvent SHALL NOT broadcast SessionErrorEvent from the match block.

    The EventProcessor.process() already yields SessionErrorEvent for
    RunErrorEvent, so the match block must not duplicate it.
    """
    session_id = "sess-err-no-sse"
    bridge, _ctx, broadcast_calls = _setup_bridge(session_id)

    event = RunErrorEvent(message="No SSE", run_id="r1", agent_name="a")

    with _patch_set_session_status():
        await bridge._handle_event(session_id, _make_envelope(session_id, event))

    session_error_events = [
        e for e in broadcast_calls if e.__class__.__name__ == "SessionErrorEvent"
    ]
    assert len(session_error_events) == 0, (
        "SessionErrorEvent must NOT be broadcast from the match block — "
        "EventProcessor.process() already handles it"
    )


# =============================================================================
# Bug 5 regression guard: RunErrorEvent transitions session to idle
# =============================================================================


@pytest.mark.anyio
async def test_run_error_event_transitions_session_to_idle() -> None:
    """RunErrorEvent sets session status to 'idle' in the event bridge.

    Bug 5 regression guard: When a RunErrorEvent is the terminal event
    (no trailing StreamCompleteEvent), the session must transition to
    'idle' so the TUI can accept new prompts. Without this, the session
    stays 'busy' forever after an error.
    """
    session_id = "sess-bug5-idle"
    bridge, _ctx, _broadcast = _setup_bridge(session_id)

    event = RunErrorEvent(
        message="Bug 5: session stuck in busy",
        run_id="run-bug5",
        agent_name="test-agent",
    )

    with _patch_set_session_status() as mock_set_status:
        await bridge._handle_event(session_id, _make_envelope(session_id, event))

    mock_set_status.assert_called_once()
    _args, kwargs = mock_set_status.call_args
    status = kwargs.get("status") or (_args[2] if len(_args) > 2 else None)
    assert status is not None
    assert status.type == "idle", (
        f"Bug 5 regression: session status should be 'idle' after RunErrorEvent, "
        f"got '{status.type}'"
    )
