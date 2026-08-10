"""Tests for P3: EventProcessorContext serialization for session resume.

P3 activates the recovery path by serializing the EventProcessorContext
after StreamCompleteEvent/RunFailedEvent and calling
``set_session_context_data()``. On the next ``_before_consumer_loop()``
call (same-process resume), the context is deserialized and restored
instead of creating a fresh one.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest

from wolfharness.agents.events.events import (
    RunFailedEvent,
    StreamCompleteEvent,
)
from wolfharness.messaging import ChatMessage
from wolfharness.orchestrator.core import EventEnvelope
from wolfharness_server.opencode_server.event_adapter import OpenCodeEventAdapter
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
    """Yield items from a list as an async iterator."""
    for item in items:
        yield item


def _make_test_ctx(
    session_id: str = "sess-p3",
    *,
    completed: int | None = None,
    msg_id: str = "msg-assistant-1",
) -> EventProcessorContext:
    """Create an EventProcessorContext with an AssistantMessage for P3 tests."""
    assistant_msg = MessageWithParts.assistant(
        message_id=msg_id,
        session_id=session_id,
        time=MessageTime(created=1000, completed=completed),
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

    def set_session_context_data(self, session_id: str, data: dict[str, Any]) -> None:
        self._resume_contexts[session_id] = data

    def get_session_context_data(self, session_id: str) -> dict[str, Any] | None:
        return self._resume_contexts.pop(session_id, None)


def _setup_bridge_for_handle(
    session_id: str = "sess-p3",
    *,
    completed: int | None = None,
    message_registered: bool = True,
) -> tuple[_FakeBridge, EventProcessorContext, list[Any]]:
    """Set up a _FakeBridge ready for _handle_event calls."""
    bridge = _FakeBridge()
    ctx = _make_test_ctx(session_id, completed=completed)

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

    return bridge, ctx, broadcast_calls


@pytest.mark.anyio
async def test_context_serialized_after_stream_complete() -> None:
    """P3: After StreamCompleteEvent, set_session_context_data is called with serialized context.

    Given: A session with a valid EventProcessorContext.
    When: StreamCompleteEvent is handled by _handle_event.
    Then: _resume_contexts[session_id] contains serialized context data
        with the correct assistant_msg_id.
    """
    session_id = "sess-p3-sc"
    bridge, ctx, _broadcast_calls = _setup_bridge_for_handle(
        session_id, completed=None, message_registered=True
    )

    # Add some mutable state to verify serialization preserves it
    ctx.response_text = "accumulated response text"
    ctx.input_tokens = 42
    ctx.output_tokens = 100

    event = StreamCompleteEvent(
        message=ChatMessage(content="done", role="assistant"),
        session_id=session_id,
    )
    envelope = EventEnvelope(source_session_id=session_id, event=event)

    with (
        __import__("unittest.mock").mock.patch(
            "wolfharness_server.opencode_server.opencode_event_bridge.set_session_status",
            new_callable=AsyncMock,
        ),
        __import__("unittest.mock").mock.patch(
            "wolfharness_server.opencode_server.opencode_event_bridge.append_message_to_session",
            new_callable=AsyncMock,
        ),
    ):
        await bridge._handle_event(session_id, envelope)

    # Verify context was serialized and stored
    resume_data = bridge._resume_contexts.get(session_id)
    assert resume_data is not None, "set_session_context_data should have been called"
    assert resume_data["session_id"] == session_id
    assert resume_data["assistant_msg_id"] == "msg-assistant-1"
    assert resume_data["response_text"] == "accumulated response text"
    assert resume_data["input_tokens"] == 42
    assert resume_data["output_tokens"] == 100


@pytest.mark.anyio
async def test_context_restored_on_resume() -> None:
    """P3: _before_consumer_loop restores context from serialized data.

    Given: set_session_context_data() was called with serialized context.
    When: _before_consumer_loop() is called for that session.
    Then: The context is deserialized and _message_registered is set to True.
    """
    session_id = "sess-p3-resume"

    # Create a context, serialize it, and store it
    original_ctx = _make_test_ctx(session_id, completed=5000, msg_id="msg-original")
    original_ctx.response_text = "restored text"
    original_ctx.input_tokens = 77
    serialized = original_ctx.serialize()

    bridge = _FakeBridge()
    bridge.server_state = MagicMock()
    bridge.server_state.working_dir = "/tmp"
    bridge.set_session_context_data(session_id, serialized)

    # Call _before_consumer_loop — should restore from serialized data
    await bridge._before_consumer_loop(session_id)

    # Verify context was restored
    restored_ctx = bridge._contexts.get(session_id)
    assert restored_ctx is not None, "Context should be restored from serialized data"
    assert restored_ctx.session_id == session_id
    assert restored_ctx.assistant_msg_id == "msg-original"
    assert restored_ctx.response_text == "restored text"
    assert restored_ctx.input_tokens == 77

    # Verify _message_registered is True (resume flag)
    assert bridge._message_registered.get(session_id) is True

    # Verify adapter was created
    assert session_id in bridge._adapters
    assert isinstance(bridge._adapters[session_id], OpenCodeEventAdapter)


@pytest.mark.anyio
async def test_serialization_failure_falls_back_to_fresh_context() -> None:
    """P3: If ctx.serialize() raises, error is logged and turn doesn't crash.

    Given: A session where ctx.serialize() raises an exception.
    When: StreamCompleteEvent is handled by _handle_event.
    Then: An error is logged, the turn does NOT crash, and no resume
        data is stored.
    """
    import wolfharness_server.opencode_server.opencode_event_bridge as bridge_module

    session_id = "sess-p3-fail"
    bridge, ctx, _broadcast_calls = _setup_bridge_for_handle(
        session_id, completed=None, message_registered=True
    )

    # Mock serialize() to raise an exception
    original_serialize = ctx.serialize
    ctx.serialize = Mock(side_effect=RuntimeError("serialization explosion"))

    event = StreamCompleteEvent(
        message=ChatMessage(content="done", role="assistant"),
        session_id=session_id,
    )
    envelope = EventEnvelope(source_session_id=session_id, event=event)

    with (
        __import__("unittest.mock").mock.patch(
            "wolfharness_server.opencode_server.opencode_event_bridge.set_session_status",
            new_callable=AsyncMock,
        ),
        __import__("unittest.mock").mock.patch(
            "wolfharness_server.opencode_server.opencode_event_bridge.append_message_to_session",
            new_callable=AsyncMock,
        ),
        __import__("unittest.mock").mock.patch.object(
            bridge_module.logger, "exception"
        ) as mock_error,
    ):
        # Should NOT raise
        await bridge._handle_event(session_id, envelope)

    # Error should have been logged
    assert mock_error.called, "An error should be logged when serialize() fails"
    error_msg = mock_error.call_args.args[0]
    assert "serialize" in error_msg.lower() or "resume" in error_msg.lower()

    # No resume data should have been stored
    assert session_id not in bridge._resume_contexts

    # _message_registered should stay True (StreamCompleteEvent no longer
    # resets it — D1 handles the reset on the next RunStartedEvent)
    assert bridge._message_registered.get(session_id) is True

    # Restore original to avoid affecting other tests
    ctx.serialize = original_serialize


@pytest.mark.anyio
async def test_context_serialized_after_run_failed() -> None:
    """P3: After RunFailedEvent, set_session_context_data is also called.

    Given: A session with a valid EventProcessorContext.
    When: RunFailedEvent is handled by _handle_event.
    Then: _resume_contexts[session_id] contains serialized context data.
    """
    session_id = "sess-p3-rf"
    bridge, ctx, _broadcast_calls = _setup_bridge_for_handle(
        session_id, completed=None, message_registered=True
    )

    ctx.response_text = "partial response before failure"

    event = RunFailedEvent(
        run_id="run-fail-p3",
        exception=RuntimeError("test failure"),
        session_id=session_id,
    )
    envelope = EventEnvelope(source_session_id=session_id, event=event)

    with (
        __import__("unittest.mock").mock.patch(
            "wolfharness_server.opencode_server.opencode_event_bridge.set_session_status",
            new_callable=AsyncMock,
        ),
        __import__("unittest.mock").mock.patch(
            "wolfharness_server.opencode_server.opencode_event_bridge.append_message_to_session",
            new_callable=AsyncMock,
        ),
    ):
        await bridge._handle_event(session_id, envelope)

    resume_data = bridge._resume_contexts.get(session_id)
    assert resume_data is not None, "set_session_context_data should have been called"
    assert resume_data["response_text"] == "partial response before failure"


@pytest.mark.anyio
async def test_fresh_context_when_no_resume_data() -> None:
    """P3: _before_consumer_loop creates fresh context when no resume data.

    Given: No set_session_context_data() call was made.
    When: _before_consumer_loop() is called.
    Then: A fresh EventProcessorContext is created (original behavior).
    """
    session_id = "sess-p3-fresh"
    bridge = _FakeBridge()
    bridge.server_state = MagicMock()
    bridge.server_state.working_dir = "/tmp"
    bridge.server_state.resolve_default_model_info = Mock(
        return_value=("test-model", "test-provider")
    )
    bridge.session_pool.sessions.get_session = Mock(return_value=None)

    await bridge._before_consumer_loop(session_id)

    ctx = bridge._contexts.get(session_id)
    assert ctx is not None, "A fresh context should be created"
    assert ctx.session_id == session_id
    assert ctx.response_text == ""
    assert ctx.input_tokens == 0

    # _message_registered should be False for a fresh context
    assert bridge._message_registered.get(session_id) is False
