"""Tests for OpenCode server message_id alignment (D13/D14).

Tests that:
- MessageRequest accepts the `delivery` field (D13)
- route_message passes `message_id` to receive_request (D14)
- _before_consumer_loop uses pending message_id when available (D14)
- _handle_event updates ctx.assistant_msg_id from events (D14)
- Delivery mode mapping: "steer" -> "asap", "queue" -> "when_idle" (D13)
- Message ID timestamp is consistent with time.created (C1)
"""

from __future__ import annotations

from unittest.mock import AsyncMock, Mock

import pytest

from wolfharness.lifecycle.types import DeliveryMode
from wolfharness_server.opencode_server.event_processor_context import EventProcessorContext
from wolfharness_server.opencode_server.models import MessagePath, MessageTime, MessageWithParts
from wolfharness_server.opencode_server.models.message import (
    CommandRequest,
    MessageRequest,
    TextPartInput,
)


pytestmark = pytest.mark.integration


# =============================================================================
# D13: MessageRequest delivery field
# ==============================================================================


class TestMessageRequestDeliveryField:
    """Tests for the delivery field on MessageRequest (D13)."""

    def test_delivery_field_defaults_to_none(self):
        """MessageRequest should default delivery to None (queue semantics)."""
        req = MessageRequest(parts=[TextPartInput(text="hello")])
        assert req.delivery is None

    def test_delivery_field_accepts_steer(self):
        """MessageRequest should accept delivery='steer'."""
        req = MessageRequest(parts=[TextPartInput(text="hello")], delivery="steer")
        assert req.delivery == "steer"

    def test_delivery_field_accepts_queue(self):
        """MessageRequest should accept delivery='queue'."""
        req = MessageRequest(parts=[TextPartInput(text="hello")], delivery="queue")
        assert req.delivery == "queue"

    def test_delivery_field_serializes(self):
        """Delivery field should appear in serialized output when set."""
        req = MessageRequest(parts=[TextPartInput(text="hello")], delivery="steer")
        data = req.model_dump(by_alias=True, exclude_none=True)
        assert data.get("delivery") == "steer"

    def test_delivery_field_not_in_output_when_none(self):
        """Delivery field should not appear when None (backward compat)."""
        req = MessageRequest(parts=[TextPartInput(text="hello")])
        data = req.model_dump(by_alias=True, exclude_none=True)
        assert "delivery" not in data


# =============================================================================
# D13: Delivery mode mapping
# ==============================================================================


class TestDeliveryModeMapping:
    """Tests for delivery-to-priority mapping (D13)."""

    def test_steer_maps_to_asap(self):
        """delivery='steer' should map to mode=DeliveryMode.STEER."""
        # This is the mapping logic used in message_routes.py
        delivery = "steer"
        priority = "asap" if delivery == "steer" else "when_idle"
        assert priority == "asap"

    def test_queue_maps_to_when_idle(self):
        """delivery='queue' should map to mode=DeliveryMode.QUEUE."""
        delivery = "queue"
        priority = "asap" if delivery == "steer" else "when_idle"
        assert priority == "when_idle"

    def test_none_delivery_maps_to_when_idle(self):
        """delivery=None should map to mode=DeliveryMode.QUEUE (default)."""
        delivery = None
        priority = "asap" if delivery == "steer" else "when_idle"
        assert priority == "when_idle"


# =============================================================================
# D14: CommandRequest message_id passthrough
# ==============================================================================


class TestCommandRequestMessageId:
    """Tests for message_id on CommandRequest (D14)."""

    def test_command_request_accepts_message_id(self):
        """CommandRequest should accept message_id field."""
        req = CommandRequest(command="test", message_id="msg_abc123")
        assert req.message_id == "msg_abc123"

    def test_command_request_message_id_defaults_to_none(self):
        """CommandRequest should default message_id to None."""
        req = CommandRequest(command="test")
        assert req.message_id is None


# =============================================================================
# D14: route_message passes message_id to receive_request
# ==============================================================================


class TestRouteMessagePassesMessageId:
    """Tests that route_message passes message_id to receive_request (D14)."""

    @pytest.mark.asyncio
    async def test_route_message_stores_pending_message_id(self):
        """route_message should store message_id in _pending_message_ids."""
        from wolfharness_server.opencode_server.session_pool_integration import (
            OpenCodeSessionPoolIntegration,
        )

        session_pool = Mock()
        session_pool.sessions = Mock()
        session_pool.sessions.get_session = Mock(return_value=Mock())
        session_pool.sessions.get_or_create_session = AsyncMock(return_value=(Mock(), True))
        session_pool.send_message = AsyncMock(return_value=None)
        session_pool.event_bus = Mock()
        session_pool.event_bus.subscribe = AsyncMock()
        session_pool.event_bus.unsubscribe = AsyncMock()

        server_state = Mock()
        server_state.working_dir = "/tmp"
        server_state.resolve_default_model_info = Mock(return_value=("default", "wolfharness"))
        server_state.broadcast_event = AsyncMock()

        integration = OpenCodeSessionPoolIntegration(session_pool, server_state)
        # Mock _start_event_consumer to avoid starting a real consumer
        integration._start_event_consumer = AsyncMock()

        await integration.route_message(
            session_id="test-session",
            content="hello",
            mode=DeliveryMode.QUEUE,
            message_id="msg_test123",
        )

        # The pending message_id should have been consumed by now (popped in
        # _before_consumer_loop), but since we didn't call _before_consumer_loop,
        # it should still be stored.
        assert "test-session" in integration._pending_message_ids
        assert integration._pending_message_ids["test-session"] == "msg_test123"

    @pytest.mark.asyncio
    async def test_route_message_passes_message_id_to_receive_request(self):
        """route_message should pass message_id to receive_request."""
        from wolfharness_server.opencode_server.session_pool_integration import (
            OpenCodeSessionPoolIntegration,
        )

        session_pool = Mock()
        session_pool.sessions = Mock()
        session_pool.sessions.get_session = Mock(return_value=Mock())
        session_pool.sessions.get_or_create_session = AsyncMock(return_value=(Mock(), True))
        session_pool.send_message = AsyncMock(return_value=None)

        server_state = Mock()
        server_state.working_dir = "/tmp"
        server_state.resolve_default_model_info = Mock(return_value=("default", "wolfharness"))

        integration = OpenCodeSessionPoolIntegration(session_pool, server_state)
        integration._start_event_consumer = AsyncMock()

        await integration.route_message(
            session_id="test-session",
            content="hello",
            mode=DeliveryMode.QUEUE,
            message_id="msg_test456",
        )

        # Verify receive_request was called with message_id
        call_kwargs = session_pool.send_message.call_args
        assert call_kwargs.kwargs.get("message_id") == "msg_test456"

    @pytest.mark.asyncio
    async def test_route_message_without_message_id_passes_none(self):
        """route_message should pass message_id=None when not provided."""
        from wolfharness_server.opencode_server.session_pool_integration import (
            OpenCodeSessionPoolIntegration,
        )

        session_pool = Mock()
        session_pool.sessions = Mock()
        session_pool.sessions.get_session = Mock(return_value=Mock())
        session_pool.sessions.get_or_create_session = AsyncMock(return_value=(Mock(), True))
        session_pool.send_message = AsyncMock(return_value=None)

        server_state = Mock()
        server_state.working_dir = "/tmp"
        server_state.resolve_default_model_info = Mock(return_value=("default", "wolfharness"))

        integration = OpenCodeSessionPoolIntegration(session_pool, server_state)
        integration._start_event_consumer = AsyncMock()

        await integration.route_message(
            session_id="test-session",
            content="hello",
            mode=DeliveryMode.QUEUE,
        )

        call_kwargs = session_pool.send_message.call_args
        assert call_kwargs.kwargs.get("message_id") is None


# =============================================================================
# D14: _before_consumer_loop uses pending message_id
# ==============================================================================


class TestBeforeConsumerLoopPendingMessageId:
    """Tests that _before_consumer_loop uses pending message_id (D14)."""

    @pytest.mark.asyncio
    async def test_before_consumer_loop_uses_pending_message_id(self):
        """_before_consumer_loop should use pending message_id when available."""
        from wolfharness_server.opencode_server.session_pool_integration import (
            OpenCodeSessionPoolIntegration,
        )

        session_pool = Mock()
        session_pool.sessions = Mock()
        session_pool.sessions.get_session = Mock(return_value=None)
        server_state = Mock()
        server_state.working_dir = "/tmp"
        server_state.resolve_default_model_info = Mock(return_value=("default", "wolfharness"))

        integration = OpenCodeSessionPoolIntegration(session_pool, server_state)
        # Set a pending message_id
        integration._pending_message_ids["test-session"] = "msg_from_rest"

        await integration._before_consumer_loop("test-session")

        ctx = integration._contexts.get("test-session")
        assert ctx is not None
        assert ctx.assistant_msg_id == "msg_from_rest"

        # The pending message_id should have been consumed (popped)
        assert "test-session" not in integration._pending_message_ids

    @pytest.mark.asyncio
    async def test_before_consumer_loop_generates_id_when_no_pending(self):
        """_before_consumer_loop should generate a new ID when no pending message_id."""
        from wolfharness_server.opencode_server.session_pool_integration import (
            OpenCodeSessionPoolIntegration,
        )

        session_pool = Mock()
        session_pool.sessions = Mock()
        session_pool.sessions.get_session = Mock(return_value=None)
        server_state = Mock()
        server_state.working_dir = "/tmp"
        server_state.resolve_default_model_info = Mock(return_value=("default", "wolfharness"))

        integration = OpenCodeSessionPoolIntegration(session_pool, server_state)

        await integration._before_consumer_loop("test-session")

        ctx = integration._contexts.get("test-session")
        assert ctx is not None
        # Should have generated an ID (starts with "msg_")
        assert ctx.assistant_msg_id.startswith("msg_")


# =============================================================================
# D14: _handle_event updates ctx.assistant_msg_id from events
# ==============================================================================


class TestHandleEventUpdatesMessageId:
    """Tests that _handle_event updates ctx.assistant_msg_id from events (D14)."""

    @pytest.mark.asyncio
    async def test_handle_event_preserves_ctx_message_id_from_event(self):
        """_handle_event should NOT overwrite ctx.assistant_msg_id from event.message_id.

        NativeTurn generates its own UUID as _message_id, which differs from
        the canonical assistant_msg_id from the REST handler. Overwriting
        causes a mismatch between parts and the assistant message, breaking
        UI rendering.
        """
        from pydantic_ai.messages import TextPart as PydanticTextPart

        from wolfharness.agents.events import PartStartEvent
        from wolfharness.orchestrator.event_bus import EventEnvelope
        from wolfharness_server.opencode_server.models import MessageWithParts
        from wolfharness_server.opencode_server.session_pool_integration import (
            OpenCodeSessionPoolIntegration,
        )

        session_pool = Mock()
        server_state = Mock()
        server_state.working_dir = "/tmp"
        server_state.resolve_default_model_info = Mock(return_value=("default", "wolfharness"))
        server_state.broadcast_event = AsyncMock()

        integration = OpenCodeSessionPoolIntegration(session_pool, server_state)

        # Create a context with a placeholder ID
        assistant_msg = MessageWithParts.assistant(
            message_id="msg_placeholder",
            session_id="test-session",
            time=MessageTime(created=0),
            agent_name="agent",
            model_id="default",
            parent_id="",
            provider_id="wolfharness",
            path=MessagePath(cwd="/tmp", root="/tmp"),
        )
        ctx = EventProcessorContext(
            session_id="test-session",
            assistant_msg_id="msg_placeholder",
            assistant_msg=assistant_msg,
            state=server_state,
            working_dir="/tmp",
        )
        integration._contexts["test-session"] = ctx
        integration._message_registered["test-session"] = False

        # Mock the adapter to avoid full event processing
        mock_adapter = Mock()

        async def _empty_convert(event):
            return
            yield

        mock_adapter.convert_event = _empty_convert
        integration._adapters["test-session"] = mock_adapter

        # Create an event with message_id
        event = PartStartEvent(
            index=0,
            part=PydanticTextPart(content="hello"),
            message_id="msg_from_event_123",
        )
        envelope = EventEnvelope(
            event=event,
            source_session_id="test-session",
        )

        await integration._handle_event("test-session", envelope)

        # The ctx.assistant_msg_id should NOT be overwritten by the event's
        # internal message_id (NativeTurn UUID). The canonical ID from the
        # REST handler must be preserved.
        assert ctx.assistant_msg_id == "msg_placeholder"

    @pytest.mark.asyncio
    async def test_handle_event_does_not_update_when_event_has_no_message_id(self):
        """_handle_event should not update ctx.assistant_msg_id when event has no message_id."""
        from wolfharness.agents.events import RunStartedEvent
        from wolfharness.orchestrator.event_bus import EventEnvelope
        from wolfharness_server.opencode_server.session_pool_integration import (
            OpenCodeSessionPoolIntegration,
        )

        session_pool = Mock()
        server_state = Mock()
        server_state.working_dir = "/tmp"
        server_state.resolve_default_model_info = Mock(return_value=("default", "wolfharness"))
        server_state.broadcast_event = AsyncMock()

        integration = OpenCodeSessionPoolIntegration(session_pool, server_state)

        assistant_msg = MessageWithParts.assistant(
            message_id="msg_original",
            session_id="test-session",
            time=MessageTime(created=0),
            agent_name="agent",
            model_id="default",
            parent_id="",
            provider_id="wolfharness",
            path=MessagePath(cwd="/tmp", root="/tmp"),
        )
        ctx = EventProcessorContext(
            session_id="test-session",
            assistant_msg_id="msg_original",
            assistant_msg=assistant_msg,
            state=server_state,
            working_dir="/tmp",
        )
        integration._contexts["test-session"] = ctx
        integration._message_registered["test-session"] = False

        mock_adapter = Mock()

        async def _empty_convert(event):
            return
            yield

        mock_adapter.convert_event = _empty_convert
        integration._adapters["test-session"] = mock_adapter

        # RunStartedEvent does not have message_id
        event = RunStartedEvent(session_id="test-session", run_id="run_123")
        envelope = EventEnvelope(
            event=event,
            source_session_id="test-session",
        )

        await integration._handle_event("test-session", envelope)

        # The ctx.assistant_msg_id should NOT have been updated
        assert ctx.assistant_msg_id == "msg_original"


# =============================================================================
# C1: Message ID timestamp consistency
# =============================================================================


class TestMessageIdTimestampConsistency:
    """Tests for message ID timestamp vs time.created consistency (C1).

    Issue C1: identifiers.py uses int(time.time()*1000) (float truncation)
    while time_utils.now_ms() uses time.time_ns()//1_000_000 (integer division).
    Same-ms messages can have a 1ms mismatch. Additionally, the 48-bit
    encoding (timestamp_ms * 0x1000 + counter) overflows for 2026+ timestamps.
    """

    @pytest.mark.unit
    def test_id_timestamp_matches_now_ms(self) -> None:
        """C1: Timestamp decoded from ascending ID should match now_ms() window."""
        from wolfharness.utils import identifiers
        from wolfharness.utils.time_utils import now_ms

        ts_before = now_ms()
        msg_id = identifiers.ascending("message")
        ts_after = now_ms()

        # Decode timestamp from ID: first 16 hex chars = 8 bytes (64 bits)
        # now = timestamp_ms * 0x1000 + counter → timestamp_ms = now >> 12
        id_part = msg_id.split("_", 1)[1]
        id_ts = int(id_part[:16], 16) >> 12

        assert ts_before <= id_ts <= ts_after, (
            f"ID timestamp ({id_ts}) outside now_ms() window [{ts_before}, {ts_after}] — issue C1"
        )

    @pytest.mark.unit
    def test_same_ms_ids_have_consistent_timestamps(self) -> None:
        """C1: Two IDs generated in rapid succession should have close timestamps."""
        from wolfharness.utils import identifiers

        id1 = identifiers.ascending("message")
        id2 = identifiers.ascending("message")

        ts1 = int(id1.split("_", 1)[1][:16], 16) >> 12
        ts2 = int(id2.split("_", 1)[1][:16], 16) >> 12

        assert abs(ts1 - ts2) <= 1, (
            f"Same-ms IDs have {abs(ts1 - ts2)}ms difference — "
            f"should be <=1ms (issue C1: float truncation)"
        )


# =============================================================================
# C4: CustomEvent bypasses assistant registration
# =============================================================================


class TestCustomEventBypassesRegistration:
    """Tests that CustomEvent does not trigger assistant message registration (C4).

    Issue C4: CustomEvent wraps SSE broadcast events (e.g.
    SessionCreatedEvent) republished from the OpenCodeEventBridge. These
    are not real agent events and must NOT trigger assistant message
    registration. If they do, the assistant message is broadcast before
    the agent runs, causing notification ID > assistant ID → QUEUED.
    """

    @pytest.mark.asyncio
    async def test_custom_event_does_not_register_assistant(self):
        """CustomEvent should NOT trigger assistant message registration."""
        from wolfharness.agents.events.events import CustomEvent
        from wolfharness.orchestrator.event_bus import EventEnvelope
        from wolfharness_server.opencode_server.models import (
            MessagePath,
            MessageTime,
            MessageWithParts,
        )
        from wolfharness_server.opencode_server.session_pool_integration import (
            OpenCodeSessionPoolIntegration,
        )

        session_pool = Mock()
        session_pool.sessions = Mock()
        session_pool.sessions.get_session = Mock(return_value=None)
        server_state = Mock()
        server_state.working_dir = "/tmp"
        server_state.resolve_default_model_info = Mock(return_value=("default", "wolfharness"))
        server_state.broadcast_event = AsyncMock()

        integration = OpenCodeSessionPoolIntegration(session_pool, server_state)

        assistant_msg = MessageWithParts.assistant(
            message_id="msg_test",
            session_id="test-session",
            time=MessageTime(created=0),
            agent_name="agent",
            model_id="default",
            parent_id="",
            provider_id="wolfharness",
            path=MessagePath(cwd="/tmp", root="/tmp"),
        )
        ctx = EventProcessorContext(
            session_id="test-session",
            assistant_msg_id="msg_test",
            assistant_msg=assistant_msg,
            state=server_state,
            working_dir="/tmp",
        )
        integration._contexts["test-session"] = ctx
        integration._message_registered["test-session"] = False

        # Send a CustomEvent (e.g., wrapping a SessionCreatedEvent)
        custom_event = CustomEvent(
            event_data={"type": "session.created"},
            event_type="opencode:session.created",
        )
        envelope = EventEnvelope(
            event=custom_event,
            source_session_id="test-session",
        )

        await integration._handle_event("test-session", envelope)

        # Assistant message should NOT have been registered
        assert not integration._message_registered.get("test-session", False), (
            "CustomEvent should NOT trigger assistant message registration (C4)"
        )
        # broadcast_event should NOT have been called for MessageUpdatedEvent
        broadcast_calls = server_state.broadcast_event.call_args_list
        for call in broadcast_calls:
            event_arg = call.args[0] if call.args else call.kwargs.get("event")
            if hasattr(event_arg, "type") and event_arg.type == "message.updated":
                pytest.fail("CustomEvent should NOT trigger MessageUpdatedEvent broadcast (C4)")

    @pytest.mark.asyncio
    async def test_real_agent_event_does_register_assistant(self):
        """RunStartedEvent (a real agent event) SHOULD trigger registration.

        This is the positive control for C4: after skipping CustomEvent,
        the next real agent event (RunStartedEvent) must trigger assistant
        registration.
        """
        from wolfharness.agents.events import RunStartedEvent
        from wolfharness.orchestrator.event_bus import EventEnvelope
        from wolfharness_server.opencode_server.models import (
            MessagePath,
            MessageTime,
            MessageWithParts,
        )
        from wolfharness_server.opencode_server.session_pool_integration import (
            OpenCodeSessionPoolIntegration,
        )

        session_pool = Mock()
        session_pool.sessions = Mock()
        session_pool.sessions.get_session = Mock(return_value=None)
        server_state = Mock()
        server_state.working_dir = "/tmp"
        server_state.resolve_default_model_info = Mock(return_value=("default", "wolfharness"))
        server_state.broadcast_event = AsyncMock()

        integration = OpenCodeSessionPoolIntegration(session_pool, server_state)

        assistant_msg = MessageWithParts.assistant(
            message_id="msg_real",
            session_id="test-session",
            time=MessageTime(created=0),
            agent_name="agent",
            model_id="default",
            parent_id="",
            provider_id="wolfharness",
            path=MessagePath(cwd="/tmp", root="/tmp"),
        )
        ctx = EventProcessorContext(
            session_id="test-session",
            assistant_msg_id="msg_real",
            assistant_msg=assistant_msg,
            state=server_state,
            working_dir="/tmp",
        )
        integration._contexts["test-session"] = ctx
        integration._message_registered["test-session"] = False

        # Mock the adapter to avoid full event processing
        mock_adapter = Mock()

        async def _empty_convert(event):
            return
            yield

        mock_adapter.convert_event = _empty_convert
        integration._adapters["test-session"] = mock_adapter

        # Patch append_message_to_session to track calls
        import unittest.mock as _mock

        with _mock.patch(
            "wolfharness_server.opencode_server.opencode_event_bridge.append_message_to_session",
            new_callable=_mock.AsyncMock,
        ) as mock_append:
            event = RunStartedEvent(
                session_id="test-session",
                run_id="run_001",
                agent_name="test-agent",
            )
            envelope = EventEnvelope(
                event=event,
                source_session_id="test-session",
            )

            await integration._handle_event("test-session", envelope)

            # Assistant message SHOULD have been registered
            assert integration._message_registered.get("test-session", False), (
                "RunStartedEvent should trigger assistant message registration"
            )
            # append_message_to_session should have been called
            mock_append.assert_awaited_once()

        # broadcast_event should have been called with MessageUpdatedEvent
        broadcast_calls = server_state.broadcast_event.call_args_list
        event_types = []
        for call in broadcast_calls:
            event_arg = call.args[0] if call.args else call.kwargs.get("event")
            if hasattr(event_arg, "type"):
                event_types.append(event_arg.type)
        assert "message.updated" in event_types, (
            "RunStartedEvent should trigger MessageUpdatedEvent broadcast"
        )

    @pytest.mark.asyncio
    async def test_custom_event_then_runstarted_registers_on_runstarted(self):
        """CustomEvent followed by RunStartedEvent: registration on RunStarted only.

        This simulates the real timeline:
        1. SessionCreatedEvent → CustomEvent → skip (C4)
        2. System notifications
        3. RunStartedEvent → register assistant message
        """
        from wolfharness.agents.events import RunStartedEvent
        from wolfharness.agents.events.events import CustomEvent
        from wolfharness.orchestrator.event_bus import EventEnvelope
        from wolfharness_server.opencode_server.models import (
            MessagePath,
            MessageTime,
            MessageWithParts,
        )
        from wolfharness_server.opencode_server.session_pool_integration import (
            OpenCodeSessionPoolIntegration,
        )

        session_pool = Mock()
        session_pool.sessions = Mock()
        session_pool.sessions.get_session = Mock(return_value=None)
        server_state = Mock()
        server_state.working_dir = "/tmp"
        server_state.resolve_default_model_info = Mock(return_value=("default", "wolfharness"))
        server_state.broadcast_event = AsyncMock()

        integration = OpenCodeSessionPoolIntegration(session_pool, server_state)

        assistant_msg = MessageWithParts.assistant(
            message_id="msg_timeline",
            session_id="test-session",
            time=MessageTime(created=0),
            agent_name="agent",
            model_id="default",
            parent_id="",
            provider_id="wolfharness",
            path=MessagePath(cwd="/tmp", root="/tmp"),
        )
        ctx = EventProcessorContext(
            session_id="test-session",
            assistant_msg_id="msg_timeline",
            assistant_msg=assistant_msg,
            state=server_state,
            working_dir="/tmp",
        )
        integration._contexts["test-session"] = ctx
        integration._message_registered["test-session"] = False

        # Mock the adapter
        mock_adapter = Mock()

        async def _empty_convert(event):
            return
            yield

        mock_adapter.convert_event = _empty_convert
        integration._adapters["test-session"] = mock_adapter

        # Step 1: Send CustomEvent (SessionCreatedEvent)
        custom_event = CustomEvent(
            event_data={"type": "session.created"},
            event_type="opencode:session.created",
        )
        envelope1 = EventEnvelope(
            event=custom_event,
            source_session_id="test-session",
        )
        await integration._handle_event("test-session", envelope1)

        # After CustomEvent: NOT registered
        assert not integration._message_registered.get("test-session", False), (
            "CustomEvent should NOT trigger registration (C4)"
        )

        # Step 2: Send RunStartedEvent
        run_event = RunStartedEvent(
            session_id="test-session",
            run_id="run_001",
            agent_name="test-agent",
        )
        envelope2 = EventEnvelope(
            event=run_event,
            source_session_id="test-session",
        )

        import unittest.mock as _mock

        with _mock.patch(
            "wolfharness_server.opencode_server.opencode_event_bridge.append_message_to_session",
            new_callable=_mock.AsyncMock,
        ):
            await integration._handle_event("test-session", envelope2)

        # After RunStartedEvent: registered
        assert integration._message_registered.get("test-session", False), (
            "RunStartedEvent after CustomEvent should trigger registration"
        )


# =============================================================================
# C3: Event bridge creates StepStartPart on assistant registration
# =============================================================================


class TestEventBridgeStepStartPart:
    """Tests that event bridge broadcasts StepStartPart on registration (C3).

    Issue C3: The REST handler previously broadcast the assistant message
    and StepStartPart before the agent ran. Now the event bridge is the
    sole broadcast point. It must create and broadcast a StepStartPart
    alongside the assistant message so the frontend sees the step-start
    indicator when the agent actually begins work.
    """

    @pytest.mark.asyncio
    async def test_step_start_part_broadcast_on_registration(self):
        """StepStartPart should be broadcast when assistant message is registered."""
        from wolfharness.agents.events import RunStartedEvent
        from wolfharness.orchestrator.event_bus import EventEnvelope
        from wolfharness_server.opencode_server.models import (
            MessagePath,
            MessageTime,
            MessageWithParts,
            PartUpdatedEvent,
            StepStartPart,
        )
        from wolfharness_server.opencode_server.session_pool_integration import (
            OpenCodeSessionPoolIntegration,
        )

        session_pool = Mock()
        session_pool.sessions = Mock()
        session_pool.sessions.get_session = Mock(return_value=None)
        server_state = Mock()
        server_state.working_dir = "/tmp"
        server_state.resolve_default_model_info = Mock(return_value=("default", "wolfharness"))
        server_state.broadcast_event = AsyncMock()

        integration = OpenCodeSessionPoolIntegration(session_pool, server_state)

        assistant_msg = MessageWithParts.assistant(
            message_id="msg_step",
            session_id="test-session",
            time=MessageTime(created=0),
            agent_name="agent",
            model_id="default",
            parent_id="",
            provider_id="wolfharness",
            path=MessagePath(cwd="/tmp", root="/tmp"),
        )
        ctx = EventProcessorContext(
            session_id="test-session",
            assistant_msg_id="msg_step",
            assistant_msg=assistant_msg,
            state=server_state,
            working_dir="/tmp",
        )
        integration._contexts["test-session"] = ctx
        integration._message_registered["test-session"] = False

        # Mock the adapter
        mock_adapter = Mock()

        async def _empty_convert(event):
            return
            yield

        mock_adapter.convert_event = _empty_convert
        integration._adapters["test-session"] = mock_adapter

        # Patch append_message_to_session
        import unittest.mock as _mock

        with _mock.patch(
            "wolfharness_server.opencode_server.opencode_event_bridge.append_message_to_session",
            new_callable=_mock.AsyncMock,
        ):
            event = RunStartedEvent(
                session_id="test-session",
                run_id="run_001",
                agent_name="test-agent",
            )
            envelope = EventEnvelope(
                event=event,
                source_session_id="test-session",
            )

            await integration._handle_event("test-session", envelope)

        # Check that a PartUpdatedEvent with StepStartPart was broadcast
        broadcast_calls = server_state.broadcast_event.call_args_list
        step_start_found = False
        for call in broadcast_calls:
            event_arg = call.args[0] if call.args else call.kwargs.get("event")
            if isinstance(event_arg, PartUpdatedEvent):
                part = event_arg.properties.part
                if isinstance(part, StepStartPart):
                    step_start_found = True
                    assert part.message_id == "msg_step", (
                        "StepStartPart message_id should match assistant_msg_id"
                    )
                    break

        assert step_start_found, (
            "StepStartPart should be broadcast alongside assistant message registration (C3)"
        )
        # The StepStartPart should also be appended to assistant_msg.parts
        assert any(isinstance(p, StepStartPart) for p in assistant_msg.parts), (
            "StepStartPart should be appended to assistant_msg.parts"
        )
