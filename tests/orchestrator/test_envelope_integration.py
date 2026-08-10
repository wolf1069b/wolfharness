"""Integration tests for EventEnvelope wrapping behavior."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from wolfharness.orchestrator.core import EventBus, EventEnvelope


pytestmark = pytest.mark.integration


if TYPE_CHECKING:
    import asyncio

    from wolfharness import AgentPool


def _queue_empty(queue: asyncio.Queue) -> bool:
    """Check if an asyncio.Queue has no buffered items without consuming them."""
    return queue.empty()


class TestEventEnvelopeIntegration:
    """Integration tests for EventEnvelope wrapping behavior."""

    async def test_child_event_routing_source_session_id(self, minimal_pool: AgentPool) -> None:
        """Parent with scope='descendants' receives EventEnvelope with child source_session_id."""
        assert minimal_pool.session_pool is not None
        controller = minimal_pool.session_pool.sessions
        await controller.get_or_create_session("parent-sid")
        await controller.get_or_create_session("child-sid", parent_session_id="parent-sid")
        bus = EventBus(session_controller=controller)
        parent_queue = await bus.subscribe("parent-sid", scope="descendants")

        event = {"type": "test", "data": "hello from child"}
        await bus.publish("child-sid", event)

        assert not _queue_empty(parent_queue), (
            "Parent subscriber should receive child events wrapped in EventEnvelope"
        )
        received = await parent_queue.get()
        assert isinstance(received, EventEnvelope), (
            f"Expected EventEnvelope, got {type(received).__name__}"
        )
        assert received.source_session_id == "child-sid", (
            f"Expected source_session_id='child-sid', got {received.source_session_id!r}"
        )
        assert received.event == event, (
            f"Wrapped event should match original. Expected {event!r}, got {received.event!r}"
        )

    async def test_transparent_forwarding(self) -> None:
        """Attribute access transparently forwards to the wrapped event via __getattr__."""
        from wolfharness.agents.events import StreamCompleteEvent
        from wolfharness.messaging import ChatMessage

        message = ChatMessage(content="hello world", role="assistant")
        event = StreamCompleteEvent(message=message)
        envelope = EventEnvelope(source_session_id="sess-1", event=event)

        # Transparent forwarding: envelope.message should equal envelope.event.message
        assert envelope.message is envelope.event.message, (
            "envelope.message should transparently forward to"
            " envelope.event.message via __getattr__"
        )
        assert envelope.message.content == "hello world", (
            "Forwarded attribute should expose the wrapped event's data"
        )

    async def test_field_precedence(self) -> None:
        """EventEnvelope.source_session_id is not shadowed by event attribute."""

        # Create an event that has its own source_session_id attribute
        class EventWithSourceSessionId:
            def __init__(self) -> None:
                self.source_session_id = "event-source"
                self.message = "hello"

        raw_event = EventWithSourceSessionId()
        envelope = EventEnvelope(source_session_id="envelope-source", event=raw_event)

        # Field precedence: envelope's own source_session_id should win
        assert envelope.source_session_id == "envelope-source", (
            "EventEnvelope.source_session_id should take precedence over event attribute"
        )
        # But other attributes should still forward transparently
        assert envelope.message == "hello", (
            "Non-envelope attributes should still forward via __getattr__"
        )
        # The event's own source_session_id should still be accessible via .event
        assert envelope.event.source_session_id == "event-source", (
            "Event's source_session_id should still be accessible via envelope.event"
        )

    async def test_envelope_is_frozen(self) -> None:
        """EventEnvelope is immutable (frozen dataclass)."""
        envelope = EventEnvelope(source_session_id="sess-1", event={"data": "test"})

        with pytest.raises(AttributeError):
            envelope.source_session_id = "sess-2"

        with pytest.raises(AttributeError):
            envelope.event = {"data": "modified"}

    async def test_envelope_repr(self) -> None:
        """EventEnvelope repr includes source_session_id and event."""
        event = {"type": "test"}
        envelope = EventEnvelope(source_session_id="sess-1", event=event)
        repr_str = repr(envelope)

        assert "EventEnvelope" in repr_str
        assert "sess-1" in repr_str
        assert "test" in repr_str
