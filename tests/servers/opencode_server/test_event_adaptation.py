"""Merged tests for event adaptation in the OpenCode server.

Combines tests from:
- test_event_adapter.py
- test_event_conversion.py
"""

from __future__ import annotations

from typing import Any
from unittest.mock import Mock

from pydantic_ai import (
    PartStartEvent as PydanticPartStartEvent,
    TextPart as PydanticTextPart,
    TextPartDelta,
)
from pydantic_ai.messages import (
    PartDeltaEvent as PydanticPartDeltaEvent,
    PartEndEvent,
)
import pytest

from wolfharness.agents.events import (
    PartDeltaEvent as AgentPoolPartDeltaEvent,
    PartStartEvent,
    RunErrorEvent,
    RunStartedEvent,
    StreamCompleteEvent,
    TextContentItem,
    ToolCallCompleteEvent,
    ToolCallProgressEvent,
    ToolCallStartEvent,
)
from wolfharness_server.opencode_server.event_adapter import OpenCodeEventAdapter
from wolfharness_server.opencode_server.event_processor_context import (
    EventProcessorContext,
)
from wolfharness_server.opencode_server.models import (
    MessagePath,
    MessageTime,
    MessageWithParts,
    PartDeltaEvent,
    PartUpdatedEvent,
    SessionErrorEvent,
    SessionStatusEvent,
)
from wolfharness_server.opencode_server.models.parts import (
    ReasoningPart,
    StepFinishPart,
    TextPart,
    ToolPart,
    ToolStateCompleted,
    ToolStateError,
    ToolStateRunning,
)


pytestmark = pytest.mark.integration


# =============================================================================
# --- Merged from test_event_adapter.py ---
# =============================================================================

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def adapter_context() -> EventProcessorContext:
    """Create an event processor context for testing the adapter."""
    session_id = "test-session"
    assistant_msg_id = "msg-001"
    assistant_msg = MessageWithParts.assistant(
        message_id=assistant_msg_id,
        session_id=session_id,
        time=MessageTime(created=0),
        agent_name="test-agent",
        model_id="test-model",
        provider_id="wolfharness",
        path=MessagePath(cwd="/tmp", root="/tmp"),
        parent_id="msg-000",
    )
    state = Mock()
    state.messages = {}
    state.messages.setdefault(session_id, [])
    state.ensure_session = Mock()
    state.storage = Mock()
    state.storage.log_message = Mock()
    state.model_variants = {}

    return EventProcessorContext(
        session_id=session_id,
        assistant_msg_id=assistant_msg_id,
        assistant_msg=assistant_msg,
        state=state,
        working_dir="/tmp",
    )


# =============================================================================
# Helper
# =============================================================================


async def _collect_events(async_gen) -> list[Any]:
    """Collect all events from an async generator."""
    return [event async for event in async_gen]


# =============================================================================
# PartStartEvent conversion
# =============================================================================


class TestPartStartEventConversion:
    """Tests for PartStartEvent -> OpenCode PartUpdatedEvent."""

    @pytest.mark.asyncio
    async def test_text_part_start_creates_text_part(
        self,
        adapter_context: EventProcessorContext,
    ) -> None:
        """PartStartEvent with TextPart yields PartUpdatedEvent with TextPart."""
        adapter = OpenCodeEventAdapter(context=adapter_context)

        event = PartStartEvent.text(index=0, content="Hello, world!")
        events = await _collect_events(adapter.convert_event(event))

        part_updated = [e for e in events if isinstance(e, PartUpdatedEvent)]
        assert len(part_updated) == 1
        assert isinstance(part_updated[0].properties.part, TextPart)
        assert part_updated[0].properties.part.text == "Hello, world!"

    @pytest.mark.asyncio
    async def test_thinking_part_start_creates_reasoning_part(
        self,
        adapter_context: EventProcessorContext,
    ) -> None:
        """PartStartEvent with ThinkingPart yields PartUpdatedEvent with ReasoningPart."""
        adapter = OpenCodeEventAdapter(context=adapter_context)

        event = PartStartEvent.thinking(index=0, content="Let me think...")
        events = await _collect_events(adapter.convert_event(event))

        part_updated = [e for e in events if isinstance(e, PartUpdatedEvent)]
        assert len(part_updated) == 1
        assert isinstance(part_updated[0].properties.part, ReasoningPart)
        assert part_updated[0].properties.part.text == "Let me think..."

    @pytest.mark.asyncio
    async def test_pydantic_text_part_start_creates_text_part(
        self,
        adapter_context: EventProcessorContext,
    ) -> None:
        """PydanticAI PartStartEvent with TextPart yields PartUpdatedEvent with TextPart."""
        adapter = OpenCodeEventAdapter(context=adapter_context)

        event = PydanticPartStartEvent(index=0, part=PydanticTextPart(content="Pydantic text"))
        events = await _collect_events(adapter.convert_event(event))

        part_updated = [e for e in events if isinstance(e, PartUpdatedEvent)]
        assert len(part_updated) == 1
        assert isinstance(part_updated[0].properties.part, TextPart)
        assert part_updated[0].properties.part.text == "Pydantic text"


# =============================================================================
# PartDeltaEvent conversion
# =============================================================================


class TestPartDeltaEventConversion:
    """Tests for PartDeltaEvent -> OpenCode PartDeltaEvent."""

    @pytest.mark.asyncio
    async def test_text_delta_yields_part_delta_event(
        self,
        adapter_context: EventProcessorContext,
    ) -> None:
        """Text delta should yield PartDeltaEvent."""
        adapter = OpenCodeEventAdapter(context=adapter_context)

        # First establish a text part
        start_event = PartStartEvent.text(index=0, content="Hello")
        await _collect_events(adapter.convert_event(start_event))

        delta_event = AgentPoolPartDeltaEvent.text(index=0, content=", world!")
        events = await _collect_events(adapter.convert_event(delta_event))

        delta_events = [e for e in events if isinstance(e, PartDeltaEvent)]
        assert len(delta_events) == 1
        assert delta_events[0].properties.delta == ", world!"
        assert delta_events[0].properties.field == "text"

    @pytest.mark.asyncio
    async def test_thinking_delta_yields_part_delta_event(
        self,
        adapter_context: EventProcessorContext,
    ) -> None:
        """Thinking delta should yield PartDeltaEvent."""
        adapter = OpenCodeEventAdapter(context=adapter_context)

        # First establish a reasoning part
        start_event = PartStartEvent.thinking(index=0, content="Thinking")
        await _collect_events(adapter.convert_event(start_event))

        delta_event = AgentPoolPartDeltaEvent.thinking(index=0, content=" more...")
        events = await _collect_events(adapter.convert_event(delta_event))

        delta_events = [e for e in events if isinstance(e, PartDeltaEvent)]
        assert len(delta_events) == 1
        assert delta_events[0].properties.delta == " more..."

    @pytest.mark.asyncio
    async def test_pydantic_text_delta_yields_part_delta_event(
        self,
        adapter_context: EventProcessorContext,
    ) -> None:
        """PydanticAI TextPartDelta should yield PartDeltaEvent."""
        adapter = OpenCodeEventAdapter(context=adapter_context)

        # First establish a text part
        start_event = PydanticPartStartEvent(index=0, part=PydanticTextPart(content="Base"))
        await _collect_events(adapter.convert_event(start_event))

        delta_event = PydanticPartDeltaEvent(
            index=0, delta=TextPartDelta(content_delta=" extended")
        )
        events = await _collect_events(adapter.convert_event(delta_event))

        delta_events = [e for e in events if isinstance(e, PartDeltaEvent)]
        assert len(delta_events) == 1
        assert delta_events[0].properties.delta == " extended"


# =============================================================================
# PartEndEvent conversion
# =============================================================================


class TestPartEndEventConversion:
    """Tests for PartEndEvent -> no output (handled internally)."""

    @pytest.mark.asyncio
    async def test_part_end_yields_nothing(
        self,
        adapter_context: EventProcessorContext,
    ) -> None:
        """PartEndEvent should not yield any OpenCode events."""
        adapter = OpenCodeEventAdapter(context=adapter_context)

        # First establish a text part so the end event has something to end
        start_event = PartStartEvent.text(index=0, content="Hello")
        await _collect_events(adapter.convert_event(start_event))

        end_event = PartEndEvent(index=0, part=PydanticTextPart(content="Hello"))
        events = await _collect_events(adapter.convert_event(end_event))

        assert len(events) == 0, f"Expected no events, got {events}"

    @pytest.mark.asyncio
    async def test_part_end_without_prior_start_yields_nothing(
        self,
        adapter_context: EventProcessorContext,
    ) -> None:
        """PartEndEvent without prior start should not crash or yield events."""
        adapter = OpenCodeEventAdapter(context=adapter_context)

        end_event = PartEndEvent(index=0, part=PydanticTextPart(content="Orphan"))
        events = await _collect_events(adapter.convert_event(end_event))

        assert len(events) == 0, f"Expected no events, got {events}"


# =============================================================================
# ToolCallStartEvent conversion
# =============================================================================


class TestToolCallStartEventConversion:
    """Tests for ToolCallStartEvent -> OpenCode PartUpdatedEvent (ToolPart)."""

    @pytest.mark.asyncio
    async def test_tool_call_start_creates_running_tool_part(
        self,
        adapter_context: EventProcessorContext,
    ) -> None:
        """ToolCallStartEvent should yield PartUpdatedEvent with ToolPart in running state."""
        adapter = OpenCodeEventAdapter(context=adapter_context)

        event = ToolCallStartEvent(
            tool_call_id="call-001",
            tool_name="bash",
            title="Running: ls -la",
            raw_input={"command": "ls -la"},
        )
        events = await _collect_events(adapter.convert_event(event))

        part_updated = [e for e in events if isinstance(e, PartUpdatedEvent)]
        assert len(part_updated) == 1
        tool_part = part_updated[0].properties.part
        assert isinstance(tool_part, ToolPart)
        assert tool_part.tool == "bash"
        assert isinstance(tool_part.state, ToolStateRunning)
        assert tool_part.state.input == {"command": "ls -la"}

    @pytest.mark.asyncio
    async def test_tool_call_start_with_empty_input(
        self,
        adapter_context: EventProcessorContext,
    ) -> None:
        """ToolCallStartEvent with no input should still create ToolPart."""
        adapter = OpenCodeEventAdapter(context=adapter_context)

        event = ToolCallStartEvent(
            tool_call_id="call-002",
            tool_name="read",
            title="Reading file",
        )
        events = await _collect_events(adapter.convert_event(event))

        part_updated = [e for e in events if isinstance(e, PartUpdatedEvent)]
        assert len(part_updated) == 1
        tool_part = part_updated[0].properties.part
        assert isinstance(tool_part, ToolPart)
        assert tool_part.tool == "read"
        assert isinstance(tool_part.state, ToolStateRunning)


# =============================================================================
# ToolCallCompleteEvent conversion
# =============================================================================


class TestToolCallCompleteEventConversion:
    """Tests for ToolCallCompleteEvent -> OpenCode PartUpdatedEvent (completed/error)."""

    @pytest.mark.asyncio
    async def test_tool_call_complete_creates_completed_tool_part(
        self,
        adapter_context: EventProcessorContext,
    ) -> None:
        """ToolCallCompleteEvent should yield PartUpdatedEvent with ToolPart in completed state."""
        adapter = OpenCodeEventAdapter(context=adapter_context)

        # First create the tool part
        start_event = ToolCallStartEvent(
            tool_call_id="call-003",
            tool_name="bash",
            title="Running: echo hello",
            raw_input={"command": "echo hello"},
        )
        await _collect_events(adapter.convert_event(start_event))

        complete_event = ToolCallCompleteEvent(
            tool_name="bash",
            tool_call_id="call-003",
            tool_input={"command": "echo hello"},
            tool_result="hello",
            agent_name="test-agent",
            message_id="msg-001",
        )
        events = await _collect_events(adapter.convert_event(complete_event))

        part_updated = [e for e in events if isinstance(e, PartUpdatedEvent)]
        assert len(part_updated) == 1
        tool_part = part_updated[0].properties.part
        assert isinstance(tool_part, ToolPart)
        assert isinstance(tool_part.state, ToolStateCompleted)
        assert tool_part.state.output == "hello"

    @pytest.mark.asyncio
    async def test_tool_call_complete_with_error_creates_error_tool_part(
        self,
        adapter_context: EventProcessorContext,
    ) -> None:
        """ToolCallCompleteEvent with error result should yield ToolPart in error state."""
        adapter = OpenCodeEventAdapter(context=adapter_context)

        # First create the tool part
        start_event = ToolCallStartEvent(
            tool_call_id="call-004",
            tool_name="bash",
            title="Running: false",
            raw_input={"command": "false"},
        )
        await _collect_events(adapter.convert_event(start_event))

        complete_event = ToolCallCompleteEvent(
            tool_name="bash",
            tool_call_id="call-004",
            tool_input={"command": "false"},
            tool_result={"error": "Command failed with exit code 1"},
            agent_name="test-agent",
            message_id="msg-001",
        )
        events = await _collect_events(adapter.convert_event(complete_event))

        part_updated = [e for e in events if isinstance(e, PartUpdatedEvent)]
        assert len(part_updated) == 1
        tool_part = part_updated[0].properties.part
        assert isinstance(tool_part, ToolPart)
        assert isinstance(tool_part.state, ToolStateError)
        assert tool_part.state.error == "Command failed with exit code 1"


# =============================================================================
# StreamCompleteEvent conversion
# =============================================================================


class TestStreamCompleteEventConversion:
    """Tests for StreamCompleteEvent -> StepFinishPart."""

    @pytest.mark.asyncio
    async def test_stream_complete_yields_step_finish_part(
        self,
        adapter_context: EventProcessorContext,
    ) -> None:
        """StreamCompleteEvent should yield PartUpdatedEvent with StepFinishPart."""
        adapter = OpenCodeEventAdapter(context=adapter_context)

        msg = Mock()
        msg.content = "Done"
        msg.usage = None
        msg.cost_info = None
        event = StreamCompleteEvent(message=msg)
        events = await _collect_events(adapter.convert_event(event))

        part_updated = [e for e in events if isinstance(e, PartUpdatedEvent)]
        step_finish = [e for e in part_updated if isinstance(e.properties.part, StepFinishPart)]
        assert len(step_finish) == 1

    @pytest.mark.asyncio
    async def test_stream_complete_updates_token_counts(
        self,
        adapter_context: EventProcessorContext,
    ) -> None:
        """StreamCompleteEvent with usage should update token counts in StepFinishPart."""
        from pydantic_ai import RequestUsage

        adapter = OpenCodeEventAdapter(context=adapter_context)

        msg = Mock()
        msg.content = "Done"
        msg.usage = RequestUsage(input_tokens=100, output_tokens=50)
        msg.cost_info = None
        event = StreamCompleteEvent(message=msg)
        events = await _collect_events(adapter.convert_event(event))

        part_updated = [e for e in events if isinstance(e, PartUpdatedEvent)]
        step_finish = [e for e in part_updated if isinstance(e.properties.part, StepFinishPart)]
        assert len(step_finish) == 1
        assert step_finish[0].properties.part.tokens.input == 100
        assert step_finish[0].properties.part.tokens.output == 50


# =============================================================================
# RunStartedEvent conversion
# =============================================================================


class TestRunStartedEventConversion:
    """Tests for RunStartedEvent -> SessionStatusEvent (busy)."""

    @pytest.mark.asyncio
    async def test_run_started_does_not_yield_session_status(
        self,
        adapter_context: EventProcessorContext,
    ) -> None:
        """RunStartedEvent should NOT yield SessionStatusEvent from the adapter.

        Session status broadcasting is handled by
        OpenCodeSessionPoolIntegration._handle_event(), not the adapter.
        The adapter only handles content/streaming events.
        """
        adapter = OpenCodeEventAdapter(context=adapter_context)

        event = RunStartedEvent(session_id="test-session", run_id="run-001")
        events = await _collect_events(adapter.convert_event(event))

        status_events = [e for e in events if isinstance(e, SessionStatusEvent)]
        assert len(status_events) == 0, (
            "RunStartedEvent should not yield SessionStatusEvent from adapter; "
            "status broadcasting is handled by OpenCodeSessionPoolIntegration._handle_event()"
        )


# =============================================================================
# RunErrorEvent conversion
# =============================================================================


class TestRunErrorEventConversion:
    """Tests for RunErrorEvent -> SessionErrorEvent."""

    @pytest.mark.asyncio
    async def test_run_error_yields_session_error_event(
        self,
        adapter_context: EventProcessorContext,
    ) -> None:
        """RunErrorEvent should yield SessionErrorEvent with correct details."""
        adapter = OpenCodeEventAdapter(context=adapter_context)

        event = RunErrorEvent(
            message="Something went wrong",
            code="ERR_001",
            run_id="run-002",
        )
        events = await _collect_events(adapter.convert_event(event))

        error_events = [e for e in events if isinstance(e, SessionErrorEvent)]
        assert len(error_events) == 1
        assert error_events[0].properties.error is not None
        assert error_events[0].properties.error.name == "ERR_001"
        assert error_events[0].properties.error.data == {"message": "Something went wrong"}
        assert error_events[0].properties.session_id == "test-session"

    @pytest.mark.asyncio
    async def test_run_error_without_code_uses_default_name(
        self,
        adapter_context: EventProcessorContext,
    ) -> None:
        """RunErrorEvent without code should use default error name."""
        adapter = OpenCodeEventAdapter(context=adapter_context)

        event = RunErrorEvent(
            message="Generic failure",
            code=None,
            run_id="run-003",
        )
        events = await _collect_events(adapter.convert_event(event))

        error_events = [e for e in events if isinstance(e, SessionErrorEvent)]
        assert len(error_events) == 1
        assert error_events[0].properties.error is not None
        assert error_events[0].properties.error.name == "RunError"
        assert error_events[0].properties.error.data == {"message": "Generic failure"}


# =============================================================================
# MessageWithParts structure preservation
# =============================================================================


class TestMessageWithPartsPreservation:
    """Tests verifying MessageWithParts structure is preserved during conversion."""

    @pytest.mark.asyncio
    async def test_text_parts_appended_to_assistant_msg(
        self,
        adapter_context: EventProcessorContext,
    ) -> None:
        """Text parts should be appended to assistant_msg.parts."""
        adapter = OpenCodeEventAdapter(context=adapter_context)

        event = PartStartEvent.text(index=0, content="Hello")
        await _collect_events(adapter.convert_event(event))

        assert len(adapter_context.assistant_msg.parts) == 1
        assert isinstance(adapter_context.assistant_msg.parts[0], TextPart)
        assert adapter_context.assistant_msg.parts[0].text == "Hello"

    @pytest.mark.asyncio
    async def test_reasoning_parts_appended_to_assistant_msg(
        self,
        adapter_context: EventProcessorContext,
    ) -> None:
        """Reasoning parts should be appended to assistant_msg.parts."""
        adapter = OpenCodeEventAdapter(context=adapter_context)

        event = PartStartEvent.thinking(index=0, content="Thinking...")
        await _collect_events(adapter.convert_event(event))

        assert len(adapter_context.assistant_msg.parts) == 1
        assert isinstance(adapter_context.assistant_msg.parts[0], ReasoningPart)
        assert adapter_context.assistant_msg.parts[0].text == "Thinking..."

    @pytest.mark.asyncio
    async def test_tool_parts_appended_to_assistant_msg(
        self,
        adapter_context: EventProcessorContext,
    ) -> None:
        """Tool parts should be appended to assistant_msg.parts."""
        adapter = OpenCodeEventAdapter(context=adapter_context)

        event = ToolCallStartEvent(
            tool_call_id="call-005",
            tool_name="bash",
            title="Running: echo test",
            raw_input={"command": "echo test"},
        )
        await _collect_events(adapter.convert_event(event))

        assert len(adapter_context.assistant_msg.parts) == 1
        assert isinstance(adapter_context.assistant_msg.parts[0], ToolPart)
        assert adapter_context.assistant_msg.parts[0].tool == "bash"

    @pytest.mark.asyncio
    async def test_stream_complete_adds_step_finish_part(
        self,
        adapter_context: EventProcessorContext,
    ) -> None:
        """StreamCompleteEvent should add StepFinishPart to assistant_msg.parts."""
        adapter = OpenCodeEventAdapter(context=adapter_context)

        msg = Mock()
        msg.content = "Done"
        msg.usage = None
        msg.cost_info = None
        event = StreamCompleteEvent(message=msg)
        await _collect_events(adapter.convert_event(event))

        step_finish_parts = [
            p for p in adapter_context.assistant_msg.parts if isinstance(p, StepFinishPart)
        ]
        assert len(step_finish_parts) == 1

    @pytest.mark.asyncio
    async def test_multiple_parts_preserved_in_order(
        self,
        adapter_context: EventProcessorContext,
    ) -> None:
        """Multiple parts should be preserved in order."""
        adapter = OpenCodeEventAdapter(context=adapter_context)

        # Text part
        await _collect_events(adapter.convert_event(PartStartEvent.text(index=0, content="Text")))
        # Tool part
        await _collect_events(
            adapter.convert_event(
                ToolCallStartEvent(
                    tool_call_id="call-006",
                    tool_name="bash",
                    title="Run",
                    raw_input={},
                )
            )
        )
        # Complete tool
        await _collect_events(
            adapter.convert_event(
                ToolCallCompleteEvent(
                    tool_name="bash",
                    tool_call_id="call-006",
                    tool_input={},
                    tool_result="ok",
                    agent_name="agent",
                    message_id="msg",
                )
            )
        )
        # Stream complete
        msg = Mock()
        msg.content = "Done"
        msg.usage = None
        msg.cost_info = None
        await _collect_events(adapter.convert_event(StreamCompleteEvent(message=msg)))

        parts = adapter_context.assistant_msg.parts
        assert len(parts) == 3  # text, tool, step_finish
        assert isinstance(parts[0], TextPart)
        assert isinstance(parts[1], ToolPart)
        assert isinstance(parts[2], StepFinishPart)


# =============================================================================
# Stream conversion
# =============================================================================


class TestStreamConversion:
    """Tests for OpenCodeEventAdapter.convert_stream."""

    @pytest.mark.asyncio
    async def test_convert_stream_yields_all_events(
        self,
        adapter_context: EventProcessorContext,
    ) -> None:
        """convert_stream should yield OpenCode events for all AgentPool events."""

        async def _make_stream():
            yield PartStartEvent.text(index=0, content="Hello")
            yield PartStartEvent.text(index=1, content="World")

        adapter = OpenCodeEventAdapter(context=adapter_context)
        events = await _collect_events(adapter.convert_stream(_make_stream()))

        part_updated = [e for e in events if isinstance(e, PartUpdatedEvent)]
        assert len(part_updated) == 2


# =============================================================================
# Conversion completeness
# =============================================================================


class TestConversionCompleteness:
    """Tests verifying all specified AgentPool events are mapped to OpenCode events."""

    @pytest.mark.asyncio
    async def test_all_required_events_produce_output(
        self,
        adapter_context: EventProcessorContext,
    ) -> None:
        """Every required AgentPool event type should produce at least one OpenCode event.

        Note: RunStartedEvent is intentionally excluded — session status broadcasting
        is handled by OpenCodeSessionPoolIntegration._handle_event(), not the adapter.
        """
        adapter = OpenCodeEventAdapter(context=adapter_context)

        required_events = [
            PartStartEvent.text(index=0, content="test"),
            AgentPoolPartDeltaEvent.text(index=0, content="test"),
            ToolCallStartEvent(tool_call_id="t1", tool_name="test", title="Test"),
            ToolCallCompleteEvent(
                tool_name="test",
                tool_call_id="t1",
                tool_input={},
                tool_result="result",
                agent_name="agent",
                message_id="msg",
            ),
            RunErrorEvent(message="error", code="CODE"),
        ]

        for event in required_events:
            events = await _collect_events(adapter.convert_event(event))
            assert len(events) >= 1, f"Event {type(event).__name__} produced no output"

    @pytest.mark.asyncio
    async def test_part_end_produces_no_output(
        self,
        adapter_context: EventProcessorContext,
    ) -> None:
        """PartEndEvent is a completion signal and should produce no output."""
        adapter = OpenCodeEventAdapter(context=adapter_context)

        end_event = PartEndEvent(index=0, part=PydanticTextPart(content="test"))
        events = await _collect_events(adapter.convert_event(end_event))

        assert len(events) == 0, "PartEndEvent should not produce any OpenCode events"

    @pytest.mark.asyncio
    async def test_no_wolfharness_events_leak_through(
        self,
        adapter_context: EventProcessorContext,
    ) -> None:
        """Converted events should all be OpenCode Event types, never raw AgentPool events."""
        adapter = OpenCodeEventAdapter(context=adapter_context)

        event = RunStartedEvent(session_id="s", run_id="r")
        events = await _collect_events(adapter.convert_event(event))

        for e in events:
            # All events should have a 'type' attribute (OpenCode events do)
            assert hasattr(e, "type"), f"Event {type(e).__name__} lacks 'type' attribute"


# =============================================================================
# --- Merged from test_event_conversion.py ---
# =============================================================================

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def event_context():
    """Create an event context for testing event conversion."""
    from wolfharness_server.opencode_server.event_processor_context import (
        EventProcessorContext,
    )

    session_id = "test-session"
    assistant_msg_id = "msg-001"
    assistant_msg = MessageWithParts.assistant(
        message_id=assistant_msg_id,
        session_id=session_id,
        time=MessageTime(created=0),
        agent_name="test-agent",
        model_id="test-model",
        provider_id="wolfharness",
        path=MessagePath(cwd="/tmp", root="/tmp"),
        parent_id="msg-000",
    )
    state = Mock()
    state.messages = {}
    state.messages.setdefault(session_id, [])
    state.ensure_session = Mock()
    state.storage = Mock()
    state.storage.log_message = Mock()
    state.model_variants = {}

    return EventProcessorContext(
        session_id=session_id,
        assistant_msg_id=assistant_msg_id,
        assistant_msg=assistant_msg,
        state=state,
        working_dir="/tmp",
    )


# =============================================================================
# Helper to collect async generator results
# =============================================================================


async def _collect_events_v2(async_gen) -> list[Any]:
    """Collect all events from an async generator."""
    return [event async for event in async_gen]


# =============================================================================
# OpenCodeEventAdapter existence tests
# =============================================================================


class TestOpenCodeEventAdapterExists:
    """Verify the OpenCodeEventAdapter class exists and is importable."""

    @pytest.mark.asyncio
    async def test_adapter_class_importable(self) -> None:
        """The OpenCodeEventAdapter class should be importable."""
        assert OpenCodeEventAdapter is not None

    @pytest.mark.asyncio
    async def test_adapter_initialization(self, event_context) -> None:
        """Adapter should accept a context."""
        adapter = OpenCodeEventAdapter(context=event_context)
        assert adapter.context is event_context


# =============================================================================
# PartStartEvent conversion tests
# =============================================================================


class TestPartStartEventConversionV2:
    """Tests for PartStartEvent -> OpenCode PartUpdatedEvent."""

    @pytest.mark.asyncio
    async def test_text_part_start_creates_text_part(
        self,
        event_context,
    ) -> None:
        """PartStartEvent with TextPart should yield PartUpdatedEvent with TextPart."""
        from wolfharness.agents.events import PartStartEvent

        adapter = OpenCodeEventAdapter(context=event_context)

        event = PartStartEvent.text(index=0, content="Hello, world!")
        events = await _collect_events_v2(adapter.convert_event(event))

        part_updated = [e for e in events if isinstance(e, PartUpdatedEvent)]
        assert len(part_updated) == 1
        assert isinstance(part_updated[0].properties.part, TextPart)
        assert part_updated[0].properties.part.text == "Hello, world!"

    @pytest.mark.asyncio
    async def test_thinking_part_start_creates_reasoning_part(
        self,
        event_context,
    ) -> None:
        """PartStartEvent with ThinkingPart should yield PartUpdatedEvent with ReasoningPart."""
        from wolfharness.agents.events import PartStartEvent

        adapter = OpenCodeEventAdapter(context=event_context)

        event = PartStartEvent.thinking(index=0, content="Let me think...")
        events = await _collect_events_v2(adapter.convert_event(event))

        part_updated = [e for e in events if isinstance(e, PartUpdatedEvent)]
        assert len(part_updated) == 1
        from wolfharness_server.opencode_server.models.parts import ReasoningPart

        assert isinstance(part_updated[0].properties.part, ReasoningPart)
        assert part_updated[0].properties.part.text == "Let me think..."

    @pytest.mark.asyncio
    async def test_pydantic_text_part_start_creates_text_part(
        self,
        event_context,
    ) -> None:
        """PydanticAI PartStartEvent with TextPart should yield PartUpdatedEvent with TextPart."""
        adapter = OpenCodeEventAdapter(context=event_context)

        event = PydanticPartStartEvent(index=0, part=PydanticTextPart(content="Pydantic text"))
        events = await _collect_events_v2(adapter.convert_event(event))

        part_updated = [e for e in events if isinstance(e, PartUpdatedEvent)]
        assert len(part_updated) == 1
        assert isinstance(part_updated[0].properties.part, TextPart)
        assert part_updated[0].properties.part.text == "Pydantic text"


# =============================================================================
# PartDeltaEvent conversion tests
# =============================================================================


class TestPartDeltaEventConversionV2:
    """Tests for PartDeltaEvent -> OpenCode PartDeltaEvent."""

    @pytest.mark.asyncio
    async def test_text_delta_yields_part_delta_event(
        self,
        event_context,
    ) -> None:
        """Text delta should yield PartDeltaEvent."""
        from wolfharness.agents.events import (
            PartDeltaEvent as AgentPoolPartDeltaEvent,
            PartStartEvent,
        )

        adapter = OpenCodeEventAdapter(context=event_context)

        # First establish a text part
        start_event = PartStartEvent.text(index=0, content="Hello")
        await _collect_events_v2(adapter.convert_event(start_event))

        delta_event = AgentPoolPartDeltaEvent.text(index=0, content=", world!")
        events = await _collect_events_v2(adapter.convert_event(delta_event))

        delta_events = [e for e in events if isinstance(e, PartDeltaEvent)]
        assert len(delta_events) == 1
        assert delta_events[0].properties.delta == ", world!"
        assert delta_events[0].properties.field == "text"

    @pytest.mark.asyncio
    async def test_thinking_delta_yields_part_delta_event(
        self,
        event_context,
    ) -> None:
        """Thinking delta should yield PartDeltaEvent."""
        from wolfharness.agents.events import (
            PartDeltaEvent as AgentPoolPartDeltaEvent,
            PartStartEvent,
        )

        adapter = OpenCodeEventAdapter(context=event_context)

        # First establish a reasoning part
        start_event = PartStartEvent.thinking(index=0, content="Thinking")
        await _collect_events_v2(adapter.convert_event(start_event))

        delta_event = AgentPoolPartDeltaEvent.thinking(index=0, content=" more...")
        events = await _collect_events_v2(adapter.convert_event(delta_event))

        delta_events = [e for e in events if isinstance(e, PartDeltaEvent)]
        assert len(delta_events) == 1
        assert delta_events[0].properties.delta == " more..."

    @pytest.mark.asyncio
    async def test_pydantic_text_delta_yields_part_delta_event(
        self,
        event_context,
    ) -> None:
        """PydanticAI TextPartDelta should yield PartDeltaEvent."""
        adapter = OpenCodeEventAdapter(context=event_context)

        # First establish a text part
        start_event = PydanticPartStartEvent(index=0, part=PydanticTextPart(content="Base"))
        await _collect_events_v2(adapter.convert_event(start_event))

        delta_event = PydanticPartDeltaEvent(
            index=0, delta=TextPartDelta(content_delta=" extended")
        )
        events = await _collect_events_v2(adapter.convert_event(delta_event))

        delta_events = [e for e in events if isinstance(e, PartDeltaEvent)]
        assert len(delta_events) == 1
        assert delta_events[0].properties.delta == " extended"


# =============================================================================
# ToolCallStartEvent conversion tests
# =============================================================================


class TestToolCallStartEventConversionV2:
    """Tests for ToolCallStartEvent -> OpenCode PartUpdatedEvent (ToolPart)."""

    @pytest.mark.asyncio
    async def test_tool_call_start_creates_running_tool_part(
        self,
        event_context,
    ) -> None:
        """ToolCallStartEvent should yield PartUpdatedEvent with ToolPart in running state."""
        adapter = OpenCodeEventAdapter(context=event_context)

        event = ToolCallStartEvent(
            tool_call_id="call-001",
            tool_name="bash",
            title="Running: ls -la",
            raw_input={"command": "ls -la"},
        )
        events = await _collect_events_v2(adapter.convert_event(event))

        part_updated = [e for e in events if isinstance(e, PartUpdatedEvent)]
        assert len(part_updated) == 1
        tool_part = part_updated[0].properties.part
        assert isinstance(tool_part, ToolPart)
        assert tool_part.tool == "bash"
        assert isinstance(tool_part.state, ToolStateRunning)
        assert tool_part.state.input == {"command": "ls -la"}

    @pytest.mark.asyncio
    async def test_tool_call_start_with_empty_input(
        self,
        event_context,
    ) -> None:
        """ToolCallStartEvent with no input should still create ToolPart."""
        adapter = OpenCodeEventAdapter(context=event_context)

        event = ToolCallStartEvent(
            tool_call_id="call-002",
            tool_name="read",
            title="Reading file",
        )
        events = await _collect_events_v2(adapter.convert_event(event))

        part_updated = [e for e in events if isinstance(e, PartUpdatedEvent)]
        assert len(part_updated) == 1
        tool_part = part_updated[0].properties.part
        assert isinstance(tool_part, ToolPart)
        assert tool_part.tool == "read"


# =============================================================================
# ToolCallCompleteEvent conversion tests
# =============================================================================


class TestToolCallCompleteEventConversionV2:
    """Tests for ToolCallCompleteEvent -> OpenCode PartUpdatedEvent (completed/error)."""

    @pytest.mark.asyncio
    async def test_tool_call_complete_creates_completed_tool_part(
        self,
        event_context,
    ) -> None:
        """ToolCallCompleteEvent should yield PartUpdatedEvent with ToolPart in completed state."""
        adapter = OpenCodeEventAdapter(context=event_context)

        # First create the tool part
        start_event = ToolCallStartEvent(
            tool_call_id="call-003",
            tool_name="bash",
            title="Running: echo hello",
            raw_input={"command": "echo hello"},
        )
        await _collect_events_v2(adapter.convert_event(start_event))

        complete_event = ToolCallCompleteEvent(
            tool_name="bash",
            tool_call_id="call-003",
            tool_input={"command": "echo hello"},
            tool_result="hello",
            agent_name="test-agent",
            message_id="msg-001",
        )
        events = await _collect_events_v2(adapter.convert_event(complete_event))

        part_updated = [e for e in events if isinstance(e, PartUpdatedEvent)]
        assert len(part_updated) == 1
        tool_part = part_updated[0].properties.part
        assert isinstance(tool_part, ToolPart)
        assert isinstance(tool_part.state, ToolStateCompleted)
        assert tool_part.state.output == "hello"

    @pytest.mark.asyncio
    async def test_tool_call_complete_with_error_creates_error_tool_part(
        self,
        event_context,
    ) -> None:
        """ToolCallCompleteEvent with error result should yield ToolPart in error state."""
        adapter = OpenCodeEventAdapter(context=event_context)

        # First create the tool part
        start_event = ToolCallStartEvent(
            tool_call_id="call-004",
            tool_name="bash",
            title="Running: false",
            raw_input={"command": "false"},
        )
        await _collect_events_v2(adapter.convert_event(start_event))

        complete_event = ToolCallCompleteEvent(
            tool_name="bash",
            tool_call_id="call-004",
            tool_input={"command": "false"},
            tool_result={"error": "Command failed with exit code 1"},
            agent_name="test-agent",
            message_id="msg-001",
        )
        events = await _collect_events_v2(adapter.convert_event(complete_event))

        part_updated = [e for e in events if isinstance(e, PartUpdatedEvent)]
        assert len(part_updated) == 1
        tool_part = part_updated[0].properties.part
        assert isinstance(tool_part, ToolPart)
        assert isinstance(tool_part.state, ToolStateError)
        assert tool_part.state.error == "Command failed with exit code 1"


# =============================================================================
# StreamCompleteEvent conversion tests
# =============================================================================


class TestStreamCompleteEventConversionV2:
    """Tests for StreamCompleteEvent -> StepFinishPart + SessionIdleEvent."""

    @pytest.mark.asyncio
    async def test_stream_complete_yields_step_finish_part(
        self,
        event_context,
    ) -> None:
        """StreamCompleteEvent should yield PartUpdatedEvent with StepFinishPart."""
        adapter = OpenCodeEventAdapter(context=event_context)

        msg = Mock()
        msg.content = "Done"
        msg.usage = None
        msg.cost_info = None
        event = StreamCompleteEvent(message=msg)
        events = await _collect_events_v2(adapter.convert_event(event))

        part_updated = [e for e in events if isinstance(e, PartUpdatedEvent)]
        step_finish = [e for e in part_updated if isinstance(e.properties.part, StepFinishPart)]
        assert len(step_finish) == 1

    @pytest.mark.asyncio
    async def test_stream_complete_updates_token_counts(
        self,
        event_context,
    ) -> None:
        """StreamCompleteEvent with usage should update token counts in StepFinishPart."""
        from pydantic_ai import RequestUsage

        adapter = OpenCodeEventAdapter(context=event_context)

        msg = Mock()
        msg.content = "Done"
        msg.usage = RequestUsage(input_tokens=100, output_tokens=50)
        msg.cost_info = None
        event = StreamCompleteEvent(message=msg)
        events = await _collect_events_v2(adapter.convert_event(event))

        part_updated = [e for e in events if isinstance(e, PartUpdatedEvent)]
        step_finish = [e for e in part_updated if isinstance(e.properties.part, StepFinishPart)]
        assert len(step_finish) == 1
        assert step_finish[0].properties.part.tokens.input == 100
        assert step_finish[0].properties.part.tokens.output == 50


class TestRunStartedEventConversionV2:
    """Tests for RunStartedEvent -> SessionStatusEvent (busy)."""

    @pytest.mark.asyncio
    async def test_run_started_does_not_yield_session_status(
        self,
        event_context,
    ) -> None:
        """RunStartedEvent should NOT yield SessionStatusEvent from the adapter.

        Session status broadcasting is now handled by
        OpenCodeSessionPoolIntegration._handle_event(), not the adapter.
        """
        adapter = OpenCodeEventAdapter(context=event_context)

        event = RunStartedEvent(session_id="test-session", run_id="run-001")
        events = await _collect_events_v2(adapter.convert_event(event))

        status_events = [e for e in events if isinstance(e, SessionStatusEvent)]
        assert len(status_events) == 0, (
            "RunStartedEvent should not yield SessionStatusEvent from adapter; "
            "status broadcasting is handled by OpenCodeSessionPoolIntegration._handle_event()"
        )


# =============================================================================
# RunErrorEvent conversion tests
# =============================================================================


class TestRunErrorEventConversionV2:
    """Tests for RunErrorEvent -> SessionErrorEvent."""

    @pytest.mark.asyncio
    async def test_run_error_yields_session_error_event(
        self,
        event_context,
    ) -> None:
        """RunErrorEvent should yield SessionErrorEvent."""
        adapter = OpenCodeEventAdapter(context=event_context)

        event = RunErrorEvent(
            message="Something went wrong",
            code="ERR_001",
            run_id="run-002",
        )
        events = await _collect_events_v2(adapter.convert_event(event))

        error_events = [e for e in events if isinstance(e, SessionErrorEvent)]
        assert len(error_events) == 1
        assert error_events[0].properties.error is not None
        assert error_events[0].properties.error.name == "ERR_001"


# =============================================================================
# RunFailedEvent conversion tests
# =============================================================================


class TestRunFailedEventConversion:
    """Tests for RunFailedEvent -> SessionErrorEvent + SessionStatusEvent (idle)."""


class TestToolCallProgressEventConversion:
    """Tests for ToolCallProgressEvent -> PartUpdatedEvent (ToolPart updates)."""

    @pytest.mark.asyncio
    async def test_tool_progress_updates_existing_tool_part(
        self,
        event_context,
    ) -> None:
        """ToolCallProgressEvent should update existing ToolPart."""
        adapter = OpenCodeEventAdapter(context=event_context)

        # First create the tool part
        start_event = ToolCallStartEvent(
            tool_call_id="call-005",
            tool_name="bash",
            title="Running: long command",
            raw_input={"command": "long command"},
        )
        await _collect_events_v2(adapter.convert_event(start_event))

        progress_event = ToolCallProgressEvent(
            tool_call_id="call-005",
            title="Still running...",
            items=[TextContentItem(text="Output line 1")],
        )
        events = await _collect_events_v2(adapter.convert_event(progress_event))

        part_updated = [e for e in events if isinstance(e, PartUpdatedEvent)]
        assert len(part_updated) == 1
        tool_part = part_updated[0].properties.part
        assert isinstance(tool_part, ToolPart)
        assert isinstance(tool_part.state, ToolStateRunning)

    @pytest.mark.asyncio
    async def test_tool_progress_updates_input_from_event_when_existing_empty(
        self,
        event_context,
    ) -> None:
        """ToolCallProgressEvent with tool_input should populate empty input.

        When a model streams tool call arguments, the initial ToolCallStartEvent
        may arrive with empty raw_input (args not yet assembled). The
        EventMapper then emits a ToolCallProgressEvent with the complete
        tool_input once args are available. The event processor must update
        the tool part's input from this progress event, otherwise parameters
        are lost and never displayed in the TUI.
        """
        adapter = OpenCodeEventAdapter(context=event_context)

        # Step 1: ToolCallStartEvent with empty raw_input (streaming args)
        start_event = ToolCallStartEvent(
            tool_call_id="call-stream-001",
            tool_name="bash",
            title="Executing: bash",
            raw_input={},
        )
        await _collect_events_v2(adapter.convert_event(start_event))

        # Step 2: ToolCallProgressEvent carries the complete tool_input
        progress_event = ToolCallProgressEvent(
            tool_call_id="call-stream-001",
            title="Running: ls -la",
            items=[],
            tool_name="bash",
            tool_input={"command": "ls -la"},
        )
        events = await _collect_events_v2(adapter.convert_event(progress_event))

        part_updated = [e for e in events if isinstance(e, PartUpdatedEvent)]
        assert len(part_updated) == 1
        tool_part = part_updated[0].properties.part
        assert isinstance(tool_part, ToolPart)
        assert isinstance(tool_part.state, ToolStateRunning)
        # The input must be populated from the progress event's tool_input
        assert tool_part.state.input.get("command") == "ls -la"

    @pytest.mark.asyncio
    async def test_tool_progress_creates_new_tool_part_if_not_exists(
        self,
        event_context,
    ) -> None:
        """ToolCallProgressEvent without prior start should create new ToolPart."""
        adapter = OpenCodeEventAdapter(context=event_context)

        progress_event = ToolCallProgressEvent(
            tool_call_id="call-006",
            title="File operation",
            items=[TextContentItem(text="Reading file...")],
            tool_name="read",
        )
        events = await _collect_events_v2(adapter.convert_event(progress_event))

        part_updated = [e for e in events if isinstance(e, PartUpdatedEvent)]
        assert len(part_updated) == 1
        tool_part = part_updated[0].properties.part
        assert isinstance(tool_part, ToolPart)
        assert tool_part.tool == "read"


# =============================================================================
# Stream conversion tests
# =============================================================================


class TestStreamConversionV2:
    """Tests for OpenCodeEventAdapter.convert_stream."""

    @pytest.mark.asyncio
    async def test_convert_stream_yields_all_events(self, event_context) -> None:
        """convert_stream should yield OpenCode events for all AgentPool events."""

        async def _make_stream():
            from wolfharness.agents.events import PartStartEvent

            yield PartStartEvent.text(index=0, content="Hello")
            yield PartStartEvent.text(index=1, content="World")

        adapter = OpenCodeEventAdapter(context=event_context)
        events = await _collect_events_v2(adapter.convert_stream(_make_stream()))

        part_updated = [e for e in events if isinstance(e, PartUpdatedEvent)]
        assert len(part_updated) == 2


# =============================================================================
# Conversion completeness tests
# =============================================================================


class TestConversionCompletenessV2:
    """Tests verifying all specified AgentPool events are mapped to OpenCode events."""

    @pytest.mark.asyncio
    async def test_all_required_events_have_handlers(
        self,
        event_context,
    ) -> None:
        """Every required AgentPool event type should have a conversion handler."""
        from wolfharness.agents.events import (
            PartDeltaEvent,
            PartStartEvent,
            ToolCallCompleteEvent,
            ToolCallStartEvent,
        )

        adapter = OpenCodeEventAdapter(context=event_context)

        # RunStartedEvent intentionally excluded — status broadcasting is now
        # handled by OpenCodeSessionPoolIntegration._handle_event(), not the adapter.
        required_events = [
            PartStartEvent.text(index=0, content="test"),
            PartDeltaEvent.text(index=0, content="test"),
            ToolCallStartEvent(tool_call_id="t1", tool_name="test", title="Test"),
            ToolCallCompleteEvent(
                tool_name="test",
                tool_call_id="t1",
                tool_input={},
                tool_result="result",
                agent_name="agent",
                message_id="msg",
            ),
        ]

        for event in required_events:
            events = await _collect_events_v2(adapter.convert_event(event))
            # Each event should produce at least one OpenCode event
            assert len(events) >= 1, f"Event {type(event).__name__} produced no output"

    @pytest.mark.asyncio
    async def test_no_wolfharness_events_leak_through(
        self,
        event_context,
    ) -> None:
        """Converted events should all be OpenCode Event types, never raw AgentPool events."""
        from wolfharness.agents.events import RunStartedEvent

        adapter = OpenCodeEventAdapter(context=event_context)

        event = RunStartedEvent(session_id="s", run_id="r")
        events = await _collect_events_v2(adapter.convert_event(event))

        for e in events:
            # All events should have a 'type' attribute (OpenCode events do)
            assert hasattr(e, "type"), f"Event {type(e).__name__} lacks 'type' attribute"
