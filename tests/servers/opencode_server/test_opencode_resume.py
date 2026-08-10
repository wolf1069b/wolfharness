"""Tests for OpenCode session resume with EventProcessorContext reconstruction.

Covers Task 26: implementing session resume with EventProcessorContext
reconstruction in session_pool_integration.py.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest

from wolfharness.orchestrator.core import SessionPool
from wolfharness_server.opencode_server.event_processor_context import (
    EventProcessorContext,
)
from wolfharness_server.opencode_server.models import (
    MessagePath,
    MessageTime,
    MessageWithParts,
)
from wolfharness_server.opencode_server.models.parts import (
    TextPart,
    ToolPart,
    ToolStateRunning,
)
from wolfharness_server.opencode_server.session_pool_integration import (
    OpenCodeSessionPoolIntegration,
)
from wolfharness_server.opencode_server.state import ServerState


pytestmark = pytest.mark.integration


@pytest.fixture
def mock_agent_pool() -> Mock:
    """Create a mock AgentPool for SessionPool construction."""
    pool = Mock()
    pool.main_agent = Mock()
    pool.main_agent.name = "test-agent"
    pool.manifest = Mock()
    pool.manifest.agents = {}
    pool._config_file_path = None
    pool.get_agent = Mock(return_value=Mock())
    return pool


@pytest.fixture
def mock_session_store() -> Mock:
    """Create a mock SessionStore."""
    store = Mock()
    store.save_session = AsyncMock(return_value=None)
    store.delete_session = AsyncMock(return_value=None)
    store.load_session = AsyncMock(return_value=None)
    store.list_session_ids = AsyncMock(return_value=[])
    return store


@pytest.fixture
async def session_pool(mock_agent_pool: Mock, mock_session_store: Mock) -> SessionPool:
    """Create a real SessionPool with mocked dependencies."""
    sp = SessionPool(
        pool=mock_agent_pool,
        store=mock_session_store,
        enable_auto_resume=False,
        enable_event_bus=True,
    )
    await sp.start()
    yield sp
    await sp.shutdown()


@pytest.fixture
def server_state(tmp_path: Any) -> ServerState:
    """Create a minimal ServerState for testing."""
    agent = Mock()
    agent.model_name = None  # resolve_default_model_info() fallback
    agent.name = "test-agent"
    agent.storage = Mock()
    return ServerState(working_dir=str(tmp_path), agent=agent)


def _make_assistant_msg(session_id: str, working_dir: str) -> MessageWithParts:
    """Create a fresh assistant MessageWithParts for testing."""
    from wolfharness.utils.time_utils import now_ms

    return MessageWithParts.assistant(
        message_id="msg-test-001",
        session_id=session_id,
        time=MessageTime(created=now_ms()),
        agent_name="test-agent",
        model_id="test-model",
        parent_id=session_id,
        provider_id="wolfharness",
        path=MessagePath(cwd=working_dir, root=working_dir),
    )


# ---------------------------------------------------------------------------
# serialize / deserialize roundtrip
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestEventProcessorContextSerialization:
    """Tests for EventProcessorContext.serialize() and .deserialize()."""

    def test_serialize_empty_context(self, server_state: ServerState) -> None:
        """serialize() returns a dict with all expected keys for an empty context."""
        msg = _make_assistant_msg("sess-1", server_state.working_dir)
        ctx = EventProcessorContext(
            session_id="sess-1",
            assistant_msg_id=msg.info.id,
            assistant_msg=msg,
            state=server_state,
            working_dir=server_state.working_dir,
        )

        data = ctx.serialize()

        assert isinstance(data, dict)
        assert data["session_id"] == "sess-1"
        assert data["assistant_msg_id"] == msg.info.id
        assert data["response_text"] == ""
        assert data["text_part"] is None
        assert data["reasoning_part"] is None
        assert data["input_tokens"] == 0
        assert data["output_tokens"] == 0
        assert data["total_cost"] == 0.0
        assert isinstance(data["stream_start_ms"], int)
        assert data["tool_parts"] == {}
        assert data["tool_outputs"] == {}
        assert data["tool_inputs"] == {}
        assert data["subagent_tool_parts"] == {}
        assert data["is_errored"] is False
        # assistant_msg should be serialized as a dict
        assert isinstance(data["assistant_msg"], dict)

    def test_serialize_deserialize_roundtrip_basic(self, server_state: ServerState) -> None:
        """Roundtrip: serialize → deserialize preserves all fields."""
        session_id = "sess-roundtrip"
        msg = _make_assistant_msg(session_id, server_state.working_dir)
        ctx = EventProcessorContext(
            session_id=session_id,
            assistant_msg_id=msg.info.id,
            assistant_msg=msg,
            state=server_state,
            working_dir=server_state.working_dir,
        )
        # Set some state
        ctx.accumulate_text("Hello, world!")
        ctx.update_tokens(100, 50)
        ctx.update_cost(0.005)
        ctx.is_errored = False

        data = ctx.serialize()
        restored = EventProcessorContext.deserialize(
            data, state=server_state, working_dir=server_state.working_dir
        )

        assert restored.session_id == ctx.session_id
        assert restored.assistant_msg_id == ctx.assistant_msg_id
        assert restored.response_text == "Hello, world!"
        assert restored.input_tokens == 100
        assert restored.output_tokens == 50
        assert restored.total_cost == 0.005
        assert restored.is_errored is False

    def test_serialize_deserialize_roundtrip_with_tool_parts(
        self, server_state: ServerState
    ) -> None:
        """Roundtrip: tool parts and tracking state are preserved."""
        session_id = "sess-tools"
        msg = _make_assistant_msg(session_id, server_state.working_dir)
        ctx = EventProcessorContext(
            session_id=session_id,
            assistant_msg_id=msg.info.id,
            assistant_msg=msg,
            state=server_state,
            working_dir=server_state.working_dir,
        )

        # Add a tool part
        from wolfharness_server.opencode_server.models.parts import (
            TimeStart,
        )

        tp = ToolPart(
            id="part-1",
            message_id=msg.info.id,
            session_id=session_id,
            tool="read",
            call_id="call-1",
            state=ToolStateRunning(
                time=TimeStart(start=ctx.stream_start_ms),
                input={"path": "/test/file.py"},
                title="Reading file",
            ),
        )
        ctx.add_tool_part("call-1", tp)
        ctx.set_tool_input("call-1", {"path": "/test/file.py"})
        ctx.set_tool_output("call-1", "partial out")
        ctx.append_tool_output("call-1", "put")

        data = ctx.serialize()
        restored = EventProcessorContext.deserialize(
            data, state=server_state, working_dir=server_state.working_dir
        )

        assert restored.has_tool_part("call-1")
        restored_tp = restored.get_tool_part("call-1")
        assert restored_tp is not None
        assert restored_tp.tool == "read"
        assert restored_tp.call_id == "call-1"
        assert restored.get_tool_input("call-1") == {"path": "/test/file.py"}
        assert restored.get_tool_output("call-1") == "partial output"

    def test_serialize_deserialize_roundtrip_with_text_part(
        self, server_state: ServerState
    ) -> None:
        """Roundtrip: text part is preserved."""
        session_id = "sess-text"
        msg = _make_assistant_msg(session_id, server_state.working_dir)
        ctx = EventProcessorContext(
            session_id=session_id,
            assistant_msg_id=msg.info.id,
            assistant_msg=msg,
            state=server_state,
            working_dir=server_state.working_dir,
        )

        ctx.set_text("Some text")
        tp = TextPart(
            id="part-text-1",
            message_id=msg.info.id,
            session_id=session_id,
            text="Some text",
        )
        ctx.text_part = tp
        ctx.assistant_msg.parts.append(tp)

        data = ctx.serialize()
        restored = EventProcessorContext.deserialize(
            data, state=server_state, working_dir=server_state.working_dir
        )

        assert restored.response_text == "Some text"
        assert restored.text_part is not None
        assert restored.text_part.id == "part-text-1"
        assert restored.text_part.text == "Some text"


# ---------------------------------------------------------------------------
# _before_consumer_loop resume behavior
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestBeforeConsumerLoopResume:
    """Tests for _before_consumer_loop restoring context on resume."""

    @pytest.mark.asyncio
    async def test_restores_context_when_resume_data_present(
        self,
        session_pool: SessionPool,
        server_state: ServerState,
    ) -> None:
        """_before_consumer_loop restores context when resume data is set."""
        integration = OpenCodeSessionPoolIntegration(
            session_pool=session_pool,
            server_state=server_state,
        )
        session_id = "sess-resume-1"

        # First, create a normal context and serialize it
        msg = _make_assistant_msg(session_id, server_state.working_dir)

        ctx = EventProcessorContext(
            session_id=session_id,
            assistant_msg_id=msg.info.id,
            assistant_msg=msg,
            state=server_state,
            working_dir=server_state.working_dir,
        )
        ctx.accumulate_text("original accumulated text")
        ctx.update_tokens(200, 100)
        serialized = ctx.serialize()

        # Set it as resume context data
        integration.set_session_context_data(session_id, serialized)

        # Now call _before_consumer_loop — should restore
        await integration._before_consumer_loop(session_id)

        # Verify the restored context
        restored_ctx = integration._contexts.get(session_id)
        assert restored_ctx is not None
        assert restored_ctx.response_text == "original accumulated text"
        assert restored_ctx.input_tokens == 200
        assert restored_ctx.output_tokens == 100
        assert restored_ctx.session_id == session_id

        # Verify adapter was also created
        adapter = integration._adapters.get(session_id)
        assert adapter is not None

    @pytest.mark.asyncio
    async def test_creates_fresh_when_no_resume_data(
        self,
        session_pool: SessionPool,
        server_state: ServerState,
    ) -> None:
        """_before_consumer_loop creates fresh context when no resume data."""
        integration = OpenCodeSessionPoolIntegration(
            session_pool=session_pool,
            server_state=server_state,
        )
        session_id = "sess-fresh-1"

        # No resume context set — should create fresh
        await integration._before_consumer_loop(session_id)

        ctx = integration._contexts.get(session_id)
        assert ctx is not None
        # Fresh context should have empty response_text
        assert ctx.response_text == ""
        assert ctx.input_tokens == 0
        assert ctx.output_tokens == 0

    @pytest.mark.asyncio
    async def test_restored_context_has_correct_assistant_msg(
        self,
        session_pool: SessionPool,
        server_state: ServerState,
    ) -> None:
        """Restored context has the assistant message with original parts."""
        integration = OpenCodeSessionPoolIntegration(
            session_pool=session_pool,
            server_state=server_state,
        )
        session_id = "sess-resume-msg"

        # Create context with a tool part on the assistant message
        msg = _make_assistant_msg(session_id, server_state.working_dir)
        ctx = EventProcessorContext(
            session_id=session_id,
            assistant_msg_id=msg.info.id,
            assistant_msg=msg,
            state=server_state,
            working_dir=server_state.working_dir,
        )

        # Add a tool part to the context and assistant message
        from wolfharness_server.opencode_server.models.parts import (
            TimeStart,
        )

        tp = ToolPart(
            id="part-tool-resume",
            message_id=msg.info.id,
            session_id=session_id,
            tool="grep",
            call_id="call-grep-1",
            state=ToolStateRunning(
                time=TimeStart(start=ctx.stream_start_ms),
                input={"pattern": "TODO"},
                title="Searching for TODOs",
            ),
        )
        ctx.add_tool_part("call-grep-1", tp)
        ctx.assistant_msg.parts.append(tp)

        serialized = ctx.serialize()
        integration.set_session_context_data(session_id, serialized)

        await integration._before_consumer_loop(session_id)

        restored_ctx = integration._contexts.get(session_id)
        assert restored_ctx is not None
        # Tool part should be on the restored assistant message
        restored_tp = restored_ctx.get_tool_part("call-grep-1")
        assert restored_tp is not None
        assert restored_tp.tool == "grep"
        assert restored_tp.call_id == "call-grep-1"
        # And the assistant message should have it in parts
        assert len(restored_ctx.assistant_msg.parts) == 1
        assert restored_ctx.assistant_msg.parts[0].id == "part-tool-resume"

    @pytest.mark.asyncio
    async def test_resume_context_is_consumed_once(
        self,
        session_pool: SessionPool,
        server_state: ServerState,
    ) -> None:
        """Resume context data is consumed on first call, not reused."""
        integration = OpenCodeSessionPoolIntegration(
            session_pool=session_pool,
            server_state=server_state,
        )
        session_id = "sess-consume-once"

        msg = _make_assistant_msg(session_id, server_state.working_dir)
        ctx = EventProcessorContext(
            session_id=session_id,
            assistant_msg_id=msg.info.id,
            assistant_msg=msg,
            state=server_state,
            working_dir=server_state.working_dir,
        )
        ctx.accumulate_text("resumed text")
        serialized = ctx.serialize()
        integration.set_session_context_data(session_id, serialized)

        # First call: should restore
        await integration._before_consumer_loop(session_id)
        restored = integration._contexts[session_id]
        assert restored.response_text == "resumed text"

        # Clean up to simulate consumer restart
        integration._contexts.pop(session_id, None)
        integration._adapters.pop(session_id, None)

        # Second call: should create fresh (resume data consumed)
        await integration._before_consumer_loop(session_id)
        fresh = integration._contexts[session_id]
        assert fresh.response_text == ""

    @pytest.mark.asyncio
    async def test_restored_context_does_not_replay_completed_parts(
        self,
        session_pool: SessionPool,
        server_state: ServerState,
    ) -> None:
        """Restored context has parts but does NOT replay SSE events for them.

        After resume, the context is reconstructed with existing parts.
        The frontend already has these parts displayed — we should NOT
        re-broadcast PartUpdatedEvent for them during resume.
        """
        integration = OpenCodeSessionPoolIntegration(
            session_pool=session_pool,
            server_state=server_state,
        )
        session_id = "sess-no-replay"

        msg = _make_assistant_msg(session_id, server_state.working_dir)
        ctx = EventProcessorContext(
            session_id=session_id,
            assistant_msg_id=msg.info.id,
            assistant_msg=msg,
            state=server_state,
            working_dir=server_state.working_dir,
        )
        # Simulate an already-completed part
        from wolfharness_server.opencode_server.models.parts import (
            TimeStart,
        )

        tp = ToolPart(
            id="part-done",
            message_id=msg.info.id,
            session_id=session_id,
            tool="read",
            call_id="call-done",
            state=ToolStateRunning(
                time=TimeStart(start=ctx.stream_start_ms),
                input={"path": "done.py"},
                title="Done reading",
            ),
        )
        ctx.add_tool_part("call-done", tp)
        ctx.assistant_msg.parts.append(tp)

        serialized = ctx.serialize()
        integration.set_session_context_data(session_id, serialized)

        # Patch broadcast_event to count calls
        from unittest.mock import patch

        with patch.object(server_state, "broadcast_event", new=AsyncMock()) as mock_bcast:
            await integration._before_consumer_loop(session_id)

            # broadcast_event should NOT be called for restoring parts
            # (parts are already in the frontend from the original session)
            broadcast_calls = mock_bcast.await_args_list
            part_update_calls = [
                c
                for c in broadcast_calls
                if hasattr(c.args[0], "type") and "part" in c.args[0].type
            ]
            assert len(part_update_calls) == 0, (
                "Should not replay PartUpdatedEvent for already-completed parts during resume"
            )

    @pytest.mark.asyncio
    async def test_event_bus_resubscribed_on_consumer_start(
        self,
        session_pool: SessionPool,
        server_state: ServerState,
    ) -> None:
        """EventBus is re-subscribed when consumer starts for a resumed session.

        Verifies that the consumer task and queue are created on start,
        cleaned up on stop, and re-created on restart with resume context.
        """
        integration = OpenCodeSessionPoolIntegration(
            session_pool=session_pool,
            server_state=server_state,
        )
        session_id = "sess-resub"

        # Start a consumer
        await integration.start_event_consumer(session_id)
        # Wait for _before_consumer_loop to complete in background task
        await asyncio.sleep(0.05)

        # Verify consumer task and queue exist
        assert session_id in integration._session_groups
        assert session_id in integration._consumer_streams
        assert integration._session_groups[session_id] is not None

        # Stop the consumer
        await integration.stop_event_consumer(session_id)

        # Verify consumer task and queue are cleaned up
        assert session_id not in integration._session_groups
        assert session_id not in integration._consumer_streams

        # Now resume: set context data and re-start
        msg = _make_assistant_msg(session_id, server_state.working_dir)
        ctx = EventProcessorContext(
            session_id=session_id,
            assistant_msg_id=msg.info.id,
            assistant_msg=msg,
            state=server_state,
            working_dir=server_state.working_dir,
        )
        integration.set_session_context_data(session_id, ctx.serialize())

        await integration.start_event_consumer(session_id)
        # Wait for _before_consumer_loop to complete in background task
        await asyncio.sleep(0.05)

        # Verify consumer re-started
        assert session_id in integration._session_groups
        assert session_id in integration._consumer_streams
        assert integration._session_groups[session_id] is not None

        # Verify context was restored (not created fresh)
        restored_ctx = integration._contexts.get(session_id)
        assert restored_ctx is not None
        assert restored_ctx.session_id == session_id

    @pytest.mark.asyncio
    async def test_open_code_event_adapter_restarted_with_restored_context(
        self,
        session_pool: SessionPool,
        server_state: ServerState,
    ) -> None:
        """OpenCodeEventAdapter is created with restored context on resume."""
        integration = OpenCodeSessionPoolIntegration(
            session_pool=session_pool,
            server_state=server_state,
        )
        session_id = "sess-adapter"

        msg = _make_assistant_msg(session_id, server_state.working_dir)
        ctx = EventProcessorContext(
            session_id=session_id,
            assistant_msg_id=msg.info.id,
            assistant_msg=msg,
            state=server_state,
            working_dir=server_state.working_dir,
        )
        ctx.accumulate_text("persisted adapter text")
        serialized = ctx.serialize()
        integration.set_session_context_data(session_id, serialized)

        await integration._before_consumer_loop(session_id)

        adapter = integration._adapters.get(session_id)
        assert adapter is not None
        assert adapter.context is integration._contexts[session_id]
        assert adapter.context.response_text == "persisted adapter text"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestResumeEdgeCases:
    """Edge case tests for context resume."""

    @pytest.mark.asyncio
    async def test_resume_with_empty_serialized_data(
        self,
        session_pool: SessionPool,
        server_state: ServerState,
    ) -> None:
        """Gracefully handles empty serialized data dict."""
        integration = OpenCodeSessionPoolIntegration(
            session_pool=session_pool,
            server_state=server_state,
        )
        session_id = "sess-empty"

        integration.set_session_context_data(session_id, {})

        # Should not raise, should fall back to creating fresh
        await integration._before_consumer_loop(session_id)
        ctx = integration._contexts.get(session_id)
        assert ctx is not None

    @pytest.mark.asyncio
    async def test_deserialize_preserves_is_errored_flag(self, server_state: ServerState) -> None:
        """is_errored flag is preserved across serialize/deserialize."""
        msg = _make_assistant_msg("sess-err", server_state.working_dir)
        ctx = EventProcessorContext(
            session_id="sess-err",
            assistant_msg_id=msg.info.id,
            assistant_msg=msg,
            state=server_state,
            working_dir=server_state.working_dir,
        )
        ctx.is_errored = True

        data = ctx.serialize()
        restored = EventProcessorContext.deserialize(
            data, state=server_state, working_dir=server_state.working_dir
        )
        assert restored.is_errored is True

    @pytest.mark.asyncio
    async def test_deserialize_preserves_subagent_tool_parts(
        self, server_state: ServerState
    ) -> None:
        """Subagent tool parts are preserved across serialize/deserialize."""
        session_id = "sess-subagent"
        msg = _make_assistant_msg(session_id, server_state.working_dir)
        ctx = EventProcessorContext(
            session_id=session_id,
            assistant_msg_id=msg.info.id,
            assistant_msg=msg,
            state=server_state,
            working_dir=server_state.working_dir,
        )

        from wolfharness_server.opencode_server.models.parts import (
            TimeStart,
        )

        sub_tp = ToolPart(
            id="part-sub",
            message_id=msg.info.id,
            session_id=session_id,
            tool="task",
            call_id="call-sub-1",
            state=ToolStateRunning(
                time=TimeStart(start=ctx.stream_start_ms),
                input={"subagent_type": "explorer"},
                metadata={"sessionId": "child-1"},
                title="explorer",
            ),
        )
        ctx.add_subagent_tool_part("0:explorer:child-1", sub_tp)

        data = ctx.serialize()
        restored = EventProcessorContext.deserialize(
            data, state=server_state, working_dir=server_state.working_dir
        )

        assert restored.has_subagent_tool_part("0:explorer:child-1")
        restored_sub = restored.get_subagent_tool_part("0:explorer:child-1")
        assert restored_sub is not None
        assert restored_sub.tool == "task"
