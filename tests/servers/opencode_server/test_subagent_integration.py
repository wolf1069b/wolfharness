"""Merged tests for subagent integration in the OpenCode server.

Combines tests from:
- test_subagent_completion_red_flags.py
- test_subagent_fixes.py
- test_subagent_handler.py
- test_subagent_sessions.py
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import FastAPI
import pytest

from wolfharness.agents.events import StreamCompleteEvent, SubAgentEvent
from wolfharness.messaging import ChatMessage
from wolfharness_server.opencode_server.dependencies import get_state
from wolfharness_server.opencode_server.event_processor_context import (
    EventProcessorContext,
)
from wolfharness_server.opencode_server.models import (
    MessagePath,
    MessageTime,
    MessageWithParts,
)
from wolfharness_server.opencode_server.routes import file_router, message_router, session_router
from wolfharness_server.opencode_server.session_pool_integration import (
    ensure_session,
    get_messages_for_session,
)
from wolfharness_server.opencode_server.stream_adapter import OpenCodeStreamAdapter
from wolfharness_toolsets.builtin.subagent_tools import SubagentTools


pytestmark = pytest.mark.integration


if TYPE_CHECKING:
    from wolfharness_server.opencode_server.event_processor import EventProcessor
    from wolfharness_server.opencode_server.state import ServerState


# =============================================================================
# Shared Helpers (from test_subagent_completion_red_flags.py)
# =============================================================================


def _make_parent_ctx(
    server_state: ServerState,
    parent_session_id: str = "parent-session",
    parent_msg_id: str = "parent-msg-1",
) -> EventProcessorContext:
    """Create a parent EventProcessorContext for subagent tests."""
    assistant_msg = MessageWithParts.assistant(
        message_id=parent_msg_id,
        session_id=parent_session_id,
        time=MessageTime(created=0),
        agent_name="lead-agent",
        model_id="test-model",
        parent_id="parent-user-1",
        provider_id="wolfharness",
        path=MessagePath(cwd="/tmp", root="/tmp"),
    )
    return EventProcessorContext(
        session_id=parent_session_id,
        assistant_msg_id=parent_msg_id,
        assistant_msg=assistant_msg,
        state=server_state,
        working_dir="/tmp",
    )


async def _process_events(
    processor: EventProcessor,
    events: list[Any],
    ctx: EventProcessorContext,
) -> list[Any]:
    """Process a sequence of events and collect all emitted SSE events."""
    emitted: list[Any] = []
    for event in events:
        emitted.extend([e async for e in processor.process(event, ctx)])
    return emitted


# =============================================================================
# --- Merged from test_subagent_completion_red_flags.py ---
# =============================================================================

# Red-Flag Test #3: inject_prompt does not re-awaken lead agent


@pytest.mark.asyncio
async def test_background_task_inject_prompt_wakes_lead_agent(
    server_state: ServerState,
) -> None:
    """inject_prompt after background task completion MUST re-awaken the lead agent.

    CURRENT BEHAVIOR (FIXED):
      inject_prompt() now delegates to SessionPool.receive_request() or
      SessionPool.inject_prompt() when no active run context exists,
      which triggers auto-resume via SessionController.
      The lead agent receives the completion notice and resumes reasoning.

    PREVIOUS BEHAVIOR (BROKEN):
      inject_prompt() was a silent no-op when no active run context existed,
      causing the lead agent to never resume after background task completion.
    """
    import inspect

    from wolfharness.agents.base_agent import BaseAgent

    source = inspect.getsource(BaseAgent.inject_prompt)

    # Verify the fixed implementation delegates to SessionPool for auto-resume
    assert "session_pool" in source, (
        "inject_prompt must reference session_pool to delegate when no run context exists"
    )
    assert "send_message" in source or "inject_prompt" in source, (
        "inject_prompt must call send_message or session_pool.inject_prompt "
        "to trigger auto-resume when no active run context is available"
    )
    assert "fire_and_forget" in source, (
        "inject_prompt must use fire_and_forget to schedule the request asynchronously"
    )

    # Verify the fallback path for shared agents (no fixed session_id)
    assert "find_sessions_by_agent_name" in source, (
        "inject_prompt must use find_sessions_by_agent_name as fallback for shared agents"
    )


# =============================================================================
# --- Merged from test_subagent_fixes.py ---
# =============================================================================


@pytest.mark.asyncio
async def test_get_session_children(async_client, server_state):
    """Test GET /session/{session_id}/children endpoint."""
    # Create parent session
    parent_resp = await async_client.post("/session", json={"title": "Parent"})
    assert parent_resp.status_code == 200
    parent_id = parent_resp.json()["id"]

    # Create child sessions
    child1_resp = await async_client.post(
        "/session", json={"title": "Child 1", "parent_id": parent_id}
    )
    assert child1_resp.status_code == 200
    child1_id = child1_resp.json()["id"]

    child2_resp = await async_client.post(
        "/session", json={"title": "Child 2", "parent_id": parent_id}
    )
    assert child2_resp.status_code == 200
    child2_id = child2_resp.json()["id"]

    # Create unrelated session
    other_resp = await async_client.post("/session", json={"title": "Other"})
    assert other_resp.status_code == 200
    other_id = other_resp.json()["id"]

    # Get children of parent
    resp = await async_client.get(f"/session/{parent_id}/children")
    assert resp.status_code == 200
    children = resp.json()

    assert len(children) == 2
    child_ids = [c["id"] for c in children]
    assert child1_id in child_ids
    assert child2_id in child_ids
    assert other_id not in child_ids

    # Verify parent_id in response
    for child in children:
        assert child["parentID"] == parent_id

    # Get children of unrelated session (should be empty)
    resp = await async_client.get(f"/session/{other_id}/children")
    assert resp.status_code == 200
    assert len(resp.json()) == 0


@pytest.mark.asyncio
async def test_task_tool_return_format():
    """Test that task tool returns structured data with metadata."""
    tools = SubagentTools()

    # Mock context
    ctx = MagicMock()
    ctx.run_ctx.depth = 0

    # Mock node (agent) using a class to satisfy runtime_checkable Protocol
    class MockStreamingAgent:
        agent_type = "agent"
        type = "native"

        def __init__(self):
            self.run_stream = MagicMock()

    mock_agent = MockStreamingAgent()

    # Mock run_stream to yield events
    async def mock_stream(*args, **kwargs):
        yield StreamCompleteEvent(message=ChatMessage(role="assistant", content="Task result"))

    mock_agent.run_stream.side_effect = mock_stream
    ctx.pool.manifest.agents = {"child_agent": mock_agent}
    ctx.pool.manifest.teams = {}
    ctx.node.session_id = "parent_session"
    ctx.events.emit_event = AsyncMock()
    ctx.create_child_session = AsyncMock(return_value="child_session_123")

    # Mock pool.session_pool.run_stream to yield the same events as the agent
    async def mock_session_run_stream(*args, **kwargs):
        async for event in mock_stream():
            yield event

    ctx.pool.session_pool.run_stream = mock_session_run_stream

    # Execute task
    result = await tools.task(
        ctx=ctx, agent_or_team="child_agent", prompt="Do work", description="Work", async_mode=False
    )

    # Verify result format
    assert isinstance(result, dict)
    assert "output" in result
    assert result["output"] == "Task result"
    assert "metadata" in result
    assert "sessionId" in result["metadata"]
    assert result["metadata"]["sessionId"] is not None
    assert isinstance(result["metadata"]["sessionId"], str)


@pytest.mark.asyncio
async def test_task_tool_async_mode_return_format():
    """Test that task tool in async mode returns structured data."""
    tools = SubagentTools()

    # Mock context
    ctx = MagicMock()
    ctx.run_ctx.depth = 0
    ctx.node.session_id = "parent_session"

    # Mock node
    class MockStreamingAgent:
        agent_type = "agent"
        type = "native"

        def __init__(self):
            self.run_stream = MagicMock()

    mock_agent = MockStreamingAgent()
    ctx.pool.manifest.agents = {"child_agent": mock_agent}
    ctx.pool.manifest.teams = {}

    # Mock internal_fs
    ctx.internal_fs.mkdirs = MagicMock()

    # Mock events.emit_event (needed for SpawnSessionStart emission)
    ctx.events.emit_event = AsyncMock()
    ctx.create_child_session = AsyncMock(return_value="child_session_123")

    # Execute task in async mode
    result = await tools.task(
        ctx=ctx, agent_or_team="child_agent", prompt="Do work", description="Work", async_mode=True
    )

    # Verify result format
    assert isinstance(result, dict)
    assert "output" in result
    assert "Task started in background" in result["output"]
    assert "metadata" in result
    assert "taskId" in result["metadata"]
    assert "sessionId" in result["metadata"]
    assert "outputFile" in result["metadata"]


# =============================================================================
# --- Merged from test_subagent_handler.py ---
# =============================================================================


@pytest.mark.asyncio
async def test_subagent_event_without_child_session_id(server_state: ServerState) -> None:
    """Test that SubAgentEvent without child_session_id works and doesn't trigger ensure_session."""
    # Setup
    session_id = "parent-session"

    from wolfharness_server.opencode_server.models import MessagePath, MessageTime

    assistant_msg = MessageWithParts.assistant(
        message_id="msg-1",
        session_id=session_id,
        time=MessageTime(created=1000),
        agent_name="test-agent",
        model_id="test-model",
        parent_id="user-msg-1",
        provider_id="test-provider",
        path=MessagePath(cwd="/tmp", root="/tmp"),
    )

    adapter = OpenCodeStreamAdapter(
        state=server_state,
        session_id=session_id,
        assistant_msg_id="msg-1",
        assistant_msg=assistant_msg,
        working_dir="/tmp",
    )

    # Create a stream with a SubAgentEvent
    async def event_stream():
        inner_event = StreamCompleteEvent(message=ChatMessage(role="assistant", content="Done"))

        yield SubAgentEvent(
            source_name="subagent",
            source_type="agent",
            event=inner_event,
            child_session_id=None,  # No child session ID
        )

    # Run process_stream with ensure_session patched
    with patch(
        "wolfharness_server.opencode_server.session_pool_integration.ensure_session"
    ) as mock_ensure:
        async for _ in adapter.process_stream(event_stream()):
            pass

        # Verify ensure_session was NOT called
        mock_ensure.assert_not_called()


# =============================================================================
# --- Merged from test_subagent_sessions.py ---
# =============================================================================


class TestSubagentSessions:
    """Integration tests for subagent session handling."""

    @pytest.fixture
    def app(self, server_state: ServerState) -> FastAPI:
        """Create a FastAPI app with message routes."""
        app = FastAPI()
        app.include_router(session_router)
        app.include_router(message_router)
        app.include_router(file_router)
        app.dependency_overrides[get_state] = lambda: server_state
        return app

    @pytest.fixture
    def mock_agent_stream(self, server_state: ServerState):
        """Mock the agent's run_stream method to yield specific events."""
        original_run_stream = server_state.agent.run_stream

        # Create a mock that we can configure per test
        mock = MagicMock()
        server_state.agent.run_stream = mock

        yield mock

        # Restore original
        server_state.agent.run_stream = original_run_stream

    @pytest.mark.asyncio
    async def test_child_session_has_parent_id(
        self,
        server_state,
    ):
        """Verify ensure_session correctly sets parent_id on the model."""
        parent_id = "ses_parent"
        child_id = "ses_child"

        # Pre-create parent
        await ensure_session(server_state, parent_id)

        # Create child with parent reference
        child_session = await ensure_session(server_state, child_id, parent_id=parent_id)

        assert child_session.id == child_id
        assert child_session.parent_id == parent_id

        # Verify persistence
        stored_session = server_state.sessions[child_id]
        assert stored_session.parent_id == parent_id

    @pytest.mark.asyncio
    async def test_backward_compatibility_non_subagent(
        self,
        async_client,
        mock_agent_stream,
        server_state,
    ):
        """Verify that regular tool usage without subagents still works."""
        # Create session
        response = await async_client.post("/session", json={"title": "Legacy"})
        assert response.status_code == 200
        session_id = response.json()["id"]

        # Mock normal stream without subagent events
        async def stream_generator(*args, **kwargs):
            from wolfharness.agents.events import PartDeltaEvent

            yield PartDeltaEvent.text(index=0, content="Normal response")
            yield StreamCompleteEvent(
                message=ChatMessage(role="assistant", content="Normal response")
            )

        mock_agent_stream.side_effect = stream_generator

        # Send message
        response = await async_client.post(
            f"/session/{session_id}/message",
            json={"parts": [{"type": "text", "text": "Hello"}]},
        )
        assert response.status_code == 200

        # Wait for processing
        await asyncio.sleep(0.2)

        # Verify no unexpected sessions were created
        assert len(server_state.sessions) == 1
        assert session_id in server_state.sessions

        # Verify message was appended
        session_messages = await get_messages_for_session(server_state, session_id)
        assert len(session_messages) > 0
