"""Tests for agent_name/model_id/provider_id propagation in assistant messages.

Regression tests for the bug where the OpenCode TUI showed correct agent name
and model in the input box, but after execution started, both agent and model
became 'default'/'wolfharness' for the lead agent AND sub-agents.

Root cause: ``_before_consumer_loop`` and ``_reconstruct_tool_parts_from_checkpoint``
hardcoded ``agent_name="wolfharness"``, ``model_id="default"``,
``provider_id="wolfharness"`` when creating the assistant message for the async
path. The sync path (``_process_message_locked``) correctly used values from
the HTTP request, but the TUI uses ``prompt_async`` which goes through the
async path.

These tests verify that:
1. ``_before_consumer_loop`` uses ``agent_name`` from the session state
   (not the hardcoded "wolfharness").
2. ``_before_consumer_loop`` uses model info from pending metadata when
   provided by the REST handler (not the hardcoded "default"/"wolfharness").
3. ``_before_consumer_loop`` falls back to "default"/"wolfharness" only when
   no model info is available.
4. ``_reconstruct_tool_parts_from_checkpoint`` uses ``agent_name`` from the
   session state (not the hardcoded "wolfharness").
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest

from wolfharness.orchestrator.session_controller import SessionState


pytestmark = pytest.mark.unit


def _make_session_state(agent_name: str, agent: Any = None) -> SessionState:
    """Build a real SessionState with the given agent_name."""
    return SessionState(
        session_id="test-session",
        agent_name=agent_name,
        agent=agent,
    )


def _make_integration(
    session_pool: Mock | None = None,
    server_state: Mock | None = None,
) -> Any:
    """Build an OpenCodeSessionPoolIntegration with mocked deps."""
    from wolfharness_server.opencode_server.session_pool_integration import (
        OpenCodeSessionPoolIntegration,
    )

    if session_pool is None:
        session_pool = Mock()
    if server_state is None:
        server_state = Mock()
        server_state.working_dir = "/tmp"
        server_state.resolve_default_model_info = Mock(return_value=("default", "wolfharness"))
        server_state.model_variants = {}
    return OpenCodeSessionPoolIntegration(session_pool, server_state)


# =============================================================================
# _before_consumer_loop: agent_name propagation
# =============================================================================


class TestBeforeConsumerLoopAgentName:
    """_before_consumer_loop must use agent_name from session state."""

    @pytest.mark.asyncio
    async def test_uses_session_agent_name_not_wolfharness(self) -> None:
        """Assistant message should carry the session's agent_name, not 'wolfharness'."""
        session_pool = Mock()
        session_pool.sessions = Mock()
        session_pool.sessions.get_session = Mock(
            return_value=_make_session_state(agent_name="engineer")
        )

        integration = _make_integration(session_pool=session_pool)

        await integration._before_consumer_loop("test-session")

        ctx = integration._contexts.get("test-session")
        assert ctx is not None
        assert ctx.assistant_msg.info.agent == "engineer"

    @pytest.mark.asyncio
    async def test_uses_different_agent_name_per_session(self) -> None:
        """Different sessions with different agents should get different agent_names."""
        session_pool = Mock()
        session_pool.sessions = Mock()

        def _get_session(session_id: str) -> SessionState:
            name = "historian" if session_id == "session-h" else "logician"
            return _make_session_state(agent_name=name)

        session_pool.sessions.get_session = Mock(side_effect=_get_session)

        integration = _make_integration(session_pool=session_pool)

        await integration._before_consumer_loop("session-h")
        await integration._before_consumer_loop("session-l")

        ctx_h = integration._contexts["session-h"]
        ctx_l = integration._contexts["session-l"]
        assert ctx_h.assistant_msg.info.agent == "historian"
        assert ctx_l.assistant_msg.info.agent == "logician"

    @pytest.mark.asyncio
    async def test_falls_back_to_wolfharness_when_no_session_state(self) -> None:
        """When session state is unavailable, fall back to 'wolfharness' (graceful degradation)."""
        session_pool = Mock()
        session_pool.sessions = Mock()
        session_pool.sessions.get_session = Mock(return_value=None)

        integration = _make_integration(session_pool=session_pool)

        await integration._before_consumer_loop("unknown-session")

        ctx = integration._contexts.get("unknown-session")
        assert ctx is not None
        # Graceful fallback — should still create a message, just with the
        # default agent_name.
        assert ctx.assistant_msg.info.agent == "wolfharness"


# =============================================================================
# _before_consumer_loop: model_id / provider_id propagation
# =============================================================================


class TestBeforeConsumerLoopModelInfo:
    """_before_consumer_loop must use model info from pending metadata."""

    @pytest.mark.asyncio
    async def test_uses_pending_model_id_and_provider_id(self) -> None:
        """REST handler model info should propagate to the assistant message."""
        session_pool = Mock()
        session_pool.sessions = Mock()
        session_pool.sessions.get_session = Mock(
            return_value=_make_session_state(agent_name="engineer")
        )

        integration = _make_integration(session_pool=session_pool)
        # Simulate the REST handler storing pending model metadata.
        # route_message stores message_id in both _pending_message_ids
        # (for the assistant_msg_id mechanism) and _pending_message_metadata
        # (for the model propagation mechanism).
        integration._pending_message_ids["test-session"] = "msg_from_rest"
        integration._pending_message_metadata["test-session"] = {
            "message_id": "msg_from_rest",
            "model_id": "gpt-4o",
            "provider_id": "openai",
        }

        await integration._before_consumer_loop("test-session")

        ctx = integration._contexts.get("test-session")
        assert ctx is not None
        assert ctx.assistant_msg.info.model_id == "gpt-4o"
        assert ctx.assistant_msg.info.provider_id == "openai"
        assert ctx.assistant_msg_id == "msg_from_rest"

    @pytest.mark.asyncio
    async def test_falls_back_to_default_when_no_pending_model(self) -> None:
        """Without pending model metadata, fall back to 'default'/'wolfharness'."""
        session_pool = Mock()
        session_pool.sessions = Mock()
        session_pool.sessions.get_session = Mock(
            return_value=_make_session_state(agent_name="engineer")
        )

        integration = _make_integration(session_pool=session_pool)

        await integration._before_consumer_loop("test-session")

        ctx = integration._contexts.get("test-session")
        assert ctx is not None
        # agent_name is still correct from session state
        assert ctx.assistant_msg.info.agent == "engineer"
        # model falls back to defaults
        assert ctx.assistant_msg.info.model_id == "default"
        assert ctx.assistant_msg.info.provider_id == "wolfharness"

    @pytest.mark.asyncio
    async def test_pending_message_id_still_works_with_metadata(self) -> None:
        """The legacy _pending_message_ids mechanism should still work alongside metadata."""
        session_pool = Mock()
        session_pool.sessions = Mock()
        session_pool.sessions.get_session = Mock(
            return_value=_make_session_state(agent_name="engineer")
        )

        integration = _make_integration(session_pool=session_pool)
        # Legacy path: only message_id, no model info
        integration._pending_message_ids["test-session"] = "msg_legacy"

        await integration._before_consumer_loop("test-session")

        ctx = integration._contexts.get("test-session")
        assert ctx is not None
        assert ctx.assistant_msg_id == "msg_legacy"
        assert ctx.assistant_msg.info.agent == "engineer"
        # No model info → defaults
        assert ctx.assistant_msg.info.model_id == "default"


# =============================================================================
# _before_consumer_loop: child sessions (sub-agents)
# =============================================================================


class TestBeforeConsumerLoopChildSession:
    """Child sessions (sub-agents) should also get the correct agent_name and model."""

    @pytest.mark.asyncio
    async def test_child_session_uses_child_agent_name(self) -> None:
        """Child session assistant message should use the child's agent_name."""
        session_pool = Mock()
        session_pool.sessions = Mock()

        def _get_session(session_id: str) -> SessionState:
            # Child session has its own agent_name (set when spawned).
            return _make_session_state(agent_name="researcher")

        session_pool.sessions.get_session = Mock(side_effect=_get_session)

        integration = _make_integration(session_pool=session_pool)

        # Simulate _on_spawn_session_start calling start_event_consumer(child_id)
        child_id = "child-session-123"
        await integration._before_consumer_loop(child_id)

        ctx = integration._contexts.get(child_id)
        assert ctx is not None
        assert ctx.assistant_msg.info.agent == "researcher"
        assert ctx.assistant_msg.info.agent != "wolfharness"

    @pytest.mark.asyncio
    async def test_child_session_uses_child_agent_model(self) -> None:
        """Child session assistant message should use the child's model, not the parent's.

        Regression test for #266: child sessions created via create_child_session()
        bypass route_message(), so _pending_message_metadata is never set.
        _create_assistant_message fell back to resolve_default_model_info() which
        returns the parent/default agent's model.
        """
        session_pool = Mock()
        session_pool.sessions = Mock()

        # Create a child agent with a DIFFERENT model from the parent
        child_agent = Mock()
        child_agent.model_name = "openai:gpt-4o-mini"

        def _get_session(session_id: str) -> SessionState:
            return _make_session_state(agent_name="researcher", agent=child_agent)

        session_pool.sessions.get_session = Mock(side_effect=_get_session)

        # Server state returns the PARENT's model (different from child's)
        server_state = Mock()
        server_state.working_dir = "/tmp"
        server_state.resolve_default_model_info = Mock(return_value=("gpt-4o", "openai"))
        server_state.model_variants = {}

        integration = _make_integration(session_pool=session_pool, server_state=server_state)

        child_id = "child-session-123"
        await integration._before_consumer_loop(child_id)

        ctx = integration._contexts.get(child_id)
        assert ctx is not None
        # agent_name should be correct (already tested above)
        assert ctx.assistant_msg.info.agent == "researcher"
        # model_id should come from the child's agent, not the server default
        assert ctx.assistant_msg.info.model_id == "gpt-4o-mini"
        assert ctx.assistant_msg.info.provider_id == "openai"
        # Explicitly assert it's NOT the parent's model
        assert ctx.assistant_msg.info.model_id != "gpt-4o"

    @pytest.mark.asyncio
    async def test_child_session_model_falls_back_when_no_agent(self) -> None:
        """When SessionState.agent is None, fall back to server default gracefully."""
        session_pool = Mock()
        session_pool.sessions = Mock()

        def _get_session(session_id: str) -> SessionState:
            # No agent instance available — just agent_name
            return _make_session_state(agent_name="researcher", agent=None)

        session_pool.sessions.get_session = Mock(side_effect=_get_session)

        server_state = Mock()
        server_state.working_dir = "/tmp"
        server_state.resolve_default_model_info = Mock(return_value=("gpt-4o", "openai"))
        server_state.model_variants = {}

        integration = _make_integration(session_pool=session_pool, server_state=server_state)

        child_id = "child-session-456"
        await integration._before_consumer_loop(child_id)

        ctx = integration._contexts.get(child_id)
        assert ctx is not None
        assert ctx.assistant_msg.info.agent == "researcher"
        # Falls back to server default when no agent instance is available
        assert ctx.assistant_msg.info.model_id == "gpt-4o"
        assert ctx.assistant_msg.info.provider_id == "openai"


# =============================================================================
# _reconstruct_tool_parts_from_checkpoint: agent_name propagation
# =============================================================================


class TestReconstructToolPartsAgentName:
    """_reconstruct_tool_parts_from_checkpoint must use agent_name from session state."""

    @pytest.mark.asyncio
    async def test_uses_session_agent_name(self) -> None:
        """Checkpoint recovery should preserve the session's agent_name on the assistant message."""
        from wolfharness_server.opencode_server.opencode_message_bridge import (
            _reconstruct_tool_parts_from_checkpoint,
        )

        session_pool = Mock()
        session_pool.sessions = Mock()
        session_pool.sessions.get_session = Mock(
            return_value=_make_session_state(agent_name="visionary")
        )

        pool = Mock()
        pool.session_pool = session_pool

        server_state = Mock()
        server_state.working_dir = "/tmp"
        server_state.resolve_default_model_info = Mock(return_value=("default", "wolfharness"))
        server_state.model_variants = {}
        server_state.pool_or_none = pool
        server_state.messages = {}

        # Build a minimal pending call mock
        pending_call = Mock()
        pending_call.tool_name = "bash"
        pending_call.tool_call_id = "call_123"
        pending_call.deferred_strategy = "external"

        _reconstruct_tool_parts_from_checkpoint(
            server_state,
            "test-session",
            [pending_call],
        )

        messages = server_state.messages.get("test-session", [])
        assert len(messages) == 1
        assistant_msg = messages[0]
        assert assistant_msg.info.agent == "visionary"
        assert assistant_msg.info.agent != "wolfharness"

    @pytest.mark.asyncio
    async def test_falls_back_to_wolfharness_when_no_session_state(self) -> None:
        """When session state is unavailable, fall back to 'wolfharness'."""
        from wolfharness_server.opencode_server.opencode_message_bridge import (
            _reconstruct_tool_parts_from_checkpoint,
        )

        session_pool = Mock()
        session_pool.sessions = Mock()
        session_pool.sessions.get_session = Mock(return_value=None)

        pool = Mock()
        pool.session_pool = session_pool

        server_state = Mock()
        server_state.working_dir = "/tmp"
        server_state.resolve_default_model_info = Mock(return_value=("default", "wolfharness"))
        server_state.model_variants = {}
        server_state.pool_or_none = pool
        server_state.messages = {}

        pending_call = Mock()
        pending_call.tool_name = "bash"
        pending_call.tool_call_id = "call_456"
        pending_call.deferred_strategy = "external"

        _reconstruct_tool_parts_from_checkpoint(
            server_state,
            "test-session",
            [pending_call],
        )

        messages = server_state.messages.get("test-session", [])
        assert len(messages) == 1
        assert messages[0].info.agent == "wolfharness"

    @pytest.mark.asyncio
    async def test_no_op_when_no_pending_calls(self) -> None:
        """Function should be a no-op when pending_calls is empty."""
        from wolfharness_server.opencode_server.opencode_message_bridge import (
            _reconstruct_tool_parts_from_checkpoint,
        )

        server_state = Mock()
        server_state.messages = {}

        _reconstruct_tool_parts_from_checkpoint(
            server_state,
            "test-session",
            [],
        )

        assert server_state.messages.get("test-session") is None


# =============================================================================
# route_message: model info passthrough
# =============================================================================


class TestRouteMessageModelPassthrough:
    """route_message should pass model info to _pending_message_metadata."""

    @pytest.mark.asyncio
    async def test_route_message_stores_model_metadata(self) -> None:
        """route_message should store model_id/provider_id in _pending_message_metadata."""
        session_pool = Mock()
        session_pool.sessions = Mock()
        session_pool.sessions.get_session = Mock(
            return_value=_make_session_state(agent_name="engineer")
        )
        session_pool.sessions.get_or_create_session = AsyncMock(
            return_value=(_make_session_state(agent_name="engineer"), False)
        )
        _run_handle = Mock()
        _run_handle.complete_event = Mock()
        _run_handle.complete_event.wait = AsyncMock()
        session_pool.send_message = AsyncMock(return_value=_run_handle)
        session_pool.receive_request = AsyncMock()
        session_pool.event_bus = Mock()
        session_pool.event_bus.subscribe = Mock()

        integration = _make_integration(session_pool=session_pool)
        integration._start_event_consumer = AsyncMock()

        await integration.route_message(
            session_id="test-session",
            content="hello",
            agent_name="engineer",
            message_id="msg_123",
            model_id="gpt-4o",
            provider_id="openai",
        )

        # The metadata should be stored for _before_consumer_loop to consume
        assert "test-session" in integration._pending_message_metadata
        meta = integration._pending_message_metadata["test-session"]
        assert meta["message_id"] == "msg_123"
        assert meta["model_id"] == "gpt-4o"
        assert meta["provider_id"] == "openai"
