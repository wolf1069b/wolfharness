"""TDD tests for PR review round 3 fixes.

Fix #3/#4 (REAL BUG): get_or_create_session_agent() calls
parent_agent.mcp.get_or_create_session(parent_session_id) to read the
parent's snapshot. If the parent session was already cleaned up,
get_or_create_session() CREATES a new phantom McpSessionContext that
will NEVER be cleaned up — memory leak.

Fix #2 (IMPROVEMENT): on_disconnect callback in _handle_websocket_client()
is only called in the except ConnectionClosed handler. If a non-
ConnectionClosed exception propagates, the callback is skipped.
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from wolfharness.mcp_server.config_snapshot import McpConfigSnapshot
from wolfharness.orchestrator.session_controller import SessionController, SessionState


if TYPE_CHECKING:
    from wolfharness.mcp_server.manager import MCPManager


# ============================================================================
# Fix #3/#4: get_or_create_session_agent must not recreate cleaned parent
# ============================================================================


@pytest.mark.integration
@pytest.mark.asyncio
async def test_get_or_create_session_agent_does_not_recreate_cleaned_parent_session() -> None:
    """get_or_create_session_agent() must not recreate a cleaned-up parent context.

    Bug: The method calls parent_agent.mcp.get_or_create_session(parent_id)
    to read the parent's snapshot. If the parent was already cleaned up,
    get_or_create_session() creates a phantom McpSessionContext that leaks.

    Steps:
    1. Create a real Agent with a real MCPManager.
    2. Register it as a parent session in SessionController.
    3. Create parent MCP session context with a snapshot.
    4. cleanup_session(parent_id) — removes the session context.
    5. Create a child session state with parent_session_id=parent_id.
    6. Call get_or_create_session_agent(child_id).
    7. Assert parent_id has NO session context (no phantom created).
    """
    from wolfharness.agents.native_agent import Agent
    from wolfharness.models.agents import NativeAgentConfig
    from wolfharness.models.manifest import AgentsManifest

    def simple_callback(message: str) -> str:
        return f"Test: {message}"

    parent_agent = Agent.from_callback(name="test_agent", callback=simple_callback)
    await parent_agent.__aenter__()
    mcp_manager: MCPManager = parent_agent.mcp

    mock_pool: MagicMock = MagicMock()
    mock_pool.manifest = AgentsManifest(
        agents={"test_agent": NativeAgentConfig(model="test")},
    )
    mock_pool.main_agent_name = "test_agent"
    mock_pool._config_file_path = None
    mock_pool.skill_capabilities = []
    mock_pool.mcp = mcp_manager
    mock_pool.get_context.return_value = MagicMock()
    mock_pool._factory.create_session_agent = AsyncMock(return_value=parent_agent)

    controller = SessionController(pool=mock_pool)

    parent_id = "test-r3-parent"
    child_id = "test-r3-child"

    child_agent = None
    try:
        parent_state = SessionState(
            session_id=parent_id,
            agent_name="test_agent",
        )
        parent_state.agent = parent_agent
        parent_state.is_per_session_agent = False
        controller._sessions[parent_id] = parent_state
        controller._session_agents[parent_id] = parent_agent

        parent_ctx = mcp_manager.get_or_create_session(parent_id)
        parent_ctx.snapshot = McpConfigSnapshot()

        await mcp_manager.cleanup_session(parent_id)
        assert mcp_manager.get_session_context(parent_id) is None

        child_state = SessionState(
            session_id=child_id,
            agent_name="test_agent",
            parent_session_id=parent_id,
        )
        controller._sessions[child_id] = child_state

        child_agent = await controller.get_or_create_session_agent(
            child_id, agent_name="test_agent"
        )

        assert mcp_manager.get_session_context(parent_id) is None, (
            "get_or_create_session_agent() must not recreate a cleaned-up "
            "parent session context via get_or_create_session()"
        )
    finally:
        if child_agent is not None:
            with contextlib.suppress(Exception):
                await child_agent.__aexit__(None, None, None)
        with contextlib.suppress(Exception):
            await parent_agent.__aexit__(None, None, None)
        with contextlib.suppress(Exception):
            await controller.close_session(child_id)
        with contextlib.suppress(Exception):
            await controller.close_session(parent_id)
        with contextlib.suppress(Exception):
            await mcp_manager.cleanup()


# ============================================================================
# Fix #3/#4 regression guard: parent snapshot read without leak when active
# ============================================================================


@pytest.mark.integration
@pytest.mark.asyncio
async def test_get_or_create_session_agent_reads_parent_snapshot_without_leaking() -> None:
    """get_or_create_session_agent() reads parent snapshot without recreating context.

    When the parent session is still active (not cleaned up), the method
    should read the existing McpSessionContext, not create a new one.

    This test passes both before and after the fix — it guards against
    regressions where the parent context is accidentally recreated.
    """
    from wolfharness.agents.native_agent import Agent
    from wolfharness.models.agents import NativeAgentConfig
    from wolfharness.models.manifest import AgentsManifest

    def simple_callback(message: str) -> str:
        return f"Test: {message}"

    parent_agent = Agent.from_callback(name="test_agent", callback=simple_callback)
    await parent_agent.__aenter__()
    mcp_manager: MCPManager = parent_agent.mcp

    mock_pool: MagicMock = MagicMock()
    mock_pool.manifest = AgentsManifest(
        agents={"test_agent": NativeAgentConfig(model="test")},
    )
    mock_pool.main_agent_name = "test_agent"
    mock_pool._config_file_path = None
    mock_pool.skill_capabilities = []
    mock_pool.mcp = mcp_manager
    mock_pool.get_context.return_value = MagicMock()
    mock_pool._factory.create_session_agent = AsyncMock(return_value=parent_agent)

    controller = SessionController(pool=mock_pool)

    parent_id = "test-r3-parent-active"
    child_id = "test-r3-child-active"

    parent_ctx_original = mcp_manager.get_or_create_session(parent_id)
    parent_ctx_original.snapshot = McpConfigSnapshot(
        pool_configs=(),
        agent_configs=(),
        session_configs=(),
        skill_configs=(),
    )

    child_agent = None
    try:
        parent_state = SessionState(
            session_id=parent_id,
            agent_name="test_agent",
        )
        parent_state.agent = parent_agent
        parent_state.is_per_session_agent = False
        controller._sessions[parent_id] = parent_state
        controller._session_agents[parent_id] = parent_agent

        child_state = SessionState(
            session_id=child_id,
            agent_name="test_agent",
            parent_session_id=parent_id,
        )
        controller._sessions[child_id] = child_state

        child_agent = await controller.get_or_create_session_agent(
            child_id, agent_name="test_agent"
        )

        parent_ctx_after = mcp_manager.get_session_context(parent_id)
        assert parent_ctx_after is not None
        assert parent_ctx_after is parent_ctx_original, (
            "Parent session context must be the same object, not recreated"
        )
        assert parent_ctx_after.snapshot is parent_ctx_original.snapshot
    finally:
        if child_agent is not None:
            with contextlib.suppress(Exception):
                await child_agent.__aexit__(None, None, None)
        with contextlib.suppress(Exception):
            await parent_agent.__aexit__(None, None, None)
        with contextlib.suppress(Exception):
            await controller.close_session(child_id)
        with contextlib.suppress(Exception):
            await controller.close_session(parent_id)
        with contextlib.suppress(Exception):
            await mcp_manager.cleanup()


# ============================================================================
# Fix #2: on_disconnect called on non-ConnectionClosed exception
# ============================================================================


@pytest.mark.unit
@pytest.mark.asyncio
async def test_on_disconnect_called_on_non_connection_closed_exception() -> None:
    """on_disconnect must be called even when a non-ConnectionClosed exception occurs.

    Bug: on_disconnect is only called in the except ConnectionClosed handler.
    If a different exception propagates, the callback is skipped.

    Fix: Move on_disconnect to the finally block so it runs regardless
    of the exception type.
    """
    import websockets.exceptions  # noqa: F401 — ensure submodule loaded

    from acp.transports import _handle_websocket_client

    on_disconnect = AsyncMock()
    mock_conn = MagicMock()
    mock_conn.close = AsyncMock()
    mock_conn._conn = None

    shutdown = asyncio.Event()
    shutdown.set()

    with (
        patch(
            "acp.agent.connection.AgentSideConnection",
            return_value=mock_conn,
        ),
        patch("acp.transports._WebSocketReadStream"),
        patch("acp.transports._WebSocketWriteStream"),
        patch("asyncio.wait", side_effect=RuntimeError("test error")),
        pytest.raises(RuntimeError, match="test error"),
    ):
        await _handle_websocket_client(
            websocket=MagicMock(),
            agent_factory=MagicMock(),
            shutdown=shutdown,
            connections=[],
            debug_file=None,
            ping_interval=None,
            pong_timeout=10.0,
            max_missed_pongs=3,
            kwargs={},
            on_disconnect=on_disconnect,
        )

    assert on_disconnect.called, (
        "on_disconnect must be called in the finally block, "
        "not only in the except ConnectionClosed handler"
    )
