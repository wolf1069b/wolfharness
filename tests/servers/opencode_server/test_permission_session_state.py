"""Tests for permission routes reading from SessionState via SessionController.

Validates A5.1 (permissions on SessionState), A5.2 (routes via SessionController),
and A5.6 (fast-path Future resolution).
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, Mock

import pytest

from wolfharness.orchestrator.core import SessionState
from wolfharness_server.opencode_server.input_provider import (
    OpenCodeInputProvider,
    PendingPermission,
)
from wolfharness_server.opencode_server.models import (
    PermissionReplyRequest,
    PermissionResolvedEvent,
)
from wolfharness_server.opencode_server.routes.permission_routes import (
    list_permissions,
    reply_to_permission,
)
from wolfharness_server.opencode_server.state import ServerState


pytestmark = pytest.mark.integration


async def test_ensure_input_provider_stores_on_session_state():
    """A5.1: ensure_input_provider stores provider on SessionState when controller is available."""
    mock_agent = Mock()
    mock_agent.agent_pool = None
    mock_agent.host_context = None

    session = SessionState(session_id="test-session", agent_name="test-agent")
    session_controller = Mock()
    session_controller.get_session = Mock(return_value=session)
    session_controller.list_sessions = Mock(return_value=[session])

    state = ServerState(
        working_dir="/tmp",
        agent=mock_agent,
        session_controller=session_controller,
    )

    provider = state.ensure_input_provider("test-session")

    # Provider should be stored on SessionState
    assert session.input_provider is provider
    assert isinstance(provider, OpenCodeInputProvider)


async def test_list_permissions_reads_from_session_controller():
    """A5.2: list_permissions iterates sessions via SessionController."""
    mock_agent = Mock()
    mock_agent.agent_pool = None
    mock_agent.host_context = None

    session = SessionState(session_id="sess-1", agent_name="test-agent")
    provider = OpenCodeInputProvider(
        state=Mock(),
        session_id="sess-1",
    )
    # Inject a pending permission manually
    future = asyncio.get_running_loop().create_future()
    provider._pending_permissions["perm-1"] = PendingPermission(
        permission_id="perm-1",
        tool_name="bash",
        args={"command": "echo hello"},
        future=future,
    )
    session.input_provider = provider

    session_controller = Mock()
    session_controller._sessions = {"sess-1": session}

    state = ServerState(
        working_dir="/tmp",
        agent=mock_agent,
        session_controller=session_controller,
    )
    state.broadcast_event = AsyncMock()  # type: ignore[method-assign]

    result = await list_permissions(state)

    assert len(result) == 1
    assert result[0].id == "perm-1"
    assert result[0].session_id == "sess-1"
    assert result[0].permission == "bash"


async def test_reply_to_permission_resolves_via_session_controller():
    """A5.2: reply_to_permission finds and resolves via SessionController."""
    mock_agent = Mock()
    mock_agent.agent_pool = None
    mock_agent.host_context = None

    session = SessionState(session_id="sess-1", agent_name="test-agent")
    provider = OpenCodeInputProvider(
        state=Mock(),
        session_id="sess-1",
    )
    future = asyncio.get_running_loop().create_future()
    provider._pending_permissions["perm-1"] = PendingPermission(
        permission_id="perm-1",
        tool_name="bash",
        args={"command": "echo hello"},
        future=future,
    )
    session.input_provider = provider

    session_controller = Mock()
    session_controller._sessions = {"sess-1": session}

    state = ServerState(
        working_dir="/tmp",
        agent=mock_agent,
        session_controller=session_controller,
    )
    state.broadcast_event = AsyncMock()  # type: ignore[method-assign]

    body = PermissionReplyRequest(reply="once")
    result = await reply_to_permission("perm-1", body, state)

    assert result is True
    assert future.done()
    assert future.result() == "once"
    # Verify broadcast was sent
    assert state.broadcast_event.await_count == 1  # type: ignore[attr-defined]
    event = state.broadcast_event.await_args.args[0]  # type: ignore[union-attr]
    assert isinstance(event, PermissionResolvedEvent)
    assert event.properties.request_id == "perm-1"
    assert event.properties.reply == "once"


async def test_reply_to_permission_not_found_with_controller():
    """A5.2: reply_to_permission returns 404 when permission not found via controller."""
    mock_agent = Mock()
    mock_agent.agent_pool = None
    mock_agent.host_context = None

    session = SessionState(session_id="sess-1", agent_name="test-agent")
    # No input_provider set, so no permissions
    session_controller = Mock()
    session_controller._sessions = {"sess-1": session}

    state = ServerState(
        working_dir="/tmp",
        agent=mock_agent,
        session_controller=session_controller,
    )

    from fastapi import HTTPException

    body = PermissionReplyRequest(reply="once")
    with pytest.raises(HTTPException) as exc_info:
        await reply_to_permission("nonexistent", body, state)

    assert exc_info.value.status_code == 404


async def test_fast_path_future_resolution():
    """A5.6: HTTP POST sets Future result; tool awaiting same Future resolves immediately.

    This test simulates the exact fast-path without broadcast overhead:
    1. Create a PendingPermission with a Future
    2. Start a task awaiting that Future
    3. Call resolve_permission() -> future.set_result()
    4. The awaiting task wakes up immediately with the result
    """
    provider = OpenCodeInputProvider(state=Mock(), session_id="sess-1")

    future: asyncio.Future[str] = asyncio.get_running_loop().create_future()
    provider._pending_permissions["perm-fast"] = PendingPermission(
        permission_id="perm-fast",
        tool_name="bash",
        args={"command": "echo fast"},
        future=future,
    )

    # Start a task awaiting the future (simulates tool side)
    async def tool_side() -> str:
        return await future

    tool_task = asyncio.create_task(tool_side())

    # Small yield to ensure the task is awaiting
    await asyncio.sleep(0)

    # Simulate HTTP POST handler resolving the permission
    resolved = provider.resolve_permission("perm-fast", "once")
    assert resolved is True

    # Tool should resolve immediately (fast path — no polling, no timeout)
    result = await asyncio.wait_for(tool_task, timeout=0.5)
    assert result == "once"


async def test_fast_path_future_always_approval():
    """A5.6: 'always' reply sets standing approval and resolves Future immediately."""
    provider = OpenCodeInputProvider(state=Mock(), session_id="sess-1")

    future: asyncio.Future[str] = asyncio.get_running_loop().create_future()
    provider._pending_permissions["perm-always"] = PendingPermission(
        permission_id="perm-always",
        tool_name="bash",
        args={"command": "echo always"},
        future=future,
    )

    # Simulate tool side: awaits future, then processes response (like get_tool_confirmation does)
    async def tool_side() -> str:
        response = await future
        # Consumer processes the response and updates approvals
        provider._handle_permission_response(response, "bash")
        return response

    tool_task = asyncio.create_task(tool_side())
    await asyncio.sleep(0)

    # HTTP handler replies "always"
    resolved = provider.resolve_permission("perm-always", "always")
    assert resolved is True

    result = await asyncio.wait_for(tool_task, timeout=0.5)
    assert result == "always"

    # Standing approval should now be recorded (by consumer side)
    assert provider._tool_approvals.get("bash") == "always"

    # Second request for same tool should auto-resolve without creating a new Future
    result2 = provider._handle_permission_response("always", "bash")
    assert result2 == "allow"


async def test_legacy_fallback_without_session_controller():
    """A5.1: Without session_controller, routes return empty / 404 (no legacy fallback)."""
    mock_agent = Mock()
    mock_agent.agent_pool = None
    mock_agent.host_context = None

    state = ServerState(working_dir="/tmp", agent=mock_agent)
    provider = OpenCodeInputProvider(state=state, session_id="sess-legacy")

    future = asyncio.get_running_loop().create_future()
    provider._pending_permissions["perm-legacy"] = PendingPermission(
        permission_id="perm-legacy",
        tool_name="bash",
        args={"command": "echo legacy"},
        future=future,
    )
    broadcast_calls = []

    async def _mock_broadcast(event):
        broadcast_calls.append(event)

    state.broadcast_event = _mock_broadcast

    # list_permissions returns empty when no session_controller
    result = await list_permissions(state)
    assert len(result) == 0

    # reply_to_permission raises 404 when no session_controller
    from fastapi import HTTPException

    body = PermissionReplyRequest(reply="once")
    with pytest.raises(HTTPException) as exc_info:
        await reply_to_permission("perm-legacy", body, state)
    assert exc_info.value.status_code == 404
