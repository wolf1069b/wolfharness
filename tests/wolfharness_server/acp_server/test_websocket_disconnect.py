"""Integration tests for WebSocket disconnect session cleanup.

Verifies that when a WebSocket connection drops,
``close_all_sessions_for_connection()`` closes all sessions associated
with that connection while leaving sessions on other connections alive.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from wolfharness_server.acp_server.session_manager import ACPSessionManager


def _make_mock_session(session_id: str) -> AsyncMock:
    """Create a mock ACPSession with async close().

    Args:
        session_id: The session ID to assign to the mock session.

    Returns:
        An AsyncMock configured with ``session_id`` and an async ``close()``.
    """
    session: AsyncMock = AsyncMock()
    session.session_id = session_id
    session.close = AsyncMock()
    return session


def _make_mock_pool() -> MagicMock:
    """Create a mock AgentPool suitable for ACPSessionManager.

    Returns:
        A MagicMock with ``session_pool``, ``sessions``, ``store``,
        and ``manifest`` configured for session lifecycle tests.
    """
    mock_pool: MagicMock = MagicMock()
    mock_sessions: MagicMock = MagicMock()
    mock_pool.session_pool.sessions = mock_sessions
    mock_sessions.close_session = AsyncMock()
    mock_pool.session_pool.close_session = AsyncMock()
    mock_pool.manifest.agents = {}
    return mock_pool


@pytest.mark.integration
@pytest.mark.asyncio
async def test_websocket_disconnect_closes_all_sessions() -> None:
    """Disconnecting a WebSocket closes all sessions on that connection.

    Creates 2 sessions on the same ``connection_id``, then calls
    ``close_all_sessions_for_connection()``. Both sessions must be
    popped from ``_acp_sessions`` and the ``connection_id`` must be
    popped from ``_connection_sessions``.
    """
    mock_pool: MagicMock = _make_mock_pool()
    manager: ACPSessionManager = ACPSessionManager(mock_pool)

    session_id_a: str = "ws-disc-session-a"
    session_id_b: str = "ws-disc-session-b"
    connection_id: str = "conn-1"

    # Create 2 mock sessions and register them manually
    session_a: AsyncMock = _make_mock_session(session_id_a)
    session_b: AsyncMock = _make_mock_session(session_id_b)

    manager._acp_sessions[session_id_a] = session_a
    manager._acp_sessions[session_id_b] = session_b
    manager._connection_sessions.setdefault(connection_id, set()).add(session_id_a)
    manager._connection_sessions[connection_id].add(session_id_b)

    # Verify pre-state
    assert len(manager._acp_sessions) == 2
    assert connection_id in manager._connection_sessions
    assert manager._connection_sessions[connection_id] == {session_id_a, session_id_b}

    # Simulate WebSocket disconnect
    await manager.close_all_sessions_for_connection(connection_id)

    # Both sessions should be popped from _acp_sessions
    assert session_id_a not in manager._acp_sessions
    assert session_id_b not in manager._acp_sessions
    assert len(manager._acp_sessions) == 0

    # connection_id should be popped from _connection_sessions
    assert connection_id not in manager._connection_sessions

    # Both sessions' close() should have been called
    session_a.close.assert_awaited_once()
    session_b.close.assert_awaited_once()

    # SessionPool.close_session should have been called for both
    # (via ACPSessionManager.close_session → SessionPool.close_session)
    mock_pool.session_pool.close_session.assert_awaited()
    assert mock_pool.session_pool.close_session.await_count == 2


@pytest.mark.integration
@pytest.mark.asyncio
async def test_websocket_disconnect_preserves_other_connections() -> None:
    """Disconnecting one WebSocket does not affect sessions on other connections.

    Creates sessions on 2 separate ``connection_id`` values, disconnects
    one connection, and verifies that only that connection's sessions
    are closed while the other connection's sessions remain alive.
    """
    mock_pool: MagicMock = _make_mock_pool()
    manager: ACPSessionManager = ACPSessionManager(mock_pool)

    session_id_a: str = "ws-disc-session-a"
    session_id_b: str = "ws-disc-session-b"
    conn_id_1: str = "conn-1"
    conn_id_2: str = "conn-2"

    # Create session A on conn-1, session B on conn-2
    session_a: AsyncMock = _make_mock_session(session_id_a)
    session_b: AsyncMock = _make_mock_session(session_id_b)

    manager._acp_sessions[session_id_a] = session_a
    manager._acp_sessions[session_id_b] = session_b
    manager._connection_sessions.setdefault(conn_id_1, set()).add(session_id_a)
    manager._connection_sessions.setdefault(conn_id_2, set()).add(session_id_b)

    # Verify pre-state
    assert len(manager._acp_sessions) == 2
    assert conn_id_1 in manager._connection_sessions
    assert conn_id_2 in manager._connection_sessions
    assert manager._connection_sessions[conn_id_1] == {session_id_a}
    assert manager._connection_sessions[conn_id_2] == {session_id_b}

    # Simulate WebSocket disconnect on conn-1 only
    await manager.close_all_sessions_for_connection(conn_id_1)

    # Session A should be closed (popped from _acp_sessions)
    assert session_id_a not in manager._acp_sessions

    # Session B should still be alive
    assert session_id_b in manager._acp_sessions
    assert manager._acp_sessions[session_id_b] is session_b

    # conn-1 should be popped from _connection_sessions
    assert conn_id_1 not in manager._connection_sessions

    # conn-2 should still be in _connection_sessions with session B's ID
    assert conn_id_2 in manager._connection_sessions
    assert manager._connection_sessions[conn_id_2] == {session_id_b}

    # Session A's close() should have been called
    session_a.close.assert_awaited_once()

    # Session B's close() should NOT have been called
    session_b.close.assert_not_awaited()

    # SessionPool.close_session should have been called only for session A
    mock_pool.session_pool.close_session.assert_awaited_once()
    call_args_list: list[tuple[str, ...]] = [
        call.args for call in mock_pool.session_pool.close_session.await_args_list
    ]
    assert (session_id_a,) in call_args_list
    assert (session_id_b,) not in call_args_list

    # Clean up session B
    await manager.close_all_sessions_for_connection(conn_id_2)
    assert session_id_b not in manager._acp_sessions
    assert conn_id_2 not in manager._connection_sessions
