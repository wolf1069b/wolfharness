"""Test: WebSocket disconnect during active run -> RunHandle cancelled with timeout.

Verifies that when ``close_all_sessions_for_connection()`` is called
(e.g. on WebSocket disconnect) for a session with an active run that
never completes, the RunHandle is cancelled via the
``SessionController._close_session_run_turn()`` timeout mechanism and
cleanup proceeds.

This is the end-to-end disconnect path (T25 + T26):
1. ``on_disconnect`` callback fires on WebSocket close.
2. ``ACPSessionManager.close_all_sessions_for_connection()`` is called.
3. ``SessionController.close_session()`` is called for each session.
4. ``_close_session_run_turn()`` handles RunHandle lifecycle (2s timeout
   + cancel).
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, Mock

import pytest

from wolfharness.lifecycle import RunState
from wolfharness.orchestrator.run import RunHandle
from wolfharness.orchestrator.session_controller import SessionController, SessionState
from wolfharness_server.acp_server.session_manager import ACPSessionManager


@pytest.mark.integration
@pytest.mark.asyncio
async def test_websocket_disconnect_during_run() -> None:
    """close_all_sessions_for_connection cancels unresponsive RunHandle.

    Simulates a WebSocket disconnect while a run is active and never
    sets ``complete_event`` (hung run). The disconnect path should:
    1. Call ``SessionController.close_session()`` (via the controller).
    2. ``_close_session_run_turn()`` acquires ``turn_lock``.
    3. Waits 2s for ``complete_event`` (timeout).
    4. Calls ``run_handle.cancel()``.
    5. Removes session from ``_sessions`` and ``_acp_sessions``.
    """
    mock_pool = Mock()
    controller = SessionController(pool=mock_pool)

    session_id = "test-ws-disconnect-active-run"
    run_id = "run-hung-on-disconnect"
    connection_id = "conn-uuid-hex-1234"

    session = SessionState(
        session_id=session_id,
        agent_name="test_agent",
    )
    session.current_run_id = run_id
    controller._sessions[session_id] = session

    run_handle = RunHandle(
        run_id=run_id,
        session_id=session_id,
        agent_type="native",
    )
    run_handle._run_state = RunState.RUNNING
    controller._runs[run_id] = run_handle

    mock_acp_session = Mock()
    mock_acp_session.close = AsyncMock()

    session_manager = ACPSessionManager(pool=mock_pool)
    mock_pool.session_pool = Mock()
    mock_pool.session_pool.sessions = controller

    async def _close_via_controller(sid: str) -> None:
        await controller.close_session(sid)

    mock_pool.session_pool.close_session = _close_via_controller
    session_manager._acp_sessions[session_id] = mock_acp_session
    session_manager._connection_sessions[connection_id] = {session_id}

    await asyncio.wait_for(
        session_manager.close_all_sessions_for_connection(connection_id),
        timeout=30.0,
    )

    assert run_handle.run_ctx.cancelled is True
    assert session_id not in controller._sessions
    assert session_id not in session_manager._acp_sessions
    assert connection_id not in session_manager._connection_sessions
    mock_acp_session.close.assert_awaited_once()
