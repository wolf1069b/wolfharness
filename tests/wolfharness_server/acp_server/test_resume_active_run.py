"""Test: resume with active run - RunHandle cancelled with timeout.

Verifies that when ``close_session()`` is called on a session with an
active run that never completes, the RunHandle is cancelled via the
timeout mechanism and cleanup proceeds.

This mirrors the ``resume_session()`` flow (T20) which calls
``SessionController.close_session()`` before recreating the session.
"""

from __future__ import annotations

import asyncio
from unittest.mock import Mock

import pytest

from wolfharness.lifecycle import RunState
from wolfharness.orchestrator.run import RunHandle
from wolfharness.orchestrator.session_controller import SessionController, SessionState


@pytest.mark.integration
@pytest.mark.asyncio
async def test_resume_with_active_run() -> None:
    """close_session cancels an unresponsive RunHandle with timeout.

    Simulates a run that never sets ``complete_event`` (hung run).
    ``close_session`` should:
    1. Call ``run_handle.close()`` (signal stop).
    2. Acquire ``turn_lock`` (succeeds - nothing holds it).
    3. Wait for ``complete_event`` with 2s timeout.
    4. On timeout, call ``run_handle.cancel()``.
    5. Proceed to remove session from ``_sessions``.
    """
    mock_pool = Mock()
    controller = SessionController(pool=mock_pool)

    session_id = "test-resume-active-run"
    run_id = "run-never-completes"

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
    # Simulate a running run: status is running, complete_event NOT set.
    run_handle._run_state = RunState.RUNNING
    controller._runs[run_id] = run_handle

    # close_session should complete within 30s (internal 2s timeout + cleanup).
    await asyncio.wait_for(controller.close_session(session_id), timeout=30.0)

    # RunHandle was cancelled.
    assert run_handle.run_ctx.cancelled is True

    # Session was cleaned up.
    assert session_id not in controller._sessions

    # RunHandle was removed from _runs via _cleanup_run or remains but
    # the session is gone — the key assertion is that close_session
    # returned without hanging.
    # complete_event may not be set because cancel() doesn't set it
    # (only complete()/fail()/_cleanup_run does). The important thing
    # is close_session didn't block forever.
