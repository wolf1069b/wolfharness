"""Tests for ACPProtocolHandler cancel behavior.

Tests that cancel_session properly delegates to cancel_run_for_session
without calling fail() on the RunHandle. After the cancel-turn-not-run
fix, the event consumer is NOT stopped before cancel — it must stay
alive to deliver the RunFailedEvent (stop_reason="cancelled") to the
client.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from wolfharness_server.acp_server.event_converter import ACPEventConverter
from wolfharness_server.acp_server.handler import ACPProtocolHandler
from wolfharness_server.acp_server.session_manager import ACPSessionManager


if TYPE_CHECKING:
    from wolfharness import AgentPool


pytestmark = pytest.mark.unit


@pytest.fixture
def mock_session_manager() -> MagicMock:
    """Mock ACPSessionManager."""
    return MagicMock(spec=ACPSessionManager)


@pytest.fixture
def mock_event_converter() -> MagicMock:
    """Mock ACPEventConverter."""
    conv = MagicMock(spec=ACPEventConverter)
    conv.convert = AsyncMock()
    return conv


@pytest.fixture
def mock_client() -> MagicMock:
    """Mock ACP Client."""
    client = MagicMock()
    client.session_update = AsyncMock()
    return client


@pytest.fixture
def acp_handler(
    minimal_pool: AgentPool,
    mock_session_manager: MagicMock,
    mock_event_converter: MagicMock,
    mock_client: MagicMock,
) -> ACPProtocolHandler:
    """Return an ACPProtocolHandler with real pool context and mocked deps."""
    return ACPProtocolHandler(
        host_context=minimal_pool.get_context(),
        session_manager=mock_session_manager,
        event_converter=mock_event_converter,
        client=mock_client,
        client_capabilities=None,
    )


@pytest.mark.anyio
async def test_cancel_session_calls_cancel_run_for_session(
    acp_handler: ACPProtocolHandler,
    minimal_pool: AgentPool,
) -> None:
    """cancel_session must call cancel_run_for_session on the session pool.

    After the cancel-turn-not-run fix, the event consumer is NOT stopped
    before cancel — it must stay alive to deliver the RunFailedEvent
    (stop_reason="cancelled") to the client via session/update.
    """
    session_id = "test-session-123"
    assert minimal_pool.session_pool is not None

    with patch.object(
        minimal_pool.session_pool.sessions,
        "cancel_run_for_session",
        new_callable=MagicMock,
    ) as mock_cancel:
        # Call cancel_session
        await acp_handler.cancel_session(session_id)

        # Verify cancel_run_for_session was called
        mock_cancel.assert_called_once_with(session_id)


@pytest.mark.anyio
async def test_cancel_session_handles_no_running_consumer(
    acp_handler: ACPProtocolHandler,
    minimal_pool: AgentPool,
) -> None:
    """cancel_session should handle case where no consumer is running."""
    session_id = "test-session-456"
    assert minimal_pool.session_pool is not None

    assert session_id not in acp_handler._session_groups

    with patch.object(
        minimal_pool.session_pool.sessions,
        "cancel_run_for_session",
        new_callable=MagicMock,
    ) as mock_cancel:
        # Should not raise even with no consumer
        await acp_handler.cancel_session(session_id)

        # Verify cancel_run_for_session was still called
        mock_cancel.assert_called_once_with(session_id)


@pytest.mark.anyio
async def test_cancel_session_without_session_pool(
    acp_handler: ACPProtocolHandler,
    minimal_pool: AgentPool,
) -> None:
    """cancel_session should be a no-op when SessionPool is None."""
    # Create a HostContext with session_pool=None using dataclasses.replace
    host_ctx = minimal_pool.get_context()
    assert host_ctx is not None
    acp_handler._host_context = dataclasses.replace(host_ctx, session_pool=None)
    session_id = "test-session-789"

    # Should not raise
    await acp_handler.cancel_session(session_id)


@pytest.mark.anyio
async def test_cancel_session_does_not_call_fail_on_run_handle(
    acp_handler: ACPProtocolHandler,
    minimal_pool: AgentPool,
) -> None:
    """cancel_session must NOT call fail() on the RunHandle.

    After the cancel-turn-not-run fix, cancel uses _interrupt() + cancelled
    flag, not fail(). Calling fail() would publish RunFailedEvent with an
    exception, causing double TurnComplete in the ACP event converter.
    """
    session_id = "test-session-no-fail"
    assert minimal_pool.session_pool is not None

    # Set up a mock run handle that cancel_run_for_session would operate on
    mock_run_handle = MagicMock()

    with (
        patch.object(acp_handler, "stop_event_consumer", new_callable=AsyncMock),
        patch.object(
            minimal_pool.session_pool.sessions,
            "cancel_run_for_session",
            new_callable=MagicMock,
        ) as mock_cancel,
    ):
        # Simulate what the real cancel_run_for_session does: call cancel()
        # on the run handle (NOT fail()).
        def fake_cancel(sid: str) -> None:
            mock_run_handle.cancel()

        mock_cancel.side_effect = fake_cancel

        await acp_handler.cancel_session(session_id)

    # fail() must NOT be called — cancel uses _interrupt() + cancelled flag
    mock_run_handle.fail.assert_not_called()
    # cancel() SHOULD be called (via cancel_run_for_session)
    mock_run_handle.cancel.assert_called_once()


@pytest.mark.anyio
async def test_cancel_session_cancels_child_subagent_runs(
    acp_handler: ACPProtocolHandler,
    minimal_pool: AgentPool,
) -> None:
    """cancel_session must cancel runs for all child (subagent) sessions.

    When a parent session is cancelled, all in-flight subagent runs must
    also be cancelled.  This test verifies that ``cancel_run_for_session``
    is called for each child session in the ``_parent_of`` tree, not just
    the parent.
    """
    parent_sid = "parent-session"
    child1_sid = "child-session-1"
    child2_sid = "child-session-2"
    grandchild_sid = "grandchild-session"
    assert minimal_pool.session_pool is not None

    # Build a parent-child tree: parent → child1, child2; child1 → grandchild
    acp_handler._parent_of[child1_sid] = parent_sid
    acp_handler._parent_of[child2_sid] = parent_sid
    acp_handler._parent_of[grandchild_sid] = child1_sid

    cancelled_sids: list[str] = []

    def track_cancel(sid: str) -> None:
        cancelled_sids.append(sid)

    with patch.object(
        minimal_pool.session_pool.sessions,
        "cancel_run_for_session",
        new_callable=MagicMock,
        side_effect=track_cancel,
    ):
        await acp_handler.cancel_session(parent_sid)

    # All sessions in the subtree must be cancelled (depth-first order):
    # grandchild, child1, child2, then parent
    assert parent_sid in cancelled_sids, "Parent session must be cancelled"
    assert child1_sid in cancelled_sids, "Child session 1 must be cancelled"
    assert child2_sid in cancelled_sids, "Child session 2 must be cancelled"
    assert grandchild_sid in cancelled_sids, "Grandchild session must be cancelled"

    # Parent must be cancelled LAST (after all children)
    assert cancelled_sids[-1] == parent_sid, (
        f"Parent must be cancelled after all children, got order: {cancelled_sids}"
    )

    # _parent_of must NOT be modified (cancel ≠ close)
    assert acp_handler._parent_of.get(child1_sid) == parent_sid
    assert acp_handler._parent_of.get(child2_sid) == parent_sid
    assert acp_handler._parent_of.get(grandchild_sid) == child1_sid


@pytest.mark.anyio
async def test_cancel_session_no_children_only_cancels_parent(
    acp_handler: ACPProtocolHandler,
    minimal_pool: AgentPool,
) -> None:
    """cancel_session with no children should only cancel the parent."""
    session_id = "lonely-session"
    assert minimal_pool.session_pool is not None

    cancelled_sids: list[str] = []

    with patch.object(
        minimal_pool.session_pool.sessions,
        "cancel_run_for_session",
        new_callable=MagicMock,
        side_effect=cancelled_sids.append,
    ):
        await acp_handler.cancel_session(session_id)

    assert cancelled_sids == [session_id]
