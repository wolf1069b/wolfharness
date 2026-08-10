"""Tests for deferred call timeout detection.

Verifies that SessionController._check_expired_calls correctly identifies
expired pending deferred calls based on their timeout and creation time.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from wolfharness.orchestrator.core import SessionController
from wolfharness.sessions.models import PendingDeferredCall, SessionData


pytestmark = pytest.mark.unit


def test_pending_deferred_call_expiry_detection() -> None:
    """A PendingDeferredCall with timeout=0 should be detected as expired."""
    long_ago = datetime.now() - timedelta(hours=1)
    call = PendingDeferredCall(
        tool_call_id="call_1",
        tool_name="bash",
        deferred_kind="external",
        deferred_strategy="block",
        created_at=long_ago,
        timeout=timedelta(seconds=0),
    )
    session_data = SessionData(
        session_id="test_session",
        agent_name="test_agent",
        pending_deferred_calls=[call],
    )
    expired = SessionController._check_expired_calls(session_data)
    assert len(expired) == 1
    assert expired[0].tool_call_id == "call_1"


def test_pending_deferred_call_not_expired_when_within_timeout() -> None:
    """A PendingDeferredCall within its timeout should NOT be detected as expired."""
    now = datetime.now()
    call = PendingDeferredCall(
        tool_call_id="call_2",
        tool_name="bash",
        deferred_kind="external",
        deferred_strategy="block",
        created_at=now,
        timeout=timedelta(hours=24),
    )
    session_data = SessionData(
        session_id="test_session",
        agent_name="test_agent",
        pending_deferred_calls=[call],
    )
    expired = SessionController._check_expired_calls(session_data)
    assert len(expired) == 0


def test_pending_deferred_call_no_timeout_never_expires() -> None:
    """A PendingDeferredCall with timeout=None should never be detected as expired."""
    long_ago = datetime.now() - timedelta(days=365)
    call = PendingDeferredCall(
        tool_call_id="call_3",
        tool_name="subagent",
        deferred_kind="external",
        deferred_strategy="continue",
        created_at=long_ago,
        timeout=None,
    )
    session_data = SessionData(
        session_id="test_session",
        agent_name="test_agent",
        pending_deferred_calls=[call],
    )
    expired = SessionController._check_expired_calls(session_data)
    assert len(expired) == 0
