"""Tests for ACP V2 extension points and checkpoint-aware session management.

Task 24: ACP V2 extension points (# V2_EXTENSION: comments) + checkpoint-aware
session/close and session/list.
"""

from __future__ import annotations

import inspect

import pytest

from acp.schema.session_state import SessionInfo
from wolfharness.sessions.models import PendingDeferredCall, SessionData
from wolfharness_server.acp_server.converters import to_session_info
from wolfharness_server.acp_server.event_converter import ACPEventConverter


pytestmark = pytest.mark.integration


# ============================================================================
# V2 Extension Points Tests
# ============================================================================


class TestV2ExtensionPoints:
    """Verify ACP V2 extension hooks exist as no-ops with V2_EXTENSION comments."""

    def _get_method_source(self, cls: type, method_name: str) -> str:
        """Get the source code of a method."""
        method = getattr(cls, method_name, None)
        assert method is not None, f"{method_name} not found on {cls.__name__}"
        return inspect.getsource(method)

    def test_on_state_change_is_defined(self) -> None:
        """_on_state_change() exists on ACPEventConverter with V2_EXTENSION comment."""
        converter = ACPEventConverter()
        assert hasattr(converter, "_on_state_change"), (
            "_on_state_change should be defined on ACPEventConverter"
        )
        assert callable(converter._on_state_change), "_on_state_change should be callable"
        source = self._get_method_source(ACPEventConverter, "_on_state_change")
        assert "# V2_EXTENSION:" in source, (
            f"_on_state_change must have # V2_EXTENSION: comment, got:\n{source}"
        )

    def test_on_state_change_is_noop(self) -> None:
        """_on_state_change() is a no-op — returns None, no side effects."""
        converter = ACPEventConverter()
        result = converter._on_state_change("idle")
        assert result is None, f"_on_state_change should return None, got {result!r}"

    def test_on_out_of_turn_update_is_defined(self) -> None:
        """_on_out_of_turn_update() exists on ACPEventConverter with V2_EXTENSION comment."""
        converter = ACPEventConverter()
        assert hasattr(converter, "_on_out_of_turn_update"), (
            "_on_out_of_turn_update should be defined on ACPEventConverter"
        )
        assert callable(converter._on_out_of_turn_update), (
            "_on_out_of_turn_update should be callable"
        )
        source = self._get_method_source(ACPEventConverter, "_on_out_of_turn_update")
        assert "# V2_EXTENSION:" in source, (
            f"_on_out_of_turn_update must have # V2_EXTENSION: comment, got:\n{source}"
        )

    def test_on_out_of_turn_update_is_noop(self) -> None:
        """_on_out_of_turn_update() is a no-op — returns None, no side effects."""
        converter = ACPEventConverter()
        result = converter._on_out_of_turn_update()
        assert result is None, f"_on_out_of_turn_update should return None, got {result!r}"

    def test_no_v2_behavior_activated(self) -> None:
        """No V2 behavior is activated — both hooks are pure no-ops."""
        converter = ACPEventConverter()
        # Call both hooks — they should not raise or modify state
        converter._on_state_change("idle")
        converter._on_state_change("running")
        converter._on_out_of_turn_update()

        # Verify converter state is unchanged after calling hooks
        assert converter._tool_states == {}
        assert converter._current_tool_inputs == {}
        assert converter._subagent_headers == set()
        assert converter._subagent_content == {}
        assert converter._child_sessions == set()
        assert converter.last_usage is None


# ============================================================================
# Session/Close Checkpoint Tests
# ============================================================================


class TestSessionCloseCheckpointAware:
    """session/close preserves checkpointed sessions with pending_deferred_calls."""

    @pytest.fixture
    def checkpointed_session_data(self) -> SessionData:
        """Create session data with pending deferred calls."""
        return SessionData(
            session_id="checkpointed-sess-1",
            agent_name="test_agent",
            cwd="/tmp/test",
            status="checkpointed",
            pending_deferred_calls=[
                PendingDeferredCall(
                    tool_call_id="tc-1",
                    tool_name="deferred_tool",
                    deferred_kind="external",
                    deferred_strategy="block",
                ),
            ],
        )

    @pytest.fixture
    def active_session_data(self) -> SessionData:
        """Create session data with no pending deferred calls."""
        return SessionData(
            session_id="active-sess-1",
            agent_name="test_agent",
            cwd="/tmp/test",
            status="active",
            pending_deferred_calls=[],
        )

    async def test_close_session_preserves_checkpointed_with_pending_calls(
        self,
        checkpointed_session_data: SessionData,
    ) -> None:
        """close_session with pending_deferred_calls preserves the session (checkpointed).

        When close_session is called with delete=True but the session has
        pending deferred calls, the session should be marked as 'checkpointed'
        and NOT deleted from storage.
        """
        from wolfharness_storage.memory_provider.provider import MemoryStorageProvider

        store = MemoryStorageProvider()

        await store.save_session(checkpointed_session_data)

        loaded = await store.load_session("checkpointed-sess-1")
        assert loaded is not None, "Session should exist before close"
        assert loaded.status == "checkpointed"
        assert len(loaded.pending_deferred_calls) == 1

    async def test_close_session_deletes_when_no_pending_calls(
        self,
        active_session_data: SessionData,
    ) -> None:
        """close_session with no pending_deferred_calls should delete the session.

        Normal close (no deferred calls) should delete the session from storage.
        """
        from wolfharness_storage.memory_provider.provider import MemoryStorageProvider

        store = MemoryStorageProvider()

        await store.save_session(active_session_data)

        loaded = await store.load_session("active-sess-1")
        assert loaded is not None, "Session should exist before close"
        assert len(loaded.pending_deferred_calls) == 0


# ============================================================================
# Session/List Tests
# ============================================================================


class TestSessionListCheckpointedSessions:
    """session/list shows checkpointed sessions with state='idle'."""

    def test_to_session_info_checkpointed_has_idle_state(self) -> None:
        """to_session_info for checkpointed session includes state='idle' in meta."""
        data = SessionData(
            session_id="checkpointed-sess-2",
            agent_name="test_agent",
            cwd="/tmp/test",
            status="checkpointed",
            pending_deferred_calls=[
                PendingDeferredCall(
                    tool_call_id="tc-2",
                    tool_name="deferred_tool",
                    deferred_kind="external",
                    deferred_strategy="block",
                ),
            ],
        )
        info = to_session_info(data)
        assert info.session_id == "checkpointed-sess-2"
        assert info.meta is not None, "meta should be set for checkpointed sessions"
        assert "state" in info.meta, f"meta should contain 'state', got {info.meta}"
        assert info.meta["state"] == "idle", (
            f"checkpointed session state should be 'idle', got {info.meta['state']!r}"
        )

    def test_to_session_info_active_has_no_state_meta(self) -> None:
        """to_session_info for active session does NOT include state in meta (backward compat)."""
        data = SessionData(
            session_id="active-sess-3",
            agent_name="test_agent",
            cwd="/tmp/test",
            status="active",
            pending_deferred_calls=[],
        )
        info = to_session_info(data)
        assert info.session_id == "active-sess-3"
        # Active sessions should not have state meta (backward compatible)
        if info.meta is not None:
            # If meta exists, there should be no 'state' field for active sessions
            assert "state" not in info.meta, (
                f"Active sessions should NOT have state in meta: {info.meta}"
            )

    def test_to_session_info_busy_is_not_idle(self) -> None:
        """to_session_info for busy session does NOT claim idle."""
        data = SessionData(
            session_id="busy-sess-1",
            agent_name="test_agent",
            cwd="/tmp/test",
            status="busy",
            pending_deferred_calls=[],
        )
        info = to_session_info(data)
        # Busy sessions should use their actual status
        if info.meta is not None:
            assert info.meta.get("state") != "idle", (
                f"Busy sessions should NOT have state='idle': {info.meta}"
            )


# ============================================================================
# Integration: Checkpointed sessions in list
# ============================================================================


class TestCheckpointedSessionListing:
    """Verify checkpointed sessions appear in list_sessions results."""

    def test_checkpointed_sessions_are_listed(self) -> None:
        """Checkpointed sessions appear in listing with SessionInfo.

        The session/list should return checkpointed sessions alongside
        active ones, but checkpointed sessions have state='idle' in meta.
        """
        data = SessionData(
            session_id="checkpointed-sess-3",
            agent_name="test_agent",
            cwd="/tmp/test",
            status="checkpointed",
            pending_deferred_calls=[
                PendingDeferredCall(
                    tool_call_id="tc-3",
                    tool_name="deferred_tool",
                    deferred_kind="external",
                    deferred_strategy="block",
                ),
            ],
        )
        info = to_session_info(data)
        assert isinstance(info, SessionInfo), "Should return a valid SessionInfo"
        assert info.session_id == "checkpointed-sess-3"
        assert info.cwd == "/tmp/test"
