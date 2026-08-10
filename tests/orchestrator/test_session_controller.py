"""Unit tests for SessionController (SessionPool Group 2.11).

Tests session lifecycle, TTL cleanup, per-session agent creation,
and MCP process limit enforcement.
"""

from __future__ import annotations

import asyncio
import inspect
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tests._controller_helpers import send_via_controller
from wolfharness.lifecycle.types import DeliveryMode
from wolfharness.orchestrator.core import (
    DEFAULT_SESSION_TTL_SECONDS,
    EventBus,
    SessionController,
    SessionState,
)
from wolfharness.orchestrator.run import RunHandle


if TYPE_CHECKING:
    from wolfharness import AgentPool


pytestmark = pytest.mark.unit

_L2_SKIP = pytest.mark.skip(reason="L2 migration: requires mock internals — remains L1 unit test")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def controller(minimal_pool: AgentPool) -> SessionController:
    """Return a SessionController backed by the real pool."""
    assert minimal_pool.session_pool is not None
    return minimal_pool.session_pool.sessions


@pytest.fixture
def mock_native_agent() -> MagicMock:
    """Return a mocked BaseAgent that looks like a native agent."""
    agent = MagicMock()
    agent.__aenter__ = AsyncMock(return_value=agent)
    agent.__aexit__ = AsyncMock(return_value=None)
    return agent


# ---------------------------------------------------------------------------
# get_or_create_session
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_get_or_create_session_creates_new(
    controller: SessionController,
) -> None:
    """A new session is created when the session_id is unknown."""
    state, was_created = await controller.get_or_create_session("sess-1", agent_name="agent-a")
    assert isinstance(state, SessionState)
    assert was_created is True
    assert state.session_id == "sess-1"
    assert state.agent_name == "agent-a"
    assert state.closed_at is None
    assert state.is_closing is False


@pytest.mark.anyio
async def test_get_or_create_session_returns_existing(
    controller: SessionController,
) -> None:
    """Calling get_or_create_session with the same ID returns the existing state."""
    first, first_created = await controller.get_or_create_session("sess-1", agent_name="agent-a")
    second, second_created = await controller.get_or_create_session("sess-1")
    assert first is second
    assert first_created is True
    assert second_created is False


@pytest.mark.anyio
async def test_get_or_create_session_updates_last_active(
    controller: SessionController,
) -> None:
    """last_active_at is refreshed when an existing session is retrieved."""
    state, _ = await controller.get_or_create_session("sess-1")
    old_ts = state.last_active_at
    await asyncio.sleep(0.01)
    state2, _ = await controller.get_or_create_session("sess-1")
    assert state2.last_active_at > old_ts


@pytest.mark.anyio
async def test_get_or_create_session_stores_metadata(
    controller: SessionController,
) -> None:
    """Arbitrary keyword metadata is stored on the session state."""
    state, _ = await controller.get_or_create_session("sess-1", user_id="u42")
    assert state.metadata == {"user_id": "u42"}


# ---------------------------------------------------------------------------
# list_sessions
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_list_sessions_returns_session_info(
    controller: SessionController,
) -> None:
    """list_sessions returns SessionInfo DTOs for all active sessions."""
    from wolfharness_server.opencode_server.models.session_info import SessionInfo

    await controller.get_or_create_session("sess-1", agent_name="agent-a")
    await controller.get_or_create_session("sess-2", agent_name="agent-b")

    infos = controller.list_sessions()

    assert len(infos) == 2
    assert all(isinstance(info, SessionInfo) for info in infos)
    assert {info.session_id for info in infos} == {"sess-1", "sess-2"}
    assert {info.agent_name for info in infos} == {"agent-a", "agent-b"}
    assert all(info.status == "idle" for info in infos)
    assert all(not info.is_per_session_agent for info in infos)


# ---------------------------------------------------------------------------
# get_or_create_session_agent - per-session native agent
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_get_or_create_session_agent_creates_native_agent(
    controller: SessionController,
    minimal_pool: AgentPool,
) -> None:
    """Per-session agent creation returns a real agent from the pool."""
    agent = await controller.get_or_create_session_agent("sess-1", agent_name="test_agent")

    assert agent is not None
    assert agent.AGENT_TYPE == "native"
    state = controller.get_session("sess-1")
    assert state is not None
    assert state.is_per_session_agent is True


@pytest.mark.anyio
async def test_get_or_create_session_agent_returns_existing_agent(
    controller: SessionController,
    minimal_pool: AgentPool,
) -> None:
    """A second call returns the cached per-session agent."""
    first = await controller.get_or_create_session_agent("sess-1", agent_name="test_agent")
    second = await controller.get_or_create_session_agent("sess-1", agent_name="test_agent")

    assert first is second


# ---------------------------------------------------------------------------
# MCP process limits
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_mcp_count_incremented_and_decremented(
    controller: SessionController,
    minimal_pool: AgentPool,
) -> None:
    """MCP count tracks per-session agent creation and destruction."""
    assert controller._mcp_process_count == 0
    await controller.get_or_create_session_agent("sess-1", agent_name="test_agent")
    assert controller._mcp_process_count == 1

    await controller.close_session("sess-1")
    assert controller._mcp_process_count == 0


@pytest.mark.anyio
async def test_mcp_count_never_negative(
    controller: SessionController,
) -> None:
    """_decrement_mcp_count clamps at zero."""
    controller._decrement_mcp_count(MagicMock())
    assert controller._mcp_process_count == 0


# ---------------------------------------------------------------------------
# close_session
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_close_session_removes_session(
    controller: SessionController,
) -> None:
    """After close_session, the session is no longer retrievable."""
    await controller.get_or_create_session("sess-1")
    await controller.close_session("sess-1")
    assert controller.get_session("sess-1") is None


@pytest.mark.anyio
async def test_close_session_is_idempotent(
    controller: SessionController,
) -> None:
    """Closing a session twice does not raise."""
    await controller.get_or_create_session("sess-1")
    await controller.close_session("sess-1")
    await controller.close_session("sess-1")  # should not raise
    assert controller.get_session("sess-1") is None


@pytest.mark.anyio
async def test_close_session_sets_closing_flag(
    controller: SessionController,
) -> None:
    """close_session marks the session as closing and records closed_at."""
    state, _ = await controller.get_or_create_session("sess-1")
    await controller.close_session("sess-1")
    # closed_at is set inside close_session
    assert state.closed_at is not None
    assert state.is_closing is True


@_L2_SKIP
@pytest.mark.anyio
async def test_close_session_exits_per_session_agent(
    controller: SessionController,
    minimal_pool: AgentPool,
    mock_native_agent: MagicMock,
) -> None:
    """A per-session agent has its async context exited on close."""
    cfg = MagicMock()
    cfg.name = "agent-a"
    minimal_pool.manifest.agents = {"agent-a": cfg}
    minimal_pool._factory.create_session_agent = AsyncMock(return_value=mock_native_agent)

    await controller.get_or_create_session_agent("sess-1", agent_name="agent-a")

    await controller.close_session("sess-1")
    mock_native_agent.__aexit__.assert_awaited_once()


@pytest.mark.anyio
async def test_cleanup_expired_sessions_calls_callback(
    minimal_pool: AgentPool,
) -> None:
    """The optional cleanup_callback is invoked for expired sessions."""
    callback = AsyncMock()
    ctrl = SessionController(pool=minimal_pool, cleanup_callback=callback)
    ctrl._session_ttl_seconds = 0.05
    await ctrl.get_or_create_session("sess-1")
    await asyncio.sleep(0.1)
    await ctrl._cleanup_expired_sessions()
    callback.assert_awaited_once_with("sess-1")


# ---------------------------------------------------------------------------
# TTL cleanup
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_cleanup_task_closes_expired_sessions(
    controller: SessionController,
) -> None:
    """Background cleanup closes sessions whose TTL has expired."""
    controller._session_ttl_seconds = 0.05
    await controller.get_or_create_session("sess-1")
    await asyncio.sleep(0.1)
    await controller._cleanup_expired_sessions()
    assert controller.get_session("sess-1") is None


@pytest.mark.anyio
async def test_cleanup_task_keeps_active_sessions(
    controller: SessionController,
) -> None:
    """Active sessions (within TTL) are not closed by cleanup."""
    controller._session_ttl_seconds = 10.0
    await controller.get_or_create_session("sess-1")
    await controller._cleanup_expired_sessions()
    assert controller.get_session("sess-1") is not None


@pytest.mark.anyio
async def test_cleanup_task_uses_callback_when_provided(
    minimal_pool: AgentPool,
) -> None:
    """When cleanup_callback is set, it is used instead of close_session."""
    callback = AsyncMock()
    ctrl = SessionController(pool=minimal_pool, cleanup_callback=callback)
    ctrl._session_ttl_seconds = 0.05
    await ctrl.get_or_create_session("sess-1")
    await asyncio.sleep(0.1)
    await ctrl._cleanup_expired_sessions()
    callback.assert_awaited_once_with("sess-1")


@pytest.mark.anyio
async def test_start_and_stop_cleanup_task(
    minimal_pool: AgentPool,
) -> None:
    """start_cleanup_task and stop_cleanup_task manage the background task."""
    # Create a fresh SessionController that hasn't started its cleanup task yet
    controller = SessionController(pool=minimal_pool)
    assert controller._cleanup_task is None
    await controller.start_cleanup_task()
    assert controller._cleanup_task is not None
    await controller.stop_cleanup_task()
    assert controller._cleanup_task is None


@pytest.mark.anyio
async def test_cleanup_loop_catches_exceptions(
    minimal_pool: AgentPool,
) -> None:
    """The cleanup loop survives exceptions and continues running."""
    controller = SessionController(pool=minimal_pool)
    controller._session_ttl_seconds = 0.01
    await controller.start_cleanup_task()
    # Force an exception by corrupting internal state
    with patch.object(controller, "_cleanup_expired_sessions", side_effect=RuntimeError("boom")):
        await asyncio.sleep(0.03)
    await controller.stop_cleanup_task()


# ---------------------------------------------------------------------------
# get_session
# ---------------------------------------------------------------------------


def test_get_session_returns_none_for_unknown(controller: SessionController) -> None:
    """get_session returns None when the session does not exist."""
    assert controller.get_session("missing") is None


def test_get_session_returns_state(controller: SessionController) -> None:
    """get_session returns the SessionState for an existing session."""
    # Note: using the sync variant via internal dict for simplicity
    state = SessionState(session_id="sess-1", agent_name="a")
    controller._sessions["sess-1"] = state
    assert controller.get_session("sess-1") is state


# ---------------------------------------------------------------------------
# Default TTL constant
# ---------------------------------------------------------------------------


def test_default_ttl_is_one_hour() -> None:
    """The default session TTL is 3600 seconds."""
    assert DEFAULT_SESSION_TTL_SECONDS == 3600.0


# ---------------------------------------------------------------------------
# receive_request
# ---------------------------------------------------------------------------
# Either idle (run completed quickly) or exactly one active run
# Because run_loop is mocked, it returns immediately, so the run
# may already be cleaned up.


# ---------------------------------------------------------------------------
# cancel_run_for_session
# ---------------------------------------------------------------------------
def test_cancel_run_for_session_noop_for_idle_session(
    controller: SessionController,
) -> None:
    """cancel_run_for_session is a no-op when the session has no active run."""
    # No session exists - should not raise
    controller.cancel_run_for_session("missing")


def test_cancel_run_for_session_noop_for_missing_run(
    controller: SessionController,
) -> None:
    """cancel_run_for_session is a no-op when current_run_id is set but run is missing."""
    state = SessionState(session_id="sess-1", agent_name="a")
    state.current_run_id = "ghost-run"
    controller._sessions["sess-1"] = state
    controller.cancel_run_for_session("sess-1")


# ---------------------------------------------------------------------------
# _cleanup_run
# ---------------------------------------------------------------------------
def test_cleanup_run_noop_for_missing_run(controller: SessionController) -> None:
    """_cleanup_run is a no-op when the run_id is unknown."""
    controller._cleanup_run("ghost")  # should not raise


def test_cleanup_run_clears_current_run_id(
    controller: SessionController,
) -> None:
    """_cleanup_run clears session.current_run_id when it matches the run_id."""
    state = SessionState(session_id="sess-cleanup", agent_name="a")
    controller._sessions["sess-cleanup"] = state

    run_handle = RunHandle(
        run_id="run-123",
        session_id="sess-cleanup",
        agent_type="native",
    )
    controller._runs["run-123"] = run_handle
    state.current_run_id = "run-123"

    controller._cleanup_run("run-123")

    assert state.current_run_id is None
    assert "run-123" not in controller._runs


# ---------------------------------------------------------------------------
# SessionState.closing alias
# ---------------------------------------------------------------------------


def test_closing_alias_reads_is_closing() -> None:
    """The closing property returns the value of is_closing."""
    state = SessionState(session_id="s", agent_name="a")
    assert state.closing is False
    state.is_closing = True
    assert state.closing is True


def test_closing_alias_writes_is_closing() -> None:
    """Setting closing updates is_closing."""
    state = SessionState(session_id="s", agent_name="a")
    state.closing = True
    assert state.is_closing is True
    assert state.closing is True


# ---------------------------------------------------------------------------
# Tests from PR #64 review (SessionController internals)
# ---------------------------------------------------------------------------


@_L2_SKIP
@pytest.mark.asyncio
async def test_steer_followup_inside_request_lock(minimal_pool: AgentPool) -> None:
    """steer()/followup() must be called inside _request_lock.

    Without this, current_run_id can be cleared between the check and
    the steer()/followup() call, causing silent message drops.
    """
    from wolfharness.orchestrator.core import SessionController

    controller = SessionController(pool=minimal_pool)
    controller._event_bus = EventBus()

    mock_agent = MagicMock()
    mock_agent.AGENT_TYPE = "native"

    session_id = "sess-toctou"
    controller._sessions[session_id] = MagicMock()
    controller._sessions[session_id].session_id = session_id
    controller._sessions[session_id].current_run_id = "fake-run-id"
    controller._sessions[session_id].closing = False
    controller._sessions[session_id].is_closing = False
    controller._sessions[session_id]._request_lock = asyncio.Lock()
    controller._sessions[session_id].turn_lock = asyncio.Lock()
    controller._sessions[session_id].input_provider = None
    controller._sessions[session_id].is_per_session_agent = False
    controller._session_agents[session_id] = mock_agent

    # Track if steer is called while lock is held
    lock_was_held_during_steer = False
    fake_run = MagicMock()
    lock = controller._sessions[session_id]._request_lock

    def _check_lock_and_steer(content: str, *, message_id: str | None = None) -> None:
        nonlocal lock_was_held_during_steer
        lock_was_held_during_steer = lock.locked()

    fake_run.steer = _check_lock_and_steer
    fake_run.followup = MagicMock()
    fake_run._run_state = MagicMock()  # Not RunState.DONE
    controller._runs["fake-run-id"] = fake_run

    await send_via_controller(controller, session_id, "steer me", mode=DeliveryMode.STEER)

    assert lock_was_held_during_steer, (
        "steer() was called outside _request_lock — TOCTOU race possible"
    )


def test_background_task_strong_reference() -> None:
    """_start_run_handle must keep strong reference to background task.

    Without a strong reference, Python's GC can destroy the task mid-execution.
    """
    import wolfharness.orchestrator.core as core_module

    source = inspect.getsource(core_module.SessionController._start_run_handle)
    assert "_background_tasks" in source, (
        "_start_run_handle must store task in _background_tasks set "
        "to prevent GC from destroying it mid-execution"
    )
    assert "add_done_callback" in source, (
        "task must have done callback to discard from _background_tasks"
    )


def test_closing_property_sets_is_closing() -> None:
    """session.closing = True already sets session.is_closing = True.

    The `closing` property is an alias for `is_closing` — its setter
    writes to `self.is_closing`. So setting `session.closing = True`
    is equivalent to setting `session.is_closing = True`.
    """
    session = SessionState(
        session_id="test-property",
        agent_name="test",
    )
    assert session.is_closing is False
    assert session.closing is False

    session.closing = True
    assert session.is_closing is True, (
        "Setting session.closing = True should also set session.is_closing = True "
        "via the property setter"
    )
    assert session.closing is True


# ---------------------------------------------------------------------------
# _background_tasks initialization (from PR #64 round-7 review)
# ---------------------------------------------------------------------------


def test_background_tasks_initialized_in_init(minimal_pool: AgentPool) -> None:
    """SessionController.__init__ must initialize _background_tasks set.

    Without early initialization, the first call to _start_run_handle
    hits a hasattr check that could mask bugs.
    """
    from wolfharness.orchestrator.core import SessionController

    controller = SessionController(pool=minimal_pool)
    assert hasattr(controller, "_background_tasks"), (
        "_background_tasks must be initialized in __init__"
    )
    assert isinstance(controller._background_tasks, set), "_background_tasks must be a set"


def test_background_task_callback_is_named_function() -> None:
    """_start_run_handle must use a named callback, not a lambda tuple hack.

    Lambda tuples like `lambda _: (a(), b())` are fragile and hard to debug.
    """
    import re

    import wolfharness.orchestrator.core as core_module

    source = inspect.getsource(core_module.SessionController._start_run_handle)
    # Should NOT contain the lambda tuple pattern
    lambda_tuple = re.findall(r"lambda.*:\s*\(.*,\s*.*\)", source)
    assert len(lambda_tuple) == 0, (
        f"_start_run_handle uses lambda tuple hack: {lambda_tuple}. "
        "Use a named callback function instead."
    )
    # Should contain a def callback
    assert "def _on_run_done" in source or "add_done_callback(" in source, (
        "_start_run_handle should use a named callback function"
    )
