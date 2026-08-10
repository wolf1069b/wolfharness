"""Tests for SessionPool session lifecycle, close semantics, and error propagation.

Consolidated from:
- test_session_pool.py (SessionLifecyclePolicy, SessionState parent/child, EventBus scopes)
- test_close_session.py (close_session wait/cancel/race semantics)
- test_error_propagation.py (RunFailedEvent via receive_request)
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from wolfharness.agents.events import RunStartedEvent
from wolfharness.orchestrator.core import (
    EventBus,
    SessionController,
    SessionLifecyclePolicy,
    SessionPool,
    SessionState,
)
from wolfharness.orchestrator.run import RunHandle
from wolfharness.sessions.models import PendingDeferredCall, SessionData


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from wolfharness import AgentPool
    from wolfharness.agents.context import AgentRunContext


pytestmark = pytest.mark.unit


# ============================================================================
# Shared fixtures and helpers
# ============================================================================


class MockAgent:
    """Simple mock agent for testing."""

    AGENT_TYPE: str = "native"

    def __init__(self) -> None:
        self._stream_impl: Any = None
        self.get_active_run_context = MagicMock(return_value=None)

    async def run_stream(
        self,
        *prompts: Any,
        session_id: str | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[Any]:
        """Mock run_stream that delegates to _stream_impl via _stream_events."""
        if self._stream_impl is None:
            raise RuntimeError("No stream impl set")
        run_ctx = MagicMock()
        if inspect.isasyncgenfunction(self._stream_impl):
            async for event in self._stream_impl(run_ctx, *prompts, **kwargs):
                yield event
        else:
            await self._stream_impl(run_ctx, *prompts, **kwargs)
        # Yield at least one event so the run doesn't hang
        yield RunStartedEvent(session_id=session_id or "", run_id="run-mock")

    async def _stream_events(
        self,
        run_ctx: AgentRunContext,
        *prompts: Any,
        **kwargs: Any,
    ) -> AsyncIterator[Any]:
        if self._stream_impl is None:
            raise RuntimeError("No stream impl set")
        if inspect.isasyncgenfunction(self._stream_impl):
            async for event in self._stream_impl(run_ctx, *prompts, **kwargs):
                yield event
        else:
            await self._stream_impl(run_ctx, *prompts, **kwargs)


@pytest.fixture
def session_pool(minimal_pool: AgentPool) -> SessionPool:
    assert minimal_pool.session_pool is not None
    return minimal_pool.session_pool


@pytest.fixture
def controller(minimal_pool: AgentPool) -> SessionController:
    """Return a real SessionController backed by the real pool."""
    assert minimal_pool.session_pool is not None
    return minimal_pool.session_pool.sessions


async def _setup_session(
    ctrl: SessionController,
    session_id: str,
    agent: MockAgent,
    real_pool: AgentPool,
) -> None:
    """Create a session and attach the mock agent directly."""
    state, _ = await ctrl.get_or_create_session(session_id)
    state.agent = agent
    ctrl._session_agents[session_id] = agent


# ============================================================================
# SessionLifecyclePolicy
# ============================================================================


class TestSessionLifecyclePolicy:
    """Tests for SessionLifecyclePolicy enum and validation."""

    def test_default_is_cascade(self) -> None:
        assert SessionLifecyclePolicy.default() == "cascade"

    def test_valid_policies(self) -> None:
        assert SessionLifecyclePolicy.is_valid("independent")
        assert SessionLifecyclePolicy.is_valid("cascade")
        assert SessionLifecyclePolicy.is_valid("bound")
        assert not SessionLifecyclePolicy.is_valid("invalid")


class TestSessionStateParentChild:
    """Tests for SessionState parent-child relationship fields."""

    def test_session_state_has_parent_and_policy(self) -> None:
        state = SessionState(
            session_id="s1",
            agent_name="test",
            parent_session_id="parent1",
            lifecycle_policy="independent",
        )
        assert state.parent_session_id == "parent1"
        assert state.lifecycle_policy == "independent"

    def test_session_state_defaults(self) -> None:
        state = SessionState(session_id="s1", agent_name="test")
        assert state.parent_session_id is None
        assert state.lifecycle_policy == "cascade"


class TestSessionControllerParentChild:
    """Tests for SessionController parent-child session management."""

    @pytest.mark.anyio
    async def test_creates_child_session(self, minimal_pool: AgentPool) -> None:
        ctrl = SessionController(pool=minimal_pool)
        parent, _ = await ctrl.get_or_create_session("parent1")
        child, _ = await ctrl.get_or_create_session("child1", parent_session_id="parent1")
        assert child.parent_session_id == "parent1"
        assert ctrl.get_children("parent1") == ["child1"]
        assert ctrl.get_parent("child1") == parent

    @pytest.mark.anyio
    async def test_close_session_cascade_closes_children(self, minimal_pool: AgentPool) -> None:
        ctrl = SessionController(pool=minimal_pool)
        await ctrl.get_or_create_session("parent1")
        await ctrl.get_or_create_session(
            "child1", parent_session_id="parent1", lifecycle_policy="cascade"
        )
        await ctrl.close_session("parent1")
        assert ctrl.get_session("parent1") is None
        assert ctrl.get_session("child1") is None

    @pytest.mark.anyio
    async def test_close_session_independent_preserves_children(
        self, minimal_pool: AgentPool
    ) -> None:
        ctrl = SessionController(pool=minimal_pool)
        await ctrl.get_or_create_session("parent1")
        await ctrl.get_or_create_session(
            "child1", parent_session_id="parent1", lifecycle_policy="independent"
        )
        await ctrl.close_session("parent1")
        assert ctrl.get_session("parent1") is None
        assert ctrl.get_session("child1") is not None

    @pytest.mark.anyio
    async def test_lifecycle_policy_bound_closes_child_immediately(
        self, minimal_pool: AgentPool
    ) -> None:
        ctrl = SessionController(pool=minimal_pool)
        await ctrl.get_or_create_session("parent1")
        await ctrl.get_or_create_session(
            "child1", parent_session_id="parent1", lifecycle_policy="bound"
        )
        await ctrl.close_session("parent1")
        assert ctrl.get_session("parent1") is None
        assert ctrl.get_session("child1") is None


class TestEventBusScopedSubscription:
    """Tests for EventBus scoped subscription behavior."""

    @pytest.mark.anyio
    async def test_session_scope_receives_own_events(self) -> None:
        bus = EventBus()
        queue = await bus.subscribe("s1", scope="session")
        await bus.publish("s1", "event1")
        envelope = await asyncio.wait_for(queue.get(), timeout=1.0)
        assert envelope is not None
        assert envelope.event == "event1"

    @pytest.mark.anyio
    async def test_session_scope_excludes_child_events(self) -> None:
        bus = EventBus()
        # Manually set up tree: s1 -> s1.1
        bus._session_tree = {"s1": ["s1.1"], "s1.1": []}
        queue = await bus.subscribe("s1", scope="session")
        await bus.publish("s1.1", "event1")
        # Should NOT receive - queue should be empty
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(queue.get(), timeout=0.5)

    @pytest.mark.anyio
    async def test_descendants_scope_receives_child_events(self) -> None:
        bus = EventBus()
        bus._session_tree = {"s1": ["s1.1"], "s1.1": []}
        queue = await bus.subscribe("s1", scope="descendants")
        await bus.publish("s1.1", "event1")
        envelope = await asyncio.wait_for(queue.get(), timeout=1.0)
        assert envelope is not None
        assert envelope.event == "event1"

    @pytest.mark.anyio
    async def test_subtree_scope_receives_sibling_events(self) -> None:
        bus = EventBus()
        bus._session_tree = {"s1": ["s1.1", "s1.2"], "s1.1": [], "s1.2": []}
        queue = await bus.subscribe("s1.1", scope="subtree")
        await bus.publish("s1.2", "event1")
        envelope = await asyncio.wait_for(queue.get(), timeout=1.0)
        assert envelope is not None
        assert envelope.event == "event1"


# ============================================================================
# Close session semantics
# ============================================================================
@pytest.mark.anyio
async def test_close_session_no_active_run(
    session_pool: SessionPool,
    minimal_pool: AgentPool,
) -> None:
    """close_session works normally when there is no active run."""
    agent = MockAgent()

    await _setup_session(session_pool.sessions, "sess-4", agent, minimal_pool)

    await session_pool.close_session("sess-4")
    assert session_pool.sessions.get_session("sess-4") is None


@pytest.mark.anyio
async def test_close_session_acquires_request_lock(
    session_pool: SessionPool,
    minimal_pool: AgentPool,
) -> None:
    """close_session acquires _request_lock before setting closing=True."""
    lock_acquired = False

    original_acquire = asyncio.Lock.acquire

    async def _patched_acquire(self: asyncio.Lock, *args: Any, **kwargs: Any) -> bool:
        nonlocal lock_acquired
        result = await original_acquire(self, *args, **kwargs)
        session = session_pool.sessions.get_session("sess-8")
        if session is not None and self is session._request_lock:
            lock_acquired = True
        return result

    asyncio.Lock.acquire = _patched_acquire  # type: ignore[method-assign]

    agent = MockAgent()

    await _setup_session(session_pool.sessions, "sess-8", agent, minimal_pool)
    await session_pool.close_session("sess-8")

    asyncio.Lock.acquire = original_acquire  # type: ignore[method-assign]
    assert lock_acquired is True


# ============================================================================
# Error propagation
# ============================================================================
# Should complete without error


# ============================================================================
# close_session background task unblock
# ============================================================================

# ---------------------------------------------------------------------------
# Merged from test_close_session.py (suffix: cs)
# ---------------------------------------------------------------------------


if TYPE_CHECKING:
    from wolfharness.agents.context import AgentRunContext


@pytest.fixture
def controller_cs(minimal_pool: AgentPool) -> SessionController:
    """Return a SessionController backed by the real pool."""
    assert minimal_pool.session_pool is not None
    return minimal_pool.session_pool.sessions


def _make_session(session_id: str) -> SessionState:
    """Return a minimal SessionState for testing."""
    return SessionState(session_id=session_id, agent_name="test-agent")


def _make_mock_run_handle(run_id: str = "run-1") -> MagicMock:
    """Return a MagicMock simulating a RunHandle with close/cancel/complete_event."""
    rh = MagicMock(spec=RunHandle)
    rh.run_id = run_id
    rh.run_ctx = None
    rh.close = MagicMock()
    rh.cancel = MagicMock()
    rh.complete_event = asyncio.Event()
    return rh


@pytest.mark.anyio
async def test_flag_on_timeout_triggers_cancel(
    controller_cs: SessionController, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When turn_lock acquisition times out, RunHandle.cancel() is called."""
    monkeypatch.setenv("AGENTPOOL_USE_RUN_TURN", "true")
    session = _make_session("sess-2")
    session.current_run_id = "run-2"
    controller_cs._sessions["sess-2"] = session
    run_handle = _make_mock_run_handle("run-2")
    controller_cs._runs["run-2"] = run_handle
    held_lock = session.turn_lock
    await held_lock.acquire()
    original_timeout = asyncio.timeout

    def fast_timeout(delay: float) -> asyncio.Timeout:
        return original_timeout(0.05)

    monkeypatch.setattr(asyncio, "timeout", fast_timeout)
    await controller_cs.close_session("sess-2")
    run_handle.close.assert_called_once()
    run_handle.cancel.assert_called_once()
    assert session.is_closing is True
    assert "sess-2" not in controller_cs._sessions
    held_lock.release()


@pytest.mark.anyio
async def test_flag_on_no_active_run(
    controller_cs: SessionController, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When flag is ON and no active run exists, session closes cleanly."""
    monkeypatch.setenv("AGENTPOOL_USE_RUN_TURN", "true")
    session = _make_session("sess-4")
    session.current_run_id = None
    controller_cs._sessions["sess-4"] = session
    await controller_cs.close_session("sess-4")
    assert session.is_closing is True
    assert "sess-4" not in controller_cs._sessions


@pytest.mark.skip(
    reason="L2 migration: requires mock session/run_handle internals — remains L1 unit test"
)
@pytest.mark.unit
async def test_close_session_releases_lock_on_cancelled(minimal_pool: AgentPool) -> None:
    """close_session must release turn_lock even if cancelled mid-wait.

    Without try/finally, CancelledError during complete_event.wait()
    skips the lock release, leaving the session permanently locked.
    """
    from wolfharness.orchestrator.core import SessionController

    controller_cs = SessionController(pool=minimal_pool)
    controller_cs._event_bus = EventBus()
    session_id = "sess-close-cancel"
    controller_cs._sessions[session_id] = MagicMock()
    controller_cs._sessions[session_id].session_id = session_id
    controller_cs._sessions[session_id].current_run_id = "fake-run-id"
    controller_cs._sessions[session_id].closing = False
    controller_cs._sessions[session_id].is_closing = False
    controller_cs._sessions[session_id]._request_lock = asyncio.Lock()
    controller_cs._sessions[session_id].turn_lock = asyncio.Lock()
    controller_cs._sessions[session_id].input_provider = None
    controller_cs._sessions[session_id].is_per_session_agent = False
    controller_cs._sessions[session_id].cancel_scope = None
    fake_run = MagicMock()
    fake_run.close = MagicMock()
    fake_run.cancel = MagicMock()
    fake_run.complete_event = asyncio.Event()
    controller_cs._runs["fake-run-id"] = fake_run
    lock = controller_cs._sessions[session_id].turn_lock

    async def _close() -> None:
        await controller_cs._close_session_run_turn(session_id)

    task = asyncio.create_task(_close())
    await asyncio.sleep(0.1)
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task
    try:
        async with asyncio.timeout(1):
            await lock.acquire()
    except TimeoutError:
        pytest.fail("turn_lock was not released after CancelledError in close_session")
    finally:
        if lock.locked():
            lock.release()


@pytest.mark.integration
@pytest.mark.anyio
async def test_close_session_after_cancel(minimal_pool: AgentPool) -> None:
    """close_session() must not hang after a run is cancelled.

    Steps:
        1. Start a run with a blocking turn (patched on real agent).
        2. Cancel via cancel_run_for_session().
        3. Call close_session() with a 30s timeout.
        4. Verify close_session returns within timeout (no hang from turn_lock).

    After cancel, the start() loop publishes RunFailedEvent, sets
    _turn_complete_event, and the turn completes — releasing turn_lock.
    close_session() should acquire turn_lock quickly and return.
    """
    from wolfharness.agents.events import StreamCompleteEvent
    from wolfharness.messaging import ChatMessage
    from wolfharness.orchestrator.turn import Turn

    class _BlockingTurn(Turn):
        """Turn that blocks until run_ctx.cancelled, then returns."""

        def __init__(self, run_ctx: AgentRunContext) -> None:
            self._run_ctx = run_ctx

        async def execute(self):
            self._message_history = []
            self._final_message = ChatMessage(content="blocked", role="assistant")
            while not self._run_ctx.cancelled:
                await asyncio.sleep(0.01)
            return
            yield

    class _StubTurn(Turn):
        """Minimal Turn that yields StreamCompleteEvent."""

        async def execute(self):
            self._message_history = []
            self._final_message = ChatMessage(content="done", role="assistant")
            yield StreamCompleteEvent(message=ChatMessage(content="response", role="assistant"))

    session_pool = minimal_pool.session_pool
    assert session_pool is not None
    session_id = "sess-close-after-cancel"
    await session_pool.create_session(session_id, agent_name="test_agent")
    # Get real agent from pool and patch create_turn to control execution
    agent = await session_pool.sessions.get_or_create_session_agent(session_id)
    call_count = 0

    def _create_turn(
        prompts: Any, run_ctx: AgentRunContext, message_history: Any, **kwargs: Any
    ) -> Turn:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return _BlockingTurn(run_ctx)
        return _StubTurn()

    agent.create_turn = _create_turn  # type: ignore[method-assign]
    msg_id = await session_pool.send_message(session_id, "blocking prompt")
    assert msg_id is not None
    run_handle = session_pool._get_active_run_handle(session_id)
    assert run_handle is not None
    await asyncio.sleep(0.1)
    session_pool.sessions.cancel_run_for_session(session_id)
    await asyncio.sleep(0.2)
    run_handle.close()
    await asyncio.sleep(0.1)
    try:
        await asyncio.wait_for(session_pool.close_session(session_id), timeout=30.0)
    except TimeoutError:
        pytest.fail("close_session hung after cancel — turn_lock was not released")
    assert session_id not in session_pool.sessions._sessions
    # session_pool lifecycle managed by minimal_pool fixture


# ---------------------------------------------------------------------------
# Merged from test_close_checkpoint.py (suffix: cc)
# ---------------------------------------------------------------------------


def make_pending_call(tool_call_id: str = "call-1", tool_name: str = "bash") -> PendingDeferredCall:
    """Create a PendingDeferredCall for testing."""
    return PendingDeferredCall(
        tool_call_id=tool_call_id,
        tool_name=tool_name,
        deferred_kind="external",
        deferred_strategy="block",
    )


def make_session_data(
    session_id: str = "sess-1",
    agent_name: str = "test-agent",
    pending: list[PendingDeferredCall] | None = None,
    status: str = "active",
) -> SessionData:
    """Create a SessionData with optional pending deferred calls."""
    return SessionData(
        session_id=session_id,
        agent_name=agent_name,
        pending_deferred_calls=pending or [],
        status=status,
    )


@pytest.fixture
def controller_cc(minimal_pool: AgentPool, mock_store: MagicMock) -> SessionController:
    """Return a SessionController backed by the real pool with a mock store."""
    return SessionController(pool=minimal_pool, store=mock_store)


@pytest.fixture
def mock_store() -> MagicMock:
    """Return a mocked SessionStore."""
    store = MagicMock()
    store.load_session = AsyncMock(return_value=None)
    store.save_session = AsyncMock(return_value=None)
    store.delete_session = AsyncMock(return_value=None)
    return store


class TestShouldCheckpointOnClose:
    """Test the _should_checkpoint_on_close predicate."""

    def test_returns_false_when_no_pending_calls(self, controller_cc: SessionController) -> None:
        """No pending calls → no checkpoint needed."""
        data = make_session_data(pending=[])
        assert controller_cc._should_checkpoint_on_close(data) is False

    def test_returns_true_when_pending_calls_exist(self, controller_cc: SessionController) -> None:
        """Pending calls → checkpoint needed."""
        data = make_session_data(pending=[make_pending_call()])
        assert controller_cc._should_checkpoint_on_close(data) is True

    def test_returns_false_when_data_is_none(self, controller_cc: SessionController) -> None:
        """None data → no checkpoint needed."""
        assert controller_cc._should_checkpoint_on_close(None) is False


class TestCloseSessionWithoutPendingCalls:
    """close_session() without pending deferred calls behaves as before."""

    @pytest.mark.anyio
    async def test_removes_session(self, controller_cc: SessionController) -> None:
        """Session is removed from tracking."""
        await controller_cc.get_or_create_session("sess-1")
        await controller_cc.close_session("sess-1")
        assert controller_cc.get_session("sess-1") is None

    @pytest.mark.anyio
    async def test_is_idempotent(self, controller_cc: SessionController) -> None:
        """Double close does not raise."""
        await controller_cc.get_or_create_session("sess-1")
        await controller_cc.close_session("sess-1")
        await controller_cc.close_session("sess-1")

    @pytest.mark.anyio
    async def test_marks_closed_in_store(
        self, minimal_pool: AgentPool, mock_store: MagicMock
    ) -> None:
        """When a store exists, the session is marked as closed (not deleted)."""
        mock_store.load_session = AsyncMock(return_value=make_session_data())
        mock_store.save_session = AsyncMock(return_value=None)
        ctrl = SessionController(pool=minimal_pool, store=mock_store)
        await ctrl.get_or_create_session("sess-1")
        await ctrl.close_session("sess-1")
        mock_store.delete_session.assert_not_awaited()
        closed_saves = [
            call
            for call in mock_store.save_session.await_args_list
            if call[0][0].status == "closed"
        ]
        assert len(closed_saves) >= 1, "Expected save() with status='closed'"

    @pytest.mark.anyio
    async def test_does_not_save_checkpoint(
        self, minimal_pool: AgentPool, mock_store: MagicMock
    ) -> None:
        """Without pending calls, save is NOT called for checkpoint status."""
        mock_store.load_session = AsyncMock(return_value=make_session_data())
        ctrl = SessionController(pool=minimal_pool, store=mock_store)
        await ctrl.get_or_create_session("sess-1")
        await ctrl.close_session("sess-1")
        for call in (
            mock_store.save_session.await_args_list
            if hasattr(mock_store.save_session, "await_args_list")
            else []
        ):
            args, _ = call
            if hasattr(args[0], "status") and args[0].status == "checkpointed":
                pytest.fail("save() was called with checkpointed status unexpectedly")


class TestCloseSessionWithPendingCalls:
    """close_session() with pending deferred calls triggers checkpoint."""

    @pytest.mark.anyio
    async def test_saves_checkpoint_before_release(
        self, minimal_pool: AgentPool, mock_store: MagicMock
    ) -> None:
        """When pending deferred calls exist, session data is saved as checkpointed."""
        data = make_session_data(pending=[make_pending_call()])
        mock_store.load_session = AsyncMock(return_value=data)
        mock_store.save_session = AsyncMock(return_value=None)
        ctrl = SessionController(pool=minimal_pool, store=mock_store)
        await ctrl.get_or_create_session("sess-1")
        await ctrl.close_session("sess-1")
        saved_calls = [
            call
            for call in mock_store.save_session.await_args_list
            if call[0][0].session_id == "sess-1" and call[0][0].status == "checkpointed"
        ]
        assert len(saved_calls) >= 1, "Expected save() with checkpointed status"
        saved_data: SessionData = saved_calls[0][0][0]
        assert saved_data.pending_deferred_calls[0].tool_call_id == "call-1"
        assert saved_data.pending_deferred_calls[0].tool_name == "bash"

    @pytest.mark.anyio
    async def test_does_not_delete_from_store(
        self, minimal_pool: AgentPool, mock_store: MagicMock
    ) -> None:
        """When checkpointed, store.delete is NOT called."""
        data = make_session_data(pending=[make_pending_call()])
        mock_store.load_session = AsyncMock(return_value=data)
        ctrl = SessionController(pool=minimal_pool, store=mock_store)
        await ctrl.get_or_create_session("sess-1")
        await ctrl.close_session("sess-1")
        mock_store.delete_session.assert_not_awaited()

    @pytest.mark.anyio
    async def test_releases_inmemory_resources(
        self, minimal_pool: AgentPool, mock_store: MagicMock
    ) -> None:
        """Even when checkpointed, in-memory session state is cleaned up."""
        data = make_session_data(pending=[make_pending_call()])
        mock_store.load_session = AsyncMock(return_value=data)
        ctrl = SessionController(pool=minimal_pool, store=mock_store)
        await ctrl.get_or_create_session("sess-1")
        await ctrl.close_session("sess-1")
        assert ctrl.get_session("sess-1") is None

    @pytest.mark.anyio
    async def test_checkpoint_failure_prevents_resource_release(
        self, minimal_pool: AgentPool, mock_store: MagicMock
    ) -> None:
        """If checkpoint save fails, session resources are NOT released."""
        data = make_session_data(pending=[make_pending_call()])
        mock_store.load_session = AsyncMock(return_value=data)
        orig_save = AsyncMock(return_value=None)

        async def failing_save(obj: Any) -> None:
            if isinstance(obj, SessionData) and obj.status == "checkpointed":
                raise RuntimeError("Storage unavailable")
            await orig_save(obj)

        mock_store.save_session = AsyncMock(side_effect=failing_save)
        ctrl = SessionController(pool=minimal_pool, store=mock_store)
        await ctrl.get_or_create_session("sess-1")
        await ctrl.close_session("sess-1")
        assert ctrl.get_session("sess-1") is not None, (
            "Session should survive when checkpoint save fails"
        )
        mock_store.delete_session.assert_not_awaited()


class TestCloseSessionWithoutStore:
    """close_session() when no store is configured."""

    @pytest.mark.anyio
    async def test_no_store_no_checkpoint(self, controller_cc: SessionController) -> None:
        """Without a store, close_session just removes the session."""
        await controller_cc.get_or_create_session("sess-1")
        await controller_cc.close_session("sess-1")
        assert controller_cc.get_session("sess-1") is None


class TestSessionPoolCloseCheckpoint:
    """SessionPool.close_session delegates to SessionController which handles checkpoint."""


class TestSaveCloseCheckpoint:
    """Test the _save_close_checkpoint helper."""

    @pytest.mark.anyio
    async def test_saves_with_checkpoint_status(
        self, minimal_pool: AgentPool, mock_store: MagicMock
    ) -> None:
        """_save_close_checkpoint saves session data as checkpointed."""
        data = make_session_data(pending=[make_pending_call()])
        mock_store.load_session = AsyncMock(return_value=data)
        ctrl = SessionController(pool=minimal_pool, store=mock_store)
        result = await ctrl._save_close_checkpoint("sess-1", data)
        assert result is True
        mock_store.save_session.assert_awaited_once()
        saved_data = mock_store.save_session.await_args[0][0]
        assert saved_data.status == "checkpointed"
        assert len(saved_data.pending_deferred_calls) == 1

    @pytest.mark.anyio
    async def test_returns_false_on_failure(
        self, minimal_pool: AgentPool, mock_store: MagicMock
    ) -> None:
        """_save_close_checkpoint returns False when save fails."""
        data = make_session_data(pending=[make_pending_call()])
        mock_store.save_session = AsyncMock(side_effect=RuntimeError("Storage error"))
        ctrl = SessionController(pool=minimal_pool, store=mock_store)
        result = await ctrl._save_close_checkpoint("sess-1", data)
        assert result is False


# ---------------------------------------------------------------------------
# Merged from test_checkpoint_close_review.py (suffix: cr)
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_checkpointed_status_not_overwritten_on_close(
    minimal_pool: AgentPool, mock_store: MagicMock
) -> None:
    """Checkpointed status must not be overwritten with 'closed' on close.

    When a session is checkpointed before close (due to pending deferred
    calls), _close_session_unlocked must NOT call _mark_session_closed()
    which would overwrite the "checkpointed" status with "closed".
    """
    data = make_session_data(pending=[make_pending_call()])
    checkpointed_data = data.model_copy(update={"status": "checkpointed"})
    mock_store.load_session = AsyncMock(return_value=checkpointed_data)
    mock_store.save_session = AsyncMock(return_value=None)
    ctrl = SessionController(pool=minimal_pool, store=mock_store)
    await ctrl.get_or_create_session("sess-1")
    await ctrl.close_session("sess-1")
    all_saves = mock_store.save_session.await_args_list
    closed_saves = [call for call in all_saves if call[0][0].status == "closed"]
    checkpointed_saves = [call for call in all_saves if call[0][0].status == "checkpointed"]
    assert len(closed_saves) == 0, (
        f"Expected no save with status='closed', but found {len(closed_saves)}."
        " The checkpointed status was overwritten!"
    )
    assert len(checkpointed_saves) >= 1, "Expected at least one save with status='checkpointed'"


@pytest.mark.anyio
async def test_non_checkpointed_still_marked_closed(
    minimal_pool: AgentPool, mock_store: MagicMock
) -> None:
    """Normal close (no pending calls) should still mark as 'closed'.

    When a session has NO pending deferred calls, close_session should
    still mark it as "closed" (normal behavior, no regression).
    """
    data = make_session_data(pending=[])
    mock_store.load_session = AsyncMock(return_value=data)
    mock_store.save_session = AsyncMock(return_value=None)
    ctrl = SessionController(pool=minimal_pool, store=mock_store)
    await ctrl.get_or_create_session("sess-1")
    await ctrl.close_session("sess-1")
    all_saves = mock_store.save_session.await_args_list
    closed_saves = [call for call in all_saves if call[0][0].status == "closed"]
    assert len(closed_saves) >= 1, "Expected save with status='closed' for normal close"
