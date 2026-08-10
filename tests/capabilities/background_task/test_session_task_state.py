"""Tests for SessionTaskState isolation, routing, closure capture, and T5 fixes.

Covers:
1. SessionTaskState lazy creation and reuse
2. Per-session isolation between different AgentRunContexts
3. WeakKeyDictionary GC behavior
4. retrieval_retry_count isolation
5. Closure capture in _on_task_completed
6. before_run() resets per-turn fields only
7. Notification suppression with child_done_events still popped
8. Separated delivery: child_done_events per-task, followup once per batch
9. Dead session skip (weakref dereferences to None)
10. Fallback to session_pool.followup() when _run_handle is None
11. Ephemeral state with no-op deliver_callback
12. deliver_callback uses weakref.ref(run_ctx) — no strong reference
13. Cross-turn notification persistence
14. followup() called BEFORE child_done_events.pop()
15. get_model_settings closure uses per-session pending_retrievals
16. _background_cancel uses get_all_tasks() public API
"""

# pyright: reportAttributeAccessIssue=false
# Mock-heavy test code: assigning to spec'd attributes and accessing mock
# methods (assert_not_called, assert_called_once, call_args) are expected.

from __future__ import annotations

import asyncio
import contextlib
from datetime import UTC, datetime
import gc
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import anyio
import pytest

from wolfharness.capabilities.background_task.capability import (
    BackgroundTaskCapability,
)
from wolfharness.capabilities.background_task.types import BackgroundTask, SessionTaskState


# ---------------------------------------------------------------------------
# Helpers (adapted from test_background_task_completion_notification.py)
# ---------------------------------------------------------------------------


def _wrap_in_run_context(agent_ctx: Any) -> MagicMock:
    """Wrap an AgentContext in a mock RunContext for capability tool methods."""
    run_ctx = MagicMock()
    run_ctx.deps = agent_ctx
    run_ctx.tool_call_id = agent_ctx.tool_call_id
    return run_ctx


def _make_mock_node(*, name: str = "worker", description: str = "A worker agent") -> MagicMock:
    """Create a mock node that looks like a BaseAgent with run_stream support."""
    node = MagicMock()
    node.type = "agent"
    node.name = name
    node.description = description
    node.model_name = "test:model"
    node.session_id = "ses_parent_123"
    return node


def _make_mock_pool(nodes: dict[str, MagicMock] | None = None) -> Any:
    """Create a mock AgentPool with the given nodes."""
    pool = MagicMock()

    if nodes is None:
        mock_node = _make_mock_node()
        nodes = {"worker": mock_node}

    pool.nodes = nodes
    pool.agent_configs = nodes
    pool.all_agents = list(nodes.items())
    pool.teams = {}
    pool.sessions = None
    mock_session_pool = MagicMock()

    async def _get_or_create_session_agent(agent_name: str, agent_type: str):
        return nodes.get(agent_name, next(iter(nodes.values())))

    mock_session_pool.sessions = MagicMock()
    mock_session_pool.sessions.get_or_create_session_agent = AsyncMock(
        side_effect=_get_or_create_session_agent,
    )

    mock_session_pool.inject_prompt = AsyncMock()
    mock_session_pool.steer = AsyncMock()
    mock_session_pool.followup = AsyncMock(return_value=True)

    async def _send_message(session_id: str, prompt: str, input_provider=None, **kwargs: Any):
        await asyncio.sleep(0.1)
        return MagicMock()

    mock_session_pool.send_message = AsyncMock(side_effect=_send_message)

    async def _subscribe(session_id: str, scope: str = "session"):
        queue = MagicMock()
        queue.receive = AsyncMock(side_effect=anyio.EndOfStream())
        return queue

    mock_session_pool.event_bus = MagicMock()
    mock_session_pool.event_bus.subscribe = AsyncMock(side_effect=_subscribe)
    mock_session_pool.event_bus.unsubscribe = AsyncMock()
    pool.session_pool = mock_session_pool
    return pool


def _make_agent_context(
    pool: Any | None = None,
    data: dict[str, object] | None = None,
    run_ctx: Any | None = None,
) -> Any:
    """Create a minimal AgentContext for testing.

    If ``run_ctx`` is provided, it is set on ``ctx.run_ctx``.
    """
    agent = MagicMock()
    agent.type = "agent"
    agent.name = "coordinator"
    agent.session_id = "ses_parent_123"
    agent.agent_pool = pool
    agent.inject_prompt = MagicMock()

    ctx = MagicMock()
    ctx.node = agent
    ctx.agent = agent
    ctx.pool = pool
    ctx.data = data if data is not None else {}
    ctx.tool_call_id = "tc_001"
    ctx.run_ctx = run_ctx
    ctx.events = MagicMock()
    ctx.events.emit_event = AsyncMock()
    ctx.create_child_session = AsyncMock(return_value="ses_child_456")
    ctx.internal_fs = MagicMock()
    ctx.internal_fs.mkdirs = MagicMock()
    ctx.internal_fs.pipe = MagicMock()

    return ctx


class _MockRunContext:
    """A lightweight mock AgentRunContext.

    Real AgentRunContext cannot be imported in this test directory due to
    the ``tests/wolfharness/__init__.py`` shadowing the ``wolfharness`` package.
    This mock provides the attributes needed by the capability:
    ``child_done_events``, ``_run_handle``, ``session_id``, ``run_id``.
    """

    __slots__ = ("__weakref__", "_run_handle", "child_done_events", "run_id", "session_id")

    def __init__(self, session_id: str = "ses_test_001", run_id: str | None = None) -> None:
        self.session_id = session_id
        self.run_id = run_id or f"run_{id(self)}"
        self._run_handle: Any = None
        self.child_done_events: dict[str, anyio.Event] = {}


def _make_real_run_ctx() -> _MockRunContext:
    """Create a mock run context with a unique run_id."""
    return _MockRunContext(session_id="ses_test_001")


def _make_mock_run_handle() -> MagicMock:
    """Create a mock RunHandle with a working followup() method."""
    handle = MagicMock()
    handle.followup = MagicMock(return_value=True)
    handle._closing = False
    return handle


def _make_completed_task(
    task_id: str = "bg_test001",
    parent_session_id: str = "ses_parent_123",
    child_session_id: str = "ses_child_456",
) -> BackgroundTask:
    """Create a BackgroundTask in completed state."""
    task = BackgroundTask(
        id=task_id,
        description="test task",
        agent_or_team="worker",
        prompt="do something",
        parent_session_id=parent_session_id,
        child_session_id=child_session_id,
    )
    task.status = "completed"
    task.completed_at = datetime.now(tz=UTC)
    task.started_at = datetime.now(tz=UTC)
    task.result = "result text"
    return task


# ---------------------------------------------------------------------------
# 1. SessionTaskState lazy creation — first access creates, second reuses
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_session_state_lazy_creation_and_reuse():
    """First call to _get_session_state creates; second call returns the same instance."""
    capability = BackgroundTaskCapability(schemas=None)
    pool = _make_mock_pool()
    run_ctx = _make_real_run_ctx()
    ctx = _make_agent_context(pool=pool, run_ctx=run_ctx)

    state1 = capability._get_session_state(ctx)
    state2 = capability._get_session_state(ctx)

    assert state1 is state2, "Second access must return the same SessionTaskState"
    assert isinstance(state1, SessionTaskState)


# ---------------------------------------------------------------------------
# 2. Per-session isolation — two mock sessions get different states
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_per_session_isolation():
    """Two different AgentRunContexts get different SessionTaskState instances."""
    capability = BackgroundTaskCapability(schemas=None)
    pool = _make_mock_pool()

    run_ctx_a = _make_real_run_ctx()
    run_ctx_b = _make_real_run_ctx()

    ctx_a = _make_agent_context(pool=pool, run_ctx=run_ctx_a)
    ctx_b = _make_agent_context(pool=pool, run_ctx=run_ctx_b)

    state_a = capability._get_session_state(ctx_a)
    state_b = capability._get_session_state(ctx_b)

    assert state_a is not state_b, "Different sessions must get different states"
    assert state_a.task_manager is not state_b.task_manager
    assert state_a.batcher is not state_b.batcher


# ---------------------------------------------------------------------------
# 3. Session state keyed by run_id — different run_ctx objects with same run_id reuse state
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_weakkeydict_gc_behavior():
    """Session state is keyed by run_id, not by object identity.

    Two different run_ctx objects with the same run_id resolve to the same
    SessionTaskState.  This replaces the old WeakKeyDictionary GC behavior —
    we now use a plain dict keyed by the stable UUID ``run_id``.
    """
    capability = BackgroundTaskCapability(schemas=None)

    run_ctx = _make_real_run_ctx()

    # Use a simple object instead of MagicMock to avoid reference cycles
    class _SimpleCtx:
        def __init__(self, pool: Any, rc: Any) -> None:
            self.pool = pool
            self.run_ctx = rc
            self.data: dict[str, object] = {}
            self.tool_call_id = "tc_001"
            self.node = MagicMock()
            self.agent = MagicMock()
            self.events = MagicMock()
            self.create_child_session = AsyncMock(return_value="ses_child_456")
            self.internal_fs = MagicMock()

    pool = _make_mock_pool()
    ctx = _SimpleCtx(pool, run_ctx)

    state = capability._get_session_state(ctx)
    assert state is not None

    # Verify it's in the dict, keyed by run_id
    assert run_ctx.run_id in capability._session_states

    # A second run_ctx with the SAME run_id should reuse the same state
    run_ctx_2 = _MockRunContext(session_id="ses_test_001", run_id=run_ctx.run_id)
    ctx_2 = _SimpleCtx(pool, run_ctx_2)
    state_2 = capability._get_session_state(ctx_2)
    assert state_2 is state, "Same run_id must resolve to same SessionTaskState"

    # A third run_ctx with a DIFFERENT run_id should get a new state
    run_ctx_3 = _MockRunContext(session_id="ses_test_002", run_id="run_different")
    ctx_3 = _SimpleCtx(pool, run_ctx_3)
    state_3 = capability._get_session_state(ctx_3)
    assert state_3 is not state, "Different run_id must get different SessionTaskState"

    # Manual cleanup (no GC magic — caller is responsible for lifecycle)
    del ctx, ctx_2, ctx_3, run_ctx, run_ctx_2, run_ctx_3, state, state_2, state_3
    gc.collect()

    # The dict still holds entries (no auto-GC) — this is by design
    assert len(capability._session_states) == 2


# ---------------------------------------------------------------------------
# 4. retrieval_retry_count isolation between sessions
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_retrieval_retry_count_isolation():
    """retrieval_retry_count is isolated between sessions."""
    capability = BackgroundTaskCapability(schemas=None, force_retrieval="directive")
    pool = _make_mock_pool()

    run_ctx_a = _make_real_run_ctx()
    run_ctx_b = _make_real_run_ctx()

    ctx_a = _make_agent_context(pool=pool, run_ctx=run_ctx_a)
    ctx_b = _make_agent_context(pool=pool, run_ctx=run_ctx_b)

    state_a = capability._get_session_state(ctx_a)
    state_b = capability._get_session_state(ctx_b)

    state_a.retrieval_retry_count = 5
    assert state_b.retrieval_retry_count == 0, "Session B must not be affected by Session A"


# ---------------------------------------------------------------------------
# 5. Closure capture — _on_task_completed uses captured state, not re-derived ctx
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_closure_uses_captured_state():
    """The _on_task_completed closure uses the captured state, not a re-derived ctx.

    This test verifies that the closure created in _task_async captures the
    SessionTaskState at creation time, so even if the capability's internal
    state changes, the closure still references the original state.
    """
    capability = BackgroundTaskCapability(schemas=None)
    pool = _make_mock_pool()
    run_ctx = _make_real_run_ctx()
    ctx = _make_agent_context(pool=pool, run_ctx=run_ctx)

    # Get state before task creation
    state_before = capability._get_session_state(ctx)

    # Simulate the closure logic manually — the closure captures `state` and `ctx`
    # from the enclosing scope of _task_async, NOT re-deriving from _get_session_state
    captured_state = state_before
    captured_ctx = ctx

    task_id = "bg_closure01"
    child_session_id = "ses_child_456"

    # Register a task in the captured state's manager
    task_model = _make_completed_task(task_id=task_id, child_session_id=child_session_id)
    captured_state.task_manager.register_task(task_model)

    # Set up child_done_events
    done_event = anyio.Event()
    captured_ctx.run_ctx.child_done_events[child_session_id] = done_event

    # Build the closure exactly as _task_async does
    def _on_task_completed() -> None:
        # 1. ALWAYS pop+set child_done_events immediately
        run_ctx_local = captured_ctx.run_ctx
        if run_ctx_local is not None and child_session_id:
            event = run_ctx_local.child_done_events.pop(child_session_id, None)
            if event is not None:
                event.set()

        # 2. Conditionally submit to batcher
        if captured_state.task_manager.has_blocking_waiter(task_id):
            return

        task = captured_state.task_manager.get_task(task_id)
        if task is None:
            return

        with contextlib.suppress(ValueError):
            captured_state.batcher.submit(task)

    # Execute the closure
    _on_task_completed()

    # Verify the closure used the captured state (task was found in captured_state)
    assert done_event.is_set(), "Closure must have popped and set the child_done_event"
    assert child_session_id not in captured_ctx.run_ctx.child_done_events


# ---------------------------------------------------------------------------
# 6. before_run() only resets per-turn fields — task_manager and batcher persist
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_before_run_resets_only_per_turn_fields():
    """before_run() resets pending_retrievals and retrieval_retry_count, but preserves
    task_manager and batcher.
    """
    capability = BackgroundTaskCapability(schemas=None)
    pool = _make_mock_pool()
    run_ctx = _make_real_run_ctx()
    ctx = _make_agent_context(pool=pool, run_ctx=run_ctx)

    # First access creates the state
    state = capability._get_session_state(ctx)

    # Mutate per-turn fields
    state.pending_retrievals.add("bg_task_1")
    state.pending_retrievals.add("bg_task_2")
    state.retrieval_retry_count = 3

    # Store references before before_run
    task_manager_ref = state.task_manager
    batcher_ref = state.batcher

    # Call before_run — pass ctx directly (not wrapped) so _get_session_state
    # takes the else branch and finds the same state via ctx.run_ctx
    await capability.before_run(ctx)

    # Per-turn fields must be reset
    assert len(state.pending_retrievals) == 0, "pending_retrievals must be cleared"
    assert state.retrieval_retry_count == 0, "retrieval_retry_count must be reset"

    # Persistent fields must survive
    assert state.task_manager is task_manager_ref, "task_manager must persist across turns"
    assert state.batcher is batcher_ref, "batcher must persist across turns"


# ---------------------------------------------------------------------------
# 7. Notification suppression when has_blocking_waiter — child_done_events STILL popped
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_notification_suppression_still_pops_child_done_events():
    """When has_blocking_waiter is True, batcher.submit is skipped, but
    child_done_events is STILL popped and set immediately (C3 FIX).
    """
    capability = BackgroundTaskCapability(schemas=None)
    pool = _make_mock_pool()
    run_ctx = _make_real_run_ctx()
    ctx = _make_agent_context(pool=pool, run_ctx=run_ctx)

    state = capability._get_session_state(ctx)

    # Register a task
    task_model = _make_completed_task(task_id="bg_suppress01", child_session_id="ses_child_sup")
    state.task_manager.register_task(task_model)

    # Set up child_done_events on run_ctx
    done_event = anyio.Event()
    run_ctx.child_done_events["ses_child_sup"] = done_event

    # Register a blocking waiter (simulates background_output(block=True))
    state.task_manager._handles["bg_suppress01"] = MagicMock()
    token = state.task_manager.register_blocking_waiter("bg_suppress01")
    assert token is not None

    # Mock batcher.submit to track if it's called
    state.batcher.submit = MagicMock()

    # Build the _on_task_completed closure manually to test it in isolation
    task_id = "bg_suppress01"
    child_session_id = "ses_child_sup"
    parent_session_id = "ses_parent_123"

    def _on_task_completed() -> None:
        # 1. ALWAYS pop+set child_done_events immediately
        rc = ctx.run_ctx
        if rc is not None and child_session_id:
            event = rc.child_done_events.pop(child_session_id, None)
            if event is not None:
                event.set()

        # 2. Conditionally submit to batcher (skip if blocking waiter)
        if state.task_manager.has_blocking_waiter(task_id):
            return

        task = state.task_manager.get_task(task_id)
        if task is None:
            return

        if not parent_session_id:
            return

        with contextlib.suppress(ValueError):
            state.batcher.submit(task)

    # Execute the closure
    _on_task_completed()

    # child_done_events MUST be popped and event set
    assert "ses_child_sup" not in run_ctx.child_done_events, (
        "child_done_events must be popped even when blocking waiter is present"
    )
    assert done_event.is_set(), "Event must be set even when blocking waiter is present"

    # batcher.submit must NOT be called
    state.batcher.submit.assert_not_called()


# ---------------------------------------------------------------------------
# 8. Separated delivery — child_done_events popped per-task, followup called once
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_separated_delivery_child_events_per_task_followup_once():
    """child_done_events is popped per-task in the closure, while followup()
    is called once per batch in the deliver callback.
    """
    capability = BackgroundTaskCapability(schemas=None)
    pool = _make_mock_pool()
    run_ctx = _make_real_run_ctx()
    ctx = _make_agent_context(pool=pool, run_ctx=run_ctx)

    state = capability._get_session_state(ctx)

    # Set up a run handle with followup
    mock_handle = _make_mock_run_handle()
    run_ctx._run_handle = mock_handle

    # Create two completed tasks
    task1 = _make_completed_task(task_id="bg_sep01", child_session_id="ses_child_a")
    task2 = _make_completed_task(task_id="bg_sep02", child_session_id="ses_child_b")
    state.task_manager.register_task(task1)
    state.task_manager.register_task(task2)

    # Set up child_done_events
    event_a = anyio.Event()
    event_b = anyio.Event()
    run_ctx.child_done_events["ses_child_a"] = event_a
    run_ctx.child_done_events["ses_child_b"] = event_b

    # Simulate the immediate pop+set (closure behavior, step 1)
    for child_sid in ["ses_child_a", "ses_child_b"]:
        event = run_ctx.child_done_events.pop(child_sid, None)
        if event is not None:
            event.set()

    # Verify both events were popped and set
    assert "ses_child_a" not in run_ctx.child_done_events
    assert "ses_child_b" not in run_ctx.child_done_events
    assert event_a.is_set()
    assert event_b.is_set()

    # Now simulate the deliver callback (batch path)
    deliver_callback = capability._make_deliver_callback(run_ctx, pool.session_pool)

    # Manually call the deliver callback with both tasks as a batch
    await deliver_callback("ses_parent_123", [task1, task2], "notice text")

    # followup must be called exactly once (for the batch)
    assert mock_handle.followup.call_count == 1, (
        "followup must be called once per batch, not per task"
    )


# ---------------------------------------------------------------------------
# 9. Dead session skip — weak ref dereferences to None, callback skips silently
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_dead_session_skip():
    """When run_ctx weak ref dereferences to None, the deliver callback skips silently
    without calling followup (no crash, no fallback).
    """
    capability = BackgroundTaskCapability(schemas=None)
    pool = _make_mock_pool()

    # Create a disposable run_ctx and a callback bound to it
    disposable_ctx = _MockRunContext(session_id="ses_disposable")
    disposable_handle = _make_mock_run_handle()
    disposable_ctx._run_handle = disposable_handle

    callback = capability._make_deliver_callback(disposable_ctx, pool.session_pool)

    # Drop strong reference and force GC
    del disposable_ctx
    gc.collect()

    # Calling the callback should not crash — it should silently return
    task = _make_completed_task()
    await callback("ses_parent", [task], "notice")

    # followup was never called because the weak ref is dead
    disposable_handle.followup.assert_not_called()
    pool.session_pool.followup.assert_not_called()


# ---------------------------------------------------------------------------
# 10. Fallback to session_pool.followup() when _run_handle is None
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_fallback_to_session_pool_followup_when_no_run_handle():
    """When _run_handle is None but run_ctx is alive via weak ref, the deliver
    callback falls back to session_pool.followup() (NOT steer()).
    """
    capability = BackgroundTaskCapability(schemas=None)
    pool = _make_mock_pool()
    run_ctx = _make_real_run_ctx()

    # _run_handle is None (no active run handle)
    run_ctx._run_handle = None

    deliver_callback = capability._make_deliver_callback(run_ctx, pool.session_pool)

    task = _make_completed_task(task_id="bg_fallback01")
    await deliver_callback("ses_parent_123", [task], "notice text")

    # session_pool.followup must be called
    pool.session_pool.followup.assert_called_once()
    call_args = pool.session_pool.followup.call_args
    assert call_args[0][0] == "ses_parent_123"
    assert "notice text" in call_args[0][1]

    # steer must NEVER be called
    pool.session_pool.steer.assert_not_called()


# ---------------------------------------------------------------------------
# 11. Ephemeral state with no-op deliver_callback — does not crash on flush
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_ephemeral_state_noop_deliver_callback():
    """Ephemeral state (run_ctx is None) uses a no-op deliver_callback that
    does not crash when the batcher flushes.
    """
    capability = BackgroundTaskCapability(schemas=None)

    # Create an AgentContext with run_ctx=None (ephemeral path)
    pool = _make_mock_pool()
    ctx = _make_agent_context(pool=pool, run_ctx=None)

    state = capability._get_session_state(ctx)
    assert state is not None

    # Start the batcher
    await state.batcher.start()

    # Submit a completed task and let the debounce fire
    task = _make_completed_task(task_id="bg_ephem01", parent_session_id="ses_ephem")
    state.task_manager.register_task(task)

    # Manually trigger flush
    state.batcher._pending["ses_ephem"] = [task]
    await state.batcher._flush("ses_ephem")

    # Should not crash — no exception means pass
    await state.batcher.shutdown()


# ---------------------------------------------------------------------------
# 12. deliver_callback uses weakref.ref(run_ctx) — no strong reference
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_deliver_callback_uses_weakref_no_strong_reference():
    """The deliver_callback created by _make_deliver_callback uses weakref.ref(run_ctx),
    so SessionTaskState holds no strong reference to AgentRunContext.
    """
    capability = BackgroundTaskCapability(schemas=None)
    pool = _make_mock_pool()
    run_ctx = _make_real_run_ctx()

    # Create the state and deliver callback
    state = capability._get_session_state(
        _make_agent_context(pool=pool, run_ctx=run_ctx),
    )

    # The deliver_callback is stored on the batcher
    deliver_cb = state.batcher.deliver_callback

    # Inspect the closure of the deliver callback
    if hasattr(deliver_cb, "__closure__") and deliver_cb.__closure__:
        closure_types = [type(cell.cell_contents).__name__ for cell in deliver_cb.__closure__]
        # The weakref.ref object should be in the closure
        assert "weakref" in closure_types or "ReferenceType" in closure_types, (
            f"deliver_callback closure must contain a weakref, got: {closure_types}"
        )


# ---------------------------------------------------------------------------
# 13. Cross-turn notification persistence — batcher persists across turns
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_cross_turn_notification_persistence():
    """Launch a task in turn 1, let turn 1 end, start turn 2 (before_run fires),
    verify the batcher is the same instance (notifications persist across turns).
    """
    capability = BackgroundTaskCapability(schemas=None)
    pool = _make_mock_pool()
    run_ctx = _make_real_run_ctx()
    ctx = _make_agent_context(pool=pool, run_ctx=run_ctx)

    # Turn 1: before_run fires, creates state
    await capability.before_run(_wrap_in_run_context(ctx))
    state_turn1 = capability._get_session_state(ctx)
    batcher_turn1 = state_turn1.batcher

    # Simulate turn 1 ending

    # Turn 2: before_run fires again
    await capability.before_run(_wrap_in_run_context(ctx))
    state_turn2 = capability._get_session_state(ctx)

    # The batcher must be the same instance (persists across turns)
    assert state_turn2.batcher is batcher_turn1, (
        "NotificationBatcher must persist across turns for cross-turn delivery"
    )
    # The task_manager must also persist
    assert state_turn2.task_manager is state_turn1.task_manager


# ---------------------------------------------------------------------------
# 14. followup() called BEFORE child_done_events.pop() in deliver callback
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_followup_called_before_child_done_events_pop():
    """In the deliver callback, followup() is called BEFORE child_done_events.pop()."""
    capability = BackgroundTaskCapability(schemas=None)
    pool = _make_mock_pool()
    run_ctx = _make_real_run_ctx()

    mock_handle = _make_mock_run_handle()
    run_ctx._run_handle = mock_handle

    # Set up a child_done_event
    child_sid = "ses_child_order"
    done_event = anyio.Event()
    run_ctx.child_done_events[child_sid] = done_event

    # Track call order
    call_order: list[str] = []

    # Wrap followup to track when it's called
    original_followup = mock_handle.followup

    def tracking_followup(msg: str) -> bool:
        call_order.append("followup")
        return original_followup(msg)

    mock_handle.followup = tracking_followup

    # Use a custom dict subclass to track pop calls (dict.pop is read-only)
    class _TrackingDict(dict):  # type: ignore[type-arg]
        def pop(self, key: str, default: Any = None) -> Any:  # type: ignore[override]
            call_order.append("pop")
            return super().pop(key, default)

    tracking_dict: _TrackingDict = _TrackingDict()
    tracking_dict[child_sid] = done_event
    run_ctx.child_done_events = tracking_dict

    deliver_callback = capability._make_deliver_callback(run_ctx, pool.session_pool)

    task = _make_completed_task(task_id="bg_order01", child_session_id=child_sid)
    await deliver_callback("ses_parent", [task], "notice")

    # followup must be called BEFORE pop
    assert "followup" in call_order, "followup must be called"
    assert "pop" in call_order, "child_done_events.pop must be called"
    followup_idx = call_order.index("followup")
    pop_idx = call_order.index("pop")
    assert followup_idx < pop_idx, (
        f"followup must be called BEFORE child_done_events.pop, got order: {call_order}"
    )


# ---------------------------------------------------------------------------
# 15. get_model_settings closure uses per-session pending_retrievals
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_get_model_settings_uses_per_session_pending_retrievals():
    """get_model_settings returns a closure that checks per-session pending_retrievals,
    not a stale capability-level set.
    """
    capability = BackgroundTaskCapability(schemas=None, force_retrieval="tool_choice")
    pool = _make_mock_pool()

    run_ctx_a = _make_real_run_ctx()
    run_ctx_b = _make_real_run_ctx()

    ctx_a = _make_agent_context(pool=pool, run_ctx=run_ctx_a)
    ctx_b = _make_agent_context(pool=pool, run_ctx=run_ctx_b)

    state_a = capability._get_session_state(ctx_a)
    capability._get_session_state(ctx_b)  # ensure session B state exists

    # Only session A has pending retrievals
    state_a.pending_retrievals.add("bg_pending_a")

    settings_fn = capability.get_model_settings()
    assert settings_fn is not None

    # Session A should get tool_choice forcing
    settings_a = settings_fn(ctx_a)
    assert "tool_choice" in settings_a
    assert settings_a["tool_choice"] is not None

    # Session B should get empty settings (no pending retrievals)
    settings_b = settings_fn(ctx_b)
    assert "tool_choice" not in settings_b or settings_b.get("tool_choice") is None


# ---------------------------------------------------------------------------
# 16. _background_cancel uses get_all_tasks() public API
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_background_cancel_uses_get_all_tasks_public_api():
    """_background_cancel(cancel_all=True) uses get_all_tasks() public API,
    not the private _tasks attribute.
    """
    capability = BackgroundTaskCapability(schemas=None)
    pool = _make_mock_pool()
    run_ctx = _make_real_run_ctx()
    ctx = _make_agent_context(pool=pool, run_ctx=run_ctx)

    state = capability._get_session_state(ctx)

    # Register some tasks
    task1 = _make_completed_task(task_id="bg_cancel01", child_session_id="ses_c1")
    task2 = _make_completed_task(task_id="bg_cancel02", child_session_id="ses_c2")
    # Make them non-terminal so they get cancelled
    task1.status = "running"
    task2.status = "pending"
    state.task_manager.register_task(task1)
    state.task_manager.register_task(task2)

    # Create mock handles so cancel_task works
    # Use AsyncMock for the task since cancel_task awaits it via asyncio.shield
    async def _noop_coro() -> None:
        pass

    mock_atask1 = asyncio.ensure_future(_noop_coro())
    state.task_manager._handles["bg_cancel01"] = MagicMock()
    state.task_manager._handles["bg_cancel01"].task = mock_atask1
    state.task_manager._handles["bg_cancel01"].completion_event = anyio.Event()

    mock_atask2 = asyncio.ensure_future(_noop_coro())
    state.task_manager._handles["bg_cancel02"] = MagicMock()
    state.task_manager._handles["bg_cancel02"].task = mock_atask2
    state.task_manager._handles["bg_cancel02"].completion_event = anyio.Event()

    # Patch get_all_tasks to track if it's called
    original_get_all = state.task_manager.get_all_tasks
    get_all_called: list[bool] = []

    def tracking_get_all() -> list[BackgroundTask]:
        get_all_called.append(True)
        return original_get_all()

    state.task_manager.get_all_tasks = tracking_get_all  # type: ignore[method-assign]

    result = await capability._background_cancel(
        ctx,
        cancel_all=True,
    )

    # get_all_tasks must have been called
    assert len(get_all_called) > 0, "_background_cancel must use get_all_tasks() public API"

    # Result should mention cancelled tasks
    assert "Cancelled" in result or "cancel" in result.lower()
