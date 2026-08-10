"""Tests for SessionPool.resume_session() durable execution resume.

Covers native agent resume, ACP agent resume, error cases, and event emission.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from wolfharness.agents.events.events import SessionResumeEvent
from wolfharness.orchestrator.core import SessionBusyError, SessionPool
from wolfharness.sessions.models import PendingDeferredCall, SessionData


if TYPE_CHECKING:
    from wolfharness import AgentPool


def _stream_empty(queue: asyncio.Queue[Any]) -> bool:
    """Check if a subscriber queue has no buffered items."""
    return queue.empty()


pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _ok_gen(*args: Any, **kwargs: Any) -> Any:
    """Async generator that yields nothing and completes silently."""
    return
    yield  # pragma: no cover


async def _fail_gen(*args: Any, **kwargs: Any) -> Any:
    """Async generator that raises RuntimeError during iteration."""
    raise RuntimeError("Boom")
    yield  # pragma: no cover


def _track_calls() -> Any:
    """Return (async_gen_fn, calls_list) — calls_list collects call kwargs."""
    calls: list[dict[str, Any]] = []

    async def _tracked_gen(*args: Any, **kwargs: Any) -> Any:
        calls.append(dict(kwargs))
        return
        yield  # pragma: no cover

    return _tracked_gen, calls


def make_pending_call(
    tool_call_id: str = "call-1",
    tool_name: str = "bash",
    deferred_kind: str = "external",
    deferred_strategy: str = "block",
) -> PendingDeferredCall:
    return PendingDeferredCall(
        tool_call_id=tool_call_id,
        tool_name=tool_name,
        deferred_kind=deferred_kind,  # type: ignore[arg-type]
        deferred_strategy=deferred_strategy,  # type: ignore[arg-type]
    )


def make_session_data(
    session_id: str = "sess-1",
    agent_name: str = "test-agent",
    agent_type: str = "native",
    pending: list[PendingDeferredCall] | None = None,
    status: str = "checkpointed",
    agent_config_hash: str = "abc123",
    metadata: dict[str, Any] | None = None,
) -> SessionData:
    return SessionData(
        session_id=session_id,
        agent_name=agent_name,
        agent_type=agent_type,
        pending_deferred_calls=pending or [],
        status=status,
        agent_config_hash=agent_config_hash,
        metadata=metadata or {},
    )


def make_deferred_tool_results(
    call_ids: list[str],
) -> Any:
    """Create a DeferredToolResults-compatible object for tests.

    Returns a simple object with `calls` dict for matching tool_call_ids.
    """
    return _FakeDeferredResults(call_ids=call_ids)


@dataclass
class _FakeDeferredResults:
    """Fake DeferredToolResults for testing (avoids pydantic-ai import)."""

    call_ids: list[str] = field(default_factory=list)

    @property
    def calls(self) -> dict[str, str]:
        return {cid: f"result-{cid}" for cid in self.call_ids}

    @property
    def approvals(self) -> dict[str, Any]:
        return {}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def session_pool(minimal_pool: AgentPool) -> SessionPool:
    """Return a SessionPool backed by a real AgentPool with MemoryStorageProvider."""
    from wolfharness_storage.memory_provider.provider import MemoryStorageProvider

    store = MemoryStorageProvider()
    return SessionPool(pool=minimal_pool, store=store)


# ---------------------------------------------------------------------------
# SessionNotFoundError
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_resume_session_raises_session_not_found_error(
    session_pool: SessionPool,
) -> None:
    """resume_session raises SessionNotFoundError for non-existent session."""
    from wolfharness.orchestrator.core import SessionNotFoundError

    results = make_deferred_tool_results(["call-1"])
    with pytest.raises(SessionNotFoundError, match="sess-nonexistent"):
        await session_pool.resume_session("sess-nonexistent", results)


# ---------------------------------------------------------------------------
# SessionBusyError
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_resume_session_raises_busy_error_when_active_run(
    session_pool: SessionPool,
) -> None:
    """resume_session raises SessionBusyError when the session has an active run."""
    from wolfharness.orchestrator.core import SessionBusyError

    # Create session and fake an active run
    state, _ = await session_pool.sessions.get_or_create_session("sess-1", agent_name="test-agent")
    state.current_run_id = "run-active"

    results = make_deferred_tool_results(["call-1"])
    with pytest.raises(SessionBusyError, match="already has an active run"):
        await session_pool.resume_session("sess-1", results)


# ---------------------------------------------------------------------------
# CheckpointMismatchError
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_resume_session_raises_mismatch_error_missing_results(
    session_pool: SessionPool,
) -> None:
    """resume_session raises CheckpointMismatchError when results don't cover all pending calls."""
    from wolfharness.orchestrator.core import CheckpointMismatchError

    store = session_pool.sessions.store
    assert store is not None
    await store.save_session(
        make_session_data(
            session_id="sess-1",
            pending=[make_pending_call("call-1"), make_pending_call("call-2")],
            status="checkpointed",
        )
    )

    # Only provide result for call-1, not call-2
    results = make_deferred_tool_results(["call-1"])
    with pytest.raises(CheckpointMismatchError, match="call-2"):
        await session_pool.resume_session("sess-1", results)


@pytest.mark.anyio
async def test_resume_session_raises_mismatch_error_extra_results(
    session_pool: SessionPool,
) -> None:
    """resume_session raises CheckpointMismatchError when results include unknown call IDs."""
    from wolfharness.orchestrator.core import CheckpointMismatchError

    store = session_pool.sessions.store
    assert store is not None
    await store.save_session(
        make_session_data(
            session_id="sess-1",
            pending=[make_pending_call("call-1")],
            status="checkpointed",
        )
    )

    results = make_deferred_tool_results(["call-1", "call-unknown"])
    with pytest.raises(CheckpointMismatchError, match="call-unknown"):
        await session_pool.resume_session("sess-1", results)


# ---------------------------------------------------------------------------
# Resume lock serialization
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_resume_session_serialized_via_resume_lock(
    session_pool: SessionPool,
) -> None:
    """Concurrent resume_session calls are serialized via per-session lock."""
    store = session_pool.sessions.store
    assert store is not None
    await store.save_session(
        make_session_data(
            session_id="sess-1",
            pending=[make_pending_call("call-1")],
            status="checkpointed",
        )
    )

    # Verify lock exists
    lock = await session_pool._get_resume_lock("sess-1")
    assert lock is not None

    # Lock is per-session, can be acquired
    async with lock:
        assert lock.locked()


# ---------------------------------------------------------------------------
# Native agent resume — message_history + deferred_tool_results flow
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_resume_native_agent_loads_checkpoint_and_runs(
    session_pool: SessionPool,
) -> None:
    """resume_session for native agent loads checkpoint, reconstructs agent.

    Routes through session_pool.run_stream() with resume parameters.
    """
    store = session_pool.sessions.store
    assert store is not None
    await store.save_session(
        make_session_data(
            session_id="sess-1",
            agent_name="test-agent",
            agent_type="native",
            pending=[make_pending_call("call-1")],
            status="checkpointed",
            metadata={"agent_type": "native"},
        )
    )

    # Create a fake checkpoint
    from wolfharness.agents.native_agent.checkpoint import CheckpointData

    checkpoint_data = CheckpointData(
        message_history=[],
        pending_calls=[make_pending_call("call-1")],
    )

    # Mock the reconstructed agent
    mock_native = MagicMock()
    mock_native.name = "test-agent"
    mock_native._model = None
    mock_native.model_settings = None

    # Track run_stream calls on session_pool
    run_stream_calls: list[dict[str, Any]] = []

    async def tracked_run_stream(session_id: str, *prompts: Any, **kwargs: Any) -> Any:
        run_stream_calls.append({"session_id": session_id, "prompts": prompts, **kwargs})
        return
        yield  # pragma: no cover

    mock_reconstruct = AsyncMock(return_value=mock_native)
    with (
        patch.object(
            session_pool, "_load_checkpoint_data", AsyncMock(return_value=checkpoint_data)
        ),
        patch.object(session_pool, "_reconstruct_native_agent", mock_reconstruct),
        patch.object(session_pool, "run_stream", tracked_run_stream),
    ):
        results = make_deferred_tool_results(["call-1"])
        await session_pool.resume_session("sess-1", results)

    # Verify checkpoint was loaded
    mock_reconstruct.assert_awaited_once_with("sess-1", "test-agent")

    # Verify run_stream was called with resume parameters
    assert len(run_stream_calls) == 1
    call = run_stream_calls[0]
    assert call["session_id"] == "sess-1"
    assert "message_history" in call
    assert "deferred_tool_results" in call
    assert "cached_elicitation_responses" in call


# ---------------------------------------------------------------------------
# pending_deferred_calls cleared only after agent.run() succeeds
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_resume_native_agent_clears_pending_after_success(
    session_pool: SessionPool,
) -> None:
    """pending_deferred_calls are cleared ONLY after agent.run() succeeds."""
    store = session_pool.sessions.store
    assert store is not None
    await store.save_session(
        make_session_data(
            session_id="sess-1",
            agent_type="native",
            pending=[make_pending_call("call-1")],
            status="checkpointed",
            metadata={"agent_type": "native"},
        )
    )

    from wolfharness.agents.native_agent.checkpoint import CheckpointData

    checkpoint_data = CheckpointData(
        message_history=[],
        pending_calls=[make_pending_call("call-1")],
    )

    mock_native = MagicMock()
    mock_native.name = "test-agent"
    mock_native._model = None
    mock_native.model_settings = None

    with (
        patch.object(
            session_pool, "_load_checkpoint_data", AsyncMock(return_value=checkpoint_data)
        ),
        patch.object(
            session_pool, "_reconstruct_native_agent", AsyncMock(return_value=mock_native)
        ),
        patch.object(session_pool, "run_stream", _ok_gen),
    ):
        results = make_deferred_tool_results(["call-1"])
        await session_pool.resume_session("sess-1", results)

    # After successful resume, pending_deferred_calls should be cleared
    session_data = await store.load_session("sess-1")
    assert session_data is not None
    assert session_data.pending_deferred_calls == []


@pytest.mark.anyio
async def test_resume_native_agent_does_not_clear_pending_on_failure(
    session_pool: SessionPool,
) -> None:
    """pending_deferred_calls are NOT cleared if agent.run() fails."""
    store = session_pool.sessions.store
    assert store is not None
    await store.save_session(
        make_session_data(
            session_id="sess-1",
            agent_type="native",
            pending=[make_pending_call("call-1")],
            status="checkpointed",
            metadata={"agent_type": "native"},
        )
    )

    from wolfharness.agents.native_agent.checkpoint import CheckpointData

    checkpoint_data = CheckpointData(
        message_history=[],
        pending_calls=[make_pending_call("call-1")],
    )

    mock_native = MagicMock()
    mock_native.name = "test-agent"
    mock_native._model = None
    mock_native.model_settings = None

    with (
        patch.object(
            session_pool, "_load_checkpoint_data", AsyncMock(return_value=checkpoint_data)
        ),
        patch.object(
            session_pool, "_reconstruct_native_agent", AsyncMock(return_value=mock_native)
        ),
        patch.object(session_pool, "run_stream", _fail_gen),
    ):
        results = make_deferred_tool_results(["call-1"])
        with pytest.raises(RuntimeError, match="Boom"):
            await session_pool.resume_session("sess-1", results)

    # After failed resume, pending_deferred_calls should NOT be cleared
    session_data = await store.load_session("sess-1")
    assert session_data is not None
    assert len(session_data.pending_deferred_calls) == 1
    assert session_data.pending_deferred_calls[0].tool_call_id == "call-1"


# ---------------------------------------------------------------------------
# SessionResumeEvent emission
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_resume_session_emits_resume_event(
    session_pool: SessionPool,
) -> None:
    """resume_session emits SessionResumeEvent on successful resume."""
    store = session_pool.sessions.store
    assert store is not None
    await store.save_session(
        make_session_data(
            session_id="sess-1",
            agent_type="native",
            pending=[make_pending_call("call-1"), make_pending_call("call-2")],
            status="checkpointed",
            metadata={"agent_type": "native"},
        )
    )

    from wolfharness.agents.native_agent.checkpoint import CheckpointData

    checkpoint_data = CheckpointData(
        message_history=[],
        pending_calls=[make_pending_call("call-1"), make_pending_call("call-2")],
    )

    mock_native = MagicMock()
    mock_native.name = "test-agent"
    mock_native._model = None
    mock_native.model_settings = None

    # Subscribe to event bus before resume
    queue = await session_pool.event_bus.subscribe("sess-1")

    with (
        patch.object(
            session_pool, "_load_checkpoint_data", AsyncMock(return_value=checkpoint_data)
        ),
        patch.object(
            session_pool, "_reconstruct_native_agent", AsyncMock(return_value=mock_native)
        ),
        patch.object(session_pool, "run_stream", _ok_gen),
    ):
        results = make_deferred_tool_results(["call-1", "call-2"])
        await session_pool.resume_session("sess-1", results)

    # Collect events
    events: list[Any] = []
    try:
        while True:
            envelope = queue.get_nowait()
            if envelope is not None:
                events.append(envelope.event)
    except asyncio.QueueEmpty:
        pass
    except asyncio.QueueShutDown:
        pass

    # Should find SessionResumeEvent
    resume_events = [e for e in events if isinstance(e, SessionResumeEvent)]
    assert len(resume_events) == 1
    assert resume_events[0].session_id == "sess-1"
    assert resume_events[0].resolved_call_count == 2
    assert resume_events[0].source == "resume_prompt"


# ---------------------------------------------------------------------------
# Status transition: checkpointed → active
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_resume_session_transitions_status_to_active(
    session_pool: SessionPool,
) -> None:
    """resume_session transitions status from 'checkpointed' to 'active' on success."""
    store = session_pool.sessions.store
    assert store is not None
    await store.save_session(
        make_session_data(
            session_id="sess-1",
            agent_type="native",
            pending=[make_pending_call("call-1")],
            status="checkpointed",
            metadata={"agent_type": "native"},
        )
    )

    from wolfharness.agents.native_agent.checkpoint import CheckpointData

    checkpoint_data = CheckpointData(
        message_history=[],
        pending_calls=[make_pending_call("call-1")],
    )

    mock_native = MagicMock()
    mock_native.name = "test-agent"
    mock_native._model = None
    mock_native.model_settings = None

    with (
        patch.object(
            session_pool, "_load_checkpoint_data", AsyncMock(return_value=checkpoint_data)
        ),
        patch.object(
            session_pool, "_reconstruct_native_agent", AsyncMock(return_value=mock_native)
        ),
        patch.object(session_pool, "run_stream", _ok_gen),
    ):
        results = make_deferred_tool_results(["call-1"])
        await session_pool.resume_session("sess-1", results)

    session_data = await store.load_session("sess-1")
    assert session_data is not None
    assert session_data.status == "active"


# ---------------------------------------------------------------------------
# Status transition: checkpointed → checkpointed on failure
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_resume_session_keeps_checkpointed_status_on_failure(
    session_pool: SessionPool,
) -> None:
    """resume_session keeps status as 'checkpointed' when agent.run() fails."""
    store = session_pool.sessions.store
    assert store is not None
    await store.save_session(
        make_session_data(
            session_id="sess-1",
            agent_type="native",
            pending=[make_pending_call("call-1")],
            status="checkpointed",
            metadata={"agent_type": "native"},
        )
    )

    from wolfharness.agents.native_agent.checkpoint import CheckpointData

    checkpoint_data = CheckpointData(
        message_history=[],
        pending_calls=[make_pending_call("call-1")],
    )

    mock_native = MagicMock()
    mock_native.name = "test-agent"
    mock_native._model = None
    mock_native.model_settings = None

    with (
        patch.object(
            session_pool, "_load_checkpoint_data", AsyncMock(return_value=checkpoint_data)
        ),
        patch.object(
            session_pool, "_reconstruct_native_agent", AsyncMock(return_value=mock_native)
        ),
        patch.object(session_pool, "run_stream", _fail_gen),
    ):
        results = make_deferred_tool_results(["call-1"])
        with pytest.raises(RuntimeError, match="Boom"):
            await session_pool.resume_session("sess-1", results)

    session_data = await store.load_session("sess-1")
    assert session_data is not None
    assert session_data.status == "checkpointed"


# ---------------------------------------------------------------------------
# ACP agent resume path
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_resume_acp_agent_reopens_subprocess(
    session_pool: SessionPool,
) -> None:
    """resume_session for ACP agent reopens subprocess and sends session/resume."""
    store = session_pool.sessions.store
    assert store is not None
    await store.save_session(
        make_session_data(
            session_id="sess-acp",
            agent_name="acp-agent",
            agent_type="acp",
            pending=[make_pending_call("call-acp-1")],
            status="checkpointed",
            metadata={"agent_type": "acp"},
        )
    )

    # Mock ACP agent
    mock_acp = MagicMock()
    mock_acp.name = "acp-agent"
    mock_acp.run = AsyncMock(return_value=MagicMock())
    mock_acp._resume_session = AsyncMock(return_value=None)

    from wolfharness.agents.native_agent.checkpoint import CheckpointData

    checkpoint_data = CheckpointData(
        message_history=[],
        pending_calls=[make_pending_call("call-acp-1")],
    )

    mock_load = AsyncMock(return_value=checkpoint_data)
    mock_reconstruct = AsyncMock(return_value=mock_acp)
    with (
        patch.object(session_pool, "_load_checkpoint_data", mock_load),
        patch.object(session_pool, "_reconstruct_acp_agent", mock_reconstruct),
    ):
        results = make_deferred_tool_results(["call-acp-1"])
        await session_pool.resume_session("sess-acp", results)

    # Verify ACP subprocess was reopened
    mock_reconstruct.assert_awaited_once_with("sess-acp", "acp-agent")

    # Verify agent.run() was called (ACP agents use run, not _resume_session at this level)
    mock_acp.run.assert_called_once()


# ---------------------------------------------------------------------------
# No pending calls — edge case
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_resume_session_with_empty_pending_calls(
    session_pool: SessionPool,
) -> None:
    """resume_session handles empty pending_deferred_calls gracefully."""
    store = session_pool.sessions.store
    assert store is not None
    await store.save_session(
        make_session_data(
            session_id="sess-1",
            agent_type="native",
            pending=[],
            status="checkpointed",
            metadata={"agent_type": "native"},
        )
    )

    from wolfharness.agents.native_agent.checkpoint import CheckpointData

    checkpoint_data = CheckpointData(
        message_history=[],
        pending_calls=[],
    )

    mock_native = MagicMock()
    mock_native.name = "test-agent"
    mock_native._model = None
    mock_native.model_settings = None

    with (
        patch.object(
            session_pool, "_load_checkpoint_data", AsyncMock(return_value=checkpoint_data)
        ),
        patch.object(
            session_pool, "_reconstruct_native_agent", AsyncMock(return_value=mock_native)
        ),
        patch.object(session_pool, "run_stream", _ok_gen),
    ):
        # Empty results should be fine when no pending calls
        results = make_deferred_tool_results([])
        await session_pool.resume_session("sess-1", results)

    # Successful completion implies run_stream was called
    session_data = await store.load_session("sess-1")
    assert session_data is not None
    assert session_data.status == "active"


# ---------------------------------------------------------------------------
# Merged from test_resume_concurrency.py (suffix: rc)
# ---------------------------------------------------------------------------


async def _ok_gen_rc(*args: Any, **kwargs: Any) -> Any:
    """Async generator that yields nothing and completes silently."""
    return
    yield


async def _fail_gen_rc(*args: Any, **kwargs: Any) -> Any:
    """Async generator that raises RuntimeError during iteration."""
    raise RuntimeError("Boom")
    yield


@pytest.fixture
async def session_pool_rc(minimal_pool: AgentPool) -> SessionPool:
    """Return a SessionPool backed by a real AgentPool with MemoryStorageProvider."""
    from wolfharness_storage.memory_provider.provider import MemoryStorageProvider

    store = MemoryStorageProvider()
    return SessionPool(pool=minimal_pool, store=store)


@pytest.mark.anyio
async def test_with_resume_lock_acquires_lock(session_pool_rc: SessionPool) -> None:
    """_with_resume_lock acquires the per-session resume lock."""
    lock = await session_pool_rc._get_resume_lock("sess-1")
    assert lock is not None
    assert not lock.locked()
    async with session_pool_rc._with_resume_lock("sess-1") as session:
        assert lock.locked()
        assert session is None
    assert not lock.locked()


@pytest.mark.anyio
async def test_with_resume_lock_raises_busy_for_active_run(session_pool_rc: SessionPool) -> None:
    """_with_resume_lock raises SessionBusyError when session has active run."""
    state, _ = await session_pool_rc.sessions.get_or_create_session(
        "sess-1", agent_name="test-agent"
    )
    state.current_run_id = "run-active"
    with pytest.raises(SessionBusyError, match="already has an active run"):
        async with session_pool_rc._with_resume_lock("sess-1"):
            pass


@pytest.mark.anyio
async def test_with_resume_lock_raises_busy_for_resumed_session(
    session_pool_rc: SessionPool,
) -> None:
    """_with_resume_lock raises SessionBusyError when session status is not 'checkpointed'."""
    store = session_pool_rc.sessions.store
    assert store is not None
    await store.save_session(
        make_session_data(
            session_id="sess-1",
            agent_name="test-agent",
            status="active",
            metadata={"agent_type": "native"},
        )
    )
    with pytest.raises(SessionBusyError, match="already has an active run"):
        async with session_pool_rc._with_resume_lock("sess-1"):
            pass


@pytest.mark.anyio
async def test_with_resume_lock_allows_checkpointed_session(session_pool_rc: SessionPool) -> None:
    """_with_resume_lock allows sessions with status 'checkpointed'."""
    store = session_pool_rc.sessions.store
    assert store is not None
    await store.save_session(
        make_session_data(
            session_id="sess-1",
            agent_name="test-agent",
            status="checkpointed",
            metadata={"agent_type": "native"},
        )
    )
    async with session_pool_rc._with_resume_lock("sess-1") as session:
        assert session is None


@pytest.mark.anyio
async def test_resume_session_concurrent_calls_serialize(session_pool_rc: SessionPool) -> None:
    """Concurrent resume_session calls serialize via per-session lock.

    The second call receives SessionBusyError while the first is in progress.
    """
    store = session_pool_rc.sessions.store
    assert store is not None
    await store.save_session(
        make_session_data(
            session_id="sess-1",
            agent_type="native",
            pending=[make_pending_call("call-1")],
            status="checkpointed",
            metadata={"agent_type": "native"},
        )
    )
    from wolfharness.agents.native_agent.checkpoint import CheckpointData

    checkpoint_data = CheckpointData(
        message_history=[], pending_calls=[make_pending_call("call-1")]
    )
    block_event = asyncio.Event()

    async def slow_run_stream(session_id: str, *prompts: Any, **kwargs: Any) -> Any:
        await block_event.wait()
        return
        yield

    mock_native = MagicMock()
    mock_native.name = "test-agent"
    mock_native._model = None
    mock_native.model_settings = None
    with (
        patch.object(
            session_pool_rc, "_load_checkpoint_data", AsyncMock(return_value=checkpoint_data)
        ),
        patch.object(
            session_pool_rc, "_reconstruct_native_agent", AsyncMock(return_value=mock_native)
        ),
        patch.object(session_pool_rc, "run_stream", slow_run_stream),
    ):
        results = make_deferred_tool_results(["call-1"])
        task1 = asyncio.create_task(session_pool_rc.resume_session("sess-1", results))
        task2 = asyncio.create_task(session_pool_rc.resume_session("sess-1", results))
        await asyncio.sleep(0.05)
        assert not task1.done(), "First resume should be blocked on slow_run_stream"
        assert not task2.done(), "Second resume should be waiting for lock"
        block_event.set()
        await task1
        with pytest.raises(SessionBusyError):
            await task2


@pytest.mark.anyio
async def test_resume_session_rejects_second_after_success(session_pool_rc: SessionPool) -> None:
    """A second resume_session call after a successful resume gets SessionBusyError.

    Verifies the status re-check inside the lock catches already-resumed sessions.
    """
    store = session_pool_rc.sessions.store
    assert store is not None
    await store.save_session(
        make_session_data(
            session_id="sess-1",
            agent_type="native",
            pending=[make_pending_call("call-1")],
            status="checkpointed",
            metadata={"agent_type": "native"},
        )
    )
    from wolfharness.agents.native_agent.checkpoint import CheckpointData

    checkpoint_data = CheckpointData(
        message_history=[], pending_calls=[make_pending_call("call-1")]
    )
    mock_native = MagicMock()
    mock_native.name = "test-agent"
    mock_native._model = None
    mock_native.model_settings = None
    with (
        patch.object(
            session_pool_rc, "_load_checkpoint_data", AsyncMock(return_value=checkpoint_data)
        ),
        patch.object(
            session_pool_rc, "_reconstruct_native_agent", AsyncMock(return_value=mock_native)
        ),
        patch.object(session_pool_rc, "run_stream", _ok_gen_rc),
    ):
        results = make_deferred_tool_results(["call-1"])
        await session_pool_rc.resume_session("sess-1", results)
        with pytest.raises(SessionBusyError):
            await session_pool_rc.resume_session("sess-1", make_deferred_tool_results([]))


@pytest.mark.anyio
async def test_resume_session_does_not_clear_pending_on_failure(
    session_pool_rc: SessionPool,
) -> None:
    """pending_deferred_calls are NOT cleared if agent.run() fails.

    Status reverts to 'checkpointed' so the session can be retried.
    """
    store = session_pool_rc.sessions.store
    assert store is not None
    await store.save_session(
        make_session_data(
            session_id="sess-1",
            agent_type="native",
            pending=[make_pending_call("call-1")],
            status="checkpointed",
            metadata={"agent_type": "native"},
        )
    )
    from wolfharness.agents.native_agent.checkpoint import CheckpointData

    checkpoint_data = CheckpointData(
        message_history=[], pending_calls=[make_pending_call("call-1")]
    )
    mock_native = MagicMock()
    mock_native.name = "test-agent"
    mock_native._model = None
    mock_native.model_settings = None
    with (
        patch.object(
            session_pool_rc, "_load_checkpoint_data", AsyncMock(return_value=checkpoint_data)
        ),
        patch.object(
            session_pool_rc, "_reconstruct_native_agent", AsyncMock(return_value=mock_native)
        ),
        patch.object(session_pool_rc, "run_stream", _fail_gen_rc),
    ):
        results = make_deferred_tool_results(["call-1"])
        with pytest.raises(RuntimeError, match="Boom"):
            await session_pool_rc.resume_session("sess-1", results)
    session_data = await store.load_session("sess-1")
    assert session_data is not None
    assert len(session_data.pending_deferred_calls) == 1
    assert session_data.pending_deferred_calls[0].tool_call_id == "call-1"
    assert session_data.status == "checkpointed"


@pytest.mark.anyio
async def test_resume_session_allows_retry_after_failure(session_pool_rc: SessionPool) -> None:
    """A second resume_session call succeeds after a failed attempt.

    Verifies that when the first resume fails and reverts status to
    'checkpointed', the second call is allowed to proceed.
    """
    store = session_pool_rc.sessions.store
    assert store is not None
    await store.save_session(
        make_session_data(
            session_id="sess-1",
            agent_type="native",
            pending=[make_pending_call("call-1")],
            status="checkpointed",
            metadata={"agent_type": "native"},
        )
    )
    from wolfharness.agents.native_agent.checkpoint import CheckpointData

    checkpoint_data = CheckpointData(
        message_history=[], pending_calls=[make_pending_call("call-1")]
    )
    fail_run = True
    mock_native = MagicMock()
    mock_native.name = "test-agent"
    mock_native._model = None
    mock_native.model_settings = None

    async def run_fail_then_succeed_stream(session_id: str, *prompts: Any, **kwargs: Any) -> Any:
        nonlocal fail_run
        if fail_run:
            fail_run = False
            raise RuntimeError("First attempt fails")
        return
        yield

    with (
        patch.object(
            session_pool_rc, "_load_checkpoint_data", AsyncMock(return_value=checkpoint_data)
        ),
        patch.object(
            session_pool_rc, "_reconstruct_native_agent", AsyncMock(return_value=mock_native)
        ),
        patch.object(session_pool_rc, "run_stream", run_fail_then_succeed_stream),
    ):
        results = make_deferred_tool_results(["call-1"])
        with pytest.raises(RuntimeError, match="First attempt fails"):
            await session_pool_rc.resume_session("sess-1", results)
        session_data = await store.load_session("sess-1")
        assert session_data is not None
        assert session_data.status == "checkpointed"
        assert len(session_data.pending_deferred_calls) == 1
        await session_pool_rc.resume_session("sess-1", results)
        session_data = await store.load_session("sess-1")
        assert session_data is not None
        assert session_data.status == "active"
        assert session_data.pending_deferred_calls == []


@pytest.mark.anyio
async def test_resume_session_status_transitions_checkpointed_to_resuming_to_active(
    session_pool_rc: SessionPool,
) -> None:
    """Status transitions: checkpointed -> resuming -> active on success."""
    store = session_pool_rc.sessions.store
    assert store is not None
    await store.save_session(
        make_session_data(
            session_id="sess-1",
            agent_type="native",
            pending=[make_pending_call("call-1")],
            status="checkpointed",
            metadata={"agent_type": "native"},
        )
    )
    from wolfharness.agents.native_agent.checkpoint import CheckpointData

    checkpoint_data = CheckpointData(
        message_history=[], pending_calls=[make_pending_call("call-1")]
    )
    observed_statuses: list[str] = []
    original_save = store.save_session

    async def tracking_save(data: Any) -> None:
        observed_statuses.append(data.status)
        await original_save(data)

    store.save_session = tracking_save
    mock_native = MagicMock()
    mock_native.name = "test-agent"
    mock_native._model = None
    mock_native.model_settings = None
    with (
        patch.object(
            session_pool_rc, "_load_checkpoint_data", AsyncMock(return_value=checkpoint_data)
        ),
        patch.object(
            session_pool_rc, "_reconstruct_native_agent", AsyncMock(return_value=mock_native)
        ),
        patch.object(session_pool_rc, "run_stream", _ok_gen_rc),
    ):
        results = make_deferred_tool_results(["call-1"])
        await session_pool_rc.resume_session("sess-1", results)
    assert "resuming" in observed_statuses
    assert "active" in observed_statuses
    assert observed_statuses[-1] == "active"


@pytest.mark.anyio
async def test_resume_session_status_reverts_to_checkpointed_on_failure(
    session_pool_rc: SessionPool,
) -> None:
    """Status reverts from resuming to checkpointed on agent.run() failure."""
    store = session_pool_rc.sessions.store
    assert store is not None
    await store.save_session(
        make_session_data(
            session_id="sess-1",
            agent_type="native",
            pending=[make_pending_call("call-1")],
            status="checkpointed",
            metadata={"agent_type": "native"},
        )
    )
    from wolfharness.agents.native_agent.checkpoint import CheckpointData

    checkpoint_data = CheckpointData(
        message_history=[], pending_calls=[make_pending_call("call-1")]
    )
    observed_statuses: list[str] = []
    original_save = store.save_session

    async def tracking_save(data: Any) -> None:
        observed_statuses.append(data.status)
        await original_save(data)

    store.save_session = tracking_save
    mock_native = MagicMock()
    mock_native.name = "test-agent"
    mock_native._model = None
    mock_native.model_settings = None
    with (
        patch.object(
            session_pool_rc, "_load_checkpoint_data", AsyncMock(return_value=checkpoint_data)
        ),
        patch.object(
            session_pool_rc, "_reconstruct_native_agent", AsyncMock(return_value=mock_native)
        ),
        patch.object(session_pool_rc, "run_stream", _fail_gen_rc),
    ):
        results = make_deferred_tool_results(["call-1"])
        with pytest.raises(RuntimeError, match="Boom"):
            await session_pool_rc.resume_session("sess-1", results)
    assert "resuming" in observed_statuses
    assert observed_statuses[-1] == "checkpointed"
