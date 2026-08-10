"""Unit tests for SessionPool message history API (Migration B).

Tests get_messages, append_message, truncate_messages, and copy_messages.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock

import pytest

from wolfharness.messaging import ChatMessage


if TYPE_CHECKING:
    from wolfharness import AgentPool
    from wolfharness.orchestrator.core import SessionPool


pytestmark = pytest.mark.unit

_L2_SKIP = pytest.mark.skip(reason="L2 migration: requires mock internals — remains L1 unit test")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_pool(minimal_pool: AgentPool) -> AgentPool:
    """Return the real pool (storage defaults to real StorageManager)."""
    return minimal_pool


@pytest.fixture
def session_pool(mock_pool: AgentPool) -> SessionPool:
    """Return a SessionPool backed by the real pool."""
    assert mock_pool.session_pool is not None
    return mock_pool.session_pool


@pytest.fixture
def sample_message() -> ChatMessage[str]:
    """Return a sample ChatMessage for testing."""
    return ChatMessage(content="hello", role="user", session_id="sess-1")


# ---------------------------------------------------------------------------
# get_messages
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_get_messages_returns_storage_messages(
    session_pool: SessionPool,
    mock_pool: AgentPool,
) -> None:
    """get_messages returns messages from real storage after append."""
    await session_pool.sessions.get_or_create_session("sess-1", agent_name="agent-a")
    msg = ChatMessage(content="hi", role="user", session_id="sess-1")
    await session_pool.append_message("sess-1", msg)

    result = await session_pool.get_messages("sess-1")

    assert any(m.content == "hi" for m in result)


@pytest.mark.anyio
async def test_get_messages_raises_keyerror_for_missing_session(
    session_pool: SessionPool,
) -> None:
    """get_messages raises KeyError when the session does not exist."""
    with pytest.raises(KeyError, match="missing-sess"):
        await session_pool.get_messages("missing-sess")


@pytest.mark.anyio
async def test_get_messages_empty_when_no_storage(
    session_pool: SessionPool,
    mock_pool: AgentPool,
) -> None:
    """get_messages returns an empty list when storage is None."""
    await session_pool.sessions.get_or_create_session("sess-1", agent_name="agent-a")
    mock_pool.storage = None

    result = await session_pool.get_messages("sess-1")

    assert result == []


# ---------------------------------------------------------------------------
# append_message
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_append_message_logs_to_storage(
    session_pool: SessionPool,
    mock_pool: AgentPool,
    sample_message: ChatMessage[str],
) -> None:
    """append_message stores the message and returns the message ID."""
    await session_pool.sessions.get_or_create_session("sess-1", agent_name="agent-a")

    result = await session_pool.append_message("sess-1", sample_message)

    assert result == sample_message.message_id
    # Verify the message was actually stored by reading it back
    msgs = await session_pool.get_messages("sess-1")
    assert any(m.message_id == sample_message.message_id for m in msgs)


@pytest.mark.anyio
async def test_append_message_raises_keyerror_for_missing_session(
    session_pool: SessionPool,
    sample_message: ChatMessage[str],
) -> None:
    """append_message raises KeyError when the session does not exist."""
    with pytest.raises(KeyError, match="missing-sess"):
        await session_pool.append_message("missing-sess", sample_message)


@pytest.mark.anyio
async def test_append_message_without_storage(
    session_pool: SessionPool,
    mock_pool: AgentPool,
    sample_message: ChatMessage[str],
) -> None:
    """append_message returns message_id even when storage is None."""
    await session_pool.sessions.get_or_create_session("sess-1", agent_name="agent-a")
    mock_pool.storage = None

    result = await session_pool.append_message("sess-1", sample_message)

    assert result == sample_message.message_id


# ---------------------------------------------------------------------------
# copy_messages
# ---------------------------------------------------------------------------


@_L2_SKIP
@pytest.mark.anyio
async def test_copy_messages_forks_via_storage(
    session_pool: SessionPool,
    mock_pool: AgentPool,
) -> None:
    """copy_messages forwards to storage.fork_conversation and returns fork point."""
    await session_pool.sessions.get_or_create_session("sess-1", agent_name="agent-a")
    await session_pool.sessions.get_or_create_session("sess-2", agent_name="agent-a")
    mock_pool.storage.fork_conversation = AsyncMock(return_value="fork-point-id")

    result = await session_pool.copy_messages("sess-1", "sess-2")

    assert result == "fork-point-id"
    mock_pool.storage.fork_conversation.assert_awaited_once_with(
        source_session_id="sess-1",
        new_session_id="sess-2",
        fork_from_message_id=None,
    )


@_L2_SKIP
@pytest.mark.anyio
async def test_copy_messages_with_up_to_message_id(
    session_pool: SessionPool,
    mock_pool: AgentPool,
) -> None:
    """copy_messages passes up_to_message_id as fork_from_message_id."""
    await session_pool.sessions.get_or_create_session("sess-1", agent_name="agent-a")
    await session_pool.sessions.get_or_create_session("sess-2", agent_name="agent-a")
    mock_pool.storage.fork_conversation = AsyncMock(return_value="msg-123")

    result = await session_pool.copy_messages("sess-1", "sess-2", up_to_message_id="msg-123")

    assert result == "msg-123"
    mock_pool.storage.fork_conversation.assert_awaited_once_with(
        source_session_id="sess-1",
        new_session_id="sess-2",
        fork_from_message_id="msg-123",
    )


@pytest.mark.anyio
async def test_copy_messages_raises_keyerror_for_missing_source(
    session_pool: SessionPool,
) -> None:
    """copy_messages raises KeyError when the source session does not exist."""
    await session_pool.sessions.get_or_create_session("sess-2", agent_name="agent-a")

    with pytest.raises(KeyError, match="missing-source"):
        await session_pool.copy_messages("missing-source", "sess-2")


@pytest.mark.anyio
async def test_copy_messages_raises_keyerror_for_missing_target(
    session_pool: SessionPool,
) -> None:
    """copy_messages raises KeyError when the target session does not exist."""
    await session_pool.sessions.get_or_create_session("sess-1", agent_name="agent-a")

    with pytest.raises(KeyError, match="missing-target"):
        await session_pool.copy_messages("sess-1", "missing-target")


@pytest.mark.anyio
async def test_copy_messages_without_storage(
    session_pool: SessionPool,
    mock_pool: AgentPool,
) -> None:
    """copy_messages returns None when storage is None."""
    await session_pool.sessions.get_or_create_session("sess-1", agent_name="agent-a")
    await session_pool.sessions.get_or_create_session("sess-2", agent_name="agent-a")
    mock_pool.storage = None

    result = await session_pool.copy_messages("sess-1", "sess-2")

    assert result is None


# ---------------------------------------------------------------------------
# truncate_messages
# ---------------------------------------------------------------------------


@_L2_SKIP
@pytest.mark.anyio
async def test_truncate_messages_calls_storage(
    session_pool: SessionPool,
    mock_pool: AgentPool,
) -> None:
    """truncate_messages forwards to storage.truncate_messages and returns count."""
    await session_pool.sessions.get_or_create_session("sess-1", agent_name="agent-a")
    mock_pool.storage.truncate_messages = AsyncMock(return_value=3)

    result = await session_pool.truncate_messages("sess-1", "msg-456")

    assert result == 3
    mock_pool.storage.truncate_messages.assert_awaited_once_with("sess-1", "msg-456")


@pytest.mark.anyio
async def test_truncate_messages_raises_keyerror_for_missing_session(
    session_pool: SessionPool,
) -> None:
    """truncate_messages raises KeyError when the session does not exist."""
    with pytest.raises(KeyError, match="missing-sess"):
        await session_pool.truncate_messages("missing-sess", "msg-123")


@pytest.mark.anyio
async def test_truncate_messages_without_storage(
    session_pool: SessionPool,
    mock_pool: AgentPool,
) -> None:
    """truncate_messages returns 0 when storage is None."""
    await session_pool.sessions.get_or_create_session("sess-1", agent_name="agent-a")
    mock_pool.storage = None

    result = await session_pool.truncate_messages("sess-1", "msg-123")

    assert result == 0


# ---------------------------------------------------------------------------
# Cache behaviour
# ---------------------------------------------------------------------------


@_L2_SKIP
@pytest.mark.anyio
async def test_cache_get_messages_returns_cached_data(
    session_pool: SessionPool,
    mock_pool: AgentPool,
) -> None:
    """Second call to get_messages uses cache; storage is only hit once."""
    await session_pool.sessions.get_or_create_session("sess-1", agent_name="agent-a")
    msg = ChatMessage(content="cached", role="user", session_id="sess-1")
    mock_pool.storage.get_session_messages = AsyncMock(return_value=[msg])

    first = await session_pool.get_messages("sess-1")
    second = await session_pool.get_messages("sess-1")

    assert first == [msg]
    assert second == [msg]
    mock_pool.storage.get_session_messages.assert_awaited_once_with("sess-1")


@_L2_SKIP
@pytest.mark.anyio
async def test_cache_append_message_invalidates_cache(
    session_pool: SessionPool,
    mock_pool: AgentPool,
) -> None:
    """append_message invalidates cache so the next get_messages hits storage."""
    await session_pool.sessions.get_or_create_session("sess-1", agent_name="agent-a")
    original = ChatMessage(content="original", role="user", session_id="sess-1")
    updated = ChatMessage(content="updated", role="assistant", session_id="sess-1")
    mock_pool.storage.get_session_messages = AsyncMock(
        side_effect=[[original], [original, updated]]
    )

    first = await session_pool.get_messages("sess-1")
    assert first == [original]

    await session_pool.append_message("sess-1", updated)

    second = await session_pool.get_messages("sess-1")
    assert second == [original, updated]
    assert mock_pool.storage.get_session_messages.await_count == 2


@_L2_SKIP
@pytest.mark.anyio
async def test_cache_truncate_messages_invalidates_cache(
    session_pool: SessionPool,
    mock_pool: AgentPool,
) -> None:
    """truncate_messages invalidates cache so the next get_messages hits storage."""
    await session_pool.sessions.get_or_create_session("sess-1", agent_name="agent-a")
    original = ChatMessage(content="original", role="user", session_id="sess-1")
    truncated = ChatMessage(content="truncated", role="user", session_id="sess-1")
    mock_pool.storage.get_session_messages = AsyncMock(side_effect=[[original], [truncated]])

    first = await session_pool.get_messages("sess-1")
    assert first == [original]

    await session_pool.truncate_messages("sess-1", "msg-123")

    second = await session_pool.get_messages("sess-1")
    assert second == [truncated]
    assert mock_pool.storage.get_session_messages.await_count == 2


@_L2_SKIP
@pytest.mark.anyio
async def test_cache_copy_messages_invalidates_target_cache(
    session_pool: SessionPool,
    mock_pool: AgentPool,
) -> None:
    """copy_messages invalidates target session cache."""
    await session_pool.sessions.get_or_create_session("sess-1", agent_name="agent-a")
    await session_pool.sessions.get_or_create_session("sess-2", agent_name="agent-a")
    target_before = ChatMessage(content="before", role="user", session_id="sess-2")
    target_after = ChatMessage(content="after", role="user", session_id="sess-2")
    mock_pool.storage.get_session_messages = AsyncMock(
        side_effect=[[target_before], [target_after]]
    )

    first = await session_pool.get_messages("sess-2")
    assert first == [target_before]

    await session_pool.copy_messages("sess-1", "sess-2")

    second = await session_pool.get_messages("sess-2")
    assert second == [target_after]
    assert mock_pool.storage.get_session_messages.await_count == 2


# ---------------------------------------------------------------------------
# RunHandle delegation tests (feature-flag gated)
# ---------------------------------------------------------------------------

from wolfharness.lifecycle import RunState  # noqa: E402
from wolfharness.orchestrator.run import RunHandle  # noqa: E402


def _make_mock_agent() -> MagicMock:
    """Return a MagicMock simulating a native Agent."""
    agent = MagicMock()
    agent.AGENT_TYPE = "native"
    return agent


def _setup_session_with_agent(
    session_pool: SessionPool,
    session_id: str,
    agent: MagicMock,
) -> MagicMock:
    """Create a session and register an agent for it on the SessionController."""
    import asyncio

    session = MagicMock()
    session.session_id = session_id
    session.current_run_id = None
    session.closing = False
    session.is_closing = False
    session._request_lock = asyncio.Lock()
    session.turn_lock = asyncio.Lock()
    session.input_provider = None
    session.agent = agent
    session_pool.sessions._sessions[session_id] = session
    session_pool.sessions._session_agents[session_id] = agent
    return session


def _setup_active_run(
    session_pool: SessionPool,
    session_id: str,
) -> MagicMock:
    """Register a mock RunHandle as the active run for a session."""
    run_handle = MagicMock(spec=RunHandle)
    run_handle.steer = MagicMock(return_value=True)
    run_handle.followup = MagicMock(return_value=True)
    run_handle.run_id = "test-run-id"
    run_handle._run_state = RunState.RUNNING
    session_pool.sessions._runs["test-run-id"] = run_handle
    session = session_pool.sessions.get_session(session_id)
    assert session is not None
    session.current_run_id = "test-run-id"
    return run_handle


# === receive_request ===


@_L2_SKIP
@pytest.mark.anyio
async def test_receive_request_flag_on_delegates_to_session_controller(
    session_pool: SessionPool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When flag is ON, receive_request delegates to SessionController which uses RunHandle."""
    monkeypatch.setenv("AGENTPOOL_USE_RUN_TURN", "true")
    agent = _make_mock_agent()
    _setup_session_with_agent(session_pool, "sess-rr-1", agent)
    session_pool.sessions._use_run_turn = lambda _agent: True  # type: ignore[method-assign]
    session_pool.sessions._consume_run = AsyncMock(return_value=None)  # type: ignore[method-assign]

    result = await session_pool.send_message("sess-rr-1", "hello")

    assert result is not None
    assert isinstance(result, str)  # message_id
    # Verify a RunHandle was created
    session = session_pool.sessions.get_session("sess-rr-1")
    assert session is not None
    assert session.current_run_id is not None
    run_handle = session_pool.sessions._runs.get(session.current_run_id)
    assert run_handle is not None
    assert run_handle.agent is agent


# === process_prompt ===


@_L2_SKIP
@pytest.mark.anyio
async def test_process_prompt_flag_on_delegates_to_run_handle(
    session_pool: SessionPool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When flag is ON, process_prompt uses RunHandle.start()."""
    monkeypatch.setenv("AGENTPOOL_USE_RUN_TURN", "true")
    agent = _make_mock_agent()
    _setup_session_with_agent(session_pool, "sess-pp-1", agent)
    session_pool._use_run_turn_for_session = lambda _sid: True  # type: ignore[method-assign]

    # Mock get_or_create_session to return the session we set up
    session = session_pool.sessions.get_session("sess-pp-1")
    session_pool.sessions.get_or_create_session = AsyncMock(  # type: ignore[method-assign]
        return_value=(session, True)
    )

    # Mock _create_run_handle to return a mock RunHandle
    mock_run = MagicMock(spec=RunHandle)
    mock_run.run_id = "test-run-pp-1"
    mock_run.start = MagicMock(return_value=_empty_async_gen())
    session_pool._create_run_handle = MagicMock(return_value=mock_run)  # type: ignore[method-assign]
    session_pool.sessions._runs["test-run-pp-1"] = mock_run

    await session_pool.process_prompt("sess-pp-1", "test prompt")

    mock_run.start.assert_called_once_with("test prompt")


# === run_stream ===


@_L2_SKIP
@pytest.mark.anyio
async def test_run_stream_flag_on_delegates_to_run_handle(
    session_pool: SessionPool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When flag is ON, run_stream yields from RunHandle.start()."""
    monkeypatch.setenv("AGENTPOOL_USE_RUN_TURN", "true")
    agent = _make_mock_agent()
    _setup_session_with_agent(session_pool, "sess-rs-1", agent)
    session_pool._use_run_turn_for_session = lambda _sid: True  # type: ignore[method-assign]

    session = session_pool.sessions.get_session("sess-rs-1")
    session_pool.sessions.get_or_create_session = AsyncMock(  # type: ignore[method-assign]
        return_value=(session, True)
    )

    # Mock _create_run_handle
    mock_run = MagicMock(spec=RunHandle)
    mock_run.run_id = "test-run-rs-1"
    mock_run.start = MagicMock(return_value=_event_async_gen(["event1", "event2"]))
    session_pool._create_run_handle = MagicMock(return_value=mock_run)  # type: ignore[method-assign]
    session_pool.sessions._runs["test-run-rs-1"] = mock_run

    events = [event async for event in session_pool.run_stream("sess-rs-1", "prompt")]

    assert events == ["event1", "event2"]
    mock_run.start.assert_called_once_with("prompt")


# === inject_prompt ===


@_L2_SKIP
@pytest.mark.anyio
async def test_inject_prompt_flag_on_delegates_to_run_handle_steer(
    session_pool: SessionPool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When flag is ON, inject_prompt delegates to RunHandle.steer()."""
    monkeypatch.setenv("AGENTPOOL_USE_RUN_TURN", "true")
    agent = _make_mock_agent()
    _setup_session_with_agent(session_pool, "sess-ip-1", agent)
    session_pool._use_run_turn_for_session = lambda _sid: True  # type: ignore[method-assign]
    run_handle = _setup_active_run(session_pool, "sess-ip-1")

    result = await session_pool.inject_prompt("sess-ip-1", "urgent message")

    assert result is True
    run_handle.steer.assert_called_once_with("urgent message")


@_L2_SKIP
@pytest.mark.anyio
async def test_inject_prompt_flag_on_no_run_returns_none(
    session_pool: SessionPool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When flag is ON and no active run, inject_prompt returns None."""
    monkeypatch.setenv("AGENTPOOL_USE_RUN_TURN", "true")
    agent = _make_mock_agent()
    _setup_session_with_agent(session_pool, "sess-ip-2", agent)
    session_pool._use_run_turn_for_session = lambda _sid: True  # type: ignore[method-assign]

    result = await session_pool.inject_prompt("sess-ip-2", "message")

    assert result is None


# === queue_prompt ===


@_L2_SKIP
@pytest.mark.anyio
async def test_queue_prompt_flag_on_delegates_to_run_handle_followup(
    session_pool: SessionPool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When flag is ON, queue_prompt delegates to RunHandle.followup()."""
    monkeypatch.setenv("AGENTPOOL_USE_RUN_TURN", "true")
    agent = _make_mock_agent()
    _setup_session_with_agent(session_pool, "sess-qp-1", agent)
    session_pool._use_run_turn_for_session = lambda _sid: True  # type: ignore[method-assign]
    run_handle = _setup_active_run(session_pool, "sess-qp-1")

    result = await session_pool.queue_prompt("sess-qp-1", "follow up")

    assert result is True
    run_handle.followup.assert_called_once_with("follow up")


@_L2_SKIP
@pytest.mark.anyio
async def test_queue_prompt_flag_on_no_run_returns_none(
    session_pool: SessionPool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When flag is ON and no active run, queue_prompt returns None."""
    monkeypatch.setenv("AGENTPOOL_USE_RUN_TURN", "true")
    agent = _make_mock_agent()
    _setup_session_with_agent(session_pool, "sess-qp-2", agent)
    session_pool._use_run_turn_for_session = lambda _sid: True  # type: ignore[method-assign]

    result = await session_pool.queue_prompt("sess-qp-2", "message")

    assert result is None


# === steer ===


@_L2_SKIP
@pytest.mark.anyio
async def test_steer_flag_on_delegates_to_run_handle_steer(
    session_pool: SessionPool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When flag is ON, steer delegates to RunHandle.steer()."""
    monkeypatch.setenv("AGENTPOOL_USE_RUN_TURN", "true")
    agent = _make_mock_agent()
    _setup_session_with_agent(session_pool, "sess-st-1", agent)
    session_pool._use_run_turn_for_session = lambda _sid: True  # type: ignore[method-assign]
    run_handle = _setup_active_run(session_pool, "sess-st-1")

    result = await session_pool.steer("sess-st-1", "steer message")

    assert result is True
    run_handle.steer.assert_called_once_with("steer message")


@_L2_SKIP
@pytest.mark.anyio
async def test_steer_flag_on_no_run_returns_none(
    session_pool: SessionPool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When flag is ON and no active run, steer returns None."""
    monkeypatch.setenv("AGENTPOOL_USE_RUN_TURN", "true")
    agent = _make_mock_agent()
    _setup_session_with_agent(session_pool, "sess-st-2", agent)
    session_pool._use_run_turn_for_session = lambda _sid: True  # type: ignore[method-assign]

    result = await session_pool.steer("sess-st-2", "message")

    assert result is None


# === followup ===


@_L2_SKIP
@pytest.mark.anyio
async def test_followup_flag_on_delegates_to_run_handle_followup(
    session_pool: SessionPool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When flag is ON, followup delegates to RunHandle.followup()."""
    monkeypatch.setenv("AGENTPOOL_USE_RUN_TURN", "true")
    agent = _make_mock_agent()
    _setup_session_with_agent(session_pool, "sess-fu-1", agent)
    session_pool._use_run_turn_for_session = lambda _sid: True  # type: ignore[method-assign]
    run_handle = _setup_active_run(session_pool, "sess-fu-1")

    result = await session_pool.followup("sess-fu-1", "followup message")

    assert result is True
    run_handle.followup.assert_called_once_with("followup message")


@_L2_SKIP
@pytest.mark.anyio
async def test_followup_flag_on_no_run_returns_none(
    session_pool: SessionPool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When flag is ON and no active run, followup returns None."""
    monkeypatch.setenv("AGENTPOOL_USE_RUN_TURN", "true")
    agent = _make_mock_agent()
    _setup_session_with_agent(session_pool, "sess-fu-2", agent)
    session_pool._use_run_turn_for_session = lambda _sid: True  # type: ignore[method-assign]

    result = await session_pool.followup("sess-fu-2", "message")

    assert result is None


# === Helpers ===


async def _empty_async_gen():
    """Async generator that yields nothing."""
    return
    yield  # type: ignore[unreachable]  # makes this an async generator


async def _event_async_gen(events: list[str]):
    """Async generator that yields the given events."""
    for event in events:
        yield event
