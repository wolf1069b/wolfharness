"""Unit tests for SessionPool public API.

Tests send_message, run_agent, revoke_message, wait_for_completion
(Task 11 — D19/D20/D21/D25).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from wolfharness.agents.events import RunErrorEvent, StreamCompleteEvent
from wolfharness.lifecycle.types import DeliveryMode
from wolfharness.messaging import ChatMessage
from wolfharness.orchestrator.session_controller import SessionNotFoundError


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
    """Return the real pool for integration testing."""
    return minimal_pool


@pytest.fixture
def session_pool(mock_pool: AgentPool) -> SessionPool:
    """Return a SessionPool backed by the real pool."""
    assert mock_pool.session_pool is not None
    return mock_pool.session_pool


# ---------------------------------------------------------------------------
# send_message
# ---------------------------------------------------------------------------


@_L2_SKIP
@pytest.mark.anyio
async def test_send_message_queue_mode(session_pool: SessionPool) -> None:
    """send_message with QUEUE mode delegates to _route_message with when_idle priority."""
    mock_session = MagicMock()
    mock_agent = MagicMock()
    session_pool.sessions.get_session = MagicMock(return_value=mock_session)  # type: ignore[method-assign]
    session_pool.sessions.get_or_create_session_agent = AsyncMock(return_value=mock_agent)  # type: ignore[method-assign]
    session_pool.sessions._route_message = AsyncMock(return_value="msg-123")  # type: ignore[method-assign]

    result = await session_pool.send_message("sess-1", "hello", mode=DeliveryMode.QUEUE)

    assert result == "msg-123"
    session_pool.sessions._route_message.assert_awaited_once_with(
        mock_session,
        mock_agent,
        "sess-1",
        "hello",
        priority="when_idle",
        deps=None,
        message_id=None,
    )


@_L2_SKIP
@pytest.mark.anyio
async def test_send_message_steer_mode(session_pool: SessionPool) -> None:
    """send_message with STEER mode delegates to _route_message with asap priority."""
    mock_session = MagicMock()
    mock_agent = MagicMock()
    session_pool.sessions.get_session = MagicMock(return_value=mock_session)  # type: ignore[method-assign]
    session_pool.sessions.get_or_create_session_agent = AsyncMock(return_value=mock_agent)  # type: ignore[method-assign]
    session_pool.sessions._route_message = AsyncMock(return_value="msg-456")  # type: ignore[method-assign]

    result = await session_pool.send_message("sess-1", "steer me", mode=DeliveryMode.STEER)

    assert result == "msg-456"
    session_pool.sessions._route_message.assert_awaited_once_with(
        mock_session,
        mock_agent,
        "sess-1",
        "steer me",
        priority="asap",
        deps=None,
        message_id=None,
    )


@_L2_SKIP
@pytest.mark.anyio
async def test_send_message_with_message_id(session_pool: SessionPool) -> None:
    """send_message passes explicit message_id through to _route_message."""
    mock_session = MagicMock()
    mock_agent = MagicMock()
    session_pool.sessions.get_session = MagicMock(return_value=mock_session)  # type: ignore[method-assign]
    session_pool.sessions.get_or_create_session_agent = AsyncMock(return_value=mock_agent)  # type: ignore[method-assign]
    session_pool.sessions._route_message = AsyncMock(return_value="custom-id")  # type: ignore[method-assign]

    result = await session_pool.send_message(
        "sess-1",
        "hello",
        message_id="custom-id",
    )

    assert result == "custom-id"
    session_pool.sessions._route_message.assert_awaited_once_with(
        mock_session,
        mock_agent,
        "sess-1",
        "hello",
        priority="when_idle",
        deps=None,
        message_id="custom-id",
    )


@_L2_SKIP
@pytest.mark.anyio
async def test_send_message_list_content(session_pool: SessionPool) -> None:
    """send_message accepts list content and passes it through without stringification."""
    mock_session = MagicMock()
    mock_agent = MagicMock()
    session_pool.sessions.get_session = MagicMock(return_value=mock_session)  # type: ignore[method-assign]
    session_pool.sessions.get_or_create_session_agent = AsyncMock(return_value=mock_agent)  # type: ignore[method-assign]
    session_pool.sessions._route_message = AsyncMock(return_value="msg-789")  # type: ignore[method-assign]
    content: list[Any] = [{"type": "text", "text": "block1"}, {"type": "text", "text": "block2"}]

    result = await session_pool.send_message("sess-1", content)

    assert result == "msg-789"
    session_pool.sessions._route_message.assert_awaited_once_with(
        mock_session,
        mock_agent,
        "sess-1",
        content,
        priority="when_idle",
        deps=None,
        message_id=None,
    )


@pytest.mark.anyio
async def test_send_message_failure_returns_none(session_pool: SessionPool) -> None:
    """send_message returns None when session is not found."""
    # No session created — send_message should return None for unknown session
    result = await session_pool.send_message("invalid-session", "hello")

    assert result is None


@_L2_SKIP
@pytest.mark.anyio
async def test_send_message_default_mode_is_queue(session_pool: SessionPool) -> None:
    """send_message defaults to QUEUE mode when mode is not specified."""
    mock_session = MagicMock()
    mock_agent = MagicMock()
    session_pool.sessions.get_session = MagicMock(return_value=mock_session)  # type: ignore[method-assign]
    session_pool.sessions.get_or_create_session_agent = AsyncMock(return_value=mock_agent)  # type: ignore[method-assign]
    session_pool.sessions._route_message = AsyncMock(return_value="msg-default")  # type: ignore[method-assign]

    await session_pool.send_message("sess-1", "hello")

    session_pool.sessions._route_message.assert_awaited_once_with(
        mock_session,
        mock_agent,
        "sess-1",
        "hello",
        priority="when_idle",
        deps=None,
        message_id=None,
    )


# ---------------------------------------------------------------------------
# run_agent
# ---------------------------------------------------------------------------


@_L2_SKIP
@pytest.mark.anyio
async def test_run_agent_basic(session_pool: SessionPool) -> None:
    """run_agent creates session, sends message, captures event, returns text."""
    # Create the session first so create_session works
    await session_pool.sessions.get_or_create_session("sess-1", agent_name="test-agent")

    # Mock create_session to use a fixed session_id
    original_create = session_pool.create_session

    async def mock_create_session(
        session_id: str,
        agent_name: str | None = None,
        parent_session_id: str | None = None,
        lifecycle_policy: str | None = None,
        **metadata: Any,
    ) -> Any:
        return await original_create(
            session_id,
            agent_name,
            parent_session_id,
            lifecycle_policy,
            **metadata,
        )

    # Mock send_message to publish a StreamCompleteEvent to the EventBus
    final_msg = ChatMessage(content="Hello from agent", role="assistant")

    async def mock_send_message(
        session_id: str,
        content: str | list[Any],
        *,
        mode: DeliveryMode = DeliveryMode.QUEUE,
        message_id: str | None = None,
    ) -> str | None:
        # Publish a StreamCompleteEvent to the EventBus
        await session_pool.event_bus.publish(
            session_id,
            StreamCompleteEvent(message=final_msg, session_id=session_id),
        )
        return "msg-1"

    # Mock wait_for_completion to return immediately
    async def mock_wait_for_completion(
        session_id: str,
        timeout: float | None = None,
    ) -> str:
        return session_id

    # Mock close_session to avoid actual cleanup
    async def mock_close_session(session_id: str) -> None:
        pass

    with (
        patch.object(session_pool, "create_session", side_effect=mock_create_session),
        patch.object(session_pool, "send_message", side_effect=mock_send_message),
        patch.object(session_pool, "wait_for_completion", side_effect=mock_wait_for_completion),
        patch.object(session_pool, "close_session", side_effect=mock_close_session),
        patch("uuid.uuid4", return_value=MagicMock(__str__=lambda _: "test-uuid")),
    ):
        result = await session_pool.run_agent("test-agent", "Say hello")

    assert result == "Hello from agent"


@_L2_SKIP
@pytest.mark.anyio
async def test_run_agent_with_parent(session_pool: SessionPool) -> None:
    """run_agent passes parent_session_id through to create_session."""
    final_msg = ChatMessage(content="result", role="assistant")

    captured_args: dict[str, Any] = {}

    async def mock_send_message(
        session_id: str,
        content: str | list[Any],
        *,
        mode: DeliveryMode = DeliveryMode.QUEUE,
        message_id: str | None = None,
    ) -> str | None:
        await session_pool.event_bus.publish(
            session_id,
            StreamCompleteEvent(message=final_msg, session_id=session_id),
        )
        return "msg-1"

    async def mock_create_session(
        session_id: str,
        agent_name: str | None = None,
        parent_session_id: str | None = None,
        lifecycle_policy: str | None = None,
        **metadata: Any,
    ) -> Any:
        captured_args["session_id"] = session_id
        captured_args["agent_name"] = agent_name
        captured_args["parent_session_id"] = parent_session_id
        captured_args["metadata"] = metadata
        # Create a real session so the EventBus subscription works
        await session_pool.sessions.get_or_create_session(session_id, agent_name=agent_name)
        return session_pool.sessions.get_session(session_id)

    async def mock_wait_for_completion(
        session_id: str,
        timeout: float | None = None,
    ) -> str:
        return session_id

    async def mock_close_session(session_id: str) -> None:
        pass

    with (
        patch.object(session_pool, "create_session", side_effect=mock_create_session),
        patch.object(session_pool, "send_message", side_effect=mock_send_message),
        patch.object(session_pool, "wait_for_completion", side_effect=mock_wait_for_completion),
        patch.object(session_pool, "close_session", side_effect=mock_close_session),
        patch("uuid.uuid4", return_value=MagicMock(__str__=lambda _: "child-uuid")),
    ):
        result = await session_pool.run_agent(
            "child-agent",
            "do work",
            parent_session_id="parent-sess",
            key="value",
        )

    assert result == "result"
    assert captured_args["agent_name"] == "child-agent"
    assert captured_args["parent_session_id"] == "parent-sess"
    assert captured_args["metadata"] == {"key": "value"}


@_L2_SKIP
@pytest.mark.anyio
async def test_run_agent_cleanup_on_error(session_pool: SessionPool) -> None:
    """run_agent ensures session is closed even when an error occurs during the run."""
    close_called = False

    async def mock_send_message(
        session_id: str,
        content: str | list[Any],
        *,
        mode: DeliveryMode = DeliveryMode.QUEUE,
        message_id: str | None = None,
    ) -> str | None:
        # Publish a RunErrorEvent to trigger an exception
        await session_pool.event_bus.publish(
            session_id,
            RunErrorEvent(message="Agent crashed", run_id="r1", agent_name="test"),
        )
        return "msg-1"

    async def mock_create_session(
        session_id: str,
        agent_name: str | None = None,
        parent_session_id: str | None = None,
        lifecycle_policy: str | None = None,
        **metadata: Any,
    ) -> Any:
        await session_pool.sessions.get_or_create_session(session_id, agent_name=agent_name)
        return session_pool.sessions.get_session(session_id)

    async def mock_wait_for_completion(
        session_id: str,
        timeout: float | None = None,
    ) -> str:
        return session_id

    async def mock_close_session(session_id: str) -> None:
        nonlocal close_called
        close_called = True

    with (
        patch.object(session_pool, "create_session", side_effect=mock_create_session),
        patch.object(session_pool, "send_message", side_effect=mock_send_message),
        patch.object(session_pool, "wait_for_completion", side_effect=mock_wait_for_completion),
        patch.object(session_pool, "close_session", side_effect=mock_close_session),
        patch("uuid.uuid4", return_value=MagicMock(__str__=lambda _: "err-uuid")),
        pytest.raises(RuntimeError, match="Agent crashed"),
    ):
        await session_pool.run_agent("test-agent", "fail please")

    assert close_called, "close_session must be called even on error"


@_L2_SKIP
@pytest.mark.anyio
async def test_run_agent_cleanup_on_send_failure(session_pool: SessionPool) -> None:
    """run_agent raises and cleans up when send_message returns None."""
    close_called = False

    async def mock_send_message(
        session_id: str,
        content: str | list[Any],
        *,
        mode: DeliveryMode = DeliveryMode.QUEUE,
        message_id: str | None = None,
    ) -> str | None:
        return None

    async def mock_create_session(
        session_id: str,
        agent_name: str | None = None,
        parent_session_id: str | None = None,
        lifecycle_policy: str | None = None,
        **metadata: Any,
    ) -> Any:
        await session_pool.sessions.get_or_create_session(session_id, agent_name=agent_name)
        return session_pool.sessions.get_session(session_id)

    async def mock_close_session(session_id: str) -> None:
        nonlocal close_called
        close_called = True

    with (
        patch.object(session_pool, "create_session", side_effect=mock_create_session),
        patch.object(session_pool, "send_message", side_effect=mock_send_message),
        patch.object(session_pool, "close_session", side_effect=mock_close_session),
        patch("uuid.uuid4", return_value=MagicMock(__str__=lambda _: "fail-uuid")),
        pytest.raises(RuntimeError, match="Failed to send message"),
    ):
        await session_pool.run_agent("test-agent", "hello")

    assert close_called, "close_session must be called even on send failure"


# ---------------------------------------------------------------------------
# revoke_message (existing wrapper — D21)
# ---------------------------------------------------------------------------


@_L2_SKIP
@pytest.mark.anyio
async def test_revoke_message_wrapper(session_pool: SessionPool) -> None:
    """revoke_message delegates to SessionController.revoke_inject."""
    session_pool.sessions.revoke_inject = MagicMock(return_value=True)  # type: ignore[method-assign]

    result = session_pool.revoke_message("sess-1", "msg-123")

    assert result is True
    session_pool.sessions.revoke_inject.assert_called_once_with("sess-1", "msg-123")


@_L2_SKIP
@pytest.mark.anyio
async def test_revoke_message_returns_false_for_unknown(session_pool: SessionPool) -> None:
    """revoke_message returns False when revoke_inject returns False."""
    session_pool.sessions.revoke_inject = MagicMock(return_value=False)  # type: ignore[method-assign]

    result = session_pool.revoke_message("sess-1", "unknown-msg")

    assert result is False


# ---------------------------------------------------------------------------
# wait_for_completion (existing wrapper — D25)
# ---------------------------------------------------------------------------


@_L2_SKIP
@pytest.mark.anyio
async def test_wait_for_completion_wrapper(session_pool: SessionPool) -> None:
    """wait_for_completion delegates to SessionController.wait_for_completion."""
    session_pool.sessions.wait_for_completion = AsyncMock(return_value="sess-1")  # type: ignore[method-assign]

    result = await session_pool.wait_for_completion("sess-1")

    assert result == "sess-1"
    session_pool.sessions.wait_for_completion.assert_awaited_once_with("sess-1", timeout=300)


@_L2_SKIP
@pytest.mark.anyio
async def test_wait_for_completion_with_timeout(session_pool: SessionPool) -> None:
    """wait_for_completion passes timeout through to SessionController."""
    session_pool.sessions.wait_for_completion = AsyncMock(return_value="sess-1")  # type: ignore[method-assign]

    result = await session_pool.wait_for_completion("sess-1", timeout=30.0)

    assert result == "sess-1"
    session_pool.sessions.wait_for_completion.assert_awaited_once_with("sess-1", timeout=30.0)


@_L2_SKIP
@pytest.mark.anyio
async def test_wait_for_completion_raises_session_not_found(session_pool: SessionPool) -> None:
    """wait_for_completion propagates SessionNotFoundError from controller."""
    session_pool.sessions.wait_for_completion = AsyncMock(  # type: ignore[method-assign]
        side_effect=SessionNotFoundError("missing-sess"),
    )

    with pytest.raises(SessionNotFoundError, match="missing-sess"):
        await session_pool.wait_for_completion("missing-sess")


# ---------------------------------------------------------------------------
# _route_message (SessionController internal method)
# ---------------------------------------------------------------------------


@_L2_SKIP
@pytest.mark.anyio
async def test_route_message_idle_session_starts_run(
    session_pool: SessionPool,
) -> None:
    """_route_message starts a new run handle when session is idle."""
    await session_pool.sessions.get_or_create_session("sess-1", agent_name="agent-a")
    session = session_pool.sessions.get_session("sess-1")
    assert session is not None

    # Mock the agent
    mock_agent = MagicMock()
    mock_agent.AGENT_TYPE = "native"

    # Mock _start_run_handle to return a message_id
    session_pool.sessions._start_run_handle = MagicMock(return_value="msg-started")  # type: ignore[method-assign]

    result = await session_pool.sessions._route_message(
        session,
        mock_agent,
        "sess-1",
        "hello",
    )

    assert result == "msg-started"
    session_pool.sessions._start_run_handle.assert_called_once()


@pytest.mark.anyio
async def test_route_message_closing_session_returns_none(
    session_pool: SessionPool,
) -> None:
    """_route_message returns None when session is closing."""
    await session_pool.sessions.get_or_create_session("sess-1", agent_name="test_agent")
    session = session_pool.sessions.get_session("sess-1")
    assert session is not None
    session.is_closing = True

    # Use a real agent from the pool — _route_message checks closing before using the agent
    agent = await session_pool.sessions.get_or_create_session_agent(
        "sess-temp-for-agent", agent_name="test_agent"
    )

    result = await session_pool.sessions._route_message(
        session,
        agent,
        "sess-1",
        "hello",
    )

    assert result is None
