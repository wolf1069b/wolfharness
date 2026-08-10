"""Integration tests for session resume data integrity.

Validates that session resume preserves conversation history, timestamps,
status fields, MCP server connections, stored data, and avoids duplicate
load_session calls.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from acp.schema import LoadSessionRequest, ResumeSessionRequest
from wolfharness.sessions.models import SessionData
from wolfharness_server.acp_server.acp_agent import AgentPoolACPAgent
from wolfharness_server.acp_server.session_manager import ACPSessionManager


if TYPE_CHECKING:
    from wolfharness import AgentPool


# ---------------------------------------------------------------------------
# Local fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_agent() -> MagicMock:
    """Create a mock agent with load_session."""
    agent = MagicMock()
    agent.name = "test_agent"
    agent.model_name = "test-model"
    agent.load_session = AsyncMock(return_value=True)
    agent.load_rules = AsyncMock()
    agent.conversation = MagicMock()
    agent.conversation.chat_messages = []
    return agent


@pytest.fixture
def mock_session_store() -> MagicMock:
    """Create a mock SessionStore for use with ACPSessionManager."""
    store = MagicMock()
    store.load_session = AsyncMock()
    store.save_session = AsyncMock()
    store.list_session_ids = AsyncMock(return_value=[])
    return store


@pytest.fixture
def session_manager(
    minimal_pool: AgentPool,
    mock_agent: MagicMock,
    mock_session_store: MagicMock,
) -> ACPSessionManager:
    """Create an ACPSessionManager with real pool and controlled session_pool methods.

    Uses minimal_pool (real AgentPool) with monkeypatched session_pool methods
    to provide controlled behavior for resume_session() testing.
    """
    assert minimal_pool.session_pool is not None
    # Monkeypatch session_pool methods that resume_session() calls
    minimal_pool.session_pool.sessions.get_or_create_session_agent = AsyncMock(  # type: ignore[assignment]
        return_value=mock_agent,
    )
    minimal_pool.session_pool._get_resume_lock = AsyncMock(return_value=asyncio.Lock())  # type: ignore[assignment]
    # Set the session store to our mock
    minimal_pool.session_pool.sessions.store = mock_session_store  # type: ignore[assignment]

    manager = ACPSessionManager(pool=minimal_pool)
    manager._acp_sessions = {}
    return manager


@pytest.fixture
def known_session_data() -> SessionData:
    """Create a SessionData with known values for integrity checks."""
    created = datetime(2025, 1, 15, 10, 30, 0, tzinfo=UTC)
    return SessionData(
        session_id="sess-abc-123",
        agent_name="test_agent",
        cwd="/tmp/test",
        project_id="proj-1",
        status="active",
        created_at=created,
        metadata={"protocol": "acp", "mcp_server_count": 1},
    )


@pytest.fixture
def checkpointed_session_data() -> SessionData:
    """Create a SessionData with status='checkpointed'."""
    return SessionData(
        session_id="sess-chk-456",
        agent_name="test_agent",
        cwd="/tmp/checkpointed",
        status="checkpointed",
        created_at=datetime(2025, 3, 1, 0, 0, 0, tzinfo=UTC),
    )


# ---------------------------------------------------------------------------
# Test 1: Conversation history preservation
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_create_then_resume_preserves_conversation_history(
    session_manager: ACPSessionManager,
    mock_agent: MagicMock,
    known_session_data: SessionData,
) -> None:
    """Resume should create and return an ACPSession without polluting the shared agent.

    session_manager.resume_session() no longer calls agent.load_session() —
    conversation history loading is now handled by SessionPool's
    get_or_create_session_agent() when creating per-session agents.
    """
    # Arrange: store session data
    session_manager.session_store.load_session = AsyncMock(return_value=known_session_data)  # type: ignore[union-attr]

    with patch("wolfharness_server.acp_server.session_manager.ACPSession") as mock_session_cls:
        mock_session_instance = MagicMock()
        mock_session_instance.session_id = "sess-abc-123"
        mock_session_instance.initialize = AsyncMock()
        mock_session_instance.initialize_mcp_servers = AsyncMock()
        mock_session_instance.agent = mock_agent
        mock_session_instance.register_update_callback = MagicMock()
        mock_session_cls.return_value = mock_session_instance

        # Act
        result = await session_manager.resume_session(
            session_id="sess-abc-123",
            client=AsyncMock(),
            acp_agent=MagicMock(),
        )

    # Assert
    assert result is not None
    assert result is mock_session_instance
    # load_session is no longer called in resume_session — it's handled by SessionPool
    mock_agent.load_session.assert_not_awaited()


# ---------------------------------------------------------------------------
# Test 2: created_at timestamp preservation
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_resume_preserves_created_at_timestamp(
    session_manager: ACPSessionManager,
    mock_agent: MagicMock,
    known_session_data: SessionData,
) -> None:
    """Resume must NOT overwrite created_at via session_store.save."""
    # Arrange
    session_manager.session_store.load_session = AsyncMock(return_value=known_session_data)  # type: ignore[union-attr]

    with patch("wolfharness_server.acp_server.session_manager.ACPSession") as mock_session_cls:
        mock_session_instance = MagicMock()
        mock_session_instance.initialize = AsyncMock()
        mock_session_instance.initialize_mcp_servers = AsyncMock()
        mock_session_instance.agent = mock_agent
        mock_session_instance.register_update_callback = MagicMock()
        mock_session_cls.return_value = mock_session_instance

        # Act
        await session_manager.resume_session(
            session_id="sess-abc-123",
            client=AsyncMock(),
            acp_agent=MagicMock(),
        )

    # Assert: session_store.save should NOT be called during resume
    # (created_at must remain unchanged from when the session was first created)
    session_manager.session_store.save_session.assert_not_awaited()  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# Test 3: status field preservation
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_resume_preserves_status_field(
    session_manager: ACPSessionManager,
    mock_agent: MagicMock,
    checkpointed_session_data: SessionData,
) -> None:
    """Resume must NOT overwrite status='checkpointed' via session_store.save."""
    # Arrange
    session_manager.session_store.load_session = AsyncMock(return_value=checkpointed_session_data)  # type: ignore[union-attr]

    with patch("wolfharness_server.acp_server.session_manager.ACPSession") as mock_session_cls:
        mock_session_instance = MagicMock()
        mock_session_instance.initialize = AsyncMock()
        mock_session_instance.initialize_mcp_servers = AsyncMock()
        mock_session_instance.agent = mock_agent
        mock_session_instance.register_update_callback = MagicMock()
        mock_session_cls.return_value = mock_session_instance

        # Act
        await session_manager.resume_session(
            session_id="sess-chk-456",
            client=AsyncMock(),
            acp_agent=MagicMock(),
        )

    # Assert: session_store.save should NOT be called (status preserved)
    session_manager.session_store.save_session.assert_not_awaited()  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# Test 4: MCP server initialization on resume
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_resume_with_mcp_servers_initializes_connections(
    session_manager: ACPSessionManager,
    mock_agent: MagicMock,
    known_session_data: SessionData,
) -> None:
    """Resume with mcp_servers must initialize MCP connections."""
    # Arrange
    session_manager.session_store.load_session = AsyncMock(return_value=known_session_data)  # type: ignore[union-attr]
    mock_mcp_server = MagicMock()

    with patch("wolfharness_server.acp_server.session_manager.ACPSession") as mock_session_cls:
        mock_session_instance = MagicMock()
        mock_session_instance.initialize = AsyncMock()
        mock_session_instance.initialize_mcp_servers = AsyncMock()
        mock_session_instance.agent = mock_agent
        mock_session_instance.register_update_callback = MagicMock()
        mock_session_cls.return_value = mock_session_instance

        # Act
        result = await session_manager.resume_session(
            session_id="sess-abc-123",
            client=AsyncMock(),
            acp_agent=MagicMock(),
            mcp_servers=[mock_mcp_server],
        )

    # Assert
    assert result is not None
    mock_session_instance.initialize_mcp_servers.assert_awaited_once()
    # Session must be added to _acp_sessions
    assert "sess-abc-123" in session_manager._acp_sessions


# ---------------------------------------------------------------------------
# Test 5: load_session preserves stored data
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_load_session_path_preserves_stored_data(
    known_session_data: SessionData,
    mock_agent: MagicMock,
) -> None:
    """acp_agent.load_session() must not call session_store.save."""
    # Arrange
    mock_connection = AsyncMock()
    acp_agent = AgentPoolACPAgent(client=mock_connection, default_agent=mock_agent)
    acp_agent._initialized = True

    # Mock session_manager.resume_session to return a valid session
    mock_session = MagicMock()
    mock_session.session_id = "sess-abc-123"
    mock_session.cwd = "/tmp/test"
    mock_session.agent = mock_agent
    mock_session.agent.conversation = MagicMock()
    mock_session.agent.conversation.chat_messages = []
    mock_session.notifications = MagicMock()
    mock_session.notifications.replay = AsyncMock()
    mock_session.send_available_commands_update = AsyncMock()
    mock_session.agent.load_rules = AsyncMock()
    acp_agent.session_manager.resume_session = AsyncMock(return_value=mock_session)
    acp_agent.session_manager.get_session = MagicMock(return_value=None)

    # Mock session_store property (read-only) to verify it's not called
    session_store = MagicMock()
    session_store.save_session = AsyncMock()
    session_store.load_session = AsyncMock(return_value=known_session_data)

    request = LoadSessionRequest(session_id="sess-abc-123", cwd="/tmp/test")

    with patch.object(
        type(acp_agent.session_manager),
        "session_store",
        new_callable=lambda: property(lambda self: session_store),
    ):
        # Act
        response = await acp_agent.load_session(request)

    # Assert: session_store.save must NOT be called (data preserved)
    session_store.save_session.assert_not_awaited()
    # Response should be a valid LoadSessionResponse
    from acp.schema import LoadSessionResponse

    assert isinstance(response, LoadSessionResponse)


# ---------------------------------------------------------------------------
# Test 6: No duplicate load_session calls
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_no_duplicate_load_session_call_on_resume(
    session_manager: ACPSessionManager,
    mock_agent: MagicMock,
    known_session_data: SessionData,
) -> None:
    """session_manager.resume_session must NOT call agent.load_session.

    Conversation history loading is now handled by SessionPool's
    get_or_create_session_agent(), not by resume_session directly.
    """
    # Arrange
    session_manager.session_store.load_session = AsyncMock(return_value=known_session_data)  # type: ignore[union-attr]

    with patch("wolfharness_server.acp_server.session_manager.ACPSession") as mock_session_cls:
        mock_session_instance = MagicMock()
        mock_session_instance.initialize = AsyncMock()
        mock_session_instance.initialize_mcp_servers = AsyncMock()
        mock_session_instance.agent = mock_agent
        mock_session_instance.register_update_callback = MagicMock()
        mock_session_cls.return_value = mock_session_instance

        # Act
        await session_manager.resume_session(
            session_id="sess-abc-123",
            client=AsyncMock(),
            acp_agent=MagicMock(),
        )

    # Assert: agent.load_session is NOT called by resume_session
    mock_agent.load_session.assert_not_awaited()


@pytest.mark.unit
async def test_no_duplicate_load_session_call_via_acp_agent(
    mock_agent: MagicMock,
) -> None:
    """acp_agent.resume_session must not call agent.load_session directly.

    session_manager.resume_session already handles load_session internally.
    acp_agent must not double-call it.
    """
    # Arrange
    mock_connection = AsyncMock()
    acp_agent = AgentPoolACPAgent(client=mock_connection, default_agent=mock_agent)
    acp_agent._initialized = True

    # Mock session_manager.resume_session to return a session
    # (session_manager handles load_session internally)
    mock_session = MagicMock()
    mock_session.session_id = "sess-abc-123"
    mock_session.cwd = "/tmp/test"
    mock_session.agent = mock_agent
    mock_session.send_available_commands_update = AsyncMock()
    mock_session.agent.load_rules = AsyncMock()
    acp_agent.session_manager.resume_session = AsyncMock(return_value=mock_session)
    acp_agent.session_manager.get_session = MagicMock(return_value=None)

    request = ResumeSessionRequest(session_id="sess-abc-123", cwd="/tmp/test")

    # Reset call count after setup
    mock_agent.load_session.reset_mock()

    # Act
    await acp_agent.resume_session(request)

    # Assert: acp_agent must NOT call agent.load_session directly
    # (session_manager.resume_session already did it)
    mock_agent.load_session.assert_not_awaited()
