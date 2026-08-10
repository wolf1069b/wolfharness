"""Manual QA test server for OpenCode server.

Starts a minimal OpenCode server with mock agent/pool for manual testing.
Uses uvicorn on port 19001.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, Mock

import uvicorn

from wolfharness.storage import StorageManager
from wolfharness.utils.streams import FileOpsTracker
from wolfharness.utils.todos import TodoTracker
from wolfharness_server.opencode_server.server import create_app


if TYPE_CHECKING:
    from fastapi import FastAPI

    from wolfharness.sessions.models import SessionData


def create_test_app() -> FastAPI:  # noqa: PLR0915
    """Create a FastAPI app with mock dependencies for QA testing."""
    # Create mock pool
    pool = Mock()
    pool.manifest = Mock()
    pool.manifest.agents = {}
    pool.manifest.opencode = Mock()

    # Storage
    from wolfharness_config.storage import MemoryStorageConfig, StorageConfig

    storage_manager = StorageManager(config=StorageConfig(providers=[MemoryStorageConfig()]))
    pool.storage = storage_manager
    pool.file_ops = FileOpsTracker()
    pool.todos = TodoTracker()
    pool.all_agents = {}

    # Sessions store
    pool.sessions = Mock()
    pool.sessions.store = Mock()
    pool.sessions.store.save = storage_manager.save_session
    pool.sessions.store.delete = storage_manager.delete_session
    pool.sessions.store.load = storage_manager.load_session
    pool.sessions.store.list_sessions = AsyncMock(return_value=[])

    # Session pool
    pool.session_pool = Mock()

    async def _mock_create_session(
        session_id: str,
        agent_name: str | None = None,
        parent_session_id: str | None = None,
        **metadata: object,
    ) -> Mock:
        from datetime import datetime

        from wolfharness.sessions.models import SessionData

        data = SessionData(
            session_id=session_id,
            agent_name=agent_name or "test-agent",
            parent_id=parent_session_id,
            created_at=datetime.now(),
            last_active=datetime.now(),
            metadata=metadata,
        )
        await storage_manager.save_session(data)
        return Mock()

    async def _mock_close_session(session_id: str) -> None:
        await storage_manager.delete_session(session_id)

    pool.session_pool.create_session = AsyncMock(side_effect=_mock_create_session)
    pool.session_pool.close_session = AsyncMock(side_effect=_mock_close_session)
    pool.session_pool.sessions = Mock()
    pool.session_pool.sessions.cancel_run_for_session = Mock()
    pool.session_pool.sessions.store = Mock()
    pool.session_pool.sessions.store.save = storage_manager.save_session
    pool.session_pool.sessions.store.delete = AsyncMock(side_effect=storage_manager.delete_session)
    pool.session_pool.sessions.store.load = AsyncMock(side_effect=storage_manager.load_session)
    pool.session_pool.sessions.store.list_sessions = AsyncMock(return_value=[])

    _mock_session_agent = Mock()
    _mock_session_agent.load_session = AsyncMock(return_value=None)
    _mock_session_agent.conversation = Mock()
    _mock_session_agent.conversation.chat_messages = []
    pool.session_pool.sessions.get_or_create_session_agent = AsyncMock(
        return_value=_mock_session_agent
    )

    def _mock_session_state():
        from datetime import datetime

        from wolfharness.orchestrator.core import SessionState

        state = Mock(spec=SessionState)
        state.created_at = datetime.now().timestamp()
        state.last_active_at = datetime.now().timestamp()
        state.session_id = "test-session"
        state.agent_name = "test-agent"
        state.parent_id = None
        state.parent_session_id = None
        state.metadata = {}
        state.current_run_id = None
        state.input_provider = None
        state.pending_questions = {}
        return state

    pool.session_pool.sessions.get_or_create_session = AsyncMock(
        return_value=(_mock_session_state(), True)
    )

    _run_handle = Mock()
    _run_handle.complete_event = Mock()
    _run_handle.complete_event.wait = AsyncMock()
    pool.session_pool.receive_request = AsyncMock(return_value=_run_handle)
    pool.session_pool.event_bus = Mock()
    pool.session_pool.event_bus.publish = AsyncMock()
    pool.session_pool.event_bus.subscribe = AsyncMock(return_value=asyncio.Queue())
    pool.session_pool.event_bus.unsubscribe = AsyncMock()
    pool.session_pool.shutdown = AsyncMock()

    # Mock env
    from upathtools.filesystems import AsyncLocalFileSystem

    env = Mock()
    env.get_fs = Mock(return_value=AsyncLocalFileSystem())
    env.cwd = "/tmp"
    env.execute_command = AsyncMock(
        return_value=Mock(success=True, result="command output", error=None)
    )

    # Mock agent
    agent = Mock()
    agent.name = "test-agent"
    agent.env = env
    agent._input_provider = None
    agent.agent_pool = pool
    agent.storage = storage_manager

    async def _list_sessions(**kwargs: object) -> list:

        ids = await storage_manager.list_session_ids()
        results: list[SessionData] = []
        for sid in ids:
            data = await storage_manager.load_session(sid)
            if data is not None:
                results.append(data)
        return results

    agent.list_sessions = _list_sessions
    agent.load_session = AsyncMock(return_value=None)

    # Use create_app which sets up the full server
    return create_app(agent=agent, working_dir="/tmp")


if __name__ == "__main__":
    app = create_test_app()
    uvicorn.run(app, host="127.0.0.1", port=19001, log_level="info")
