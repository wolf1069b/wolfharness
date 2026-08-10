"""Red flag test: title generation must NOT block agent message processing.

Bug: On the first message in a session, ``_maybe_generate_title()`` is
``await``-ed inside ``run_message_phases()`` *before* the agent even
starts streaming.  Because title generation does a full LLM round-trip
(``Agent.run()``) with **no timeout**, a slow or unresponsive title model
blocks the entire message response.

The fix makes title generation fire-and-forget (background task) in
``_run_message_locked`` and adds a 15-second timeout in
``_generate_title_core`` as a safety net.

These tests should FAIL before the fix and PASS afterwards.
"""

from __future__ import annotations

import asyncio
import os
import time
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pytest

from tests.servers.opencode_server.conftest import run_message_phases
from wolfharness.storage.manager import (
    SessionMetadata,
    SessionMetadataGeneratedEvent,
    StorageManager,
)
from wolfharness.utils.time_utils import now_ms
from wolfharness_server.opencode_server.models import (
    MessageWithParts,
    Session,
    TextPartInput,
    TimeCreatedUpdated,
    UserMessage,
)
from wolfharness_server.opencode_server.models.message import MessageRequest, TimeCreated
from wolfharness_server.opencode_server.routes.message_routes import (
    _maybe_generate_title,
)
from wolfharness_server.opencode_server.session_pool_integration import get_messages_for_session
from wolfharness_server.opencode_server.state import ServerState


pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_state(tmp_path: Any) -> ServerState:
    """Build a minimal ServerState with mocked heavy components."""
    from wolfharness_config.storage import MemoryStorageConfig, StorageConfig

    agent = Mock()
    agent.name = "test-agent"
    agent.tools = []

    storage_mgr = StorageManager(config=StorageConfig(providers=[MemoryStorageConfig()]))

    pool = Mock()
    pool.storage = storage_mgr
    pool.manifest = Mock(model_variants={})
    pool.manifest.agents = {}

    agent.agent_pool = pool
    agent.host_context = pool
    agent._agent_pool = pool  # state.py resolves _pool via agent._agent_pool
    agent.storage = storage_mgr

    env = Mock()
    from upathtools.filesystems import AsyncLocalFileSystem

    env.get_fs = Mock(return_value=AsyncLocalFileSystem())
    env.cwd = str(tmp_path)
    agent.env = env

    # Set up session pool mocks for run_message_phases
    pool.session_pool = Mock()
    pool.session_pool.sessions = Mock()
    pool.session_pool.sessions.get_or_create_session = AsyncMock(return_value=(Mock(), True))
    pool.session_pool.sessions.get_or_create_session_agent = AsyncMock(return_value=Mock())
    pool.session_pool.sessions.get_session = Mock(return_value=None)
    _run_handle = Mock()
    _run_handle.complete_event = asyncio.Event()
    pool.session_pool.send_message = AsyncMock(return_value=_run_handle)
    pool.session_pool.event_bus = Mock()
    from tests._helpers.mock_stream import EmptyReceiveStream

    pool.session_pool.event_bus.subscribe = AsyncMock(return_value=EmptyReceiveStream())
    pool.session_pool.event_bus.unsubscribe = AsyncMock()

    state = ServerState(working_dir=str(tmp_path), agent=agent)
    # Initialize backward-compat dicts removed from ServerState dataclass
    state.messages = {}
    # No session_pool_integration — run_message_phases will use the
    # fallback path via session_pool.sessions.get_or_create_session.
    return state


def _seed_session(state: ServerState, session_id: str) -> None:
    """Insert a bare session + one user message into state."""
    now = now_ms()
    state.sessions[session_id] = Session(
        id=session_id,
        project_id="test",
        directory=state.working_dir,
        title="New Session",
        version="1",
        time=TimeCreatedUpdated(created=now, updated=now),
    )
    state.messages[session_id] = []

    user_msg = UserMessage(
        id="msg_user_001",
        session_id=session_id,
        time=TimeCreated.now(),
        agent="default",
    )
    state.messages[session_id].append(MessageWithParts(info=user_msg))


class _AsyncIteratorMock:
    """Minimal async iterator that yields items then stops."""

    def __init__(self, items: list[Any]) -> None:
        self._items = list(items)

    def __aiter__(self) -> _AsyncIteratorMock:
        return self

    async def __anext__(self) -> Any:
        if not self._items:
            raise StopAsyncIteration
        return self._items.pop(0)


def _make_mock_usage() -> Mock:
    """Create a mock usage object compatible with Tokens.from_pydantic_ai."""
    mock_usage = Mock()
    mock_usage.details = {}
    mock_usage.input_tokens = 0
    mock_usage.output_tokens = 0
    mock_usage.total_tokens = 0
    mock_usage.cache_read_tokens = 0
    mock_usage.cache_write_tokens = 0
    return mock_usage


# ---------------------------------------------------------------------------
# Red-flag tests
# ---------------------------------------------------------------------------


class TestTitleGenerationDoesNotBlockAgent:
    """Title generation must not delay agent response start.

    The fix changes ``await _maybe_generate_title()`` to a fire-and-forget
    background task in ``_run_message_locked``.  We verify this by
    measuring wall-clock time.
    """

    async def test_run_message_locked_returns_fast_despite_slow_title(
        self,
        tmp_path: Any,
    ) -> None:
        """_run_message_locked must return fast even with slow title gen.

        We mock _maybe_generate_title to take 1.5s via asyncio.sleep.
        If _run_message_locked still ``await``s it, total >= 1.5s.
        After the fix (fire-and-forget background task), total < 0.5s.
        """
        state = _make_state(tmp_path)
        session_id = "ses_blocked"
        _seed_session(state, session_id)

        state.agent.run_stream = AsyncMock(return_value=_AsyncIteratorMock([]))

        async def slow_maybe_generate_title(
            state_: ServerState,
            session_id_: str,
            user_prompt: Any,
        ) -> None:
            await asyncio.sleep(1.5)

        request = MessageRequest(parts=[TextPartInput(text="hello")])
        user_msg_id = "msg_user_001"
        user_msg = state.messages[session_id][0]

        start = time.monotonic()

        with (
            patch(
                "wolfharness_server.opencode_server.routes.message_routes._maybe_generate_title",
                slow_maybe_generate_title,
            ),
            patch(
                "wolfharness_server.opencode_server.routes.message_routes.extract_user_prompt_from_parts",
                new=AsyncMock(return_value=["hello"]),
            ),
            patch(
                "wolfharness_server.opencode_server.routes.message_routes.OpenCodeStreamAdapter"
            ) as mock_adapter_class,
        ):
            mock_adapter_instance = mock_adapter_class.return_value
            mock_adapter_instance.process_stream = Mock(
                return_value=_AsyncIteratorMock([]),
            )
            mock_adapter_instance.finalize = Mock(return_value=[])
            mock_adapter_instance.response_text = ""
            mock_adapter_instance.usage = _make_mock_usage()
            mock_adapter_instance.cost_info = None

            messages = await get_messages_for_session(state, session_id)
            user_msg = messages[0]
            await run_message_phases(session_id, request, state, user_msg_id, user_msg)

        elapsed = time.monotonic() - start

        # Clean up lingering background tasks.
        await state.cleanup_tasks()

        # RED FLAG: Before the fix, the 1.5s title delay is included.
        assert elapsed < 1.5, (
            f"_run_message_locked took {elapsed:.2f}s — agent processing "
            f"is blocked by slow title generation"
        )

    async def test_maybe_generate_title_e2e_returns_fast(self, tmp_path: Any) -> None:
        """End-to-end: _maybe_generate_title called from _run_message_locked.

        should not block, even with a slow title model.

        This test uses a real slow _generate_title_core mock to test the
        full path through log_session -> _generate_title_from_prompt.
        """
        state = _make_state(tmp_path)
        session_id = "ses_e2e"
        _seed_session(state, session_id)

        state.agent.run_stream = AsyncMock(return_value=_AsyncIteratorMock([]))

        async def slow_core(
            self_: StorageManager,
            sid: str,
            prompt: str,
        ) -> SessionMetadata:
            await asyncio.sleep(2.0)
            return SessionMetadata(title="Slow Title", emoji="🐢", icon="mdi:turtle")

        request = MessageRequest(parts=[TextPartInput(text="hello")])
        user_msg_id = "msg_user_001"

        start = time.monotonic()

        with (
            patch.dict(os.environ, {}, clear=False),
            patch(
                "wolfharness_server.opencode_server.routes.message_routes.extract_user_prompt_from_parts",
                new=AsyncMock(return_value=["hello"]),
            ),
            patch(
                "wolfharness_server.opencode_server.routes.message_routes.OpenCodeStreamAdapter"
            ) as mock_adapter_class,
        ):
            os.environ.pop("PYTEST_CURRENT_TEST", None)

            with patch.object(StorageManager, "_generate_title_core", slow_core):
                mock_adapter_instance = mock_adapter_class.return_value
                mock_adapter_instance.process_stream = Mock(
                    return_value=_AsyncIteratorMock([]),
                )
                mock_adapter_instance.finalize = Mock(return_value=[])
                mock_adapter_instance.response_text = ""
                mock_adapter_instance.usage = _make_mock_usage()
                mock_adapter_instance.cost_info = None

                messages = await get_messages_for_session(state, session_id)
                user_msg = messages[0]
                await run_message_phases(session_id, request, state, user_msg_id, user_msg)

        elapsed = time.monotonic() - start

        # Clean up lingering background tasks.
        await state.cleanup_tasks()

        # RED FLAG: Before the fix, _run_message_locked awaits
        # _maybe_generate_title which awaits _generate_title_core (2s).
        assert elapsed < 1.5, (
            f"_run_message_locked took {elapsed:.2f}s with slow title "
            f"model — title generation should be fire-and-forget"
        )


class TestTitleStillGeneratedAsynchronously:
    """Title must still be generated even when it doesn't block."""

    async def test_maybe_generate_title_respects_session_metadata_disable(
        self,
        tmp_path: Any,
    ) -> None:
        """SessionPool metadata can disable title generation for child sessions."""
        state = _make_state(tmp_path)
        session_id = "ses_title_disabled"
        _seed_session(state, session_id)

        session_state = Mock(metadata={"generate_title": False})
        state.pool.session_pool.sessions.get_session = Mock(return_value=session_state)

        log_session = AsyncMock()
        with patch.object(state.pool.storage, "log_session", log_session):
            await _maybe_generate_title(state, session_id, ["hello"])

        log_session.assert_not_awaited()

    async def test_title_eventually_appears(self, tmp_path: Any) -> None:
        """Title should appear in session after background generation completes."""
        state = _make_state(tmp_path)
        session_id = "ses_title_async"
        _seed_session(state, session_id)

        mock_metadata = SessionMetadata(title="Async Title", emoji="⚡", icon="mdi:lightning")

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("PYTEST_CURRENT_TEST", None)

            with patch.object(
                StorageManager,
                "_generate_title_core",
                return_value=mock_metadata,
            ):
                await _maybe_generate_title(state, session_id, ["hello"])
                # Give the background task time to complete
                await asyncio.sleep(0.5)

        stored_title = await state.pool.storage.get_session_title(session_id)
        assert stored_title == "Async Title", (
            f"Title was '{stored_title}' — expected 'Async Title'. "
            f"Background title generation may not be working."
        )

    async def test_metadata_generated_signal_still_fires(self, tmp_path: Any) -> None:
        """The metadata_generated signal must still fire after generation."""
        state = _make_state(tmp_path)
        session_id = "ses_title_signal"
        _seed_session(state, session_id)

        mock_metadata = SessionMetadata(
            title="Signal Title", emoji="\ud83d\udce1", icon="mdi:antenna"
        )
        signal_titles: list[str] = []

        def on_signal(event):
            signal_titles.append(event.metadata.title)

        state.pool.storage.metadata_generated.connect(on_signal)

        async def mock_core_with_signal(self_, sid, prompt):
            event = SessionMetadataGeneratedEvent(session_id=sid, metadata=mock_metadata)
            await self_.metadata_generated.emit(event)
            return mock_metadata

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("PYTEST_CURRENT_TEST", None)

            with patch.object(
                StorageManager,
                "_generate_title_core",
                mock_core_with_signal,
            ):
                await state.pool.storage.log_session(
                    session_id=session_id,
                    node_name="test-agent",
                    initial_prompt="hello",
                )
                await asyncio.sleep(0.5)

        assert "Signal Title" in signal_titles, (
            f"metadata_generated signal was not fired. Collected: {signal_titles}"
        )


class TestTitleGenerationTimeout:
    """Title generation must have a timeout safety net."""

    async def test_generate_title_core_timeout_prevents_hang(self, tmp_path: Any) -> None:
        """_generate_title_core should not hang forever on a stuck model.

        The 15-second asyncio.wait_for timeout must cancel a stuck LLM call,
        preventing zombie tasks even with fire-and-forget.
        """
        state = _make_state(tmp_path)
        session_id = "ses_title_timeout"
        _seed_session(state, session_id)

        call_count = 0

        async def stuck_title(
            self_: StorageManager,
            session_id_: str,
            prompt_text: str,
        ) -> SessionMetadata | None:
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(300)
            return SessionMetadata(title="Never", emoji="💀", icon="mdi:skull")

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("PYTEST_CURRENT_TEST", None)

            with patch.object(StorageManager, "_generate_title_core", stuck_title):
                await _maybe_generate_title(state, session_id, ["hello"])

                # Poll until background_tasks is empty or 25s elapses.
                start = time.monotonic()
                while state.background_tasks and (time.monotonic() - start) < 25:
                    await asyncio.sleep(0.5)

                elapsed = time.monotonic() - start

        assert elapsed < 25, f"Title generation took {elapsed:.1f}s — timeout not working"
        assert call_count >= 1, "Title generation was never attempted"
