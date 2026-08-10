"""Tests for ACPSession.process_prompt() passing client_supports_turn_complete flag.

Verifies that process_prompt derives the client_supports_turn_complete flag from
self.client_capabilities.turn_complete and passes it to ACPEventConverter.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from acp.schema import TextContentBlock
from acp.schema.capabilities import ClientCapabilities
from wolfharness import Agent, AgentPool
from wolfharness_server.acp_server.event_converter import ACPEventConverter
from wolfharness_server.acp_server.session import ACPSession


pytestmark = pytest.mark.unit


if TYPE_CHECKING:
    from collections.abc import AsyncIterator


@pytest.fixture
def agent_pool() -> AgentPool:
    """Create a real agent pool with a test agent."""

    def simple_callback(message: str) -> str:
        return f"Response: {message}"

    from wolfharness.models.agents import NativeAgentConfig
    from wolfharness.models.manifest import AgentsManifest

    manifest = AgentsManifest(agents={"test_agent": NativeAgentConfig(model="test")})

    pool = AgentPool(manifest)
    Agent.from_callback(name="test_agent", callback=simple_callback, agent_pool=pool)
    # pool.register() removed; agent created from callback/config above
    return pool


@pytest.fixture
def mock_acp_agent() -> MagicMock:
    """Create a mock ACP agent with tasks support."""
    mock = MagicMock()
    mock._task_group.start_soon = lambda fn, *args: None  # type: ignore[assignment,method-assign]
    return mock


async def _run_stream_empty(*args: Any, **kwargs: Any) -> AsyncIterator[Any]:
    """Empty async generator for mocking agent.run_stream."""
    return
    yield  # Make this an async generator


class TestProcessPromptTurnCompleteFlag:
    """RED FLAG: process_prompt must pass client_supports_turn_complete to ACPEventConverter."""

    @pytest.mark.anyio
    async def test_process_prompt_passes_turn_complete_true(
        self,
        agent_pool: AgentPool,
        mock_acp_agent: MagicMock,
    ) -> None:
        """When client_capabilities.turn_complete=True, ACPEventConverter must be.

        created with client_supports_turn_complete=True.

        """
        agent = agent_pool.manifest.agents["test_agent"].get_agent(pool=agent_pool)
        mock_client = AsyncMock()

        session = ACPSession(
            session_id="test-session",
            agent=agent,
            cwd="/tmp",
            client=mock_client,
            acp_agent=mock_acp_agent,
            client_capabilities=ClientCapabilities(turn_complete=True),
        )

        # Mock run_stream to yield nothing
        agent.run_stream = _run_stream_empty  # type: ignore[method-assign]

        # Mock with_session_providers as no-op async context manager
        @asynccontextmanager
        async def _noop_ctx(*args: Any, **kwargs: Any) -> AsyncIterator[None]:
            yield

        agent._with_session_capabilities = _noop_ctx  # type: ignore[method-assign]

        captured_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        original_init = ACPEventConverter.__init__

        def _capture_init(self: ACPEventConverter, *args: Any, **kwargs: Any) -> None:
            captured_calls.append((args, kwargs))
            original_init(self, *args, **kwargs)

        with patch.object(ACPEventConverter, "__init__", _capture_init):
            await session.process_prompt([TextContentBlock(text="hello")])

        assert len(captured_calls) == 1
        _args, kwargs = captured_calls[0]
        assert kwargs.get("client_supports_turn_complete") is True

    @pytest.mark.anyio
    async def test_process_prompt_passes_turn_complete_false(
        self,
        agent_pool: AgentPool,
        mock_acp_agent: MagicMock,
    ) -> None:
        """When client_capabilities.turn_complete=False, ACPEventConverter must be.

        created with client_supports_turn_complete=False.

        """
        agent = agent_pool.manifest.agents["test_agent"].get_agent(pool=agent_pool)
        mock_client = AsyncMock()

        session = ACPSession(
            session_id="test-session",
            agent=agent,
            cwd="/tmp",
            client=mock_client,
            acp_agent=mock_acp_agent,
            client_capabilities=ClientCapabilities(turn_complete=False),
        )

        agent.run_stream = _run_stream_empty  # type: ignore[method-assign]

        @asynccontextmanager
        async def _noop_ctx(*args: Any, **kwargs: Any) -> AsyncIterator[None]:
            yield

        agent._with_session_capabilities = _noop_ctx  # type: ignore[method-assign]

        captured_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        original_init = ACPEventConverter.__init__

        def _capture_init(self: ACPEventConverter, *args: Any, **kwargs: Any) -> None:
            captured_calls.append((args, kwargs))
            original_init(self, *args, **kwargs)

        with patch.object(ACPEventConverter, "__init__", _capture_init):
            await session.process_prompt([TextContentBlock(text="hello")])

        assert len(captured_calls) == 1
        _args, kwargs = captured_calls[0]
        assert kwargs.get("client_supports_turn_complete") is False

    @pytest.mark.anyio
    async def test_process_prompt_defaults_turn_complete_when_none(
        self,
        agent_pool: AgentPool,
        mock_acp_agent: MagicMock,
    ) -> None:
        """When client_capabilities.turn_complete=None, ACPEventConverter must be.

        created with client_supports_turn_complete=False (default).

        """
        agent = agent_pool.manifest.agents["test_agent"].get_agent(pool=agent_pool)
        mock_client = AsyncMock()

        session = ACPSession(
            session_id="test-session",
            agent=agent,
            cwd="/tmp",
            client=mock_client,
            acp_agent=mock_acp_agent,
            client_capabilities=ClientCapabilities(turn_complete=None),
        )

        agent.run_stream = _run_stream_empty  # type: ignore[method-assign]

        @asynccontextmanager
        async def _noop_ctx(*args: Any, **kwargs: Any) -> AsyncIterator[None]:
            yield

        agent._with_session_capabilities = _noop_ctx  # type: ignore[method-assign]

        captured_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        original_init = ACPEventConverter.__init__

        def _capture_init(self: ACPEventConverter, *args: Any, **kwargs: Any) -> None:
            captured_calls.append((args, kwargs))
            original_init(self, *args, **kwargs)

        with patch.object(ACPEventConverter, "__init__", _capture_init):
            await session.process_prompt([TextContentBlock(text="hello")])

        assert len(captured_calls) == 1
        _args, kwargs = captured_calls[0]
        assert kwargs.get("client_supports_turn_complete") is False
