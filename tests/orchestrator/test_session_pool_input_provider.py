"""Tests for SessionPool input_provider propagation.

Verifies that input_provider is correctly forwarded through SessionPool
run_stream -> _run_stream_run_turn -> get_or_create_session_agent
so that elicitation does NOT fall back to StdlibInputProvider.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock

import pytest

from wolfharness.agents.events import RunStartedEvent, StreamCompleteEvent
from wolfharness.messaging import ChatMessage
from wolfharness.ui.base import InputProvider


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from wolfharness import AgentPool
    from wolfharness.agents.context import AgentRunContext
    from wolfharness.orchestrator.core import SessionPool


pytestmark = pytest.mark.unit


class FakeInputProvider(InputProvider):
    """A fake input provider for testing propagation."""

    async def get_tool_confirmation(self, context: Any, tool_description: str = "") -> Any:
        return "allow"

    async def get_elicitation(self, params: Any) -> Any:
        return {"action": "accept", "content": {}}


@pytest.fixture
def session_pool(minimal_pool: AgentPool) -> SessionPool:
    """Return a SessionPool backed by the real pool."""
    assert minimal_pool.session_pool is not None
    return minimal_pool.session_pool


@pytest.fixture
def mock_agent() -> MagicMock:
    """Return a mocked BaseAgent that captures kwargs passed to _stream_events."""
    agent = MagicMock()
    agent.get_active_run_context.return_value = None
    agent.AGENT_TYPE = "native"
    agent._input_provider = None

    async def _fake_stream(
        run_ctx: AgentRunContext,
        *prompts: Any,
        **kwargs: Any,
    ) -> AsyncIterator[Any]:
        """Yield a RunStartedEvent then a StreamCompleteEvent."""
        session_id = kwargs.get("session_id", "default")
        yield RunStartedEvent(session_id=session_id, run_id="run-1")
        msg = ChatMessage(content="test response", role="assistant")
        yield StreamCompleteEvent(message=msg)

    agent._stream_events = _fake_stream
    return agent


def _make_fake_run_handle(agent: MagicMock) -> MagicMock:
    """Create a fake RunHandle whose start() yields events from the mock agent."""
    fake_handle = MagicMock()
    fake_handle.run_id = "fake-run-id"
    fake_handle.session_id = "test-session"
    fake_handle.agent_type = "native"
    fake_handle.status = "pending"
    fake_handle.agent = agent
    fake_handle.steer = MagicMock(return_value=False)
    fake_handle.cancel = MagicMock()
    fake_handle.complete = MagicMock()

    async def _fake_start(initial_prompt: str) -> AsyncIterator[Any]:
        """Yield a RunStartedEvent then a StreamCompleteEvent."""
        yield RunStartedEvent(session_id="test-session", run_id="fake-run-id")
        msg = ChatMessage(content="test response", role="assistant")
        yield StreamCompleteEvent(message=msg)

    fake_handle.start = _fake_start
    return fake_handle


class TestSessionPoolRunStreamInputProvider:
    """RED FLAG: input_provider must be forwarded through SessionPool.run_stream()."""

    @pytest.mark.anyio
    async def test_run_stream_without_input_provider_does_not_crash(
        self,
        session_pool: SessionPool,
        minimal_pool: AgentPool,
        mock_agent: MagicMock,
    ) -> None:
        """run_stream() without input_provider should still work (backward compat)."""
        session_id = "test-session"

        # Mock get_or_create_session_agent to return mock_agent
        async def _fake_get_agent(
            sid: str,
            agent_name: str | None = None,
            input_provider: Any | None = None,
        ) -> MagicMock:
            return mock_agent

        session_pool.sessions.get_or_create_session_agent = _fake_get_agent

        # Mock _create_run_handle to avoid needing a real RunHandle
        fake_handle = _make_fake_run_handle(mock_agent)
        session_pool._create_run_handle = MagicMock(return_value=fake_handle)  # type: ignore[method-assign]

        # Count events to ensure stream completes
        event_count = 0
        async for _event in session_pool.run_stream(session_id, "hello"):
            event_count += 1

        assert event_count > 0, "Stream should yield events even without input_provider"

    @pytest.mark.anyio
    async def test_process_prompt_forwards_kwargs_to_run_turn(
        self,
        session_pool: SessionPool,
        minimal_pool: AgentPool,
        mock_agent: MagicMock,
    ) -> None:
        """process_prompt() must forward input_provider to get_or_create_session_agent.

        This tests the middle layer: process_prompt -> _process_prompt_run_turn
        -> get_or_create_session_agent.
        """
        fake_provider = FakeInputProvider()
        session_id = "test-session"

        captured_kwargs: dict[str, Any] | None = None

        async def _capturing_get_agent(
            sid: str,
            agent_name: str | None = None,
            input_provider: Any | None = None,
        ) -> MagicMock:
            nonlocal captured_kwargs
            captured_kwargs = {"input_provider": input_provider}
            return mock_agent

        session_pool.sessions.get_or_create_session_agent = _capturing_get_agent

        # Mock _create_run_handle to avoid needing a real RunHandle
        fake_handle = _make_fake_run_handle(mock_agent)
        session_pool._create_run_handle = MagicMock(return_value=fake_handle)  # type: ignore[method-assign]

        await session_pool.process_prompt(session_id, "hello", input_provider=fake_provider)

        assert captured_kwargs is not None, "get_or_create_session_agent was never called"
        assert captured_kwargs["input_provider"] is fake_provider, (
            f"Expected FakeInputProvider in get_or_create_session_agent, "
            f"got {captured_kwargs['input_provider']}"
        )


class TestRunTurnInputProvider:
    """Tests for _run_stream_run_turn forwarding input_provider to agent creation and stream."""

    @pytest.mark.anyio
    async def test_run_turn_passes_input_provider_to_get_or_create_session_agent(
        self,
        session_pool: SessionPool,
        minimal_pool: AgentPool,
        mock_agent: MagicMock,
    ) -> None:
        """_run_stream_run_turn must pass input_provider to get_or_create_session_agent."""
        fake_provider = FakeInputProvider()
        session_id = "test-session"

        captured_agent_kwargs: dict[str, Any] | None = None

        async def _capturing_get_agent(
            sid: str,
            agent_name: str | None = None,
            input_provider: Any | None = None,
        ) -> MagicMock:
            nonlocal captured_agent_kwargs
            captured_agent_kwargs = {"input_provider": input_provider}
            return mock_agent

        session_pool.sessions.get_or_create_session_agent = _capturing_get_agent

        # Mock _create_run_handle to avoid needing a real RunHandle
        fake_handle = _make_fake_run_handle(mock_agent)
        session_pool._create_run_handle = MagicMock(return_value=fake_handle)  # type: ignore[method-assign]

        # Directly call run_stream (which calls _run_stream_run_turn internally)
        async for _event in session_pool.run_stream(
            session_id, "hello", input_provider=fake_provider
        ):
            pass

        assert captured_agent_kwargs is not None, "get_or_create_session_agent was never called"
        assert captured_agent_kwargs["input_provider"] is fake_provider, (
            f"Expected FakeInputProvider, got {captured_agent_kwargs['input_provider']}"
        )

    @pytest.mark.anyio
    async def test_run_turn_passes_input_provider_to_stream_events(
        self,
        session_pool: SessionPool,
        minimal_pool: AgentPool,
        mock_agent: MagicMock,
    ) -> None:
        """_run_stream_run_turn must set input_provider on the agent.

        In the new API, input_provider is set on the agent via
        ``agent._input_provider = input_provider`` inside
        ``get_or_create_session_agent``, and on the session via
        ``session.input_provider = input_provider``.
        """
        fake_provider = FakeInputProvider()
        session_id = "test-session"

        async def _capturing_get_agent(
            sid: str,
            agent_name: str | None = None,
            input_provider: Any | None = None,
        ) -> MagicMock:
            # Simulate what the real get_or_create_session_agent does:
            # set _input_provider on the agent.
            mock_agent._input_provider = input_provider
            return mock_agent

        session_pool.sessions.get_or_create_session_agent = _capturing_get_agent

        # Mock _create_run_handle so RunHandle.start() works with mock agent
        fake_handle = _make_fake_run_handle(mock_agent)
        session_pool._create_run_handle = MagicMock(return_value=fake_handle)  # type: ignore[method-assign]

        # Directly call run_stream (which calls _run_stream_run_turn internally)
        async for _event in session_pool.run_stream(
            session_id, "hello", input_provider=fake_provider
        ):
            pass

        assert mock_agent._input_provider is fake_provider, (
            f"Expected FakeInputProvider on agent._input_provider, got {mock_agent._input_provider}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
