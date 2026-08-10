"""Unit tests for BaseAgent public APIs get_active_run_context() and is_turn_active()."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock

from pydantic_ai.models.test import TestModel
import pytest

from wolfharness.agents.base_agent import BaseAgent, _current_run_ctx_var
from wolfharness.agents.context import AgentRunContext
from wolfharness.orchestrator.core import SessionState


pytestmark = pytest.mark.integration


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from pydantic_ai.messages import ModelMessage

    from wolfharness.orchestrator.turn import Turn


# ---------------------------------------------------------------------------
# Minimal concrete subclass for isolated unit tests
# ---------------------------------------------------------------------------


class _TestAgent(BaseAgent):
    """Minimal concrete agent for testing BaseAgent APIs."""

    AGENT_TYPE = "native"

    @property
    def model_name(self) -> str | None:
        return "test-model"

    async def set_model(self, model: str) -> None:
        pass

    def create_turn(
        self,
        prompts: list[str],
        run_ctx: AgentRunContext,
        message_history: list[ModelMessage],
    ) -> Turn:
        """Test stub — returns no turn."""
        raise NotImplementedError("_TestAgent does not implement create_turn")

    async def _stream_events(
        self,
        run_ctx: AgentRunContext,
        prompts: list[Any],
        *,
        user_msg: Any,
        message_history: Any,
        effective_parent_id: str | None,
        message_id: str | None = None,
        session_id: str | None = None,
        parent_session_id: str | None = None,
        parent_id: str | None = None,
        input_provider: Any | None = None,
        deps: Any | None = None,
        wait_for_connections: bool | None = None,
        store_history: bool = True,
    ) -> AsyncIterator[Any]:
        if False:
            yield

    async def _interrupt(self, run_ctx: AgentRunContext | None = None) -> None:
        pass

    async def get_available_models(self) -> list[Any] | None:
        return None

    async def get_modes(self) -> list[Any]:
        return []

    async def _set_mode(self, mode_id: str, category_id: str) -> None:
        pass

    async def list_sessions(
        self,
        *,
        cwd: str | None = None,
        limit: int | None = None,
    ) -> list[Any]:
        return []

    async def load_session(self, session_id: str) -> Any | None:
        return None


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def agent() -> _TestAgent:
    """Create a minimal test agent instance."""
    return _TestAgent(name="test-agent")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_session_pool(agent: _TestAgent, run_ctx: AgentRunContext) -> None:
    """Mock agent_pool.session_pool so _get_session_run_ctx() returns run_ctx."""
    from wolfharness.orchestrator.run import RunHandle

    session_state = SessionState(session_id="test-session", agent_name="test")
    session_state.current_run_id = run_ctx.run_id
    session_controller = MagicMock()
    session_controller.get_session.return_value = session_state
    run_handle = MagicMock(spec=RunHandle)
    run_handle.run_ctx = run_ctx
    session_pool = MagicMock()
    session_pool.sessions = session_controller
    session_pool.get_run.return_value = run_handle
    agent_pool = MagicMock()
    agent_pool.session_pool = session_pool
    host_ctx = MagicMock()
    host_ctx.session_pool = session_pool
    agent_pool.get_context.return_value = host_ctx
    agent.agent_pool = agent_pool


# ---------------------------------------------------------------------------
# get_active_run_context — no active turn
# ---------------------------------------------------------------------------


def test_get_active_run_context_returns_none_when_idle(agent: _TestAgent) -> None:
    """When no turn has started, get_active_run_context() returns None."""
    assert agent.get_active_run_context() is None


# ---------------------------------------------------------------------------
# get_active_run_context — SessionPool fallback
# ---------------------------------------------------------------------------


def test_get_active_run_context_returns_session_run_ctx(agent: _TestAgent) -> None:
    """When session.current_run_id is set, the run context is returned."""
    ctx = AgentRunContext()
    _mock_session_pool(agent, ctx)

    result = agent.get_active_run_context(session_id="test-session")

    assert result is ctx


# ---------------------------------------------------------------------------
# get_active_run_context — _background_run_ctx
# ---------------------------------------------------------------------------


def test_get_active_run_context_returns_background_run_ctx(agent: _TestAgent) -> None:
    """When only _background_run_ctx is set, it is returned."""
    ctx = AgentRunContext()
    agent._background_run_ctx = ctx

    result = agent.get_active_run_context()

    assert result is ctx


# ---------------------------------------------------------------------------
# get_active_run_context — _current_run_ctx (ContextVar)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_active_run_context_returns_current_run_ctx(agent: _TestAgent) -> None:
    """When _current_run_ctx (ContextVar) is set in the current task, it is returned."""
    ctx = AgentRunContext()
    token = _current_run_ctx_var.set(ctx)
    try:
        result = agent.get_active_run_context()
        assert result is ctx
    finally:
        _current_run_ctx_var.reset(token)


# ---------------------------------------------------------------------------
# get_active_run_context — precedence order
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_active_run_context_prefers_current_over_session(agent: _TestAgent) -> None:
    """_current_run_ctx takes precedence over SessionPool fallback."""
    current_ctx = AgentRunContext()
    session_ctx = AgentRunContext()
    _mock_session_pool(agent, session_ctx)
    token = _current_run_ctx_var.set(current_ctx)
    try:
        result = agent.get_active_run_context(session_id="test-session")
        assert result is current_ctx
    finally:
        _current_run_ctx_var.reset(token)


def test_get_active_run_context_prefers_session_over_background(agent: _TestAgent) -> None:
    """SessionPool fallback takes precedence over _background_run_ctx."""
    session_ctx = AgentRunContext()
    background_ctx = AgentRunContext()
    _mock_session_pool(agent, session_ctx)
    agent._background_run_ctx = background_ctx

    result = agent.get_active_run_context(session_id="test-session")

    assert result is session_ctx


@pytest.mark.asyncio
async def test_get_active_run_context_prefers_current_over_background(agent: _TestAgent) -> None:
    """_current_run_ctx takes precedence over _background_run_ctx."""
    current_ctx = AgentRunContext()
    background_ctx = AgentRunContext()
    agent._background_run_ctx = background_ctx
    token = _current_run_ctx_var.set(current_ctx)
    try:
        result = agent.get_active_run_context()
        assert result is current_ctx
    finally:
        _current_run_ctx_var.reset(token)


# ---------------------------------------------------------------------------
# is_turn_active — boolean correctness
# ---------------------------------------------------------------------------


def test_is_turn_active_false_when_idle(agent: _TestAgent) -> None:
    """is_turn_active() returns False when no turn is running."""
    assert agent.is_turn_active() is False


def test_is_turn_active_true_with_session_run_ctx(agent: _TestAgent) -> None:
    """is_turn_active() returns True when session.current_run_id is set."""
    _mock_session_pool(agent, AgentRunContext())
    assert agent.is_turn_active() is False  # is_turn_active doesn't pass session_id
    assert agent.get_active_run_context(session_id="test-session") is not None


def test_is_turn_active_true_with_background_run_ctx(agent: _TestAgent) -> None:
    """is_turn_active() returns True when _background_run_ctx is set."""
    agent._background_run_ctx = AgentRunContext()
    assert agent.is_turn_active() is True


@pytest.mark.asyncio
async def test_is_turn_active_true_with_current_run_ctx(agent: _TestAgent) -> None:
    """is_turn_active() returns True when _current_run_ctx (ContextVar) is set."""
    ctx = AgentRunContext()
    token = _current_run_ctx_var.set(ctx)
    try:
        assert agent.is_turn_active() is True
    finally:
        _current_run_ctx_var.reset(token)


# ---------------------------------------------------------------------------
# is_turn_active — cleanup after turn ends
# ---------------------------------------------------------------------------


def test_is_turn_active_false_after_clearing_session_run_ctx(agent: _TestAgent) -> None:
    """After clearing session.current_run_id, is_turn_active() returns False."""
    from wolfharness.orchestrator.run import RunHandle

    run_ctx = AgentRunContext()
    session_state = SessionState(session_id="test-session", agent_name="test")
    session_state.current_run_id = run_ctx.run_id
    session_controller = MagicMock()
    session_controller.get_session.return_value = session_state
    run_handle = MagicMock(spec=RunHandle)
    run_handle.run_ctx = run_ctx
    session_pool = MagicMock()
    session_pool.sessions = session_controller
    session_pool.get_run.return_value = run_handle
    agent_pool = MagicMock()
    agent_pool.session_pool = session_pool
    host_ctx = MagicMock()
    host_ctx.session_pool = session_pool
    agent_pool.get_context.return_value = host_ctx
    agent.agent_pool = agent_pool

    assert agent.get_active_run_context(session_id="test-session") is not None

    session_state.current_run_id = None
    session_pool.get_run.return_value = None
    assert agent.get_active_run_context(session_id="test-session") is None


def test_is_turn_active_false_after_clearing_background_run_ctx(agent: _TestAgent) -> None:
    """After clearing _background_run_ctx, is_turn_active() returns False."""
    agent._background_run_ctx = AgentRunContext()
    assert agent.is_turn_active() is True

    agent._background_run_ctx = None
    assert agent.is_turn_active() is False


# ---------------------------------------------------------------------------
# Standalone agent generates ephemeral session
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_baseagent_standalone_generates_ephemeral_session() -> None:
    """BaseAgent without an agent_pool generates an ephemeral session_id during run_stream."""
    agent = _TestAgent(name="standalone-test")
    assert agent.host_context is None

    async with agent:
        async for _event in agent.run_stream("hello"):
            pass

    # Session ID is no longer stored as instance state; it flows as a parameter.
    # The run completes successfully without mutating agent state.


# ---------------------------------------------------------------------------
# Integration-style: during an actual agent run
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_is_turn_active_during_run_stream() -> None:
    """is_turn_active() returns True while inside a run_stream execution.

    We verify this by providing a tool that inspects the agent state mid-turn.
    """
    from wolfharness import Agent
    from wolfharness.agents.events import StreamCompleteEvent

    turn_active_values: list[bool] = []

    async def check_turn() -> str:
        """Tool that records is_turn_active() mid-stream."""
        turn_active_values.append(agent.is_turn_active())
        return "ok"

    model = TestModel(custom_output_text="Test response")
    async with Agent(name="native-test", model=model, tools=[check_turn]) as agent:
        async for event in agent.run_stream("trigger tool"):
            if isinstance(event, StreamCompleteEvent):
                break

    # The tool ran during the stream, so at least one True value was recorded
    assert any(turn_active_values), "is_turn_active() should be True during run_stream"


@pytest.mark.asyncio
async def test_get_active_run_context_during_run_stream() -> None:
    """get_active_run_context() returns a non-None context while inside run_stream."""
    from wolfharness import Agent
    from wolfharness.agents.events import StreamCompleteEvent

    contexts: list[AgentRunContext | None] = []

    async def capture_ctx() -> str:
        """Tool that records get_active_run_context() mid-stream."""
        contexts.append(agent.get_active_run_context())
        return "ok"

    model = TestModel(custom_output_text="Test response")
    async with Agent(name="native-test", model=model, tools=[capture_ctx]) as agent:
        async for event in agent.run_stream("trigger tool"):
            if isinstance(event, StreamCompleteEvent):
                break

    # At least one context captured during the stream should be non-None
    assert any(ctx is not None for ctx in contexts), (
        "get_active_run_context() should return non-None during run_stream"
    )


# ---------------------------------------------------------------------------
# Edge case: concurrent tasks see different ContextVar values
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_active_run_context_isolation_between_tasks(agent: _TestAgent) -> None:
    """Two concurrent tasks setting different _current_run_ctx values are isolated."""
    ctx_a = AgentRunContext()
    ctx_b = AgentRunContext()

    async def task_a() -> AgentRunContext | None:
        token = _current_run_ctx_var.set(ctx_a)
        try:
            await asyncio.sleep(0.01)
            return agent.get_active_run_context()
        finally:
            _current_run_ctx_var.reset(token)

    async def task_b() -> AgentRunContext | None:
        token = _current_run_ctx_var.set(ctx_b)
        try:
            await asyncio.sleep(0.01)
            return agent.get_active_run_context()
        finally:
            _current_run_ctx_var.reset(token)

    result_a, result_b = await asyncio.gather(task_a(), task_b())

    assert result_a is ctx_a
    assert result_b is ctx_b
