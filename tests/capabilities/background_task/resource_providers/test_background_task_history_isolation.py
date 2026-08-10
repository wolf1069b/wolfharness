"""Red-flag regression test for parallel background task history isolation.

Reproduces the reported issue that parallel background tasks appear to
share input/history between independent subagents.

Root cause: ``BackgroundTaskProvider`` calls ``node.run_stream()`` without
an explicit ``message_history`` parameter.  Inside ``BaseAgent._run_stream_once``
(line 768), the fallback is ``self.conversation`` — a single ``MessageHistory``
instance variable on the node.  When two parallel tasks delegate to the same
agent mode (same node from the pool), they both read and write the same
``MessageHistory``, causing conversation state to leak between independent
subagents.

The code even warns about this in ``BaseAgent.run_stream`` (line 668-669):
"This is single-session state — concurrent run_stream calls on the same
agent instance will overwrite each other's context."

The tests below distinguish between three potential sources of state leakage:
- **Shared message_history** (conversation state) — BUG: currently broken
- **Shared deps** (dependencies dict) — already isolated via spread-merge
- **Shared session_id** — already isolated via ``create_child_session()``
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from pydantic_ai import RunContext
import pytest

from wolfharness import ChatMessage
from wolfharness.agents.base_agent import BaseAgent
from wolfharness.agents.context import AgentContext
from wolfharness.agents.events import StreamCompleteEvent
from wolfharness.capabilities.background_task.capability import (
    BackgroundTaskCapability,
)
from wolfharness.delegation import AgentPool
from wolfharness.messaging import MessageHistory


pytestmark = pytest.mark.anyio


def _wrap_in_run_context(agent_ctx):
    """Wrap an AgentContext in a mock RunContext for capability tool methods."""
    run_ctx = MagicMock(spec=RunContext)
    run_ctx.deps = agent_ctx
    run_ctx.tool_call_id = agent_ctx.tool_call_id
    return run_ctx


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_mock_node(*, name: str = "worker", description: str = "A worker agent") -> MagicMock:
    """Create a mock node that looks like a BaseAgent with run_stream support."""
    node = MagicMock(spec=BaseAgent)
    node.type = "agent"
    node.name = name
    node.description = description
    node.model_name = "test:model"
    node.session_id = "ses_parent_123"
    # Give the node a real conversation so we can check it stays clean
    node.conversation = MessageHistory()
    return node


def _make_mock_pool(nodes: dict[str, MagicMock] | None = None) -> AgentPool:
    """Create a mock AgentPool with the given nodes."""
    pool = MagicMock(spec=AgentPool)

    if nodes is None:
        mock_node = _make_mock_node()
        nodes = {"worker": mock_node}

    pool.manifest = MagicMock()
    pool.manifest.agents = nodes
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

    # Proxy run_stream calls to the actual node so test captures still work
    def _proxy_run_stream(child_session_id, formatted_prompt, **kwargs):
        return nodes["worker"].run_stream(
            formatted_prompt,
            session_id=child_session_id,
            **kwargs,
        )

    mock_session_pool.run_stream = MagicMock(side_effect=_proxy_run_stream)
    mock_session_pool.inject_prompt = AsyncMock()
    mock_session_pool.steer = AsyncMock()
    mock_session_pool.followup = AsyncMock(return_value=True)

    _receive_calls: list[dict[str, Any]] = []

    async def _send_message(session_id: str, prompt: str, input_provider=None, **kwargs: Any):
        _receive_calls.append({"session_id": session_id, "prompt": prompt, "kwargs": kwargs})
        return MagicMock()

    mock_session_pool.send_message = AsyncMock(side_effect=_send_message)
    mock_session_pool._receive_calls = _receive_calls
    mock_session_pool.event_bus = MagicMock()

    async def _subscribe(session_id: str, scope: str = "session"):
        queue = MagicMock()
        queue.get = AsyncMock(side_effect=asyncio.QueueShutDown())
        return queue

    mock_session_pool.event_bus.subscribe = AsyncMock(side_effect=_subscribe)
    mock_session_pool.event_bus.unsubscribe = AsyncMock()
    pool.session_pool = mock_session_pool
    return pool


def _make_agent_context(
    pool: AgentPool | None = None,
    data: dict[str, Any] | None = None,
    *,
    unique_session_ids: bool = False,
) -> AgentContext:
    """Create a minimal AgentContext for testing.

    Args:
        pool: Agent pool mock
        data: Context data
        unique_session_ids: When True, create_child_session returns unique IDs
            for each call instead of a fixed string.  This matches real behavior
            where ``generate_session_id()`` always returns unique values.
    """
    agent = MagicMock(spec=BaseAgent)
    agent.type = "agent"
    agent.name = "coordinator"
    agent.session_id = "ses_parent_123"
    agent.agent_pool = pool
    agent.inject_prompt = MagicMock()

    ctx = MagicMock(spec=AgentContext)
    ctx.node = agent
    ctx.agent = agent
    ctx.pool = pool
    ctx.data = data if data is not None else {}
    ctx.tool_call_id = "tc_001"

    # Mock AgentRunContext for session state resolution
    mock_run_ctx = MagicMock()
    mock_run_ctx.session_id = "ses_parent_123"
    mock_run_ctx._run_handle = None
    mock_run_ctx.child_done_events = {}
    ctx.run_ctx = mock_run_ctx

    ctx.events = MagicMock()
    ctx.events.emit_event = AsyncMock()
    ctx.internal_fs = MagicMock()
    ctx.internal_fs.mkdirs = MagicMock()
    ctx.internal_fs.pipe = MagicMock()

    if unique_session_ids:
        session_counter = 0

        async def _create_unique_session(**kwargs: Any) -> str:
            nonlocal session_counter
            session_counter += 1
            return f"ses_child_{session_counter}"

        ctx.create_child_session = AsyncMock(side_effect=_create_unique_session)
    else:
        ctx.create_child_session = AsyncMock(return_value="ses_child_456")

    return ctx


async def _collect_stream_events(events: list[Any]) -> Any:
    """Create an async generator that yields the given events."""
    for event in events:
        yield event


# ===========================================================================
# CORE REGRESSION TEST: message_history isolation
#
# When two parallel background tasks are launched with the same agent mode,
# they share the same node instance from the pool.  Without an explicit
# message_history parameter, both tasks use node.conversation — a single
# shared MessageHistory.  This causes history leakage.
# ===========================================================================


class TestPerTaskHistoryIsolation:
    """Red-flag regression tests for parallel background task history isolation.

    These tests verify that each parallel background task receives isolated
    conversation history, not the node's shared ``self.conversation``.
    """

    @pytest.mark.unit
    async def test_async_task_receives_fresh_message_history(self) -> None:
        """A background (async_mode=True) task must pass a fresh
        MessageHistory to run_stream, not reuse the node's conversation.
        """
        capability = BackgroundTaskCapability(schemas=None)
        node = _make_mock_node()
        pool = _make_mock_pool(nodes={"worker": node})
        # Force legacy path to verify message_history is passed to run_stream
        ctx = _make_agent_context(pool=pool)

        captured_kwargs: list[dict[str, Any]] = []

        complete_event = StreamCompleteEvent(
            message=ChatMessage(content="Result", role="assistant"),
        )

        def _capturing_run_stream(*args: Any, **kwargs: Any) -> Any:
            captured_kwargs.append(kwargs)
            return _collect_stream_events([complete_event])

        node.run_stream = MagicMock(side_effect=_capturing_run_stream)

        with patch(
            "wolfharness.capabilities.background_task.capability._generate_task_id",
            return_value="bg_isoasyn1",
        ):
            await capability._task(
                _wrap_in_run_context(ctx),
                agent="worker",
                message="isolated async task",
                async_mode=True,
            )

        await asyncio.sleep(0.3)

        # Async tasks use send_message instead of run_stream
        receive_calls = pool.session_pool._receive_calls
        assert len(receive_calls) == 1, f"Expected 1 send_message call, got {len(receive_calls)}"
        call = receive_calls[0]
        assert call["session_id"] != "ses_parent_123", (
            "session_id must be a new child session, not the parent"
        )

    @pytest.mark.unit
    async def test_sync_task_receives_fresh_message_history(self) -> None:
        """A synchronous (async_mode=False) task must also pass a fresh
        MessageHistory to run_stream.
        """
        capability = BackgroundTaskCapability(schemas=None)
        node = _make_mock_node()
        pool = _make_mock_pool(nodes={"worker": node})
        # Force legacy path to verify message_history is passed to run_stream
        ctx = _make_agent_context(pool=pool)

        captured_kwargs: list[dict[str, Any]] = []

        complete_event = StreamCompleteEvent(
            message=ChatMessage(content="Result", role="assistant"),
        )

        def _capturing_run_stream(*args: Any, **kwargs: Any) -> Any:
            captured_kwargs.append(kwargs)
            return _collect_stream_events([complete_event])

        node.run_stream = MagicMock(side_effect=_capturing_run_stream)

        await capability._task(
            _wrap_in_run_context(ctx),
            agent="worker",
            message="isolated sync task",
            async_mode=False,
        )

        assert len(captured_kwargs) == 1
        kwargs = captured_kwargs[0]

        assert "session_id" in kwargs, (
            "run_stream() must be called with session_id kwarg for sync tasks too"
        )
        assert kwargs["session_id"] != "ses_parent_123"

    @pytest.mark.unit
    async def test_parallel_async_tasks_get_distinct_histories(self) -> None:
        """Two parallel background tasks must each receive a distinct
        MessageHistory object — not the same shared one.

        This is the core red-flag regression test.  When two tasks share
        the same node from the pool, they MUST get independent
        MessageHistory instances to prevent cross-task conversation leakage.
        """
        capability = BackgroundTaskCapability(schemas=None)
        node = _make_mock_node()
        pool = _make_mock_pool(nodes={"worker": node})
        # Force legacy path to verify message_history is passed to run_stream
        ctx = _make_agent_context(pool=pool, unique_session_ids=True)

        captured_kwargs: list[dict[str, Any]] = []

        complete_event = StreamCompleteEvent(
            message=ChatMessage(content="Result", role="assistant"),
        )

        def _capturing_run_stream(*args: Any, **kwargs: Any) -> Any:
            captured_kwargs.append(kwargs)
            return _collect_stream_events([complete_event])

        node.run_stream = MagicMock(side_effect=_capturing_run_stream)

        for i in range(2):
            with patch(
                "wolfharness.capabilities.background_task.capability._generate_task_id",
                return_value=f"bg_parl{i:02d}",
            ):
                await capability._task(
                    _wrap_in_run_context(ctx),
                    agent="worker",
                    message=f"parallel task {i}",
                    async_mode=True,
                )

        await asyncio.sleep(0.3)

        # Async tasks use send_message instead of run_stream
        receive_calls = pool.session_pool._receive_calls
        assert len(receive_calls) == 2, f"Expected 2 send_message calls, got {len(receive_calls)}"

        session_a = receive_calls[0]["session_id"]
        session_b = receive_calls[1]["session_id"]

        assert session_a is not None, "Task 0 must receive a session_id"
        assert session_b is not None, "Task 1 must receive a session_id"
        assert session_a != session_b, "Parallel tasks must receive DISTINCT session_ids"

    @pytest.mark.unit
    async def test_node_conversation_not_polluted_by_async_task(self) -> None:
        """After a background task runs, the node's own conversation must
        NOT contain the task's prompt or response.
        """
        capability = BackgroundTaskCapability(schemas=None)
        node = _make_mock_node()
        pool = _make_mock_pool(nodes={"worker": node})
        ctx = _make_agent_context(pool=pool)

        complete_event = StreamCompleteEvent(
            message=ChatMessage(content="Task result", role="assistant"),
        )

        node.run_stream = MagicMock(
            return_value=_collect_stream_events([complete_event]),
        )

        with patch(
            "wolfharness.capabilities.background_task.capability._generate_task_id",
            return_value="bg_nopoll1",
        ):
            await capability._task(
                _wrap_in_run_context(ctx),
                agent="worker",
                message="secret task prompt",
                async_mode=True,
            )

        await asyncio.sleep(0.3)

        assert len(node.conversation.chat_messages) == 0, (
            "Node's shared conversation must not be polluted by delegated task messages — each task should use an isolated MessageHistory"
        )


# ===========================================================================
# DISTINCTION: session_id isolation (already works)
#
# Session IDs are already unique per task via create_child_session().
# These tests confirm session_id isolation is NOT the broken dimension.
# ===========================================================================


class TestSessionIdIsolation:
    """Verify that session_id is already properly isolated between tasks.

    This distinguishes session_id leakage from message_history leakage.
    """

    @pytest.mark.unit
    async def test_parallel_async_tasks_receive_distinct_session_ids(self) -> None:
        """Each parallel background task must receive a distinct session_id."""
        capability = BackgroundTaskCapability(schemas=None)
        node = _make_mock_node()
        pool = _make_mock_pool(nodes={"worker": node})
        # Force legacy path to verify session_id is passed to run_stream
        ctx = _make_agent_context(pool=pool, unique_session_ids=True)

        captured_kwargs: list[dict[str, Any]] = []

        def _capturing_run_stream(*args: Any, **kwargs: Any) -> Any:
            captured_kwargs.append(kwargs)
            return _collect_stream_events(
                [
                    StreamCompleteEvent(
                        message=ChatMessage(content="Done", role="assistant"),
                    ),
                ],
            )

        node.run_stream = MagicMock(side_effect=_capturing_run_stream)

        for i in range(2):
            with patch(
                "wolfharness.capabilities.background_task.capability._generate_task_id",
                return_value=f"bg_sid{i:02d}",
            ):
                await capability._task(
                    _wrap_in_run_context(ctx),
                    agent="worker",
                    message=f"task {i}",
                    async_mode=True,
                )

        await asyncio.sleep(0.3)

        receive_calls = pool.session_pool._receive_calls
        assert len(receive_calls) == 2
        sid_0 = receive_calls[0]["session_id"]
        sid_1 = receive_calls[1]["session_id"]

        assert sid_0 is not None, "Task 0 session_id must not be None"
        assert sid_1 is not None, "Task 1 session_id must not be None"
        assert sid_0 != sid_1, (
            f"Parallel tasks must have distinct session_ids, got {sid_0!r} and {sid_1!r}"
        )


# ===========================================================================
# DISTINCTION: deps isolation (already works)
#
# deps is already isolated via spread-merge creating a new dict per task.
# These tests confirm deps leakage is NOT the broken dimension.
# ===========================================================================


class TestDepsIsolation:
    """Verify that deps dicts are already properly isolated between tasks.

    This distinguishes deps leakage from message_history leakage.
    """

    @pytest.mark.unit
    async def test_parallel_async_tasks_receive_distinct_deps_dicts(self) -> None:
        """Each parallel background task must receive its own deps dict."""
        capability = BackgroundTaskCapability(schemas=None)
        node = _make_mock_node()
        pool = _make_mock_pool(nodes={"worker": node})
        # Force legacy path to verify deps is passed to run_stream
        ctx = _make_agent_context(
            pool=pool,
            data={"delegation_depth": 0, "project": "diag"},
            unique_session_ids=True,
        )

        captured_kwargs: list[dict[str, Any]] = []

        def _capturing_run_stream(*args: Any, **kwargs: Any) -> Any:
            captured_kwargs.append(kwargs)
            return _collect_stream_events(
                [
                    StreamCompleteEvent(
                        message=ChatMessage(content="Done", role="assistant"),
                    ),
                ],
            )

        node.run_stream = MagicMock(side_effect=_capturing_run_stream)

        for i in range(2):
            with patch(
                "wolfharness.capabilities.background_task.capability._generate_task_id",
                return_value=f"bg_dep{i:02d}",
            ):
                await capability._task(
                    _wrap_in_run_context(ctx),
                    agent="worker",
                    message=f"task {i}",
                    async_mode=True,
                )

        await asyncio.sleep(0.3)

        receive_calls = pool.session_pool._receive_calls
        print(f"DEBUG receive_calls: {receive_calls}")
        assert len(receive_calls) == 2
        deps_0 = receive_calls[0]["kwargs"].get("deps")
        deps_1 = receive_calls[1]["kwargs"].get("deps")

        assert deps_0 is not None, "Task 0 deps must not be None"
        assert deps_1 is not None, "Task 1 deps must not be None"
        assert deps_0 is not deps_1, "Parallel tasks must NOT share the same deps dict object"
        assert deps_0["delegation_depth"] == 1
        assert deps_1["delegation_depth"] == 1
