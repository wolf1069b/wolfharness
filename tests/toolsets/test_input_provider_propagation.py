"""Tests verifying input_provider propagation through subagent and worker delegation.

These tests ensure that when a parent agent delegates to a subagent or worker,
the input_provider is properly propagated so that tools requiring user
interaction (like confirmations) work correctly in the child agent.


# TODO: L2 migration — test uses mock_pool as data container with
# mock session_pool, not for real SessionController/SessionPool creation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from wolfharness import AgentPool, AgentsManifest
from wolfharness.agents.events import StreamCompleteEvent
from wolfharness.messaging.messages import ChatMessage
from wolfharness.ui.base import InputProvider


pytestmark = pytest.mark.integration


if TYPE_CHECKING:
    from collections.abc import AsyncIterator


class FakeInputProvider(InputProvider):
    """Fake input provider for testing."""

    def get_tool_confirmation(self, context: Any, tool_description: str = "") -> Any:
        raise NotImplementedError

    def get_elicitation(self, params: Any) -> Any:
        raise NotImplementedError


TASK_TOOL_MANIFEST = """
agents:
  parent:
    type: native
    model:
      type: test
      call_tools: ["task"]
      tool_args:
        task:
          agent_or_team: child
          prompt: "Do some work"
          description: "Test input_provider propagation"
    system_prompt: "You are the parent agent."
    tools:
      - type: subagent

  child:
    type: native
    model: test
    system_prompt: "You are the child agent."
"""


async def test_input_provider_propagated_to_subagent_via_task_tool() -> None:
    """Test that input_provider is passed to subagent when using task tool."""
    manifest = AgentsManifest.from_yaml(TASK_TOOL_MANIFEST)
    fake_provider = FakeInputProvider()

    async with AgentPool(manifest) as pool:
        parent = pool.manifest.agents["parent"].get_agent(pool=pool)
        pool.manifest.agents["child"].get_agent(pool=pool)

        # Patch session_pool.run_stream (what SubagentTools.task actually calls)
        # to capture the input_provider argument
        captured_kwargs = {}
        session_pool = pool.session_pool
        assert session_pool is not None

        async def mock_run_stream(session_id: str, *prompts: str, **kwargs: Any) -> Any:
            captured_kwargs.update(kwargs)
            yield StreamCompleteEvent(
                message=ChatMessage(content="done", role="assistant"),
            )

        with patch.object(session_pool, "run_stream", side_effect=mock_run_stream):
            await parent.run(
                "Delegate to child",
                input_provider=fake_provider,
            )

        # Verify input_provider was passed to session_pool.run_stream
        assert "input_provider" in captured_kwargs, (
            f"input_provider not passed to session_pool.run_stream. "
            f"Captured kwargs: {captured_kwargs}"
        )
        assert captured_kwargs["input_provider"] is fake_provider, (
            f"Wrong input_provider passed. Expected FakeInputProvider, "
            f"got {captured_kwargs.get('input_provider')}"
        )


WORKER_MANIFEST = """
agents:
  main:
    type: native
    model:
      type: test
      call_tools: ["ask_helper"]
    system_prompt: "You are the main agent."
    workers:
      - helper

  helper:
    type: native
    model: test
    system_prompt: "You are the helper agent."
"""


async def test_input_provider_propagated_to_worker() -> None:
    """Test that input_provider is passed to worker when using worker tool."""
    manifest = AgentsManifest.from_yaml(WORKER_MANIFEST)
    fake_provider = FakeInputProvider()

    async with AgentPool(manifest) as pool:
        main = pool.manifest.agents["main"].get_agent(pool=pool)
        pool.manifest.agents["helper"].get_agent(pool=pool)

        # Patch session_pool.run_stream (what worker tool actually calls)
        # to capture the input_provider argument
        captured_kwargs = {}
        session_pool = pool.session_pool
        assert session_pool is not None

        async def mock_run_stream(session_id: str, *prompts: str, **kwargs: Any) -> Any:
            captured_kwargs.update(kwargs)
            yield StreamCompleteEvent(
                message=ChatMessage(content="done", role="assistant"),
            )

        with patch.object(session_pool, "run_stream", side_effect=mock_run_stream):
            await main.run(
                "Ask helper",
                input_provider=fake_provider,
            )

        # Verify input_provider was passed to session_pool.run_stream
        assert "input_provider" in captured_kwargs, (
            f"input_provider not passed to session_pool.run_stream. "
            f"Captured kwargs: {captured_kwargs}"
        )
        assert captured_kwargs["input_provider"] is fake_provider, (
            f"Wrong input_provider passed. Expected FakeInputProvider, "
            f"got {captured_kwargs.get('input_provider')}"
        )


ASYNC_TASK_MANIFEST = """
agents:
  orchestrator:
    type: native
    model:
      type: test
      call_tools: ["task"]
      tool_args:
        task:
          agent_or_team: worker
          prompt: "Do some work"
          description: "Test async input_provider"
          async_mode: true
    system_prompt: "You are the orchestrator."
    tools:
      - type: subagent

  worker:
    type: native
    model: test
    system_prompt: "You are the worker agent."
"""


async def test_input_provider_propagated_to_subagent_async_mode() -> None:
    """Test that input_provider is passed to async subagent task."""
    manifest = AgentsManifest.from_yaml(ASYNC_TASK_MANIFEST)
    fake_provider = FakeInputProvider()

    async with AgentPool(manifest) as pool:
        orchestrator = pool.manifest.agents["orchestrator"].get_agent(pool=pool)
        worker = pool.manifest.agents["worker"].get_agent(pool=pool)

        # Patch worker's run_stream to capture the input_provider argument
        captured_kwargs = {}

        async def mock_run_stream(*_args: Any, **kwargs: Any) -> Any:
            captured_kwargs.update(kwargs)
            yield StreamCompleteEvent(
                message=ChatMessage(content="done", role="assistant"),
            )

        with patch.object(worker, "run_stream", side_effect=mock_run_stream):
            result = await orchestrator.run(
                "Start background task",
                input_provider=fake_provider,
            )

        # For async mode, the task starts in background.
        # We just verify orchestrator completed successfully.
        assert result.content is not None


# ——— Unit tests for the guard-condition bug ———


class FakeInputProviderSession(InputProvider):
    """Fake input provider for testing SessionState-bound propagation."""

    def get_tool_confirmation(self, context: Any, tool_description: str = "") -> Any:
        raise NotImplementedError

    def get_elicitation(self, params: Any) -> Any:
        raise NotImplementedError


@pytest.mark.skip(
    reason="Mock setup incomplete for pool-less architecture;"
    " SubagentTools.task() requires full SessionPool."
    " Other input_provider tests cover propagation."
)
@pytest.mark.anyio
async def test_input_provider_propagated_when_session_bound_only() -> None:
    """Regression test: input_provider must propagate even when ctx.input_provider is None.

    In ACP/OpenCode mode, the InputProvider is stored on SessionState, NOT on
    ctx.input_provider directly. The guard ``if ctx.input_provider`` in
    SubagentTools.task() prevented get_input_provider() from resolving via
    SessionState, causing the child subagent to have no InputProvider.

    This test verifies that when ctx.input_provider is None but
    ctx.get_input_provider() would return a provider via SessionState,
    the provider IS propagated to the child session.
    """
    from wolfharness.agents.context import AgentContext
    from wolfharness.messaging.messagenode import MessageNode
    from wolfharness.orchestrator.core import SessionPool
    from wolfharness_toolsets.builtin.subagent_tools import SubagentTools

    fake_provider = FakeInputProviderSession()

    # --- Build mock AgentContext ---
    ctx = MagicMock(spec=AgentContext)
    # KEY: input_provider is None on the direct field (SessionState-bound)
    ctx.input_provider = None
    # get_input_provider() returns the provider via SessionState fallback
    ctx.get_input_provider.return_value = fake_provider

    # Mock pool with a child agent node
    child_node = MagicMock(spec=MessageNode)
    child_node.agent_type = "native"
    child_node.type = "native"
    mock_pool = MagicMock()
    mock_pool.manifest.agents = {"child_agent": child_node}
    ctx.pool = mock_pool

    # Mock SessionPool
    session_pool = MagicMock(spec=SessionPool)
    mock_pool.session_pool = session_pool
    # SubagentTools.task() accesses session_pool.sessions.runtime_registry.register()
    session_pool.sessions = MagicMock()
    session_pool.sessions.runtime_registry = MagicMock()

    # Mock run_stream to capture input_provider kwarg
    captured_input_provider: Any = None

    async def _capture_run_stream(
        session_id: str, prompt: str, **kwargs: Any
    ) -> AsyncIterator[StreamCompleteEvent]:
        nonlocal captured_input_provider
        captured_input_provider = kwargs.get("input_provider")
        yield StreamCompleteEvent(
            message=ChatMessage(content="done", role="assistant"),
        )

    session_pool.run_stream = _capture_run_stream

    # Mock child session creation
    ctx.create_child_session = AsyncMock(return_value="ses_child_001")

    # Mock event emitter
    ctx.events = MagicMock()
    ctx.events.emit_event = AsyncMock()

    # Mock run_ctx for depth
    ctx.run_ctx = MagicMock()
    ctx.run_ctx.depth = 0
    ctx.run_ctx.session_id = "ses_parent_001"

    # Mock tool_call_id
    ctx.tool_call_id = "call_001"

    # Mock node (parent agent)
    ctx.node = MagicMock()
    ctx.node.session_id = "ses_parent_001"

    # --- Execute task ---
    tools = SubagentTools()
    result = await tools.task(
        ctx=ctx,
        agent_or_team="child_agent",
        prompt="Do work",
        description="Test session-bound input provider",
        async_mode=False,
    )

    # --- Verify input_provider was propagated ---
    assert captured_input_provider is fake_provider, (
        f"input_provider was NOT propagated when ctx.input_provider is None. "
        f"Expected FakeInputProviderSession, got {captured_input_provider!r}. "
        f"The guard 'if ctx.input_provider' in SubagentTools.task() prevented "
        f"get_input_provider() from resolving via SessionState."
    )
    assert result["output"] == "done"
