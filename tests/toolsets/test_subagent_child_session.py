"""Tests for SubagentTools child session creation, spawn dedup, and depth guard.

Verifies RFC-0028 Task T9 requirements:
- Exactly one SpawnSessionStart emitted per delegation from task()
- SpawnSessionStart is emitted from task(), NOT from session stream wrapping
- ctx.run_ctx.depth is used instead of getattr(ctx, "current_depth", 0)
- MAX_DELEGATION_DEPTH guard is enforced before child session creation
- session_id, parent_session_id, and depth are passed into child run_stream()
- No identifier.ascending("session") for provider-owned child IDs
- Child SessionData persists with correct parent_id
- RunStartedEvent.session_id matches SpawnSessionStart.child_session_id
- DelegationDepthError raised at max depth
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from wolfharness import AgentPool, AgentsManifest
from wolfharness.agents.context import AgentContext, AgentRunContext
from wolfharness.agents.events import (
    RunStartedEvent,
    SpawnSessionStart,
    SubAgentEvent,
)
from wolfharness.agents.exceptions import MAX_DELEGATION_DEPTH, DelegationDepthError
from wolfharness.orchestrator.core import EventEnvelope
from wolfharness_storage.memory_provider.provider import MemoryStorageProvider
from wolfharness_toolsets.builtin.subagent_tools import SubagentTools


pytestmark = pytest.mark.integration


def _stream_empty(queue: asyncio.Queue[Any]) -> bool:
    return queue.empty()


# ---------------------------------------------------------------------------
# Single SpawnSessionStart per delegation (integration-level)
# ---------------------------------------------------------------------------


async def test_single_spawn_session_start_per_delegation() -> None:
    """task() emits exactly one SpawnSessionStart — not duplicated by stream wrapping."""
    manifest = AgentsManifest.from_yaml("""
agents:
  worker:
    model:
      type: test
      custom_output_text: "Done"
    system_prompt: You are a worker.

  orchestrator:
    model:
      type: test
      call_tools: ["task"]
      tool_args:
        task:
          agent_or_team: worker
          prompt: "Do work"
          description: "Test spawn"
    tools:
      - type: subagent
""")
    spawn_count = 0

    async with AgentPool(manifest) as pool:
        orchestrator: Any = pool.manifest.agents["orchestrator"].get_agent(pool=pool)

        async for envelope in orchestrator.run_stream("Delegate", session_id="ses_test"):
            event = envelope.event if isinstance(envelope, EventEnvelope) else envelope
            if isinstance(event, SpawnSessionStart):
                spawn_count += 1

    assert spawn_count == 1, (
        f"Expected exactly 1 SpawnSessionStart, got {spawn_count}. "
        "The event should be emitted once from task(), not duplicated by session stream."
    )


# ---------------------------------------------------------------------------
# RunStartedEvent.session_id == SpawnSessionStart.child_session_id
# ---------------------------------------------------------------------------


async def test_run_started_session_id_matches_spawn_child_id() -> None:
    """RunStartedEvent from child agent carries same session_id as SpawnSessionStart.

    SpawnSessionStart.child_session_id should match.
    """
    manifest = AgentsManifest.from_yaml("""
agents:
  worker:
    model:
      type: test
      custom_output_text: "Child done"
    system_prompt: You are a worker.

  orchestrator:
    model:
      type: test
      call_tools: ["task"]
      tool_args:
        task:
          agent_or_team: worker
          prompt: "Work"
          description: "Session match test"
    tools:
      - type: subagent
""")
    child_session_id_from_spawn: str | None = None
    child_session_ids_from_run_started: list[str] = []

    async with AgentPool(manifest) as pool:
        orchestrator: Any = pool.manifest.agents["orchestrator"].get_agent(pool=pool)
        assert pool.session_pool is not None

        # Subscribe to parent with descendants scope to catch child events
        queue = await pool.session_pool.event_bus.subscribe("ses_test", scope="descendants")

        async for envelope in orchestrator.run_stream("Delegate", session_id="ses_test"):
            event = envelope.event if isinstance(envelope, EventEnvelope) else envelope
            if isinstance(event, SpawnSessionStart):
                child_session_id_from_spawn = event.child_session_id

        # Drain remaining events from the queue
        while True:
            try:
                env = queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            except asyncio.QueueShutDown:
                break
            if env is None:
                break
            # Events are now wrapped in EventEnvelope by the EventBus
            event = env.event if isinstance(env, EventEnvelope) else env
            if isinstance(event, RunStartedEvent):
                child_session_ids_from_run_started.append(event.session_id)
            elif isinstance(event, SubAgentEvent) and isinstance(event.event, RunStartedEvent):
                child_session_ids_from_run_started.append(event.event.session_id)

        await pool.session_pool.event_bus.unsubscribe("ses_test", queue)

    assert child_session_id_from_spawn is not None, "SpawnSessionStart was not emitted"
    assert child_session_ids_from_run_started, "No RunStartedEvent found in child events"
    assert child_session_id_from_spawn in child_session_ids_from_run_started, (
        f"RunStartedEvent.session_id {child_session_ids_from_run_started} "
        f"should contain SpawnSessionStart.child_session_id {child_session_id_from_spawn}"
    )


# ---------------------------------------------------------------------------
# Child SessionData persists with correct parent_id
# ---------------------------------------------------------------------------


async def test_child_session_data_persists_with_parent_id() -> None:
    """Child session created by task() is persisted with correct parent_id."""
    store = MemoryStorageProvider()
    manifest = AgentsManifest.from_yaml("""
agents:
  worker:
    model:
      type: test
      custom_output_text: "Done"
    system_prompt: Worker.

  orchestrator:
    model:
      type: test
      call_tools: ["task"]
      tool_args:
        task:
          agent_or_team: worker
          prompt: "Work"
          description: "Persist test"
    tools:
      - type: subagent
""")

    async with AgentPool(manifest) as pool:
        # Swap in our observable store
        assert pool.session_pool is not None
        pool.session_pool.sessions.store = store

        orch: Any = pool.manifest.agents["orchestrator"].get_agent(pool=pool)

        child_session_id_from_spawn: str | None = None

        async for envelope in orch.run_stream("Delegate", session_id="ses_test"):
            event = envelope.event if isinstance(envelope, EventEnvelope) else envelope
            if isinstance(event, SpawnSessionStart):
                child_session_id_from_spawn = event.child_session_id

        assert child_session_id_from_spawn is not None, "SpawnSessionStart not emitted"

        # Verify child session was persisted (must check before pool shutdown)
        child_data = await store.load_session(child_session_id_from_spawn)
        assert child_data is not None, (
            f"Child session {child_session_id_from_spawn} was not persisted in store"
        )
        assert child_data.parent_id == "ses_test", (
            f"Child parent_id={child_data.parent_id}, expected=ses_test"
        )
        assert child_data.agent_name == "worker"


# ---------------------------------------------------------------------------
# DelegationDepthError at max depth
# ---------------------------------------------------------------------------


async def test_delegation_depth_error_at_max_depth() -> None:
    """DelegationDepthError is raised when current depth >= MAX_DELEGATION_DEPTH."""
    manifest = AgentsManifest.from_yaml("""
agents:
  worker:
    model:
      type: test
      custom_output_text: "Done"
    system_prompt: Worker.

  orchestrator:
    model:
      type: test
      call_tools: ["task"]
      tool_args:
        task:
          agent_or_team: worker
          prompt: "Work"
          description: "Depth test"
    tools:
      - type: subagent
""")
    async with AgentPool(manifest) as pool:
        orch: Any = pool.manifest.agents["orchestrator"].get_agent(pool=pool)

        tools_provider = SubagentTools()

        ctx = AgentContext(node=orch)
        ctx.pool = pool
        ctx.run_ctx = AgentRunContext(depth=MAX_DELEGATION_DEPTH)

        with pytest.raises(DelegationDepthError) as exc_info:
            await tools_provider.task(
                ctx=ctx,
                agent_or_team="worker",
                prompt="Should fail",
                description="Depth overflow",
            )

        assert exc_info.value.current_depth == MAX_DELEGATION_DEPTH + 1


# ---------------------------------------------------------------------------
# Depth guard enforced BEFORE child session creation
# ---------------------------------------------------------------------------


async def test_depth_guard_before_session_creation() -> None:
    """When depth >= MAX_DELEGATION_DEPTH, no child session is created."""
    manifest = AgentsManifest.from_yaml("""
agents:
  worker:
    model:
      type: test
      custom_output_text: "Done"
    system_prompt: Worker.

  orchestrator:
    model:
      type: test
      call_tools: ["task"]
      tool_args:
        task:
          agent_or_team: worker
          prompt: "Work"
          description: "Depth guard test"
    tools:
      - type: subagent
""")
    async with AgentPool(manifest) as pool:
        orch: Any = pool.manifest.agents["orchestrator"].get_agent(pool=pool)

        tools_provider = SubagentTools()

        ctx = AgentContext(node=orch)
        ctx.pool = pool
        ctx.run_ctx = AgentRunContext(depth=MAX_DELEGATION_DEPTH)

        # create_child_session should NOT be called because depth guard fires first
        with (
            patch.object(ctx, "create_child_session", new_callable=AsyncMock) as mock_create,
            pytest.raises(DelegationDepthError),
        ):
            await tools_provider.task(
                ctx=ctx,
                agent_or_team="worker",
                prompt="Should fail before session creation",
                description="Depth guard",
            )

        mock_create.assert_not_called()


# ---------------------------------------------------------------------------
# Depth uses ctx.run_ctx.depth, not getattr
# ---------------------------------------------------------------------------


async def test_task_uses_run_ctx_depth() -> None:
    """task() uses ctx.run_ctx.depth for computing delegation depth."""
    manifest = AgentsManifest.from_yaml("""
agents:
  worker:
    model:
      type: test
      custom_output_text: "Done"
    system_prompt: Worker.

  orchestrator:
    model:
      type: test
      call_tools: ["task"]
      tool_args:
        task:
          agent_or_team: worker
          prompt: "Work"
          description: "Depth source test"
    tools:
      - type: subagent
""")
    spawn_depth: int | None = None

    async with AgentPool(manifest) as pool:
        orch: Any = pool.manifest.agents["orchestrator"].get_agent(pool=pool)

        # With depth=0 (default top-level), child should be depth=1
        async for envelope in orch.run_stream("Delegate", session_id="ses_test"):
            event = envelope.event if isinstance(envelope, EventEnvelope) else envelope
            if isinstance(event, SpawnSessionStart):
                spawn_depth = event.depth

    assert spawn_depth == 1, f"Expected depth=1 for first delegation, got {spawn_depth}"


# ---------------------------------------------------------------------------
# No identifier.ascending("session") for child IDs in subagent_tools.py
# ---------------------------------------------------------------------------


async def test_subagent_tools_does_not_import_identifiers() -> None:
    """subagent_tools module does not import identifier — uses create_child_session instead."""
    import wolfharness_toolsets.builtin.subagent_tools as mod

    # The module should not have 'identifier' in its namespace
    assert not hasattr(mod, "identifier"), (
        "subagent_tools should not import 'identifier' — it should use ctx.create_child_session()"
    )
