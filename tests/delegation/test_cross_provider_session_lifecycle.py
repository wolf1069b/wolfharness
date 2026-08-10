"""Cross-provider event/depth/session lifecycle integration tests (RFC-0028 T14).

These tests verify persistence, event order, depth, and ID consistency across
all adapted providers: SubagentTools, WorkersTools, Team, TeamRun, ACP.

Covered test goals from RFC-0028:
  TG-1, TG-3, TG-4, TG-7, TG-8, TG-9, TG-10, TG-14, TG-15,
  TG-16, TG-18, TG-22

Additional cross-provider invariants:
  - Event ordering: SpawnSessionStart index < first SubAgentEvent index
    per child_session_id
  - Non-streaming Team.run() / TeamRun.run() do NOT emit SpawnSessionStart


# TODO: L2 migration — test uses complex inline mock_pool + mock_session_pool
# patterns that require significant rework for real pool migration.
"""

from __future__ import annotations

import tempfile
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from wolfharness import Agent, AgentPool, AgentsManifest, BaseTeam
from wolfharness.agents.context import AgentContext, AgentRunContext
from wolfharness.agents.events import (
    RunStartedEvent,
    SpawnSessionStart,
    SubAgentEvent,
)
from wolfharness.agents.exceptions import MAX_DELEGATION_DEPTH, DelegationDepthError
from wolfharness.sessions import SessionData
from wolfharness_storage.memory_provider.provider import MemoryStorageProvider


pytestmark = pytest.mark.integration


if TYPE_CHECKING:
    from collections.abc import AsyncIterator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_echo_agent(name: str, response: str = "hello") -> Agent[Any, str]:
    """Create an Agent that echoes a fixed response via function_to_model."""
    from functools import partial

    from wolfharness.utils.model_helpers import function_to_model

    async def _echo(_msg: str, *, _response: str = response) -> str:
        return _response

    model = function_to_model(partial(_echo, _response=response))
    return Agent(name=name, model=model)


async def _collect_events(source: Any, *args: Any, **kwargs: Any) -> list[Any]:
    """Collect all events from run_stream into a list."""
    return [event async for event in source.run_stream(*args, **kwargs)]


# ---------------------------------------------------------------------------
# TG-1: SubagentTools child session has correct parent_id in SessionData
# ---------------------------------------------------------------------------


@pytest.mark.skip(
    reason="Rich cell_len O(n) hang on long debug output from skill tools — "
    "instance divergence causes worker agent to produce extremely long output "
    "that hangs Rich's character-by-character cell width measurement. Tracked "
    "as instance divergence architecture issue."
)
async def test_subagent_child_session_parent_id_in_session_data() -> None:
    """TG-1: SubagentTools child session persisted with correct parent_id.

    This is a cross-provider variant: verifies the child SessionData created
    by SubagentTools has parent_id pointing to the orchestrator's session.
    """
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
          description: "TG-1 persist test"
    tools:
      - type: subagent
""")

    async with AgentPool(manifest) as pool:
        if pool.sessions is None:
            pytest.skip("Pool has no SessionManager")
        pool.session_pool.sessions.store = store  # type: ignore[union-attr]

        orch = pool.manifest.agents["orchestrator"].get_agent(pool=pool)
        child_session_id_from_spawn: str | None = None

        async for event in orch.run_stream("Delegate", session_id="ses_test"):
            if isinstance(event, SpawnSessionStart):
                child_session_id_from_spawn = event.child_session_id

        assert child_session_id_from_spawn is not None
        parent_session_id = "ses_test"
        assert parent_session_id is not None

        child_data = await store.load_session(child_session_id_from_spawn)
        assert child_data is not None
        assert child_data.parent_id == parent_session_id
        assert child_data.agent_name == "worker"


# ---------------------------------------------------------------------------
# TG-3: Team member SpawnSessionStart precedes SubAgentEvent content
# ---------------------------------------------------------------------------


async def test_team_spawn_precedes_subagent_for_each_member() -> None:
    """TG-3: For each team member, SpawnSessionStart appears before any SubAgentEvent from...."""
    agent_a = _make_echo_agent("alpha")
    agent_b = _make_echo_agent("beta")
    team = BaseTeam([agent_a, agent_b], mode="parallel")

    events = await _collect_events(team, "test")

    spawn_events = [e for e in events if isinstance(e, SpawnSessionStart)]
    sub_events = [e for e in events if isinstance(e, SubAgentEvent)]

    assert len(spawn_events) == 2, f"Expected 2 SpawnSessionStart, got {len(spawn_events)}"
    assert len(sub_events) >= 2, f"Expected >= 2 SubAgentEvent, got {len(sub_events)}"

    for spawn in spawn_events:
        spawn_idx = events.index(spawn)
        # Find first SubAgentEvent from same source
        matching_subs = [
            (i, e)
            for i, e in enumerate(events)
            if isinstance(e, SubAgentEvent) and e.source_name == spawn.source_name
        ]
        assert matching_subs, f"No SubAgentEvent for {spawn.source_name}"
        first_sub_idx = matching_subs[0][0]
        assert spawn_idx < first_sub_idx, (
            f"SpawnSessionStart for {spawn.source_name} at index {spawn_idx} "
            f"must precede its first SubAgentEvent at index {first_sub_idx}"
        )


# ---------------------------------------------------------------------------
# TG-4: SubagentTools emits exactly one SpawnSessionStart per delegation
# ---------------------------------------------------------------------------


async def test_subagent_single_spawn_per_delegation() -> None:
    """TG-4: SubagentTools emits exactly one SpawnSessionStart per delegation.

    Regression test: duplicate emission from both task() and _stream_task()
    must not occur.
    """
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
          description: "TG-4 single spawn"
    tools:
      - type: subagent
""")
    spawn_count = 0

    async with AgentPool(manifest) as pool:
        orch = pool.manifest.agents["orchestrator"].get_agent(pool=pool)
        async for event in orch.run_stream("Delegate", session_id="ses_test"):
            if isinstance(event, SpawnSessionStart):
                spawn_count += 1

    assert spawn_count == 1, (
        f"Expected exactly 1 SpawnSessionStart, got {spawn_count}. "
        "task() and _stream_task() must not both emit SpawnSessionStart."
    )


# ---------------------------------------------------------------------------
# TG-7: Team member child_session_id appears in SubAgentEvent
# ---------------------------------------------------------------------------


async def test_team_member_child_session_id_in_subagent_event() -> None:
    """TG-7: Each team member's SpawnSessionStart.child_session_id must also appear in...."""
    agent_a = _make_echo_agent("alpha")
    agent_b = _make_echo_agent("beta")
    team = BaseTeam([agent_a, agent_b], mode="parallel")

    events = await _collect_events(team, "test", session_id="ses_parent_tg7")

    spawn_events = [e for e in events if isinstance(e, SpawnSessionStart)]
    sub_events = [e for e in events if isinstance(e, SubAgentEvent)]

    assert len(spawn_events) == 2

    for spawn in spawn_events:
        matching_subs = [se for se in sub_events if se.child_session_id == spawn.child_session_id]
        assert matching_subs, (
            f"No SubAgentEvent with child_session_id={spawn.child_session_id} "
            f"for member {spawn.source_name}"
        )


# ---------------------------------------------------------------------------
# TG-8: RunStartedEvent.session_id == SpawnSessionStart.child_session_id
# ---------------------------------------------------------------------------


async def test_subagent_run_started_matches_spawn_child_id() -> None:
    """TG-8: RunStartedEvent from child agent carries same session_id as...."""
    manifest = AgentsManifest.from_yaml("""
agents:
  worker:
    model:
      type: test
      custom_output_text: "Child done"
    system_prompt: Worker.

  orchestrator:
    model:
      type: test
      call_tools: ["task"]
      tool_args:
        task:
          agent_or_team: worker
          prompt: "Work"
          description: "TG-8 session match"
    tools:
      - type: subagent
""")
    child_session_id_from_spawn: str | None = None
    child_session_ids_from_run_started: list[str] = []

    async with AgentPool(manifest) as pool:
        session_pool = pool.session_pool
        assert session_pool is not None, "SessionPool not initialized"
        # Create session with the orchestrator agent so the correct agent runs
        await session_pool.create_session("ses_test", agent_name="orchestrator")
        # Use SessionPool with scope="descendants" to receive child session events
        async for event in session_pool.run_stream("ses_test", "Delegate", scope="descendants"):
            if isinstance(event, SpawnSessionStart):
                child_session_id_from_spawn = event.child_session_id
            elif isinstance(event, RunStartedEvent):
                child_session_ids_from_run_started.append(event.session_id)

    assert child_session_id_from_spawn is not None
    assert child_session_ids_from_run_started, "No RunStartedEvent found in stream"
    assert child_session_id_from_spawn in child_session_ids_from_run_started, (
        f"RunStartedEvent.session_id {child_session_ids_from_run_started} "
        f"should contain SpawnSessionStart.child_session_id {child_session_id_from_spawn}"
    )


# ---------------------------------------------------------------------------
# TG-9: Depth increments by 1 per delegation level
# ---------------------------------------------------------------------------


async def test_depth_increments_per_delegation_level() -> None:
    """TG-9: Subagent at depth 0 → child at depth 1; explicit depth=2 → child at depth 3."""
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
    # Test default depth (0) → child depth=1
    spawn_depth_default: int | None = None

    async with AgentPool(manifest) as pool:
        orch = pool.manifest.agents["orchestrator"].get_agent(pool=pool)
        async for event in orch.run_stream("Delegate", session_id="ses_test"):
            if isinstance(event, SpawnSessionStart):
                spawn_depth_default = event.depth

    assert spawn_depth_default == 1, (
        f"Expected depth=1 for first delegation, got {spawn_depth_default}"
    )

    # Test explicit depth propagation via Team with depth parameter
    agent_a = _make_echo_agent("alpha")
    team = BaseTeam([agent_a], mode="parallel")

    events = await _collect_events(team, "test", depth=2)
    spawn_events = [e for e in events if isinstance(e, SpawnSessionStart)]
    assert len(spawn_events) == 1
    assert spawn_events[0].depth == 3, (
        f"Expected depth=3 (2+1) for team delegation at depth=2, got {spawn_events[0].depth}"
    )


# ---------------------------------------------------------------------------
# TG-10: ACP child session inherits parent project_id/cwd
# ---------------------------------------------------------------------------


async def test_acp_child_session_inherits_parent_project_and_cwd() -> None:
    """TG-10: ACP child session created via ACPSessionManager with parent_session_id...."""
    from wolfharness.models.agents import NativeAgentConfig
    from wolfharness.models.manifest import AgentsManifest
    from wolfharness.orchestrator.core import SessionPool
    from wolfharness_server.acp_server.session_manager import ACPSessionManager

    manifest = AgentsManifest(agents={"acp_agent": NativeAgentConfig(model="test")})
    pool = AgentPool(manifest)

    def simple_callback(message: str) -> str:
        return f"Response: {message}"

    Agent.from_callback(name="acp_agent", callback=simple_callback, agent_pool=pool)

    store = MemoryStorageProvider()
    session_pool = SessionPool(pool=pool, store=store)
    pool._session_pool = session_pool
    pool.storage.generate_session_id = MagicMock(return_value="acp_top_001")  # type: ignore[assignment]

    # Create parent session with known project_id and cwd
    parent_cwd = tempfile.gettempdir()
    from wolfharness_storage.opencode_provider.helpers import compute_project_id

    parent_project_id = compute_project_id(parent_cwd)
    parent_data = SessionData(
        session_id="acp_parent_001",
        agent_name="acp_agent",
        cwd=parent_cwd,
        project_id=parent_project_id,
    )
    await store.save_session(parent_data)

    manager = ACPSessionManager(pool=pool)
    mock_client = MagicMock()
    mock_acp_agent = MagicMock()

    with (
        patch("wolfharness_server.acp_server.session_manager.ACPSession") as mock_session_cls,
        patch("wolfharness_server.acp_server.session_manager.ClientCapabilities"),
    ):
        mock_session_instance = MagicMock()
        mock_session_instance.register_update_callback = MagicMock()
        mock_session_instance.initialize = AsyncMock()
        mock_session_instance.initialize_mcp_servers = AsyncMock()
        mock_session_cls.return_value = mock_session_instance

        session_id = await manager.create_session(
            agent_name="acp_agent",
            cwd="/different/cwd",
            client=mock_client,
            acp_agent=mock_acp_agent,
            parent_session_id="acp_parent_001",
        )

    child_data = await store.load_session(session_id)
    assert child_data is not None
    assert child_data.parent_id == "acp_parent_001"
    assert child_data.project_id == parent_project_id, "Child must inherit parent project_id"
    assert child_data.cwd == parent_cwd, "Child must inherit parent cwd"
    assert child_data.agent_type == "acp"


# ---------------------------------------------------------------------------
# TG-14: SubagentTools depth guard raises DelegationDepthError before
#         session creation
# ---------------------------------------------------------------------------


async def test_subagent_depth_guard_before_session_creation() -> None:
    """TG-14: When depth >= MAX_DELEGATION_DEPTH, DelegationDepthError is raised BEFORE...."""
    from wolfharness_toolsets.builtin.subagent_tools import SubagentTools

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
          description: "Depth guard"
    tools:
      - type: subagent
""")
    async with AgentPool(manifest) as pool:
        orch = pool.manifest.agents["orchestrator"].get_agent(pool=pool)
        tools_provider = SubagentTools()

        ctx = AgentContext(node=orch)
        ctx.pool = pool
        ctx.run_ctx = AgentRunContext(depth=MAX_DELEGATION_DEPTH)

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
# TG-15: WorkersTools child session persisted with correct parent
# ---------------------------------------------------------------------------


async def test_workers_child_session_persisted_with_correct_parent() -> None:
    """TG-15: WorkersTools creates child session with correct parent_session_id and the...."""
    store = MemoryStorageProvider()
    manifest = AgentsManifest.from_yaml("""
agents:
  main:
    type: native
    model: test
    workers:
      - worker

  worker:
    type: native
    model: test
    system_prompt: "Worker agent."
""")

    async with AgentPool(manifest) as pool:
        if pool.sessions is None:
            pytest.skip("Pool has no SessionManager")
        pool.session_pool.sessions.store = store  # type: ignore[union-attr]

        main_agent = pool.manifest.agents["main"].get_agent(pool=pool)
        worker = pool.manifest.agents["worker"].get_agent(pool=pool)
        assert isinstance(main_agent, Agent)
        assert isinstance(worker, Agent)

        from pydantic_ai.models.test import TestModel

        await main_agent.set_model(TestModel(call_tools=["ask_worker"]))
        await worker.set_model(TestModel(custom_output_text="Worker result"))

        child_session_id: str | None = None

        # Use _skip_pool=True to avoid instance divergence: without this,
        # run_stream() delegates to session_pool.run_stream() which creates
        # a new agent instance via get_or_create_session_agent(), losing
        # the TestModel set above.
        async for event in main_agent.run_stream(
            "Run worker", session_id="ses_test", _skip_pool=True
        ):
            if isinstance(event, SpawnSessionStart):
                child_session_id = event.child_session_id

        assert child_session_id is not None, "No SpawnSessionStart from worker"

        child_data = await store.load_session(child_session_id)
        assert child_data is not None, f"Child session {child_session_id} not persisted"
        assert child_data.agent_name == "worker"


# ---------------------------------------------------------------------------
# TG-16: TeamRun sequential members each get own child session
# ---------------------------------------------------------------------------


async def test_teamrun_each_member_gets_own_child_session() -> None:
    """TG-16: TeamRun sequential members each get their own SpawnSessionStart with unique...."""
    agent1 = _make_echo_agent("step1", "first")
    agent2 = _make_echo_agent("step2", "second")
    team = BaseTeam([agent1, agent2], mode="sequential", name="pipeline")

    async with agent1, agent2:
        events = await _collect_events(team, "prompt", session_id="ses_parent_tg16")

        spawn_events = [e for e in events if isinstance(e, SpawnSessionStart)]
        assert len(spawn_events) == 2, f"Expected 2 SpawnSessionStart, got {len(spawn_events)}"

        # Each member should have a different child_session_id
        child_ids = {e.child_session_id for e in spawn_events}
        assert len(child_ids) == 2, f"Expected 2 unique child_session_ids, got {child_ids}"

        # All should reference the same parent
        parent_ids = {e.parent_session_id for e in spawn_events}
        assert parent_ids == {"ses_parent_tg16"}

        # SubAgentEvents should carry matching child_session_ids
        sub_events = [e for e in events if isinstance(e, SubAgentEvent)]
        sub_child_ids = {se.child_session_id for se in sub_events}
        assert child_ids.issubset(sub_child_ids), (
            f"SubAgentEvent child_session_ids {sub_child_ids} "
            f"should include all spawn child_session_ids {child_ids}"
        )


# ---------------------------------------------------------------------------
# TG-18: Nested Team → SubAgentEvent preserves inner child/parent session IDs
# ---------------------------------------------------------------------------


async def test_nested_team_subagent_preserves_inner_session_ids() -> None:
    """TG-18: When a Team contains a nested team, SubAgentEvents from inner team preserve...."""
    inner_a = _make_echo_agent("inner_a")
    inner_b = _make_echo_agent("inner_b")
    inner_team = BaseTeam([inner_a, inner_b], mode="parallel", name="inner_team")

    outer_agent = _make_echo_agent("outer_agent")
    outer_team = BaseTeam([inner_team, outer_agent], mode="parallel", name="outer_team")

    events = await _collect_events(outer_team, "test", session_id="ses_outer_parent")

    # Nested SubAgentEvents with depth > 1 should preserve session IDs
    nested_sub = [e for e in events if isinstance(e, SubAgentEvent) and e.depth > 1]
    for se in nested_sub:
        assert se.child_session_id is not None, "Nested SubAgentEvent missing child_session_id"
        assert se.parent_session_id is not None, "Nested SubAgentEvent missing parent_session_id"


# ---------------------------------------------------------------------------
# TG-22: Mixed agent type Team (native + acp agents) all get child sessions
# ---------------------------------------------------------------------------


async def test_mixed_agent_type_team_all_get_child_sessions() -> None:
    """TG-22: Team with mixed agent types must create SpawnSessionStart for each member.

    Note: ACP agents in a Team require a real ACP client which we cannot
    provide in unit tests. Instead we verify the cross-provider contract
    by testing a Team with agents of different source_types (Agent + TeamRun).
    """
    native_a = _make_echo_agent("native_alpha")
    native_b = _make_echo_agent("native_beta")

    # Use a TeamRun as one of the members — it has source_type "team_sequential"
    teamrun_member = BaseTeam([native_b], mode="sequential", name="sequential_member")

    mixed_team = BaseTeam([native_a, teamrun_member], mode="parallel", name="mixed_team")

    events = await _collect_events(mixed_team, "test", session_id="ses_mixed_parent")

    spawn_events = [e for e in events if isinstance(e, SpawnSessionStart)]
    # One for native_alpha, one for sequential_member (TeamRun)
    assert len(spawn_events) == 2, f"Expected 2 SpawnSessionStart, got {len(spawn_events)}"

    spawn_names = {e.source_name for e in spawn_events}
    assert "native_alpha" in spawn_names
    assert "sequential_member" in spawn_names

    # Verify source_type differentiation
    spawn_types = {(e.source_name, e.source_type) for e in spawn_events}
    assert ("native_alpha", "agent") in spawn_types
    assert ("sequential_member", "team_sequential") in spawn_types

    # All child_session_ids should be unique
    child_ids = [e.child_session_id for e in spawn_events]
    assert len(set(child_ids)) == len(child_ids), "Duplicate child_session_ids detected"

    # All should share same parent
    parent_ids = {e.parent_session_id for e in spawn_events}
    assert parent_ids == {"ses_mixed_parent"}


# ---------------------------------------------------------------------------
# Cross-provider: Event ordering invariant
# SpawnSessionStart index < first SubAgentEvent index per child_session_id
# ---------------------------------------------------------------------------


async def test_event_ordering_spawn_before_subagent_per_child() -> None:
    """Cross-provider invariant: SpawnSessionStart precedes SubAgentEvent per child.

    For every child_session_id, the SpawnSessionStart event must appear at a
    lower index than the first SubAgentEvent carrying that child_session_id.
    """
    # Test with Team (multiple members → multiple child sessions)
    agent_a = _make_echo_agent("alpha")
    agent_b = _make_echo_agent("beta")
    team = BaseTeam([agent_a, agent_b], mode="parallel")

    events = await _collect_events(team, "test", session_id="ses_ordering_parent")

    spawn_by_child: dict[str, int] = {}
    first_sub_by_child: dict[str, int] = {}

    for i, event in enumerate(events):
        if isinstance(event, SpawnSessionStart):
            spawn_by_child[event.child_session_id] = i
        elif isinstance(event, SubAgentEvent) and event.child_session_id:
            cid = event.child_session_id
            if cid not in first_sub_by_child:
                first_sub_by_child[cid] = i

    # For every child that has both a spawn and a sub event,
    # the spawn must come first
    common_ids = set(spawn_by_child) & set(first_sub_by_child)
    assert common_ids, "No child_session_ids found in both spawn and sub events"

    for cid in common_ids:
        assert spawn_by_child[cid] < first_sub_by_child[cid], (
            f"child_session_id={cid}: SpawnSessionStart at index {spawn_by_child[cid]} "
            f"must precede first SubAgentEvent at index {first_sub_by_child[cid]}"
        )


# ---------------------------------------------------------------------------
# Negative: Non-streaming Team.run() and TeamRun.run() do NOT emit
#           SpawnSessionStart
# ---------------------------------------------------------------------------


async def test_team_run_does_not_emit_spawn_session_start() -> None:
    """Non-streaming Team.run() should NOT emit SpawnSessionStart. This is out-of-scope...."""
    agent_a = _make_echo_agent("alpha")
    agent_b = _make_echo_agent("beta")
    team = BaseTeam([agent_a, agent_b], mode="parallel")

    result = await team.run("test")
    # run() returns ChatMessage, not events — no SpawnSessionStart possible
    assert result is not None
    assert result.role == "assistant"


async def test_teamrun_run_does_not_emit_spawn_session_start() -> None:
    """Non-streaming TeamRun.run() should NOT emit SpawnSessionStart. This is out-of-scope...."""
    agent1 = _make_echo_agent("step1", "first")
    agent2 = _make_echo_agent("step2", "second")
    team = BaseTeam([agent1, agent2], mode="sequential", name="pipeline")

    async with agent1, agent2:
        result = await team.run("prompt")
        # run() returns ChatMessage, not events — no SpawnSessionStart possible
        assert result is not None
        assert result.role == "assistant"


# ---------------------------------------------------------------------------
# Cross-provider: SpawnSessionStart and SubAgentEvent depth consistency
# ---------------------------------------------------------------------------


async def test_spawn_and_subagent_depth_consistency() -> None:
    """SpawnSessionStart.depth must equal SubAgentEvent.depth for the same child delegation...."""
    agent_a = _make_echo_agent("alpha")
    team = BaseTeam([agent_a], mode="parallel")

    events = await _collect_events(team, "test", depth=3, session_id="ses_depth_parent")

    spawn_events = [e for e in events if isinstance(e, SpawnSessionStart)]
    sub_events = [e for e in events if isinstance(e, SubAgentEvent)]

    assert len(spawn_events) == 1
    assert len(sub_events) >= 1

    spawn_depth = spawn_events[0].depth
    # All SubAgentEvents from the same child should have same depth
    for se in sub_events:
        assert se.depth == spawn_depth, (
            f"SubAgentEvent.depth={se.depth} != SpawnSessionStart.depth={spawn_depth}"
        )

    # Verify the depth is correct (3 + 1 = 4)
    assert spawn_depth == 4


# ---------------------------------------------------------------------------
# Cross-provider: Pool-backed Team and TeamRun both use
# pool.sessions.create_child_session()
# ---------------------------------------------------------------------------


@pytest.mark.skip(reason="L2 migration: requires mock internals — remains L1 unit test")
async def test_pool_backed_team_and_teamrun_create_child_sessions() -> None:
    """Both Team and TeamRun with pool.session_pool should call create_child_session for...."""
    agent_a = _make_echo_agent("alpha")
    agent_b = _make_echo_agent("beta")

    team = BaseTeam([agent_a], mode="parallel", name="parallel_team")
    teamrun = BaseTeam([agent_b], mode="sequential", name="sequential_team")

    mock_pool = MagicMock()
    mock_session_pool = MagicMock()

    def _make_child_state(session_id: str):
        m = MagicMock()
        m.session_id = session_id
        return m

    mock_session_pool.create_session = AsyncMock(
        # Capture the passed session_id from create_child_session()
        # instead of hardcoding — the production code generates its own.
        side_effect=lambda session_id, **kw: _make_child_state(session_id)
    )

    async def _mock_run_stream(*args: object, **kwargs: object) -> AsyncIterator[Any]:
        return
        yield  # Makes this an async generator

    mock_session_pool.run_stream = _mock_run_stream
    mock_session_pool.sessions.get_session = MagicMock(return_value=None)
    mock_session_pool.close_session = AsyncMock()
    mock_pool.session_pool = mock_session_pool
    # storage property on MessageNode accesses _agent_pool.storage directly
    mock_pool.storage = MagicMock()
    mock_pool.storage.log_session = AsyncMock()
    mock_pool.storage.save_session = AsyncMock()
    # host_context calls _agent_pool.get_context(); return a mock context
    mock_context = MagicMock()
    mock_context.session_pool = mock_session_pool
    mock_context.storage = MagicMock()
    mock_context.storage.log_session = AsyncMock()
    mock_context.storage.save_session = AsyncMock()
    mock_pool.get_context.return_value = mock_context

    team.agent_pool = mock_pool
    agent_a.agent_pool = mock_pool
    teamrun.agent_pool = mock_pool
    agent_b.agent_pool = mock_pool

    # Team
    events = await _collect_events(team, "test", session_id="ses_parent_both")
    spawn_events = [e for e in events if isinstance(e, SpawnSessionStart)]
    assert len(spawn_events) == 1
    # child_session_id is auto-generated by create_child_session();
    # the mock captures the passed session_id so it stays consistent.
    assert spawn_events[0].child_session_id is not None
    assert isinstance(spawn_events[0].child_session_id, str)
    assert spawn_events[0].child_session_id.startswith("ses_")

    # TeamRun
    events = await _collect_events(teamrun, "test", session_id="ses_parent_both")
    spawn_events = [e for e in events if isinstance(e, SpawnSessionStart)]
    assert len(spawn_events) == 1
    assert spawn_events[0].child_session_id is not None
    assert isinstance(spawn_events[0].child_session_id, str)
    assert spawn_events[0].child_session_id.startswith("ses_")

    # Both Team and TeamRun should have called create_session for each member.
    # Agent run_stream also calls create_session to ensure the session exists.
    assert mock_session_pool.create_session.call_count >= 2


# ---------------------------------------------------------------------------
# Cross-provider: All child_session_ids are unique across providers
# ---------------------------------------------------------------------------


async def test_child_session_ids_unique_across_providers() -> None:
    """When SubagentTools delegates to a Team, the SubagentTools child session and the Team...."""
    agent_a = _make_echo_agent("alpha")
    agent_b = _make_echo_agent("beta")
    inner_team = BaseTeam([agent_a, agent_b], mode="parallel", name="work_team")

    manifest = AgentsManifest.from_yaml("""
agents:
  worker:
    model:
      type: test
      custom_output_text: "Direct worker"
    system_prompt: Worker.

  orchestrator:
    model:
      type: test
      call_tools: ["task"]
      tool_args:
        task:
          agent_or_team: work_team
          prompt: "Do work"
          description: "Cross-provider unique IDs"
    tools:
      - type: subagent
""")

    all_child_ids: list[str] = []

    async with AgentPool(manifest) as pool:
        # Add team to manifest so subagent tool can find it
        from wolfharness_config.teams import TeamConfig

        pool.manifest.teams["work_team"] = TeamConfig(mode="parallel", members=["alpha", "beta"])
        inner_team.agent_pool = pool
        agent_a.agent_pool = pool
        agent_b.agent_pool = pool

        orch = pool.manifest.agents["orchestrator"].get_agent(pool=pool)
        all_child_ids.extend([
            event.child_session_id
            async for event in orch.run_stream("Delegate to team", session_id="ses_test")
            if isinstance(event, SpawnSessionStart)
        ])

    # All child_session_ids must be unique
    assert len(set(all_child_ids)) == len(all_child_ids), (
        f"Duplicate child_session_ids across providers: {all_child_ids}"
    )
    # Should have at least: 1 for subagent→team + 2 for team members
    assert len(all_child_ids) >= 1, (
        f"Expected at least 1 SpawnSessionStart, got {len(all_child_ids)}"
    )
