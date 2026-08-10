"""Tests for Team/TeamRun.run_stream() session hierarchy and depth adaptation.

Consolidated from:
- test_team_run_stream_session.py (Team.run_stream session/depth tests)
- test_team_run_stream_depth.py (TeamRun.run_stream depth/session tests)


# TODO: L2 migration — test uses complex inline mock_pool + mock_session_pool
# patterns that require significant rework for real pool migration.
"""

from __future__ import annotations

import inspect
from typing import Any
from unittest.mock import ANY, AsyncMock, MagicMock

import pytest

from wolfharness import Agent
from wolfharness.agents.events import (
    SpawnSessionStart,
    StreamCompleteEvent,
    SubAgentEvent,
)
from wolfharness.agents.exceptions import MAX_DELEGATION_DEPTH, DelegationDepthError
from wolfharness.delegation.base_team import BaseTeam
from wolfharness.messaging import ChatMessage


pytestmark = pytest.mark.integration


pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning:wolfharness.agents.base_agent")


# ============================================================================
# Helpers
# ============================================================================


def _make_echo_agent(name: str, response: str = "hello") -> Agent[Any, str]:
    """Create an Agent that echoes a fixed response via function_to_model."""
    from functools import partial

    from wolfharness.utils.model_helpers import function_to_model

    async def _echo(_msg: str, *, _response: str = response) -> str:
        return _response

    model = function_to_model(partial(_echo, _response=response))
    return Agent(name=name, model=model)


async def _collect_events(team: BaseTeam[Any, Any], *args: Any, **kwargs: Any) -> list[Any]:
    """Collect all events from run_stream into a list."""
    return [event async for event in team.run_stream(*args, **kwargs)]


# ============================================================================
# Team.run_stream signature / depth guard
# ============================================================================


def test_team_run_stream_accepts_depth_param() -> None:
    """Team.run_stream() should accept depth parameter with default 0."""
    sig = inspect.signature(BaseTeam.run_stream)
    assert "depth" in sig.parameters
    assert sig.parameters["depth"].default == 0


async def test_team_run_stream_depth_guard() -> None:
    """Team.run_stream() should raise DelegationDepthError when depth exceeds maximum."""
    agent_a = Agent(name="a", model="test")
    agent_b = Agent(name="b", model="test")
    team = BaseTeam([agent_a, agent_b])

    with pytest.raises(DelegationDepthError) as exc_info:
        async for _ in team.run_stream("prompt", depth=MAX_DELEGATION_DEPTH):
            pass

    assert exc_info.value.current_depth == MAX_DELEGATION_DEPTH + 1


async def test_team_run_stream_depth_at_limit_ok() -> None:
    """Team.run_stream() should NOT raise at depth = MAX - 1."""
    from wolfharness.utils.model_helpers import function_to_model

    async def echo(msg: str) -> str:
        return msg

    model = function_to_model(echo)
    agent_a = Agent(name="a", model=model)
    team = BaseTeam([agent_a])

    events = [event async for event in team.run_stream("hi", depth=MAX_DELEGATION_DEPTH - 1)]
    assert len(events) > 0


# ============================================================================
# Team.run_stream: SpawnSessionStart emission
# ============================================================================


async def test_team_run_stream_emits_spawn_session_start() -> None:
    """Each member should emit SpawnSessionStart before SubAgentEvent content."""
    from wolfharness.utils.model_helpers import function_to_model

    async def echo(msg: str) -> str:
        return msg

    model = function_to_model(echo)
    agent_a = Agent(name="alpha", model=model)
    agent_b = Agent(name="beta", model=model)
    team = BaseTeam([agent_a, agent_b])

    events = [event async for event in team.run_stream("test")]

    spawn_events = [e for e in events if isinstance(e, SpawnSessionStart)]
    sub_events = [e for e in events if isinstance(e, SubAgentEvent)]

    assert len(spawn_events) == 2
    spawn_names = {e.source_name for e in spawn_events}
    assert spawn_names == {"alpha", "beta"}

    for sp in spawn_events:
        assert sp.depth == 1
        assert sp.spawn_mechanism == "spawn"
        assert sp.source_type == "agent"

    assert len(sub_events) >= 2


async def test_spawn_session_start_precedes_subagent_for_member() -> None:
    """For each member, SpawnSessionStart should appear before any SubAgentEvent."""
    from wolfharness.utils.model_helpers import function_to_model

    async def echo(msg: str) -> str:
        return msg

    model = function_to_model(echo)
    agent_a = Agent(name="alpha", model=model)
    team = BaseTeam([agent_a])

    events = [event async for event in team.run_stream("test")]

    spawn_idx = None
    sub_idx = None
    for i, e in enumerate(events):
        if isinstance(e, SpawnSessionStart) and e.source_name == "alpha":
            spawn_idx = i
        if isinstance(e, SubAgentEvent) and e.source_name == "alpha" and sub_idx is None:
            sub_idx = i

    assert spawn_idx is not None
    assert sub_idx is not None
    assert spawn_idx < sub_idx


# ============================================================================
# Team.run_stream: child session IDs
# ============================================================================


async def test_subagent_event_preserves_session_ids() -> None:
    """SubAgentEvent should carry child_session_id and parent_session_id."""
    from wolfharness.utils.model_helpers import function_to_model

    async def echo(msg: str) -> str:
        return msg

    model = function_to_model(echo)
    agent_a = Agent(name="alpha", model=model)
    team = BaseTeam([agent_a])

    events = [
        event async for event in team.run_stream("test", session_id="parent_ses_123", depth=2)
    ]

    sub_events = [e for e in events if isinstance(e, SubAgentEvent)]
    assert len(sub_events) >= 1

    for se in sub_events:
        assert se.child_session_id is not None
        assert se.child_session_id.startswith("ses_")
        assert se.parent_session_id == "parent_ses_123"
        assert se.depth == 3


async def test_spawn_session_start_carries_session_ids() -> None:
    """SpawnSessionStart should carry child_session_id and parent_session_id."""
    from wolfharness.utils.model_helpers import function_to_model

    async def echo(msg: str) -> str:
        return msg

    model = function_to_model(echo)
    agent_a = Agent(name="alpha", model=model)
    team = BaseTeam([agent_a])

    events = [event async for event in team.run_stream("test", session_id="ses_parent_abc")]

    spawn_events = [e for e in events if isinstance(e, SpawnSessionStart)]
    assert len(spawn_events) == 1
    sp = spawn_events[0]
    assert sp.child_session_id.startswith("ses_")
    assert sp.parent_session_id == "ses_parent_abc"


async def test_out_of_pool_team_generates_session_ids() -> None:
    """Team without pool should generate session IDs and not crash."""
    from wolfharness.utils.model_helpers import function_to_model

    async def echo(msg: str) -> str:
        return msg

    model = function_to_model(echo)
    agent_a = Agent(name="alpha", model=model)
    agent_b = Agent(name="beta", model=model)
    team = BaseTeam([agent_a, agent_b])

    events = [event async for event in team.run_stream("hello")]

    spawn_events = [e for e in events if isinstance(e, SpawnSessionStart)]
    sub_events = [e for e in events if isinstance(e, SubAgentEvent)]

    assert len(spawn_events) == 2
    assert len(sub_events) >= 2

    for sp in spawn_events:
        assert sp.child_session_id.startswith("ses_")
        assert sp.parent_session_id == ""

    for se in sub_events:
        assert se.child_session_id is not None
        assert se.child_session_id.startswith("ses_")


@pytest.mark.skip(reason="L2 migration: requires mock internals — remains L1 unit test")
async def test_pool_backed_team_creates_child_sessions() -> None:
    """Team with pool.sessions should call create_child_session for each member."""
    from wolfharness.utils.model_helpers import function_to_model

    async def echo(msg: str) -> str:
        return msg

    model = function_to_model(echo)
    agent_a = Agent(name="alpha", model=model)
    agent_b = Agent(name="beta", model=model)
    team = BaseTeam([agent_a, agent_b])

    mock_pool = AsyncMock()
    mock_sessions = AsyncMock()
    mock_sessions.create_session = AsyncMock(
        side_effect=[
            MagicMock(session_id="ses_child_alpha"),
            MagicMock(session_id="ses_child_beta"),
        ]
    )
    # _resolve_scoped_team_nodes calls sessions.get_or_create_session_agent
    # which must return the original agent so child_session_ids keys match.
    mock_sessions.sessions = AsyncMock()
    mock_sessions.sessions.get_or_create_session_agent = AsyncMock(side_effect=[agent_a, agent_b])
    mock_pool.session_pool = mock_sessions
    # Provide manifest with agents dict so _resolve_scoped_team_nodes
    # can check pool_agents for scoped session creation.
    from types import SimpleNamespace

    mock_pool.manifest = SimpleNamespace(
        agents={"alpha": None, "beta": None},
        teams={},
    )
    # host_context calls _agent_pool.get_context(); configure it to return
    # a context with the session_pool so team code can resolve sessions.
    # Use MagicMock (not AsyncMock) for get_context since it's synchronous.
    mock_pool.get_context = MagicMock(
        return_value=MagicMock(
            session_pool=mock_sessions,
            manifest=mock_pool.manifest,
            storage=AsyncMock(),
        )
    )
    team.agent_pool = mock_pool

    events = [event async for event in team.run_stream("test", session_id="ses_parent")]

    assert mock_sessions.create_session.call_count == 2

    spawn_events = [e for e in events if isinstance(e, SpawnSessionStart)]
    spawn_sids = {e.child_session_id for e in spawn_events}
    assert spawn_sids == {"ses_child_alpha", "ses_child_beta"}

    sub_events = [e for e in events if isinstance(e, SubAgentEvent)]
    for se in sub_events:
        assert se.child_session_id in {"ses_child_alpha", "ses_child_beta"}
        assert se.parent_session_id == "ses_parent"


async def test_team_kwargs_session_id_depth_popped() -> None:
    """Passing session_id/depth in kwargs should not cause duplicate keyword errors."""
    from wolfharness.utils.model_helpers import function_to_model

    async def echo(msg: str) -> str:
        return msg

    model = function_to_model(echo)
    agent_a = Agent(name="alpha", model=model)
    team = BaseTeam([agent_a])

    events = [
        event
        async for event in team.run_stream(
            "test",
            session_id="ses_from_kwargs",
            depth=5,
        )
    ]

    spawn_events = [e for e in events if isinstance(e, SpawnSessionStart)]
    sub_events = [e for e in events if isinstance(e, SubAgentEvent)]
    assert len(spawn_events) >= 1
    assert len(sub_events) >= 1


async def test_team_run_unchanged() -> None:
    """Team.run() should not be affected by run_stream() changes."""
    from wolfharness.utils.model_helpers import function_to_model

    async def echo(msg: str) -> str:
        return msg

    model = function_to_model(echo)
    agent_a = Agent(name="alpha", model=model)
    agent_b = Agent(name="beta", model=model)
    team = BaseTeam([agent_a, agent_b])

    result = await team.run("test")
    assert result is not None
    assert result.role == "assistant"


async def test_nested_subagent_event_session_ids_preserved() -> None:
    """Nested SubAgentEvent IDs should be preserved when a member is itself a Team."""
    from wolfharness.utils.model_helpers import function_to_model

    async def echo(msg: str) -> str:
        return msg

    model = function_to_model(echo)
    inner_a = Agent(name="inner_a", model=model)
    inner_b = Agent(name="inner_b", model=model)
    inner_team = BaseTeam([inner_a, inner_b], name="inner_team")

    outer_agent = Agent(name="outer_agent", model=model)
    outer_team = BaseTeam([inner_team, outer_agent], name="outer_team")

    events = [event async for event in outer_team.run_stream("test")]

    nested_sub = [e for e in events if isinstance(e, SubAgentEvent) and e.depth > 1]
    for se in nested_sub:
        assert se.child_session_id is not None
        assert se.parent_session_id is not None


# ============================================================================
# TeamRun.run_stream: depth parameter
# ============================================================================


async def test_teamrun_run_stream_accepts_depth_without_type_error() -> None:
    """TeamRun.run_stream(..., depth=1, require_all=False) must not raise TypeError."""
    agent1 = _make_echo_agent("a1", "first")
    agent2 = _make_echo_agent("a2", "second")
    team = BaseTeam([agent1, agent2], mode="sequential", name="seq")

    async with agent1, agent2:
        events = await _collect_events(team, "prompt", depth=1, require_all=False)
        assert len(events) > 0


async def test_teamrun_run_stream_default_depth_is_zero() -> None:
    """Without explicit depth, default is 0 and child_depth should be 1."""
    agent1 = _make_echo_agent("a1", "first")
    team = BaseTeam([agent1], mode="sequential", name="seq")

    async with agent1:
        events = await _collect_events(team, "prompt")
        sub_events = [e for e in events if isinstance(e, SubAgentEvent)]
        for se in sub_events:
            assert se.depth == 1


async def test_teamrun_run_stream_depth_propagates_to_sub_events() -> None:
    """Explicit depth=2 should produce SubAgentEvent with depth=3."""
    agent1 = _make_echo_agent("a1", "result")
    team = BaseTeam([agent1], mode="sequential", name="seq")

    async with agent1:
        events = await _collect_events(team, "prompt", depth=2)
        sub_events = [e for e in events if isinstance(e, SubAgentEvent)]
        for se in sub_events:
            assert se.depth == 3


# ============================================================================
# TeamRun.run_stream: child sessions
# ============================================================================


async def test_teamrun_each_member_gets_own_child_session() -> None:
    """Each team member should get its own SpawnSessionStart + SubAgentEvent."""
    agent1 = _make_echo_agent("a1", "first")
    agent2 = _make_echo_agent("a2", "second")
    team = BaseTeam([agent1, agent2], mode="sequential", name="seq")

    async with agent1, agent2:
        events = await _collect_events(team, "prompt", session_id="parent-123")

        spawn_events = [e for e in events if isinstance(e, SpawnSessionStart)]
        assert len(spawn_events) == 2

        child_ids = {e.child_session_id for e in spawn_events}
        assert len(child_ids) == 2

        for se in spawn_events:
            assert se.parent_session_id == "parent-123"
            assert se.source_name in {"a1", "a2"}


async def test_teamrun_sub_events_carry_child_session_ids() -> None:
    """SubAgentEvent wrappers should carry child_session_id and parent_session_id."""
    agent1 = _make_echo_agent("a1", "first")
    team = BaseTeam([agent1], mode="sequential", name="seq")

    async with agent1:
        events = await _collect_events(team, "prompt", session_id="parent-456")
        sub_events = [e for e in events if isinstance(e, SubAgentEvent)]
        assert len(sub_events) > 0

        for se in sub_events:
            assert se.child_session_id is not None
            assert se.parent_session_id == "parent-456"


async def test_teamrun_spawn_session_start_fields() -> None:
    """SpawnSessionStart events should have correct fields."""
    agent1 = _make_echo_agent("a1", "result")
    team = BaseTeam([agent1], mode="sequential", name="seq")

    async with agent1:
        events = await _collect_events(team, "prompt", session_id="parent-789")
        spawn_events = [e for e in events if isinstance(e, SpawnSessionStart)]
        assert len(spawn_events) == 1

        se = spawn_events[0]
        assert se.source_name == "a1"
        assert se.source_type == "agent"
        assert se.spawn_mechanism == "spawn"
        assert se.parent_session_id == "parent-789"
        assert se.depth == 1


async def test_teamrun_child_session_fallback_without_pool() -> None:
    """Without a pool, child sessions should use generate_session_id() as fallback."""
    agent1 = _make_echo_agent("a1", "result")
    team = BaseTeam([agent1], mode="sequential", name="seq")

    async with agent1:
        events = await _collect_events(team, "prompt", session_id="parent-abc")
        spawn_events = [e for e in events if isinstance(e, SpawnSessionStart)]
        assert len(spawn_events) == 1
        assert spawn_events[0].child_session_id.startswith("ses_")


@pytest.mark.skip(reason="L2 migration: requires mock internals — remains L1 unit test")
async def test_teamrun_child_session_uses_pool_sessions() -> None:
    """With a pool, child sessions should be created via pool.sessions.create_child_session()."""
    agent1 = _make_echo_agent("a1", "result")
    team = BaseTeam([agent1], mode="sequential", name="seq")

    mock_pool = MagicMock()
    mock_sessions = AsyncMock()
    mock_sessions.create_session = AsyncMock(return_value=MagicMock(session_id="child-via-pool"))
    # _resolve_scoped_team_nodes calls sessions.get_or_create_session_agent
    # which must return the original agent so child_session_ids keys match.
    mock_sessions.sessions = AsyncMock()
    mock_sessions.sessions.get_or_create_session_agent = AsyncMock(return_value=agent1)
    mock_pool.session_pool = mock_sessions
    # Provide manifest with agents dict so _resolve_scoped_team_nodes
    # can check pool_agents for scoped session creation.
    from types import SimpleNamespace

    mock_pool.manifest = SimpleNamespace(
        agents={"a1": None},
        teams={},
    )
    # host_context calls _agent_pool.get_context(); configure it to return
    # a context with the session_pool so team code can resolve sessions.
    mock_pool.get_context.return_value = MagicMock(
        session_pool=mock_sessions,
        manifest=mock_pool.manifest,
        storage=AsyncMock(),
    )
    team.agent_pool = mock_pool

    async with agent1:
        events = await _collect_events(team, "prompt", session_id="parent-via-pool")
        spawn_events = [e for e in events if isinstance(e, SpawnSessionStart)]
        assert len(spawn_events) == 1
        assert spawn_events[0].child_session_id == "child-via-pool"

        mock_sessions.create_session.assert_called_once_with(
            session_id=ANY,
            parent_session_id="parent-via-pool",
            agent_name="a1",
            agent_type="agent",
            generate_title=False,
        )


# ============================================================================
# TeamRun.run_stream: sequential handoff
# ============================================================================


async def test_teamrun_sequential_handoff_uses_stream_complete_content() -> None:
    """The second agent should receive the first agent's StreamComplete content."""
    agent1 = _make_echo_agent("a1", "first output")
    agent2 = _make_echo_agent("a2", "second output")
    team = BaseTeam([agent1, agent2], mode="sequential", name="seq")

    received_prompts: list[tuple[str, ...]] = []
    original_run_stream = agent2.run_stream

    async def _capturing_run_stream(*prompts: Any, **kwargs: Any) -> Any:
        received_prompts.append(prompts)
        async for event in original_run_stream(*prompts, **kwargs):
            yield event

    agent2.run_stream = _capturing_run_stream  # type: ignore[assignment]

    async with agent1, agent2:
        await _collect_events(team, "initial prompt", session_id="parent-handoff")

    assert len(received_prompts) == 1
    assert received_prompts[0] == ("first output",)


# ============================================================================
# TeamRun.run_stream: depth guard
# ============================================================================


async def test_teamrun_depth_guard_raises() -> None:
    """Exceeding MAX_DELEGATION_DEPTH should raise DelegationDepthError."""
    agent1 = _make_echo_agent("a1", "result")
    team = BaseTeam([agent1], mode="sequential", name="seq")

    async with agent1:
        with pytest.raises(DelegationDepthError):
            async for _ in team.run_stream("prompt", depth=MAX_DELEGATION_DEPTH):
                pass


async def test_teamrun_depth_guard_at_boundary() -> None:
    """Depth = MAX - 1 should still work."""
    agent1 = _make_echo_agent("a1", "result")
    team = BaseTeam([agent1], mode="sequential", name="seq")

    async with agent1:
        events = await _collect_events(team, "prompt", depth=MAX_DELEGATION_DEPTH - 1)
        sub_events = [e for e in events if isinstance(e, SubAgentEvent)]
        assert len(sub_events) > 0
        assert sub_events[0].depth == MAX_DELEGATION_DEPTH


# ============================================================================
# TeamRun.run_stream: nested SubAgentEvent depth
# ============================================================================


async def test_teamrun_nested_subagent_depth_incremented() -> None:
    """When a member yields a SubAgentEvent, depth should be incremented by 1."""
    agent1 = _make_echo_agent("a1", "result")
    inner_complete = StreamCompleteEvent(
        message=ChatMessage(role="assistant", content="inner result"),
    )
    inner_sub = SubAgentEvent(
        source_name="inner_agent",
        source_type="agent",
        event=inner_complete,
        depth=2,
        child_session_id="inner-child-123",
        parent_session_id="inner-parent-456",
    )

    original_run_stream = agent1.run_stream

    async def _nested_run_stream(*prompts: Any, **kwargs: Any) -> Any:
        async for event in original_run_stream(*prompts, **kwargs):
            yield event
        yield inner_sub

    agent1.run_stream = _nested_run_stream  # type: ignore[assignment]

    team = BaseTeam([agent1], mode="sequential", name="seq")
    async with agent1:
        events = await _collect_events(team, "prompt", depth=1, session_id="parent-nested")

        for e in events:
            if isinstance(e, SubAgentEvent) and e.source_name == "inner_agent":
                assert e.depth == 3
                assert e.child_session_id == "inner-child-123"
                assert e.parent_session_id == "inner-parent-456"


# ============================================================================
# TeamRun.run_stream: kwargs pop semantics
# ============================================================================


async def test_teamrun_session_id_popped_from_kwargs() -> None:
    """session_id in kwargs should be popped and not forwarded as duplicate."""
    agent1 = _make_echo_agent("a1", "result")
    team = BaseTeam([agent1], mode="sequential", name="seq")

    async with agent1:
        events = await _collect_events(team, "prompt", session_id="ses-123")
        assert len(events) > 0


async def test_teamrun_depth_popped_from_kwargs() -> None:
    """Depth in kwargs should be popped; explicit parameter takes precedence."""
    agent1 = _make_echo_agent("a1", "result")
    team = BaseTeam([agent1], mode="sequential", name="seq")

    async with agent1:
        events = await _collect_events(team, "prompt", depth=5, session_id="ses-depth")
        sub_events = [e for e in events if isinstance(e, SubAgentEvent)]
        for se in sub_events:
            assert se.depth == 6


# ============================================================================
# TeamRun.run_stream: require_all preserved
# ============================================================================


async def test_teamrun_require_all_still_propagates_errors() -> None:
    """require_all=True should still raise on member failure."""
    failing_agent = _make_echo_agent("fail", "nope")

    async def _failing_stream(*_prompts: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("Agent failed")
        yield

    failing_agent.run_stream = _failing_stream  # type: ignore[assignment]

    team = BaseTeam([failing_agent], mode="sequential", name="seq")
    async with failing_agent:
        with pytest.raises(ValueError, match="Chain broken"):
            await _collect_events(team, "prompt", require_all=True)


async def test_teamrun_require_all_false_continues_on_error() -> None:
    """require_all=False should continue when a member fails."""
    failing_agent = _make_echo_agent("fail", "nope")

    async def _failing_stream(*_prompts: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("Agent failed")
        yield

    failing_agent.run_stream = _failing_stream  # type: ignore[assignment]

    good_agent = _make_echo_agent("good", "I survived")
    team = BaseTeam([failing_agent, good_agent], mode="sequential", name="seq")

    async with failing_agent, good_agent:
        events = await _collect_events(team, "prompt", require_all=False)
        sub_events = [e for e in events if isinstance(e, SubAgentEvent) and e.source_name == "good"]
        assert len(sub_events) > 0
