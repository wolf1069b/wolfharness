"""Comprehensive tests for graph-based teams.

Tests cover parallel teams (Fork+Join), sequential teams (chained Steps),
mixed teams, error handling, streaming, signal emission, and backward
compatibility with legacy Team/TeamRun APIs.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import anyio
import pytest

from wolfharness import Agent
from wolfharness.agents.base_agent import _current_run_ctx_var
from wolfharness.agents.context import AgentRunContext
from wolfharness.agents.events import (
    SpawnSessionStart,
    StreamCompleteEvent,
    SubAgentEvent,
)
from wolfharness.delegation.base_team import BaseTeam
from wolfharness.delegation.graph_team import (
    _MemberOutput,
    _TeamGraphState,
    build_team_graph,
    run_team_graph,
)
from wolfharness.messaging import AgentResponse, ChatMessage, TeamResponse
from wolfharness.messaging.messagenode import MessageNode
from wolfharness.talk import Talk


pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_echo_agent(name: str, response: str = "hello") -> Agent[Any, str]:
    """Create an Agent that echoes a fixed response via TestModel."""
    from pydantic_ai.models.test import TestModel

    model = TestModel(custom_output_text=response)
    return Agent(name=name, model=model)


class _FakeSessionAgents:
    """Minimal session agent registry for scoped team tests."""

    def __init__(self) -> None:
        self.created_agents: dict[str, Agent[Any, str]] = {}
        self.states: dict[str, Any] = {}

    def add_state(
        self,
        session_id: str,
        *,
        agent_name: str = "agent",
        parent_session_id: str | None = None,
        **metadata: Any,
    ) -> Any:
        state = SimpleNamespace(
            session_id=session_id,
            agent_name=agent_name,
            parent_session_id=parent_session_id,
            metadata=metadata,
        )
        self.states[session_id] = state
        return state

    def get_session(self, session_id: str) -> Any | None:
        return self.states.get(session_id)

    def _state_to_data(self, state: Any) -> Any:
        return SimpleNamespace(
            session_id=state.session_id,
            agent_name=state.agent_name,
            parent_id=state.parent_session_id,
            metadata=state.metadata,
        )

    async def get_or_create_session_agent(
        self,
        session_id: str,
        agent_name: str | None = None,
        input_provider: Any | None = None,
    ) -> Agent[Any, str]:
        if session_id not in self.created_agents:
            name = agent_name or "agent"
            self.created_agents[session_id] = _make_echo_agent(name, f"child:{name}:{session_id}")
        return self.created_agents[session_id]


class _FakeSessionPool:
    """Minimal SessionPool facade used by `Team.execute` scoped mode."""

    def __init__(self) -> None:
        self.sessions = _FakeSessionAgents()
        self.created: list[dict[str, str]] = []
        self.closed: list[str] = []

    async def create_session(
        self,
        session_id: str,
        agent_name: str | None = None,
        parent_session_id: str | None = None,
        lifecycle_policy: str | None = None,
        **metadata: Any,
    ) -> Any:
        self.created.append({
            "session_id": session_id,
            "agent_name": agent_name or "",
            "parent_session_id": parent_session_id or "",
            "lifecycle_policy": lifecycle_policy or "",
            "team_name": str(metadata.get("team_name") or ""),
            "team_run_id": str(metadata.get("team_run_id") or ""),
            "generate_title": str(metadata.get("generate_title")),
        })
        return self.sessions.add_state(
            session_id,
            agent_name=agent_name or "",
            parent_session_id=parent_session_id,
            **metadata,
        )

    async def close_session(self, session_id: str) -> None:
        self.closed.append(session_id)


class _FakeStorage:
    """Storage facade that records scoped session persistence calls."""

    def __init__(self) -> None:
        self.saved: list[Any] = []
        self.deleted_sessions: list[str] = []
        self.deleted_messages: list[str] = []

    async def save_session(self, data: Any) -> None:
        self.saved.append(data)

    async def delete_session_messages(self, session_id: str) -> int:
        self.deleted_messages.append(session_id)
        return 0

    async def delete_session(self, session_id: str) -> bool:
        self.deleted_sessions.append(session_id)
        return True


class _FakeAgentPool:
    """Minimal AgentPool facade with team-scoped session support."""

    def __init__(self, agents: list[Agent[Any, str]]) -> None:
        from types import SimpleNamespace

        from wolfharness.mcp_server.manager import MCPManager

        self.session_pool = _FakeSessionPool()
        self.all_agents = {agent.name: agent for agent in agents}
        self.storage = _FakeStorage()
        self.mcp = MCPManager("fake-pool")
        # Provide a minimal manifest with agents and teams dicts for
        # _resolve_scoped_team_nodes which accesses pool.manifest.agents
        # and pool.manifest.teams.
        self.manifest = SimpleNamespace(
            agents={agent.name: agent for agent in agents},
            teams={},
        )

    def get_context(self) -> SimpleNamespace:
        """Return a HostContext-like object for host_context property."""
        from types import SimpleNamespace

        return SimpleNamespace(
            session_pool=self.session_pool,
            storage=self.storage,
            manifest=self.manifest,
            mcp=self.mcp,
        )


class FailingAgent(MessageNode[Any, Any]):
    """An agent that always raises an exception."""

    def __init__(self, name: str, exc_msg: str = "intentional failure") -> None:
        super().__init__(name=name)
        self.exc_msg = exc_msg

    async def run(self, *prompts: Any, **kwargs: Any) -> ChatMessage[Any]:
        msg = self.exc_msg
        raise RuntimeError(msg)

    async def get_stats(self) -> Any:
        return None

    def run_iter(self, *prompts: Any, **kwargs: Any) -> Any:
        pass

    def get_context(self, data: Any = None, input_provider: Any = None) -> Any:
        return None


class FlakyAgent(MessageNode[Any, Any]):
    """An agent that fails once and then succeeds."""

    def __init__(self, name: str) -> None:
        super().__init__(name=name)
        self.calls = 0

    async def run(self, *prompts: Any, **kwargs: Any) -> ChatMessage[Any]:
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("temporary failure")
        return ChatMessage(content="recovered", role="assistant", name=self.name)

    async def get_stats(self) -> Any:
        return None

    def run_iter(self, *prompts: Any, **kwargs: Any) -> Any:
        pass

    def get_context(self, data: Any = None, input_provider: Any = None) -> Any:
        return None


class InvalidAgent(MessageNode[Any, Any]):
    """An agent that raises a non-runtime failure."""

    def __init__(self, name: str) -> None:
        super().__init__(name=name)
        self.calls = 0

    async def run(self, *prompts: Any, **kwargs: Any) -> ChatMessage[Any]:
        self.calls += 1
        raise ValueError("invalid member configuration")

    async def get_stats(self) -> Any:
        return None

    def run_iter(self, *prompts: Any, **kwargs: Any) -> Any:
        pass

    def get_context(self, data: Any = None, input_provider: Any = None) -> Any:
        return None


class SlowAgent(MessageNode[Any, Any]):
    """An agent that sleeps longer than a configured team timeout."""

    def __init__(self, name: str, delay: float) -> None:
        super().__init__(name=name)
        self.delay = delay

    async def run(self, *prompts: Any, **kwargs: Any) -> ChatMessage[Any]:
        await anyio.sleep(self.delay)
        return ChatMessage(content="late", role="assistant", name=self.name)

    async def get_stats(self) -> Any:
        return None

    def run_iter(self, *prompts: Any, **kwargs: Any) -> Any:
        pass

    def get_context(self, data: Any = None, input_provider: Any = None) -> Any:
        return None


async def _collect_events(source: Any, *args: Any, **kwargs: Any) -> list[Any]:
    """Collect all events from run_stream into a list."""
    events: list[Any] = []
    events.extend([event async for event in source.run_stream(*args, **kwargs)])
    return events


# ---------------------------------------------------------------------------
# 1. Parallel team with 3 agents
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_parallel_team_with_three_agents() -> None:
    """Parallel Team runs 3 agents via Fork+Join and collects all responses."""
    agent_a = _make_echo_agent("alpha", "response_a")
    agent_b = _make_echo_agent("beta", "response_b")
    agent_c = _make_echo_agent("gamma", "response_c")

    team = BaseTeam([agent_a, agent_b, agent_c], name="parallel_three")

    result = await team.run("test prompt")

    assert result is not None
    assert result.role == "assistant"
    # Parallel team returns list of contents
    contents = result.content
    assert isinstance(contents, list)
    assert len(contents) == 3
    assert "response_a" in contents
    assert "response_b" in contents
    assert "response_c" in contents

    # Metadata should track all agent names
    assert "agent_names" in result.metadata
    agent_names = cast(list[str], result.metadata["agent_names"])
    assert set(agent_names) == {"alpha", "beta", "gamma"}


@pytest.mark.anyio
async def test_parallel_team_execute_returns_team_response() -> None:
    """Team.execute() returns a TeamResponse with timing and responses."""
    agent_a = _make_echo_agent("alpha", "A")
    agent_b = _make_echo_agent("beta", "B")

    team = BaseTeam([agent_a, agent_b], name="parallel_two")
    response = await team.execute("prompt")

    assert isinstance(response, TeamResponse)
    assert len(response) == 2

    names = {r.agent_name for r in response}
    assert names == {"alpha", "beta"}

    for r in response:
        assert r.message is not None
        assert r.timing is not None
        assert r.timing >= 0


@pytest.mark.anyio
async def test_parallel_team_execute_uses_scoped_child_sessions_by_default() -> None:
    """Team.execute creates child-session member agents inside SessionPool turns."""
    agent_a = _make_echo_agent("alpha", "shared_a")
    agent_b = _make_echo_agent("beta", "shared_b")
    pool = _FakeAgentPool([agent_a, agent_b])
    pool.session_pool.sessions.add_state("parent-session", agent_name="rebuttal_agent")
    team = BaseTeam([agent_a, agent_b], name="parallel_scoped", agent_pool=pool)  # type: ignore[arg-type]
    run_ctx = AgentRunContext(session_id="parent-session")

    token = _current_run_ctx_var.set(run_ctx)
    try:
        response = await team.execute("prompt")
    finally:
        _current_run_ctx_var.reset(token)

    assert response.child_session_ids.keys() == {"alpha", "beta"}
    assert {item["agent_name"] for item in pool.session_pool.created} == {"alpha", "beta"}
    assert {item["parent_session_id"] for item in pool.session_pool.created} == {"parent-session"}
    assert {item["lifecycle_policy"] for item in pool.session_pool.created} == {"cascade"}
    assert {item["team_name"] for item in pool.session_pool.created} == {"parallel_scoped"}
    assert {item["generate_title"] for item in pool.session_pool.created} == {"False"}
    assert set(pool.session_pool.closed) == set(response.child_session_ids.values())
    saved_ids = [item.session_id for item in pool.storage.saved]
    assert saved_ids[0] == "parent-session"
    assert set(saved_ids[1:]) == set(response.child_session_ids.values())
    child_saved = [item for item in pool.storage.saved if item.session_id != "parent-session"]
    assert all(item.metadata.get("generate_title") is False for item in child_saved)
    assert set(pool.storage.deleted_messages) == set(response.child_session_ids.values())
    assert set(pool.storage.deleted_sessions) == set(response.child_session_ids.values())

    contents = {str(item.message.content) for item in response if item.message is not None}
    for agent_name, session_id in response.child_session_ids.items():
        assert f"child:{agent_name}:{session_id}" in contents


@pytest.mark.anyio
async def test_teamrun_stream_uses_scoped_child_sessions_without_titles() -> None:
    """TeamRun streaming child sessions should not request generated titles."""
    agent_a = _make_echo_agent("alpha", "A")
    agent_b = _make_echo_agent("beta", "B")
    pool = _FakeAgentPool([agent_a, agent_b])
    team = BaseTeam(
        [agent_a, agent_b], mode="sequential", name="sequential_scoped", agent_pool=pool
    )  # type: ignore[arg-type]

    events = [
        event
        async for event in team.run_stream(
            "prompt",
            session_id="parent-session",
            parent_session_id="parent-session",
        )
    ]

    spawn_events = [event for event in events if isinstance(event, SpawnSessionStart)]
    assert len(spawn_events) == 2
    assert {item["agent_name"] for item in pool.session_pool.created} == {"alpha", "beta"}
    assert {item["parent_session_id"] for item in pool.session_pool.created} == {"parent-session"}
    assert {item["generate_title"] for item in pool.session_pool.created} == {"False"}


# ---------------------------------------------------------------------------
# 2. Sequential team with 3 agents
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_sequential_team_with_three_agents() -> None:
    """TeamRun chains 3 agents sequentially, passing output through pipeline."""
    agent_1 = _make_echo_agent("step1", "first")
    agent_2 = _make_echo_agent("step2", "second")
    agent_3 = _make_echo_agent("step3", "third")

    pipeline = BaseTeam([agent_1, agent_2, agent_3], mode="sequential", name="sequential_three")

    async with agent_1, agent_2, agent_3:
        result = await pipeline.run("start")

    assert result is not None
    assert result.role == "assistant"
    # TeamRun with result_mode="last" returns last agent's output
    assert result.content == "third"

    # Metadata should track execution order
    assert "execution_order" in result.metadata
    order = cast(list[str], result.metadata["execution_order"])
    assert order == ["step1", "step2", "step3"]


@pytest.mark.anyio
async def test_sequential_team_execute_iter_yields_in_order() -> None:
    """TeamRun.execute_iter yields AgentResponse and Talk in correct order."""
    agent_1 = _make_echo_agent("s1", "out1")
    agent_2 = _make_echo_agent("s2", "out2")

    pipeline = BaseTeam([agent_1, agent_2], mode="sequential", name="seq_two")

    async with agent_1, agent_2:
        items = [i async for i in pipeline.execute_iter("prompt")]

    # Should yield: AgentResponse(s1), Talk(s1->s2), AgentResponse(s2)
    agent_responses = [i for i in items if isinstance(i, AgentResponse)]
    talks = [i for i in items if isinstance(i, Talk)]

    assert len(agent_responses) == 2
    assert agent_responses[0].agent_name == "s1"
    assert agent_responses[1].agent_name == "s2"

    # One talk for the edge between the two agents
    assert len(talks) == 1
    first_talk = cast(Talk, talks[0])
    assert first_talk.source.name == "s1"
    targets = cast(list[MessageNode[Any, Any]], first_talk.targets)
    assert len(targets) == 1
    assert targets[0].name == "s2"


# ---------------------------------------------------------------------------
# 3. Mixed team: sequential containing parallel sub-team
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_mixed_team_sequential_contains_parallel() -> None:
    """TeamRun containing a Team executes parallel sub-team then next agent."""
    agent_a = _make_echo_agent("a", "A")
    agent_b = _make_echo_agent("b", "B")
    agent_c = _make_echo_agent("c", "C")

    parallel_sub = BaseTeam([agent_a, agent_b], name="parallel_sub")
    mixed = BaseTeam([parallel_sub, agent_c], mode="sequential", name="mixed_seq_par")

    async with agent_a, agent_b, agent_c:
        result = await mixed.run("start")

    assert result is not None
    assert result.content == "C"

    # execution_order should include the parallel sub-team and final agent
    assert "execution_order" in result.metadata
    order = cast(list[str], result.metadata["execution_order"])
    assert "parallel_sub" in order
    assert "c" in order


@pytest.mark.anyio
async def test_mixed_team_streaming() -> None:
    """Mixed team streaming yields SubAgentEvents from parallel sub-team and sequential agent."""
    agent_a = _make_echo_agent("a", "A")
    agent_b = _make_echo_agent("b", "B")
    agent_c = _make_echo_agent("c", "C")

    parallel_sub = BaseTeam([agent_a, agent_b], name="parallel_sub")
    mixed = BaseTeam([parallel_sub, agent_c], mode="sequential", name="mixed_stream")

    async with agent_a, agent_b, agent_c:
        events = await _collect_events(mixed, "start", session_id="ses_mixed")

    # Should have SpawnSessionStart for each member
    spawns = [e for e in events if isinstance(e, SpawnSessionStart)]
    assert len(spawns) == 2, f"Expected 2 spawns, got {len(spawns)}"

    spawn_names = {s.source_name for s in spawns}
    assert "parallel_sub" in spawn_names
    assert "c" in spawn_names


# ---------------------------------------------------------------------------
# 4. Error handling: one agent fails in parallel team
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_parallel_team_one_agent_fails() -> None:
    """Parallel team with one failing agent collects exceptions for others."""
    agent_ok = _make_echo_agent("ok_agent", "success")
    agent_fail = FailingAgent("fail_agent", "boom")

    team = BaseTeam([agent_ok, agent_fail], name="partial_fail")
    result = await team.run("test")

    # Should still return a result (partial success)
    assert result is not None

    # One success should be in content
    contents = result.content
    assert isinstance(contents, list)
    assert "success" in contents

    # Errors should be tracked in metadata
    assert "errors" in result.metadata
    errors = cast(dict[str, str], result.metadata["errors"])
    assert "fail_agent" in errors
    assert "boom" in errors["fail_agent"]


@pytest.mark.anyio
async def test_parallel_team_execute_with_error_mode() -> None:
    """Team.execute() handles fail_all vs collect_exceptions modes."""
    agent_ok = _make_echo_agent("ok", "fine")
    agent_fail = FailingAgent("fail", "explosion")

    # Default: collect_exceptions
    team = BaseTeam([agent_ok, agent_fail], name="collect")
    response = await team.execute("prompt")

    # Should have one success and one error
    assert len(response) == 1
    assert len(response.errors) == 1
    assert "fail" in response.errors


@pytest.mark.anyio
async def test_parallel_team_execute_applies_member_timeout() -> None:
    """Team.execute() should collect timed-out graph members as errors."""
    agent_ok = SlowAgent("ok", delay=0)
    agent_slow = SlowAgent("slow", delay=0.2)

    team = BaseTeam([agent_ok, agent_slow], name="timeout_team", member_timeout=0.01)
    response = await team.execute("prompt")

    assert len(response) == 1
    assert response[0].agent_name == "ok"
    assert "slow" in response.errors
    assert isinstance(response.errors["slow"], TimeoutError)


@pytest.mark.anyio
async def test_parallel_team_execute_retries_transient_member_failure() -> None:
    """Team.execute() retries runtime member failures when requested."""
    agent = FlakyAgent("flaky")
    team = BaseTeam([agent], name="retry_team")

    response = await team.execute("prompt", member_retry_attempts=1)

    assert agent.calls == 2
    assert not response.errors
    assert len(response) == 1
    assert response[0].message is not None
    assert response[0].message.content == "recovered"


@pytest.mark.anyio
async def test_parallel_team_execute_does_not_retry_non_runtime_failure() -> None:
    """Team.execute() does not retry non-runtime member failures."""
    agent = InvalidAgent("invalid")
    team = BaseTeam([agent], name="retry_team")

    response = await team.execute("prompt", member_retry_attempts=1)

    assert agent.calls == 1
    assert isinstance(response.errors["invalid"], ValueError)


# ---------------------------------------------------------------------------
# 5. Streaming events from graph-based teams
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_parallel_team_streaming_events() -> None:
    """Parallel Team streaming yields SubAgentEvents from each member."""
    agent_a = _make_echo_agent("alpha", "A")
    agent_b = _make_echo_agent("beta", "B")

    team = BaseTeam([agent_a, agent_b], name="parallel_stream")

    events = await _collect_events(team, "test", session_id="ses_stream")

    # Should have SpawnSessionStart for each member
    spawns = [e for e in events if isinstance(e, SpawnSessionStart)]
    assert len(spawns) == 2

    # Should have SubAgentEvents wrapping member streams
    subs = [e for e in events if isinstance(e, SubAgentEvent)]
    assert len(subs) >= 2

    # Each member should produce at least a StreamCompleteEvent
    complete_events = [e for e in subs if isinstance(e.event, StreamCompleteEvent)]
    assert len(complete_events) == 2


@pytest.mark.anyio
async def test_sequential_team_streaming_events() -> None:
    """TeamRun streaming yields nested SubAgentEvents for each sequential member."""
    agent_1 = _make_echo_agent("step1", "first")
    agent_2 = _make_echo_agent("step2", "second")

    pipeline = BaseTeam([agent_1, agent_2], mode="sequential", name="seq_stream")

    async with agent_1, agent_2:
        events = await _collect_events(pipeline, "start", session_id="ses_seq")

    spawns = [e for e in events if isinstance(e, SpawnSessionStart)]
    assert len(spawns) == 2
    assert spawns[0].source_name == "step1"
    assert spawns[1].source_name == "step2"

    # Depth should be 1 for direct children
    for spawn in spawns:
        assert spawn.depth == 1

    subs = [e for e in events if isinstance(e, SubAgentEvent)]
    assert len(subs) >= 2


# ---------------------------------------------------------------------------
# 6. Signal emission during graph run
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_parallel_team_signals_emitted() -> None:
    """Parallel team run emits message_received and message_sent for each member."""
    agent_a = _make_echo_agent("alpha", "A")
    agent_b = _make_echo_agent("beta", "B")

    received_a: list[ChatMessage[Any]] = []
    sent_a: list[ChatMessage[Any]] = []
    received_b: list[ChatMessage[Any]] = []
    sent_b: list[ChatMessage[Any]] = []

    agent_a.message_received.connect(received_a.append)
    agent_a.message_sent.connect(sent_a.append)
    agent_b.message_received.connect(received_b.append)
    agent_b.message_sent.connect(sent_b.append)

    team = BaseTeam([agent_a, agent_b], name="signal_test")
    await team.run("prompt")

    # Each agent should have received and sent signals
    assert len(received_a) == 1
    assert len(sent_a) == 1
    assert len(received_b) == 1
    assert len(sent_b) == 1

    # Sent messages should contain the agent responses
    assert sent_a[0].content == "A"
    assert sent_b[0].content == "B"


@pytest.mark.anyio
async def test_sequential_team_signals_emitted() -> None:
    """TeamRun emits message_received and message_sent for each step in chain."""
    agent_1 = _make_echo_agent("s1", "out1")
    agent_2 = _make_echo_agent("s2", "out2")

    received: dict[str, list[ChatMessage[Any]]] = {"s1": [], "s2": []}
    sent: dict[str, list[ChatMessage[Any]]] = {"s1": [], "s2": []}

    agent_1.message_received.connect(received["s1"].append)
    agent_1.message_sent.connect(sent["s1"].append)
    agent_2.message_received.connect(received["s2"].append)
    agent_2.message_sent.connect(sent["s2"].append)

    pipeline = BaseTeam([agent_1, agent_2], mode="sequential", name="seq_signals")

    async with agent_1, agent_2:
        await pipeline.run("prompt")

    # Both agents should have received signals
    assert len(received["s1"]) >= 1
    assert len(sent["s1"]) >= 1
    assert len(received["s2"]) >= 1
    assert len(sent["s2"]) >= 1


# ---------------------------------------------------------------------------
# 7. Backward compat: old Team/TeamRun APIs unchanged
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_team_or_operator_creates_teamrun() -> None:
    """Team | Agent still creates a sequential TeamRun (backward compat)."""
    agent_a = _make_echo_agent("a", "A")
    agent_b = _make_echo_agent("b", "B")

    # Using | operator
    pipeline = agent_a | agent_b
    assert isinstance(pipeline, BaseTeam)
    assert pipeline.mode == "sequential"

    async with agent_a, agent_b:
        result = await pipeline.run("start")
    assert result.content == "B"


@pytest.mark.anyio
async def test_team_and_operator_creates_team() -> None:
    """Agent & Agent still creates a parallel Team (backward compat)."""
    agent_a = _make_echo_agent("a", "A")
    agent_b = _make_echo_agent("b", "B")

    # Using & operator
    team = agent_a & agent_b
    assert isinstance(team, BaseTeam)
    assert team.mode == "parallel"

    result = await team.run("start")
    contents = result.content
    assert isinstance(contents, list)
    assert "A" in contents
    assert "B" in contents


@pytest.mark.anyio
async def test_teamrun_run_iter_backward_compat() -> None:
    """TeamRun.run_iter still yields ChatMessage per member (backward compat)."""
    agent_1 = _make_echo_agent("s1", "out1")
    agent_2 = _make_echo_agent("s2", "out2")

    pipeline = BaseTeam([agent_1, agent_2], mode="sequential", name="compat_iter")

    async with agent_1, agent_2:
        messages = [m async for m in pipeline.run_iter("prompt")]

    assert len(messages) == 2
    assert messages[0].content == "out1"
    assert messages[1].content == "out2"


@pytest.mark.anyio
async def test_team_run_iter_backward_compat() -> None:
    """Team.run_iter still yields ChatMessage as they arrive (backward compat)."""
    agent_a = _make_echo_agent("a", "A")
    agent_b = _make_echo_agent("b", "B")

    team = BaseTeam([agent_a, agent_b], name="compat_run_iter")

    messages = [m async for m in team.run_iter("prompt")]

    assert len(messages) == 2
    contents = {m.content for m in messages}
    assert contents == {"A", "B"}


@pytest.mark.anyio
async def test_team_talk_stats_populated() -> None:
    """Team execution populates team_talk stats for monitoring."""
    agent_a = _make_echo_agent("a", "A")
    agent_b = _make_echo_agent("b", "B")

    team = BaseTeam([agent_a, agent_b], name="stats_test")
    await team.execute("prompt")

    stats = team.execution_stats
    # Should have talks recorded (one per member)
    assert stats.num_connections >= 2


@pytest.mark.anyio
async def test_teamrun_talk_stats_populated() -> None:
    """TeamRun execution populates team_talk stats for monitoring."""
    agent_1 = _make_echo_agent("s1", "out1")
    agent_2 = _make_echo_agent("s2", "out2")

    pipeline = BaseTeam([agent_1, agent_2], name="stats_seq")

    async with agent_1, agent_2:
        await pipeline.execute("prompt")

    stats = pipeline.execution_stats
    # Should have talks recorded (one per edge + possibly last_talk)
    assert stats.num_connections >= 1


# ---------------------------------------------------------------------------
# 8. Graph builder internals
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_build_team_graph_creates_fork_join_topology() -> None:
    """build_team_graph creates a GraphBuilder with Fork->members->Join topology."""
    agent_a = _make_echo_agent("a", "A")
    agent_b = _make_echo_agent("b", "B")

    builder = build_team_graph([agent_a, agent_b])
    graph = builder.build()

    # Graph should be buildable and runnable
    state = _TeamGraphState(prompts=("test",))
    result: list[_MemberOutput] = await graph.run(state=state)

    assert len(result) == 2
    names = {o.agent_name for o in result}
    assert names == {"a", "b"}

    for output in result:
        assert output.response is not None
        assert output.exception is None


@pytest.mark.anyio
async def test_run_team_graph_returns_team_response() -> None:
    """run_team_graph returns a TeamResponse with all results."""
    agent_a = _make_echo_agent("a", "A")
    agent_b = _make_echo_agent("b", "B")

    state = _TeamGraphState(prompts=("prompt",))
    response = await run_team_graph([agent_a, agent_b], state)

    assert isinstance(response, TeamResponse)
    assert len(response) == 2
    assert len(response.errors) == 0
    assert response.start_time is not None


@pytest.mark.anyio
async def test_run_team_graph_applies_member_timeout() -> None:
    """run_team_graph should honor _TeamGraphState.member_timeout."""
    agent_slow = SlowAgent("slow", delay=0.2)

    state = _TeamGraphState(prompts=("prompt",), member_timeout=0.01)
    response = await run_team_graph([agent_slow], state)

    assert len(response) == 0
    assert "slow" in response.errors
    assert isinstance(response.errors["slow"], TimeoutError)


@pytest.mark.anyio
async def test_graph_state_shared_prompt_prepended() -> None:
    """_TeamGraphState.shared_prompt is prepended to member inputs."""
    agent_a = _make_echo_agent("a", "A")

    state = _TeamGraphState(
        prompts=("world",),
        shared_prompt="hello",
    )
    builder = build_team_graph([agent_a])
    graph = builder.build()
    result: list[_MemberOutput] = await graph.run(state=state)

    assert len(result) == 1
    assert result[0].response is not None


# ---------------------------------------------------------------------------
# 9. Edge cases
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_single_agent_parallel_team() -> None:
    """Team with a single member is effectively a passthrough."""
    agent = _make_echo_agent("solo", "only")

    team = BaseTeam([agent], name="single")
    result = await team.run("test")

    contents = result.content
    assert isinstance(contents, list)
    assert len(contents) == 1
    assert contents[0] == "only"


@pytest.mark.anyio
async def test_single_agent_sequential_team() -> None:
    """TeamRun with a single member is effectively a passthrough."""
    agent = _make_echo_agent("solo", "only")

    pipeline = BaseTeam([agent], mode="sequential", name="single_seq")
    async with agent:
        result = await pipeline.run("test")

    assert result.content == "only"


@pytest.mark.anyio
async def test_team_structure_diagram() -> None:
    """Team.get_structure_diagram generates a mermaid flowchart."""
    agent_a = _make_echo_agent("a", "A")
    agent_b = _make_echo_agent("b", "B")

    team = BaseTeam([agent_a, agent_b], name="diagram_test")
    diagram = team.get_structure_diagram()

    assert "flowchart TD" in diagram
    assert "a" in diagram
    assert "b" in diagram
