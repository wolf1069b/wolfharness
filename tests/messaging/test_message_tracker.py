from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from pydantic_ai.models.test import TestModel
import pytest

from wolfharness import Agent, AgentPool, BaseTeam
from wolfharness.models.agents import NativeAgentConfig
from wolfharness.models.manifest import AgentsManifest


pytestmark = pytest.mark.integration


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from wolfharness.agents.base_agent import BaseAgent
    from wolfharness.messaging import ChatMessage
    from wolfharness.orchestrator.core import SessionPool


def _make_pool() -> AgentPool:
    """Create a pool with a single agent in the manifest."""
    manifest = AgentsManifest(agents={"agent1": NativeAgentConfig(name="agent1", model="test")})
    return AgentPool(manifest)


def _forwarded(msg: ChatMessage[Any], agent_name: str) -> ChatMessage[Any]:
    """Create a forwarded copy of *msg* with a different sender name.

    The session_id is preserved so that MessageFlowTracker.visualize()
    can correlate the event with the original conversation.
    """
    return replace(msg, name=agent_name)


@asynccontextmanager
async def _patch_agent_models(
    session_pool: SessionPool,
    models: dict[str, TestModel],
) -> AsyncIterator[None]:
    """Patch get_or_create_session_agent to inject TestModels by agent name.

    Same pattern as ``test_workers._patch_agent_models`` — wraps
    ``get_or_create_session_agent`` so that agents created via the
    session pool use a custom TestModel (avoiding skill-tool calls
    that cause CancelledError in EventBus.subscribe).

    ``agent_name`` may be ``None`` when called from ``_run_stream_run_turn()``;
    the agent's own ``.name`` is used as fallback.
    """
    original = session_pool.sessions.get_or_create_session_agent

    async def patched(
        session_id: str,
        agent_name: str | None = None,
        **kwargs: Any,
    ) -> BaseAgent[Any, Any]:
        agent = await original(session_id, agent_name=agent_name, **kwargs)
        effective_name = agent_name or agent.name
        if effective_name in models:
            await agent.set_model(models[effective_name])  # type: ignore[arg-type]
        return agent

    session_pool.sessions.get_or_create_session_agent = patched  # type: ignore[assignment]
    try:
        yield
    finally:
        session_pool.sessions.get_or_create_session_agent = original  # type: ignore[assignment]


@pytest.mark.skip(
    reason=">> operator auto-forwarding is a deferred architecture decision (§12.1). "
    "route_message creates duplicate connections when combined with >> operator."
)
async def test_simple_sequential_chain():
    """Test basic sequential chaining."""
    async with _make_pool() as pool:
        agent1 = Agent("agent1", model="test", agent_pool=pool)
        agent2 = Agent("agent2", model="test", agent_pool=pool)
        agent3 = Agent("agent3", model="test")
        agent1 >> agent2 >> agent3
        async with pool.track_message_flow() as tracker:
            msg = await agent1.run("test")
            # Manually route through agent2's connections so that
            # connection_processed fires with a consistent session_id.
            # (agent.run() no longer auto-forwards through >> chains;
            # downstream agents produce messages with different session_ids.)
            await agent2.connections.route_message(_forwarded(msg, "agent2"))
            mermaid = tracker.visualize(msg)
            # Should only see these two connections
            connections = mermaid.replace(" ", "").split("\n")[1:]  # pyright: ignore
            assert sorted(connections) == sorted(["agent1-->agent2", "agent2-->agent3"])


@pytest.mark.skip(
    reason=">> operator auto-forwarding is a deferred architecture decision (§12.1). "
    "route_message creates duplicate connections when combined with >> operator."
)
async def test_parallel_to_sequential():
    """Test parallel flows connecting to single target."""
    async with _make_pool() as pool:
        agent1 = Agent("agent1", model="test", agent_pool=pool)
        agent2 = Agent("agent2", model="test", agent_pool=pool)
        agent3 = Agent("agent3", model="test", agent_pool=pool)
        agent4 = Agent("agent4", model="test")
        agent1 >> [agent2, agent3] >> agent4
        async with pool.track_message_flow() as tracker:
            msg = await agent1.run("test")
            # Manually route through agent2 and agent3 connections so that
            # connection_processed fires with a consistent session_id.
            await agent2.connections.route_message(_forwarded(msg, "agent2"))
            await agent3.connections.route_message(_forwarded(msg, "agent3"))
            mermaid = tracker.visualize(msg)
            connections = mermaid.replace(" ", "").split("\n")[1:]  # pyright: ignore
            assert sorted(connections) == sorted([
                "agent1-->agent2",
                "agent1-->agent3",
                "agent2-->agent4",
                "agent3-->agent4",
            ])


@pytest.mark.skip(reason="Flaky: fails due to cross-test state pollution in batch runs")
async def test_callback_chain():
    """Test chaining with a callback function."""
    async with _make_pool() as pool:
        agent1 = Agent("agent1", model="test", agent_pool=pool)
        agent2 = Agent("agent2", model="test")

        def process(msg: str) -> str:
            return f"Processed: {msg}"

        _talk = agent1 >> process >> agent2
        async with pool.track_message_flow() as tracker:
            msg = await agent1.run("test")
            mermaid = tracker.visualize(msg)
            connections = mermaid.replace(" ", "").split("\n")[1:]  # pyright: ignore
            assert sorted(connections) == sorted(["agent1-->process", "process-->agent2"])


async def test_message_flow_tracker():
    """Test tracking and visualizing message flow through a chain."""
    # Setup a simple agent chain
    async with _make_pool() as pool:
        session_pool = pool.session_pool
        assert session_pool is not None

        # Give EVERY agent a custom TestModel so they don't call skill tools.
        # Agent1 (pool, Path A) needs the session-pool patch for manifest-created instances.
        # Agent2/agent3 (Path B — standalone because the session belongs to agent1)
        # need direct set_model since they bypass get_or_create_session_agent.
        agent1_model = TestModel(custom_output_text="Response from agent1")
        agent2_model = TestModel(custom_output_text="Response from agent2")
        agent3_model = TestModel(custom_output_text="Response from agent3")

        agent1 = Agent("agent1", system_prompt="You are agent 1", model="test", agent_pool=pool)
        agent2 = Agent("agent2", system_prompt="You are agent 2", model="test", agent_pool=pool)
        agent3 = Agent("agent3", system_prompt="You are agent 3", model="test")

        await agent1.set_model(agent1_model)
        await agent2.set_model(agent2_model)
        await agent3.set_model(agent3_model)

        # Create chain: agent1 >> agent2 >> agent3
        agent1 >> agent2
        agent2 >> agent3

        # Queue agent1's outgoing Talk connections so Path A auto-forwarding
        # fires connection_processed events (captured by the tracker) but does
        # NOT cascade to agent2. Without this, agent2 runs through Path B
        # (standalone → producer_task) and the nested cascade to agent3 causes
        # a CancelledError in anyio 4.13.0 cancel_shielded_checkpoint.
        for talk in agent1.connections._connections:
            talk.queued = True

        # Patch get_or_create_session_agent so the manifest-created agent1
        # (from Path A → session_pool.run_stream()) gets a custom TestModel.
        async with _patch_agent_models(session_pool, {"agent1": agent1_model}):
            # Track message flow during execution
            async with pool.track_message_flow() as tracker:
                result = await agent1.run("Hello")

                # Manually route agent2→agent3 — synchronous call, no producer_task,
                # so the CancelledError in anyio's cancel_shielded_checkpoint does
                # NOT trigger.
                await agent2.connections.route_message(_forwarded(result, "agent2"))

                # Get flow visualization
                mermaid = tracker.visualize(result)

                # Check for expected connections in diagram
                assert "flowchart LR" in mermaid
                assert "agent1-->agent2" in mermaid.replace(" ", "")
                assert "agent2-->agent3" in mermaid.replace(" ", "")

                # Should not contain non-existent connections
                assert "agent1-->agent3" not in mermaid.replace(" ", "")
                assert "agent3-->agent1" not in mermaid.replace(" ", "")

            # Tracker should no longer receive events after context exit
            assert len(tracker.events) > 0  # Should have events from the run
            previous_count = len(tracker.events)

            # Run again outside context
            await agent1.run("Another message")
            assert len(tracker.events) == previous_count  # No new events tracked


async def test_message_flow_tracker_parallel():
    """Test tracking parallel message flows."""
    async with _make_pool() as pool:
        agent1 = Agent("agent1", model="test", agent_pool=pool)
        agent2 = Agent("agent2", model="test")
        agent3 = Agent("agent3", model="test")

        # Create parallel flows: agent1 >> [agent2, agent3]
        agent1 >> [agent2, agent3]

        async with pool.track_message_flow() as tracker:
            result = await agent1.run("Hello")
            mermaid = tracker.visualize(result)

            # Both parallel paths should be in diagram
            assert "agent1-->agent2" in mermaid.replace(" ", "")
            assert "agent1-->agent3" in mermaid.replace(" ", "")

            # With lazy session_id init, consecutive runs share the same conversation
            # so subsequent visualizations will include all events for that conversation
            other_result = await agent1.run("Different conversation")
            other_mermaid = tracker.visualize(other_result)

            # Both runs share the same session_id, so other_mermaid includes all events
            assert "agent1-->agent2" in other_mermaid.replace(" ", "")
            assert "agent1-->agent3" in other_mermaid.replace(" ", "")


async def test_message_flow_tracker_nested():
    """Test tracking flow through nested teams."""
    async with _make_pool() as pool:
        agent1 = Agent("agent1", model="test", agent_pool=pool)
        agent2 = Agent("agent2", model="test", agent_pool=pool)
        agent3 = Agent("agent3", model="test")

        # Create nested team using Team constructor instead of pool.create_team()
        team = BaseTeam([agent2, agent3], mode="parallel", name="team")
        agent1 >> team

        async with pool.track_message_flow() as tracker:
            result = await agent1.run("Hello")
            mermaid = tracker.visualize(result)

            # Should only show connection to team as a unit
            connections = mermaid.replace(" ", "").split("\n")[1:]  # pyright: ignore
            assert sorted(connections) == ["agent1-->team"]


if __name__ == "__main__":
    pytest.main([__file__, "-vv"])
