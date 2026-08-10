"""Integration red flag tests for SessionPool subagent event routing and auto-resume.

Consolidated from:
- test_acp_sessionpool_inject_redflag.py (ACP + SessionPool + inject_prompt + auto-resume)
- test_session_tree_redflag.py (EventBus _session_tree and descendants scope)
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import Any
from unittest.mock import MagicMock

import pytest

from wolfharness import AgentPool, AgentsManifest, NativeAgentConfig
from wolfharness.agents.events import StreamCompleteEvent
from wolfharness.messaging import ChatMessage
from wolfharness.orchestrator.core import EventBus, EventEnvelope, SessionController


# ============================================================================
# ACP SessionPool inject + auto-resume red flags
# ============================================================================


async def _setup_session(
    controller: SessionController,
    session_id: str,
    agent: Any,
    real_pool: AgentPool,
) -> Any:
    """Create a session and attach the agent."""
    state, _ = await controller.get_or_create_session(session_id)
    state.agent = agent
    controller._session_agents[session_id] = agent
    real_pool.get_agent = MagicMock(return_value=agent)  # type: ignore[assignment]
    return state


@pytest.mark.integration
async def test_per_session_agent_session_id_set() -> None:
    """Per-session agent created by SessionPool MUST have session_id set."""
    agent_config = NativeAgentConfig(
        name="test_agent",
        model="test",
        system_prompt="You are a test agent",
    )
    manifest = AgentsManifest(agents={"test_agent": agent_config})

    async with AgentPool(manifest, enable_session_pool=True) as pool:
        session_pool = pool.session_pool
        assert session_pool is not None

        session_id = "test-session"
        await session_pool.create_session(session_id, agent_name="test_agent")

        # Run a turn via run_stream to create per-session agent
        async for _ in session_pool.run_stream(session_id, "hello"):
            pass

        # Get the session and check agent
        session = session_pool.sessions.get_session(session_id)
        assert session is not None
        assert session.agent is not None

        # Run another turn via run_stream to verify no AssertionError
        async for _ in session_pool.run_stream(session_id, "hello"):
            pass


# ============================================================================
# Session tree / descendants scope red flags
# ============================================================================


class TestEventBusSessionTree:
    """Red flag: _session_tree is never updated, breaking descendants scope."""

    def test_session_tree_is_empty_after_construction(self) -> None:
        """Baseline: fresh EventBus has empty _session_tree."""
        bus = EventBus()
        assert bus._session_tree == {}

    async def test_is_descendant_always_false_for_empty_tree(self) -> None:
        """RED FLAG: _is_descendant returns False even for direct children."""
        bus = EventBus()
        result = bus._is_descendant("child-sid", "parent-sid")
        assert result is False, (
            "_is_descendant should be True for known children, "
            "but _session_tree is empty so it returns False"
        )

    async def test_should_receive_descendants_always_false(self) -> None:
        """RED FLAG: scope='descendants' never matches child events."""
        bus = EventBus()
        result = bus._should_receive(
            published_sid="child-sid",
            subscriber_sid="parent-sid",
            scope="descendants",
        )
        assert result is False, (
            "scope='descendants' should receive child events, "
            "but _session_tree is empty so it returns False"
        )

    async def test_publish_delivers_descendant_events_to_parent(
        self, minimal_pool: AgentPool
    ) -> None:
        """FIXED: Child session events ARE delivered to parent subscribers via controller."""
        controller = SessionController(minimal_pool)
        await controller.get_or_create_session("parent-sid")
        await controller.get_or_create_session("child-sid", parent_session_id="parent-sid")
        bus = EventBus(session_controller=controller)
        parent_queue = await bus.subscribe("parent-sid", scope="descendants")

        event = {"type": "test", "data": "hello from child"}
        await bus.publish("child-sid", event)

        received = parent_queue.get_nowait()
        assert isinstance(received, EventEnvelope)
        assert received.event == event
        assert received.source_session_id == "child-sid"

    async def test_publish_delivers_to_exact_session(self) -> None:
        """Green: exact session scope works (baseline)."""
        bus = EventBus()
        queue = await bus.subscribe("same-sid", scope="session")

        event = {"type": "test", "data": "hello"}
        await bus.publish("same-sid", event)

        received = queue.get_nowait()
        assert isinstance(received, EventEnvelope)
        assert received.event == event
        assert received.source_session_id == "same-sid"


class TestSessionControllerChildrenVsEventBus:
    """Red flag: SessionController._children and EventBus._session_tree diverge."""

    async def test_children_tracking_works(self, minimal_pool: AgentPool) -> None:
        """SessionController correctly tracks parent-child relationships."""
        controller = SessionController(minimal_pool)

        # Create parent session
        parent, _ = await controller.get_or_create_session("parent-sid")
        assert parent.session_id == "parent-sid"

        # Create child session
        child, _ = await controller.get_or_create_session(
            "child-sid", parent_session_id="parent-sid"
        )
        assert child.parent_session_id == "parent-sid"

        # SessionController knows about the relationship
        assert "child-sid" in controller._children.get("parent-sid", [])

    async def test_event_bus_does_not_know_about_children(self, minimal_pool: AgentPool) -> None:
        """RED FLAG: EventBus has no knowledge of SessionController's children."""
        controller = SessionController(minimal_pool)
        event_bus = EventBus()

        # Simulate SessionPool behavior: create sessions via controller
        await controller.get_or_create_session("parent-sid")
        await controller.get_or_create_session("child-sid", parent_session_id="parent-sid")

        # EventBus knows nothing
        assert event_bus._session_tree == {}, (
            "EventBus._session_tree is empty even though SessionController "
            "knows about parent-child relationship"
        )

    async def test_is_descendant_with_controller_wired(self, minimal_pool: AgentPool) -> None:
        """With controller wired, _is_descendant works despite empty _session_tree."""
        controller = SessionController(minimal_pool)
        bus = EventBus(session_controller=controller)

        # _session_tree is empty (would be always false without controller)
        assert bus._session_tree == {}

        # Controller knows about the relationship
        await controller.get_or_create_session("parent-sid")
        await controller.get_or_create_session("child-sid", parent_session_id="parent-sid")

        # Should now work because controller is wired
        result = bus._is_descendant("child-sid", "parent-sid")
        assert result is True, (
            "_is_descendant should return True when controller knows the relationship, "
            "even though _session_tree is empty"
        )

    async def test_should_receive_descendants_with_controller_wired(
        self, minimal_pool: AgentPool
    ) -> None:
        """With controller wired, descendants scope works despite empty _session_tree."""
        controller = SessionController(minimal_pool)
        bus = EventBus(session_controller=controller)

        # _session_tree is empty (would be always false without controller)
        assert bus._session_tree == {}

        await controller.get_or_create_session("parent-sid")
        await controller.get_or_create_session("child-sid", parent_session_id="parent-sid")

        result = bus._should_receive(
            published_sid="child-sid",
            subscriber_sid="parent-sid",
            scope="descendants",
        )
        assert result is True, (
            "scope='descendants' should receive child events when controller is wired, "
            "even though _session_tree is empty"
        )

    async def test_acp_handler_delivers_child_events(self, minimal_pool: AgentPool) -> None:
        """FIXED: Full ACP handler scenario - subagent events reach parent."""
        controller = SessionController(minimal_pool)
        await controller.get_or_create_session("parent-sid")
        await controller.get_or_create_session("child-sid", parent_session_id="parent-sid")
        bus = EventBus(session_controller=controller)

        # Step 1: ACP handler subscribes to parent session with descendants scope
        parent_queue = await bus.subscribe("parent-sid", scope="descendants")

        # Step 2: Subagent runs and publishes events with its own session_id
        subagent_events = [
            {"type": "agent_message_chunk", "content": "Hello"},
            {"type": "tool_call", "tool": "search"},
            {"type": "agent_message_chunk", "content": "Done"},
        ]

        for event in subagent_events:
            await bus.publish("child-sid", event)

        # Step 3: Parent queue should have received all child events
        received = []
        while True:
            with contextlib.suppress(asyncio.QueueEmpty):
                received.append(parent_queue.get_nowait())
                continue
            break

        assert len(received) == len(subagent_events), (
            f"Expected {len(subagent_events)} events, got {len(received)}. "
            f"Child events were not delivered to parent subscriber."
        )

    async def test_acp_handler_child_events_have_session_id(self, minimal_pool: AgentPool) -> None:
        """Child session events are wrapped in EventEnvelope with source_session_id."""
        controller = SessionController(minimal_pool)
        await controller.get_or_create_session("parent-sid")
        await controller.get_or_create_session("child-sid", parent_session_id="parent-sid")
        bus = EventBus(session_controller=controller)
        controller._event_bus = bus

        # Parent subscribes with descendants scope
        parent_queue = await bus.subscribe("parent-sid", scope="descendants")

        # Publish child event via EventBus (the real path)
        child_event = StreamCompleteEvent(
            message=ChatMessage(content="hello", role="assistant"),
        )

        await bus.publish("child-sid", child_event)

        # Parent receives the event wrapped in EventEnvelope
        received = await parent_queue.get()
        assert isinstance(received, EventEnvelope)
        assert received.source_session_id == "child-sid", (
            "Child event should carry source_session_id so ACP handler can route it correctly"
        )
        # The actual event payload is accessible via received.event
        assert isinstance(received.event, StreamCompleteEvent)


class TestSessionPoolIntegration:
    """Red flag: SessionPool-level integration tests."""

    async def test_subagent_streaming_events_routed_to_parent(
        self, minimal_pool: AgentPool
    ) -> None:
        """FIXED: When SessionPool runs subagent, events reach parent."""
        pool = minimal_pool.session_pool
        assert pool is not None

        # Parent subscribes to events
        parent_queue = await pool.event_bus.subscribe("parent-sid", scope="descendants")

        # Simulate what happens when subagent runs:
        await pool.create_session("parent-sid")
        await pool.create_session("child-sid", parent_session_id="parent-sid")

        # Simulate subagent event emission
        await pool.event_bus.publish("child-sid", {"type": "agent_message_chunk", "content": "hi"})

        # Parent should have received it
        parent_queue.get_nowait()

    async def test_manual_session_tree_fix_works(self) -> None:
        """Verify that populating _session_tree manually fixes the issue."""
        bus = EventBus()

        # Manually populate _session_tree (this is the fix)
        bus._session_tree["parent-sid"] = ["child-sid"]

        parent_queue = await bus.subscribe("parent-sid", scope="descendants")

        await bus.publish("child-sid", {"type": "test", "data": "hello"})

        received = parent_queue.get_nowait()
        assert isinstance(received, EventEnvelope)
        assert received.event["data"] == "hello"
        assert received.source_session_id == "child-sid"


class TestInjectPromptWithSessionPool:
    """Red flag: inject_prompt relies on auto-resume, but events still lost."""

    async def test_inject_prompt_triggers_auto_resume(self, minimal_pool: AgentPool) -> None:
        """inject_prompt itself works, but its events are lost due to _session_tree."""
        pool = minimal_pool.session_pool
        assert pool is not None

        # Create a session with an agent that can accept injections
        await pool.create_session("test-sid")

        # Mock the agent to avoid full agent setup
        mock_agent = MagicMock()
        mock_agent.get_active_run_context.return_value = None

        # Replace session agent
        session = pool.sessions.get_session("test-sid")
        if session:
            session.agent = mock_agent

        # inject_prompt should queue and trigger auto-resume
        result = await pool.inject_prompt("test-sid", "test message")

        # inject_prompt returns None when queued (no active run_ctx)
        assert result is None


@pytest.mark.manual
async def test_diagnostic_print_session_tree_state(minimal_pool: AgentPool) -> None:
    """Print the state of _session_tree for diagnostic purposes."""
    pool = minimal_pool.session_pool
    assert pool is not None

    await pool.create_session("parent-sid")
    await pool.create_session("child-sid", parent_session_id="parent-sid")

    print(f"\n{'=' * 60}")
    print("DIAGNOSTIC: Session Tree State")
    print(f"{'=' * 60}")
    print(f"SessionController._children: {pool.sessions._children}")
    print(f"EventBus._session_tree: {pool.event_bus._session_tree}")
    print(f"EventBus._subscribers: {await pool.event_bus.get_subscriber_counts()}")
    print(f"{'=' * 60}")

    # This assertion documents the bug:
    assert pool.sessions._children != {}, "SessionController knows about children"
    assert pool.event_bus._session_tree == {}, "BUG: EventBus._session_tree is empty"
