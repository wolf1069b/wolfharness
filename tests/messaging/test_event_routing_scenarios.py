"""Verification tests for RFC-0015 Cross-Session Event Routing.

These tests verify the two QA scenarios defined in the Verification Strategy:
1. Event Propagation Chain: Grandchild → Child → Parent
2. Loop Prevention: Events don't re-enter source session
"""

import pytest

from wolfharness.agents.events import RunStartedEvent, SubAgentEvent
from wolfharness.messaging.event_manager import EventManager


pytestmark = pytest.mark.unit


class TestEventPropagationChain:
    """Scenario 1: Grandchild event reaches Parent (Mocked)."""

    @pytest.mark.asyncio
    async def test_grandchild_event_reaches_parent(self):
        """Test that events propagate through the chain C → B → A with correct depth.

        Verification steps:
        1. Emit event from Manager C (Child of B)
        2. Assert Manager C receives Raw Event
        3. Assert Manager B receives SubAgentEvent(depth=1)
        4. Assert Manager A (Parent of B) receives SubAgentEvent(depth=2)
        Expected Result: Correct wrapping and depth increment
        """
        # Setup: Create 3-level hierarchy (Grandparent -> Parent -> Child)
        # A (grandparent) <- B (parent) <- C (child)
        grandparent_events = []
        parent_events = []
        child_events = []

        # Create managers
        grandparent = EventManager(session_id="session-a")
        parent = EventManager(
            session_id="session-b", parent_session_id="session-a", parent=grandparent
        )
        child = EventManager(session_id="session-c", parent_session_id="session-b", parent=parent)

        # Track events at each level by patching emit_agent_event
        original_gp = grandparent.emit_agent_event
        original_parent = parent.emit_agent_event
        original_child = child.emit_agent_event

        async def gp_tracker(event, source_session_id=None):
            from copy import copy

            grandparent_events.append(copy(event))
            return await original_gp(event, source_session_id)

        async def parent_tracker(event, source_session_id=None):
            from copy import copy

            parent_events.append(copy(event))
            return await original_parent(event, source_session_id)

        async def child_tracker(event, source_session_id=None):
            from copy import copy

            child_events.append(copy(event))
            return await original_child(event, source_session_id)

        grandparent.emit_agent_event = gp_tracker
        parent.emit_agent_event = parent_tracker
        child.emit_agent_event = child_tracker

        # Step 1: Emit event from child (C)
        original_event = RunStartedEvent(session_id="session-c", run_id="run-001")
        await child.emit_agent_event(original_event)

        # Step 2-4: Verify propagation chain
        # The event should be wrapped as it propagates up
        assert len(child_events) == 1, f"Child should receive 1 event, got {len(child_events)}"
        assert len(parent_events) == 1, f"Parent should receive 1 event, got {len(parent_events)}"
        assert len(grandparent_events) == 1, (
            f"Grandparent should receive 1 event, got {len(grandparent_events)}"
        )

        # Verify child receives the raw event (not wrapped - wrapping happens on forward)
        child_event = child_events[0]
        assert isinstance(child_event, RunStartedEvent), (
            f"Child event should be RunStartedEvent, got {type(child_event)}"
        )

        # Verify parent receives SubAgentEvent with depth=2
        # (depth starts at 1, incremented by child's _forward_to_parent)
        parent_event = parent_events[0]
        assert isinstance(parent_event, SubAgentEvent), (
            f"Parent event should be SubAgentEvent, got {type(parent_event)}"
        )
        assert parent_event.depth == 2, f"Parent depth should be 2, got {parent_event.depth}"
        assert "session-c" in parent_event.path, (
            f"Parent path should contain session-c, got {parent_event.path}"
        )

        # Verify grandparent receives SubAgentEvent with depth=3
        # (further incremented by parent's _forward_to_parent)
        gp_event = grandparent_events[0]
        assert isinstance(gp_event, SubAgentEvent), (
            f"Grandparent event should be SubAgentEvent, got {type(gp_event)}"
        )
        assert gp_event.depth == 3, f"Grandparent depth should be 3, got {gp_event.depth}"
        assert "session-c" in gp_event.path, (
            f"Grandparent path should contain session-c, got {gp_event.path}"
        )
        assert "session-b" in gp_event.path, (
            f"Grandparent path should contain session-b, got {gp_event.path}"
        )

    @pytest.mark.asyncio
    async def test_event_wrapping_preserves_original(self):
        """Test that the original event is preserved when wrapped."""
        parent = EventManager(session_id="parent-1")
        child = EventManager(session_id="child-1", parent_session_id="parent-1", parent=parent)

        received_events = []
        original = parent.emit_agent_event

        async def tracker(event):
            received_events.append(event)
            return await original(event)

        parent.emit_agent_event = tracker

        original_event = RunStartedEvent(session_id="child-1", run_id="run-002")
        await child.emit_agent_event(original_event)

        assert len(received_events) == 1
        wrapped = received_events[0]
        assert isinstance(wrapped, SubAgentEvent)
        assert wrapped.child_session_id == "child-1"
        assert wrapped.parent_session_id == "parent-1"
        assert wrapped.event == original_event


class TestLoopPrevention:
    """Scenario 2: Event does not re-enter source session."""

    @pytest.mark.asyncio
    async def test_loop_detection_raises_error(self):
        """Test that event routing loop is detected and rejected.

        Verification steps:
        1. Manager A linked to Manager B
        2. Mock routing to attempt reflection back to A
        3. Assert Manager A does NOT receive its own event
        Expected Result: Event dropped by loop detection
        """
        parent = EventManager(session_id="session-parent")
        child = EventManager(
            session_id="session-child", parent_session_id="session-parent", parent=parent
        )

        # Create an event that simulates attempting to route back to parent
        # (parent's session_id already in path)
        looping_event = SubAgentEvent(
            source_name="test",
            source_type="agent",
            event=RunStartedEvent(session_id="test", run_id="run-003"),
            depth=2,
            path=["some-session", "session-parent"],  # parent id already in path
            child_session_id="test-child",
            parent_session_id="session-parent",
        )

        # Attempting to forward should raise RuntimeError
        with pytest.raises(RuntimeError) as exc_info:
            await child._forward_to_parent(looping_event)

        assert "loop detected" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_loop_prevention_in_emit_agent_event(self):
        """Test loop prevention works through the public API."""
        # Setup a cyclic reference (bad configuration)
        manager_a = EventManager(session_id="manager-a")
        manager_b = EventManager(
            session_id="manager-b", parent_session_id="manager-a", parent=manager_a
        )
        # Create a cycle by setting manager_a's parent to manager_b
        # (this is a misconfiguration but tests loop detection)
        manager_a.parent = manager_b
        manager_a.parent_session_id = "manager-b"

        # Create event that already contains manager-b in path
        event = SubAgentEvent(
            source_name="test",
            source_type="agent",
            event=RunStartedEvent(session_id="test", run_id="run-004"),
            depth=1,
            path=["manager-b"],  # Will trigger loop when manager-a tries to forward
            child_session_id="test-child",
            parent_session_id="manager-b",
        )

        # Emit from manager_b which forwards to manager_a
        # Then manager_a tries to forward back to manager_b (loop!)
        with pytest.raises(RuntimeError) as exc_info:
            await manager_b.emit_agent_event(event)

        assert "loop detected" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_valid_routing_no_loop(self):
        """Test that valid non-looping routing works correctly."""
        # A <- B <- C (no cycle)
        grandparent = EventManager(session_id="gp-1")
        parent = EventManager(session_id="p-1", parent_session_id="gp-1", parent=grandparent)

        received = []
        original = grandparent.emit_agent_event

        async def tracker(event):
            received.append(event)
            return await original(event)

        grandparent.emit_agent_event = tracker

        # This should work without raising
        event = RunStartedEvent(session_id="p-1", run_id="run-005")
        await parent.emit_agent_event(event)

        # Verify it was received by grandparent
        assert len(received) == 1
        assert isinstance(received[0], SubAgentEvent)
        assert "p-1" in received[0].path
