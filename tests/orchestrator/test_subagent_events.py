"""Behavioral locking tests for OpenCode subagent event handling.

These tests capture the CURRENT behavior of OpenCode subagent event routing
through `_event_consumer_loop()` in `session_pool_integration.py`. They serve
as regression guards before any migration of subagent event handling to the
session pool layer.

Key behaviors locked:
- Parent session receives SpawnSessionStart before child stream begins
- Child events are published to child session's event bus
- ToolPart is created in parent session when subagent spawns
- ToolPart transitions from "running" to "completed" when child finishes
- ToolPart transitions to "error" on RunErrorEvent
- Nested subagents (depth >= 2) create recursive ToolParts
- descendants scope delivers child events to parent subscriber
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock

import pytest

from wolfharness import AgentPool, AgentsManifest, NativeAgentConfig
from wolfharness.agents.events import (
    RunErrorEvent,
    SpawnSessionStart,
    StreamCompleteEvent,
)
from wolfharness.messaging import ChatMessage
from wolfharness.orchestrator.core import EventEnvelope
from wolfharness_server.opencode_server.models.parts import (
    ToolPart,
    ToolStateCompleted,
    ToolStateError,
    ToolStateRunning,
)
from wolfharness_server.opencode_server.session_pool_integration import (
    OpenCodeSessionPoolIntegration,
)


pytestmark = pytest.mark.unit


class MockServerState:
    """Minimal mock of OpenCode ServerState for testing.

    Disables SessionPool message routing so events are stored in-memory
    and _create_subagent_tool_part can find and mutate messages directly.
    """

    def __init__(self) -> None:
        self.messages: dict[str, list[Any]] = {}
        self.events: list[Any] = []
        self.working_dir = "/tmp"
        self.agent = MagicMock()
        self.agent.name = "test-agent"
        self.agent.model_name = None
        self.pool: Any = None
        self.pool_or_none: Any = None
        self.session_controller: Any = None
        self.session_status: dict[str, Any] = {}
        self.config = MagicMock()
        # Required by ensure_session() and related helpers in
        # session_pool_integration.py that access ServerState attributes
        # directly (not via getattr).
        self.sessions: dict[str, Any] = {}
        self.session_locks: dict[str, asyncio.Lock] = {}

    def ensure_runtime_session_state(self, session_id: str) -> None:
        """No-op stub for ServerState.ensure_runtime_session_state."""

    def ensure_input_provider(self, session_id: str) -> Any:
        """No-op stub for ServerState.ensure_input_provider."""
        return None

    async def mark_session_idle(self, session_id: str) -> None:
        """No-op stub for ServerState.mark_session_idle."""

    async def broadcast_event(self, event: Any) -> None:
        self.events.append(event)

    def resolve_default_model_info(self) -> tuple[str, str]:
        return "default", "wolfharness"


def _get_last_assistant_message(state: MockServerState, session_id: str) -> Any | None:
    """Get the last assistant message for a session."""
    messages = state.messages.get(session_id, [])
    for msg in reversed(messages):
        if hasattr(msg, "info") and hasattr(msg.info, "role") and msg.info.role == "assistant":
            return msg
    return None


def _get_tool_part_for_child(msg: Any, child_session_id: str) -> ToolPart | None:
    """Find the ToolPart representing a child session."""
    for part in msg.parts:
        if isinstance(part, ToolPart) and isinstance(
            part.state, (ToolStateRunning, ToolStateCompleted, ToolStateError)
        ):
            meta = part.state.metadata
            if isinstance(meta, dict) and meta.get("sessionId") == child_session_id:
                return part
    return None


def _count_tool_parts(msg: Any) -> int:
    """Count ToolParts in a message."""
    return sum(1 for part in msg.parts if isinstance(part, ToolPart))


async def _wait_for(
    condition: Any,
    *,
    timeout: float = 2.0,
    interval: float = 0.01,
    description: str = "condition",
) -> None:
    """Poll until condition() returns True, raising AssertionError on timeout.

    Replaces fixed ``asyncio.sleep()`` calls with deterministic synchronization.
    """
    import time

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return
        await asyncio.sleep(interval)
    raise AssertionError(f"Timed out waiting for {description} after {timeout}s")


# ============================================================================
# Helpers
# ============================================================================


async def _setup_parent_child_sessions(
    session_pool: Any,
    parent_session_id: str,
    child_session_id: str,
    integration: OpenCodeSessionPoolIntegration,
) -> None:
    """Create parent and child sessions and start parent event consumer."""
    await session_pool.create_session(parent_session_id, agent_name="test_agent")
    await session_pool.create_session(
        child_session_id,
        parent_session_id=parent_session_id,
        agent_name="worker",
    )
    await integration._start_event_consumer(parent_session_id)
    # Give consumer time to subscribe before publishing events
    await asyncio.sleep(0.05)


async def _publish_spawn_event(
    session_pool: Any,
    parent_session_id: str,
    child_session_id: str,
    source_name: str = "worker",
    depth: int = 1,
) -> SpawnSessionStart:
    """Publish a SpawnSessionStart event and return it."""
    spawn_event = SpawnSessionStart(
        child_session_id=child_session_id,
        parent_session_id=parent_session_id,
        tool_call_id="tc-1",
        spawn_mechanism="task",
        source_name=source_name,
        source_type="agent",
        depth=depth,
        description=f"Task: {source_name}",
        metadata={"prompt": "do something"},
        model_id="test-model",
    )
    await session_pool.event_bus.publish(parent_session_id, spawn_event)
    await asyncio.sleep(0.1)
    return spawn_event


async def _publish_child_completion(
    session_pool: Any,
    child_session_id: str,
    content: str = "Task completed",
) -> StreamCompleteEvent:
    """Publish a StreamCompleteEvent for a child session."""
    complete_event = StreamCompleteEvent(
        message=ChatMessage(role="assistant", content=content),
        session_id=child_session_id,
    )
    await session_pool.event_bus.publish(child_session_id, complete_event)
    await asyncio.sleep(0.1)
    return complete_event


async def _publish_child_error(
    session_pool: Any,
    child_session_id: str,
    message: str = "Subagent failed",
) -> RunErrorEvent:
    """Publish a RunErrorEvent for a child session."""
    error_event = RunErrorEvent(
        message=message,
        code="AGENT_ERROR",
    )
    # RunErrorEvent doesn't have session_id attribute by default, but
    # _event_consumer_loop reads event_session_id from the event object.
    # We need to set it for the consumer to route it correctly.
    error_event.session_id = child_session_id  # type: ignore[attr-defined]
    await session_pool.event_bus.publish(child_session_id, error_event)
    await asyncio.sleep(0.1)
    return error_event


# ============================================================================
# Test: Parent session receives SpawnSessionStart before child stream begins
# ============================================================================


@pytest.mark.anyio
async def test_parent_receives_spawn_session_start_before_child_stream() -> None:
    """Parent consumer receives SpawnSessionStart before child events flow.

    The _event_consumer_loop subscribes with scope='descendants', so when
    SpawnSessionStart is published on the parent session, the parent consumer
    sees it first and can set up the child consumer before child events arrive.
    """
    agent_config = NativeAgentConfig(
        name="test_agent",
        model="test",
        system_prompt="You are a test agent",
    )
    manifest = AgentsManifest(agents={"test_agent": agent_config})

    async with AgentPool(manifest) as pool:
        session_pool = pool.session_pool
        assert session_pool is not None
        await session_pool.start()

        server_state = MockServerState()
        server_state.pool = pool

        integration = OpenCodeSessionPoolIntegration(
            session_pool=session_pool,
            server_state=server_state,  # type: ignore[arg-type]
        )

        parent_id = "parent-spawn-order"
        child_id = "child-spawn-order"

        await _setup_parent_child_sessions(session_pool, parent_id, child_id, integration)

        # Track order of events received by looking at broadcast events
        server_state.events.clear()

        # Publish SpawnSessionStart on parent
        await _publish_spawn_event(session_pool, parent_id, child_id)

        # Verify parent got the spawn event processed (assistant msg created)
        assistant_msg = _get_last_assistant_message(server_state, parent_id)
        assert assistant_msg is not None, (
            "Parent should have created assistant message after SpawnSessionStart"
        )

        # Now publish a child event
        await _publish_child_completion(session_pool, child_id, "done")

        # The ToolPart should have transitioned to Completed
        tool_part = _get_tool_part_for_child(assistant_msg, child_id)
        assert tool_part is not None
        assert isinstance(tool_part.state, ToolStateCompleted), (
            "ToolPart should be Completed after child stream finishes"
        )

        await integration._stop_event_consumer(parent_id)
        await session_pool.shutdown()


# ============================================================================
# Test: Child events are published to child session's event bus
# ============================================================================


@pytest.mark.anyio
async def test_child_events_published_to_child_event_bus() -> None:
    """Events published on child session ID are routable via child subscription.

    This verifies the EventBus can deliver events to a child-session subscriber
    independently of the parent consumer.
    """
    agent_config = NativeAgentConfig(
        name="test_agent",
        model="test",
        system_prompt="You are a test agent",
    )
    manifest = AgentsManifest(agents={"test_agent": agent_config})

    async with AgentPool(manifest) as pool:
        session_pool = pool.session_pool
        assert session_pool is not None
        await session_pool.start()

        parent_id = "parent-child-bus"
        child_id = "child-child-bus"

        await session_pool.create_session(parent_id, agent_name="test_agent")
        await session_pool.create_session(
            child_id,
            parent_session_id=parent_id,
            agent_name="worker",
        )

        # Subscribe directly to child session (not descendants)
        child_queue = await session_pool.event_bus.subscribe(child_id, scope="session")

        # Publish completion on child session
        complete_event = StreamCompleteEvent(
            message=ChatMessage(role="assistant", content="child-only"),
            session_id=child_id,
        )
        await session_pool.event_bus.publish(child_id, complete_event)

        # Child subscriber should receive the event
        received = await asyncio.wait_for(child_queue.get(), timeout=1.0)
        assert isinstance(received, EventEnvelope)
        assert received.source_session_id == child_id
        assert isinstance(received.event, StreamCompleteEvent)
        assert received.event.message.content == "child-only"

        await session_pool.event_bus.unsubscribe(child_id, child_queue)
        await session_pool.shutdown()


# ============================================================================
# Test: ToolPart is created in parent session when subagent spawns
# ============================================================================


@pytest.mark.anyio
async def test_toolpart_created_in_parent_on_spawn() -> None:
    """SpawnSessionStart causes a ToolPart with Running state to be created.

    The _event_consumer_loop handles SpawnSessionStart by calling
    _create_subagent_tool_part, which appends a ToolPart to the parent
    session's latest assistant message.
    """
    agent_config = NativeAgentConfig(
        name="test_agent",
        model="test",
        system_prompt="You are a test agent",
    )
    manifest = AgentsManifest(agents={"test_agent": agent_config})

    async with AgentPool(manifest) as pool:
        session_pool = pool.session_pool
        assert session_pool is not None
        await session_pool.start()

        server_state = MockServerState()
        server_state.pool = pool

        integration = OpenCodeSessionPoolIntegration(
            session_pool=session_pool,
            server_state=server_state,  # type: ignore[arg-type]
        )

        parent_id = "parent-toolpart-created"
        child_id = "child-toolpart-created"

        await _setup_parent_child_sessions(session_pool, parent_id, child_id, integration)

        # Before spawn: no assistant message / no ToolPart
        assert _get_last_assistant_message(server_state, parent_id) is None

        # Publish spawn event
        await _publish_spawn_event(session_pool, parent_id, child_id, source_name="analyzer")

        # After spawn: assistant message exists with ToolPart in Running state
        assistant_msg = _get_last_assistant_message(server_state, parent_id)
        assert assistant_msg is not None, "Assistant message should be created on SpawnSessionStart"

        tool_part = _get_tool_part_for_child(assistant_msg, child_id)
        assert tool_part is not None, "ToolPart should be created for child session"
        assert isinstance(tool_part.state, ToolStateRunning), (
            f"Expected ToolStateRunning, got {type(tool_part.state).__name__}"
        )
        assert tool_part.tool == "task"
        assert tool_part.state.metadata is not None
        assert tool_part.state.metadata.get("sessionId") == child_id
        assert tool_part.state.metadata.get("title") == "analyzer"

        await integration._stop_event_consumer(parent_id)
        await session_pool.shutdown()


# ============================================================================
# Test: ToolPart transitions from running to completed when child finishes
# ============================================================================


@pytest.mark.anyio
async def test_toolpart_transitions_running_to_completed() -> None:
    """StreamCompleteEvent on child session updates parent ToolPart to Completed.

    The _event_consumer_loop detects child StreamCompleteEvent and calls
    _update_parent_toolpart, which mutates the parent assistant message's
    ToolPart from ToolStateRunning to ToolStateCompleted.
    """
    agent_config = NativeAgentConfig(
        name="test_agent",
        model="test",
        system_prompt="You are a test agent",
    )
    manifest = AgentsManifest(agents={"test_agent": agent_config})

    async with AgentPool(manifest) as pool:
        session_pool = pool.session_pool
        assert session_pool is not None
        await session_pool.start()

        server_state = MockServerState()
        server_state.pool = pool

        integration = OpenCodeSessionPoolIntegration(
            session_pool=session_pool,
            server_state=server_state,  # type: ignore[arg-type]
        )

        parent_id = "parent-running-to-completed"
        child_id = "child-running-to-completed"

        await _setup_parent_child_sessions(session_pool, parent_id, child_id, integration)
        await _publish_spawn_event(session_pool, parent_id, child_id)

        # Verify initial Running state
        assistant_msg = _get_last_assistant_message(server_state, parent_id)
        assert assistant_msg is not None
        tool_part = _get_tool_part_for_child(assistant_msg, child_id)
        assert tool_part is not None
        assert isinstance(tool_part.state, ToolStateRunning)
        start_time = tool_part.state.time.start

        # Publish child completion
        await _publish_child_completion(session_pool, child_id, "All done!")

        # Verify Completed state
        assistant_msg = _get_last_assistant_message(server_state, parent_id)
        assert assistant_msg is not None
        tool_part = _get_tool_part_for_child(assistant_msg, child_id)
        assert tool_part is not None
        assert isinstance(tool_part.state, ToolStateCompleted), (
            f"ToolPart should be Completed, got {type(tool_part.state).__name__}"
        )
        assert tool_part.state.output == "All done!"
        assert tool_part.state.time.start == start_time, "Start time should be preserved"
        assert tool_part.state.time.end is not None, "End time should be set"

        await integration._stop_event_consumer(parent_id)
        await session_pool.shutdown()


# ============================================================================
# Test: ToolPart transitions to error on RunErrorEvent
# ============================================================================


@pytest.mark.anyio
async def test_toolpart_transitions_to_error_on_run_error() -> None:
    """RunErrorEvent on child session updates parent ToolPart to Error state.

    The _event_consumer_loop detects child RunErrorEvent and calls
    _update_parent_toolpart_error, which mutates the ToolPart to ToolStateError.
    """
    agent_config = NativeAgentConfig(
        name="test_agent",
        model="test",
        system_prompt="You are a test agent",
    )
    manifest = AgentsManifest(agents={"test_agent": agent_config})

    async with AgentPool(manifest) as pool:
        session_pool = pool.session_pool
        assert session_pool is not None
        await session_pool.start()

        server_state = MockServerState()
        server_state.pool = pool

        integration = OpenCodeSessionPoolIntegration(
            session_pool=session_pool,
            server_state=server_state,  # type: ignore[arg-type]
        )

        parent_id = "parent-error-transition"
        child_id = "child-error-transition"

        await _setup_parent_child_sessions(session_pool, parent_id, child_id, integration)
        await _publish_spawn_event(session_pool, parent_id, child_id)

        # Verify initial Running state
        assistant_msg = _get_last_assistant_message(server_state, parent_id)
        assert assistant_msg is not None
        tool_part = _get_tool_part_for_child(assistant_msg, child_id)
        assert tool_part is not None
        assert isinstance(tool_part.state, ToolStateRunning)

        # Publish child error
        await _publish_child_error(session_pool, child_id, "Critical failure in subagent")

        # Verify Error state
        assistant_msg = _get_last_assistant_message(server_state, parent_id)
        assert assistant_msg is not None
        tool_part = _get_tool_part_for_child(assistant_msg, child_id)
        assert tool_part is not None
        assert isinstance(tool_part.state, ToolStateError), (
            f"ToolPart should be Error, got {type(tool_part.state).__name__}"
        )
        assert tool_part.state.error == "Critical failure in subagent"
        assert tool_part.state.time.end is not None, "End time should be set on error"

        await integration._stop_event_consumer(parent_id)
        await session_pool.shutdown()


# ============================================================================
# Test: Nested subagents (depth >= 2) create recursive ToolParts
# ============================================================================


@pytest.mark.anyio
async def test_nested_subagents_create_recursive_toolparts() -> None:
    """Depth >= 2 subagents cause recursive child consumers and nested ToolParts.

    When a SpawnSessionStart with depth=2 arrives, the parent consumer spawns
    a child consumer for depth=1, which in turn spawns its own child consumer
    for depth=2. Each level creates its own ToolPart.
    """
    agent_config = NativeAgentConfig(
        name="test_agent",
        model="test",
        system_prompt="You are a test agent",
    )
    manifest = AgentsManifest(agents={"test_agent": agent_config})

    async with AgentPool(manifest) as pool:
        session_pool = pool.session_pool
        assert session_pool is not None
        await session_pool.start()

        server_state = MockServerState()
        server_state.pool = pool

        integration = OpenCodeSessionPoolIntegration(
            session_pool=session_pool,
            server_state=server_state,  # type: ignore[arg-type]
        )

        grandparent_id = "grandparent"
        parent_id = "parent-nested"
        child_id = "child-nested"

        # Create session hierarchy
        await session_pool.create_session(grandparent_id, agent_name="test_agent")
        await session_pool.create_session(
            parent_id,
            parent_session_id=grandparent_id,
            agent_name="worker1",
        )
        await session_pool.create_session(
            child_id,
            parent_session_id=parent_id,
            agent_name="worker2",
        )

        # Start consumer at grandparent level
        await integration._start_event_consumer(grandparent_id)
        await asyncio.sleep(0.05)

        # Publish depth=1 spawn at grandparent -> creates parent consumer
        await _publish_spawn_event(
            session_pool, grandparent_id, parent_id, source_name="worker1", depth=1
        )

        # Wait for grandparent's ToolPart to be created before publishing depth=2
        await _wait_for(
            lambda: (
                _get_last_assistant_message(server_state, grandparent_id) is not None
                and _count_tool_parts(_get_last_assistant_message(server_state, grandparent_id))
                >= 1
            ),
            description="grandparent ToolPart for parent",
        )

        # Publish depth=2 spawn at parent -> creates child consumer
        # Note: we publish on parent_id because the parent consumer is subscribed there
        # But actually, the parent consumer subscribes with descendants scope to parent_id,
        # and the spawn event for child would be published on parent_id by the agent runtime.
        # Let's publish on parent_id.
        await _publish_spawn_event(
            session_pool, parent_id, child_id, source_name="worker2", depth=2
        )

        # Wait for parent's ToolPart to be created before asserting
        await _wait_for(
            lambda: (
                _get_last_assistant_message(server_state, parent_id) is not None
                and _count_tool_parts(_get_last_assistant_message(server_state, parent_id)) >= 1
            ),
            description="parent ToolPart for child",
        )

        # Both levels should have created assistant messages and ToolParts
        gp_msg = _get_last_assistant_message(server_state, grandparent_id)
        assert gp_msg is not None, "Grandparent should have assistant message"
        assert _count_tool_parts(gp_msg) >= 1, (
            "Grandparent should have at least 1 ToolPart (for parent)"
        )

        parent_msg = _get_last_assistant_message(server_state, parent_id)
        assert parent_msg is not None, "Parent should have assistant message"
        assert _count_tool_parts(parent_msg) >= 1, (
            "Parent should have at least 1 ToolPart (for child)"
        )

        # Verify depth-1 ToolPart exists in grandparent
        tool_part_parent = _get_tool_part_for_child(gp_msg, parent_id)
        assert tool_part_parent is not None
        assert isinstance(tool_part_parent.state, ToolStateRunning)

        # Verify depth-2 ToolPart exists in parent
        tool_part_child = _get_tool_part_for_child(parent_msg, child_id)
        assert tool_part_child is not None
        assert isinstance(tool_part_child.state, ToolStateRunning)

        # Complete the deepest child
        await _publish_child_completion(session_pool, child_id, "deep task done")

        # Wait for parent's ToolPart to transition to Completed before asserting
        await _wait_for(
            lambda: (
                _get_last_assistant_message(server_state, parent_id) is not None
                and _get_tool_part_for_child(
                    _get_last_assistant_message(server_state, parent_id), child_id
                )
                is not None
                and isinstance(
                    _get_tool_part_for_child(
                        _get_last_assistant_message(server_state, parent_id), child_id
                    ).state,
                    ToolStateCompleted,
                )
            ),
            description="parent ToolPart for child to be Completed",
        )

        # The parent (depth=1) ToolPart for child should be Completed
        parent_msg = _get_last_assistant_message(server_state, parent_id)
        assert parent_msg is not None
        tool_part_child = _get_tool_part_for_child(parent_msg, child_id)
        assert tool_part_child is not None
        assert isinstance(tool_part_child.state, ToolStateCompleted), (
            f"Depth-2 ToolPart should be Completed, got {type(tool_part_child.state).__name__}"
        )

        await integration._stop_event_consumer(grandparent_id)
        await session_pool.shutdown()


# ============================================================================
# Test: descendants scope delivers child events to parent subscriber
# ============================================================================


@pytest.mark.anyio
async def test_descendants_scope_delivers_child_events_to_parent() -> None:
    """EventBus scope='descendants' routes child session events to parent queue.

    This is the foundational mechanism that makes _event_consumer_loop work:
    the parent consumer subscribes with descendants scope, so when the child
    publishes events, they are wrapped in EventEnvelope and delivered to the
    parent's queue.
    """
    agent_config = NativeAgentConfig(
        name="test_agent",
        model="test",
        system_prompt="You are a test agent",
    )
    manifest = AgentsManifest(agents={"test_agent": agent_config})

    async with AgentPool(manifest) as pool:
        session_pool = pool.session_pool
        assert session_pool is not None
        await session_pool.start()

        parent_id = "parent-descendants"
        child_id = "child-descendants"

        await session_pool.create_session(parent_id, agent_name="test_agent")
        await session_pool.create_session(
            child_id,
            parent_session_id=parent_id,
            agent_name="worker",
        )

        # Parent subscribes with descendants scope
        parent_queue = await session_pool.event_bus.subscribe(parent_id, scope="descendants")

        # Child publishes an event
        child_event = StreamCompleteEvent(
            message=ChatMessage(role="assistant", content="from child"),
            session_id=child_id,
        )
        await session_pool.event_bus.publish(child_id, child_event)

        # Parent queue should receive it wrapped in EventEnvelope
        received = await asyncio.wait_for(parent_queue.get(), timeout=1.0)
        assert isinstance(received, EventEnvelope)
        assert received.source_session_id == child_id
        assert isinstance(received.event, StreamCompleteEvent)
        assert received.event.message.content == "from child"

        # A session-scoped subscriber on parent should NOT get child events
        exact_queue = await session_pool.event_bus.subscribe(parent_id, scope="session")
        await session_pool.event_bus.publish(child_id, child_event)

        # exact_queue should be empty (timeout)
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(exact_queue.get(), timeout=0.2)

        await session_pool.event_bus.unsubscribe(parent_id, parent_queue)
        await session_pool.event_bus.unsubscribe(parent_id, exact_queue)
        await session_pool.shutdown()


# ============================================================================
# Test: Child consumer is spawned and handles child events independently
# ============================================================================


@pytest.mark.anyio
async def test_child_consumer_spawned_and_handles_events() -> None:
    """SpawnSessionStart spawns a child consumer that processes child events.

    When _event_consumer_loop sees SpawnSessionStart, it creates a child task
    subscribed to the child session. The child consumer handles child-specific
    events (e.g. converting StreamCompleteEvent to OpenCode SSE events).
    """
    agent_config = NativeAgentConfig(
        name="test_agent",
        model="test",
        system_prompt="You are a test agent",
    )
    manifest = AgentsManifest(agents={"test_agent": agent_config})

    async with AgentPool(manifest) as pool:
        session_pool = pool.session_pool
        assert session_pool is not None
        await session_pool.start()

        server_state = MockServerState()
        server_state.pool = pool

        integration = OpenCodeSessionPoolIntegration(
            session_pool=session_pool,
            server_state=server_state,  # type: ignore[arg-type]
        )

        parent_id = "parent-child-consumer"
        child_id = "child-consumer"

        await _setup_parent_child_sessions(session_pool, parent_id, child_id, integration)

        # Clear events before spawn so we can count new ones
        server_state.events.clear()
        await _publish_spawn_event(session_pool, parent_id, child_id)

        # Verify parent ToolPart was created (spawn was processed)
        assistant_msg = _get_last_assistant_message(server_state, parent_id)
        assert assistant_msg is not None
        tool_part = _get_tool_part_for_child(assistant_msg, child_id)
        assert tool_part is not None
        assert isinstance(tool_part.state, ToolStateRunning)

        # Publish completion on child
        await _publish_child_completion(session_pool, child_id, "child handled")

        # Give time for both consumers to process
        await asyncio.sleep(0.1)

        # Parent ToolPart should be Completed (parent consumer handled child event)
        assistant_msg = _get_last_assistant_message(server_state, parent_id)
        assert assistant_msg is not None
        tool_part = _get_tool_part_for_child(assistant_msg, child_id)
        assert tool_part is not None
        assert isinstance(tool_part.state, ToolStateCompleted)

        await integration._stop_event_consumer(parent_id)
        await session_pool.shutdown()


# ============================================================================
# Test: SpawnSessionStart without existing assistant message creates one
# ============================================================================


@pytest.mark.anyio
async def test_spawn_creates_assistant_message_if_missing() -> None:
    """If no assistant message exists, SpawnSessionStart creates one.

    The _event_consumer_loop registers the assistant message on first non-spawn
    event OR on SpawnSessionStart (to ensure ToolPart has a message to attach to).
    """
    agent_config = NativeAgentConfig(
        name="test_agent",
        model="test",
        system_prompt="You are a test agent",
    )
    manifest = AgentsManifest(agents={"test_agent": agent_config})

    async with AgentPool(manifest) as pool:
        session_pool = pool.session_pool
        assert session_pool is not None
        await session_pool.start()

        server_state = MockServerState()
        server_state.pool = pool

        integration = OpenCodeSessionPoolIntegration(
            session_pool=session_pool,
            server_state=server_state,  # type: ignore[arg-type]
        )

        parent_id = "parent-no-msg"
        child_id = "child-no-msg"

        await session_pool.create_session(parent_id, agent_name="test_agent")
        await session_pool.create_session(
            child_id,
            parent_session_id=parent_id,
            agent_name="worker",
        )

        # No messages exist yet
        assert parent_id not in server_state.messages

        await integration._start_event_consumer(parent_id)
        await asyncio.sleep(0.05)

        # Publish spawn
        await _publish_spawn_event(session_pool, parent_id, child_id)

        # Assistant message should now exist
        assert parent_id in server_state.messages
        assistant_msg = _get_last_assistant_message(server_state, parent_id)
        assert assistant_msg is not None
        assert hasattr(assistant_msg, "parts")
        assert len(assistant_msg.parts) >= 1  # At least the ToolPart

        await integration._stop_event_consumer(parent_id)
        await session_pool.shutdown()
