"""Tests for zed-mode SpawnSessionStart -> ToolCallStart conversion.

Verifies that when ACPEventConverter is configured with
``subagent_display_mode="zed"``, a SpawnSessionStart event yields a
ToolCallStart notification with the correct ``field_meta`` subagent session
info and a valid UUID tool_call_id.
"""

from __future__ import annotations

from typing import Any
import uuid

from pydantic_ai.models.test import TestModel
import pytest

from acp.schema import ToolCallStart
from wolfharness import Agent
from wolfharness.agents.context import (
    MAX_SUBAGENT_DEPTH,
    AgentContext,
    AgentRunContext,
    SubagentDepthError,
)
from wolfharness.agents.events import SpawnSessionStart
from wolfharness.orchestrator.core import EventBus
from wolfharness_server.acp_server.event_converter import ACPEventConverter


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def zed_converter() -> ACPEventConverter:
    """Converter configured for zed subagent display mode."""
    c = ACPEventConverter(subagent_display_mode="zed")
    c._current_message_id = "test-msg-id"
    return c


def _make_spawn_event(
    child_session_id: str = "child_ses_abc123",
    source_name: str = "coder",
    description: str = "Coding subagent",
    spawn_mechanism: str = "spawn",
) -> SpawnSessionStart:
    """Create a minimal SpawnSessionStart for testing."""
    return SpawnSessionStart(
        child_session_id=child_session_id,
        parent_session_id="parent_ses_xyz",
        source_name=source_name,
        source_type="agent",
        description=description,
        spawn_mechanism=spawn_mechanism,  # type: ignore[arg-type]
        depth=1,
    )


async def _collect(converter: ACPEventConverter, event) -> list[object]:
    """Collect all ACP updates from a converter for a single event."""
    return [u async for u in converter.convert(event)]


# ---------------------------------------------------------------------------
# Zed-mode SpawnSessionStart -> ToolCallStart
# ---------------------------------------------------------------------------


class TestZedModeSpawnSessionStart:
    """Zed-mode SpawnSessionStart conversion to ToolCallStart."""

    @pytest.mark.unit
    async def test_spawn_yields_tool_call_start(self, zed_converter: ACPEventConverter):
        """SpawnSessionStart in zed mode yields a single ToolCallStart."""
        event = _make_spawn_event()
        updates = await _collect(zed_converter, event)

        assert len(updates) == 1
        assert isinstance(updates[0], ToolCallStart)

    @pytest.mark.unit
    async def test_tool_call_id_is_valid_uuid(self, zed_converter: ACPEventConverter):
        """tool_call_id on the yielded ToolCallStart must be a valid UUID."""
        event = _make_spawn_event()
        updates = await _collect(zed_converter, event)
        tcs: ToolCallStart = updates[0]  # type: ignore[assignment]

        # Should be a valid UUID string (not None, not empty)
        assert tcs.tool_call_id is not None
        parsed = uuid.UUID(tcs.tool_call_id)  # raises ValueError on invalid
        assert str(parsed) == tcs.tool_call_id

    @pytest.mark.unit
    async def test_title_is_task_and_status_pending(self, zed_converter: ACPEventConverter):
        """ToolCallStart must have title='coder: Coding subagent' and status='pending'."""
        event = _make_spawn_event()
        updates = await _collect(zed_converter, event)
        tcs: ToolCallStart = updates[0]  # type: ignore[assignment]

        assert tcs.title == "coder: Coding subagent"
        assert tcs.status == "pending"

    @pytest.mark.unit
    async def test_field_meta_session_id_matches_child(self, zed_converter: ACPEventConverter):
        """field_meta.subagent_session_info.session_id must match child_session_id."""
        child_id = "child_ses_001"
        event = _make_spawn_event(child_session_id=child_id)
        updates = await _collect(zed_converter, event)
        tcs: ToolCallStart = updates[0]  # type: ignore[assignment]

        assert tcs.field_meta is not None
        sub_info = tcs.field_meta.get("subagent_session_info", {})
        assert sub_info.get("session_id") == child_id

    @pytest.mark.unit
    async def test_field_meta_message_start_index_is_zero(self, zed_converter: ACPEventConverter):
        """field_meta.subagent_session_info.message_start_index must be 0."""
        event = _make_spawn_event()
        updates = await _collect(zed_converter, event)
        tcs: ToolCallStart = updates[0]  # type: ignore[assignment]

        assert tcs.field_meta is not None
        sub_info = tcs.field_meta.get("subagent_session_info", {})
        assert sub_info.get("message_start_index") == 0

    @pytest.mark.unit
    async def test_field_meta_has_tool_name_task(self, zed_converter: ACPEventConverter):
        """field_meta must include tool_name='task'."""
        event = _make_spawn_event()
        updates = await _collect(zed_converter, event)
        tcs: ToolCallStart = updates[0]  # type: ignore[assignment]

        assert tcs.field_meta is not None
        assert tcs.field_meta.get("tool_name") == "task"

    @pytest.mark.unit
    async def test_legacy_mode_does_not_yield_tool_call_start(
        self,
    ):
        """Legacy mode SpawnSessionStart yields AgentMessageChunk, not ToolCallStart."""
        converter = ACPEventConverter(subagent_display_mode="legacy")
        converter._current_message_id = "test-msg-id"
        event = _make_spawn_event()
        updates = await _collect(converter, event)

        assert len(updates) == 1
        assert not isinstance(updates[0], ToolCallStart)


# ---------------------------------------------------------------------------
# Multiple spawns
# ---------------------------------------------------------------------------


class TestZedModeMultipleSpawns:
    """Multiple SpawnSessionStart events in zed mode."""

    @pytest.mark.unit
    async def test_two_spawns_each_have_unique_tool_call_ids(
        self, zed_converter: ACPEventConverter
    ):
        """Each SpawnSessionStart should get a different tool_call_id."""
        event_a = _make_spawn_event(child_session_id="child_a")
        event_b = _make_spawn_event(child_session_id="child_b")

        updates_a = await _collect(zed_converter, event_a)
        updates_b = await _collect(zed_converter, event_b)

        tcs_a: ToolCallStart = updates_a[0]  # type: ignore[assignment]
        tcs_b: ToolCallStart = updates_b[0]  # type: ignore[assignment]

        assert tcs_a.tool_call_id != tcs_b.tool_call_id


# ---------------------------------------------------------------------------
# 9.1: create_child_session auto-emits SpawnSessionStart with tool_call_id
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.anyio
async def test_create_child_session_auto_emits_spawn_with_tool_call_id() -> None:
    """create_child_session auto-emits SpawnSessionStart with correct tool_call_id.

    Given: An AgentContext with run_ctx set and an EventBus.
    When: create_child_session is called with tool_call_id="test-123".
    Then: SpawnSessionStart is emitted with tool_call_id="test-123" and depth=1.
    """
    event_bus = EventBus()
    run_ctx = AgentRunContext(session_id="parent-ses", event_bus=event_bus)
    agent = Agent(name="test-agent", model=TestModel())

    ctx: AgentContext[Any] = agent.get_context(run_ctx=run_ctx)

    recv = await event_bus.subscribe("parent-ses", scope="session")

    child_sid = await ctx.create_child_session(
        agent_name="child-agent",
        agent_type="native",
        tool_call_id="test-123",
    )

    envelope = await recv.get()
    event = envelope.event

    assert isinstance(event, SpawnSessionStart)
    assert event.tool_call_id == "test-123"
    assert event.depth == 1
    assert event.child_session_id == child_sid
    assert event.parent_session_id == "parent-ses"

    await event_bus.unsubscribe("parent-ses", recv)


# ---------------------------------------------------------------------------
# 9.2: tool_call_id flows ctx → event → converter consistently (end-to-end)
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.anyio
async def test_tool_call_id_flows_event_to_converter_consistently() -> None:
    """tool_call_id on SpawnSessionStart appears in converter's ToolCallStart.

    Given: A SpawnSessionStart with tool_call_id="flow-tc-id".
    When: The event is converted by a zed-mode ACPEventConverter.
    Then: The yielded ToolCallStart has the same tool_call_id.
    """
    converter = ACPEventConverter(subagent_display_mode="zed")
    converter._current_message_id = "test-msg-id"

    event = SpawnSessionStart(
        child_session_id="child_flow_001",
        parent_session_id="parent_ses",
        tool_call_id="flow-tc-id",
        spawn_mechanism="spawn",
        source_name="coder",
        source_type="agent",
        depth=1,
        description="Flow test",
    )

    updates: list[Any] = []
    updates.extend([update async for update in converter.convert(event)])

    assert len(updates) == 1
    tcs: ToolCallStart = updates[0]  # type: ignore[assignment]
    assert tcs.tool_call_id == "flow-tc-id"


# ---------------------------------------------------------------------------
# 9.11: MAX_SUBAGENT_DEPTH enforcement — SubagentDepthError at depth 6
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.anyio
async def test_max_subagent_depth_raises_at_depth_6() -> None:
    """create_child_session raises SubagentDepthError when depth exceeds MAX_SUBAGENT_DEPTH.

    Given: An AgentRunContext with depth=MAX_SUBAGENT_DEPTH (5).
    When: create_child_session is called (child_depth = 6 > MAX_SUBAGENT_DEPTH).
    Then: SubagentDepthError is raised.
    """
    run_ctx = AgentRunContext(session_id="parent-ses", depth=MAX_SUBAGENT_DEPTH)
    agent = Agent(name="test-agent", model=TestModel())

    ctx: AgentContext[Any] = agent.get_context(run_ctx=run_ctx)

    with pytest.raises(SubagentDepthError):
        await ctx.create_child_session(
            agent_name="child-agent",
            agent_type="native",
        )


# ---------------------------------------------------------------------------
# 9.14: team.py yield pattern unaffected by auto-emit changes
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_team_py_uses_yield_spawn_pattern() -> None:
    """base_team.py still uses 'yield SpawnSessionStart(...)' pattern (not create_child_session).

    Given: The base_team.py source file.
    When: We inspect it for SpawnSessionStart usage.
    Then: It uses 'yield SpawnSessionStart(...)' in an async generator, NOT create_child_session().
    """
    import inspect

    from wolfharness.delegation import base_team

    source = inspect.getsource(base_team)

    assert "yield SpawnSessionStart(" in source, (
        "base_team.py must still use 'yield SpawnSessionStart(...)' pattern"
    )
    assert "create_child_session" not in source, (
        "base_team.py must NOT call create_child_session — it uses the yield pattern"
    )
