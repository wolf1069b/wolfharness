"""Tests for ``_meta`` guardrails in legacy subagent display mode.

Ensures that ``subagent_session_info`` is NOT emitted in ``field_meta``
when the converter is in legacy mode (the default). The ``subagent_session_info``
field is exclusive to ``subagent_display_mode="zed"`` and must not leak
into legacy client protocols.

``SubAgentEvent`` content rendering is delegated to child consumers by the
simplified architecture — the parent converter intentionally drops these
events. Only ``SpawnSessionStart`` guardrails remain here.
"""

from __future__ import annotations

import pytest

from wolfharness.agents.events import SpawnSessionStart
from wolfharness_server.acp_server.event_converter import ACPEventConverter


@pytest.fixture
def converter() -> ACPEventConverter:
    """Create a converter with default (legacy) subagent display mode."""
    c = ACPEventConverter()
    c._current_message_id = "test-msg-id"
    return c


def _dump(update: object) -> dict[str, object]:
    """Convert an ACP update to a dict for assertion."""
    if hasattr(update, "model_dump"):
        return update.model_dump(exclude_none=True)  # type: ignore[union-attr]
    return {"_str": str(update)}


def _has_subagent_session_info(update_dict: dict[str, object]) -> bool:
    """Check if the update dict contains subagent_session_info in field_meta."""
    field_meta = update_dict.get("field_meta")
    if isinstance(field_meta, dict):
        return "subagent_session_info" in field_meta
    return False


# ---------------------------------------------------------------------------
# SpawnSessionStart — legacy mode
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_spawn_session_start_legacy_no_subagent_session_info(
    converter: ACPEventConverter,
):
    """SpawnSessionStart in legacy mode must NOT emit field_meta.subagent_session_info."""
    event = SpawnSessionStart(
        child_session_id="child_001",
        parent_session_id="parent_001",
        tool_call_id="tc_001",
        spawn_mechanism="spawn",
        source_name="coder",
        source_type="agent",
        description="Analyzing code",
    )
    updates = [u async for u in converter.convert(event)]

    assert len(updates) >= 1
    for update in updates:
        d = _dump(update)
        assert not _has_subagent_session_info(d), (
            f"Legacy SpawnSessionStart emitted subagent_session_info: {d.get('field_meta')}"
        )


@pytest.mark.unit
async def test_spawn_session_start_legacy_is_text_not_tool_call(
    converter: ACPEventConverter,
):
    """SpawnSessionStart in legacy mode yields AgentMessageChunk, not ToolCallStart."""
    event = SpawnSessionStart(
        child_session_id="child_001",
        parent_session_id="parent_001",
        tool_call_id="tc_001",
        spawn_mechanism="spawn",
        source_name="coder",
        source_type="agent",
        description="Analyzing code",
    )
    updates = [u async for u in converter.convert(event)]

    assert len(updates) >= 1
    for update in updates:
        d = _dump(update)
        # Legacy mode yields AgentMessageChunk (session_update = "agent_message_chunk"),
        # NOT ToolCallStart
        assert d.get("session_update") == "agent_message_chunk", (
            f"Expected agent_message_chunk, got {d.get('session_update')}"
        )


@pytest.mark.unit
async def test_spawn_session_start_task_mechanism_legacy_no_subagent_session_info(
    converter: ACPEventConverter,
):
    """SpawnSessionStart with task spawn_mechanism in legacy mode must not leak meta."""
    event = SpawnSessionStart(
        child_session_id="child_002",
        parent_session_id="parent_001",
        tool_call_id="tc_002",
        spawn_mechanism="task",
        source_name="researcher",
        source_type="agent",
        description="Searching docs",
    )
    updates = [u async for u in converter.convert(event)]

    assert len(updates) >= 1
    for update in updates:
        d = _dump(update)
        assert not _has_subagent_session_info(d), (
            f"Legacy task SpawnSessionStart leaked subagent_session_info: {d.get('field_meta')}"
        )


@pytest.mark.unit
async def test_spawn_session_start_legacy_child_session_tracked(
    converter: ACPEventConverter,
):
    """SpawnSessionStart in legacy mode still tracks child session in _child_sessions."""
    event = SpawnSessionStart(
        child_session_id="child_track_001",
        parent_session_id="parent_001",
        spawn_mechanism="spawn",
        source_name="helper",
        source_type="agent",
        description="Helping",
    )
    _ = [u async for u in converter.convert(event)]

    assert "child_track_001" in converter._child_sessions


# ---------------------------------------------------------------------------
# 9.13: Legacy mode unchanged — no SubagentRunInfo, no _meta
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_legacy_mode_no_subagent_run_info_or_field_meta(
    converter: ACPEventConverter,
):
    """Legacy mode SpawnSessionStart has no SubagentRunInfo and no field_meta.

    Given: A SpawnSessionStart in legacy mode (default converter).
    When: The event is converted.
    Then: No update contains 'subagent' field (SubagentRunInfo) or 'field_meta'.
    """
    event = SpawnSessionStart(
        child_session_id="child_legacy_001",
        parent_session_id="parent_001",
        tool_call_id="tc_legacy_001",
        spawn_mechanism="spawn",
        source_name="coder",
        source_type="agent",
        description="Legacy guardrail test",
    )
    updates = [u async for u in converter.convert(event)]

    assert len(updates) >= 1
    for update in updates:
        d = _dump(update)
        # Legacy mode must NOT have SubagentRunInfo
        assert "subagent" not in d, f"Legacy mode leaked SubagentRunInfo: {d.get('subagent')}"
        # Legacy mode must NOT have field_meta
        assert d.get("field_meta") is None, f"Legacy mode leaked field_meta: {d.get('field_meta')}"
