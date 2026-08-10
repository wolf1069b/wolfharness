"""Tests for ``subagent_meta`` property and qwen-mode ``SpawnSessionStart`` conversion.

Verifies that:

- ``subagent_meta`` property returns ``None`` or the expected dict based on
  ``subagent_context``, regardless of ``subagent_display_mode``.
- ``SpawnSessionStart`` in ``"qwen"`` mode yields ``ToolCallStart`` with
  ``kind="other"``, ``status="pending"``, and no ``field_meta`` /
  ``SubagentRunInfo``.
- ``"legacy"`` and ``"zed"`` modes behave as expected for regression.
"""

from __future__ import annotations

import pytest

from acp.schema import AgentMessageChunk, ToolCallStart
from wolfharness.agents.events import SpawnSessionStart
from wolfharness_server.acp_server.event_converter import (
    ACPEventConverter,
    SubagentContext,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def qwen_converter() -> ACPEventConverter:
    """Converter configured for qwen subagent display mode."""
    c = ACPEventConverter(subagent_display_mode="qwen")
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
# subagent_meta property
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_subagent_meta_returns_none_when_no_context() -> None:
    """ACPEventConverter without subagent_context returns None for subagent_meta."""
    converter = ACPEventConverter()
    assert converter.subagent_meta is None


@pytest.mark.unit
def test_subagent_meta_returns_dict_when_context_set() -> None:
    """ACPEventConverter with subagent_context returns the expected meta dict."""
    converter = ACPEventConverter(
        subagent_context=SubagentContext(
            parent_tool_call_id="tc-1",
            subagent_type="coder",
        ),
    )
    expected = {
        "parentToolCallId": "tc-1",
        "subagentType": "coder",
        "provenance": "subagent",
    }
    assert converter.subagent_meta == expected


# ---------------------------------------------------------------------------
# Qwen-mode SpawnSessionStart
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.anyio
async def test_qwen_mode_spawn_yields_tool_call_start_kind_other(
    qwen_converter: ACPEventConverter,
) -> None:
    """SpawnSessionStart in qwen mode yields ToolCallStart with kind='other' and no field_meta."""
    event = _make_spawn_event()
    updates = await _collect(qwen_converter, event)

    assert len(updates) == 1
    assert isinstance(updates[0], ToolCallStart)
    tcs: ToolCallStart = updates[0]  # type: ignore[assignment]
    assert tcs.kind == "other"
    assert tcs.status == "pending"
    assert tcs.field_meta is None


@pytest.mark.unit
@pytest.mark.anyio
async def test_qwen_mode_spawn_no_subagent_run_info(
    qwen_converter: ACPEventConverter,
) -> None:
    """ToolCallStart from qwen mode has no SubagentRunInfo in its serialized form."""
    event = _make_spawn_event()
    updates = await _collect(qwen_converter, event)

    tcs: ToolCallStart = updates[0]  # type: ignore[assignment]
    d = tcs.model_dump(exclude_none=True)
    assert "subagent" not in d, "qwen mode ToolCallStart must not contain SubagentRunInfo"


# ---------------------------------------------------------------------------
# Legacy-mode SpawnSessionStart — regression
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.anyio
async def test_legacy_mode_spawn_yields_agent_message_chunk() -> None:
    """SpawnSessionStart in legacy mode yields AgentMessageChunk, not ToolCallStart."""
    converter = ACPEventConverter(subagent_display_mode="legacy")
    converter._current_message_id = "test-msg-id"
    event = _make_spawn_event()
    updates = await _collect(converter, event)

    assert len(updates) == 1
    assert isinstance(updates[0], AgentMessageChunk)
    chunk: AgentMessageChunk = updates[0]  # type: ignore[assignment]
    assert chunk.field_meta is None
    d = chunk.model_dump(exclude_none=True)
    assert "subagent_session_info" not in d.get("field_meta", {})


# ---------------------------------------------------------------------------
# Zed-mode SpawnSessionStart — regression
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.anyio
async def test_zed_mode_spawn_yields_tool_call_start_kind_other() -> None:
    """SpawnSessionStart in zed mode yields ToolCallStart with field_meta + session_id."""
    converter = ACPEventConverter(subagent_display_mode="zed")
    converter._current_message_id = "test-msg-id"
    child_id = "child_ses_001"
    event = _make_spawn_event(child_session_id=child_id)
    updates = await _collect(converter, event)

    assert len(updates) == 1
    assert isinstance(updates[0], ToolCallStart)
    tcs: ToolCallStart = updates[0]  # type: ignore[assignment]
    assert tcs.kind == "other"
    assert tcs.field_meta is not None
    sub_info = tcs.field_meta.get("subagent_session_info", {})
    assert sub_info.get("session_id") == child_id


# ---------------------------------------------------------------------------
# Default (legacy) converter — subagent_context state
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_legacy_converter_default_subagent_context_is_none() -> None:
    """Default (legacy) converter has subagent_context=None and subagent_meta=None."""
    converter = ACPEventConverter()
    assert converter.subagent_context is None
    assert converter.subagent_meta is None


# ---------------------------------------------------------------------------
# Legacy-mode child session — subagent_meta
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_legacy_mode_child_subagent_meta_returns_dict() -> None:
    """Legacy-mode child session converter returns the expected subagent_meta dict."""
    converter = ACPEventConverter(
        subagent_display_mode="legacy",
        subagent_context=SubagentContext(
            parent_tool_call_id="tc-1",
            subagent_type="coder",
        ),
    )
    expected = {
        "parentToolCallId": "tc-1",
        "subagentType": "coder",
        "provenance": "subagent",
    }
    assert converter.subagent_meta == expected
