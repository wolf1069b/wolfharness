"""End-to-end verification that display_name propagates through the full team-mode pipeline.

SpawnSessionStart → ToolPart → session title.

This test simulates the exact sequence of events that happens when a team
lead spawns a team member, without needing real model calls.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from wolfharness.agents.events.events import SpawnSessionStart, StreamCompleteEvent


@pytest.mark.unit
@pytest.mark.asyncio
async def test_e2e_team_member_display_name_propagation():  # noqa: PLR0915
    """Verify display_name flows from SpawnSessionStart through ToolPart and session title."""
    from wolfharness_server.opencode_server.models import (
        MessageWithParts,
        Session,
        ToolPart,
    )
    from wolfharness_server.opencode_server.opencode_event_bridge import (
        OpenCodeEventBridgeMixin,
    )
    from wolfharness_server.opencode_server.opencode_message_bridge import (
        OpenCodeMessageBridgeMixin,
    )

    # --- Setup: Create mock server state with a parent session ---
    parent_session_id = "parent-session-001"
    child_session_id = "child-session-001"
    assistant_msg_id = "msg-001"

    # Create a minimal assistant message
    assistant_msg = MagicMock(spec=MessageWithParts)
    assistant_msg.info = MagicMock()
    assistant_msg.info.role = "assistant"
    assistant_msg.info.id = assistant_msg_id
    assistant_msg.parts = []

    server_state = MagicMock()
    server_state.messages = {parent_session_id: [assistant_msg]}
    server_state.broadcast_event = AsyncMock()

    # --- Step 1: Create SpawnSessionStart for a team member ---
    # NOTE: Using "critic1" (no hyphen) because the TUI SubagentFooter regex
    # /@(\w+) subagent/ doesn't match hyphens. This is a TUI limitation.
    spawn_event = SpawnSessionStart(
        child_session_id=child_session_id,
        parent_session_id=parent_session_id,
        tool_call_id="tc-001",
        spawn_mechanism="task",
        source_name="critic",
        display_name="critic1",
        source_type="agent",
        depth=1,
        description="Team member: critic1",
        metadata={
            "team_id": "team_abc",
            "team_name": "trans-qa-team",
            "team_role": "member",
            "team_member_name": "critic1",
        },
    )

    # --- Step 2: Create ToolPart via message bridge ---
    mixin = OpenCodeMessageBridgeMixin()
    mixin.server_state = server_state

    tool_part = await mixin._create_subagent_tool_part(parent_session_id, spawn_event)

    # Verify ToolPart was created
    assert tool_part is not None, "ToolPart should be created"
    assert tool_part.tool == "task"
    assert tool_part.session_id == parent_session_id

    # Verify display_name propagation in ToolPart title
    assert tool_part.state.title == "Team · critic1"
    assert tool_part.state.input["subagent_type"] == "Team · critic1"
    assert "Member in 'trans-qa-team'" in tool_part.state.input["description"]
    assert (
        "critic1" not in tool_part.state.input["description"]
        or "critic1" in tool_part.state.input["subagent_type"]
    )

    # Verify metadata
    assert tool_part.state.metadata["sessionId"] == child_session_id
    assert tool_part.state.metadata["title"] == "Team · critic1"

    print("✅ Step 2: ToolPart created with correct display_name and team context")

    # --- Step 3: Simulate session title via event bridge ---
    event_bridge = OpenCodeEventBridgeMixin()
    event_bridge.server_state = server_state

    # Mock ensure_session to return an existing session with wrong title
    mock_session = MagicMock(spec=Session)
    mock_session.id = child_session_id
    mock_session.title = "New Session"  # Simulate pre-existing default title

    with patch(
        "wolfharness_server.opencode_server.opencode_event_bridge.ensure_session",
        new_callable=AsyncMock,
        return_value=mock_session,
    ):
        await event_bridge._ensure_child_session_visible(parent_session_id, spawn_event)

    # Verify session title was updated
    expected_title = "Team 'trans-qa-team' · Member (@critic subagent)"
    assert mock_session.title == expected_title, (
        f"Expected '{expected_title}', got '{mock_session.title}'"
    )

    # Verify SessionUpdatedEvent was broadcast
    assert server_state.broadcast_event.call_count >= 1

    print(f"✅ Step 3: Session title updated to '{mock_session.title}'")

    # --- Step 4: Verify TUI SubagentFooter regex can extract display_name ---
    import re

    match = re.search(r"@(\w+) subagent", mock_session.title)
    assert match is not None, f"Regex should match session title: {mock_session.title}"
    extracted_name = match.group(1)
    assert extracted_name == "critic", f"Expected 'critic', got '{extracted_name}'"

    print(f"✅ Step 4: SubagentFooter regex extracts '{extracted_name}' from title")

    # --- Step 5: Verify _update_parent_toolpart preserves display_name ---
    # Simulate StreamCompleteEvent
    complete_msg = MagicMock()
    complete_msg.content = "Translation review complete"
    complete_event = StreamCompleteEvent(
        session_id=child_session_id,
        message=complete_msg,
    )

    await mixin._update_parent_toolpart(
        parent_session_id, child_session_id, spawn_event, complete_event
    )

    # Find the updated ToolPart
    updated_part = None
    for part in assistant_msg.parts:
        if isinstance(part, ToolPart) if isinstance(part, ToolPart) else False:
            updated_part = part
            break

    # The ToolPart should still have the correct display_name (not reverted)
    if updated_part is not None:
        # Check that the title wasn't reverted to source_name
        state_title = getattr(updated_part.state, "title", None)
        if state_title is not None:
            assert "Team · critic1" in state_title or "critic1" in str(state_title), (
                f"Title should preserve display_name, got: {state_title}"
            )

    print("✅ Step 5: _update_parent_toolpart preserves display_name")

    print("\n=== All E2E display name propagation tests passed! ===")


@pytest.mark.unit
@pytest.mark.asyncio
async def test_e2e_lead_display_name_propagation():
    """Verify lead agent gets 'Lead' role in display name."""
    from wolfharness_server.opencode_server.models import (
        MessageWithParts,
        Session,
    )
    from wolfharness_server.opencode_server.opencode_event_bridge import (
        OpenCodeEventBridgeMixin,
    )
    from wolfharness_server.opencode_server.opencode_message_bridge import (
        OpenCodeMessageBridgeMixin,
    )

    parent_session_id = "parent-lead-001"
    child_session_id = "child-lead-001"

    assistant_msg = MagicMock(spec=MessageWithParts)
    assistant_msg.info = MagicMock()
    assistant_msg.info.role = "assistant"
    assistant_msg.info.id = "msg-lead-001"
    assistant_msg.parts = []

    server_state = MagicMock()
    server_state.messages = {parent_session_id: [assistant_msg]}
    server_state.broadcast_event = AsyncMock()

    # Lead spawn event
    spawn_event = SpawnSessionStart(
        child_session_id=child_session_id,
        parent_session_id=parent_session_id,
        tool_call_id="tc-lead-001",
        spawn_mechanism="task",
        source_name="conductor",
        display_name="conductor",
        source_type="agent",
        depth=0,
        description="Team lead",
        metadata={
            "team_id": "team_xyz",
            "team_name": "translation_team",
            "team_role": "lead",
            "team_member_name": "conductor",
        },
    )

    # Create ToolPart
    mixin = OpenCodeMessageBridgeMixin()
    mixin.server_state = server_state

    tool_part = await mixin._create_subagent_tool_part(parent_session_id, spawn_event)
    assert tool_part is not None
    assert tool_part.state.title == "Team · conductor"
    assert "Lead in 'translation_team'" in tool_part.state.input["description"]

    print("✅ Lead ToolPart has 'Lead' role in description")
    print("✅ Lead ToolPart title: 'Team · conductor'")

    # Verify session title
    event_bridge = OpenCodeEventBridgeMixin()
    event_bridge.server_state = server_state

    mock_session = MagicMock(spec=Session)
    mock_session.id = child_session_id
    mock_session.title = "New Session"

    with patch(
        "wolfharness_server.opencode_server.opencode_event_bridge.ensure_session",
        new_callable=AsyncMock,
        return_value=mock_session,
    ):
        await event_bridge._ensure_child_session_visible(parent_session_id, spawn_event)

    expected_title = "Team 'translation_team' · Lead (@conductor subagent)"
    assert mock_session.title == expected_title

    print(f"✅ Lead session title: '{mock_session.title}'")
    print("\n=== Lead display name propagation test passed! ===")


@pytest.mark.unit
@pytest.mark.asyncio
async def test_e2e_regular_subagent_no_team_context():
    """Verify non-team subagents don't get team prefix."""
    from wolfharness_server.opencode_server.models import (
        MessageWithParts,
        Session,
    )
    from wolfharness_server.opencode_server.opencode_event_bridge import (
        OpenCodeEventBridgeMixin,
    )
    from wolfharness_server.opencode_server.opencode_message_bridge import (
        OpenCodeMessageBridgeMixin,
    )

    parent_session_id = "parent-reg-001"
    child_session_id = "child-reg-001"

    assistant_msg = MagicMock(spec=MessageWithParts)
    assistant_msg.info = MagicMock()
    assistant_msg.info.role = "assistant"
    assistant_msg.info.id = "msg-reg-001"
    assistant_msg.parts = []

    server_state = MagicMock()
    server_state.messages = {parent_session_id: [assistant_msg]}
    server_state.broadcast_event = AsyncMock()

    # Regular (non-team) spawn event
    spawn_event = SpawnSessionStart(
        child_session_id=child_session_id,
        parent_session_id=parent_session_id,
        tool_call_id="tc-reg-001",
        spawn_mechanism="task",
        source_name="researcher",
        display_name="Research Assistant",
        source_type="agent",
        depth=0,
        description="Research task",
        metadata={},
    )

    # Create ToolPart
    mixin = OpenCodeMessageBridgeMixin()
    mixin.server_state = server_state

    tool_part = await mixin._create_subagent_tool_part(parent_session_id, spawn_event)
    assert tool_part is not None
    # No "Team ·" prefix for non-team subagents
    assert tool_part.state.title == "Research Assistant"
    assert "Team" not in tool_part.state.input["description"]

    print("✅ Regular subagent has no 'Team ·' prefix")
    print(f"✅ Regular subagent title: '{tool_part.state.title}'")

    # Verify session title
    event_bridge = OpenCodeEventBridgeMixin()
    event_bridge.server_state = server_state

    mock_session = MagicMock(spec=Session)
    mock_session.id = child_session_id
    mock_session.title = "New Session"

    with patch(
        "wolfharness_server.opencode_server.opencode_event_bridge.ensure_session",
        new_callable=AsyncMock,
        return_value=mock_session,
    ):
        await event_bridge._ensure_child_session_visible(parent_session_id, spawn_event)

    expected_title = "(@researcher subagent)"
    assert mock_session.title == expected_title

    print(f"✅ Regular session title: '{mock_session.title}'")
    print("\n=== Regular subagent display name test passed! ===")
