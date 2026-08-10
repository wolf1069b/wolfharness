"""Test: session_pool_integration correctly notifies parent on subagent completion.

This tests the _update_parent_toolpart path in OpenCodeSessionPoolIntegration,
which is the actual code path used in production (not EventProcessor directly).
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest

from wolfharness.agents.events import SpawnSessionStart, StreamCompleteEvent
from wolfharness.messaging import ChatMessage
from wolfharness_server.opencode_server.models import (
    MessagePath,
    MessageTime,
    MessageWithParts,
)
from wolfharness_server.opencode_server.models.parts import (
    TimeStart,
    ToolPart,
    ToolStateCompleted,
    ToolStateRunning,
)


pytestmark = pytest.mark.integration


def _make_assistant_msg(session_id: str = "parent-test") -> MessageWithParts:
    return MessageWithParts.assistant(
        message_id="parent-msg-1",
        session_id=session_id,
        time=MessageTime(created=0),
        agent_name="lead-agent",
        model_id="test-model",
        parent_id="parent-user-1",
        provider_id="wolfharness",
        path=MessagePath(cwd="/tmp", root="/tmp"),
    )


@pytest.mark.asyncio
async def test_update_parent_toolpart_via_session_pool_integration(
    server_state: Any,
) -> None:
    """RED FLAG: _update_parent_toolpart must find ToolPart via state.metadata.

    ToolPart stores sessionId in state.metadata, NOT in part.metadata.
    This test verifies the fix for the bug where _update_parent_toolpart
    checked part.metadata instead of part.state.metadata.
    """
    from wolfharness_server.opencode_server.session_pool_integration import (
        OpenCodeSessionPoolIntegration,
    )

    parent_session_id = "parent-sp-test"
    child_session_id = "child-sp-test"

    # Setup: create assistant message with a ToolPart
    assistant_msg = _make_assistant_msg(parent_session_id)
    server_state.messages[parent_session_id] = [assistant_msg]

    # Create a ToolPart with sessionId in state.metadata (not part.metadata)
    tool_part = ToolPart(
        id="part-1",
        message_id="parent-msg-1",
        session_id=parent_session_id,
        tool="task",
        call_id="call-1",
        state=ToolStateRunning(
            time=TimeStart(start=0),
            input={},
            metadata={"sessionId": child_session_id, "title": "Worker"},
            title="Worker",
        ),
        metadata=None,  # part.metadata is None!
    )
    assistant_msg.parts.append(tool_part)

    # Mock session_pool
    mock_session_pool = AsyncMock()
    integration = OpenCodeSessionPoolIntegration(mock_session_pool, server_state)

    # Create spawn event
    spawn = SpawnSessionStart(
        child_session_id=child_session_id,
        parent_session_id=parent_session_id,
        tool_call_id="tc-1",
        spawn_mechanism="task",
        source_name="worker",
        source_type="agent",
        depth=1,
        description="Test task",
        metadata={"prompt": "test"},
        model_id="test-model",
    )

    # Create StreamCompleteEvent
    stream_complete = StreamCompleteEvent(
        message=ChatMessage(role="assistant", content="Done!"),
        session_id=child_session_id,
    )

    # Call _update_parent_toolpart
    await integration._update_parent_toolpart(
        parent_session_id=parent_session_id,
        child_session_id=child_session_id,
        spawn_event=spawn,
        event=stream_complete,
    )

    # Verify ToolPart was updated to Completed
    updated_part = assistant_msg.parts[0]
    assert isinstance(updated_part, ToolPart)
    assert isinstance(updated_part.state, ToolStateCompleted), (
        f"Expected ToolStateCompleted but got {type(updated_part.state).__name__}. "
        f"The _update_parent_toolpart failed to find the ToolPart."
    )
    assert updated_part.state.output == "Done!"
