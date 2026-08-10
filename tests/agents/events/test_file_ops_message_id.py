"""Tests for opencode_message_id propagation to FileOpsTracker.

Verifies that file change records carry the OpenCode message ID (not turn_id)
so that file rollback via ``get_revert_operations(since_message_id=X)`` works
correctly.

Covers OpenSpec tasks 7.4-7.7 for session-revert-stage-clear-commit.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from wolfharness import Agent
from wolfharness.agents.context import AgentContext, AgentRunContext
from wolfharness.agents.events import DiffContentItem, StreamEventEmitter
from wolfharness.utils.streams import FileOpsTracker


@pytest.mark.unit
async def test_record_change_uses_opencode_message_id_not_turn_id() -> None:
    """FileChange.message_id matches opencode_message_id, not turn_id.

    Given an AgentRunContext with opencode_message_id="msg_test_123" and
    turn_id="uuid-456", record_change via file_edit_progress should produce
    a FileChange with message_id="msg_test_123".
    """
    # Given: a pool with FileOpsTracker and an AgentContext wired with
    # opencode_message_id set on the run context.
    tracker = FileOpsTracker()
    pool = MagicMock()
    pool.file_ops = tracker

    agent = Agent(name="test_agent", model="test")
    agent.session_id = "session-1"
    agent._pool = pool  # type: ignore[private-type]

    run_ctx = AgentRunContext(
        opencode_message_id="msg_test_123",
        turn_id="uuid-456",
    )
    ctx = AgentContext(node=agent, run_ctx=run_ctx, pool=pool)

    emitter = StreamEventEmitter(ctx)

    # When: file_edit_progress records a change
    await emitter.file_edit_progress(
        path="src/main.py",
        old_text="def foo(): pass",
        new_text="def foo():\n    return 42",
        status="completed",
    )

    # Then: the FileChange carries opencode_message_id, not turn_id
    assert len(tracker.changes) == 1
    change = tracker.changes[0]
    assert change.message_id == "msg_test_123"
    assert change.message_id != "uuid-456"


@pytest.mark.unit
async def test_get_revert_operations_filters_by_message_id() -> None:
    """get_revert_operations(since_message_id=X) returns changes from X onwards.

    Records changes for message "msg_A" and message "msg_B", then verifies:
    - since_message_id="msg_A" returns changes from msg_A onwards (incl msg_B)
    - since_message_id="msg_B" returns only msg_B's changes
    - earlier changes (before msg_A) are preserved in self.changes
    """
    tracker = FileOpsTracker()

    # Record changes under two different message IDs
    tracker.record_change(
        path="file_a.py",
        old_content=None,
        new_content="content_a",
        operation="create",
        message_id="msg_A",
    )
    tracker.record_change(
        path="file_b.py",
        old_content="old_b",
        new_content="new_b",
        operation="edit",
        message_id="msg_B",
    )

    # When: get_revert_operations since msg_A
    revert_from_a = tracker.get_revert_operations(since_message_id="msg_A")

    # Then: returns both files (msg_A and msg_B onwards)
    paths_from_a = {path for path, _ in revert_from_a}
    assert paths_from_a == {"file_a.py", "file_b.py"}

    # When: get_revert_operations since msg_B
    revert_from_b = tracker.get_revert_operations(since_message_id="msg_B")

    # Then: returns only file_b
    paths_from_b = {path for path, _ in revert_from_b}
    assert paths_from_b == {"file_b.py"}

    # Earlier changes are preserved in self.changes
    assert len(tracker.changes) == 2
    assert tracker.changes[0].message_id == "msg_A"
    assert tracker.changes[1].message_id == "msg_B"


@pytest.mark.unit
async def test_progress_items_passes_opencode_message_id() -> None:
    """progress_items() (diff rendering path) passes opencode_message_id.

    When tool_call_progress is called with DiffContentItem items,
    record_change should receive the opencode_message_id from the run context.
    """
    tracker = FileOpsTracker()
    pool = MagicMock()
    pool.file_ops = tracker

    agent = Agent(name="test_agent", model="test")
    agent.session_id = "session-1"
    agent._pool = pool  # type: ignore[private-type]

    run_ctx = AgentRunContext(
        opencode_message_id="msg_diff_001",
        turn_id="turn-uuid-abc",
    )
    ctx = AgentContext(node=agent, run_ctx=run_ctx, pool=pool)

    emitter = StreamEventEmitter(ctx)

    # When: tool_call_progress with a DiffContentItem
    diff_item = DiffContentItem(
        path="src/utils.py",
        old_text="old_code",
        new_text="new_code",
    )
    await emitter.tool_call_progress(
        title="Editing file",
        items=[diff_item],
    )

    # Then: the FileChange carries the opencode_message_id
    assert len(tracker.changes) == 1
    assert tracker.changes[0].message_id == "msg_diff_001"


@pytest.mark.unit
async def test_file_edit_progress_passes_opencode_message_id() -> None:
    """file_edit_progress() (edit path) passes opencode_message_id.

    When file_edit_progress is called, record_change should receive
    the opencode_message_id from the run context.
    """
    tracker = FileOpsTracker()
    pool = MagicMock()
    pool.file_ops = tracker

    agent = Agent(name="test_agent", model="test")
    agent.session_id = "session-1"
    agent._pool = pool  # type: ignore[private-type]

    run_ctx = AgentRunContext(
        opencode_message_id="msg_edit_002",
        turn_id="turn-uuid-def",
    )
    ctx = AgentContext(node=agent, run_ctx=run_ctx, pool=pool)

    emitter = StreamEventEmitter(ctx)

    # When: file_edit_progress records a change
    await emitter.file_edit_progress(
        path="src/main.py",
        old_text="original",
        new_text="modified",
        status="completed",
    )

    # Then: the FileChange carries the opencode_message_id
    assert len(tracker.changes) == 1
    assert tracker.changes[0].message_id == "msg_edit_002"


@pytest.mark.unit
async def test_fallback_to_turn_id_when_opencode_message_id_is_none() -> None:
    """When opencode_message_id is None, fallback to turn_id.

    Given an AgentRunContext with opencode_message_id=None and
    turn_id="uuid-789", record_change should produce a FileChange
    with message_id="uuid-789" (fallback works).
    """
    tracker = FileOpsTracker()
    pool = MagicMock()
    pool.file_ops = tracker

    agent = Agent(name="test_agent", model="test")
    agent.session_id = "session-1"
    agent._pool = pool  # type: ignore[private-type]

    run_ctx = AgentRunContext(
        opencode_message_id=None,
        turn_id="uuid-789",
    )
    ctx = AgentContext(node=agent, run_ctx=run_ctx, pool=pool)

    emitter = StreamEventEmitter(ctx)

    # When: file_edit_progress records a change
    await emitter.file_edit_progress(
        path="src/main.py",
        old_text="old",
        new_text="new",
        status="completed",
    )

    # Then: the FileChange falls back to turn_id
    assert len(tracker.changes) == 1
    assert tracker.changes[0].message_id == "uuid-789"


@pytest.mark.unit
async def test_message_id_none_when_no_run_ctx() -> None:
    """When run_ctx is None, message_id is None (no crash)."""
    tracker = FileOpsTracker()
    pool = MagicMock()
    pool.file_ops = tracker

    agent = Agent(name="test_agent", model="test")
    agent.session_id = "session-1"
    agent._pool = pool  # type: ignore[private-type]

    ctx = AgentContext(node=agent, run_ctx=None, pool=pool)
    emitter = StreamEventEmitter(ctx)

    # When: file_edit_progress records a change without run_ctx
    await emitter.file_edit_progress(
        path="src/main.py",
        old_text="old",
        new_text="new",
        status="completed",
    )

    # Then: the FileChange has message_id=None
    assert len(tracker.changes) == 1
    assert tracker.changes[0].message_id is None
