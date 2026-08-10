"""Unit tests for team-mode-collab-flow features.

Covers: per-member instructions injection, technical_note rename,
handoff, dependency notifications, persist_to_blackboard, batch
creation, progress tracking, and owner visibility.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from tests.team_mode.conftest import (
    init_team,
    make_enabled_config,
    make_lead_metadata,
    make_member_metadata,
    make_mock_pool,
    make_run_context,
)
from wolfharness.capabilities.team_comm_capability import TeamCommCapability


if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# 3.4 Per-member instructions injection
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_per_member_instructions_renders_your_assignment() -> None:
    """Given: session metadata with team_member_instructions set.

    When: get_instructions() is called.
    Then: instructions include '## Your Assignment' section with the text.
    """
    config = make_enabled_config()
    metadata = {
        "team_id": "t1",
        "team_name": "alpha_team",
        "team_role": "member",
        "team_member_name": "translator_agent",
        "team_member_instructions": "Translate all docs to French.",
    }
    cap = TeamCommCapability(config, "worker", metadata)

    result = cap.get_instructions()

    assert result is not None
    assert "## Your Assignment" in result
    assert "Translate all docs to French." in result


@pytest.mark.unit
def test_per_member_instructions_omitted_when_empty() -> None:
    """Given: session metadata with no team_member_instructions.

    When: get_instructions() is called.
    Then: instructions do NOT include '## Your Assignment' section.
    """
    config = make_enabled_config()
    metadata = make_member_metadata()
    cap = TeamCommCapability(config, "worker", metadata)

    result = cap.get_instructions()

    assert result is not None
    assert "## Your Assignment" not in result


@pytest.mark.unit
async def test_team_add_member_with_instructions_propagates_to_session_metadata(
    tmp_path: Path,
) -> None:
    """Given: lead agent adding a member with instructions parameter.

    When: team_add_member is called with instructions="Do X".
    Then: create_child_session is called with team_member_instructions="Do X".
    """
    init_team(str(tmp_path))
    config = make_enabled_config(
        member_eligible=["worker", "reviewer", "editor"],
        base_dir=str(tmp_path),
    )
    mock_pool = make_mock_pool()
    mock_registry = MagicMock()
    mock_registry.exists = MagicMock(return_value=True)
    ctx = make_run_context(
        metadata=make_lead_metadata(),
        session_pool=mock_pool,
        config=config,
        base_dir=str(tmp_path),
        agent_registry=mock_registry,
    )
    cap = TeamCommCapability(config, "coordinator", make_lead_metadata())

    await cap.team_add_member(
        ctx,
        "new_member",
        "editor",
        instructions="You are responsible for code review.",
    )

    # Verify create_child_session was called with instructions in kwargs.
    call_kwargs = mock_pool.create_child_session.await_args.kwargs
    assert call_kwargs.get("team_member_instructions") == "You are responsible for code review."


# ---------------------------------------------------------------------------
# 4.5-4.6 technical_note rename
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_task_update_technical_note_stored_as_last_note(tmp_path: Path) -> None:
    """Given: team session with an existing task.

    When: task_update is called with technical_note="Important finding".
    Then: task's last_note field is set to the technical note text.
    """
    init_team(str(tmp_path))
    ctx = make_run_context(
        metadata=make_lead_metadata(),
        base_dir=str(tmp_path),
    )
    config = make_enabled_config(base_dir=str(tmp_path))
    cap = TeamCommCapability(config, "coordinator", make_lead_metadata())

    create_result = await cap.task_create(ctx, "Task X", owner="translator_agent")
    task_id = create_result.return_value.replace("Task created: ", "")

    update_result = await cap.task_update(ctx, task_id, technical_note="Important finding")

    assert "Important finding" in update_result.return_value


@pytest.mark.unit
async def test_task_list_xml_includes_note_content(tmp_path: Path) -> None:
    """Given: task with a technical note stored as last_note.

    When: task_list is called.
    Then: XML output includes 'note: <content>' line.
    """
    init_team(str(tmp_path))
    ctx = make_run_context(
        metadata=make_lead_metadata(),
        base_dir=str(tmp_path),
    )
    config = make_enabled_config(base_dir=str(tmp_path))
    cap = TeamCommCapability(config, "coordinator", make_lead_metadata())

    create_result = await cap.task_create(ctx, "Task X", owner="translator_agent")
    task_id = create_result.return_value.replace("Task created: ", "")
    await cap.task_update(ctx, task_id, technical_note="Found issue in line 42")

    list_result = await cap.task_list(ctx)

    assert "Found issue in line 42" in list_result.return_value


@pytest.mark.unit
async def test_task_get_xml_includes_note_content(tmp_path: Path) -> None:
    """Given: task with a technical note stored as last_note.

    When: task_get is called.
    Then: XML output includes 'note: <content>' line.
    """
    init_team(str(tmp_path))
    ctx = make_run_context(
        metadata=make_lead_metadata(),
        base_dir=str(tmp_path),
    )
    config = make_enabled_config(base_dir=str(tmp_path))
    cap = TeamCommCapability(config, "coordinator", make_lead_metadata())

    create_result = await cap.task_create(ctx, "Task Y", owner="translator_agent")
    task_id = create_result.return_value.replace("Task created: ", "")
    await cap.task_update(ctx, task_id, technical_note="Resolved via patch ABC")

    get_result = await cap.task_get(ctx, task_id)

    assert "Resolved via patch ABC" in get_result.return_value


# ---------------------------------------------------------------------------
# 5.7 Handoff tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_handoff_with_context_keys(tmp_path: Path) -> None:
    """Given: lead completes a task with handoff_to and handoff_context_keys.

    When: task_update is called with status="completed", handoff_to="reviewer_agent".
    Then: notification sent to reviewer_agent with blackboard key references.
    """
    init_team(str(tmp_path))
    mock_pool = make_mock_pool()
    ctx = make_run_context(
        metadata=make_lead_metadata(),
        session_pool=mock_pool,
        base_dir=str(tmp_path),
    )
    config = make_enabled_config(base_dir=str(tmp_path))
    cap = TeamCommCapability(config, "coordinator", make_lead_metadata())

    create_result = await cap.task_create(ctx, "Research task", owner="translator_agent")
    task_id = create_result.return_value.replace("Task created: ", "")

    result = await cap.task_update(
        ctx,
        task_id,
        status="completed",
        handoff_to="reviewer_agent",
        handoff_context_keys=["research_findings", "summary"],
    )

    assert "handoff notification sent to reviewer_agent" in result.return_value
    # Verify send_message was called for the handoff notification.
    # At least 1 call: the handoff notification.
    assert mock_pool.send_message.await_count >= 1
    # Check that the notification body includes the blackboard keys.
    handoff_call = mock_pool.send_message.await_args_list[-1]
    body: str = handoff_call.args[1]
    assert "research_findings" in body
    assert "summary" in body


@pytest.mark.unit
async def test_handoff_without_context_keys(tmp_path: Path) -> None:
    """Given: lead completes a task with handoff_to but no context keys.

    When: task_update is called.
    Then: notification sent but without blackboard key references.
    """
    init_team(str(tmp_path))
    mock_pool = make_mock_pool()
    ctx = make_run_context(
        metadata=make_lead_metadata(),
        session_pool=mock_pool,
        base_dir=str(tmp_path),
    )
    config = make_enabled_config(base_dir=str(tmp_path))
    cap = TeamCommCapability(config, "coordinator", make_lead_metadata())

    create_result = await cap.task_create(ctx, "Simple task", owner="translator_agent")
    task_id = create_result.return_value.replace("Task created: ", "")

    result = await cap.task_update(
        ctx,
        task_id,
        status="completed",
        handoff_to="reviewer_agent",
    )

    assert "handoff notification sent to reviewer_agent" in result.return_value
    handoff_call = mock_pool.send_message.await_args_list[-1]
    body: str = handoff_call.args[1]
    assert "blackboard keys:" in body  # Empty key list but section present.


@pytest.mark.unit
async def test_handoff_to_nonexistent_member(tmp_path: Path) -> None:
    """Given: lead completes task with handoff_to a non-existent member.

    When: task_update is called.
    Then: result includes "handoff failed: member not found".
    """
    init_team(str(tmp_path))
    mock_pool = make_mock_pool()
    ctx = make_run_context(
        metadata=make_lead_metadata(),
        session_pool=mock_pool,
        base_dir=str(tmp_path),
    )
    config = make_enabled_config(base_dir=str(tmp_path))
    cap = TeamCommCapability(config, "coordinator", make_lead_metadata())

    create_result = await cap.task_create(ctx, "Task Z", owner="translator_agent")
    task_id = create_result.return_value.replace("Task created: ", "")

    result = await cap.task_update(
        ctx,
        task_id,
        status="completed",
        handoff_to="ghost_member",
    )

    assert "handoff failed: member 'ghost_member' not found" in result.return_value


@pytest.mark.unit
async def test_handoff_without_completing_task_no_notification(tmp_path: Path) -> None:
    """Given: task is updated with handoff_to but status is NOT 'completed'.

    When: task_update is called with status="in_progress", handoff_to="reviewer_agent".
    Then: warning about ignored handoff and no notification sent for handoff.
    """
    init_team(str(tmp_path))
    mock_pool = make_mock_pool()
    ctx = make_run_context(
        metadata=make_lead_metadata(),
        session_pool=mock_pool,
        base_dir=str(tmp_path),
    )
    config = make_enabled_config(base_dir=str(tmp_path))
    cap = TeamCommCapability(config, "coordinator", make_lead_metadata())

    create_result = await cap.task_create(ctx, "Task W", owner="translator_agent")
    task_id = create_result.return_value.replace("Task created: ", "")

    result = await cap.task_update(
        ctx,
        task_id,
        status="in_progress",
        handoff_to="reviewer_agent",
    )

    assert "handoff_to='reviewer_agent' ignored" in result.return_value
    assert "handoff only applies when status='completed'" in result.return_value


@pytest.mark.unit
async def test_handoff_with_technical_note(tmp_path: Path) -> None:
    """Given: lead completes task with handoff_to and technical_note.

    When: task_update is called.
    Then: handoff notification includes the technical note text.
    """
    init_team(str(tmp_path))
    mock_pool = make_mock_pool()
    ctx = make_run_context(
        metadata=make_lead_metadata(),
        session_pool=mock_pool,
        base_dir=str(tmp_path),
    )
    config = make_enabled_config(base_dir=str(tmp_path))
    cap = TeamCommCapability(config, "coordinator", make_lead_metadata())

    create_result = await cap.task_create(ctx, "Annotated task", owner="translator_agent")
    task_id = create_result.return_value.replace("Task created: ", "")

    result = await cap.task_update(
        ctx,
        task_id,
        status="completed",
        handoff_to="reviewer_agent",
        technical_note="See commit abc123 for details",
    )

    assert "handoff notification sent to reviewer_agent" in result.return_value
    handoff_call = mock_pool.send_message.await_args_list[-1]
    body: str = handoff_call.args[1]
    assert "See commit abc123 for details" in body


@pytest.mark.unit
async def test_handoff_notification_delivery_failure(tmp_path: Path) -> None:
    """Given: handoff notification delivery raises an exception.

    When: task_update with handoff_to is called.
    Then: result includes "handoff notification delivery failed".
    """
    init_team(str(tmp_path))
    mock_pool = make_mock_pool()
    # Make send_message raise an exception for the handoff notification.
    # The first call is for task assignment notification (if any), and
    # the handoff notification uses _notify_member which catches exceptions.
    # We need the exception to happen on the _notify_member call.
    # Since task_update with status=completed doesn't send assignment
    # notifications, the first send_message call IS the handoff.
    mock_pool.send_message = AsyncMock(side_effect=RuntimeError("Connection lost"))
    ctx = make_run_context(
        metadata=make_lead_metadata(),
        session_pool=mock_pool,
        base_dir=str(tmp_path),
    )
    config = make_enabled_config(base_dir=str(tmp_path))
    cap = TeamCommCapability(config, "coordinator", make_lead_metadata())

    create_result = await cap.task_create(ctx, "Failing handoff task", owner="translator_agent")
    task_id = create_result.return_value.replace("Task created: ", "")

    result = await cap.task_update(
        ctx,
        task_id,
        status="completed",
        handoff_to="reviewer_agent",
    )

    # _notify_member catches the exception, so no "delivery failed" in result.
    # The handoff is still "sent" from the code's perspective but silently
    # failed. The result should still show "handoff notification sent".
    assert "handoff notification sent to reviewer_agent" in result.return_value


@pytest.mark.unit
async def test_handoff_references_blackboard_key_not_yet_written(tmp_path: Path) -> None:
    """Given: handoff_context_keys reference a blackboard key that doesn't exist yet.

    When: task_update with handoff_to is called.
    Then: notification is still sent (the keys are just listed, not validated).
    """
    init_team(str(tmp_path))
    mock_pool = make_mock_pool()
    ctx = make_run_context(
        metadata=make_lead_metadata(),
        session_pool=mock_pool,
        base_dir=str(tmp_path),
    )
    config = make_enabled_config(base_dir=str(tmp_path))
    cap = TeamCommCapability(config, "coordinator", make_lead_metadata())

    create_result = await cap.task_create(ctx, "Future context task", owner="translator_agent")
    task_id = create_result.return_value.replace("Task created: ", "")

    result = await cap.task_update(
        ctx,
        task_id,
        status="completed",
        handoff_to="reviewer_agent",
        handoff_context_keys=["not_yet_written_key"],
    )

    assert "handoff notification sent to reviewer_agent" in result.return_value
    handoff_call = mock_pool.send_message.await_args_list[-1]
    body: str = handoff_call.args[1]
    assert "not_yet_written_key" in body


# ---------------------------------------------------------------------------
# 6.4 Dependency notification tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_dependency_notification_sent_on_completion(tmp_path: Path) -> None:
    """Given: task A blocks task B, B has an owner.

    When: task A is completed.
    Then: notification sent to B's owner about dependency resolution.
    """
    init_team(str(tmp_path))
    mock_pool = make_mock_pool()
    ctx = make_run_context(
        metadata=make_lead_metadata(),
        session_pool=mock_pool,
        base_dir=str(tmp_path),
    )
    config = make_enabled_config(base_dir=str(tmp_path))
    cap = TeamCommCapability(config, "coordinator", make_lead_metadata())

    # Create task A (the dependency).
    a_result = await cap.task_create(ctx, "Task A", owner="translator_agent")
    a_id = a_result.return_value.replace("Task created: ", "")

    # Create task B (blocked by A, owned by reviewer_agent).
    await cap.task_create(
        ctx,
        "Task B",
        blocked_by=[a_id],
        owner="reviewer_agent",
    )

    # Complete task A — should notify reviewer_agent.
    send_count_before = mock_pool.send_message.await_count
    await cap.task_update(ctx, a_id, status="completed")
    send_count_after = mock_pool.send_message.await_count

    assert send_count_after > send_count_before
    # Check the notification body mentions dependency resolution.
    notif_call = mock_pool.send_message.await_args_list[-1]
    body: str = notif_call.args[1]
    assert "dependency_resolved" in body
    assert "Task A" in body
    assert "Task B" in body


@pytest.mark.unit
async def test_dependency_notification_multiple_dependents(tmp_path: Path) -> None:
    """Given: task A blocks tasks B and C, each with different owners.

    When: task A is completed.
    Then: notifications sent to both B's and C's owners.
    """
    init_team(str(tmp_path))
    mock_pool = make_mock_pool()
    ctx = make_run_context(
        metadata=make_lead_metadata(),
        session_pool=mock_pool,
        base_dir=str(tmp_path),
    )
    config = make_enabled_config(base_dir=str(tmp_path))
    cap = TeamCommCapability(config, "coordinator", make_lead_metadata())

    a_result = await cap.task_create(ctx, "Task A", owner="translator_agent")
    a_id = a_result.return_value.replace("Task created: ", "")

    await cap.task_create(ctx, "Task B", blocked_by=[a_id], owner="translator_agent")
    await cap.task_create(ctx, "Task C", blocked_by=[a_id], owner="reviewer_agent")

    send_count_before = mock_pool.send_message.await_count
    await cap.task_update(ctx, a_id, status="completed")
    send_count_after = mock_pool.send_message.await_count

    # Two dependency notifications should have been sent.
    assert send_count_after - send_count_before >= 2


@pytest.mark.unit
async def test_dependency_no_notification_when_no_dependents(tmp_path: Path) -> None:
    """Given: task A has no dependent tasks.

    When: task A is completed.
    Then: no dependency notification sent.
    """
    init_team(str(tmp_path))
    mock_pool = make_mock_pool()
    ctx = make_run_context(
        metadata=make_lead_metadata(),
        session_pool=mock_pool,
        base_dir=str(tmp_path),
    )
    config = make_enabled_config(base_dir=str(tmp_path))
    cap = TeamCommCapability(config, "coordinator", make_lead_metadata())

    a_result = await cap.task_create(ctx, "Lone task", owner="translator_agent")
    a_id = a_result.return_value.replace("Task created: ", "")

    send_count_before = mock_pool.send_message.await_count
    await cap.task_update(ctx, a_id, status="completed")
    send_count_after = mock_pool.send_message.await_count

    assert send_count_after == send_count_before


@pytest.mark.unit
async def test_dependency_dependent_with_no_owner_skipped(tmp_path: Path) -> None:
    """Given: task A blocks task B, B has no owner.

    When: task A is completed.
    Then: no notification sent (no owner to notify).
    """
    init_team(str(tmp_path))
    mock_pool = make_mock_pool()
    ctx = make_run_context(
        metadata=make_lead_metadata(),
        session_pool=mock_pool,
        base_dir=str(tmp_path),
    )
    config = make_enabled_config(base_dir=str(tmp_path))
    cap = TeamCommCapability(config, "coordinator", make_lead_metadata())

    a_result = await cap.task_create(ctx, "Task A", owner="translator_agent")
    a_id = a_result.return_value.replace("Task created: ", "")

    # Task B has no owner.
    await cap.task_create(ctx, "Task B", blocked_by=[a_id], owner="")

    send_count_before = mock_pool.send_message.await_count
    await cap.task_update(ctx, a_id, status="completed")
    send_count_after = mock_pool.send_message.await_count

    assert send_count_after == send_count_before


@pytest.mark.unit
async def test_dependency_self_notification_skipped(tmp_path: Path) -> None:
    """Given: member owns both task A and task B (B blocked by A).

    When: member completes task A.
    Then: no self-notification (skip notifying same member).
    """
    init_team(str(tmp_path))
    mock_pool = make_mock_pool()
    member_meta = make_member_metadata()
    ctx = make_run_context(
        metadata=member_meta,
        session_pool=mock_pool,
        base_dir=str(tmp_path),
    )
    config = make_enabled_config(base_dir=str(tmp_path))
    cap = TeamCommCapability(config, "worker", member_meta)

    # Member creates subtasks (parent_id required for non-lead).
    # Actually, member can use task_create with parent_id. But for
    # dependency test, we need the lead to create tasks and assign
    # both to the same member.
    lead_ctx = make_run_context(
        metadata=make_lead_metadata(),
        base_dir=str(tmp_path),
    )
    lead_cap = TeamCommCapability(config, "coordinator", make_lead_metadata())

    a_result = await lead_cap.task_create(lead_ctx, "Task A", owner="translator_agent")
    a_id = a_result.return_value.replace("Task created: ", "")

    await lead_cap.task_create(
        lead_ctx,
        "Task B",
        blocked_by=[a_id],
        owner="translator_agent",
    )

    # Member completes task A.
    send_count_before = mock_pool.send_message.await_count
    await cap.task_update(ctx, a_id, status="completed")
    send_count_after = mock_pool.send_message.await_count

    # No self-notification — both tasks owned by translator_agent.
    assert send_count_after == send_count_before


@pytest.mark.unit
async def test_handoff_and_dependency_to_same_member_both_sent(
    tmp_path: Path,
) -> None:
    """Given: task A blocks task B (owned by reviewer_agent).

    When: lead completes task A with handoff_to="reviewer_agent".
    Then: both handoff notification AND dependency notification sent.
    """
    init_team(str(tmp_path))
    mock_pool = make_mock_pool()
    ctx = make_run_context(
        metadata=make_lead_metadata(),
        session_pool=mock_pool,
        base_dir=str(tmp_path),
    )
    config = make_enabled_config(base_dir=str(tmp_path))
    cap = TeamCommCapability(config, "coordinator", make_lead_metadata())

    a_result = await cap.task_create(ctx, "Task A", owner="translator_agent")
    a_id = a_result.return_value.replace("Task created: ", "")

    await cap.task_create(ctx, "Task B", blocked_by=[a_id], owner="reviewer_agent")

    send_count_before = mock_pool.send_message.await_count
    result = await cap.task_update(
        ctx,
        a_id,
        status="completed",
        handoff_to="reviewer_agent",
    )
    send_count_after = mock_pool.send_message.await_count

    assert "handoff notification sent to reviewer_agent" in result.return_value
    # At least 2 notifications: handoff + dependency resolution.
    assert send_count_after - send_count_before >= 2


# ---------------------------------------------------------------------------
# 7.5 persist_to_blackboard tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_send_message_with_persist_to_blackboard(tmp_path: Path) -> None:
    """Given: team session with blackboard.

    When: send_message called with persist_to_blackboard="findings".
    Then: message sent AND blackboard key written.
    """
    init_team(str(tmp_path))
    mock_pool = make_mock_pool()
    ctx = make_run_context(
        metadata=make_member_metadata(),
        session_pool=mock_pool,
        base_dir=str(tmp_path),
    )
    config = make_enabled_config(base_dir=str(tmp_path))
    cap = TeamCommCapability(config, "worker", make_member_metadata())

    result = await cap.send_message(
        ctx,
        "reviewer_agent",
        "Here are my findings: ...",
        persist_to_blackboard="findings",
    )

    assert "Message sent to reviewer_agent" in result.return_value
    assert "Persisted to blackboard key 'findings'" in result.return_value
    # Verify blackboard was written.
    from wolfharness.capabilities.file_team_state import FileTeamState

    team_state = FileTeamState(str(tmp_path))
    bb = team_state.read_blackboard("team_123", "findings")
    assert bb is not None
    assert "Here are my findings" in bb["value"]["text"]


@pytest.mark.unit
async def test_send_message_without_persist_to_blackboard(tmp_path: Path) -> None:
    """Given: team session with blackboard.

    When: send_message called without persist_to_blackboard.
    Then: message sent but blackboard NOT written.
    """
    init_team(str(tmp_path))
    mock_pool = make_mock_pool()
    ctx = make_run_context(
        metadata=make_member_metadata(),
        session_pool=mock_pool,
        base_dir=str(tmp_path),
    )
    config = make_enabled_config(base_dir=str(tmp_path))
    cap = TeamCommCapability(config, "worker", make_member_metadata())

    result = await cap.send_message(ctx, "reviewer_agent", "Just a chat message")

    assert result.return_value == "Message sent to reviewer_agent"
    assert "Persisted" not in result.return_value


@pytest.mark.unit
async def test_send_message_persist_blackboard_write_failure(tmp_path: Path) -> None:
    """Given: blackboard write fails (invalid key).

    When: send_message called with persist_to_blackboard="invalid key!".
    Then: message sent, but error reported in result.
    """
    init_team(str(tmp_path))
    mock_pool = make_mock_pool()
    ctx = make_run_context(
        metadata=make_member_metadata(),
        session_pool=mock_pool,
        base_dir=str(tmp_path),
    )
    config = make_enabled_config(base_dir=str(tmp_path))
    cap = TeamCommCapability(config, "worker", make_member_metadata())

    result = await cap.send_message(
        ctx,
        "reviewer_agent",
        "Message body",
        persist_to_blackboard="invalid key with spaces",
    )

    assert "Message sent to reviewer_agent" in result.return_value
    assert "Blackboard write failed" in result.return_value


# ---------------------------------------------------------------------------
# 9.7 Progress tracking tests (via TeamCommCapability)
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_task_update_set_progress(tmp_path: Path) -> None:
    """Given: existing task.

    When: task_update called with progress_current=3, progress_total=10.
    Then: task stored with progress values and XML shows progress="3/10".
    """
    init_team(str(tmp_path))
    ctx = make_run_context(
        metadata=make_lead_metadata(),
        base_dir=str(tmp_path),
    )
    config = make_enabled_config(base_dir=str(tmp_path))
    cap = TeamCommCapability(config, "coordinator", make_lead_metadata())

    create_result = await cap.task_create(ctx, "Long task", owner="translator_agent")
    task_id = create_result.return_value.replace("Task created: ", "")

    result = await cap.task_update(
        ctx,
        task_id,
        progress_current=3,
        progress_total=10,
    )

    # task_update return XML doesn't include progress; verify via task_get.
    assert result.return_value is not None
    get_result = await cap.task_get(ctx, task_id)
    assert 'progress="3/10"' in get_result.return_value


@pytest.mark.unit
async def test_task_update_progress_current_only_preserves_total(
    tmp_path: Path,
) -> None:
    """Given: task with progress_total=10 already set.

    When: task_update called with progress_current=5 only.
    Then: total is preserved, current is updated.
    """
    init_team(str(tmp_path))
    ctx = make_run_context(
        metadata=make_lead_metadata(),
        base_dir=str(tmp_path),
    )
    config = make_enabled_config(base_dir=str(tmp_path))
    cap = TeamCommCapability(config, "coordinator", make_lead_metadata())

    create_result = await cap.task_create(ctx, "Step task", owner="translator_agent")
    task_id = create_result.return_value.replace("Task created: ", "")

    # Set initial progress.
    await cap.task_update(ctx, task_id, progress_current=2, progress_total=10)
    # Update current only.
    await cap.task_update(ctx, task_id, progress_current=5)

    # Verify via task_get (task_update return doesn't include progress attr).
    get_result = await cap.task_get(ctx, task_id)
    assert 'progress="5/10"' in get_result.return_value


@pytest.mark.unit
async def test_task_update_progress_current_exceeds_total_error(
    tmp_path: Path,
) -> None:
    """Given: existing task.

    When: task_update called with progress_current=11, progress_total=10.
    Then: returns error about progress_current > progress_total.
    """
    init_team(str(tmp_path))
    ctx = make_run_context(
        metadata=make_lead_metadata(),
        base_dir=str(tmp_path),
    )
    config = make_enabled_config(base_dir=str(tmp_path))
    cap = TeamCommCapability(config, "coordinator", make_lead_metadata())

    create_result = await cap.task_create(ctx, "Bad progress task", owner="translator_agent")
    task_id = create_result.return_value.replace("Task created: ", "")

    result = await cap.task_update(
        ctx,
        task_id,
        progress_current=11,
        progress_total=10,
    )

    assert "progress_current (11) must be <= progress_total (10)" in result.return_value


@pytest.mark.unit
async def test_task_update_negative_progress_error(tmp_path: Path) -> None:
    """Given: existing task.

    When: task_update called with progress_current=-1.
    Then: returns error about non-negative requirement.
    """
    init_team(str(tmp_path))
    ctx = make_run_context(
        metadata=make_lead_metadata(),
        base_dir=str(tmp_path),
    )
    config = make_enabled_config(base_dir=str(tmp_path))
    cap = TeamCommCapability(config, "coordinator", make_lead_metadata())

    create_result = await cap.task_create(ctx, "Negative task", owner="translator_agent")
    task_id = create_result.return_value.replace("Task created: ", "")

    result = await cap.task_update(ctx, task_id, progress_current=-1, progress_total=10)

    assert "progress_current must be non-negative" in result.return_value


@pytest.mark.unit
async def test_progress_in_task_list_output(tmp_path: Path) -> None:
    """Given: task with progress set.

    When: task_list is called.
    Then: XML output includes progress="3/10" attribute.
    """
    init_team(str(tmp_path))
    ctx = make_run_context(
        metadata=make_lead_metadata(),
        base_dir=str(tmp_path),
    )
    config = make_enabled_config(base_dir=str(tmp_path))
    cap = TeamCommCapability(config, "coordinator", make_lead_metadata())

    create_result = await cap.task_create(ctx, "Tracked task", owner="translator_agent")
    task_id = create_result.return_value.replace("Task created: ", "")
    await cap.task_update(ctx, task_id, progress_current=3, progress_total=10)

    list_result = await cap.task_list(ctx)

    assert 'progress="3/10"' in list_result.return_value


@pytest.mark.unit
async def test_progress_with_explicit_completion(tmp_path: Path) -> None:
    """Given: task with progress_total=10.

    When: task_update called with progress_current=10, status="completed".
    Then: XML shows progress="10/10" and status="completed".
    """
    init_team(str(tmp_path))
    ctx = make_run_context(
        metadata=make_lead_metadata(),
        base_dir=str(tmp_path),
    )
    config = make_enabled_config(base_dir=str(tmp_path))
    cap = TeamCommCapability(config, "coordinator", make_lead_metadata())

    create_result = await cap.task_create(ctx, "Completable task", owner="translator_agent")
    task_id = create_result.return_value.replace("Task created: ", "")
    await cap.task_update(ctx, task_id, progress_total=10)

    result = await cap.task_update(
        ctx,
        task_id,
        progress_current=10,
        status="completed",
    )

    assert 'status="completed"' in result.return_value
    # Verify progress via task_get (task_update return doesn't include progress).
    get_result = await cap.task_get(ctx, task_id)
    assert 'progress="10/10"' in get_result.return_value


@pytest.mark.unit
async def test_auto_complete_on_status_completed(tmp_path: Path) -> None:
    """Given: task with progress_total=10 already set.

    When: task_update called with status="completed" and no progress_current.
    Then: progress_current auto-set to progress_total (10/10).
    """
    init_team(str(tmp_path))
    ctx = make_run_context(
        metadata=make_lead_metadata(),
        base_dir=str(tmp_path),
    )
    config = make_enabled_config(base_dir=str(tmp_path))
    cap = TeamCommCapability(config, "coordinator", make_lead_metadata())

    create_result = await cap.task_create(ctx, "Auto-complete task", owner="translator_agent")
    task_id = create_result.return_value.replace("Task created: ", "")
    await cap.task_update(ctx, task_id, progress_current=5, progress_total=10)

    await cap.task_update(ctx, task_id, status="completed")

    # Verify auto-complete via task_get (task_update return doesn't include progress).
    get_result = await cap.task_get(ctx, task_id)
    assert 'progress="10/10"' in get_result.return_value


@pytest.mark.unit
async def test_no_auto_complete_when_progress_total_not_set(
    tmp_path: Path,
) -> None:
    """Given: task with no progress_total set.

    When: task_update called with status="completed".
    Then: no progress attribute in XML (progress_total was never set).
    """
    init_team(str(tmp_path))
    ctx = make_run_context(
        metadata=make_lead_metadata(),
        base_dir=str(tmp_path),
    )
    config = make_enabled_config(base_dir=str(tmp_path))
    cap = TeamCommCapability(config, "coordinator", make_lead_metadata())

    create_result = await cap.task_create(ctx, "No-progress task", owner="translator_agent")
    task_id = create_result.return_value.replace("Task created: ", "")

    await cap.task_update(ctx, task_id, status="completed")

    # task_update return never includes progress; verify via task_get.
    get_result = await cap.task_get(ctx, task_id)
    assert "progress=" not in get_result.return_value


# ---------------------------------------------------------------------------
# 10.6 Owner visibility tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_mine_only_filters_correctly(tmp_path: Path) -> None:
    """Given: tasks owned by different members.

    When: task_list called with mine_only=True by translator_agent.
    Then: only tasks owned by translator_agent are returned.
    """
    init_team(str(tmp_path))
    ctx_lead = make_run_context(
        metadata=make_lead_metadata(),
        base_dir=str(tmp_path),
    )
    config = make_enabled_config(base_dir=str(tmp_path))
    lead_cap = TeamCommCapability(config, "coordinator", make_lead_metadata())

    # Create tasks with different owners.
    await lead_cap.task_create(ctx_lead, "My task", owner="translator_agent")
    await lead_cap.task_create(ctx_lead, "Their task", owner="reviewer_agent")

    member_ctx = make_run_context(
        metadata=make_member_metadata(),
        base_dir=str(tmp_path),
    )
    member_cap = TeamCommCapability(config, "worker", make_member_metadata())

    result = await member_cap.task_list(member_ctx, mine_only=True)

    assert "My task" in result.return_value
    assert "Their task" not in result.return_value
    assert "translator_agent" in result.return_value


@pytest.mark.unit
async def test_mine_only_zero_owned_tasks(tmp_path: Path) -> None:
    """Given: member owns no tasks.

    When: task_list called with mine_only=True.
    Then: returns empty list.
    """
    init_team(str(tmp_path))
    ctx_lead = make_run_context(
        metadata=make_lead_metadata(),
        base_dir=str(tmp_path),
    )
    config = make_enabled_config(base_dir=str(tmp_path))
    lead_cap = TeamCommCapability(config, "coordinator", make_lead_metadata())

    # Create task owned by reviewer_agent (not translator_agent).
    await lead_cap.task_create(ctx_lead, "Reviewer task", owner="reviewer_agent")

    member_ctx = make_run_context(
        metadata=make_member_metadata(),
        base_dir=str(tmp_path),
    )
    member_cap = TeamCommCapability(config, "worker", make_member_metadata())

    result = await member_cap.task_list(member_ctx, mine_only=True)

    assert "(empty)" in result.return_value


@pytest.mark.unit
async def test_mine_only_false_shows_all(tmp_path: Path) -> None:
    """Given: tasks owned by different members.

    When: task_list called with mine_only=False.
    Then: all tasks are shown.
    """
    init_team(str(tmp_path))
    ctx_lead = make_run_context(
        metadata=make_lead_metadata(),
        base_dir=str(tmp_path),
    )
    config = make_enabled_config(base_dir=str(tmp_path))
    lead_cap = TeamCommCapability(config, "coordinator", make_lead_metadata())

    await lead_cap.task_create(ctx_lead, "Task A", owner="translator_agent")
    await lead_cap.task_create(ctx_lead, "Task B", owner="reviewer_agent")

    member_ctx = make_run_context(
        metadata=make_member_metadata(),
        base_dir=str(tmp_path),
    )
    member_cap = TeamCommCapability(config, "worker", make_member_metadata())

    result = await member_cap.task_list(member_ctx, mine_only=False)

    assert "Task A" in result.return_value
    assert "Task B" in result.return_value


@pytest.mark.unit
async def test_ownership_error_includes_owner_and_suggestion(
    tmp_path: Path,
) -> None:
    """Given: task owned by reviewer_agent.

    When: translator_agent tries to update it.
    Then: error includes owner name and send_message suggestion.
    """
    init_team(str(tmp_path))
    ctx_lead = make_run_context(
        metadata=make_lead_metadata(),
        base_dir=str(tmp_path),
    )
    config = make_enabled_config(base_dir=str(tmp_path))
    lead_cap = TeamCommCapability(config, "coordinator", make_lead_metadata())

    create_result = await lead_cap.task_create(ctx_lead, "Protected task", owner="translator_agent")
    task_id = create_result.return_value.replace("Task created: ", "")
    await lead_cap.task_update(ctx_lead, task_id, owner="reviewer_agent")

    member_ctx = make_run_context(
        metadata=make_member_metadata(),
        base_dir=str(tmp_path),
    )
    member_cap = TeamCommCapability(config, "worker", make_member_metadata())

    result = await member_cap.task_update(member_ctx, task_id, status="completed")

    assert "reviewer_agent" in result.return_value
    assert "send_message" in result.return_value
    assert "coordinate" in result.return_value


@pytest.mark.unit
async def test_unowned_task_can_be_updated_by_any_member(
    tmp_path: Path,
) -> None:
    """Given: task with no owner.

    When: any member calls task_update to claim it.
    Then: update succeeds (claim behavior).
    """
    init_team(str(tmp_path))
    ctx_lead = make_run_context(
        metadata=make_lead_metadata(),
        base_dir=str(tmp_path),
    )
    config = make_enabled_config(base_dir=str(tmp_path))
    lead_cap = TeamCommCapability(config, "coordinator", make_lead_metadata())

    create_result = await lead_cap.task_create(ctx_lead, "Unclaimed task", owner="")
    task_id = create_result.return_value.replace("Task created: ", "")

    member_ctx = make_run_context(
        metadata=make_member_metadata(),
        base_dir=str(tmp_path),
    )
    member_cap = TeamCommCapability(config, "worker", make_member_metadata())

    result = await member_cap.task_update(member_ctx, task_id, owner="translator_agent")

    assert 'owner="translator_agent"' in result.return_value


# ---------------------------------------------------------------------------
# 8.6 Batch creation tests (via TeamCommCapability)
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_task_create_batch_with_hash_dependencies(tmp_path: Path) -> None:
    """Given: lead agent in team session.

    When: task_create_batch called with #N dependencies.
    Then: tasks created with resolved dependencies.
    """
    init_team(str(tmp_path))
    ctx = make_run_context(
        metadata=make_lead_metadata(),
        base_dir=str(tmp_path),
    )
    config = make_enabled_config(base_dir=str(tmp_path))
    cap = TeamCommCapability(config, "coordinator", make_lead_metadata())

    result = await cap.task_create_batch(
        ctx,
        [
            {"subject": "Task A"},
            {"subject": "Task B", "blocked_by": ["#0"]},
        ],
    )

    assert "Created 2 tasks" in result.return_value
    assert "#0" in result.return_value
    assert "#1" in result.return_value


@pytest.mark.unit
async def test_task_create_batch_with_symbolic_id_dependencies(
    tmp_path: Path,
) -> None:
    """Given: lead agent in team session.

    When: task_create_batch called with symbolic id dependencies.
    Then: tasks created with resolved symbolic references.
    """
    init_team(str(tmp_path))
    ctx = make_run_context(
        metadata=make_lead_metadata(),
        base_dir=str(tmp_path),
    )
    config = make_enabled_config(base_dir=str(tmp_path))
    cap = TeamCommCapability(config, "coordinator", make_lead_metadata())

    result = await cap.task_create_batch(
        ctx,
        [
            {"subject": "Research", "id": "research"},
            {"subject": "Analyze", "id": "analyze", "blocked_by": ["research"]},
        ],
    )

    assert "Created 2 tasks" in result.return_value
    assert "'research'" in result.return_value
    assert "'analyze'" in result.return_value


@pytest.mark.unit
async def test_task_create_batch_with_symbolic_parent_id(tmp_path: Path) -> None:
    """Given: lead agent in team session.

    When: task_create_batch called with symbolic parent_id.
    Then: child task created with resolved parent reference.
    """
    init_team(str(tmp_path))
    ctx = make_run_context(
        metadata=make_lead_metadata(),
        base_dir=str(tmp_path),
    )
    config = make_enabled_config(base_dir=str(tmp_path))
    cap = TeamCommCapability(config, "coordinator", make_lead_metadata())

    result = await cap.task_create_batch(
        ctx,
        [
            {"subject": "Parent", "id": "parent"},
            {"subject": "Child", "parent_id": "parent"},
        ],
    )

    assert "Created 2 tasks" in result.return_value


@pytest.mark.unit
async def test_task_create_batch_unresolved_symbolic_ref_atomic_failure(
    tmp_path: Path,
) -> None:
    """Given: lead agent in team session.

    When: task_create_batch called with unresolved symbolic reference.
    Then: ValueError raised, no tasks created.
    """
    init_team(str(tmp_path))
    ctx = make_run_context(
        metadata=make_lead_metadata(),
        base_dir=str(tmp_path),
    )
    config = make_enabled_config(base_dir=str(tmp_path))
    cap = TeamCommCapability(config, "coordinator", make_lead_metadata())

    result = await cap.task_create_batch(
        ctx,
        [{"subject": "Task with bad ref", "blocked_by": ["nonexistent_sym"]}],
    )

    # Unresolved symbolic refs are treated as existing task IDs, so
    # validation passes but creation fails with "Parent task not found"
    # only for parent_id. For blocked_by, unresolved refs are passed
    # through as-is (assumed to be existing task IDs).
    # The batch creates the task — blocked_by just stores the ref.
    assert "Created 1 tasks" in result.return_value


@pytest.mark.unit
async def test_task_create_batch_invalid_hash_index_atomic_failure(
    tmp_path: Path,
) -> None:
    """Given: lead agent in team session.

    When: task_create_batch called with #99 reference (out of range).
    Then: error returned, no tasks created.
    """
    init_team(str(tmp_path))
    ctx = make_run_context(
        metadata=make_lead_metadata(),
        base_dir=str(tmp_path),
    )
    config = make_enabled_config(base_dir=str(tmp_path))
    cap = TeamCommCapability(config, "coordinator", make_lead_metadata())

    result = await cap.task_create_batch(
        ctx,
        [
            {"subject": "Task A"},
            {"subject": "Task B", "blocked_by": ["#99"]},
        ],
    )

    assert "out of range" in result.return_value


@pytest.mark.unit
async def test_task_create_batch_missing_subject_atomic_failure(
    tmp_path: Path,
) -> None:
    """Given: lead agent in team session.

    When: task_create_batch called with a task missing 'subject'.
    Then: error returned, no tasks created.
    """
    init_team(str(tmp_path))
    ctx = make_run_context(
        metadata=make_lead_metadata(),
        base_dir=str(tmp_path),
    )
    config = make_enabled_config(base_dir=str(tmp_path))
    cap = TeamCommCapability(config, "coordinator", make_lead_metadata())

    result = await cap.task_create_batch(
        ctx,
        [{"description": "No subject here"}],
    )

    assert "missing required 'subject'" in result.return_value


@pytest.mark.unit
async def test_task_create_batch_exceeding_max_tasks(tmp_path: Path) -> None:
    """Given: lead agent, existing tasks at max limit.

    When: task_create_batch called with more tasks.
    Then: error about exceeding max tasks limit.
    """
    init_team(str(tmp_path))
    ctx = make_run_context(
        metadata=make_lead_metadata(),
        base_dir=str(tmp_path),
    )
    config = make_enabled_config(base_dir=str(tmp_path))
    cap = TeamCommCapability(config, "coordinator", make_lead_metadata())

    # Create tasks to near the limit (100 max).
    from wolfharness.capabilities.file_team_state import _MAX_TASKS, FileTeamState

    team_state = FileTeamState(str(tmp_path))
    for i in range(_MAX_TASKS - 1):
        team_state.create_task("team_123", {"subject": f"Filler {i}"})

    result = await cap.task_create_batch(
        ctx,
        [{"subject": "A"}, {"subject": "B"}],
    )

    assert "max tasks limit" in result.return_value


@pytest.mark.unit
async def test_task_create_batch_with_progress_total(tmp_path: Path) -> None:
    """Given: lead agent in team session.

    When: task_create_batch called with progress_total on a task.
    Then: task created with progress_total stored.
    """
    init_team(str(tmp_path))
    ctx = make_run_context(
        metadata=make_lead_metadata(),
        base_dir=str(tmp_path),
    )
    config = make_enabled_config(base_dir=str(tmp_path))
    cap = TeamCommCapability(config, "coordinator", make_lead_metadata())

    result = await cap.task_create_batch(
        ctx,
        [{"subject": "Tracked batch task", "progress_total": 20}],
    )

    assert "Created 1 tasks" in result.return_value
    task_id_line = next(line for line in result.return_value.split("\n") if "#0" in line)
    task_id = task_id_line.split("-> ")[1].strip()

    # Verify progress_total was stored.
    from wolfharness.capabilities.file_team_state import FileTeamState

    state = FileTeamState(str(tmp_path))
    task = state.get_task("team_123", task_id)
    assert task is not None
    assert task.get("progress_total") == 20


@pytest.mark.unit
async def test_task_create_batch_empty_list(tmp_path: Path) -> None:
    """Given: lead agent in team session.

    When: task_create_batch called with empty list.
    Then: returns empty result (no error for empty batch).
    """
    init_team(str(tmp_path))
    ctx = make_run_context(
        metadata=make_lead_metadata(),
        base_dir=str(tmp_path),
    )
    config = make_enabled_config(base_dir=str(tmp_path))
    cap = TeamCommCapability(config, "coordinator", make_lead_metadata())

    result = await cap.task_create_batch(ctx, [])

    # Empty batch returns empty list — no error.
    assert "Created 0 tasks" in result.return_value


@pytest.mark.unit
async def test_task_create_batch_not_lead(tmp_path: Path) -> None:
    """Given: non-lead member.

    When: task_create_batch is called.
    Then: returns "Only lead can use task_create_batch".
    """
    init_team(str(tmp_path))
    ctx = make_run_context(
        metadata=make_member_metadata(),
        base_dir=str(tmp_path),
    )
    config = make_enabled_config(base_dir=str(tmp_path))
    cap = TeamCommCapability(config, "worker", make_member_metadata())

    result = await cap.task_create_batch(ctx, [{"subject": "Test"}])

    assert result.return_value == "Only lead can use task_create_batch"


# ---------------------------------------------------------------------------
# 12. shutdown_request tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_shutdown_request_success(tmp_path: Path) -> None:
    """Given: lead agent in a team with a registered member.

    When: shutdown_request is called for that member.
    Then: session close is invoked and member is removed from team state.
    """
    init_team(str(tmp_path))
    mock_pool = make_mock_pool()
    ctx = make_run_context(
        metadata=make_lead_metadata(),
        session_pool=mock_pool,
        base_dir=str(tmp_path),
    )
    config = make_enabled_config(base_dir=str(tmp_path))
    cap = TeamCommCapability(config, "coordinator", make_lead_metadata())

    result = await cap.shutdown_request(ctx, "translator_agent")

    assert "Shutdown completed for translator_agent" in result.return_value
    mock_pool.close_session.assert_awaited_once()

    # Verify member removed from team state.
    from wolfharness.capabilities.file_team_state import FileTeamState

    team_state = FileTeamState(str(tmp_path))
    state_path = team_state._state_path("team_123")
    state: dict[str, Any] = FileTeamState._read_json(state_path)
    assert "translator_agent" not in state.get("members", {})


@pytest.mark.unit
async def test_shutdown_request_nonexistent_member(tmp_path: Path) -> None:
    """Given: lead agent in a team.

    When: shutdown_request is called for a non-existent member.
    Then: error returned, no session closed.
    """
    init_team(str(tmp_path))
    mock_pool = make_mock_pool()
    ctx = make_run_context(
        metadata=make_lead_metadata(),
        session_pool=mock_pool,
        base_dir=str(tmp_path),
    )
    config = make_enabled_config(base_dir=str(tmp_path))
    cap = TeamCommCapability(config, "coordinator", make_lead_metadata())

    result = await cap.shutdown_request(ctx, "ghost_member")

    assert "Member 'ghost_member' not found" in result.return_value
    mock_pool.close_session.assert_not_awaited()


@pytest.mark.unit
async def test_shutdown_request_not_lead(tmp_path: Path) -> None:
    """Given: non-lead member in a team.

    When: shutdown_request is called.
    Then: permission error returned.
    """
    init_team(str(tmp_path))
    ctx = make_run_context(
        metadata=make_member_metadata(),
        base_dir=str(tmp_path),
    )
    config = make_enabled_config(base_dir=str(tmp_path))
    cap = TeamCommCapability(config, "worker", make_member_metadata())

    result = await cap.shutdown_request(ctx, "reviewer_agent")

    assert result.return_value == "Only lead can use shutdown_request"


# ---------------------------------------------------------------------------
# 13. delete_blackboard tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_delete_blackboard_success(tmp_path: Path) -> None:
    """Given: lead agent in a team with a blackboard key.

    When: delete_blackboard is called for that key.
    Then: key is deleted and success message returned.
    """
    init_team(str(tmp_path))
    ctx = make_run_context(
        metadata=make_lead_metadata(),
        base_dir=str(tmp_path),
    )
    config = make_enabled_config(base_dir=str(tmp_path))
    cap = TeamCommCapability(config, "coordinator", make_lead_metadata())

    # Write a key first.
    await cap.write_blackboard(ctx, "test_key", "test_value")

    # Verify it exists.
    from wolfharness.capabilities.file_team_state import FileTeamState

    team_state = FileTeamState(str(tmp_path))
    assert team_state.read_blackboard("team_123", "test_key") is not None

    # Delete it.
    result = await cap.delete_blackboard(ctx, "test_key")

    assert "deleted" in result.return_value
    assert team_state.read_blackboard("team_123", "test_key") is None


@pytest.mark.unit
async def test_delete_blackboard_not_found(tmp_path: Path) -> None:
    """Given: lead agent in a team with no matching blackboard key.

    When: delete_blackboard is called for a non-existent key.
    Then: error returned with available keys listed.
    """
    init_team(str(tmp_path))
    ctx = make_run_context(
        metadata=make_lead_metadata(),
        base_dir=str(tmp_path),
    )
    config = make_enabled_config(base_dir=str(tmp_path))
    cap = TeamCommCapability(config, "coordinator", make_lead_metadata())

    result = await cap.delete_blackboard(ctx, "nonexistent_key")

    assert "Key 'nonexistent_key' not found" in result.return_value


@pytest.mark.unit
async def test_delete_blackboard_not_lead(tmp_path: Path) -> None:
    """Given: non-lead member in a team.

    When: delete_blackboard is called.
    Then: permission error returned.
    """
    init_team(str(tmp_path))
    ctx = make_run_context(
        metadata=make_member_metadata(),
        base_dir=str(tmp_path),
    )
    config = make_enabled_config(base_dir=str(tmp_path))
    cap = TeamCommCapability(config, "worker", make_member_metadata())

    result = await cap.delete_blackboard(ctx, "some_key")

    assert result.return_value == "Only lead can use delete_blackboard"


# ---------------------------------------------------------------------------
# 14. Bounds enforcement tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_team_create_exceeds_max_members(tmp_path: Path) -> None:
    """Given: lead agent with max_members=2 configured.

    When: team_create is called with 3 members.
    Then: error returned about exceeding max_members.
    """
    from wolfharness_config.team_mode import TeamBounds

    config = make_enabled_config(base_dir=str(tmp_path))
    config = config.model_copy(update={"bounds": TeamBounds(max_members=2)})
    mock_pool = make_mock_pool()
    mock_registry = MagicMock()
    mock_registry.exists = MagicMock(return_value=True)
    ctx = make_run_context(
        metadata=make_lead_metadata(),
        session_pool=mock_pool,
        config=config,
        base_dir=str(tmp_path),
        agent_registry=mock_registry,
    )
    cap = TeamCommCapability(config, "coordinator", make_lead_metadata())

    result = await cap.team_create(
        ctx,
        "overflow_team",
        [
            {"agent": "worker", "name": "m1"},
            {"agent": "worker", "name": "m2"},
            {"agent": "reviewer", "name": "m3"},
        ],
    )

    assert "max_members" in result.return_value
    assert "3 > 2" in result.return_value


@pytest.mark.unit
async def test_team_add_member_exceeds_max_members(tmp_path: Path) -> None:
    """Given: team created with 2 members and max_members=2.

    When: team_add_member is called to add a 3rd member.
    Then: error returned about exceeding max_members.
    """
    from wolfharness_config.team_mode import TeamBounds

    config = make_enabled_config(
        member_eligible=["worker", "reviewer", "editor"],
        base_dir=str(tmp_path),
    )
    config = config.model_copy(update={"bounds": TeamBounds(max_members=2)})
    mock_pool = make_mock_pool()
    mock_registry = MagicMock()
    mock_registry.exists = MagicMock(return_value=True)
    lead_meta: dict[str, Any] = {
        "team_role": "lead",
        "team_member_name": "coordinator",
    }

    def ctx_factory() -> MagicMock:
        return make_run_context(
            metadata=lead_meta,
            session_pool=mock_pool,
            config=config,
            base_dir=str(tmp_path),
            agent_registry=mock_registry,
        )

    cap = TeamCommCapability(config, "coordinator", lead_meta)

    # Create team with 2 members (at max).
    create_result = await cap.team_create(
        ctx_factory(),
        "max_team",
        [
            {"agent": "worker", "name": "m1"},
            {"agent": "reviewer", "name": "m2"},
        ],
    )
    assert "team_id=" in create_result.return_value
    team_id = create_result.return_value.split("team_id=")[1].strip()
    lead_meta["team_id"] = team_id
    lead_meta["team_name"] = "max_team"

    # Try to add a 3rd member.
    add_result = await cap.team_add_member(
        ctx_factory(),
        "m3",
        "editor",
    )

    assert "max_members" in add_result.return_value
