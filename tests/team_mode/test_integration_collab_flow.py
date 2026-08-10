"""Integration tests for team-mode-collab-flow features.

L2 tests that exercise multiple TeamCommCapability methods in sequence
using real FileTeamState on tmp_path and mock SessionPool.
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
    make_mock_registry,
    make_run_context,
)
from wolfharness.capabilities.team_comm_capability import TeamCommCapability


if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# 11.1 Full team lifecycle with per-member instructions
# ---------------------------------------------------------------------------


@pytest.mark.integration
async def test_full_lifecycle_with_per_member_instructions(tmp_path: Path) -> None:
    """Full team lifecycle: create → add member with instructions → verify.

    Turn 1: team_create with a member that has instructions
    Turn 2: team_status
    Turn 3: task_create
    Turn 4: task_list
    Turn 5: team_delete
    """
    config = make_enabled_config(
        member_eligible=["worker", "reviewer"],
        base_dir=str(tmp_path),
    )
    mock_pool = make_mock_pool()
    mock_registry = make_mock_registry()
    mock_delegation = MagicMock()
    mock_delegation.create_child_session = AsyncMock(return_value="child_001")

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
            delegation=mock_delegation,
        )

    cap = TeamCommCapability(config, "coordinator", lead_meta)

    # Turn 1: team_create with instructions on a member.
    create_result = await cap.team_create(
        ctx_factory(),
        "instr_team",
        [
            {"agent": "worker", "name": "analyst", "instructions": "Do analysis."},
            {"agent": "reviewer", "name": "checker"},
        ],
    )
    assert "team_id=" in create_result.return_value
    team_id = create_result.return_value.split("team_id=")[1].strip()
    lead_meta["team_id"] = team_id
    lead_meta["team_name"] = "instr_team"

    # Verify instructions propagated to create_child_session kwargs.
    create_calls = mock_pool.create_child_session.await_args_list
    analyst_call = next(c for c in create_calls if c.kwargs.get("team_member_name") == "analyst")
    assert analyst_call.kwargs.get("team_member_instructions") == "Do analysis."
    checker_call = next(c for c in create_calls if c.kwargs.get("team_member_name") == "checker")
    assert checker_call.kwargs.get("team_member_instructions") == ""

    # Turn 2: team_status.
    status = await cap.team_status(ctx_factory())
    assert "instr_team" in status.return_value

    # Turn 3: task_create.
    task_result = await cap.task_create(ctx_factory(), "Integration task", owner="translator_agent")
    assert task_result.return_value.startswith("Task created: ")

    # Turn 4: task_list.
    list_result = await cap.task_list(ctx_factory())
    assert "Integration task" in list_result.return_value

    # Turn 5: team_delete.
    delete_result = await cap.team_delete(ctx_factory())
    assert delete_result.return_value == "Team deleted"


# ---------------------------------------------------------------------------
# 11.2 Handoff flow
# ---------------------------------------------------------------------------


@pytest.mark.integration
async def test_handoff_flow_with_dependencies(tmp_path: Path) -> None:
    """Create tasks with deps → complete with handoff → notification + dep resolved."""
    init_team(str(tmp_path))
    mock_pool = make_mock_pool()
    ctx = make_run_context(
        metadata=make_lead_metadata(team_id="team_123"),
        session_pool=mock_pool,
        base_dir=str(tmp_path),
    )
    config = make_enabled_config(base_dir=str(tmp_path))
    cap = TeamCommCapability(config, "coordinator", make_lead_metadata(team_id="team_123"))

    # Create task A (no deps).
    a_result = await cap.task_create(ctx, "Research phase", owner="translator_agent")
    a_id = a_result.return_value.replace("Task created: ", "")

    # Create task B (blocked by A, owned by reviewer_agent).
    b_result = await cap.task_create(
        ctx,
        "Review phase",
        blocked_by=[a_id],
        owner="reviewer_agent",
    )
    b_id = b_result.return_value.replace("Task created: ", "")

    # Write a blackboard key for handoff context.
    await cap.write_blackboard(ctx, "research_output", "Key findings here")

    # Complete task A with handoff to reviewer_agent.
    send_before = mock_pool.send_message.await_count
    result = await cap.task_update(
        ctx,
        a_id,
        status="completed",
        handoff_to="reviewer_agent",
        handoff_context_keys=["research_output"],
    )
    send_after = mock_pool.send_message.await_count

    assert "handoff notification sent to reviewer_agent" in result.return_value
    # At least 2 notifications: handoff + dependency resolution.
    assert send_after - send_before >= 2

    # Verify task B is now unblocked.
    tasks = cap._get_team_state(cap._resolve_agent_context(ctx)).list_tasks("team_123")
    task_b = next(t for t in tasks if t["task_id"] == b_id)
    assert task_b["is_unblocked"] is True


# ---------------------------------------------------------------------------
# 11.3 send_message with persist_to_blackboard
# ---------------------------------------------------------------------------


@pytest.mark.integration
async def test_send_message_with_persist_to_blackboard_integration(
    tmp_path: Path,
) -> None:
    """send_message with persist_to_blackboard writes to both inbox and blackboard."""
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
        "Critical findings: the API has a bug in endpoint X.",
        persist_to_blackboard="critical_findings",
    )

    assert "Message sent to reviewer_agent" in result.return_value
    assert "Persisted to blackboard key 'critical_findings'" in result.return_value

    # Verify blackboard was actually written.
    from wolfharness.capabilities.file_team_state import FileTeamState

    team_state = FileTeamState(str(tmp_path))
    bb = team_state.read_blackboard("team_123", "critical_findings")
    assert bb is not None
    assert "Critical findings" in bb["value"]["text"]
    assert bb["written_by"] == "worker"

    # Verify message was sent via session_pool.
    mock_pool.send_message.assert_awaited()


# ---------------------------------------------------------------------------
# 11.4 Protocol template rendering
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_protocol_template_rendering_with_communication_channels() -> None:
    """Verify the protocol template renders with all Communication Channels sections."""
    config = make_enabled_config()
    cap = TeamCommCapability(config, "coordinator", make_lead_metadata(team_id="team_123"))

    instructions = cap.get_instructions()

    assert instructions is not None
    assert "## Communication Channels" in instructions
    assert "### Tasks" in instructions
    assert "### Blackboard" in instructions
    assert "### Messages" in instructions
    assert "## Guidelines" not in instructions
    assert "task_create_batch" in instructions
    assert "technical_note" in instructions
    assert "progress_current" in instructions


# ---------------------------------------------------------------------------
# 11.5 Batch task creation with #N and symbolic id references
# ---------------------------------------------------------------------------


@pytest.mark.integration
async def test_batch_creation_with_hash_and_symbolic_refs(tmp_path: Path) -> None:
    """Create a batch with both #N and symbolic id references, verify resolution."""
    init_team(str(tmp_path))
    ctx = make_run_context(
        metadata=make_lead_metadata(team_id="team_123"),
        base_dir=str(tmp_path),
    )
    config = make_enabled_config(base_dir=str(tmp_path))
    cap = TeamCommCapability(config, "coordinator", make_lead_metadata(team_id="team_123"))

    result = await cap.task_create_batch(
        ctx,
        [
            {"subject": "Setup", "id": "setup"},
            {"subject": "Build", "id": "build", "blocked_by": ["setup"]},
            {"subject": "Test", "blocked_by": ["#0", "#1"]},
        ],
    )

    assert "Created 3 tasks" in result.return_value

    # Verify dependencies were resolved.
    from wolfharness.capabilities.file_team_state import FileTeamState

    team_state = FileTeamState(str(tmp_path))
    tasks = team_state.list_tasks("team_123")
    test_task = next(t for t in tasks if t["subject"] == "Test")
    build_task = next(t for t in tasks if t["subject"] == "Build")
    setup_task = next(t for t in tasks if t["subject"] == "Setup")

    # Test task should be blocked by both setup and build.
    assert setup_task["task_id"] in test_task["blocked_by"]
    assert build_task["task_id"] in test_task["blocked_by"]
    # Build task should be blocked by setup.
    assert setup_task["task_id"] in build_task["blocked_by"]
    # Test task should NOT be unblocked yet.
    assert test_task["is_unblocked"] is False


# ---------------------------------------------------------------------------
# 11.6 Progress tracking lifecycle
# ---------------------------------------------------------------------------


@pytest.mark.integration
async def test_progress_tracking_lifecycle(tmp_path: Path) -> None:
    """Update progress → task_list shows progress → auto-complete on completion."""
    init_team(str(tmp_path))
    ctx = make_run_context(
        metadata=make_lead_metadata(team_id="team_123"),
        base_dir=str(tmp_path),
    )
    config = make_enabled_config(base_dir=str(tmp_path))
    cap = TeamCommCapability(config, "coordinator", make_lead_metadata(team_id="team_123"))

    # Create task.
    create_result = await cap.task_create(ctx, "Long-running task", owner="translator_agent")
    task_id = create_result.return_value.replace("Task created: ", "")

    # Set progress_total.
    await cap.task_update(ctx, task_id, progress_total=10)

    # Update progress to 3/10.
    await cap.task_update(ctx, task_id, progress_current=3)

    # task_list should show progress.
    list_result = await cap.task_list(ctx)
    assert 'progress="3/10"' in list_result.return_value

    # Update progress to 7/10.
    await cap.task_update(ctx, task_id, progress_current=7)
    list_result = await cap.task_list(ctx)
    assert 'progress="7/10"' in list_result.return_value

    # Complete the task — should auto-set progress to 10/10.
    complete_result = await cap.task_update(ctx, task_id, status="completed")
    assert 'status="completed"' in complete_result.return_value
    # Verify progress via task_get (task_update return doesn't include progress attr).
    get_result = await cap.task_get(ctx, task_id)
    assert 'progress="10/10"' in get_result.return_value


# ---------------------------------------------------------------------------
# 11.7 Ownership enforcement
# ---------------------------------------------------------------------------


@pytest.mark.integration
async def test_ownership_enforcement_with_send_message_suggestion(
    tmp_path: Path,
) -> None:
    """Non-lead attempts other's task → error → send_message suggestion in message."""
    init_team(str(tmp_path))
    mock_pool = make_mock_pool()
    lead_ctx = make_run_context(
        metadata=make_lead_metadata(team_id="team_123"),
        base_dir=str(tmp_path),
    )
    config = make_enabled_config(base_dir=str(tmp_path))
    lead_cap = TeamCommCapability(config, "coordinator", make_lead_metadata(team_id="team_123"))

    # Lead creates a task and assigns to reviewer_agent.
    create_result = await lead_cap.task_create(lead_ctx, "Review task", owner="translator_agent")
    task_id = create_result.return_value.replace("Task created: ", "")
    await lead_cap.task_update(lead_ctx, task_id, owner="reviewer_agent")

    # Translator_agent (member) tries to update reviewer_agent's task.
    member_ctx = make_run_context(
        metadata=make_member_metadata(),
        session_pool=mock_pool,
        base_dir=str(tmp_path),
    )
    member_cap = TeamCommCapability(config, "worker", make_member_metadata())

    result = await member_cap.task_update(member_ctx, task_id, status="completed")

    # Should get ownership error with suggestion.
    assert "reviewer_agent" in result.return_value
    assert "send_message" in result.return_value

    # Member then sends a message to reviewer_agent as suggested.
    msg_result = await member_cap.send_message(
        member_ctx,
        "reviewer_agent",
        "Can you complete the review task?",
    )
    assert msg_result.return_value == "Message sent to reviewer_agent"


# ---------------------------------------------------------------------------
# 11.8 Handoff + dependency to same member
# ---------------------------------------------------------------------------


@pytest.mark.integration
async def test_handoff_and_dependency_to_same_member(tmp_path: Path) -> None:
    """Both handoff and dependency notifications sent to the same member."""
    init_team(str(tmp_path))
    mock_pool = make_mock_pool()
    ctx = make_run_context(
        metadata=make_lead_metadata(team_id="team_123"),
        session_pool=mock_pool,
        base_dir=str(tmp_path),
    )
    config = make_enabled_config(base_dir=str(tmp_path))
    cap = TeamCommCapability(config, "coordinator", make_lead_metadata(team_id="team_123"))

    # Create task A (no deps).
    a_result = await cap.task_create(ctx, "Phase 1", owner="translator_agent")
    a_id = a_result.return_value.replace("Task created: ", "")

    # Create task B (blocked by A, owned by reviewer_agent).
    await cap.task_create(
        ctx,
        "Phase 2",
        blocked_by=[a_id],
        owner="reviewer_agent",
    )

    # Complete task A with handoff to reviewer_agent (same as B's owner).
    send_before = mock_pool.send_message.await_count
    result = await cap.task_update(
        ctx,
        a_id,
        status="completed",
        handoff_to="reviewer_agent",
        handoff_context_keys=["phase1_results"],
    )
    send_after = mock_pool.send_message.await_count

    # Both handoff and dependency notifications should be sent.
    assert "handoff notification sent to reviewer_agent" in result.return_value
    assert send_after - send_before >= 2

    # Verify at least one notification mentions handoff and another mentions
    # dependency_resolved.
    bodies = [str(c.args[1]) for c in mock_pool.send_message.await_args_list]
    has_handoff = any("handoff" in b for b in bodies[-2:])
    has_dep = any("dependency_resolved" in b for b in bodies[-2:])
    assert has_handoff
    assert has_dep


# ---------------------------------------------------------------------------
# 11.9 Agent-in-the-loop: full agent→tool→agent flow with FunctionModel
# ---------------------------------------------------------------------------


@pytest.mark.integration
async def test_agent_in_the_loop_team_tools(team_mode_pool: Any) -> None:
    """Given: a real AgentPool with team_mode enabled and TestModel agents.

    When: the coordinator agent is run with a FunctionModel that scripts
        a sequence of tool calls: team_create → task_create → task_list →
        team_status.

    Then: all four tool calls are issued through the full agent loop and
        tool returns are received in the message history.
    """
    from tests.team_mode.conftest import (
        _tool_call_names,
        _tool_returns_by_name,
        make_lifecycle_model,
    )

    session_id = "test-agent-in-loop"
    await team_mode_pool.session_pool.create_session(
        session_id,
        agent_name="coordinator",
        team_role="lead",
        team_member_name="coordinator",
    )
    agent = await team_mode_pool.session_pool.sessions.get_or_create_session_agent(
        session_id,
    )
    model = make_lifecycle_model([
        (
            "team_create",
            {
                "name": "test_team",
                "members": [{"agent": "worker", "name": "worker_1"}],
            },
        ),
        ("task_create", {"subject": "Test task", "owner": "worker_1"}),
        ("task_list", {}),
        ("team_status", {}),
    ])
    await agent.set_model(model)

    events = [
        event
        async for event in team_mode_pool.session_pool.run_stream(
            session_id,
            "Create a team and assign a task",
        )
    ]

    # Collect all messages from the agent run result.
    from wolfharness.agents.events.events import StreamCompleteEvent

    final_message = None
    for e in events:
        if isinstance(e, StreamCompleteEvent):
            final_message = e.message
            break

    assert final_message is not None, "No StreamCompleteEvent received"
    all_messages = final_message.messages

    tool_calls = _tool_call_names(all_messages)
    assert "team_create" in tool_calls
    assert "task_create" in tool_calls
    assert "task_list" in tool_calls
    assert "team_status" in tool_calls

    # Verify tool returns were received.
    team_create_returns = _tool_returns_by_name(all_messages, "team_create")
    assert len(team_create_returns) >= 1
    assert "created" in str(team_create_returns[0].content).lower()

    task_create_returns = _tool_returns_by_name(all_messages, "task_create")
    assert len(task_create_returns) >= 1
    assert "Task created" in str(task_create_returns[0].content)

    task_list_returns = _tool_returns_by_name(all_messages, "task_list")
    assert len(task_list_returns) >= 1

    team_status_returns = _tool_returns_by_name(all_messages, "team_status")
    assert len(team_status_returns) >= 1
