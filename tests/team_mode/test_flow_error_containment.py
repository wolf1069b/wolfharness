"""L2 integration tests for soft/hard error isolation in team mode.

These tests verify that failures within team operations are contained as
soft errors (returning error strings) rather than hard errors (raising
exceptions that crash the agent run).

The tests use the real ``team_mode_pool`` fixture and exercise
``TeamCommCapability`` methods directly via ``build_agent_context`` and
``make_mock_run_context`` helpers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic_ai.messages import ToolReturn
import pytest

from tests.team_mode.conftest import build_agent_context, make_mock_run_context
from wolfharness.capabilities.file_team_state import FileTeamState
from wolfharness.capabilities.team_comm_capability import TeamCommCapability


if TYPE_CHECKING:
    from wolfharness import AgentPool
    from wolfharness_config.team_mode import TeamModeConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_LEAD_METADATA: dict[str, Any] = {
    "team_role": "lead",
    "team_member_name": "coordinator",
}


async def _setup_lead_session(
    pool: AgentPool[Any],
    session_id: str,
) -> tuple[str, TeamModeConfig, Any]:
    """Create a lead session and return (session_id, config, agent_ctx).

    Args:
        pool: Real AgentPool instance.
        session_id: Unique session identifier for the lead.

    Returns:
        A tuple of (session_id, team_mode_config, agent_ctx).
    """
    manifest = pool.manifest
    team_mode_config: TeamModeConfig | None = manifest.team_mode
    assert team_mode_config is not None

    session_pool = pool.session_pool
    assert session_pool is not None
    await session_pool.create_session(
        session_id,
        agent_name="coordinator",
        team_role="lead",
        team_member_name="coordinator",
    )

    agent_ctx = build_agent_context(pool, session_id, team_mode_config)
    return session_id, team_mode_config, agent_ctx


async def _create_team(
    cap: TeamCommCapability,
    agent_ctx: Any,
    team_name: str,
    members: list[dict[str, str]],
) -> str:
    """Call team_create and return the extracted team_id.

    Args:
        cap: TeamCommCapability instance with lead metadata.
        agent_ctx: Real AgentContext from build_agent_context.
        team_name: Human-readable team name.
        members: List of member dicts with ``agent`` and ``name`` keys.

    Returns:
        The team_id string extracted from the team_create result.
    """
    ctx = make_mock_run_context(agent_ctx)
    result = await cap.team_create(ctx, team_name, members)
    assert "team_id=" in result.return_value
    team_id = result.return_value.split("team_id=")[1].strip()
    # Update lead metadata so subsequent tool calls find the team_id.
    agent_ctx.session.metadata["team_id"] = team_id
    agent_ctx.session.metadata["team_name"] = team_name
    cap._session_metadata["team_id"] = team_id
    cap._session_metadata["team_name"] = team_name
    return team_id


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
async def test_member_crash_returns_error_to_lead(
    team_mode_pool: AgentPool[Any],
) -> None:
    """Given: a team with 2 members, one member session is closed (crashed).

    When: the lead calls ``send_message`` to the crashed member.

    Then: the lead receives an error string (``"Failed to deliver
        message to '{to}'"``) — not an exception.  The run is not
        crashed; the lead can continue operating.
    """
    session_id, config, agent_ctx = await _setup_lead_session(
        team_mode_pool,
        "err-containment-lead-001",
    )

    cap = TeamCommCapability(
        config,
        "coordinator",
        session_metadata={**_LEAD_METADATA},
    )

    team_id = await _create_team(
        cap,
        agent_ctx,
        "crash_test_team",
        [
            {"agent": "worker", "name": "worker_1"},
            {"agent": "reviewer", "name": "reviewer_1"},
        ],
    )

    # Extract member session IDs from team state.
    base_dir = config.effective_base_dir
    team_state = FileTeamState(base_dir)
    state = team_state._read_json(team_state._state_path(team_id))
    members: dict[str, dict[str, str]] = state.get("members", {})
    worker_session_id = members.get("worker_1", {}).get("session_id", "")
    assert worker_session_id != ""

    # Simulate member crash: close the worker's session.
    session_pool = team_mode_pool.session_pool
    assert session_pool is not None
    await session_pool.close_session(worker_session_id)

    # Verify the session is gone.
    closed_session = session_pool.sessions.get_session(worker_session_id)
    assert closed_session is None

    # Lead tries to send a message to the crashed member.
    ctx = make_mock_run_context(agent_ctx)
    result = await cap.send_message(ctx, "worker_1", "Are you there?")

    # The result is an error string, NOT an exception.
    assert isinstance(result, ToolReturn)
    assert "Failed to deliver" in result.return_value or "not found" in result.return_value.lower()

    # Cleanup: close the lead session.
    await session_pool.close_session(session_id)


@pytest.mark.integration
async def test_team_delete_cleans_up_all_members(
    team_mode_pool: AgentPool[Any],
) -> None:
    """Given: a team with 2 members, one member session is already closed (error state).

    When: the lead calls ``team_delete``.

    Then: all remaining member sessions are closed and the team state
        is cleaned up.  ``team_delete`` does not raise an exception
        even when some members are already in an error state.
    """
    session_id, config, agent_ctx = await _setup_lead_session(
        team_mode_pool,
        "err-cleanup-lead-001",
    )

    cap = TeamCommCapability(
        config,
        "coordinator",
        session_metadata={**_LEAD_METADATA},
    )

    team_id = await _create_team(
        cap,
        agent_ctx,
        "cleanup_test_team",
        [
            {"agent": "worker", "name": "worker_1"},
            {"agent": "reviewer", "name": "reviewer_1"},
        ],
    )

    base_dir = config.effective_base_dir
    team_state = FileTeamState(base_dir)
    state = team_state._read_json(team_state._state_path(team_id))
    members: dict[str, dict[str, str]] = state.get("members", {})
    worker_session_id = members.get("worker_1", {}).get("session_id", "")
    reviewer_session_id = members.get("reviewer_1", {}).get("session_id", "")
    assert worker_session_id != ""
    assert reviewer_session_id != ""

    # Put one member in error state by closing its session.
    session_pool = team_mode_pool.session_pool
    assert session_pool is not None
    await session_pool.close_session(worker_session_id)

    # Verify worker session is gone but reviewer is still active.
    assert session_pool.sessions.get_session(worker_session_id) is None
    assert session_pool.sessions.get_session(reviewer_session_id) is not None

    # team_delete should close all members (even the already-closed one)
    # without raising.
    ctx = make_mock_run_context(agent_ctx)
    result = await cap.team_delete(ctx)

    assert result.return_value == "Team deleted"

    # Verify both member sessions are now gone.
    assert session_pool.sessions.get_session(worker_session_id) is None
    assert session_pool.sessions.get_session(reviewer_session_id) is None

    # Verify team state is cleaned up.
    state_path = team_state._state_path(team_id)
    assert not state_path.exists()

    # Cleanup.
    await session_pool.close_session(session_id)


@pytest.mark.integration
async def test_blackboard_write_failure_is_soft_error(
    team_mode_pool: AgentPool[Any],
) -> None:
    """Given: a team with a blackboard key at version 1.

    When: ``write_blackboard`` is called with a stale
        ``expected_version=0`` (version conflict).

    Then: the tool returns a ``"Conflict: current version is 1"``
        error string — NOT an exception.  The run is not crashed;
        the agent can retry with the correct version.
    """
    session_id, config, agent_ctx = await _setup_lead_session(
        team_mode_pool,
        "err-bb-lead-001",
    )

    cap = TeamCommCapability(
        config,
        "coordinator",
        session_metadata={**_LEAD_METADATA},
    )

    await _create_team(
        cap,
        agent_ctx,
        "bb_error_team",
        [{"agent": "worker", "name": "worker_1"}],
    )

    ctx = make_mock_run_context(agent_ctx)

    # First write succeeds (version=1).
    write_result = await cap.write_blackboard(ctx, "status", "in_progress")
    assert write_result.return_value == "Written, version=1"

    # Second write with stale expected_version=0 → conflict (soft error).
    conflict_result = await cap.write_blackboard(
        ctx,
        "status",
        "done",
        expected_version=0,
    )

    assert isinstance(conflict_result, ToolReturn)
    assert conflict_result.return_value == "Conflict: current version is 1"

    # Verify the blackboard value was NOT overwritten.
    read_result = await cap.read_blackboard(ctx, "status")
    rv = (
        "\n".join(read_result.return_value)
        if isinstance(read_result.return_value, list)
        else read_result.return_value
    )
    assert "<blackboard" in rv
    assert "in_progress" in rv
    assert 'version="1"' in rv

    # Cleanup.
    session_pool = team_mode_pool.session_pool
    assert session_pool is not None
    await cap.team_delete(ctx)
    await session_pool.close_session(session_id)
