"""L2 integration tests for budget/quota exhaustion in team mode.

These tests verify that resource constraints (``bounds`` in
``TeamModeConfig``) are enforced by the team tools:

- ``max_members``: ``team_create`` rejects teams exceeding the limit.
- ``max_member_turns``: ``send_message`` rejects further messages to a
  member that has reached its turn limit.
- ``max_wall_clock_minutes``: the ``started_at`` timestamp is recorded
  during ``team_create`` as a prerequisite for wall-clock enforcement.

The tests use a custom inline YAML config with bounds set low and the
real ``AgentPool`` + ``SessionPool`` stack (no mocks for pool/session).
"""

from __future__ import annotations

import datetime
from typing import TYPE_CHECKING, Any

import pytest
import yamling

from tests.team_mode.conftest import build_agent_context, make_mock_run_context
from wolfharness import AgentPool, AgentsManifest
from wolfharness.capabilities.file_team_state import FileTeamState
from wolfharness.capabilities.team_comm_capability import TeamCommCapability


if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path

    from wolfharness_config.team_mode import TeamModeConfig


# ---------------------------------------------------------------------------
# Custom config with low bounds
# ---------------------------------------------------------------------------

_BUDGET_CONFIG_YAML = """\
agents:
  coordinator:
    type: native
    model: test
    system_prompt: "You are a team coordinator"
  worker:
    type: native
    model: test
    system_prompt: "You are a team worker"
  reviewer:
    type: native
    model: test
    system_prompt: "You are a team reviewer"
  editor:
    type: native
    model: test
    system_prompt: "You are a team editor"

team_mode:
  enabled: true
  member_eligible: [worker, reviewer, editor]
  lead_eligible: [coordinator]
  base_dir: {base_dir}
  bounds:
    max_members: 2
    max_parallel_members: 2
    max_member_turns: 3
    max_wall_clock_minutes: 1
"""


def _build_budget_manifest(base_dir: str) -> AgentsManifest:
    """Parse inline YAML with low bounds into an ``AgentsManifest``.

    Args:
        base_dir: Filesystem path for team state files.

    Returns:
        An ``AgentsManifest`` with ``bounds.max_members=2``,
        ``max_member_turns=3``, ``max_wall_clock_minutes=1``.
    """
    raw = yamling.load_yaml(
        _BUDGET_CONFIG_YAML.format(base_dir=base_dir),
        verify_type=dict,
    )
    return AgentsManifest.model_validate(raw)


@pytest.fixture
async def budget_pool(tmp_path: Path) -> AsyncIterator[AgentPool[Any]]:
    """Real ``AgentPool`` with low bounds for budget exhaustion testing."""
    manifest = _build_budget_manifest(str(tmp_path))
    async with AgentPool(manifest) as pool:
        yield pool


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
        pool: Real AgentPool instance with budget config.
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
        The team_id string, or the full error string if team_create failed.
    """
    ctx = make_mock_run_context(agent_ctx)
    result = await cap.team_create(ctx, team_name, members)
    if "team_id=" in result.return_value:
        team_id = result.return_value.split("team_id=")[1].strip()
        agent_ctx.session.metadata["team_id"] = team_id
        agent_ctx.session.metadata["team_name"] = team_name
        cap._session_metadata["team_id"] = team_id
        cap._session_metadata["team_name"] = team_name
        return team_id
    return result


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
async def test_max_members_enforced(budget_pool: AgentPool[Any]) -> None:
    """Given: a team mode config with ``max_members: 2``.

    When: ``team_create`` is called with 3 members (exceeding the limit).

    Then: the call returns an error string containing
        ``"exceeds max_members"`` with the actual and limit values.
        No team is created.
    """
    session_id, config, agent_ctx = await _setup_lead_session(
        budget_pool,
        "budget-members-lead-001",
    )

    cap = TeamCommCapability(
        config,
        "coordinator",
        session_metadata={**_LEAD_METADATA},
    )

    ctx = make_mock_run_context(agent_ctx)
    result = await cap.team_create(
        ctx,
        "oversized_team",
        [
            {"agent": "worker", "name": "m1"},
            {"agent": "reviewer", "name": "m2"},
            {"agent": "editor", "name": "m3"},
        ],
    )

    assert "exceeds max_members" in result.return_value
    assert "3" in result.return_value
    assert "2" in result.return_value

    # Cleanup.
    session_pool = budget_pool.session_pool
    assert session_pool is not None
    await session_pool.close_session(session_id)


@pytest.mark.integration
async def test_max_member_turns_enforced(budget_pool: AgentPool[Any]) -> None:
    """Given: a team with ``max_member_turns: 3`` and one member.

    When: ``send_message`` is called to the member 4 times (exceeding
        the turn limit on the 4th call).

    Then: the first 3 messages succeed.  The 4th call returns an error
        string containing ``"exceeded max turns"``.
    """
    session_id, config, agent_ctx = await _setup_lead_session(
        budget_pool,
        "budget-turns-lead-001",
    )

    cap = TeamCommCapability(
        config,
        "coordinator",
        session_metadata={**_LEAD_METADATA},
    )

    team_id = await _create_team(
        cap,
        agent_ctx,
        "turns_test_team",
        [{"agent": "worker", "name": "worker_1"}],
    )
    assert isinstance(team_id, str)
    assert len(team_id) > 0

    ctx = make_mock_run_context(agent_ctx)

    # Send 3 messages — all should succeed (turn_count goes 0→1, 1→2, 2→3).
    for i in range(3):
        result = await cap.send_message(ctx, "worker_1", f"message {i + 1}")
        assert result.return_value == "Message sent to worker_1", (
            f"Message {i + 1} should succeed, got: {result.return_value}"
        )

    # 4th message should be rejected (turn_count=3 >= max_member_turns=3).
    result = await cap.send_message(ctx, "worker_1", "message 4")
    assert "exceeded max turns" in result.return_value
    assert "3" in result.return_value

    # Cleanup.
    session_pool = budget_pool.session_pool
    assert session_pool is not None
    await cap.team_delete(ctx)
    await session_pool.close_session(session_id)


@pytest.mark.integration
async def test_wall_clock_timeout(budget_pool: AgentPool[Any]) -> None:
    """Wall-clock timeout infrastructure test with low bounds.

    Given: a team with ``max_wall_clock_minutes: 1`` and ``started_at``
        recorded during ``team_create``.

    When: the ``started_at`` timestamp is manually set to 2 minutes ago
        (exceeding the 1-minute wall-clock limit).

    Then: the ``started_at`` field is present in the team state and the
        elapsed time exceeds ``max_wall_clock_minutes``.  The
        ``team_status`` tool still returns the team status, confirming
        the ``started_at`` infrastructure is in place for wall-clock
        enforcement.
    """
    session_id, config, agent_ctx = await _setup_lead_session(
        budget_pool,
        "budget-clock-lead-001",
    )

    cap = TeamCommCapability(
        config,
        "coordinator",
        session_metadata={**_LEAD_METADATA},
    )

    team_id = await _create_team(
        cap,
        agent_ctx,
        "clock_test_team",
        [{"agent": "worker", "name": "worker_1"}],
    )
    assert isinstance(team_id, str)
    assert len(team_id) > 0

    # Verify started_at is recorded.
    base_dir = config.effective_base_dir
    team_state = FileTeamState(base_dir)
    state_path = team_state._state_path(team_id)
    state: dict[str, Any] = FileTeamState._read_json(state_path)
    assert "started_at" in state
    started_at_str: str = state["started_at"]
    assert started_at_str is not None

    # Parse the ISO timestamp to confirm it's valid.
    started_at = datetime.datetime.fromisoformat(started_at_str)
    assert started_at is not None

    # Manually set started_at to 2 minutes ago (exceeding max_wall_clock_minutes=1).
    past_time = datetime.datetime.now(datetime.UTC) - datetime.timedelta(minutes=2)
    state["started_at"] = past_time.isoformat()
    FileTeamState._atomic_write(state_path, state)

    # Compute elapsed time — it should exceed max_wall_clock_minutes.
    elapsed = datetime.datetime.now(datetime.UTC) - past_time
    assert elapsed.total_seconds() > config.bounds.max_wall_clock_minutes * 60

    # team_status still returns the team status (wall-clock enforcement
    # in tool bodies is a future enhancement; started_at is the
    # prerequisite infrastructure).
    ctx = make_mock_run_context(agent_ctx)
    status_result = await cap.team_status(ctx)
    assert "clock_test_team" in status_result.return_value
    assert "worker_1" in status_result.return_value

    # Cleanup.
    session_pool = budget_pool.session_pool
    assert session_pool is not None
    await cap.team_delete(ctx)
    await session_pool.close_session(session_id)
