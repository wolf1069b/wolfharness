"""Integration tests for team_create with config default members.

These tests exercise the team_create flow with defaults config providing
default members when the LLM passes an empty members list. Uses real
FileTeamState on tmp_path, with mocked SessionPool and DelegationService.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from wolfharness.capabilities.team_comm_capability import TeamCommCapability
from wolfharness_config.team_mode import MemberSpec, TeamDefaultsConfig, TeamModeConfig


def _make_defaults_config(base_dir: str) -> TeamModeConfig:
    """Create an enabled TeamModeConfig with defaults for testing."""
    return TeamModeConfig(
        enabled=True,
        member_eligible=["translator", "reviewer"],
        lead_eligible=["coordinator"],
        base_dir=base_dir,
        defaults=TeamDefaultsConfig(
            team_name="auto_integration_team",
            members=[
                MemberSpec(name="translator", agent="translator"),
                MemberSpec(name="reviewer", agent="reviewer"),
            ],
        ),
    )


def _make_run_context(
    metadata: dict[str, Any],
    session_pool: MagicMock,
    config: TeamModeConfig,
    agent_registry: MagicMock,
    delegation: MagicMock,
    session_id: str = "lead_session_001",
) -> MagicMock:
    """Create a mock RunContext with AgentContextDeps deps for integration tests."""
    from wolfharness.capabilities.agent_context import AgentContextDeps

    agent_ctx = MagicMock(spec=AgentContextDeps)
    agent_ctx.session.metadata = metadata
    agent_ctx.host.session_pool = session_pool
    agent_ctx.team_mode_config = config
    agent_ctx.agent_registry = agent_registry
    agent_ctx.session.session_id = session_id
    agent_ctx.delegation = delegation

    ctx = MagicMock()
    ctx.deps = agent_ctx
    return ctx


@pytest.mark.integration
async def test_team_create_with_config_default_members(tmp_path: Any) -> None:
    """Given: TeamCommCapability with defaults config, lead role.

    When: team_create is called with empty members.
    Then: uses defaults.members to create the team with child sessions.
    """
    from wolfharness.capabilities.file_team_state import FileTeamState

    config = _make_defaults_config(str(tmp_path))

    mock_pool = MagicMock()
    mock_pool.send_message = AsyncMock(return_value="msg_id")
    mock_pool.close_session = AsyncMock()
    mock_pool.sessions = MagicMock()
    mock_pool.sessions.get_or_create_session_agent = AsyncMock()
    mock_pool.event_bus = None

    mock_registry = MagicMock()
    mock_registry.exists = MagicMock(return_value=True)

    child_ids = iter(["child_translator", "child_reviewer"])

    def _make_child_state() -> Any:
        state = MagicMock()
        state.session_id = next(child_ids)
        return state

    mock_pool.create_child_session = AsyncMock(side_effect=lambda **kw: _make_child_state())
    mock_delegation = MagicMock()
    mock_delegation.create_child_session = AsyncMock(
        side_effect=lambda *a, **kw: next(child_ids, "child_session"),
    )

    lead_metadata: dict[str, Any] = {
        "team_role": "lead",
        "team_member_name": "coordinator",
    }
    ctx = _make_run_context(lead_metadata, mock_pool, config, mock_registry, mock_delegation)
    cap = TeamCommCapability(config, "coordinator", lead_metadata)

    result = await cap.team_create(ctx, "my_team", [])

    assert "Team 'my_team' created with 2 members" in result.return_value
    assert "team_id=" in result.return_value
    team_id = result.return_value.split("team_id=")[1].strip()

    # FileTeamState should have the team on disk.
    team_state = FileTeamState(str(tmp_path))
    state_path = team_state._state_path(team_id)
    assert state_path.exists()

    state = team_state._read_json(state_path)
    assert state["team_name"] == "my_team"
    assert "translator" in state["members"]
    assert "reviewer" in state["members"]

    # SessionPool.create_child_session should have been called for each member.
    assert mock_pool.create_child_session.await_count == 2
    assert mock_pool.send_message.await_count == 2


@pytest.mark.integration
async def test_team_create_config_default_members_graceful_degradation(
    tmp_path: Any,
) -> None:
    """Given: defaults config, but delegation.create_child_session raises.

    When: team_create is called with empty members.
    Then: error message returned, no crash, team state cleaned up.
    """
    config = _make_defaults_config(str(tmp_path))

    mock_pool = MagicMock()
    mock_pool.send_message = AsyncMock(return_value="msg_id")
    mock_pool.close_session = AsyncMock()
    mock_pool.sessions = MagicMock()
    mock_pool.sessions.get_or_create_session_agent = AsyncMock()
    mock_pool.event_bus = None
    mock_pool.create_child_session = AsyncMock(side_effect=RuntimeError("Session creation failed"))

    mock_registry = MagicMock()
    mock_registry.exists = MagicMock(return_value=True)

    mock_delegation = MagicMock()
    mock_delegation.create_child_session = AsyncMock(
        side_effect=RuntimeError("Session creation failed"),
    )

    lead_metadata: dict[str, Any] = {
        "team_role": "lead",
        "team_member_name": "coordinator",
    }
    ctx = _make_run_context(lead_metadata, mock_pool, config, mock_registry, mock_delegation)
    cap = TeamCommCapability(config, "coordinator", lead_metadata)

    result = await cap.team_create(ctx, "my_team", [])

    assert "Failed to create team" in result.return_value
    assert "Session creation failed" in result.return_value


@pytest.mark.integration
async def test_team_create_defaults_member_skills_injected(tmp_path: Any) -> None:
    """R6: defaults member with ``skills`` → skill blocks injected.

    End-to-end through the defaults fallback: lead calls team_create with
    empty members, defaults provide members with per-member skills, and
    the child session receives rendered skill instructions.
    """
    config = TeamModeConfig(
        enabled=True,
        member_eligible=["translator", "reviewer"],
        lead_eligible=["coordinator"],
        base_dir=str(tmp_path),
        defaults=TeamDefaultsConfig(
            team_name="auto_skill_team",
            members=[
                MemberSpec(name="translator_a", agent="translator", skills=["lodestone"]),
                MemberSpec(name="reviewer_a", agent="reviewer"),
            ],
        ),
    )

    mock_pool = MagicMock()
    mock_pool.send_message = AsyncMock(return_value="msg_id")
    mock_pool.close_session = AsyncMock()
    mock_pool.sessions = MagicMock()
    mock_pool.sessions.get_or_create_session_agent = AsyncMock()
    mock_pool.event_bus = None

    child_ids = iter(["child_translator", "child_reviewer"])

    def _make_child_state() -> Any:
        state = MagicMock()
        state.session_id = next(child_ids)
        return state

    mock_pool.create_child_session = AsyncMock(side_effect=lambda **kw: _make_child_state())

    mock_registry = MagicMock()
    mock_registry.exists = MagicMock(return_value=True)
    mock_delegation = MagicMock()
    mock_delegation.create_child_session = AsyncMock(
        side_effect=lambda *a, **kw: next(child_ids, "child_session"),
    )

    lead_metadata: dict[str, Any] = {
        "team_role": "lead",
        "team_member_name": "coordinator",
    }
    ctx = _make_run_context(lead_metadata, mock_pool, config, mock_registry, mock_delegation)
    cap = TeamCommCapability(config, "coordinator", lead_metadata)

    from unittest.mock import patch

    with patch(
        "wolfharness_toolsets.builtin.skills.load_skill_for_node",
        new=AsyncMock(return_value="Skill docs"),
    ) as mock_load:
        result = await cap.team_create(ctx, "my_team", [])

    assert "Team 'my_team' created with 2 members" in result.return_value
    # Only the translator member requested a skill — one load.
    assert mock_load.await_count == 1
    assert mock_load.await_args.kwargs["node_name"] == "translator"
    assert mock_load.await_args.kwargs["include_assembly"] is False

    # The translator child session received the skill block; the reviewer did not.
    create_calls = mock_pool.create_child_session.await_args_list
    translator_call = next(
        c for c in create_calls if c.kwargs.get("team_member_name") == "translator_a"
    )
    reviewer_call = next(
        c for c in create_calls if c.kwargs.get("team_member_name") == "reviewer_a"
    )
    assert (
        '<skill-instruction name="lodestone">' in translator_call.kwargs["team_member_instructions"]
    )
    assert "Skill docs" in translator_call.kwargs["team_member_instructions"]
    assert reviewer_call.kwargs["team_member_instructions"] == ""


@pytest.mark.integration
async def test_team_create_defaults_member_instructions_propagated(tmp_path: Any) -> None:
    """D4a regression: defaults MemberSpec.instructions is no longer dropped.

    Previously the defaults fallback mapped only {"name", "agent"}, silently
    discarding ``MemberSpec.instructions``. With the fix, both instructions
    and skills flow into the member's ``## Your Assignment`` section.
    """
    config = TeamModeConfig(
        enabled=True,
        member_eligible=["translator"],
        lead_eligible=["coordinator"],
        base_dir=str(tmp_path),
        defaults=TeamDefaultsConfig(
            team_name="auto_instr_team",
            members=[
                MemberSpec(
                    name="translator_a",
                    agent="translator",
                    instructions="Translate to Italian",
                    skills=["lodestone"],
                ),
            ],
        ),
    )

    mock_pool = MagicMock()
    mock_pool.send_message = AsyncMock(return_value="msg_id")
    mock_pool.close_session = AsyncMock()
    mock_pool.sessions = MagicMock()
    mock_pool.sessions.get_or_create_session_agent = AsyncMock()
    mock_pool.event_bus = None
    mock_pool.create_child_session = AsyncMock(
        side_effect=lambda **kw: MagicMock(session_id="child_translator")
    )

    mock_registry = MagicMock()
    mock_registry.exists = MagicMock(return_value=True)
    mock_delegation = MagicMock()
    mock_delegation.create_child_session = AsyncMock(return_value="child_translator")

    lead_metadata: dict[str, Any] = {
        "team_role": "lead",
        "team_member_name": "coordinator",
    }
    ctx = _make_run_context(lead_metadata, mock_pool, config, mock_registry, mock_delegation)
    cap = TeamCommCapability(config, "coordinator", lead_metadata)

    from unittest.mock import patch

    with patch(
        "wolfharness_toolsets.builtin.skills.load_skill_for_node",
        new=AsyncMock(return_value="Skill docs"),
    ):
        result = await cap.team_create(ctx, "my_team", [])

    assert "Team 'my_team' created with 1 members" in result.return_value
    call_kwargs = mock_pool.create_child_session.await_args.kwargs
    instructions: str = call_kwargs["team_member_instructions"]
    # Skills first, then the lead/defaults instructions.
    assert "<skill-instruction" in instructions
    assert "Translate to Italian" in instructions
    assert instructions.index("<skill-instruction") < instructions.index("Translate to Italian")


@pytest.mark.integration
async def test_team_create_config_default_members_then_delete(tmp_path: Any) -> None:
    """Given: team_create with config default members creates a team.

    When: team_delete is called afterwards.
    Then: team is successfully deleted.
    """
    from wolfharness.capabilities.file_team_state import FileTeamState

    config = _make_defaults_config(str(tmp_path))

    mock_pool = MagicMock()
    mock_pool.send_message = AsyncMock(return_value="msg_id")
    mock_pool.close_session = AsyncMock()
    mock_pool.sessions = MagicMock()
    mock_pool.sessions.get_or_create_session_agent = AsyncMock()
    mock_pool.event_bus = None

    mock_registry = MagicMock()
    mock_registry.exists = MagicMock(return_value=True)

    child_ids = iter(["child_translator", "child_reviewer"])

    def _make_child_state() -> Any:
        state = MagicMock()
        state.session_id = next(child_ids)
        return state

    mock_pool.create_child_session = AsyncMock(side_effect=lambda **kw: _make_child_state())
    mock_delegation = MagicMock()
    mock_delegation.create_child_session = AsyncMock(
        side_effect=lambda *a, **kw: next(child_ids, "child_session"),
    )

    lead_metadata: dict[str, Any] = {
        "team_role": "lead",
        "team_member_name": "coordinator",
    }
    ctx = _make_run_context(lead_metadata, mock_pool, config, mock_registry, mock_delegation)
    cap = TeamCommCapability(config, "coordinator", lead_metadata)

    # Create the team with empty members (uses defaults config).
    create_result = await cap.team_create(ctx, "my_team", [])
    assert "Team 'my_team' created with 2 members" in create_result.return_value
    team_id = create_result.return_value.split("team_id=")[1].strip()

    # Verify team exists on disk.
    team_state = FileTeamState(str(tmp_path))
    assert team_state._state_path(team_id).exists()

    # Write team_id into metadata so team_delete can find it.
    lead_metadata["team_id"] = team_id
    lead_metadata["team_name"] = "my_team"

    # Now delete the team.
    result = await cap.team_delete(ctx)

    assert result.return_value == "Team deleted"
    assert not team_state._state_path(team_id).exists()
    assert mock_pool.close_session.await_count == 2
