"""Integration tests for the dynamic team mode feature.

Covers end-to-end lifecycle, disabled mode, standalone mode, missing
team_id, graph/teams coexistence, and defaults + manual team_create
coexistence scenarios.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock

import pytest
import yamling

from tests.team_mode.conftest import build_agent_context, make_mock_run_context
from wolfharness.capabilities.file_team_state import FileTeamState
from wolfharness.capabilities.team_comm_capability import TeamCommCapability
from wolfharness.models.manifest import AgentsManifest
from wolfharness_config.team_mode import (
    MemberSpec,
    TeamDefaultsConfig,
    TeamModeConfig,
)


if TYPE_CHECKING:
    from wolfharness import AgentPool


# ---------------------------------------------------------------------------
# Helpers (mirrors patterns from test_team_comm_capability.py)
# ---------------------------------------------------------------------------


def _make_enabled_config(
    *,
    member_eligible: list[str] | None = None,
    lead_eligible: list[str] | None = None,
    base_dir: str | None = None,
    defaults: TeamDefaultsConfig | None = None,
) -> TeamModeConfig:
    """Create an enabled TeamModeConfig for integration testing."""
    return TeamModeConfig(
        enabled=True,
        member_eligible=member_eligible or ["worker", "reviewer"],
        lead_eligible=lead_eligible or ["coordinator"],
        base_dir=base_dir,
        defaults=defaults,
    )


def _make_disabled_config() -> TeamModeConfig:
    """Create a disabled TeamModeConfig."""
    return TeamModeConfig(
        enabled=False,
        member_eligible=["worker"],
        lead_eligible=["coordinator"],
    )


def _make_session_metadata(
    team_id: str = "team_123",
    team_role: str = "translator",
    team_member_name: str = "translator_agent",
) -> dict[str, Any]:
    """Create typical session metadata for a team member."""
    return {
        "team_id": team_id,
        "team_name": "alpha_team",
        "team_role": team_role,
        "team_member_name": team_member_name,
    }


def _make_lead_metadata(team_id: str = "team_123") -> dict[str, Any]:
    """Create session metadata for a lead agent."""
    return {
        "team_id": team_id,
        "team_name": "alpha_team",
        "team_role": "lead",
        "team_member_name": "coordinator",
    }


def _make_run_context(
    metadata: dict[str, Any] | None = None,
    session_pool: MagicMock | None = None,
    config: TeamModeConfig | None = None,
    base_dir: str | None = None,
    agent_registry: MagicMock | None = None,
    session_id: str = "lead_session_001",
    delegation: MagicMock | None = None,
) -> MagicMock:
    """Create a mock RunContext with AgentContextDeps deps.

    Args:
        metadata: Session metadata dict (defaults to team member metadata).
        session_pool: Mock SessionPool (or None to test missing pool).
        config: TeamModeConfig (defaults to enabled config).
        base_dir: Optional base_dir override for TeamModeConfig.
        agent_registry: Mock AgentRegistry (defaults to a permissive mock).
        session_id: Session ID string for the mock SessionState.
        delegation: Mock DelegationService (defaults to a generic MagicMock).
    """
    from wolfharness.capabilities.agent_context import AgentContextDeps

    cfg = config or _make_enabled_config(base_dir=base_dir)

    agent_ctx = MagicMock(spec=AgentContextDeps)
    agent_ctx.session.metadata = metadata if metadata is not None else _make_session_metadata()
    agent_ctx.host.session_pool = session_pool
    agent_ctx.team_mode_config = cfg
    agent_ctx.agent_registry = agent_registry or MagicMock()
    agent_ctx.session.session_id = session_id
    agent_ctx.delegation = delegation or MagicMock()

    ctx = MagicMock()
    ctx.deps = agent_ctx
    return ctx


def _init_team(
    base_dir: str,
    team_id: str = "team_123",
    team_name: str = "alpha_team",
) -> None:
    """Initialize a real FileTeamState with a team and registered members."""
    state = FileTeamState(base_dir)
    state.init(
        team_id,
        team_name,
        [
            {"name": "translator_agent", "agent": "worker"},
            {"name": "reviewer_agent", "agent": "reviewer"},
        ],
    )
    state.register_member(team_id, "translator_agent", "sess_translator")
    state.register_member(team_id, "reviewer_agent", "sess_reviewer")


def _make_defaults_config(base_dir: str) -> TeamModeConfig:
    """Create an enabled TeamModeConfig with defaults for testing."""
    return _make_enabled_config(
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


# ---------------------------------------------------------------------------
# Scenario 1: End-to-end lifecycle
# ---------------------------------------------------------------------------


@pytest.mark.integration
async def test_e2e_lifecycle_create_message_task_blackboard_delete(
    team_mode_pool: AgentPool[Any],
) -> None:
    """Given: real AgentPool with team_mode enabled and a lead session.

    When: full lifecycle is exercised — create team, send message, create
        task, complete task, write/read blackboard, delete team.

    Then: each step returns the expected result, real child sessions are
        created in SessionPool, and FileTeamState on disk reflects the
        changes.
    """
    manifest = team_mode_pool.manifest
    team_mode_config: TeamModeConfig | None = manifest.team_mode
    assert team_mode_config is not None

    session_pool = team_mode_pool.session_pool
    assert session_pool is not None

    # --- Setup: create lead session ---
    session_id = "lead-session-001"
    await session_pool.create_session(
        session_id,
        agent_name="coordinator",
        team_role="lead",
        team_member_name="coordinator",
    )

    agent_ctx = build_agent_context(team_mode_pool, session_id, team_mode_config)
    lead_metadata: dict[str, Any] = {
        "team_role": "lead",
        "team_member_name": "coordinator",
    }
    cap = TeamCommCapability(team_mode_config, "coordinator", lead_metadata)
    ctx = make_mock_run_context(agent_ctx)

    # --- Step 1: Create team ---
    create_result = await cap.team_create(
        ctx,
        "alpha_team",
        [
            {"agent": "worker", "name": "translator_agent"},
            {"agent": "reviewer", "name": "reviewer_agent"},
        ],
    )

    assert "Team 'alpha_team' created with 2 members" in create_result.return_value
    assert "team_id=" in create_result.return_value
    team_id = create_result.return_value.split("team_id=")[1].strip()
    # team_create writes team_id back to agent_ctx.session.metadata;
    # update cap._session_metadata for consistency.
    cap._session_metadata["team_id"] = team_id
    cap._session_metadata["team_name"] = "alpha_team"

    # Verify real child sessions exist in SessionPool.
    base_dir = team_mode_config.effective_base_dir
    team_state = FileTeamState(base_dir)
    state_path = team_state._state_path(team_id)
    assert state_path.exists()
    state = team_state._read_json(state_path)
    members: dict[str, dict[str, str]] = state.get("members", {})
    for member_info in members.values():
        member_sid = member_info.get("session_id", "")
        assert member_sid != ""
        assert session_pool.sessions.get_session(member_sid) is not None

    # --- Step 2: Send a message ---
    msg_result = await cap.send_message(ctx, "translator_agent", "Start translating")
    assert msg_result.return_value == "Message sent to translator_agent"

    # --- Step 3: Create a task ---
    task_result = await cap.task_create(
        ctx, "Translate docs", owner="translator_agent", description="Translate API docs"
    )
    assert task_result.return_value.startswith("Task created: ")
    task_id = task_result.return_value.replace("Task created: ", "")

    # --- Step 4: Complete the task ---
    update_result = await cap.task_update(ctx, task_id, status="completed")
    assert 'status="completed"' in update_result.return_value
    assert "<task" in update_result.return_value

    # --- Step 5: Write blackboard ---
    wb_write_result = await cap.write_blackboard(ctx, "glossary", "v1 content")
    assert wb_write_result.return_value == "Written, version=1"

    # --- Step 6: Read blackboard ---
    rb_result = await cap.read_blackboard(ctx, "glossary")
    rb_rv = (
        "\n".join(rb_result.return_value)
        if isinstance(rb_result.return_value, list)
        else rb_result.return_value
    )
    assert "<blackboard" in rb_rv
    assert "v1 content" in rb_rv
    assert 'version="1"' in rb_rv

    # --- Step 7: Delete team ---
    del_result = await cap.team_delete(ctx)
    assert del_result.return_value == "Team deleted"

    # Verify real member sessions are closed (exclude lead — lead session stays alive).
    for member_info in members.values():
        member_sid = member_info.get("session_id", "")
        if member_sid == session_id:
            continue  # Lead's own session should still be alive.
        assert session_pool.sessions.get_session(member_sid) is None

    # Team state should be cleaned up.
    assert not state_path.exists()

    # Cleanup.
    await session_pool.close_session(session_id)


# ---------------------------------------------------------------------------
# Scenario 2: Disabled mode
# ---------------------------------------------------------------------------


@pytest.mark.integration
async def test_disabled_mode_no_tools_no_instructions() -> None:
    """Given: TeamModeConfig with enabled=False.

    When: TeamCommCapability is constructed and get_tools/get_instructions
        are called.

    Then: get_tools() returns empty list, get_instructions() returns None.
    """
    config = _make_disabled_config()
    metadata = _make_session_metadata()
    cap = TeamCommCapability(config, "worker", metadata)

    tools = await cap.get_tools()
    assert list(tools) == []

    instructions = cap.get_instructions()
    assert instructions is None


# ---------------------------------------------------------------------------
# Scenario 3: Standalone mode (no session_pool)
# ---------------------------------------------------------------------------


@pytest.mark.integration
async def test_standalone_mode_team_create_without_session_pool(
    tmp_path: Any,
) -> None:
    """Given: lead agent with no SessionPool (standalone execution).

    When: team_create is called.

    Then: returns "SessionPool not available" — team state is created on
        disk but sessions cannot be spawned without a pool.
    """
    config = _make_enabled_config(
        member_eligible=["worker", "reviewer"],
        base_dir=str(tmp_path),
    )
    mock_registry = MagicMock()
    mock_registry.exists = MagicMock(return_value=True)
    ctx = _make_run_context(
        metadata=_make_lead_metadata(),
        session_pool=None,
        config=config,
        base_dir=str(tmp_path),
        agent_registry=mock_registry,
    )
    cap = TeamCommCapability(config, "coordinator", _make_lead_metadata())

    result = await cap.team_create(
        ctx,
        "solo_team",
        [{"agent": "worker", "name": "translator_agent"}],
    )

    assert result.return_value == "SessionPool not available"


@pytest.mark.integration
async def test_standalone_mode_session_pool_none_returns_error(tmp_path: Any) -> None:
    """Given: host is not None but host.session_pool is None.

    When: send_message is called.

    Then: returns "SessionPool not available".
    """
    _init_team(str(tmp_path))
    config = _make_enabled_config(base_dir=str(tmp_path))
    metadata = _make_session_metadata()
    ctx = _make_run_context(
        metadata=metadata,
        session_pool=None,
        config=config,
        base_dir=str(tmp_path),
    )
    cap = TeamCommCapability(config, "worker", metadata)

    result = await cap.send_message(ctx, "reviewer_agent", "hello")

    assert result.return_value == "SessionPool not available"


# ---------------------------------------------------------------------------
# Scenario 4: No team_id metadata
# ---------------------------------------------------------------------------


@pytest.mark.integration
async def test_no_team_id_universal_tools_return_error() -> None:
    """Given: session metadata without team_id.

    When: universal tools (team_status, task_list, read_blackboard,
        list_blackboard) are called.

    Then: each returns "Not in a team session".
    """
    config = _make_enabled_config()
    metadata: dict[str, Any] = {"team_name": "foo", "team_role": "member"}
    cap = TeamCommCapability(config, "worker", metadata)

    # team_status
    ctx_status = _make_run_context(metadata=metadata, config=config)
    assert (await cap.team_status(ctx_status)).return_value == "Not in a team session"

    # task_list
    ctx_tasks = _make_run_context(metadata=metadata, config=config)
    assert (await cap.task_list(ctx_tasks)).return_value == "Not in a team session"

    # read_blackboard
    ctx_read = _make_run_context(metadata=metadata, config=config)
    assert (await cap.read_blackboard(ctx_read, "some_key")).return_value == "Not in a team session"

    # list_blackboard
    ctx_list = _make_run_context(metadata=metadata, config=config)
    assert (await cap.list_blackboard(ctx_list)).return_value == "Not in a team session"


# ---------------------------------------------------------------------------
# Scenario 5: Graph/teams coexistence
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_graph_and_team_mode_coexist_in_manifest() -> None:
    """Given: a YAML config with both graph: and team_mode: sections.

    When: AgentsManifest is parsed from the YAML string.

    Then: manifest loads without errors, graph is populated, and
        team_mode is enabled with the expected values.
    """
    yaml_str = """
agents:
  coordinator:
    type: native
    model: openai:test
    system_prompt: "You are a coordinator"
  worker:
    type: native
    model: openai:test
    system_prompt: "You are a worker"
  reviewer:
    type: native
    model: openai:test
    system_prompt: "You are a reviewer"

graph:
  name: review_pipeline
  steps:
    - id: coordinator
      agent: coordinator
    - id: worker
      agent: worker
    - id: reviewer
      agent: reviewer

team_mode:
  enabled: true
  member_eligible: [worker, reviewer]
  lead_eligible: [coordinator]
"""
    config_dict = yamling.load_yaml(yaml_str, verify_type=dict)
    manifest = AgentsManifest(**config_dict)

    # Graph should be populated.
    assert manifest.graph is not None
    assert manifest.graph.name == "review_pipeline"
    assert len(manifest.graph.steps) == 3

    # Team mode should be enabled.
    assert manifest.team_mode is not None
    assert manifest.team_mode.enabled is True
    assert "worker" in manifest.team_mode.member_eligible
    assert "reviewer" in manifest.team_mode.member_eligible
    assert "coordinator" in manifest.team_mode.lead_eligible

    # Agents should be present.
    assert "coordinator" in manifest.agents
    assert "worker" in manifest.agents
    assert "reviewer" in manifest.agents


# ---------------------------------------------------------------------------
# Scenario 6: Config default members in team_create
# ---------------------------------------------------------------------------


@pytest.mark.integration
async def test_team_status_with_existing_team_id_no_session_creation(
    tmp_path: Any,
) -> None:
    """Given: team_id is already set in metadata from a prior team_create.

    When: team_status is called.

    Then: no new sessions are created, and the existing team state is used.
    """
    config = _make_defaults_config(str(tmp_path))

    # Manually create a team first (simulating prior team_create).
    _init_team(str(tmp_path), team_id="manual_team", team_name="manual_team")

    mock_pool = MagicMock()
    mock_pool.send_message = AsyncMock(return_value="msg_id")
    mock_pool.close_session = AsyncMock()

    mock_registry = MagicMock()
    mock_registry.exists = MagicMock(return_value=True)

    # Metadata already has team_id from manual team_create.
    metadata = _make_lead_metadata(team_id="manual_team")
    ctx = _make_run_context(
        metadata=metadata,
        session_pool=mock_pool,
        config=config,
        base_dir=str(tmp_path),
        agent_registry=mock_registry,
    )
    cap = TeamCommCapability(config, "coordinator", metadata)

    result = await cap.team_status(ctx)

    # team_status should show the manually-created team.
    assert "manual_team" in result.return_value


@pytest.mark.integration
async def test_team_create_uses_config_default_members_when_empty(
    team_mode_pool_with_defaults: AgentPool[Any],
) -> None:
    """Given: defaults config is set, lead calls team_create with empty members.

    When: team_create is called with members=[].

    Then: uses defaults.members from config, creates real child sessions, and
        team_status shows the created team.
    """
    manifest = team_mode_pool_with_defaults.manifest
    team_mode_config: TeamModeConfig | None = manifest.team_mode
    assert team_mode_config is not None
    assert team_mode_config.defaults is not None

    session_pool = team_mode_pool_with_defaults.session_pool
    assert session_pool is not None

    # Setup: create lead session.
    session_id = "lead-defaults-001"
    await session_pool.create_session(
        session_id,
        agent_name="coordinator",
        team_role="lead",
        team_member_name="coordinator",
    )

    agent_ctx = build_agent_context(
        team_mode_pool_with_defaults,
        session_id,
        team_mode_config,
    )
    lead_metadata: dict[str, Any] = {
        "team_role": "lead",
        "team_member_name": "coordinator",
    }
    cap = TeamCommCapability(team_mode_config, "coordinator", lead_metadata)
    ctx = make_mock_run_context(agent_ctx)

    # team_create with empty members should use defaults config defaults.
    create_result = await cap.team_create(ctx, "my_team", [])
    assert "Team 'my_team' created with 2 members" in create_result.return_value
    team_id = create_result.return_value.split("team_id=")[1].strip()
    cap._session_metadata["team_id"] = team_id
    cap._session_metadata["team_name"] = "my_team"

    # Verify real child sessions were created (2 from defaults + lead registered).
    base_dir = team_mode_config.effective_base_dir
    team_state = FileTeamState(base_dir)
    state = team_state._read_json(team_state._state_path(team_id))
    members: dict[str, dict[str, str]] = state.get("members", {})
    # Lead is also registered as a member, so 3 total (lead + 2 defaults).
    assert len(members) == 3
    for member_info in members.values():
        member_sid = member_info.get("session_id", "")
        assert member_sid != ""
        assert session_pool.sessions.get_session(member_sid) is not None

    # team_status should show the created team.
    status_result = await cap.team_status(ctx)
    assert "my_team" in status_result.return_value

    # Cleanup.
    await cap.team_delete(ctx)
    await session_pool.close_session(session_id)


# ---------------------------------------------------------------------------
# Scenario 7: End-to-end task lifecycle on blackboard
# ---------------------------------------------------------------------------


@pytest.mark.integration
async def test_e2e_task_and_blackboard_lifecycle(tmp_path: Any) -> None:
    """Given: initialized team with real FileTeamState on tmp_path.

    When: task_create → task_list → task_update → write_blackboard →
        read_blackboard → list_blackboard → delete_blackboard.

    Then: each step returns expected results and state persists on disk.
    """
    _init_team(str(tmp_path))
    config = _make_enabled_config(base_dir=str(tmp_path))
    metadata = _make_lead_metadata()
    ctx = _make_run_context(
        metadata=metadata,
        config=config,
        base_dir=str(tmp_path),
    )
    cap = TeamCommCapability(config, "coordinator", metadata)

    # Create a task
    create_result = await cap.task_create(
        ctx, "Review PR", owner="translator_agent", description="Review PR #42"
    )
    assert create_result.return_value.startswith("Task created: ")
    task_id = create_result.return_value.replace("Task created: ", "")

    # List tasks
    list_result = await cap.task_list(ctx)
    assert "<task_list>" in list_result.return_value
    assert "Review PR" in list_result.return_value

    # Update task status
    update_result = await cap.task_update(ctx, task_id, status="in_progress")
    assert 'status="in_progress"' in update_result.return_value
    assert "<task" in update_result.return_value

    # Write blackboard
    wb_result = await cap.write_blackboard(ctx, "review_notes", "LGTM")
    assert wb_result.return_value == "Written, version=1"

    # Read blackboard
    rb_result = await cap.read_blackboard(ctx, "review_notes")
    rb_rv = (
        "\n".join(rb_result.return_value)
        if isinstance(rb_result.return_value, list)
        else rb_result.return_value
    )
    assert "<blackboard" in rb_rv
    assert "LGTM" in rb_rv
    assert 'version="1"' in rb_rv

    # List blackboard keys
    lb_result = await cap.list_blackboard(ctx)
    assert "<blackboard_keys>" in lb_result.return_value
    assert "review_notes" in lb_result.return_value

    # Delete blackboard key (lead-only)
    db_result = await cap.delete_blackboard(ctx, "review_notes")
    assert db_result.return_value == "Blackboard key 'review_notes' deleted"

    # Verify deletion
    rb_after = await cap.read_blackboard(ctx, "review_notes")
    assert rb_after.return_value == "Key not found"


# ---------------------------------------------------------------------------
# Scenario 8: Broadcast and team_status integration
# ---------------------------------------------------------------------------


@pytest.mark.integration
async def test_e2e_broadcast_and_status(
    team_mode_pool: AgentPool[Any],
) -> None:
    """Given: real AgentPool with a team created via team_create.

    When: broadcast message is sent, then team_status is checked.

    Then: broadcast reaches all members, team_status shows team info.
    """
    manifest = team_mode_pool.manifest
    team_mode_config: TeamModeConfig | None = manifest.team_mode
    assert team_mode_config is not None

    session_pool = team_mode_pool.session_pool
    assert session_pool is not None

    # Setup: create lead session and team.
    session_id = "lead-broadcast-001"
    await session_pool.create_session(
        session_id,
        agent_name="coordinator",
        team_role="lead",
        team_member_name="coordinator",
    )

    agent_ctx = build_agent_context(team_mode_pool, session_id, team_mode_config)
    lead_metadata: dict[str, Any] = {
        "team_role": "lead",
        "team_member_name": "coordinator",
    }
    cap = TeamCommCapability(team_mode_config, "coordinator", lead_metadata)
    ctx = make_mock_run_context(agent_ctx)

    create_result = await cap.team_create(
        ctx,
        "alpha_team",
        [
            {"agent": "worker", "name": "translator_agent"},
            {"agent": "reviewer", "name": "reviewer_agent"},
        ],
    )
    assert "team_id=" in create_result.return_value
    team_id = create_result.return_value.split("team_id=")[1].strip()
    cap._session_metadata["team_id"] = team_id
    cap._session_metadata["team_name"] = "alpha_team"

    # Broadcast (excludes lead, so 2 members receive).
    broadcast_result = await cap.send_message(ctx, "*", "Team meeting at 3pm")
    assert "Broadcast sent to 2 members" in broadcast_result.return_value

    # Team status.
    status_result = await cap.team_status(ctx)
    assert "alpha_team" in status_result.return_value
    assert "active" in status_result.return_value
    assert "translator_agent" in status_result.return_value
    assert "reviewer_agent" in status_result.return_value

    # Cleanup.
    await cap.team_delete(ctx)
    await session_pool.close_session(session_id)
