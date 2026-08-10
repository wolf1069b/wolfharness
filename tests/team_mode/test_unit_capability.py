"""Tests for TeamCommCapability skeleton, registration, and per-session."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from pydantic_ai.tools import ToolDefinition
import pytest

from wolfharness.capabilities.team_comm_capability import TeamCommCapability
from wolfharness_config.team_mode import TeamBounds, TeamModeConfig


# ---- Helpers ----


def _make_enabled_config(
    *,
    member_eligible: list[str] | None = None,
    lead_eligible: list[str] | None = None,
    protocol_template: str | None = None,
    base_dir: str | None = None,
    notice_delivery_mode: str = "steer",
    max_watch_timeout: int = 120,
) -> TeamModeConfig:
    """Create an enabled TeamModeConfig for testing.

    Args:
        member_eligible: Agent names eligible as members.
        lead_eligible: Agent names eligible as leads.
        protocol_template: Custom protocol template string.
        base_dir: Base directory for team state files.
        notice_delivery_mode: Delivery mode for notifications.
        max_watch_timeout: Max watch timeout in seconds.

    Returns:
        A frozen TeamModeConfig with enabled=True.
    """
    return TeamModeConfig(
        enabled=True,
        member_eligible=member_eligible or ["worker"],
        lead_eligible=lead_eligible or ["coordinator"],
        protocol_template=protocol_template
        or "Team={team_name}, Role={role}, Member={member_name}",
        base_dir=base_dir,
        notice_delivery_mode=notice_delivery_mode,
        max_watch_timeout=max_watch_timeout,
    )


def _make_disabled_config() -> TeamModeConfig:
    """Create a disabled TeamModeConfig for testing."""
    return TeamModeConfig(
        enabled=False,
        member_eligible=["worker"],
        lead_eligible=["coordinator"],
    )


def _make_session_metadata() -> dict[str, Any]:
    """Create typical session metadata for a team session."""
    return {
        "team_id": "team_123",
        "team_name": "alpha_team",
        "team_role": "translator",
        "team_member_name": "translator_agent",
    }


# ---- Skeleton tests ----


@pytest.mark.unit
def test_skeleton_get_instructions_renders_template_with_metadata() -> None:
    """Given: enabled config + session metadata.

    When: get_instructions() is called.
    Then: returns rendered protocol template with actual metadata values.
    """
    config = _make_enabled_config()
    metadata = _make_session_metadata()
    cap = TeamCommCapability(config, "worker", metadata)

    result = cap.get_instructions()

    assert result is not None
    assert "alpha_team" in result
    assert "translator" in result
    assert "translator_agent" in result


@pytest.mark.unit
def test_skeleton_get_instructions_returns_none_when_disabled() -> None:
    """Given: disabled config + session metadata.

    When: get_instructions() is called.
    Then: returns None.
    """
    config = _make_disabled_config()
    metadata = _make_session_metadata()
    cap = TeamCommCapability(config, "worker", metadata)

    result = cap.get_instructions()

    assert result is None


@pytest.mark.unit
def test_skeleton_get_instructions_returns_none_when_no_metadata() -> None:
    """Given: enabled config + session_metadata=None.

    When: get_instructions() is called.
    Then: returns None (no session context to render template with).
    """
    config = _make_enabled_config()
    cap = TeamCommCapability(config, "worker", session_metadata=None)

    result = cap.get_instructions()

    assert result is None


@pytest.mark.unit
def test_skeleton_get_instructions_returns_none_when_empty_metadata() -> None:
    """Given: enabled config + empty session metadata dict.

    When: get_instructions() is called.
    Then: returns None.
    """
    config = _make_enabled_config()
    cap = TeamCommCapability(config, "worker", session_metadata={})

    result = cap.get_instructions()

    assert result is None


@pytest.mark.unit
async def test_skeleton_get_tools_returns_15_tools_when_enabled() -> None:
    """Given: enabled config with T9 universal tools + T6 lead-only tools.

    When: get_tools() is called.
    Then: returns 15 tools (send_message, task_create, task_list,
        task_update, task_get, read_blackboard, write_blackboard,
        list_blackboard, team_status, team_create, team_delete,
        delete_blackboard, shutdown_request, team_add_member,
        task_create_batch).
    """
    config = _make_enabled_config()
    cap = TeamCommCapability(config, "worker", _make_session_metadata())

    result = await cap.get_tools()

    tool_names = {t.name for t in result}
    assert tool_names == {
        "send_message",
        "task_create",
        "task_list",
        "task_update",
        "task_get",
        "read_blackboard",
        "write_blackboard",
        "list_blackboard",
        "team_status",
        "team_create",
        "team_delete",
        "delete_blackboard",
        "shutdown_request",
        "team_add_member",
        "task_create_batch",
    }


@pytest.mark.unit
async def test_skeleton_get_tools_returns_empty_when_disabled() -> None:
    """Given: disabled config.

    When: get_tools() is called.
    Then: returns empty list.
    """
    config = _make_disabled_config()
    cap = TeamCommCapability(config, "worker", _make_session_metadata())

    result = await cap.get_tools()

    assert list(result) == []


@pytest.mark.unit
def test_skeleton_get_instructions_uses_agent_name_as_default_member() -> None:
    """Given: enabled config + metadata without team_member_name key.

    When: get_instructions() is called.
    Then: uses agent_name as the default member_name.
    """
    config = _make_enabled_config()
    metadata: dict[str, Any] = {"team_name": "beta", "team_role": "lead"}
    cap = TeamCommCapability(config, "coordinator", metadata)

    result = cap.get_instructions()

    assert result is not None
    assert "coordinator" in result
    assert "beta" in result
    assert "lead" in result


@pytest.mark.unit
def test_skeleton_get_instructions_uses_unknown_for_missing_keys() -> None:
    """Given: enabled config + metadata with only team_id.

    When: get_instructions() is called.
    Then: uses 'unknown' for missing team_name and team_role.
    """
    config = _make_enabled_config()
    metadata: dict[str, Any] = {"team_id": "t1"}
    cap = TeamCommCapability(config, "worker", metadata)

    result = cap.get_instructions()

    assert result is not None
    assert "unknown" in result


# ---- Registration tests ----


def _make_factory() -> Any:
    """Create a minimal AgentFactory for testing _compile_agent_capabilities.

    Returns an AgentFactory with a mock pool (the method under test
    does not access self._pool).
    """
    from wolfharness.host.factory import AgentFactory

    mock_pool = MagicMock()
    return AgentFactory(mock_pool)


def _make_host_context(team_mode: Any) -> Any:
    """Create a mock HostContext with the given manifest.team_mode.

    Args:
        team_mode: Value to return for host_context.manifest.team_mode.

    Returns:
        A MagicMock configured with .manifest.team_mode and .skills_tools_provider=None.
    """
    mock = MagicMock()
    mock.manifest.team_mode = team_mode
    mock.skill_capabilities = []
    return mock


def _make_native_config(team_mode: Any = None) -> Any:
    """Create a minimal NativeAgentConfig for testing.

    Args:
        team_mode: TeamModeConfig or None for the per-agent overlay.

    Returns:
        A NativeAgentConfig with model='openai:test' and no tools.
    """
    from wolfharness.models.agents import NativeAgentConfig

    return NativeAgentConfig(
        name="test_agent",
        model="openai:test",
        tools=[],
        team_mode=team_mode,
    )


@pytest.mark.unit
def test_registration_team_comm_added_when_enabled_and_eligible() -> None:
    """Given: enabled global team_mode, agent in member_eligible.

    When: _compile_agent_capabilities() is called.
    Then: TeamCommCapability is present in the returned capability list.
    """
    config = _make_enabled_config(member_eligible=["test_agent"])
    factory = _make_factory()
    host_ctx = _make_host_context(team_mode=config)
    cfg = _make_native_config()

    caps = factory._compile_agent_capabilities("test_agent", cfg, host_ctx)

    team_caps = [c for c in caps if isinstance(c, TeamCommCapability)]
    assert len(team_caps) == 1
    assert team_caps[0]._agent_name == "test_agent"
    # Shared instance at compile time has no session metadata.
    assert team_caps[0]._session_metadata == {}


@pytest.mark.unit
def test_registration_no_team_comm_when_disabled() -> None:
    """Given: disabled global team_mode, agent in member_eligible.

    When: _compile_agent_capabilities() is called.
    Then: no TeamCommCapability in the returned list.
    """
    config = _make_disabled_config()
    factory = _make_factory()
    host_ctx = _make_host_context(team_mode=config)
    cfg = _make_native_config()

    caps = factory._compile_agent_capabilities("worker", cfg, host_ctx)

    team_caps = [c for c in caps if isinstance(c, TeamCommCapability)]
    assert len(team_caps) == 0


@pytest.mark.unit
def test_registration_no_team_comm_when_agent_not_eligible() -> None:
    """Given: enabled global team_mode, agent NOT in any eligible list.

    When: _compile_agent_capabilities() is called.
    Then: no TeamCommCapability in the returned list.
    """
    config = _make_enabled_config(
        member_eligible=["other_agent"],
        lead_eligible=["other_lead"],
    )
    factory = _make_factory()
    host_ctx = _make_host_context(team_mode=config)
    cfg = _make_native_config()

    caps = factory._compile_agent_capabilities("test_agent", cfg, host_ctx)

    team_caps = [c for c in caps if isinstance(c, TeamCommCapability)]
    assert len(team_caps) == 0


@pytest.mark.unit
def test_registration_team_comm_for_lead_eligible() -> None:
    """Given: enabled global team_mode, agent in lead_eligible.

    When: _compile_agent_capabilities() is called.
    Then: TeamCommCapability is present.
    """
    config = _make_enabled_config(lead_eligible=["coordinator"])
    factory = _make_factory()
    host_ctx = _make_host_context(team_mode=config)
    cfg = _make_native_config()

    caps = factory._compile_agent_capabilities("coordinator", cfg, host_ctx)

    team_caps = [c for c in caps if isinstance(c, TeamCommCapability)]
    assert len(team_caps) == 1


@pytest.mark.unit
def test_registration_no_team_comm_when_global_is_none() -> None:
    """Given: global team_mode is None, per-agent team_mode is None.

    When: _compile_agent_capabilities() is called.
    Then: no TeamCommCapability in the returned list.
    """
    factory = _make_factory()
    host_ctx = _make_host_context(team_mode=None)
    cfg = _make_native_config()

    caps = factory._compile_agent_capabilities("test_agent", cfg, host_ctx)

    team_caps = [c for c in caps if isinstance(c, TeamCommCapability)]
    assert len(team_caps) == 0


# ---- Per-session tests ----


@pytest.mark.unit
def test_per_session_get_instructions_renders_with_actual_metadata() -> None:
    """Given: enabled config + session metadata containing team_id.

    When: TeamCommCapability is constructed with session metadata.
    Then: get_instructions() renders the template with actual metadata values.
    """
    config = _make_enabled_config()
    metadata = _make_session_metadata()
    cap = TeamCommCapability(config, "worker", metadata)

    result = cap.get_instructions()

    assert result is not None
    assert "alpha_team" in result
    assert "translator" in result
    assert "translator_agent" in result


@pytest.mark.unit
def test_per_session_shared_instance_has_no_instructions() -> None:
    """Given: shared instance created at compile time (session_metadata=None).

    When: get_instructions() is called.
    Then: returns None (no session context yet).
    """
    config = _make_enabled_config()
    shared_cap = TeamCommCapability(config, "worker", session_metadata=None)

    result = shared_cap.get_instructions()

    assert result is None


@pytest.mark.unit
def test_per_session_replacement_provides_instructions() -> None:
    """Given: shared instance (no metadata) replaced by per-session instance.

    When: per-session instance's get_instructions() is called.
    Then: returns rendered instructions with actual team metadata.
    """
    config = _make_enabled_config()
    shared_cap = TeamCommCapability(config, "worker", session_metadata=None)
    assert shared_cap.get_instructions() is None

    # Simulate per-session replacement.
    metadata = _make_session_metadata()
    per_session_cap = TeamCommCapability(config, "worker", metadata)

    result = per_session_cap.get_instructions()

    assert result is not None
    assert "alpha_team" in result


# ---- T7 Universal tool tests ----


def _make_run_context(
    metadata: dict[str, Any] | None = None,
    session_pool: MagicMock | None = None,
    config: TeamModeConfig | None = None,
    base_dir: str | None = None,
    agent_registry: MagicMock | None = None,
    session_id: str | None = None,
    delegation: MagicMock | None = None,
) -> MagicMock:
    """Create a mock RunContext with AgentContextDeps deps.

    Args:
        metadata: Session metadata dict (defaults to team session metadata).
        session_pool: Mock SessionPool (or None to test missing pool).
        config: TeamModeConfig (defaults to enabled config).
        base_dir: Optional base_dir override for TeamModeConfig.
        agent_registry: Mock AgentRegistry (defaults to a permissive mock).
        session_id: Optional session_id string for the mock SessionState.
        delegation: Mock DelegationService (defaults to a generic MagicMock).

    Returns:
        A MagicMock whose .deps is a mock AgentContextDeps.
    """
    from wolfharness.capabilities.agent_context import AgentContextDeps

    cfg = config or _make_enabled_config(base_dir=base_dir)

    # Ensure mock pool has needed async methods for _create_member_session.
    # Only set up if not already configured (avoids overriding explicit test setups).
    if session_pool is not None:
        if not isinstance(session_pool.create_child_session, AsyncMock):
            _child_state = MagicMock()
            _child_state.session_id = "child_session"
            session_pool.create_child_session = AsyncMock(return_value=_child_state)
        if not isinstance(
            getattr(session_pool.sessions, "get_or_create_session_agent", None),
            AsyncMock,
        ):
            session_pool.sessions = MagicMock()
            session_pool.sessions.get_or_create_session_agent = AsyncMock()
        # event_bus: set to None unless explicitly configured as MagicMock with publish
        _eb = session_pool.event_bus
        if not (
            isinstance(_eb, MagicMock) and isinstance(getattr(_eb, "publish", None), AsyncMock)
        ):
            session_pool.event_bus = None

    agent_ctx = MagicMock(spec=AgentContextDeps)
    agent_ctx.session.metadata = metadata if metadata is not None else _make_session_metadata()
    agent_ctx.host.session_pool = session_pool
    agent_ctx.team_mode_config = cfg
    agent_ctx.agent_registry = agent_registry or MagicMock()
    agent_ctx.session.session_id = session_id or "lead_session_001"
    agent_ctx.delegation = delegation or MagicMock()

    ctx = MagicMock()
    ctx.deps = agent_ctx
    return ctx


def _make_lead_metadata(team_id: str = "team_123") -> dict[str, Any]:
    """Create session metadata for a lead agent."""
    return {
        "team_id": team_id,
        "team_name": "alpha_team",
        "team_role": "lead",
        "team_member_name": "coordinator",
    }


def _init_team(base_dir: str, team_id: str = "team_123") -> None:
    """Initialize a real FileTeamState with a team and members."""
    from wolfharness.capabilities.file_team_state import FileTeamState

    state = FileTeamState(base_dir)
    state.init(
        team_id,
        "alpha_team",
        [
            {"name": "translator_agent", "agent": "worker"},
            {"name": "reviewer_agent", "agent": "reviewer"},
        ],
    )
    state.register_member(team_id, "translator_agent", "sess_translator")
    state.register_member(team_id, "reviewer_agent", "sess_reviewer")


def _make_add_member_setup(
    tmp_path: Any,
    *,
    member_eligible: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> tuple[TeamCommCapability, Any, MagicMock, MagicMock, MagicMock]:
    """Create a standard setup for team_add_member tests.

    Returns:
        (capability, ctx, mock_pool, mock_delegation, mock_registry)
    """
    _init_team(str(tmp_path))
    config = _make_enabled_config(
        member_eligible=member_eligible or ["worker", "reviewer", "editor"],
        base_dir=str(tmp_path),
    )
    mock_registry = MagicMock()
    mock_registry.exists = MagicMock(return_value=True)
    mock_pool = MagicMock()
    mock_pool.send_message = AsyncMock(return_value="msg_id")
    mock_pool.close_session = AsyncMock()
    # Mock create_child_session to return a state with session_id
    mock_child_state = MagicMock()
    mock_child_state.session_id = "child_session_new"
    mock_pool.create_child_session = AsyncMock(return_value=mock_child_state)
    mock_pool.sessions = MagicMock()
    mock_pool.sessions.get_or_create_session_agent = AsyncMock()
    mock_pool.event_bus = None  # Skip SpawnSessionStart emission in unit tests
    mock_delegation = MagicMock()
    mock_delegation.create_child_session = AsyncMock(return_value="child_session_new")
    ctx = _make_run_context(
        metadata=metadata or _make_lead_metadata(),
        session_pool=mock_pool,
        config=config,
        base_dir=str(tmp_path),
        agent_registry=mock_registry,
        delegation=mock_delegation,
    )
    cap = TeamCommCapability(config, "coordinator", metadata or _make_lead_metadata())
    return cap, ctx, mock_pool, mock_delegation, mock_registry


@pytest.mark.unit
async def test_send_message_happy_path(tmp_path: Any) -> None:
    """Given: team session with registered members + mock session_pool.

    When: send_message is called with valid recipient.
    Then: returns "Message sent to {to}" and session_pool.send_message called.
    """
    _init_team(str(tmp_path))
    mock_pool = MagicMock()
    mock_pool.send_message = AsyncMock(return_value="msg_id_123")
    ctx = _make_run_context(session_pool=mock_pool, base_dir=str(tmp_path))
    config = _make_enabled_config(base_dir=str(tmp_path))
    cap = TeamCommCapability(config, "worker", _make_session_metadata())

    result = await cap.send_message(ctx, "reviewer_agent", "hello")

    assert result.return_value == "Message sent to reviewer_agent"
    mock_pool.send_message.assert_awaited_once()


@pytest.mark.unit
async def test_send_message_broadcast_returns_error() -> None:
    """Given: team session.

    When: send_message is called with to='*'.
    Then: returns "Broadcast is lead-only" error.
    """
    ctx = _make_run_context()
    cap = TeamCommCapability(_make_enabled_config(), "worker", _make_session_metadata())

    result = await cap.send_message(ctx, "*", "announcement")

    assert result.return_value == "Broadcast is lead-only"


@pytest.mark.unit
async def test_send_message_no_team_id() -> None:
    """Given: session metadata without team_id.

    When: send_message is called.
    Then: returns "Not in a team session".
    """
    ctx = _make_run_context(metadata={"team_name": "foo"})
    cap = TeamCommCapability(_make_enabled_config(), "worker", {"team_name": "foo"})

    result = await cap.send_message(ctx, "reviewer_agent", "hello")

    assert result.return_value == "Not in a team session"


@pytest.mark.unit
async def test_send_message_no_session_pool(tmp_path: Any) -> None:
    """Given: team session but session_pool is None.

    When: send_message is called.
    Then: returns "SessionPool not available".
    """
    _init_team(str(tmp_path))
    ctx = _make_run_context(session_pool=None, base_dir=str(tmp_path))
    config = _make_enabled_config(base_dir=str(tmp_path))
    cap = TeamCommCapability(config, "worker", _make_session_metadata())

    result = await cap.send_message(ctx, "reviewer_agent", "hello")

    assert result.return_value == "SessionPool not available"


@pytest.mark.unit
async def test_send_message_member_not_found(tmp_path: Any) -> None:
    """Given: team session but recipient not registered.

    When: send_message is called with unknown member.
    Then: returns error about member not found.
    """
    _init_team(str(tmp_path))
    mock_pool = MagicMock()
    ctx = _make_run_context(session_pool=mock_pool, base_dir=str(tmp_path))
    config = _make_enabled_config(base_dir=str(tmp_path))
    cap = TeamCommCapability(config, "worker", _make_session_metadata())

    result = await cap.send_message(ctx, "nonexistent", "hello")

    assert "not found" in result.return_value


@pytest.mark.unit
async def test_send_message_uses_steer_by_default(tmp_path: Any) -> None:
    """Given: team session, default config (notice_delivery_mode=steer).

    When: send_message is called.
    Then: session_pool.send_message called with DeliveryMode.STEER.
    """
    from wolfharness.lifecycle.types import DeliveryMode

    _init_team(str(tmp_path))
    mock_pool = MagicMock()
    mock_pool.send_message = AsyncMock(return_value="msg_steer")
    ctx = _make_run_context(session_pool=mock_pool, base_dir=str(tmp_path))
    config = _make_enabled_config(base_dir=str(tmp_path))
    cap = TeamCommCapability(config, "worker", _make_session_metadata())

    result = await cap.send_message(ctx, "reviewer_agent", "test msg")

    assert result.return_value == "Message sent to reviewer_agent"
    call_kwargs = mock_pool.send_message.call_args
    assert call_kwargs.kwargs["mode"] is DeliveryMode.STEER


# ---- T9 Bounds enforcement tests ----


@pytest.mark.unit
async def test_bounds_max_members_exceeded(tmp_path: Any) -> None:
    """Given: lead agent with 4 members but max_members=3.

    When: team_create is called with 4 members.
    Then: returns error about exceeding max_members.
    """
    config = _make_enabled_config(
        member_eligible=["worker", "reviewer", "editor", "writer"],
        base_dir=str(tmp_path),
    )
    config = config.model_copy(update={"bounds": TeamBounds(max_members=3)})
    mock_registry = MagicMock()
    mock_registry.exists = MagicMock(return_value=True)
    mock_pool = MagicMock()
    mock_pool.create_session = AsyncMock()
    mock_pool.send_message = AsyncMock(return_value="msg_id")
    ctx = _make_run_context(
        metadata=_make_lead_metadata(),
        session_pool=mock_pool,
        config=config,
        base_dir=str(tmp_path),
        agent_registry=mock_registry,
    )
    cap = TeamCommCapability(config, "coordinator", _make_lead_metadata())

    result = await cap.team_create(
        ctx,
        "big_team",
        [
            {"agent": "worker", "name": "m1"},
            {"agent": "reviewer", "name": "m2"},
            {"agent": "editor", "name": "m3"},
            {"agent": "writer", "name": "m4"},
        ],
    )

    assert "exceeds max_members" in result.return_value
    assert "4" in result.return_value
    assert "3" in result.return_value


@pytest.mark.unit
async def test_bounds_max_members_ok(tmp_path: Any) -> None:
    """Given: lead agent with 2 members and max_members=3.

    When: team_create is called with 2 members.
    Then: returns success message (within bounds).
    """
    config = _make_enabled_config(
        member_eligible=["worker", "reviewer"],
        base_dir=str(tmp_path),
    )
    config = config.model_copy(update={"bounds": TeamBounds(max_members=3)})
    mock_registry = MagicMock()
    mock_registry.exists = MagicMock(return_value=True)
    mock_pool = MagicMock()
    mock_pool.send_message = AsyncMock(return_value="msg_id")
    mock_delegation = MagicMock()
    mock_delegation.create_child_session = AsyncMock(return_value="child_session_001")
    ctx = _make_run_context(
        metadata=_make_lead_metadata(),
        session_pool=mock_pool,
        config=config,
        base_dir=str(tmp_path),
        agent_registry=mock_registry,
        delegation=mock_delegation,
    )
    cap = TeamCommCapability(config, "coordinator", _make_lead_metadata())

    result = await cap.team_create(
        ctx,
        "ok_team",
        [
            {"agent": "worker", "name": "translator_agent"},
            {"agent": "reviewer", "name": "reviewer_agent"},
        ],
    )

    assert "Team 'ok_team' created with 2 members" in result.return_value


@pytest.mark.unit
async def test_bounds_started_at_recorded(tmp_path: Any) -> None:
    """Given: lead agent successfully creates a team.

    When: team_create completes.
    Then: state.json contains a 'started_at' field with an ISO timestamp.
    """
    from wolfharness.capabilities.file_team_state import FileTeamState

    config = _make_enabled_config(
        member_eligible=["worker", "reviewer"],
        base_dir=str(tmp_path),
    )
    mock_registry = MagicMock()
    mock_registry.exists = MagicMock(return_value=True)
    mock_pool = MagicMock()
    mock_pool.send_message = AsyncMock(return_value="msg_id")
    mock_delegation = MagicMock()
    mock_delegation.create_child_session = AsyncMock(return_value="child_session_001")
    ctx = _make_run_context(
        metadata=_make_lead_metadata(),
        session_pool=mock_pool,
        config=config,
        base_dir=str(tmp_path),
        agent_registry=mock_registry,
        delegation=mock_delegation,
    )
    cap = TeamCommCapability(config, "coordinator", _make_lead_metadata())

    result = await cap.team_create(
        ctx,
        "timed_team",
        [
            {"agent": "worker", "name": "translator_agent"},
            {"agent": "reviewer", "name": "reviewer_agent"},
        ],
    )

    assert "team_id=" in result.return_value
    team_id = result.return_value.split("team_id=")[1].strip()
    state = FileTeamState._read_json(FileTeamState(str(tmp_path))._state_path(team_id))
    assert "started_at" in state
    assert state["started_at"] is not None
    import datetime

    datetime.datetime.fromisoformat(state["started_at"])


@pytest.mark.unit
async def test_bounds_inbox_max_bytes_exceeded(tmp_path: Any) -> None:
    """Given: team session with inbox_max_bytes set very small.

    When: send_message is called with a body that would exceed the inbox limit.
    Then: returns error about inbox exceeding max size.
    """
    _init_team(str(tmp_path))
    config = _make_enabled_config(base_dir=str(tmp_path))
    config = config.model_copy(update={"inbox_max_bytes": 50})
    mock_pool = MagicMock()
    mock_pool.send_message = AsyncMock(return_value="msg_id")
    ctx = _make_run_context(
        session_pool=mock_pool,
        config=config,
        base_dir=str(tmp_path),
    )
    cap = TeamCommCapability(config, "worker", _make_session_metadata())

    # First message should succeed (inbox is empty, body is small).
    result1 = await cap.send_message(ctx, "reviewer_agent", "hi")
    assert result1.return_value == "Message sent to reviewer_agent"

    # Second message with a large body should exceed the inbox limit.
    big_body = "x" * 100
    result2 = await cap.send_message(ctx, "reviewer_agent", big_body)

    assert "Inbox exceeds max size" in result2.return_value


@pytest.mark.unit
async def test_bounds_max_member_turns_exceeded(tmp_path: Any) -> None:
    """Given: team session where recipient has reached max_member_turns.

    When: send_message is called for that recipient.
    Then: returns error about member exceeding max turns.
    """
    _init_team(str(tmp_path))
    config = _make_enabled_config(base_dir=str(tmp_path))
    config = config.model_copy(update={"bounds": TeamBounds(max_member_turns=2)})
    mock_pool = MagicMock()
    mock_pool.send_message = AsyncMock(return_value="msg_id")
    ctx = _make_run_context(
        session_pool=mock_pool,
        config=config,
        base_dir=str(tmp_path),
    )
    cap = TeamCommCapability(config, "worker", _make_session_metadata())

    # Send 2 messages (should succeed, turn_count goes 0->1, 1->2).
    result1 = await cap.send_message(ctx, "reviewer_agent", "msg1")
    assert result1.return_value == "Message sent to reviewer_agent"
    result2 = await cap.send_message(ctx, "reviewer_agent", "msg2")
    assert result2.return_value == "Message sent to reviewer_agent"

    # Third message should be rejected (turn_count=2 >= max_member_turns=2).
    result3 = await cap.send_message(ctx, "reviewer_agent", "msg3")
    assert "exceeded max turns" in result3.return_value
    assert "2" in result3.return_value


@pytest.mark.unit
async def test_bounds_blackboard_max_size_exceeded(tmp_path: Any) -> None:
    """Given: team session with max_size_mb=1 (minimum allowed).

    When: write_blackboard is called with > 1MB of data.
    Then: returns error about blackboard exceeding max size.
    """
    _init_team(str(tmp_path))
    config = _make_enabled_config(base_dir=str(tmp_path))
    config = config.model_copy(
        update={"blackboard": config.blackboard.model_copy(update={"max_size_mb": 1})}
    )
    ctx = _make_run_context(
        config=config,
        base_dir=str(tmp_path),
    )
    cap = TeamCommCapability(config, "worker", _make_session_metadata())

    # Write > 1MB of data.
    big_value = "x" * (1024 * 1024 + 1)
    result = await cap.write_blackboard(ctx, "big_key", big_value)
    assert "Blackboard write exceeds max size" in result.return_value
    assert "MB" in result.return_value


@pytest.mark.unit
async def test_bounds_blackboard_max_size_cleans_up_file(tmp_path: Any) -> None:
    """Given: team session with max_size_mb=1 (minimum allowed).

    When: write_blackboard is called with > 1MB of data.
    Then: returns error about blackboard exceeding max size AND the
          oversized file is deleted from disk so subsequent reads
          return None instead of stale oversized data.
    """
    _init_team(str(tmp_path))
    config = _make_enabled_config(base_dir=str(tmp_path))
    config = config.model_copy(
        update={"blackboard": config.blackboard.model_copy(update={"max_size_mb": 1})}
    )
    ctx = _make_run_context(
        config=config,
        base_dir=str(tmp_path),
    )
    cap = TeamCommCapability(config, "worker", _make_session_metadata())

    # Write > 1MB of data.
    big_value = "x" * (1024 * 1024 + 1)
    result = await cap.write_blackboard(ctx, "big_key", big_value)
    assert "Blackboard write exceeds max size" in result.return_value

    # The oversized file must be cleaned up — reading the key should return
    # "Key not found" (not the oversized data as a list).
    read_result = await cap.read_blackboard(ctx, "big_key")
    assert isinstance(read_result.return_value, str)
    assert "not found" in read_result.return_value.lower()


@pytest.mark.unit
async def test_task_create_happy_path(tmp_path: Any) -> None:
    """Given: team session with initialized state.

    When: task_create is called with subject and description.
    Then: returns "Task created: {task_id}" and task is persisted.
    """
    _init_team(str(tmp_path))
    ctx = _make_run_context(
        metadata=_make_lead_metadata(),
        base_dir=str(tmp_path),
    )
    config = _make_enabled_config(base_dir=str(tmp_path))
    cap = TeamCommCapability(config, "coordinator", _make_lead_metadata())

    result = await cap.task_create(
        ctx, "Translate docs", owner="translator_agent", description="Translate API docs to French"
    )

    assert result.return_value.startswith("Task created: ")


@pytest.mark.unit
async def test_task_create_with_owner(tmp_path: Any) -> None:
    """Given: team session with initialized state.

    When: task_create is called with owner parameter.
    Then: task is created with the specified owner.
    """
    _init_team(str(tmp_path))
    ctx = _make_run_context(
        metadata=_make_lead_metadata(),
        base_dir=str(tmp_path),
    )
    config = _make_enabled_config(base_dir=str(tmp_path))
    cap = TeamCommCapability(config, "coordinator", _make_lead_metadata())

    result = await cap.task_create(
        ctx, "Translate docs", owner="translator_agent", description="Translate API docs to French"
    )

    assert result.return_value.startswith("Task created: ")
    task_id = result.return_value.replace("Task created: ", "")
    task = cap._get_team_state(  # type: ignore[union-attr]
        cap._resolve_agent_context(ctx)
    ).get_task("team_123", task_id)
    assert task is not None
    assert task.get("owner") == "translator_agent"


@pytest.mark.unit
async def test_task_create_no_team_id() -> None:
    """Given: session metadata without team_id.

    When: task_create is called.
    Then: returns "Not in a team session".
    """
    ctx = _make_run_context(metadata={"team_name": "foo", "team_role": "lead"})
    cap = TeamCommCapability(
        _make_enabled_config(),
        "coordinator",
        {"team_name": "foo", "team_role": "lead"},
    )

    result = await cap.task_create(ctx, "Task", owner="translator_agent")

    assert result.return_value == "Not in a team session"


@pytest.mark.unit
async def test_task_list_returns_tasks(tmp_path: Any) -> None:
    """Given: team session with existing tasks.

    When: task_list is called.
    Then: returns JSON array with at least one task.
    """
    _init_team(str(tmp_path))
    ctx = _make_run_context(
        metadata=_make_lead_metadata(),
        base_dir=str(tmp_path),
    )
    config = _make_enabled_config(base_dir=str(tmp_path))
    cap = TeamCommCapability(config, "coordinator", _make_lead_metadata())

    await cap.task_create(ctx, "Task A", owner="translator_agent")
    await cap.task_create(ctx, "Task B", owner="translator_agent")

    result = await cap.task_list(ctx)

    assert "<task_list>" in result.return_value
    assert "Task A" in result.return_value
    assert "Task B" in result.return_value


@pytest.mark.unit
async def test_task_update_changes_status(tmp_path: Any) -> None:
    """Given: team session with an existing task.

    When: task_update is called with status="completed".
    Then: returns updated task JSON with status="completed".
    """
    _init_team(str(tmp_path))
    ctx = _make_run_context(
        metadata=_make_lead_metadata(),
        base_dir=str(tmp_path),
    )
    config = _make_enabled_config(base_dir=str(tmp_path))
    cap = TeamCommCapability(config, "coordinator", _make_lead_metadata())

    create_result = await cap.task_create(ctx, "Task X", owner="translator_agent")
    task_id = create_result.return_value.replace("Task created: ", "")

    update_result = await cap.task_update(ctx, task_id, status="completed")

    assert 'status="completed"' in update_result.return_value
    assert "<task" in update_result.return_value


@pytest.mark.unit
async def test_task_update_no_updates_specified(tmp_path: Any) -> None:
    """Given: team session with a task.

    When: task_update called with empty status and owner.
    Then: returns "No updates specified".
    """
    _init_team(str(tmp_path))
    lead_meta = _make_lead_metadata()
    ctx = _make_run_context(metadata=lead_meta, base_dir=str(tmp_path))
    config = _make_enabled_config(base_dir=str(tmp_path))
    cap = TeamCommCapability(config, "coordinator", lead_meta)

    create_result = await cap.task_create(ctx, "Test task", owner="translator_agent")
    task_id = create_result.return_value.replace("Task created: ", "")
    result = await cap.task_update(ctx, task_id)

    assert result.return_value == "No updates specified"


@pytest.mark.unit
async def test_read_blackboard_returns_value(tmp_path: Any) -> None:
    """Given: team session with a blackboard key written.

    When: read_blackboard is called.
    Then: returns JSON with the value and version.
    """
    _init_team(str(tmp_path))
    ctx = _make_run_context(base_dir=str(tmp_path))
    config = _make_enabled_config(base_dir=str(tmp_path))
    cap = TeamCommCapability(config, "worker", _make_session_metadata())

    await cap.write_blackboard(ctx, "config", "value1")
    result = await cap.read_blackboard(ctx, "config")
    rv = (
        "\n".join(result.return_value)
        if isinstance(result.return_value, list)
        else result.return_value
    )

    assert "<blackboard" in rv
    assert "value1" in rv
    assert 'version="1"' in rv


@pytest.mark.unit
async def test_read_blackboard_pagination_default_limit(tmp_path: Any) -> None:
    """Given: blackboard key with 300 lines.

    When: read_blackboard is called with default params.
    Then: returns first 200 lines with has_more hint.
    """
    _init_team(str(tmp_path))
    ctx = _make_run_context(base_dir=str(tmp_path))
    config = _make_enabled_config(base_dir=str(tmp_path))
    cap = TeamCommCapability(config, "worker", _make_session_metadata())

    lines = [f"line_{i}" for i in range(300)]
    await cap.write_blackboard(ctx, "log", "\n".join(lines))
    result = await cap.read_blackboard(ctx, "log")
    rv = "\n".join(result.return_value)

    assert "line_0" in rv
    assert "line_199" in rv
    assert "line_200" not in rv
    assert "has_more=true" in rv
    assert 'total_lines="300"' in rv
    assert 'offset="0"' in rv
    assert 'limit="200"' in rv


@pytest.mark.unit
async def test_read_blackboard_pagination_offset(tmp_path: Any) -> None:
    """Given: blackboard key with 300 lines.

    When: read_blackboard is called with offset=200, limit=100.
    Then: returns lines 200-299 without has_more hint.
    """
    _init_team(str(tmp_path))
    ctx = _make_run_context(base_dir=str(tmp_path))
    config = _make_enabled_config(base_dir=str(tmp_path))
    cap = TeamCommCapability(config, "worker", _make_session_metadata())

    lines = [f"line_{i}" for i in range(300)]
    await cap.write_blackboard(ctx, "log", "\n".join(lines))
    result = await cap.read_blackboard(ctx, "log", limit=100, offset=200)
    rv = "\n".join(result.return_value)

    assert "line_200" in rv
    assert "line_299" in rv
    assert "line_199" not in rv
    assert "has_more" not in rv
    assert 'offset="200"' in rv


@pytest.mark.unit
async def test_read_blackboard_pagination_context(tmp_path: Any) -> None:
    """Given: blackboard key with 100 lines.

    When: read_blackboard is called with context=50, limit=20.
    Then: returns 20 lines centered around line 50 (lines 40-59).
    """
    _init_team(str(tmp_path))
    ctx = _make_run_context(base_dir=str(tmp_path))
    config = _make_enabled_config(base_dir=str(tmp_path))
    cap = TeamCommCapability(config, "worker", _make_session_metadata())

    lines = [f"line_{i}" for i in range(100)]
    await cap.write_blackboard(ctx, "log", "\n".join(lines))
    result = await cap.read_blackboard(ctx, "log", limit=20, context=50)
    rv = "\n".join(result.return_value)

    assert "line_40" in rv
    assert "line_59" in rv
    assert "line_39" not in rv
    assert "line_60" not in rv
    assert 'offset="40"' in rv


@pytest.mark.unit
async def test_read_blackboard_pagination_context_near_start(tmp_path: Any) -> None:
    """Given: blackboard key with 100 lines.

    When: read_blackboard is called with context=5, limit=20.
    Then: offset is clamped to 0, returns lines 0-19.
    """
    _init_team(str(tmp_path))
    ctx = _make_run_context(base_dir=str(tmp_path))
    config = _make_enabled_config(base_dir=str(tmp_path))
    cap = TeamCommCapability(config, "worker", _make_session_metadata())

    lines = [f"line_{i}" for i in range(100)]
    await cap.write_blackboard(ctx, "log", "\n".join(lines))
    result = await cap.read_blackboard(ctx, "log", limit=20, context=5)
    rv = "\n".join(result.return_value)

    assert "line_0" in rv
    assert "line_19" in rv
    assert 'offset="0"' in rv


@pytest.mark.unit
async def test_read_blackboard_pagination_no_more(tmp_path: Any) -> None:
    """Given: blackboard key with 50 lines.

    When: read_blackboard is called with default limit=200.
    Then: returns all 50 lines, no has_more hint.
    """
    _init_team(str(tmp_path))
    ctx = _make_run_context(base_dir=str(tmp_path))
    config = _make_enabled_config(base_dir=str(tmp_path))
    cap = TeamCommCapability(config, "worker", _make_session_metadata())

    lines = [f"line_{i}" for i in range(50)]
    await cap.write_blackboard(ctx, "log", "\n".join(lines))
    result = await cap.read_blackboard(ctx, "log")
    rv = "\n".join(result.return_value)

    assert "line_0" in rv
    assert "line_49" in rv
    assert "has_more" not in rv
    assert 'total_lines="50"' in rv


@pytest.mark.unit
async def test_read_blackboard_key_not_found(tmp_path: Any) -> None:
    """Given: team session with empty blackboard.

    When: read_blackboard is called with unknown key.
    Then: returns "Key not found".
    """
    _init_team(str(tmp_path))
    ctx = _make_run_context(base_dir=str(tmp_path))
    config = _make_enabled_config(base_dir=str(tmp_path))
    cap = TeamCommCapability(config, "worker", _make_session_metadata())

    result = await cap.read_blackboard(ctx, "nonexistent")

    assert result.return_value == "Key not found"


@pytest.mark.unit
async def test_write_blackboard_returns_version(tmp_path: Any) -> None:
    """Given: team session.

    When: write_blackboard is called with a new key.
    Then: returns "Written, version=1".
    """
    _init_team(str(tmp_path))
    ctx = _make_run_context(base_dir=str(tmp_path))
    config = _make_enabled_config(base_dir=str(tmp_path))
    cap = TeamCommCapability(config, "worker", _make_session_metadata())

    result = await cap.write_blackboard(ctx, "key1", "val1")

    assert result.return_value == "Written, version=1"


@pytest.mark.unit
async def test_write_blackboard_conflict(tmp_path: Any) -> None:
    """Given: team session with existing blackboard key at version 1.

    When: write_blackboard called with expected_version=0 (wrong).
    Then: returns "Conflict: current version is 1".
    """
    _init_team(str(tmp_path))
    ctx = _make_run_context(base_dir=str(tmp_path))
    config = _make_enabled_config(base_dir=str(tmp_path))
    cap = TeamCommCapability(config, "worker", _make_session_metadata())

    await cap.write_blackboard(ctx, "key1", "val1")
    result = await cap.write_blackboard(ctx, "key1", "val2", expected_version=0)

    assert result.return_value == "Conflict: current version is 1"


@pytest.mark.unit
async def test_write_blackboard_append_mode(tmp_path: Any) -> None:
    """Given: team session with existing blackboard key.

    When: write_blackboard called with mode="append".
    Then: new value is concatenated to the existing value.
    """
    _init_team(str(tmp_path))
    ctx = _make_run_context(base_dir=str(tmp_path))
    config = _make_enabled_config(base_dir=str(tmp_path))
    cap = TeamCommCapability(config, "worker", _make_session_metadata())

    await cap.write_blackboard(ctx, "findings", "first finding")
    result = await cap.write_blackboard(ctx, "findings", "second finding", mode="append")

    assert result.return_value == "Written, version=2"

    # Read back and verify append
    read_result = await cap.read_blackboard(ctx, "findings")
    rv = (
        "\n".join(read_result.return_value)
        if isinstance(read_result.return_value, list)
        else read_result.return_value
    )
    assert "first finding" in rv
    assert "second finding" in rv


@pytest.mark.unit
async def test_write_blackboard_append_to_empty_key(tmp_path: Any) -> None:
    """Given: team session, append to a non-existent key.

    When: write_blackboard called with mode="append" on new key.
    Then: value is written as-is (no existing content to append to).
    """
    _init_team(str(tmp_path))
    ctx = _make_run_context(base_dir=str(tmp_path))
    config = _make_enabled_config(base_dir=str(tmp_path))
    cap = TeamCommCapability(config, "worker", _make_session_metadata())

    result = await cap.write_blackboard(ctx, "new_key", "first entry", mode="append")

    assert result.return_value == "Written, version=1"
    read_result = await cap.read_blackboard(ctx, "new_key")
    rv = (
        "\n".join(read_result.return_value)
        if isinstance(read_result.return_value, list)
        else read_result.return_value
    )
    assert "first entry" in rv


@pytest.mark.unit
async def test_list_blackboard_returns_keys(tmp_path: Any) -> None:
    """Given: team session with multiple blackboard keys.

    When: list_blackboard is called.
    Then: returns JSON array of sorted key names.
    """
    _init_team(str(tmp_path))
    ctx = _make_run_context(base_dir=str(tmp_path))
    config = _make_enabled_config(base_dir=str(tmp_path))
    cap = TeamCommCapability(config, "worker", _make_session_metadata())

    await cap.write_blackboard(ctx, "zebra", "z")
    await cap.write_blackboard(ctx, "alpha", "a")

    result = await cap.list_blackboard(ctx)

    assert "<blackboard_keys>" in result.return_value
    assert "alpha" in result.return_value
    assert "zebra" in result.return_value


@pytest.mark.unit
async def test_team_status_returns_formatted_string(tmp_path: Any) -> None:
    """Given: team session with initialized state and members.

    When: team_status is called.
    Then: returns formatted string with team name, status, and members.
    """
    _init_team(str(tmp_path))
    ctx = _make_run_context(base_dir=str(tmp_path))
    config = _make_enabled_config(base_dir=str(tmp_path))
    cap = TeamCommCapability(config, "worker", _make_session_metadata())

    result = await cap.team_status(ctx)

    assert "alpha_team" in result.return_value
    assert "active" in result.return_value
    assert "translator_agent" in result.return_value
    assert "reviewer_agent" in result.return_value


@pytest.mark.unit
async def test_team_status_no_team_id() -> None:
    """Given: session metadata without team_id.

    When: team_status is called.
    Then: returns "Not in a team session".
    """
    ctx = _make_run_context(metadata={"team_name": "foo"})
    cap = TeamCommCapability(_make_enabled_config(), "worker", {"team_name": "foo"})

    result = await cap.team_status(ctx)

    assert result.return_value == "Not in a team session"


@pytest.mark.unit
async def test_disabled_config_registers_no_tools() -> None:
    """Given: disabled config.

    When: TeamCommCapability is constructed.
    Then: no tools are registered and get_tools() returns empty list.
    """
    config = _make_disabled_config()
    cap = TeamCommCapability(config, "worker", _make_session_metadata())

    result = await cap.get_tools()

    assert list(result) == []


# ---- T8 Lead-only tool tests ----


@pytest.mark.unit
async def test_team_create_success(tmp_path: Any) -> None:
    """Given: lead agent with eligible members and mock delegation service.

    When: team_create is called with 2 eligible members.
    Then: returns success message with team_id and creates child sessions.
    """
    config = _make_enabled_config(
        member_eligible=["worker", "reviewer"],
        base_dir=str(tmp_path),
    )
    mock_registry = MagicMock()
    mock_registry.exists = MagicMock(return_value=True)
    mock_pool = MagicMock()
    mock_pool.send_message = AsyncMock(return_value="msg_id")
    mock_delegation = MagicMock()
    mock_delegation.create_child_session = AsyncMock(return_value="child_session_001")
    ctx = _make_run_context(
        metadata=_make_lead_metadata(),
        session_pool=mock_pool,
        config=config,
        base_dir=str(tmp_path),
        agent_registry=mock_registry,
        delegation=mock_delegation,
    )
    cap = TeamCommCapability(config, "coordinator", _make_lead_metadata())

    result = await cap.team_create(
        ctx,
        "my_team",
        [
            {"agent": "worker", "name": "translator_agent"},
            {"agent": "reviewer", "name": "reviewer_agent"},
        ],
    )

    assert "Team 'my_team' created with 2 members" in result.return_value
    assert "team_id=" in result.return_value
    assert mock_pool.create_child_session.await_count == 2
    assert mock_pool.send_message.await_count == 2


@pytest.mark.unit
async def test_team_create_not_lead() -> None:
    """Given: non-lead agent (team_role='translator').

    When: team_create is called.
    Then: returns "Only lead can use team_create".
    """
    ctx = _make_run_context()
    cap = TeamCommCapability(_make_enabled_config(), "worker", _make_session_metadata())

    result = await cap.team_create(ctx, "test", [])

    assert result.return_value == "Only lead can use team_create"


@pytest.mark.unit
async def test_team_create_agent_not_in_registry(tmp_path: Any) -> None:
    """Given: lead agent but member agent not in registry.

    When: team_create is called.
    Then: returns "Agent '{name}' not found in registry".
    """
    config = _make_enabled_config(
        member_eligible=["ghost"],
        base_dir=str(tmp_path),
    )
    mock_registry = MagicMock()
    mock_registry.exists = MagicMock(return_value=False)
    ctx = _make_run_context(
        metadata=_make_lead_metadata(),
        config=config,
        base_dir=str(tmp_path),
        agent_registry=mock_registry,
    )
    cap = TeamCommCapability(config, "coordinator", _make_lead_metadata())

    result = await cap.team_create(
        ctx,
        "test_team",
        [{"agent": "ghost", "name": "ghost_member"}],
    )

    assert "not found in registry" in result.return_value


@pytest.mark.unit
async def test_team_create_agent_not_eligible(tmp_path: Any) -> None:
    """Given: lead agent, agent exists in registry but not in member_eligible.

    When: team_create is called.
    Then: returns "Agent '{name}' is not eligible for team membership".
    """
    config = _make_enabled_config(
        member_eligible=["worker"],
        base_dir=str(tmp_path),
    )
    mock_registry = MagicMock()
    mock_registry.exists = MagicMock(return_value=True)
    ctx = _make_run_context(
        metadata=_make_lead_metadata(),
        config=config,
        base_dir=str(tmp_path),
        agent_registry=mock_registry,
    )
    cap = TeamCommCapability(config, "coordinator", _make_lead_metadata())

    result = await cap.team_create(
        ctx,
        "test_team",
        [{"agent": "non_eligible", "name": "member1"}],
    )

    assert "not eligible for team membership" in result.return_value


@pytest.mark.unit
async def test_team_delete_success(tmp_path: Any) -> None:
    """Given: lead agent with initialized team.

    When: team_delete is called.
    Then: closes all member sessions and returns "Team deleted".
    """
    _init_team(str(tmp_path))
    mock_pool = MagicMock()
    mock_pool.close_session = AsyncMock()
    ctx = _make_run_context(
        metadata=_make_lead_metadata(),
        session_pool=mock_pool,
        base_dir=str(tmp_path),
    )
    config = _make_enabled_config(base_dir=str(tmp_path))
    cap = TeamCommCapability(config, "coordinator", _make_lead_metadata())

    result = await cap.team_delete(ctx)

    assert result.return_value == "Team deleted"
    # Two members registered in _init_team.
    assert mock_pool.close_session.await_count == 2


@pytest.mark.unit
async def test_team_delete_not_lead() -> None:
    """Given: non-lead agent (team_role='translator').

    When: team_delete is called.
    Then: returns "Only lead can use team_delete".
    """
    ctx = _make_run_context()
    cap = TeamCommCapability(_make_enabled_config(), "worker", _make_session_metadata())

    result = await cap.team_delete(ctx)

    assert result.return_value == "Only lead can use team_delete"


@pytest.mark.unit
async def test_shutdown_request_success(tmp_path: Any) -> None:
    """Given: lead agent with initialized team.

    When: shutdown_request is called with a valid member name.

    Then: closes the member's session and removes the member from the
        members dict entirely (hard remove).
    """
    _init_team(str(tmp_path))
    mock_pool = MagicMock()
    mock_pool.close_session = AsyncMock()
    ctx = _make_run_context(
        metadata=_make_lead_metadata(),
        session_pool=mock_pool,
        base_dir=str(tmp_path),
    )
    config = _make_enabled_config(base_dir=str(tmp_path))
    cap = TeamCommCapability(config, "coordinator", _make_lead_metadata())

    result = await cap.shutdown_request(ctx, "translator_agent")

    assert result.return_value == "Shutdown completed for translator_agent"
    mock_pool.close_session.assert_awaited_once_with("sess_translator")
    # Verify member is removed from the members dict entirely (hard remove).
    from wolfharness.capabilities.file_team_state import FileTeamState

    team_state = FileTeamState(str(tmp_path))
    state = team_state._read_json(team_state._state_path("team_123"))
    assert "translator_agent" not in state.get("members", {})


@pytest.mark.unit
async def test_shutdown_request_not_lead() -> None:
    """Given: non-lead agent (team_role='translator').

    When: shutdown_request is called.
    Then: returns "Only lead can use shutdown_request".
    """
    ctx = _make_run_context()
    cap = TeamCommCapability(_make_enabled_config(), "worker", _make_session_metadata())

    result = await cap.shutdown_request(ctx, "some_member")

    assert result.return_value == "Only lead can use shutdown_request"


@pytest.mark.unit
async def test_delete_blackboard_success(tmp_path: Any) -> None:
    """Given: lead agent with a blackboard key written.

    When: delete_blackboard is called.
    Then: key is removed and returns "Blackboard key '{key}' deleted".
    """
    _init_team(str(tmp_path))
    ctx = _make_run_context(
        metadata=_make_lead_metadata(),
        base_dir=str(tmp_path),
    )
    config = _make_enabled_config(base_dir=str(tmp_path))
    cap = TeamCommCapability(config, "coordinator", _make_lead_metadata())

    # Write a key first so we can delete it.
    await cap.write_blackboard(ctx, "test_key", "test_value")
    result = await cap.delete_blackboard(ctx, "test_key")

    assert result.return_value == "Blackboard key 'test_key' deleted"
    # Verify it's gone.
    read_result = await cap.read_blackboard(ctx, "test_key")
    assert read_result.return_value == "Key not found"


@pytest.mark.unit
async def test_delete_blackboard_not_lead() -> None:
    """Given: non-lead agent (team_role='translator').

    When: delete_blackboard is called.
    Then: returns "Only lead can use delete_blackboard".
    """
    ctx = _make_run_context()
    cap = TeamCommCapability(_make_enabled_config(), "worker", _make_session_metadata())

    result = await cap.delete_blackboard(ctx, "some_key")

    assert result.return_value == "Only lead can use delete_blackboard"


@pytest.mark.unit
async def test_broadcast_lead(tmp_path: Any) -> None:
    """Given: lead agent sends broadcast (to='*').

    When: send_message is called with to='*'.
    Then: all members receive the message.
    """
    _init_team(str(tmp_path))
    mock_pool = MagicMock()
    mock_pool.send_message = AsyncMock(return_value="msg_id")
    ctx = _make_run_context(
        metadata=_make_lead_metadata(),
        session_pool=mock_pool,
        base_dir=str(tmp_path),
    )
    config = _make_enabled_config(base_dir=str(tmp_path))
    cap = TeamCommCapability(config, "coordinator", _make_lead_metadata())

    result = await cap.send_message(ctx, "*", "announcement")

    assert "Broadcast sent to 2 members" in result.return_value
    assert mock_pool.send_message.await_count == 2


@pytest.mark.unit
async def test_broadcast_not_lead() -> None:
    """Given: non-lead agent (team_role='translator') sends broadcast.

    When: send_message is called with to='*'.
    Then: returns "Broadcast is lead-only".
    """
    ctx = _make_run_context()
    cap = TeamCommCapability(_make_enabled_config(), "worker", _make_session_metadata())

    result = await cap.send_message(ctx, "*", "announcement")

    assert result.return_value == "Broadcast is lead-only"


@pytest.mark.unit
async def test_message_size_exceeds_limit() -> None:
    """Given: message body exceeding message_max_bytes.

    When: send_message is called.
    Then: returns error about message exceeding max size.
    """
    config = _make_enabled_config()
    config = config.model_copy(update={"message_max_bytes": 10})
    ctx = _make_run_context(config=config)
    cap = TeamCommCapability(config, "worker", _make_session_metadata())

    big_body = "x" * 100
    result = await cap.send_message(ctx, "reviewer_agent", big_body)

    assert "exceeds max size" in result.return_value
    assert "100" in result.return_value
    assert "10" in result.return_value


@pytest.mark.unit
async def test_send_message_queue_mode(tmp_path: Any) -> None:
    """Given: config with notice_delivery_mode='queue'.

    When: send_message is called.
    Then: DeliveryMode.QUEUE is used.
    """
    from wolfharness.lifecycle.types import DeliveryMode

    _init_team(str(tmp_path))
    mock_pool = MagicMock()
    mock_pool.send_message = AsyncMock(return_value="msg_queue")
    ctx = _make_run_context(
        session_pool=mock_pool,
        base_dir=str(tmp_path),
    )
    config = _make_enabled_config(base_dir=str(tmp_path), notice_delivery_mode="queue")
    cap = TeamCommCapability(config, "worker", _make_session_metadata())

    result = await cap.send_message(
        ctx,
        "reviewer_agent",
        "queued msg",
        message_type="escalation",
    )

    assert result.return_value == "Message sent to reviewer_agent"
    call_kwargs = mock_pool.send_message.call_args
    assert call_kwargs.kwargs["mode"] is DeliveryMode.QUEUE


# ---- Config default members tests ----


@pytest.mark.unit
async def test_session_lock_is_per_instance_not_global(tmp_path: Any) -> None:
    """Given: two TeamCommCapability instances (different teams/agents).

    When: both call _create_member_session concurrently.
    Then: both proceed in parallel — the session creation lock is per-instance,
          not a global module-level lock that serializes all teams.
    """
    config = _make_enabled_config(base_dir=str(tmp_path))

    mock_pool = MagicMock()

    ctx1 = _make_run_context(
        metadata=_make_lead_metadata(team_id="team_A"),
        session_pool=mock_pool,
        config=config,
        base_dir=str(tmp_path),
        session_id="lead_session_A",
    )
    ctx2 = _make_run_context(
        metadata=_make_lead_metadata(team_id="team_B"),
        session_pool=mock_pool,
        config=config,
        base_dir=str(tmp_path),
        session_id="lead_session_B",
    )

    # Track concurrent in-flight create_child_session calls.
    # Set up AFTER _make_run_context so it doesn't override our custom function.
    in_flight = 0
    max_in_flight = 0
    counter_lock = asyncio.Lock()

    async def track_create_child_session(**kwargs: Any) -> Any:
        nonlocal in_flight, max_in_flight
        async with counter_lock:
            in_flight += 1
            max_in_flight = max(max_in_flight, in_flight)
        await asyncio.sleep(0.05)  # Small delay to ensure overlap
        async with counter_lock:
            in_flight -= 1
        child_state = MagicMock()
        child_state.session_id = f"child_{kwargs.get('agent_name', 'unknown')}"
        return child_state

    mock_pool.create_child_session = track_create_child_session
    mock_pool.event_bus = None
    mock_pool.pool = MagicMock()
    mock_pool.pool.manifest = MagicMock()
    mock_pool.pool.manifest.agents = {}

    cap1 = TeamCommCapability(config, "coordinator", _make_lead_metadata(team_id="team_A"))
    cap2 = TeamCommCapability(config, "coordinator2", _make_lead_metadata(team_id="team_B"))

    # Run two _create_member_session calls concurrently from different instances.
    await asyncio.gather(
        cap1._create_member_session(
            ctx1.deps,
            "worker",
            "lead_session_A",
            "test",
            team_id="team_A",
            team_member_name="worker_A",
        ),
        cap2._create_member_session(
            ctx2.deps,
            "worker",
            "lead_session_B",
            "test",
            team_id="team_B",
            team_member_name="worker_B",
        ),
    )

    # With per-instance locks, both calls should be in-flight simultaneously.
    assert max_in_flight == 2, (
        f"Expected max_in_flight=2 (parallel), got {max_in_flight}. "
        f"Session creation lock may be global instead of per-instance."
    )


@pytest.mark.unit
async def test_team_create_uses_config_default_members(tmp_path: Any) -> None:
    """Given: lead agent with defaults config, team_create called with empty members.

    When: team_create is called with members=[].
    Then: uses defaults.members from config to create the team.
    """
    from wolfharness_config.team_mode import MemberSpec, TeamDefaultsConfig

    config = _make_enabled_config(
        member_eligible=["translator", "reviewer"],
        base_dir=str(tmp_path),
    ).model_copy(
        update={
            "defaults": TeamDefaultsConfig(
                team_name="default_team",
                members=[
                    MemberSpec(name="translator", agent="translator"),
                    MemberSpec(name="reviewer", agent="reviewer"),
                ],
            )
        }
    )
    mock_registry = MagicMock()
    mock_registry.exists = MagicMock(return_value=True)
    mock_pool = MagicMock()
    mock_pool.send_message = AsyncMock(return_value="msg_id")
    mock_delegation = MagicMock()
    mock_delegation.create_child_session = AsyncMock(return_value="child_session_001")
    ctx = _make_run_context(
        metadata=_make_lead_metadata(),
        session_pool=mock_pool,
        config=config,
        base_dir=str(tmp_path),
        agent_registry=mock_registry,
        delegation=mock_delegation,
    )
    cap = TeamCommCapability(config, "coordinator", _make_lead_metadata())

    result = await cap.team_create(ctx, "my_team", [])

    assert "Team 'my_team' created with 2 members" in result.return_value
    assert mock_pool.create_child_session.await_count == 2
    assert mock_pool.send_message.await_count == 2


@pytest.mark.unit
async def test_resolve_agent_context_from_runtime_context(tmp_path: Any) -> None:
    """Test _resolve_agent_context extracts AgentContextDeps from runtime context.

    In production, PydanticAI wraps our AgentContextDeps inside agents.context.AgentContext.data.
    The tool functions receive ctx.deps = agents.context.AgentContext, and our
    capabilities.agent_context.AgentContextDeps is at ctx.deps.data.
    """
    from wolfharness.agents.context import AgentContext as RuntimeAgentContext
    from wolfharness.capabilities.agent_context import AgentContextDeps
    from wolfharness.orchestrator.session_controller import SessionState

    # Create our capabilities AgentContextDeps (the frozen dataclass).
    session = SessionState(session_id="test-session-123", agent_name="test_agent")
    session.metadata = {"team_id": "test-team"}
    cap_ctx = AgentContextDeps(
        agent_registry=MagicMock(),
        delegation=MagicMock(),
        session=session,
        scope=MagicMock(),
        host=MagicMock(),
    )

    # Create the runtime AgentContext (what PydanticAI actually passes to tools).
    # NodeContext requires a `node` field; AgentContext extends it with `data`.
    runtime_ctx = RuntimeAgentContext(node=MagicMock())
    runtime_ctx.data = cap_ctx

    # Create a mock RunContext with deps=runtime_ctx.
    ctx = MagicMock()
    ctx.deps = runtime_ctx

    # Create TeamCommCapability.
    config = _make_enabled_config(base_dir=str(tmp_path))
    cap = TeamCommCapability(config, "test_agent")

    # This should NOT raise 'AgentContextDeps object has no attribute session'.
    result = cap._resolve_agent_context(ctx)
    assert result is cap_ctx
    assert result.session is not None
    assert result.session.metadata.get("team_id") == "test-team"


@pytest.mark.unit
async def test_team_create_empty_members_no_defaults(tmp_path: Any) -> None:
    """Given: lead agent with defaults=None, team_create called with empty members.

    When: team_create is called with members=[].
    Then: creates team with 0 members (no crash, no defaults fallback).
    """
    config = _make_enabled_config(
        member_eligible=["worker"],
        base_dir=str(tmp_path),
    )
    mock_registry = MagicMock()
    mock_registry.exists = MagicMock(return_value=True)
    mock_pool = MagicMock()
    mock_pool.send_message = AsyncMock(return_value="msg_id")
    mock_pool.create_child_session = AsyncMock()
    mock_pool.sessions = MagicMock()
    mock_pool.sessions.get_or_create_session_agent = AsyncMock()
    mock_pool.event_bus = None
    mock_delegation = MagicMock()
    mock_delegation.create_child_session = AsyncMock(return_value="child_001")
    ctx = _make_run_context(
        metadata=_make_lead_metadata(),
        session_pool=mock_pool,
        config=config,
        base_dir=str(tmp_path),
        agent_registry=mock_registry,
        delegation=mock_delegation,
    )
    cap = TeamCommCapability(config, "coordinator", _make_lead_metadata())

    result = await cap.team_create(ctx, "empty_team", [])

    assert "Team 'empty_team' created with 0 members" in result.return_value
    assert mock_pool.create_child_session.await_count == 0


# ---- prepare_tools role-based filtering tests ----


def _make_tool_def(
    name: str,
    *,
    description: str | None = None,
    to_description: str | None = None,
) -> ToolDefinition:
    """Create a minimal ToolDefinition for testing.

    Args:
        name: Tool name.
        description: Optional tool description.
        to_description: Optional description for the ``to`` parameter
            (only used for ``send_message``).

    Returns:
        A ToolDefinition with a simple parameters_json_schema.
    """
    properties: dict[str, Any] = {}
    if to_description is not None:
        properties["to"] = {
            "type": "string",
            "description": to_description,
        }
    return ToolDefinition(
        name=name,
        description=description,
        parameters_json_schema={
            "type": "object",
            "properties": properties,
            "required": list(properties.keys()),
        },
    )


def _all_tool_names() -> list[str]:
    """Return all 14 team tool names."""
    return [
        "send_message",
        "task_create",
        "task_list",
        "task_update",
        "read_blackboard",
        "write_blackboard",
        "list_blackboard",
        "team_status",
        "team_create",
        "team_delete",
        "delete_blackboard",
        "shutdown_request",
        "team_add_member",
    ]


@pytest.mark.unit
async def test_prepare_tools_lead_returns_all_tools() -> None:
    """Given: lead agent with all 13 tool defs.

    When: prepare_tools() is called.
    Then: all 13 tool defs returned unchanged.
    """
    config = _make_enabled_config()
    cap = TeamCommCapability(config, "coordinator", _make_lead_metadata())
    tool_defs = [_make_tool_def(name) for name in _all_tool_names()]
    ctx = MagicMock()

    result = await cap.prepare_tools(ctx, tool_defs)

    assert len(result) == 13
    result_names = {td.name for td in result}
    assert result_names == set(_all_tool_names())


@pytest.mark.unit
async def test_prepare_tools_member_filters_lead_only_tools() -> None:
    """Given: non-lead member with all 13 tool defs.

    When: prepare_tools() is called.
    Then: lead-only tools (team_create, team_delete,
        delete_blackboard, shutdown_request, team_add_member)
        are filtered out.  8 universal tools remain (task_create
        is available to members but restricted to subtask creation
        by runtime permission checks).
    """
    config = _make_enabled_config()
    cap = TeamCommCapability(config, "worker", _make_session_metadata())
    tool_defs = [_make_tool_def(name) for name in _all_tool_names()]
    ctx = MagicMock()

    result = await cap.prepare_tools(ctx, tool_defs)

    result_names = {td.name for td in result}
    assert "team_create" not in result_names
    assert "team_delete" not in result_names
    assert "delete_blackboard" not in result_names
    assert "shutdown_request" not in result_names
    assert "team_add_member" not in result_names
    assert len(result) == 8
    # Universal tools remain (task_create included for subtask creation).
    for name in (
        "send_message",
        "task_create",
        "task_list",
        "task_update",
        "read_blackboard",
        "write_blackboard",
        "list_blackboard",
        "team_status",
    ):
        assert name in result_names


@pytest.mark.unit
async def test_prepare_tools_member_strips_broadcast_from_send_message() -> None:
    """Given: non-lead member's send_message tool def with broadcast in description.

    When: prepare_tools() is called.
    Then: send_message ``to`` parameter description is updated to omit
        broadcast, and a ``pattern`` constraint is added.
    """
    config = _make_enabled_config()
    cap = TeamCommCapability(config, "worker", _make_session_metadata())
    send_message_def = _make_tool_def(
        "send_message",
        to_description='Recipient member name. "*" broadcasts to all members.',
    )
    ctx = MagicMock()

    result = await cap.prepare_tools(ctx, [send_message_def])

    assert len(result) == 1
    to_prop = result[0].parameters_json_schema["properties"]["to"]
    assert "broadcast" not in to_prop["description"].lower()
    assert to_prop["pattern"] == r"^[^*]+$"


@pytest.mark.unit
async def test_prepare_tools_lead_keeps_broadcast_in_send_message() -> None:
    """Given: lead agent's send_message tool def with broadcast in description.

    When: prepare_tools() is called.
    Then: send_message ``to`` parameter description is unchanged (no
        pattern constraint added).
    """
    config = _make_enabled_config()
    cap = TeamCommCapability(config, "coordinator", _make_lead_metadata())
    original_desc = 'Recipient member name. "*" broadcasts to all members.'
    send_message_def = _make_tool_def("send_message", to_description=original_desc)
    ctx = MagicMock()

    result = await cap.prepare_tools(ctx, [send_message_def])

    assert len(result) == 1
    to_prop = result[0].parameters_json_schema["properties"]["to"]
    assert to_prop["description"] == original_desc
    assert "pattern" not in to_prop


@pytest.mark.unit
async def test_prepare_tools_no_session_metadata_returns_all() -> None:
    """Given: shared instance with no session metadata (compile time).

    When: prepare_tools() is called.
    Then: all tool defs returned unchanged (no role to filter by).
    """
    config = _make_enabled_config()
    cap = TeamCommCapability(config, "worker", session_metadata=None)
    tool_defs = [_make_tool_def(name) for name in _all_tool_names()]
    ctx = MagicMock()

    result = await cap.prepare_tools(ctx, tool_defs)

    assert len(result) == 13


# ---- get_instructions role-specific capabilities tests ----


@pytest.mark.unit
def test_get_instructions_lead_includes_broadcast_capability() -> None:
    """Given: lead agent session metadata.

    When: get_instructions() is called.
    Then: instructions include the lead capabilities section mentioning
        broadcast (to="*") and lead-only tools.
    """
    config = _make_enabled_config()
    cap = TeamCommCapability(config, "coordinator", _make_lead_metadata())

    result = cap.get_instructions()

    assert result is not None
    assert "Your Capabilities (Lead)" in result
    assert 'to="*"' in result
    assert "create and delete teams" in result


@pytest.mark.unit
def test_get_instructions_member_includes_individual_messaging_only() -> None:
    """Given: non-lead member session metadata.

    When: get_instructions() is called.
    Then: instructions include the member capabilities section that
        mentions individual messaging and states broadcast is not
        available.
    """
    config = _make_enabled_config()
    cap = TeamCommCapability(config, "worker", _make_session_metadata())

    result = cap.get_instructions()

    assert result is not None
    assert "Your Capabilities (Member)" in result
    assert "individual" in result.lower()
    assert "not available" in result.lower()
    # Should NOT mention lead-only tools in capabilities section.
    assert "Your Capabilities (Lead)" not in result


# ---- team_add_member and shutdown_request tests ----


@pytest.mark.unit
async def test_team_add_member_success(tmp_path: Any) -> None:
    """Given: lead agent with initialized team and eligible agent.

    When: team_add_member is called with a valid agent name.
    Then: returns success message and creates child session.
    """
    cap, ctx, mock_pool, _mock_delegation, _mock_registry = _make_add_member_setup(tmp_path)

    result = await cap.team_add_member(ctx, "new_member", "editor")

    assert result.return_value == "Member 'new_member' added to team (lifecycle=persistent)"
    mock_pool.create_child_session.assert_awaited_once()
    # 1 initial prompt to new_member + 2 broadcast notifications to
    # existing members (translator_agent, reviewer_agent) since
    # broadcast_on_create defaults to True.
    assert mock_pool.send_message.await_count == 3

    # Verify agent field in team state is the actual agent type, not the member name.
    from wolfharness.capabilities.file_team_state import FileTeamState

    team_state = FileTeamState(str(tmp_path))
    state = team_state._read_json(team_state._state_path("team_123"))
    assert state["members"]["new_member"]["agent"] == "editor"
    assert state["members"]["new_member"]["session_id"] == "child_session_new"

    # Verify team_member_sessions metadata was updated.
    agent_ctx = cap._resolve_agent_context(ctx)
    assert "child_session_new" in agent_ctx.session.metadata.get("team_member_sessions", [])


@pytest.mark.unit
async def test_team_add_member_not_lead() -> None:
    """Given: non-lead agent (team_role='translator').

    When: team_add_member is called.
    Then: returns "Only lead can use team_add_member".
    """
    ctx = _make_run_context()
    cap = TeamCommCapability(_make_enabled_config(), "worker", _make_session_metadata())

    result = await cap.team_add_member(ctx, "new_member", "worker")

    assert result.return_value == "Only lead can use team_add_member"


@pytest.mark.unit
async def test_team_add_member_agent_not_eligible(tmp_path: Any) -> None:
    """Given: lead agent, agent not in member_eligible.

    When: team_add_member is called.
    Then: returns "Agent '{agent}' is not eligible".
    """
    cap, ctx, _mock_pool, _mock_delegation, _mock_registry = _make_add_member_setup(
        tmp_path,
        member_eligible=["worker", "reviewer"],
    )

    result = await cap.team_add_member(ctx, "new_member", "non_eligible")

    assert "not eligible" in result.return_value


@pytest.mark.unit
async def test_team_add_member_duplicate_name(tmp_path: Any) -> None:
    """Given: lead agent, member name already exists in team state.

    When: team_add_member is called with existing name.
    Then: returns "Member '{name}' already exists".
    """
    cap, ctx, _mock_pool, _mock_delegation, _mock_registry = _make_add_member_setup(
        tmp_path,
        member_eligible=["worker", "reviewer"],
    )

    result = await cap.team_add_member(ctx, "translator_agent", "worker")

    assert result.return_value == "Member 'translator_agent' already exists"


@pytest.mark.unit
async def test_team_add_member_with_notify(tmp_path: Any) -> None:
    """Given: lead agent adding a member with notify message.

    When: team_add_member is called with notify="New member joining".
    Then: notify message is sent to existing members (excluding lead and new member).
    """
    cap, ctx, mock_pool, _mock_delegation, _mock_registry = _make_add_member_setup(tmp_path)

    result = await cap.team_add_member(
        ctx,
        "new_member",
        "editor",
        notify="New member joining",
    )

    assert "added to team" in result.return_value
    # send_message called: 1 for initial prompt + 2 for broadcast (translator + reviewer)
    assert mock_pool.send_message.await_count == 3

    # Verify broadcast was sent to existing members but NOT to lead or new member.
    # The notify text is embedded in the auto-generated broadcast message.
    broadcast_calls = [
        c
        for c in mock_pool.send_message.await_args_list
        if len(c.args) > 1 and "New member joining" in c.args[1]
    ]
    broadcast_targets = {c.args[0] for c in broadcast_calls}
    assert "sess_translator" in broadcast_targets  # existing member received
    assert "sess_reviewer" in broadcast_targets  # existing member received
    assert "child_session_new" not in broadcast_targets  # new member excluded


@pytest.mark.unit
async def test_team_add_member_ephemeral(tmp_path: Any) -> None:
    """Given: lead agent adding an ephemeral member.

    When: team_add_member is called with lifecycle="ephemeral".
    Then: returns success with lifecycle=ephemeral and cleanup task scheduled.
    """
    cap, ctx, _mock_pool, _mock_delegation, _mock_registry = _make_add_member_setup(tmp_path)

    result = await cap.team_add_member(
        ctx,
        "temp_member",
        "editor",
        lifecycle="ephemeral",
    )

    assert result.return_value == "Member 'temp_member' added to team (lifecycle=ephemeral)"
    # Verify the member was registered in team state.
    from wolfharness.capabilities.file_team_state import FileTeamState

    team_state = FileTeamState(str(tmp_path))
    sid = team_state.get_member_session_id("team_123", "temp_member")
    assert sid == "child_session_new"


@pytest.mark.unit
async def test_shutdown_request_removes_member(tmp_path: Any) -> None:
    """Given: lead agent with initialized team.

    When: shutdown_request is called with a valid member name.

    Then: closes the member's session, removes the member from team
        state, writes audit to blackboard, and returns success.
    """
    _init_team(str(tmp_path))
    mock_pool = MagicMock()
    mock_pool.close_session = AsyncMock()
    ctx = _make_run_context(
        metadata=_make_lead_metadata(),
        session_pool=mock_pool,
        base_dir=str(tmp_path),
    )
    config = _make_enabled_config(base_dir=str(tmp_path))
    cap = TeamCommCapability(config, "coordinator", _make_lead_metadata())

    result = await cap.shutdown_request(ctx, "translator_agent")

    assert result.return_value == "Shutdown completed for translator_agent"
    mock_pool.close_session.assert_awaited_once_with("sess_translator")
    # Verify member removed from team state (hard remove, not soft shutdown).
    from wolfharness.capabilities.file_team_state import FileTeamState

    team_state = FileTeamState(str(tmp_path))
    sid = team_state.get_member_session_id("team_123", "translator_agent")
    assert sid is None

    # Verify blackboard was written with sanitized key.
    bb_result = team_state.read_blackboard("team_123", "member_update/translator_agent")
    assert bb_result is not None
    assert bb_result["value"]["action"] == "removed"


@pytest.mark.unit
async def test_shutdown_request_not_found(tmp_path: Any) -> None:
    """Given: lead agent with initialized team.

    When: shutdown_request is called with unknown member name.
    Then: returns "Member '{name}' not found".
    """
    _init_team(str(tmp_path))
    mock_pool = MagicMock()
    mock_pool.close_session = AsyncMock()
    ctx = _make_run_context(
        metadata=_make_lead_metadata(),
        session_pool=mock_pool,
        base_dir=str(tmp_path),
    )
    config = _make_enabled_config(base_dir=str(tmp_path))
    cap = TeamCommCapability(config, "coordinator", _make_lead_metadata())

    result = await cap.shutdown_request(ctx, "nonexistent")

    assert result.return_value == "Member 'nonexistent' not found"
    mock_pool.close_session.assert_not_awaited()


@pytest.mark.unit
async def test_shutdown_request_cannot_shutdown_self(tmp_path: Any) -> None:
    """Given: lead agent tries to shut down themselves.

    When: shutdown_request is called with lead's own member name.
    Then: returns "Cannot shut down yourself".
    """
    _init_team(str(tmp_path))
    mock_pool = MagicMock()
    mock_pool.close_session = AsyncMock()
    ctx = _make_run_context(
        metadata=_make_lead_metadata(),
        session_pool=mock_pool,
        base_dir=str(tmp_path),
    )
    config = _make_enabled_config(base_dir=str(tmp_path))
    cap = TeamCommCapability(config, "coordinator", _make_lead_metadata())

    result = await cap.shutdown_request(ctx, "coordinator")

    assert result.return_value == "Cannot shut down yourself"
    mock_pool.close_session.assert_not_awaited()


# ---- Additional team_add_member / shutdown_request coverage ----


@pytest.mark.unit
async def test_team_add_member_non_ascii_name(tmp_path: Any) -> None:
    """Member name with Chinese characters should succeed (blackboard key sanitized)."""
    import re

    cap, ctx, _mock_pool, _mock_delegation, _mock_registry = _make_add_member_setup(tmp_path)
    result = await cap.team_add_member(ctx, "推理员B", "editor")
    assert "added to team" in result.return_value

    # Verify blackboard key was sanitized (non-ASCII chars replaced with _).
    from wolfharness.capabilities.file_team_state import FileTeamState

    safe_name = re.sub(r"[^a-zA-Z0-9_]", "_", "推理员B")
    team_state = FileTeamState(str(tmp_path))
    bb = team_state.read_blackboard("team_123", f"member_update/{safe_name}")
    assert bb is not None
    assert bb["value"]["name"] == "推理员B"


@pytest.mark.unit
async def test_team_add_member_hyphen_in_name(tmp_path: Any) -> None:
    """Member name with hyphens should succeed."""
    import re

    cap, ctx, _mock_pool, _mock_delegation, _mock_registry = _make_add_member_setup(tmp_path)
    result = await cap.team_add_member(ctx, "logician-B", "editor")
    assert "added to team" in result.return_value

    # Verify blackboard key was sanitized (hyphen replaced with _).
    from wolfharness.capabilities.file_team_state import FileTeamState

    safe_name = re.sub(r"[^a-zA-Z0-9_]", "_", "logician-B")
    team_state = FileTeamState(str(tmp_path))
    bb = team_state.read_blackboard("team_123", f"member_update/{safe_name}")
    assert bb is not None


@pytest.mark.unit
async def test_team_add_member_agent_field_correct(tmp_path: Any) -> None:
    """The agent field in team state should be the agent type, not the member name."""
    cap, ctx, _mock_pool, _mock_delegation, _mock_registry = _make_add_member_setup(tmp_path)
    await cap.team_add_member(ctx, "my_member", "editor")

    from wolfharness.capabilities.file_team_state import FileTeamState

    team_state = FileTeamState(str(tmp_path))
    state = team_state._read_json(team_state._state_path("team_123"))
    assert state["members"]["my_member"]["agent"] == "editor"
    assert state["members"]["my_member"]["agent"] != "my_member"  # NOT the member name


@pytest.mark.unit
async def test_team_add_member_roster_includes_work_summary(tmp_path: Any) -> None:
    """Given: team with existing members who have tasks (in_progress and completed).

    When: team_add_member is called.
    Then: the initial prompt to the new member includes work-status for each member.
    """
    from wolfharness.capabilities.file_team_state import FileTeamState

    cap, ctx, mock_pool, _mock_delegation, _mock_registry = _make_add_member_setup(tmp_path)

    # Create tasks for existing members.
    team_state = FileTeamState(str(tmp_path))
    team_state.create_task(
        "team_123",
        {
            "subject": "Translate chapter 1",
            "owner": "translator_agent",
            "status": "in_progress",
            "content": "",
        },
    )
    team_state.create_task(
        "team_123",
        {
            "subject": "Review draft",
            "owner": "reviewer_agent",
            "status": "completed",
            "content": "",
        },
    )

    await cap.team_add_member(ctx, "new_member", "editor")

    # The first send_message call is the initial prompt to the new member.
    first_call = mock_pool.send_message.await_args_list[0]
    prompt_text: str = first_call.args[1]

    assert "Currently working on: Translate chapter 1" in prompt_text
    assert "Just completed: Review draft" in prompt_text


@pytest.mark.unit
async def test_team_add_member_max_members_exceeded(tmp_path: Any) -> None:
    """Should fail when adding a member would exceed max_members.

    The team from _init_team has 2 non-lead members (translator_agent,
    reviewer_agent).  max_members counts non-lead members only, so with
    max_members=2 the team is already full.
    """
    _init_team(str(tmp_path))
    config = _make_enabled_config(
        member_eligible=["worker", "reviewer", "editor"],
        base_dir=str(tmp_path),
    )
    config = config.model_copy(update={"bounds": TeamBounds(max_members=2)})
    mock_registry = MagicMock()
    mock_registry.exists = MagicMock(return_value=True)
    mock_pool = MagicMock()
    mock_pool.send_message = AsyncMock(return_value="msg_id")
    mock_delegation = MagicMock()
    mock_delegation.create_child_session = AsyncMock(return_value="child_session_new")
    ctx = _make_run_context(
        metadata=_make_lead_metadata(),
        session_pool=mock_pool,
        config=config,
        base_dir=str(tmp_path),
        agent_registry=mock_registry,
        delegation=mock_delegation,
    )
    cap = TeamCommCapability(config, "coordinator", _make_lead_metadata())

    result = await cap.team_add_member(ctx, "extra", "editor")

    assert "max_members" in result.return_value


@pytest.mark.unit
async def test_team_add_member_agent_not_in_registry(tmp_path: Any) -> None:
    """Should fail when agent doesn't exist in the registry."""
    cap, ctx, _mock_pool, _mock_delegation, mock_registry = _make_add_member_setup(tmp_path)
    mock_registry.exists = MagicMock(return_value=False)  # Agent not in registry

    result = await cap.team_add_member(ctx, "new_member", "nonexistent_agent")

    assert "not found in registry" in result.return_value


@pytest.mark.unit
async def test_team_add_member_team_member_sessions_updated(tmp_path: Any) -> None:
    """session.metadata['team_member_sessions'] should include the new session_id."""
    cap, ctx, _mock_pool, _mock_delegation, _mock_registry = _make_add_member_setup(tmp_path)
    await cap.team_add_member(ctx, "new_member", "editor")

    agent_ctx = cap._resolve_agent_context(ctx)
    sessions = agent_ctx.session.metadata.get("team_member_sessions", [])
    assert "child_session_new" in sessions


@pytest.mark.unit
async def test_shutdown_request_non_ascii_name(tmp_path: Any) -> None:
    """Shutting down a member with non-ASCII name should succeed (blackboard key sanitized)."""
    import re

    # First add a member with Chinese name.
    cap, ctx, _mock_pool, _mock_delegation, _mock_registry = _make_add_member_setup(tmp_path)
    await cap.team_add_member(ctx, "推理员", "editor")

    # Then shut it down.
    result = await cap.shutdown_request(ctx, "推理员")
    assert "Shutdown completed" in result.return_value

    # Verify blackboard was written with sanitized key.
    from wolfharness.capabilities.file_team_state import FileTeamState

    safe_name = re.sub(r"[^a-zA-Z0-9_]", "_", "推理员")
    team_state = FileTeamState(str(tmp_path))
    bb = team_state.read_blackboard("team_123", f"member_update/{safe_name}")
    assert bb is not None
    assert bb["value"]["action"] == "removed"
    assert bb["value"]["name"] == "推理员"


@pytest.mark.unit
async def test_team_add_member_notify_excludes_lead_and_new_member(tmp_path: Any) -> None:
    """Notify should be sent to existing members but NOT to lead or the new member."""
    cap, ctx, mock_pool, _mock_delegation, _mock_registry = _make_add_member_setup(tmp_path)
    await cap.team_add_member(ctx, "new_member", "editor", notify="Heads up")

    # Get all send_message calls where content contains "Heads up"
    notify_calls = [
        c
        for c in mock_pool.send_message.await_args_list
        if len(c.args) > 1 and "Heads up" in c.args[1]
    ]
    notify_targets = {c.args[0] for c in notify_calls}
    # Lead's session should NOT be in notify targets.
    # New member's session ("child_session_new") should NOT be in notify targets.
    assert "child_session_new" not in notify_targets
    # Existing members (translator_agent, reviewer_agent) SHOULD be in notify targets.
    assert "sess_translator" in notify_targets
    assert "sess_reviewer" in notify_targets


@pytest.mark.unit
async def test_send_message_to_nonexistent_member_no_phantom(tmp_path: Any) -> None:
    """Given: lead in a team.

    When: send_message to a non-existent member.
    Then: returns error AND does NOT create a phantom entry in team state.
    """
    _init_team(str(tmp_path))
    config = _make_enabled_config(base_dir=str(tmp_path))
    mock_pool = MagicMock()
    mock_pool.send_message = AsyncMock(return_value="msg_id")
    ctx = _make_run_context(
        metadata=_make_lead_metadata(),
        session_pool=mock_pool,
        config=config,
        base_dir=str(tmp_path),
    )
    cap = TeamCommCapability(config, "coordinator", _make_lead_metadata())

    result = await cap.send_message(ctx, "ghost_member", "hello")

    assert "not found" in result.return_value
    # Verify NO phantom entry was created in team state.
    from wolfharness.capabilities.file_team_state import FileTeamState

    team_state = FileTeamState(str(tmp_path))
    state = team_state._read_json(team_state._state_path("team_123"))
    assert "ghost_member" not in state.get("members", {})


@pytest.mark.unit
async def test_team_status_shows_added_member_no_team_mode_config(tmp_path: Any) -> None:
    """BUG-001 regression: team_status finds team state even when.

    team_mode_config is None in the per-turn AgentContextDeps.
    team_create stores base_dir in session.metadata['team_base_dir'].
    _get_team_state falls back to that when team_mode_config is None.
    """
    _init_team(str(tmp_path))
    config = _make_enabled_config(
        member_eligible=["worker", "reviewer", "editor"],
        base_dir=str(tmp_path),
    )
    mock_registry = MagicMock()
    mock_registry.exists = MagicMock(return_value=True)
    mock_pool = MagicMock()
    mock_pool.send_message = AsyncMock(return_value="msg_id")
    mock_child_state = MagicMock()
    mock_child_state.session_id = "child_new"
    mock_pool.create_child_session = AsyncMock(return_value=mock_child_state)
    mock_pool.sessions = MagicMock()
    mock_pool.sessions.get_or_create_session_agent = AsyncMock()
    mock_pool.event_bus = None
    mock_delegation = MagicMock()
    mock_delegation.create_child_session = AsyncMock(return_value="child_new")
    metadata = _make_lead_metadata()
    ctx = _make_run_context(
        metadata=metadata,
        session_pool=mock_pool,
        config=config,
        base_dir=str(tmp_path),
        agent_registry=mock_registry,
        delegation=mock_delegation,
    )
    cap = TeamCommCapability(config, "coordinator", metadata)

    # Simulate team_create storing base_dir in metadata.
    agent_ctx = cap._resolve_agent_context(ctx)
    agent_ctx.session.metadata["team_base_dir"] = str(tmp_path)

    await cap.team_add_member(ctx, "historian_backup", "editor")

    # Now simulate team_mode_config=None (the bug scenario).
    agent_ctx.team_mode_config = None  # type: ignore[misc]

    status = await cap.team_status(ctx)
    assert "historian_backup" in status.return_value, (
        f"BUG-001: member missing when team_mode_config=None\n{status.return_value}"
    )


@pytest.mark.unit
async def test_delete_blackboard_nonexistent_key_returns_not_found(
    tmp_path: Any,
) -> None:
    """NOTE-001: delete_blackboard on non-existent key returns not found.

    Consistent with read_blackboard behavior.
    """
    _init_team(str(tmp_path))
    config = _make_enabled_config(base_dir=str(tmp_path))
    ctx = _make_run_context(
        metadata=_make_lead_metadata(),
        config=config,
        base_dir=str(tmp_path),
    )
    cap = TeamCommCapability(config, "coordinator", _make_lead_metadata())

    result = await cap.delete_blackboard(ctx, "nonexistent_key")
    assert "not found" in result.return_value, (
        f"Should return 'not found' for missing key, got: {result.return_value}"
    )


@pytest.mark.unit
async def test_task_update_invalid_task_id_returns_friendly_error(
    tmp_path: Any,
) -> None:
    """BUG-01: task_update with invalid task_id returns friendly error.

    Not raw FileNotFoundError.
    """
    _init_team(str(tmp_path))
    config = _make_enabled_config(base_dir=str(tmp_path))
    ctx = _make_run_context(
        metadata=_make_lead_metadata(),
        config=config,
        base_dir=str(tmp_path),
    )
    cap = TeamCommCapability(config, "coordinator", _make_lead_metadata())

    result = await cap.task_update(ctx, "nonexistent_task_id", status="completed")
    assert "Task not found" in result.return_value
    assert "nonexistent_task_id" in result.return_value
    assert "Errno" not in result.return_value
    assert "No such file" not in result.return_value


@pytest.mark.unit
async def test_team_create_with_prompt(tmp_path: Any) -> None:
    """Given: lead agent with prompt parameter.

    When: team_create is called with a prompt.
    Then: prompt is appended to the initial member message.
    """
    from unittest.mock import AsyncMock, MagicMock

    _init_team(str(tmp_path))
    mock_registry = MagicMock()
    mock_registry.exists = MagicMock(return_value=True)
    mock_pool = MagicMock()
    mock_pool.send_message = AsyncMock(return_value="msg_1")
    mock_pool.sessions = MagicMock()
    mock_pool.sessions.get_or_create_session_agent = AsyncMock()
    mock_child_state = MagicMock()
    mock_child_state.session_id = "child_001"
    mock_pool.create_child_session = AsyncMock(return_value=mock_child_state)
    mock_pool.event_bus = MagicMock()
    mock_pool.event_bus.publish = AsyncMock()
    mock_delegation = MagicMock()
    mock_delegation.create_child_session = AsyncMock(return_value="child_001")
    config = _make_enabled_config(
        member_eligible=["worker", "reviewer"],
        lead_eligible=["coordinator"],
        base_dir=str(tmp_path),
    )
    ctx = _make_run_context(
        session_pool=mock_pool,
        metadata=_make_lead_metadata(),
        config=config,
        base_dir=str(tmp_path),
        agent_registry=mock_registry,
        delegation=mock_delegation,
    )
    cap = TeamCommCapability(config, "coordinator", _make_lead_metadata())

    result = await cap.team_create(
        ctx,
        "test_team",
        [{"agent": "worker", "name": "worker"}],
        prompt="Analyze the system logs for errors",
    )

    assert "team_id=" in result.return_value
    # Check that send_message was called with a prompt containing the task
    call_args = mock_pool.send_message.call_args
    sent_body = call_args.args[1]
    assert "## Task" in sent_body
    assert "Analyze the system logs for errors" in sent_body


@pytest.mark.unit
async def test_list_blackboard_watch_no_changes(tmp_path: Any) -> None:
    """Given: team session with blackboard keys.

    When: list_blackboard called with watch=True, timeout=1.
    Then: returns current keys after timeout (no changes detected).
    """
    _init_team(str(tmp_path))
    ctx = _make_run_context(base_dir=str(tmp_path))
    config = _make_enabled_config(base_dir=str(tmp_path))
    cap = TeamCommCapability(config, "worker", _make_session_metadata())

    await cap.write_blackboard(ctx, "key1", "val1")
    result = await cap.list_blackboard(ctx, watch=True, timeout=1)

    assert "key1" in result.return_value
    assert "watch timeout" in result.return_value


@pytest.mark.unit
async def test_list_blackboard_watch_detects_change(tmp_path: Any) -> None:
    """Given: team session with blackboard keys, watch=True.

    When: another write happens during watch.
    Then: returns updated keys promptly.
    """
    _init_team(str(tmp_path))
    ctx = _make_run_context(base_dir=str(tmp_path))
    config = _make_enabled_config(base_dir=str(tmp_path))
    cap = TeamCommCapability(config, "worker", _make_session_metadata())

    await cap.write_blackboard(ctx, "key1", "val1")

    # Schedule a write after 1.5s (while watch is polling at 1s intervals)
    async def _delayed_write() -> None:
        await asyncio.sleep(1.5)
        await cap.write_blackboard(ctx, "key2", "val2")

    task = asyncio.create_task(_delayed_write())
    result = await cap.list_blackboard(ctx, watch=True, timeout=5)
    await task

    assert "key1" in result.return_value
    assert "key2" in result.return_value
    assert "watch timeout" not in result.return_value


@pytest.mark.unit
async def test_team_status_watch_no_changes(tmp_path: Any) -> None:
    """Given: team session.

    When: team_status called with watch=True, timeout=1.
    Then: returns current status after timeout (no changes detected).
    """
    _init_team(str(tmp_path))
    config = _make_enabled_config(base_dir=str(tmp_path))
    ctx = _make_run_context(
        metadata=_make_lead_metadata(),
        config=config,
        base_dir=str(tmp_path),
    )
    cap = TeamCommCapability(config, "coordinator", _make_lead_metadata())

    # Create team first
    mock_pool = MagicMock()
    mock_pool.send_message = AsyncMock(return_value="msg_1")
    mock_pool.sessions._session_agents = {}
    mock_pool.event_bus = MagicMock()
    mock_pool.event_bus.publish = AsyncMock()
    ctx2 = _make_run_context(
        session_pool=mock_pool,
        metadata=_make_lead_metadata(),
        config=config,
        base_dir=str(tmp_path),
    )
    await cap.team_create(
        ctx2,
        "test_team",
        [{"agent": "worker", "name": "worker"}],
    )

    result = await cap.team_status(ctx, watch=True, timeout=1)
    assert "watch timeout" in result.return_value


@pytest.mark.unit
async def test_list_blackboard_watch_task_ids_timeout(tmp_path: Any) -> None:
    """Given: team with tasks, watch=True with watch_task_ids.

    When: no task changes during timeout.
    Then: returns current keys with timeout message.
    """
    _init_team(str(tmp_path))
    ctx = _make_run_context(
        metadata=_make_lead_metadata(),
        base_dir=str(tmp_path),
    )
    config = _make_enabled_config(base_dir=str(tmp_path))
    cap = TeamCommCapability(config, "coordinator", _make_lead_metadata())

    create_result = await cap.task_create(ctx, "Task A", owner="translator_agent")
    task_id = create_result.return_value.replace("Task created: ", "")
    await cap.write_blackboard(ctx, "key1", "val1")

    result = await cap.list_blackboard(ctx, watch=True, timeout=1, watch_task_ids=[task_id])

    assert "key1" in result.return_value
    assert "watch timeout" in result.return_value


@pytest.mark.unit
async def test_list_blackboard_watch_task_ids_detects_change(tmp_path: Any) -> None:
    """Given: team with tasks, watch=True with watch_task_ids.

    When: a watched task is updated during watch.
    Then: returns promptly without timeout.
    """
    _init_team(str(tmp_path))
    ctx = _make_run_context(
        metadata=_make_lead_metadata(),
        base_dir=str(tmp_path),
    )
    config = _make_enabled_config(base_dir=str(tmp_path))
    cap = TeamCommCapability(config, "coordinator", _make_lead_metadata())

    create_result = await cap.task_create(ctx, "Task A", owner="translator_agent")
    task_id = create_result.return_value.replace("Task created: ", "")
    await cap.write_blackboard(ctx, "key1", "val1")

    async def _delayed_update() -> None:
        await asyncio.sleep(1.5)
        await cap.task_update(ctx, task_id, status="in_progress")

    task = asyncio.create_task(_delayed_update())
    result = await cap.list_blackboard(ctx, watch=True, timeout=5, watch_task_ids=[task_id])
    await task

    assert "key1" in result.return_value
    assert "watch timeout" not in result.return_value


@pytest.mark.unit
async def test_list_blackboard_watch_task_ids_ignores_unrelated_change(
    tmp_path: Any,
) -> None:
    """Given: team with tasks, watch=True with watch_task_ids targeting task A.

    When: task B changes (not in watch list) and timeout expires.
    Then: returns with timeout (watched task unchanged).
    """
    _init_team(str(tmp_path))
    ctx = _make_run_context(
        metadata=_make_lead_metadata(),
        base_dir=str(tmp_path),
    )
    config = _make_enabled_config(base_dir=str(tmp_path))
    cap = TeamCommCapability(config, "coordinator", _make_lead_metadata())

    create_a = await cap.task_create(ctx, "Task A", owner="translator_agent")
    task_a_id = create_a.return_value.replace("Task created: ", "")
    create_b = await cap.task_create(ctx, "Task B", owner="translator_agent")
    task_b_id = create_b.return_value.replace("Task created: ", "")

    async def _delayed_update() -> None:
        await asyncio.sleep(1.5)
        await cap.task_update(ctx, task_b_id, status="in_progress")

    task = asyncio.create_task(_delayed_update())
    result = await cap.list_blackboard(ctx, watch=True, timeout=3, watch_task_ids=[task_a_id])
    await task

    assert "watch timeout" in result.return_value


@pytest.mark.unit
async def test_team_status_watch_task_ids_timeout(tmp_path: Any) -> None:
    """Given: team with tasks, team_status watch=True with watch_task_ids.

    When: no task changes during timeout.
    Then: returns current status with timeout message.
    """
    _init_team(str(tmp_path))
    config = _make_enabled_config(base_dir=str(tmp_path))
    ctx = _make_run_context(
        metadata=_make_lead_metadata(),
        config=config,
        base_dir=str(tmp_path),
    )
    cap = TeamCommCapability(config, "coordinator", _make_lead_metadata())

    create_result = await cap.task_create(ctx, "Task A", owner="translator_agent")
    task_id = create_result.return_value.replace("Task created: ", "")

    result = await cap.team_status(ctx, watch=True, timeout=1, watch_task_ids=[task_id])
    assert "watch timeout" in result.return_value


@pytest.mark.unit
async def test_team_status_watch_task_ids_detects_change(tmp_path: Any) -> None:
    """Given: team with tasks, team_status watch=True with watch_task_ids.

    When: a watched task is updated during watch.
    Then: returns promptly without timeout.
    """
    _init_team(str(tmp_path))
    config = _make_enabled_config(base_dir=str(tmp_path))
    ctx = _make_run_context(
        metadata=_make_lead_metadata(),
        config=config,
        base_dir=str(tmp_path),
    )
    cap = TeamCommCapability(config, "coordinator", _make_lead_metadata())

    create_result = await cap.task_create(ctx, "Task A", owner="translator_agent")
    task_id = create_result.return_value.replace("Task created: ", "")

    async def _delayed_update() -> None:
        await asyncio.sleep(1.5)
        await cap.task_update(ctx, task_id, status="completed")

    task = asyncio.create_task(_delayed_update())
    result = await cap.team_status(ctx, watch=True, timeout=5, watch_task_ids=[task_id])
    await task

    assert "watch timeout" not in result.return_value


@pytest.mark.unit
async def test_list_blackboard_watch_timeout_zero_uses_config_max(tmp_path: Any) -> None:
    """Given: team with blackboard, config max_watch_timeout=1.

    When: list_blackboard called with watch=True, timeout=0 (no limit).
    Then: uses config max (1s), returns with timeout after ~1s.
    """
    _init_team(str(tmp_path))
    ctx = _make_run_context(base_dir=str(tmp_path))
    config = _make_enabled_config(base_dir=str(tmp_path), max_watch_timeout=1)
    cap = TeamCommCapability(config, "worker", _make_session_metadata())

    await cap.write_blackboard(ctx, "key1", "val1")
    result = await cap.list_blackboard(ctx, watch=True, timeout=0)

    assert "key1" in result.return_value
    assert "watch timeout" in result.return_value


@pytest.mark.unit
async def test_list_blackboard_watch_timeout_capped_by_config(tmp_path: Any) -> None:
    """Given: team with blackboard, config max_watch_timeout=2.

    When: list_blackboard called with watch=True, timeout=999 (exceeds config).
    Then: capped at 2s, returns with timeout after ~2s.
    """
    _init_team(str(tmp_path))
    ctx = _make_run_context(base_dir=str(tmp_path))
    config = _make_enabled_config(base_dir=str(tmp_path), max_watch_timeout=2)
    cap = TeamCommCapability(config, "worker", _make_session_metadata())

    await cap.write_blackboard(ctx, "key1", "val1")
    result = await cap.list_blackboard(ctx, watch=True, timeout=999)

    assert "key1" in result.return_value
    assert "watch timeout" in result.return_value


@pytest.mark.unit
async def test_team_status_watch_timeout_zero_uses_config_max(tmp_path: Any) -> None:
    """Given: team, config max_watch_timeout=1.

    When: team_status called with watch=True, timeout=0 (no limit).
    Then: uses config max (1s), returns with timeout after ~1s.
    """
    _init_team(str(tmp_path))
    config = _make_enabled_config(base_dir=str(tmp_path), max_watch_timeout=1)
    ctx = _make_run_context(
        metadata=_make_lead_metadata(),
        config=config,
        base_dir=str(tmp_path),
    )
    cap = TeamCommCapability(config, "coordinator", _make_lead_metadata())

    mock_pool = MagicMock()
    mock_pool.send_message = AsyncMock(return_value="msg_1")
    mock_pool.sessions._session_agents = {}
    mock_pool.event_bus = MagicMock()
    mock_pool.event_bus.publish = AsyncMock()
    ctx2 = _make_run_context(
        session_pool=mock_pool,
        metadata=_make_lead_metadata(),
        config=config,
        base_dir=str(tmp_path),
    )
    await cap.team_create(
        ctx2,
        "test_team",
        [{"agent": "worker", "name": "worker"}],
    )

    result = await cap.team_status(ctx, watch=True, timeout=0)
    assert "watch timeout" in result.return_value


# ---- Nested subtask + permission tests ----


def _make_member_metadata(team_id: str = "team_123") -> dict[str, Any]:
    """Create session metadata for a non-lead member agent."""
    return {
        "team_id": team_id,
        "team_name": "alpha_team",
        "team_role": "member",
        "team_member_name": "translator_agent",
    }


@pytest.mark.unit
async def test_member_can_create_subtask(tmp_path: Any) -> None:
    """Given: non-lead member with a parent task already created by lead.

    When: member calls task_create with parent_id set.
    Then: subtask is created successfully.
    """
    _init_team(str(tmp_path))
    lead_ctx = _make_run_context(
        metadata=_make_lead_metadata(),
        base_dir=str(tmp_path),
    )
    config = _make_enabled_config(base_dir=str(tmp_path))
    lead_cap = TeamCommCapability(config, "coordinator", _make_lead_metadata())

    parent_result = await lead_cap.task_create(lead_ctx, "Parent task", owner="translator_agent")
    parent_id = parent_result.return_value.replace("Task created: ", "")

    member_ctx = _make_run_context(
        metadata=_make_member_metadata(),
        base_dir=str(tmp_path),
    )
    member_cap = TeamCommCapability(config, "worker", _make_member_metadata())

    result = await member_cap.task_create(
        member_ctx, "Subtask", parent_id=parent_id, owner="translator_agent"
    )

    assert result.return_value.startswith("Task created: ")


@pytest.mark.unit
async def test_member_cannot_create_top_level_task(tmp_path: Any) -> None:
    """Given: non-lead member.

    When: member calls task_create without parent_id (top-level).
    Then: returns "Only lead can use task_create".
    """
    _init_team(str(tmp_path))
    ctx = _make_run_context(
        metadata=_make_member_metadata(),
        base_dir=str(tmp_path),
    )
    config = _make_enabled_config(base_dir=str(tmp_path))
    cap = TeamCommCapability(config, "worker", _make_member_metadata())

    result = await cap.task_create(ctx, "Top-level task", owner="translator_agent")

    assert result.return_value == "Only lead can use task_create"


@pytest.mark.unit
async def test_lead_can_create_top_level_and_subtask(tmp_path: Any) -> None:
    """Given: lead agent.

    When: lead creates a top-level task and then a subtask.
    Then: both succeed.
    """
    _init_team(str(tmp_path))
    ctx = _make_run_context(
        metadata=_make_lead_metadata(),
        base_dir=str(tmp_path),
    )
    config = _make_enabled_config(base_dir=str(tmp_path))
    cap = TeamCommCapability(config, "coordinator", _make_lead_metadata())

    top_result = await cap.task_create(ctx, "Top task", owner="translator_agent")
    assert top_result.return_value.startswith("Task created: ")
    parent_id = top_result.return_value.replace("Task created: ", "")

    sub_result = await cap.task_create(
        ctx, "Sub task", parent_id=parent_id, owner="translator_agent"
    )
    assert sub_result.return_value.startswith("Task created: ")


@pytest.mark.unit
async def test_subtask_with_invalid_parent_id(tmp_path: Any) -> None:
    """Given: any team member.

    When: task_create called with non-existent parent_id.
    Then: returns "Parent task not found".
    """
    _init_team(str(tmp_path))
    ctx = _make_run_context(
        metadata=_make_member_metadata(),
        base_dir=str(tmp_path),
    )
    config = _make_enabled_config(base_dir=str(tmp_path))
    cap = TeamCommCapability(config, "worker", _make_member_metadata())

    result = await cap.task_create(
        ctx, "Orphan subtask", parent_id="task_fake123", owner="translator_agent"
    )

    assert "Parent task not found" in result.return_value


@pytest.mark.unit
async def test_task_list_default_shows_only_top_level(tmp_path: Any) -> None:
    """Given: team with top-level and subtasks.

    When: task_list called with default params.
    Then: only top-level tasks are shown.
    """
    _init_team(str(tmp_path))
    ctx = _make_run_context(
        metadata=_make_lead_metadata(),
        base_dir=str(tmp_path),
    )
    config = _make_enabled_config(base_dir=str(tmp_path))
    cap = TeamCommCapability(config, "coordinator", _make_lead_metadata())

    top_result = await cap.task_create(ctx, "Top task", owner="translator_agent")
    top_id = top_result.return_value.replace("Task created: ", "")
    await cap.task_create(ctx, "Sub task", parent_id=top_id, owner="translator_agent")

    result = await cap.task_list(ctx)

    assert "Top task" in result.return_value
    assert "Sub task" not in result.return_value


@pytest.mark.unit
async def test_task_list_include_children_nests_subtasks(tmp_path: Any) -> None:
    """Given: team with top-level and subtasks.

    When: task_list called with include_children=True.
    Then: subtasks are nested inside parent tasks as <subtask> elements.
    """
    _init_team(str(tmp_path))
    ctx = _make_run_context(
        metadata=_make_lead_metadata(),
        base_dir=str(tmp_path),
    )
    config = _make_enabled_config(base_dir=str(tmp_path))
    cap = TeamCommCapability(config, "coordinator", _make_lead_metadata())

    top_result = await cap.task_create(ctx, "Top task", owner="translator_agent")
    top_id = top_result.return_value.replace("Task created: ", "")
    await cap.task_create(ctx, "Sub task", parent_id=top_id, owner="translator_agent")

    result = await cap.task_list(ctx, include_children=True)

    assert "<task_list>" in result.return_value
    assert "Top task" in result.return_value
    assert "<subtask" in result.return_value
    assert "Sub task" in result.return_value
    assert "</subtask>" in result.return_value


@pytest.mark.unit
async def test_task_list_parent_id_filter(tmp_path: Any) -> None:
    """Given: team with top-level and subtasks.

    When: task_list called with parent_id set.
    Then: only direct children of that parent are shown.
    """
    _init_team(str(tmp_path))
    ctx = _make_run_context(
        metadata=_make_lead_metadata(),
        base_dir=str(tmp_path),
    )
    config = _make_enabled_config(base_dir=str(tmp_path))
    cap = TeamCommCapability(config, "coordinator", _make_lead_metadata())

    top_result = await cap.task_create(ctx, "Top task", owner="translator_agent")
    top_id = top_result.return_value.replace("Task created: ", "")
    await cap.task_create(ctx, "Child A", parent_id=top_id, owner="translator_agent")
    await cap.task_create(ctx, "Child B", parent_id=top_id, owner="translator_agent")

    result = await cap.task_list(ctx, parent_id=top_id)

    assert "<task_list>" in result.return_value
    assert "Child A" in result.return_value
    assert "Child B" in result.return_value
    assert "Top task" not in result.return_value


@pytest.mark.unit
async def test_member_can_update_own_task(tmp_path: Any) -> None:
    """Given: member owns a task.

    When: member calls task_update on their own task.
    Then: update succeeds.
    """
    _init_team(str(tmp_path))
    lead_ctx = _make_run_context(
        metadata=_make_lead_metadata(),
        base_dir=str(tmp_path),
    )
    config = _make_enabled_config(base_dir=str(tmp_path))
    lead_cap = TeamCommCapability(config, "coordinator", _make_lead_metadata())

    create_result = await lead_cap.task_create(
        lead_ctx, "Task for member", owner="translator_agent"
    )
    task_id = create_result.return_value.replace("Task created: ", "")

    # Lead assigns the task to the member (using member name from metadata).
    await lead_cap.task_update(lead_ctx, task_id, owner="translator_agent")

    # Member updates their own task.
    member_ctx = _make_run_context(
        metadata=_make_member_metadata(),
        base_dir=str(tmp_path),
    )
    member_cap = TeamCommCapability(config, "worker", _make_member_metadata())

    result = await member_cap.task_update(member_ctx, task_id, status="in_progress")

    assert 'status="in_progress"' in result.return_value


@pytest.mark.unit
async def test_member_cannot_update_other_member_task(tmp_path: Any) -> None:
    """Given: task owned by another member.

    When: a different member calls task_update.
    Then: returns "Permission denied".
    """
    _init_team(str(tmp_path))
    lead_ctx = _make_run_context(
        metadata=_make_lead_metadata(),
        base_dir=str(tmp_path),
    )
    config = _make_enabled_config(base_dir=str(tmp_path))
    lead_cap = TeamCommCapability(config, "coordinator", _make_lead_metadata())

    create_result = await lead_cap.task_create(lead_ctx, "Owned task", owner="translator_agent")
    task_id = create_result.return_value.replace("Task created: ", "")

    # Lead assigns to "reviewer_agent".
    await lead_cap.task_update(lead_ctx, task_id, owner="reviewer_agent")

    # "worker" (translator_agent) tries to update a task owned by reviewer_agent.
    member_ctx = _make_run_context(
        metadata=_make_member_metadata(),
        base_dir=str(tmp_path),
    )
    member_cap = TeamCommCapability(config, "worker", _make_member_metadata())

    result = await member_cap.task_update(member_ctx, task_id, status="completed")

    assert "owned by 'reviewer_agent'" in result.return_value
    assert "send_message(to='reviewer_agent'" in result.return_value


@pytest.mark.unit
async def test_member_can_claim_unclaimed_task(tmp_path: Any) -> None:
    """Given: task with no owner.

    When: member calls task_update to set owner.
    Then: update succeeds (member can claim unclaimed tasks).
    """
    _init_team(str(tmp_path))
    lead_ctx = _make_run_context(
        metadata=_make_lead_metadata(),
        base_dir=str(tmp_path),
    )
    config = _make_enabled_config(base_dir=str(tmp_path))
    lead_cap = TeamCommCapability(config, "coordinator", _make_lead_metadata())

    create_result = await lead_cap.task_create(lead_ctx, "Unclaimed task", owner="")
    task_id = create_result.return_value.replace("Task created: ", "")

    member_ctx = _make_run_context(
        metadata=_make_member_metadata(),
        base_dir=str(tmp_path),
    )
    member_cap = TeamCommCapability(config, "worker", _make_member_metadata())

    result = await member_cap.task_update(member_ctx, task_id, owner="worker")

    assert 'owner="worker"' in result.return_value


@pytest.mark.unit
async def test_lead_can_update_any_task(tmp_path: Any) -> None:
    """Given: task owned by a member.

    When: lead calls task_update.
    Then: update succeeds (lead bypasses ownership check).
    """
    _init_team(str(tmp_path))
    ctx = _make_run_context(
        metadata=_make_lead_metadata(),
        base_dir=str(tmp_path),
    )
    config = _make_enabled_config(base_dir=str(tmp_path))
    cap = TeamCommCapability(config, "coordinator", _make_lead_metadata())

    create_result = await cap.task_create(ctx, "Task owned by someone", owner="translator_agent")
    task_id = create_result.return_value.replace("Task created: ", "")
    await cap.task_update(ctx, task_id, owner="reviewer_agent")

    result = await cap.task_update(ctx, task_id, status="completed")

    assert 'status="completed"' in result.return_value


@pytest.mark.unit
async def test_task_get_returns_task_details(tmp_path: Any) -> None:
    """Given: team session with an existing task.

    When: task_get is called with a valid task_id.
    Then: returns task details as XML.
    """
    _init_team(str(tmp_path))
    ctx = _make_run_context(
        metadata=_make_lead_metadata(),
        base_dir=str(tmp_path),
    )
    config = _make_enabled_config(base_dir=str(tmp_path))
    cap = TeamCommCapability(config, "coordinator", _make_lead_metadata())

    create_result = await cap.task_create(
        ctx, "Get me", owner="translator_agent", description="Detailed description"
    )
    task_id = create_result.return_value.replace("Task created: ", "")

    result = await cap.task_get(ctx, task_id)

    assert "<task" in result.return_value
    assert "Get me" in result.return_value
    assert "Detailed description" in result.return_value


@pytest.mark.unit
async def test_task_get_not_found(tmp_path: Any) -> None:
    """Given: team session.

    When: task_get is called with non-existent task_id.
    Then: returns "Task not found".
    """
    _init_team(str(tmp_path))
    ctx = _make_run_context(
        metadata=_make_lead_metadata(),
        base_dir=str(tmp_path),
    )
    config = _make_enabled_config(base_dir=str(tmp_path))
    cap = TeamCommCapability(config, "coordinator", _make_lead_metadata())

    result = await cap.task_get(ctx, "task_nonexistent")

    assert "Task not found" in result.return_value


@pytest.mark.unit
async def test_task_get_with_include_children(tmp_path: Any) -> None:
    """Given: task with subtasks.

    When: task_get called with include_children=True.
    Then: subtasks are nested inside the task XML.
    """
    _init_team(str(tmp_path))
    ctx = _make_run_context(
        metadata=_make_lead_metadata(),
        base_dir=str(tmp_path),
    )
    config = _make_enabled_config(base_dir=str(tmp_path))
    cap = TeamCommCapability(config, "coordinator", _make_lead_metadata())

    top_result = await cap.task_create(ctx, "Parent task", owner="translator_agent")
    top_id = top_result.return_value.replace("Task created: ", "")
    await cap.task_create(ctx, "Child task", parent_id=top_id, owner="translator_agent")

    result = await cap.task_get(ctx, top_id, include_children=True)

    assert "<task" in result.return_value
    assert "Parent task" in result.return_value
    assert "<subtask" in result.return_value
    assert "Child task" in result.return_value
    assert "</subtask>" in result.return_value


# ------------------------------------------------------------------
# after_run — unfinished task reminder harness
# ------------------------------------------------------------------


@pytest.mark.unit
async def test_after_run_reminder_for_unfinished_tasks(tmp_path: Any) -> None:
    """Given: team member with an in_progress task.

    When: after_run fires after the member's run completes.
    Then: routes a reminder message to the member's own session via
        session_pool.send_message with QUEUE mode.
    """
    from wolfharness.capabilities.file_team_state import FileTeamState

    _init_team(str(tmp_path))
    team_state = FileTeamState(str(tmp_path))
    team_state.create_task(
        "team_123",
        {
            "subject": "Translate chapter 1",
            "owner": "translator_agent",
            "status": "in_progress",
            "content": "",
        },
    )

    mock_pool = MagicMock()
    mock_pool.send_message = AsyncMock(return_value="msg_id")
    metadata = _make_session_metadata()
    ctx = _make_run_context(
        metadata=metadata,
        session_pool=mock_pool,
        base_dir=str(tmp_path),
        session_id="sess_translator",
    )
    ctx.deps.session.closing = False
    ctx.deps.session.is_closing = False

    config = _make_enabled_config(base_dir=str(tmp_path))
    cap = TeamCommCapability(config, "worker", metadata)

    mock_result = MagicMock()
    result = await cap.after_run(ctx, result=mock_result)

    assert result is mock_result
    mock_pool.send_message.assert_awaited_once()
    # Verify QUEUE mode was used.
    call_args = mock_pool.send_message.await_args
    call_kwargs = call_args.kwargs
    from wolfharness.lifecycle.types import DeliveryMode

    assert call_kwargs.get("mode") is DeliveryMode.QUEUE
    # Verify message content (2nd positional arg) mentions the unfinished task.
    msg_content: str = call_args.args[1]
    assert "Translate chapter 1" in msg_content
    assert "in_progress" in msg_content
    # Verify reminder count was incremented.
    assert metadata.get("_task_reminder_count") == 1


@pytest.mark.unit
async def test_after_run_no_reminder_for_lead(tmp_path: Any) -> None:
    """Given: lead agent with an in_progress task.

    When: after_run fires.
    Then: no reminder is sent (lead doesn't get task reminders).
    """
    from wolfharness.capabilities.file_team_state import FileTeamState

    _init_team(str(tmp_path))
    team_state = FileTeamState(str(tmp_path))
    team_state.create_task(
        "team_123",
        {
            "subject": "Coordinate work",
            "owner": "coordinator",
            "status": "in_progress",
            "content": "",
        },
    )

    mock_pool = MagicMock()
    mock_pool.send_message = AsyncMock(return_value="msg_id")
    ctx = _make_run_context(
        metadata=_make_lead_metadata(),
        session_pool=mock_pool,
        base_dir=str(tmp_path),
    )
    ctx.deps.session.closing = False
    ctx.deps.session.is_closing = False

    config = _make_enabled_config(base_dir=str(tmp_path))
    cap = TeamCommCapability(config, "coordinator", _make_lead_metadata())

    mock_result = MagicMock()
    await cap.after_run(ctx, result=mock_result)

    mock_pool.send_message.assert_not_awaited()


@pytest.mark.unit
async def test_after_run_no_reminder_when_session_closing(tmp_path: Any) -> None:
    """Given: team member with unfinished tasks, but session is being closed.

    When: after_run fires.
    Then: no reminder is sent (shutdown path handles notification instead).
    """
    from wolfharness.capabilities.file_team_state import FileTeamState

    _init_team(str(tmp_path))
    team_state = FileTeamState(str(tmp_path))
    team_state.create_task(
        "team_123",
        {
            "subject": "Translate chapter 1",
            "owner": "translator_agent",
            "status": "in_progress",
            "content": "",
        },
    )

    mock_pool = MagicMock()
    mock_pool.send_message = AsyncMock(return_value="msg_id")
    metadata = _make_session_metadata()
    ctx = _make_run_context(
        metadata=metadata,
        session_pool=mock_pool,
        base_dir=str(tmp_path),
        session_id="sess_translator",
    )
    ctx.deps.session.closing = True

    config = _make_enabled_config(base_dir=str(tmp_path))
    cap = TeamCommCapability(config, "worker", metadata)

    mock_result = MagicMock()
    await cap.after_run(ctx, result=mock_result)

    mock_pool.send_message.assert_not_awaited()


@pytest.mark.unit
async def test_after_run_no_duplicate_reminder(tmp_path: Any) -> None:
    """Given: team member already received a reminder (count=1).

    When: after_run fires again.
    Then: no second reminder is sent (max 1 per session).
    """
    from wolfharness.capabilities.file_team_state import FileTeamState

    _init_team(str(tmp_path))
    team_state = FileTeamState(str(tmp_path))
    team_state.create_task(
        "team_123",
        {
            "subject": "Translate chapter 1",
            "owner": "translator_agent",
            "status": "in_progress",
            "content": "",
        },
    )

    mock_pool = MagicMock()
    mock_pool.send_message = AsyncMock(return_value="msg_id")
    metadata = _make_session_metadata()
    metadata["_task_reminder_count"] = 1  # Already reminded once.
    ctx = _make_run_context(
        metadata=metadata,
        session_pool=mock_pool,
        base_dir=str(tmp_path),
        session_id="sess_translator",
    )
    ctx.deps.session.closing = False
    ctx.deps.session.is_closing = False

    config = _make_enabled_config(base_dir=str(tmp_path))
    cap = TeamCommCapability(config, "worker", metadata)

    mock_result = MagicMock()
    await cap.after_run(ctx, result=mock_result)

    mock_pool.send_message.assert_not_awaited()


@pytest.mark.unit
async def test_after_run_no_reminder_when_no_unfinished_tasks(tmp_path: Any) -> None:
    """Given: team member with only completed tasks.

    When: after_run fires.
    Then: no reminder is sent.
    """
    from wolfharness.capabilities.file_team_state import FileTeamState

    _init_team(str(tmp_path))
    team_state = FileTeamState(str(tmp_path))
    team_state.create_task(
        "team_123",
        {
            "subject": "Translate chapter 1",
            "owner": "translator_agent",
            "status": "completed",
            "content": "",
        },
    )

    mock_pool = MagicMock()
    mock_pool.send_message = AsyncMock(return_value="msg_id")
    metadata = _make_session_metadata()
    ctx = _make_run_context(
        metadata=metadata,
        session_pool=mock_pool,
        base_dir=str(tmp_path),
        session_id="sess_translator",
    )
    ctx.deps.session.closing = False
    ctx.deps.session.is_closing = False

    config = _make_enabled_config(base_dir=str(tmp_path))
    cap = TeamCommCapability(config, "worker", metadata)

    mock_result = MagicMock()
    await cap.after_run(ctx, result=mock_result)

    mock_pool.send_message.assert_not_awaited()


# ------------------------------------------------------------------
# shutdown_request — unfinished task warning in return value
# ------------------------------------------------------------------


@pytest.mark.unit
async def test_shutdown_request_warns_about_unfinished_tasks(tmp_path: Any) -> None:
    """Given: lead shuts down a member who has an in_progress task.

    When: shutdown_request is called.
    Then: return value includes a warning about the unfinished task.
    """
    from wolfharness.capabilities.file_team_state import FileTeamState

    _init_team(str(tmp_path))
    team_state = FileTeamState(str(tmp_path))
    team_state.create_task(
        "team_123",
        {
            "subject": "Translate chapter 3",
            "owner": "translator_agent",
            "status": "in_progress",
            "content": "",
        },
    )

    mock_pool = MagicMock()
    mock_pool.close_session = AsyncMock()
    ctx = _make_run_context(
        metadata=_make_lead_metadata(),
        session_pool=mock_pool,
        base_dir=str(tmp_path),
    )
    config = _make_enabled_config(base_dir=str(tmp_path))
    cap = TeamCommCapability(config, "coordinator", _make_lead_metadata())

    result = await cap.shutdown_request(ctx, "translator_agent")

    assert "Shutdown completed for translator_agent" in result.return_value
    assert "Warning" in result.return_value
    assert "Translate chapter 3" in result.return_value
    assert "in_progress" not in result.return_value  # status word not in output
    assert "unfinished" in result.return_value
    mock_pool.close_session.assert_awaited_once_with("sess_translator")


@pytest.mark.unit
async def test_shutdown_request_no_warning_when_tasks_completed(tmp_path: Any) -> None:
    """Given: lead shuts down a member whose tasks are all completed.

    When: shutdown_request is called.
    Then: return value is the standard message with no warning.
    """
    from wolfharness.capabilities.file_team_state import FileTeamState

    _init_team(str(tmp_path))
    team_state = FileTeamState(str(tmp_path))
    team_state.create_task(
        "team_123",
        {
            "subject": "Translate chapter 3",
            "owner": "translator_agent",
            "status": "completed",
            "content": "",
        },
    )

    mock_pool = MagicMock()
    mock_pool.close_session = AsyncMock()
    ctx = _make_run_context(
        metadata=_make_lead_metadata(),
        session_pool=mock_pool,
        base_dir=str(tmp_path),
    )
    config = _make_enabled_config(base_dir=str(tmp_path))
    cap = TeamCommCapability(config, "coordinator", _make_lead_metadata())

    result = await cap.shutdown_request(ctx, "translator_agent")

    assert result.return_value == "Shutdown completed for translator_agent"
    mock_pool.close_session.assert_awaited_once_with("sess_translator")
