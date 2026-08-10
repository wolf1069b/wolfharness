"""L2 integration tests for role-aware tool filtering via ``prepare_tools``.

These tests verify that ``TeamCommCapability.prepare_tools()`` correctly
filters and modifies tool definitions based on the agent's team role
(``lead`` vs ``member``) as determined by ``session_metadata``.

The tests use the real ``team_mode_pool`` fixture (real ``AgentPool`` with
``TestModel`` agents and ``team_mode`` enabled) and the
``build_agent_context`` / ``make_mock_run_context`` helpers from
``tests.team_mode.conftest`` to construct realistic contexts.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic_ai.tools import ToolDefinition
import pytest

from tests.team_mode.conftest import build_agent_context, make_mock_run_context
from wolfharness.capabilities.team_comm_capability import TeamCommCapability


if TYPE_CHECKING:
    from wolfharness import AgentPool
    from wolfharness_config.team_mode import TeamModeConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_LEAD_ONLY_TOOLS: frozenset[str] = frozenset(
    {
        "team_create",
        "team_delete",
        "delete_blackboard",
        "shutdown_request",
        "team_add_member",
    },
)

_ALL_TOOL_NAMES: list[str] = [
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

_BROADCAST_DESC = 'Recipient member name. "*" broadcasts to all members (lead-only).'


def _make_tool_def(
    name: str,
    *,
    to_description: str | None = None,
) -> ToolDefinition:
    """Create a minimal ``ToolDefinition`` for testing.

    Args:
        name: Tool name.
        to_description: Optional description for the ``to`` parameter
            (only used for ``send_message``).

    Returns:
        A ``ToolDefinition`` with a simple ``parameters_json_schema``.
    """
    properties: dict[str, Any] = {}
    if to_description is not None:
        properties["to"] = {
            "type": "string",
            "description": to_description,
        }
    return ToolDefinition(
        name=name,
        description=f"Tool: {name}",
        parameters_json_schema={
            "type": "object",
            "properties": properties,
            "required": list(properties.keys()),
        },
    )


def _make_all_tool_defs() -> list[ToolDefinition]:
    """Create all 14 team tool definitions.

    The ``send_message`` tool def includes the broadcast mention in the
    ``to`` parameter description, matching production behavior.
    """
    return [
        _make_tool_def(
            "send_message",
            to_description=_BROADCAST_DESC,
        ),
        *(_make_tool_def(name) for name in _ALL_TOOL_NAMES if name != "send_message"),
    ]


async def _create_lead_session(
    pool: AgentPool[Any],
    session_id: str,
    team_role: str,
) -> str:
    """Create a session in the real pool with the given team role.

    Args:
        pool: Real AgentPool instance.
        session_id: Unique session identifier.
        team_role: Either ``"lead"`` or ``"member"``.

    Returns:
        The session ID.
    """
    session_pool = pool.session_pool
    assert session_pool is not None
    await session_pool.create_session(
        session_id,
        agent_name="coordinator" if team_role == "lead" else "worker",
        team_role=team_role,
        team_member_name=f"{team_role}_agent",
    )
    return session_id


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
async def test_lead_sees_all_13_tools(team_mode_pool: AgentPool[Any]) -> None:
    """Given: lead agent with all 13 team tool definitions.

    When: ``prepare_tools()`` is called with lead session metadata.

    Then: all 13 tool definitions are returned unchanged, including
        the 6 lead-only tools (team_create, team_delete,
        delete_blackboard, shutdown_request, team_add_member).
    """
    manifest = team_mode_pool.manifest
    team_mode_config: TeamModeConfig | None = manifest.team_mode
    assert team_mode_config is not None

    session_id = await _create_lead_session(team_mode_pool, "lead-role-test-001", "lead")
    agent_ctx = build_agent_context(team_mode_pool, session_id, team_mode_config)

    cap = TeamCommCapability(
        team_mode_config,
        "coordinator",
        session_metadata={"team_role": "lead", "team_member_name": "coordinator"},
    )
    ctx = make_mock_run_context(agent_ctx)
    tool_defs = _make_all_tool_defs()

    result = await cap.prepare_tools(ctx, tool_defs)

    result_names = {td.name for td in result}
    assert len(result) == 13
    assert result_names == set(_ALL_TOOL_NAMES)
    for lead_tool in _LEAD_ONLY_TOOLS:
        assert lead_tool in result_names


@pytest.mark.integration
async def test_member_sees_only_9_universal_tools(
    team_mode_pool: AgentPool[Any],
) -> None:
    """Given: non-lead member with all 13 team tool definitions.

    When: ``prepare_tools()`` is called with member session metadata.

    Then: only 8 universal tool definitions are returned.  The 5
        lead-only tools (team_create, team_delete,
        delete_blackboard, shutdown_request, team_add_member)
        are filtered out entirely.  task_create is available to
        members but restricted to subtask creation by runtime
        permission checks.
    """
    manifest = team_mode_pool.manifest
    team_mode_config: TeamModeConfig | None = manifest.team_mode
    assert team_mode_config is not None

    session_id = await _create_lead_session(team_mode_pool, "member-role-test-001", "member")
    agent_ctx = build_agent_context(team_mode_pool, session_id, team_mode_config)

    cap = TeamCommCapability(
        team_mode_config,
        "worker",
        session_metadata={
            "team_role": "member",
            "team_member_name": "worker_agent",
        },
    )
    ctx = make_mock_run_context(agent_ctx)
    tool_defs = _make_all_tool_defs()

    result = await cap.prepare_tools(ctx, tool_defs)

    result_names = {td.name for td in result}
    assert len(result) == 8
    for lead_tool in _LEAD_ONLY_TOOLS:
        assert lead_tool not in result_names
    universal_tools = {
        "send_message",
        "task_create",
        "task_list",
        "task_update",
        "read_blackboard",
        "write_blackboard",
        "list_blackboard",
        "team_status",
    }
    assert result_names == universal_tools


@pytest.mark.integration
async def test_member_send_message_has_no_broadcast(
    team_mode_pool: AgentPool[Any],
) -> None:
    """Given: non-lead member's ``send_message`` tool def with broadcast in description.

    When: ``prepare_tools()`` is called with member session metadata.

    Then: the ``send_message`` tool def's ``to`` parameter description
        is updated to omit the broadcast mention, and a ``pattern``
        constraint ``r"^[^*]+$"`` is added to reject ``"*"`` at the
        schema level.
    """
    manifest = team_mode_pool.manifest
    team_mode_config: TeamModeConfig | None = manifest.team_mode
    assert team_mode_config is not None

    session_id = await _create_lead_session(team_mode_pool, "member-broadcast-test-001", "member")
    agent_ctx = build_agent_context(team_mode_pool, session_id, team_mode_config)

    cap = TeamCommCapability(
        team_mode_config,
        "worker",
        session_metadata={
            "team_role": "member",
            "team_member_name": "worker_agent",
        },
    )
    ctx = make_mock_run_context(agent_ctx)
    tool_defs = _make_all_tool_defs()

    result = await cap.prepare_tools(ctx, tool_defs)

    send_message_defs = [td for td in result if td.name == "send_message"]
    assert len(send_message_defs) == 1
    to_prop = send_message_defs[0].parameters_json_schema["properties"]["to"]
    assert "broadcast" not in to_prop["description"].lower()
    assert to_prop["pattern"] == r"^[^*]+$"


@pytest.mark.integration
async def test_lead_send_message_allows_broadcast(
    team_mode_pool: AgentPool[Any],
) -> None:
    """Given: lead agent's ``send_message`` tool def with broadcast in description.

    When: ``prepare_tools()`` is called with lead session metadata.

    Then: the ``send_message`` tool def's ``to`` parameter description
        is unchanged (broadcast mention preserved), and no ``pattern``
        constraint is added — the lead can use ``to="*"`` for broadcast.
    """
    manifest = team_mode_pool.manifest
    team_mode_config: TeamModeConfig | None = manifest.team_mode
    assert team_mode_config is not None

    session_id = await _create_lead_session(team_mode_pool, "lead-broadcast-test-001", "lead")
    agent_ctx = build_agent_context(team_mode_pool, session_id, team_mode_config)

    cap = TeamCommCapability(
        team_mode_config,
        "coordinator",
        session_metadata={"team_role": "lead", "team_member_name": "coordinator"},
    )
    ctx = make_mock_run_context(agent_ctx)
    tool_defs = _make_all_tool_defs()

    result = await cap.prepare_tools(ctx, tool_defs)

    send_message_defs = [td for td in result if td.name == "send_message"]
    assert len(send_message_defs) == 1
    to_prop = send_message_defs[0].parameters_json_schema["properties"]["to"]
    assert to_prop["description"] == _BROADCAST_DESC
    assert "pattern" not in to_prop


@pytest.mark.integration
async def test_no_metadata_returns_all_tools(
    team_mode_pool: AgentPool[Any],
) -> None:
    """Given: shared compile-time instance with no session metadata.

    When: ``prepare_tools()`` is called.

    Then: all 14 tool definitions are returned unchanged (no role to
        filter by — the compile-time default returns everything).
    """
    manifest = team_mode_pool.manifest
    team_mode_config: TeamModeConfig | None = manifest.team_mode
    assert team_mode_config is not None

    cap = TeamCommCapability(
        team_mode_config,
        "coordinator",
        session_metadata=None,
    )
    ctx = make_mock_run_context(
        build_agent_context(
            team_mode_pool,
            await _create_lead_session(team_mode_pool, "no-meta-test-001", "lead"),
            team_mode_config,
        ),
    )
    tool_defs = _make_all_tool_defs()

    result = await cap.prepare_tools(ctx, tool_defs)

    assert len(result) == 13
    result_names = {td.name for td in result}
    assert result_names == set(_ALL_TOOL_NAMES)
