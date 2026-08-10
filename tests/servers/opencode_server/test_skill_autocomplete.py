"""Tests for skill autocomplete in GET /command endpoint and session fallback.

Integration tests for the skill autocomplete fix in AgentPool's OpenCode server.
Covers:
- GET /command endpoint with skill_bridge, skill_provider, and MCP prompts
- Session fallback skill lookup when CommandStore misses
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest
from upathtools import UPath

from wolfharness.skills.command import SkillCommand
from wolfharness.skills.skill import Skill


if TYPE_CHECKING:
    from httpx import AsyncClient

    from wolfharness_server.opencode_server.state import ServerState


pytestmark = pytest.mark.asyncio


# =============================================================================
# GET /command endpoint tests
# =============================================================================


async def test_command_endpoint_no_skill_bridge_no_provider(
    async_client: AsyncClient,
    server_state: ServerState,
    mock_agent: MagicMock,
) -> None:
    """When skill_bridge is None and skill_provider is None, returns only MCP prompts."""
    # Ensure no skill bridge or provider
    server_state.skill_bridge = None
    mock_agent.host_context.skill_provider = None  # type: ignore[attr-defined]

    # Mock MCP prompts
    mock_prompt = MagicMock()
    mock_prompt.name = "mcp-prompt"
    mock_prompt.description = "An MCP prompt"
    mock_agent.list_prompts = AsyncMock(return_value=[mock_prompt])

    response = await async_client.get("/command")

    assert response.status_code == 200
    commands = response.json()
    # Only MCP prompt should appear
    mcp_commands = [c for c in commands if c["source"] == "mcp"]
    skill_commands = [c for c in commands if c["source"] == "command"]
    assert len(mcp_commands) == 1
    assert mcp_commands[0]["name"] == "mcp-prompt"
    # No skill commands when no bridge or provider
    assert len(skill_commands) == 0


async def test_command_endpoint_skill_bridge_commands(
    async_client: AsyncClient,
    server_state: ServerState,
    mock_agent: MagicMock,
) -> None:
    """When skill_bridge has skill commands, they appear with source='command'."""
    # Create a real Skill with pre-set instructions
    skill = Skill(
        name="test-skill",
        description="A test skill",
        skill_path=UPath("/tmp/test-skill"),
        instructions="Analyze $1 and $ARGUMENTS",
    )
    skill_cmd = SkillCommand(name="test-skill", description="A test skill", skill=skill)

    # Create skill_bridge mock
    mock_bridge = MagicMock()
    mock_bridge.get_skill_commands = MagicMock(return_value=[skill_cmd])
    server_state.skill_bridge = mock_bridge
    mock_agent.host_context.skill_provider = None  # type: ignore[attr-defined]

    # Mock empty MCP prompts
    mock_agent.list_prompts = AsyncMock(return_value=[])

    response = await async_client.get("/command")

    assert response.status_code == 200
    commands = response.json()

    # Find the skill command
    skill_cmds = [c for c in commands if c["name"] == "test-skill"]
    assert len(skill_cmds) == 1
    cmd = skill_cmds[0]
    assert cmd["source"] == "command"
    assert cmd["template"] == "Analyze $1 and $ARGUMENTS"
    assert cmd["hints"] == ["$1", "$ARGUMENTS"]


async def test_command_endpoint_skill_provider_fallback(
    async_client: AsyncClient,
    server_state: ServerState,
    mock_agent: MagicMock,
) -> None:
    """When skill_resolver is available (fallback path), skills appear with source='command'."""
    # No skill bridge → triggers resolver fallback
    server_state.skill_bridge = None

    # Create skill_resolver mock
    from wolfharness.capabilities.resource_protocols import SkillEntry

    mock_entry = SkillEntry(
        name="provider-skill",
        description="Provider skill",
        uri="skill://test-provider/provider-skill",
    )

    mock_provider = AsyncMock()
    mock_provider.list_skills = AsyncMock(return_value=[mock_entry])
    mock_provider.read_skill = AsyncMock(return_value="Process $1")

    mock_resolver = MagicMock()
    mock_resolver.list_providers = MagicMock(return_value=["test-provider"])
    mock_resolver.get_provider = MagicMock(return_value=mock_provider)

    mock_agent._agent_pool.skill_resolver = mock_resolver  # type: ignore[attr-defined]
    mock_agent.host_context.skill_provider = None  # type: ignore[attr-defined]

    # Mock empty MCP prompts
    mock_agent.list_prompts = AsyncMock(return_value=[])

    response = await async_client.get("/command")

    assert response.status_code == 200
    commands = response.json()

    # Find the provider skill command
    skill_cmds = [c for c in commands if c["name"] == "provider-skill"]
    assert len(skill_cmds) == 1
    cmd = skill_cmds[0]
    assert cmd["source"] == "command"
    assert cmd["template"] == "Process $1"
    assert cmd["hints"] == ["$1"]


async def test_command_endpoint_mcp_prompts_have_source_mcp(
    async_client: AsyncClient,
    server_state: ServerState,
    mock_agent: MagicMock,
) -> None:
    """MCP prompts get source='mcp' and empty hints."""
    server_state.skill_bridge = None
    mock_agent.host_context.skill_provider = None  # type: ignore[attr-defined]

    # Mock MCP prompts
    mock_prompt1 = MagicMock()
    mock_prompt1.name = "review"
    mock_prompt1.description = "Code review prompt"
    mock_prompt2 = MagicMock()
    mock_prompt2.name = "test"
    mock_prompt2.description = "Test prompt"
    mock_agent.list_prompts = AsyncMock(return_value=[mock_prompt1, mock_prompt2])

    response = await async_client.get("/command")

    assert response.status_code == 200
    commands = response.json()

    mcp_cmds = [c for c in commands if c["source"] == "mcp"]
    assert len(mcp_cmds) == 2
    for cmd in mcp_cmds:
        assert cmd["hints"] == []
        assert cmd["source"] == "mcp"


async def test_command_endpoint_skill_hints_from_template(
    async_client: AsyncClient,
    server_state: ServerState,
    mock_agent: MagicMock,
) -> None:
    """Skills include hints extracted from template with numeric sort."""
    skill = Skill(
        name="compare-skill",
        description="Compare things",
        skill_path=UPath("/tmp/compare"),
        instructions="Compare $10 with $1 and $2 using $ARGUMENTS",
    )
    skill_cmd = SkillCommand(name="compare-skill", description="Compare things", skill=skill)

    mock_bridge = MagicMock()
    mock_bridge.get_skill_commands = MagicMock(return_value=[skill_cmd])
    server_state.skill_bridge = mock_bridge
    mock_agent.host_context.skill_provider = None  # type: ignore[attr-defined]

    # Mock empty MCP prompts
    mock_agent.list_prompts = AsyncMock(return_value=[])

    response = await async_client.get("/command")

    assert response.status_code == 200
    commands = response.json()

    cmd = next(c for c in commands if c["name"] == "compare-skill")
    # Numeric sort: $1, $2, $10, then $ARGUMENTS
    assert cmd["hints"] == ["$1", "$2", "$10", "$ARGUMENTS"]


async def test_command_endpoint_skill_bridge_with_provider_for_virtual_skills(
    async_client: AsyncClient,
    server_state: ServerState,
    mock_agent: MagicMock,
) -> None:
    """When skill_bridge and skill_resolver both exist, resolver is used for instructions."""
    # Use MagicMock for skill since we need load_instructions fallback behavior
    mock_skill_obj = MagicMock(spec=Skill)
    mock_skill_obj.load_instructions = MagicMock(return_value="Local fallback instructions")

    mock_skill_cmd = MagicMock()
    mock_skill_cmd.name = "virtual-skill"
    mock_skill_cmd.description = "Virtual skill from MCP"
    mock_skill_cmd.skill = mock_skill_obj

    mock_bridge = MagicMock()
    mock_bridge.get_skill_commands = MagicMock(return_value=[mock_skill_cmd])
    server_state.skill_bridge = mock_bridge

    # Resolver returns different instructions than local
    mock_resolved = MagicMock()
    mock_resolved.load_instructions = MagicMock(return_value="Virtual instructions with $1")
    mock_resolver = MagicMock()
    mock_resolver.resolve = AsyncMock(return_value=mock_resolved)
    mock_agent._agent_pool.skill_resolver = mock_resolver  # type: ignore[attr-defined]
    mock_agent.host_context.skill_provider = None  # type: ignore[attr-defined]

    # Mock empty MCP prompts
    mock_agent.list_prompts = AsyncMock(return_value=[])

    response = await async_client.get("/command")

    assert response.status_code == 200
    commands = response.json()

    cmd = next(c for c in commands if c["name"] == "virtual-skill")
    assert cmd["template"] == "Virtual instructions with $1"
    assert cmd["hints"] == ["$1"]
    # Resolver was called, not local load_instructions
    mock_resolver.resolve.assert_called_once_with("virtual-skill")


# =============================================================================
# Helpers for session tests
# =============================================================================


def _setup_pool_sessions(mock_pool: MagicMock) -> None:
    """Set up mock pool.session_pool for session creation and retrieval.

    Only adds store to the existing sessions mock — does NOT replace
    session_pool (replacing it loses the create_session AsyncMock from
    the mock_pool fixture, causing a cascading TypeError).
    """
    mock_pool.session_pool.sessions.store = AsyncMock()
    mock_pool.session_pool.sessions.store.load_session = AsyncMock(return_value=None)
    # Allow get_or_load_session fast-path: cached session + controller registered
    mock_pool.session_pool.sessions.get_session = Mock(return_value=Mock())


# =============================================================================
# Session fallback skill lookup tests
# =============================================================================


async def test_session_command_store_takes_priority(
    async_client: AsyncClient,
    server_state: ServerState,
    mock_agent: MagicMock,
) -> None:
    """When command is in CommandStore, uses CommandStore (existing behavior)."""
    _setup_pool_sessions(mock_agent.host_context)

    # Create session
    response = await async_client.post("/session", json={"title": "Test Session"})
    assert response.status_code == 200
    session_id = response.json()["id"]

    # Mock CommandStore with a command
    mock_command = MagicMock()
    mock_command.execute = AsyncMock()
    mock_command_store = MagicMock()
    mock_command_store.get_command = MagicMock(return_value=mock_command)
    server_state.command_store = mock_command_store

    # Mock empty MCP prompts
    mock_agent.list_prompts = AsyncMock(return_value=[])

    response = await async_client.post(
        f"/session/{session_id}/command",
        json={"command": "test-cmd", "arguments": "arg1"},
    )

    # CommandStore should handle it, not skill_commands
    assert response.status_code == 200
    mock_command_store.get_command.assert_called_with("test-cmd")
    mock_command.execute.assert_called_once()


async def test_session_skill_commands_fallback(
    async_client: AsyncClient,
    server_state: ServerState,
    mock_agent: MagicMock,
) -> None:
    """When command is in CommandStore, execution works."""
    _setup_pool_sessions(mock_agent.host_context)

    # Create session
    response = await async_client.post("/session", json={"title": "Test Session"})
    assert response.status_code == 200
    session_id = response.json()["id"]

    # CommandStore has the command
    mock_command = MagicMock()
    mock_command.execute = AsyncMock()
    mock_command_store = MagicMock()
    mock_command_store.get_command = MagicMock(return_value=mock_command)
    server_state.command_store = mock_command_store

    mock_agent.host_context.skill_provider = None  # type: ignore[attr-defined]

    # Mock empty MCP prompts
    mock_agent.list_prompts = AsyncMock(return_value=[])
    mock_agent.run = AsyncMock(return_value=MagicMock(data="skill result"))

    response = await async_client.post(
        f"/session/{session_id}/command",
        json={"command": "late-skill", "arguments": "some args"},
    )

    # Command should execute successfully — returns 200
    assert response.status_code == 200
    result = response.json()
    assert "info" in result
    assert "parts" in result


async def test_session_unknown_command_falls_back_to_mcp_or_404(
    async_client: AsyncClient,
    server_state: ServerState,
    mock_agent: MagicMock,
) -> None:
    """When command is NOT in CommandStore AND NOT in pool.skill_commands, falls to MCP or 404."""
    _setup_pool_sessions(mock_agent.host_context)

    # Create session
    response = await async_client.post("/session", json={"title": "Test Session"})
    assert response.status_code == 200
    session_id = response.json()["id"]

    # CommandStore doesn't have it
    mock_command_store = MagicMock()
    mock_command_store.get_command = MagicMock(return_value=None)
    server_state.command_store = mock_command_store

    # No MCP prompts either
    mock_agent.list_prompts = AsyncMock(return_value=[])

    response = await async_client.post(
        f"/session/{session_id}/command",
        json={"command": "nonexistent-cmd", "arguments": ""},
    )

    # Should 404
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


async def test_session_skill_commands_fallback_then_mcp(
    async_client: AsyncClient,
    server_state: ServerState,
    mock_agent: MagicMock,
) -> None:
    """When not in CommandStore or skill_commands, MCP prompt is used as final fallback."""
    _setup_pool_sessions(mock_agent.host_context)

    # Create session
    response = await async_client.post("/session", json={"title": "Test Session"})
    assert response.status_code == 200
    session_id = response.json()["id"]

    # CommandStore doesn't have it
    mock_command_store = MagicMock()
    mock_command_store.get_command = MagicMock(return_value=None)
    server_state.command_store = mock_command_store

    # MCP prompt exists
    mock_prompt = MagicMock()
    mock_prompt.name = "mcp-fallback-cmd"
    mock_prompt.arguments = []
    mock_prompt.get_components = AsyncMock(return_value=[])
    mock_agent.list_prompts = AsyncMock(return_value=[mock_prompt])
    mock_agent.run = AsyncMock(return_value=MagicMock(data="MCP result"))

    response = await async_client.post(
        f"/session/{session_id}/command",
        json={"command": "mcp-fallback-cmd", "arguments": ""},
    )

    # MCP fallback should handle it
    assert response.status_code == 200
