"""Unit tests for team-member skill injection (skills parameter).

Covers spec requirements R1-R5 for ``team_create`` / ``team_add_member``:
- R1/R2: member ``skills`` accepted and injected as instruction text
- R3: visibility checked against the member agent's node scope
- R4: load failures degrade to error text, never abort member creation
- R5: pure instruction injection — ``include_assembly=False``, no tool/MCP assembly
- Dedup: duplicate skill names injected exactly once
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tests.team_mode.conftest import (
    init_team,
    make_enabled_config,
    make_lead_metadata,
    make_mock_pool,
    make_run_context,
)
from wolfharness.capabilities.team_comm_capability import TeamCommCapability


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_cap() -> TeamCommCapability[Any]:
    """Create a TeamCommCapability for a lead agent."""
    return TeamCommCapability(make_enabled_config(), "coordinator", make_lead_metadata())


def _make_ctx(mock_pool: MagicMock, tmp_path: Any) -> MagicMock:
    """Create a mock run context wired to the mock pool and permissive registry."""
    mock_registry = MagicMock()
    mock_registry.exists = MagicMock(return_value=True)
    return make_run_context(
        metadata=make_lead_metadata(),
        session_pool=mock_pool,
        config=make_enabled_config(
            member_eligible=["worker", "reviewer", "editor"],
            base_dir=str(tmp_path),
        ),
        base_dir=str(tmp_path),
        agent_registry=mock_registry,
    )


def _make_mock_pool_with_created_sessions(tmp_path: Any) -> MagicMock:
    """Return a mock pool whose create_child_session yields unique ids."""
    init_team(str(tmp_path))
    mock_pool = make_mock_pool()
    mock_pool.create_child_session = AsyncMock(
        side_effect=lambda **kwargs: MagicMock(
            session_id=f"child_{kwargs.get('team_member_name', 'x')}"
        )
    )
    return mock_pool


# ---------------------------------------------------------------------------
# R1: team_create member skills
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_team_create_with_member_skills_injects_instruction_blocks(
    tmp_path: Any,
) -> None:
    """R1: member dict with ``skills`` injects <skill-instruction> blocks."""
    mock_pool = _make_mock_pool_with_created_sessions(tmp_path)
    ctx = _make_ctx(mock_pool, tmp_path)
    cap = _make_cap()

    with patch(
        "wolfharness_toolsets.builtin.skills.load_skill_for_node",
        new=AsyncMock(return_value="Skill instructions text"),
    ) as mock_load:
        result = await cap.team_create(
            ctx,
            "my_team",
            [{"agent": "worker", "name": "worker_a", "skills": ["lodestone"]}],
        )

    assert "Team 'my_team' created" in result.return_value
    call_kwargs = mock_pool.create_child_session.await_args.kwargs
    instructions: str = call_kwargs["team_member_instructions"]
    assert '<skill-instruction name="lodestone">' in instructions
    assert "Skill instructions text" in instructions
    # node_name is the member agent, not the lead.
    assert mock_load.await_args.kwargs["node_name"] == "worker"
    assert mock_load.await_args.kwargs["include_assembly"] is False


@pytest.mark.unit
async def test_team_create_without_skills_no_injection(tmp_path: Any) -> None:
    """R1: member dict without ``skills`` behaves exactly as before."""
    mock_pool = _make_mock_pool_with_created_sessions(tmp_path)
    ctx = _make_ctx(mock_pool, tmp_path)
    cap = _make_cap()

    with patch(
        "wolfharness_toolsets.builtin.skills.load_skill_for_node", new=AsyncMock()
    ) as mock_load:
        result = await cap.team_create(
            ctx,
            "my_team",
            [{"agent": "worker", "name": "worker_a", "instructions": "Do X"}],
        )

    assert "Team 'my_team' created" in result.return_value
    call_kwargs = mock_pool.create_child_session.await_args.kwargs
    assert call_kwargs["team_member_instructions"] == "Do X"
    mock_load.assert_not_awaited()


@pytest.mark.unit
async def test_team_create_skills_before_instructions(tmp_path: Any) -> None:
    """R5 ordering: skills appear BEFORE lead-supplied instructions."""
    mock_pool = _make_mock_pool_with_created_sessions(tmp_path)
    ctx = _make_ctx(mock_pool, tmp_path)
    cap = _make_cap()

    with patch(
        "wolfharness_toolsets.builtin.skills.load_skill_for_node",
        new=AsyncMock(return_value="Skill docs"),
    ):
        await cap.team_create(
            ctx,
            "my_team",
            [
                {
                    "agent": "worker",
                    "name": "worker_a",
                    "instructions": "Review now",
                    "skills": ["lodestone"],
                },
            ],
        )

    instructions: str = mock_pool.create_child_session.await_args.kwargs["team_member_instructions"]
    skill_pos = instructions.index("<skill-instruction")
    instr_pos = instructions.index("Review now")
    assert skill_pos < instr_pos
    assert "\n\n" in instructions  # skills_content + blank line + instructions


@pytest.mark.unit
async def test_team_create_skills_dedup(tmp_path: Any) -> None:
    """Dedup: repeated skill names inject exactly once, order preserved."""
    mock_pool = _make_mock_pool_with_created_sessions(tmp_path)
    ctx = _make_ctx(mock_pool, tmp_path)
    cap = _make_cap()

    with patch(
        "wolfharness_toolsets.builtin.skills.load_skill_for_node",
        new=AsyncMock(return_value="x"),
    ) as mock_load:
        await cap.team_create(
            ctx,
            "my_team",
            [{"agent": "worker", "name": "worker_a", "skills": ["a", "b", "a"]}],
        )

    assert mock_load.await_count == 2
    instructions: str = mock_pool.create_child_session.await_args.kwargs["team_member_instructions"]
    assert instructions.count("<skill-instruction") == 2
    assert instructions.index('name="a"') < instructions.index('name="b"')


# ---------------------------------------------------------------------------
# R2: team_add_member skills
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_team_add_member_with_skills_injects_block(tmp_path: Any) -> None:
    """R2: team_add_member ``skills`` param injects <skill-instruction> block."""
    mock_pool = _make_mock_pool_with_created_sessions(tmp_path)
    ctx = _make_ctx(mock_pool, tmp_path)
    cap = _make_cap()

    with patch(
        "wolfharness_toolsets.builtin.skills.load_skill_for_node",
        new=AsyncMock(return_value="Skill text"),
    ) as mock_load:
        result = await cap.team_add_member(
            ctx,
            "new_worker",
            "worker",
            skills=["lodestone"],
        )

    assert "Member 'new_worker' added to team" in result.return_value
    call_kwargs = mock_pool.create_child_session.await_args.kwargs
    instructions: str = call_kwargs["team_member_instructions"]
    assert '<skill-instruction name="lodestone">' in instructions
    assert call_kwargs["team_member_name"] == "new_worker"
    assert mock_load.await_args.kwargs["node_name"] == "worker"
    assert mock_load.await_args.kwargs["include_assembly"] is False


@pytest.mark.unit
async def test_team_add_member_without_skills_no_injection(tmp_path: Any) -> None:
    """R2: omitted skills param behaves exactly as before."""
    mock_pool = _make_mock_pool_with_created_sessions(tmp_path)
    ctx = _make_ctx(mock_pool, tmp_path)
    cap = _make_cap()

    with patch(
        "wolfharness_toolsets.builtin.skills.load_skill_for_node", new=AsyncMock()
    ) as mock_load:
        result = await cap.team_add_member(
            ctx,
            "new_worker",
            "worker",
            instructions="Be careful",
        )

    assert "Member 'new_worker' added to team" in result.return_value
    call_kwargs = mock_pool.create_child_session.await_args.kwargs
    assert call_kwargs["team_member_instructions"] == "Be careful"
    mock_load.assert_not_awaited()


@pytest.mark.unit
async def test_team_add_member_empty_skills_list_no_injection(tmp_path: Any) -> None:
    """Having an empty skills list is identical to omitting it."""
    mock_pool = _make_mock_pool_with_created_sessions(tmp_path)
    ctx = _make_ctx(mock_pool, tmp_path)
    cap = _make_cap()

    with patch(
        "wolfharness_toolsets.builtin.skills.load_skill_for_node", new=AsyncMock()
    ) as mock_load:
        await cap.team_add_member(ctx, "new_worker", "worker", skills=[])

    mock_load.assert_not_awaited()


# ---------------------------------------------------------------------------
# R3/R4: visibility + failures degrade to error text
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_skill_not_visible_renders_error_text_member_created(tmp_path: Any) -> None:
    """R3/R4: load_skill_for_node's not-found text renders; session still created."""
    mock_pool = _make_mock_pool_with_created_sessions(tmp_path)
    ctx = _make_ctx(mock_pool, tmp_path)
    cap = _make_cap()

    with patch(
        "wolfharness_toolsets.builtin.skills.load_skill_for_node",
        new=AsyncMock(return_value=("Skill 'lodestone' not found. Available skills: some_other")),
    ):
        result = await cap.team_add_member(ctx, "new_worker", "worker", skills=["lodestone"])

    assert "Member 'new_worker' added to team" in result.return_value
    instructions: str = mock_pool.create_child_session.await_args.kwargs["team_member_instructions"]
    assert "Skill 'lodestone' not found" in instructions


@pytest.mark.unit
async def test_skill_load_exception_renders_error_text_member_created(tmp_path: Any) -> None:
    """R4: resolver exception degrades to error text; member creation succeeds."""
    mock_pool = _make_mock_pool_with_created_sessions(tmp_path)
    ctx = _make_ctx(mock_pool, tmp_path)
    cap = _make_cap()

    with patch(
        "wolfharness_toolsets.builtin.skills.load_skill_for_node",
        new=AsyncMock(side_effect=RuntimeError("boom")),
    ):
        result = await cap.team_add_member(ctx, "new_worker", "worker", skills=["bad"])

    assert "Member 'new_worker' added to team" in result.return_value
    instructions: str = mock_pool.create_child_session.await_args.kwargs["team_member_instructions"]
    assert "Error: Failed to load skill 'bad'" in instructions
    assert "RuntimeError: boom" in instructions


# ---------------------------------------------------------------------------
# R5: pure instruction injection (no tool/MCP assembly)
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_skill_injection_no_tool_assembly_contract(tmp_path: Any) -> None:
    """R5: load_skill_for_node called with include_assembly=False; no extra kwargs."""
    mock_pool = _make_mock_pool_with_created_sessions(tmp_path)
    ctx = _make_ctx(mock_pool, tmp_path)
    cap = _make_cap()

    with patch(
        "wolfharness_toolsets.builtin.skills.load_skill_for_node",
        new=AsyncMock(return_value="docs"),
    ) as mock_load:
        await cap.team_add_member(ctx, "new_worker", "worker", skills=["s1"])

    # The call passed exactly: ctx, name, node_name=member, include_assembly=False
    await_args = mock_load.await_args
    assert await_args is not None
    assert await_args.kwargs == {"node_name": "worker", "include_assembly": False}
    # create_child_session carries only metadata — no capabilities/tools kwargs.
    call_kwargs = mock_pool.create_child_session.await_args.kwargs
    assert "capabilities" not in call_kwargs
    assert "tools" not in call_kwargs


# ---------------------------------------------------------------------------
# URI / reference-path support
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_skill_uri_reference_path_display_name(tmp_path: Any) -> None:
    """skill://foo/references/guide.md → display name foo/references/guide.md."""
    mock_pool = _make_mock_pool_with_created_sessions(tmp_path)
    ctx = _make_ctx(mock_pool, tmp_path)
    cap = _make_cap()

    with patch(
        "wolfharness_toolsets.builtin.skills.load_skill_for_node",
        new=AsyncMock(return_value="ref content"),
    ) as mock_load:
        await cap.team_add_member(
            ctx,
            "new_worker",
            "worker",
            skills=["skill://lodestone/references/guide.md"],
        )

    instructions: str = mock_pool.create_child_session.await_args.kwargs["team_member_instructions"]
    assert '<skill-instruction name="lodestone/references/guide.md">' in instructions
    # The full URI is passed through to load_skill_for_node.
    assert mock_load.await_args.args[1] == "skill://lodestone/references/guide.md"


# ---------------------------------------------------------------------------
# Defensive: bad-typed skills key
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_team_create_skills_non_list_degrades_gracefully(tmp_path: Any) -> None:
    """A non-list ``skills`` value is treated as empty — no crash."""
    mock_pool = _make_mock_pool_with_created_sessions(tmp_path)
    ctx = _make_ctx(mock_pool, tmp_path)
    cap = _make_cap()

    with patch(
        "wolfharness_toolsets.builtin.skills.load_skill_for_node", new=AsyncMock()
    ) as mock_load:
        result = await cap.team_create(
            ctx,
            "my_team",
            [{"agent": "worker", "name": "worker_a", "skills": "lodestone"}],  # type: ignore[dict-item]
        )

    assert "Team 'my_team' created" in result.return_value
    mock_load.assert_not_awaited()
    instructions: str = mock_pool.create_child_session.await_args.kwargs["team_member_instructions"]
    assert instructions == ""
