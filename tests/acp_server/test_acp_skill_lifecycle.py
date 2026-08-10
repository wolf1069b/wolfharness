"""Unit tests for ACPSession skill lifecycle.

Covers init_client_skills, _watch_skill_changes, and skill command execution.
Tasks T7.1-T7.4 (init_client_skills integration), T8.1-T8.5 (dynamic skill
updates), and T9.1-T9.5 (skill command execution).
"""

from __future__ import annotations

import asyncio
from pathlib import PurePosixPath
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from wolfharness.capabilities.change_event import ChangeEvent
from wolfharness.skills.command import SkillCommand
from wolfharness.skills.skill import Skill


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_skill(
    name: str = "test-skill",
    description: str = "A test skill",
    *,
    user_invocable: bool = True,
    instructions: str | None = "Test instructions content",
) -> Skill:
    """Create a minimal Skill instance for testing."""
    return Skill(
        name=name,
        description=description,
        skill_path=PurePosixPath(f"skill://{name}"),
        instructions=instructions,
        user_invocable=user_invocable,
    )


def _make_skill_command(
    skill: Skill | None = None,
    name: str = "test-skill",
    description: str = "A test skill",
) -> SkillCommand:
    """Create a minimal SkillCommand for testing."""
    if skill is None:
        skill = _make_skill(name=name, description=description)
    return SkillCommand(
        name=name,
        description=description,
        skill=skill,
        skill_uri=f"skill://{name}",
    )


def _make_mock_session(
    *,
    skills_list: list[Skill] | None = None,
    extension_registry: Any | None = None,
) -> MagicMock:
    """Create a mock ACPSession with the minimum attributes needed for skill tests.

    The returned mock has the async methods (init_client_skills,
    send_available_commands_update, _watch_skill_changes, close,
    execute_slash_command) bound to the real implementations so that
    we can test behaviour without a full ACPSession construction.
    """
    from slashed import CommandStore

    from wolfharness_server.acp_server.commands.skill_commands import ACPSkillBridge

    session = MagicMock()

    # Internal state
    session.command_store = CommandStore()
    session.command_store._initialize_sync()
    session._skill_bridge = ACPSkillBridge()
    session._skill_change_task = None
    session._skill_register_lock = asyncio.Lock()
    session._remote_commands = []
    session._update_callbacks = []
    session.log = MagicMock()
    session.fs = MagicMock()
    session.session_id = "test-session-id"
    session.cwd = "/test"

    # Mock host_context (set directly on session since it's a property on ACPSession)
    host_ctx = MagicMock()
    skills_manager = MagicMock()
    skills_manager.list_skills.return_value = skills_list or []
    skills_manager.add_skills_directory = AsyncMock()
    host_ctx.skills_registry = skills_manager
    host_ctx.extension_registry = extension_registry
    host_ctx.manifest = MagicMock()
    host_ctx.manifest.get_command_configs.return_value = None
    session.host_context = host_ctx

    # Mock agent with staged_content
    staged_content = MagicMock()
    session.agent = MagicMock()
    session.agent.staged_content = staged_content
    session.agent.name = "test-agent"
    session.agent.host_context = host_ctx
    session.agent.get_context = MagicMock(return_value=MagicMock(data=session, node=session.agent))

    # Mock notifications
    session.notifications = MagicMock()
    session.notifications.update_commands = AsyncMock()
    session.notifications.send_agent_text = AsyncMock()

    # Bind real methods
    from wolfharness_server.acp_server.session import ACPSession

    session._register_skill_commands = ACPSession._register_skill_commands.__get__(
        session, ACPSession
    )
    session._start_skill_change_watcher = ACPSession._start_skill_change_watcher.__get__(
        session, ACPSession
    )
    session._watch_skill_changes = ACPSession._watch_skill_changes.__get__(session, ACPSession)
    session.init_client_skills = ACPSession.init_client_skills.__get__(session, ACPSession)
    session.send_available_commands_update = ACPSession.send_available_commands_update.__get__(
        session, ACPSession
    )
    session.get_acp_commands = ACPSession.get_acp_commands.__get__(session, ACPSession)
    session._notify_command_update = ACPSession._notify_command_update.__get__(session, ACPSession)
    session.close = ACPSession.close.__get__(session, ACPSession)
    session.execute_slash_command = ACPSession.execute_slash_command.__get__(session, ACPSession)
    session._send_toast = AsyncMock()

    return session


def _async_iter(events: list[Any | None]):
    """Create an async iterator yielding the given events.

    ``None`` entries are skipped (they represent sentinel values in the
    merge_change_streams protocol). The iterator completes after all
    events are yielded.
    """

    async def _gen():
        for event in events:
            if event is None:
                continue
            yield event

    return _gen()


# ---------------------------------------------------------------------------
# T7 — init_client_skills() integration
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_init_client_skills_registers_client_skills_in_command_store() -> None:
    """T7.1: After init_client_skills() completes.

    Client-discovered skills appear in command_store.
    """
    skill = _make_skill(name="client-skill", description="A client-side skill")
    session = _make_mock_session(skills_list=[skill])

    await session.init_client_skills()

    # The skill should have been registered as a slash command
    cmd = session.command_store.get_command("client-skill")
    assert cmd is not None
    assert cmd.name == "client-skill"


@pytest.mark.unit
async def test_init_client_skills_calls_send_available_commands_update() -> None:
    """T7.2: send_available_commands_update() is called after client skill registration."""
    skill = _make_skill(name="client-skill", description="A client-side skill")
    session = _make_mock_session(skills_list=[skill])
    session.send_available_commands_update = AsyncMock()

    await session.init_client_skills()

    session.send_available_commands_update.assert_called_once()


@pytest.mark.unit
async def test_init_client_skills_with_no_skills_directory_no_error() -> None:
    """T7.3: init_client_skills() with no .claude/skills directory (no error, no new commands)."""
    session = _make_mock_session(skills_list=[])
    session.send_available_commands_update = AsyncMock()

    # Should not raise
    await session.init_client_skills()

    # add_skills_directory was called (attempted to add .claude/skills)
    session.agent.host_context.skills_registry.add_skills_directory.assert_called_once()
    # send_available_commands_update was still called
    session.send_available_commands_update.assert_called_once()
    # No skill commands should be registered (built-in help/exit may exist)
    all_cmds = session.command_store.list_commands()
    skill_cmds = [c for c in all_cmds if c.category == "skill"]
    assert len(skill_cmds) == 0


@pytest.mark.unit
async def test_pool_and_client_skills_coexist_without_duplicates() -> None:
    """T7.4: Pool + client skills coexist in command_store without duplicates."""
    pool_skill = _make_skill(name="pool-skill", description="Pool-level skill")
    client_skill = _make_skill(name="client-skill", description="Client-level skill")
    # Pool skill pre-registered via _register_skill_commands in __post_init__
    session = _make_mock_session(skills_list=[pool_skill, client_skill])
    session._register_skill_commands()

    # Both skills should be in command_store
    assert session.command_store.get_command("pool-skill") is not None
    assert session.command_store.get_command("client-skill") is not None

    # Re-register with same skills (simulating init_client_skills re-registering)
    # _register_skill_commands uses replace=True, so no duplicates
    session._register_skill_commands()

    pool_cmds = [c for c in session.command_store.list_commands() if c.name == "pool-skill"]
    client_cmds = [c for c in session.command_store.list_commands() if c.name == "client-skill"]
    assert len(pool_cmds) == 1
    assert len(client_cmds) == 1


# ---------------------------------------------------------------------------
# T8 — Dynamic skill updates
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_watch_skill_changes_calls_register_on_skills_changed_event() -> None:
    """T8.1: _watch_skill_changes() calls _register_skill_commands().

    Called when skills_changed event arrives.
    """
    event = ChangeEvent(capability_name="test-cap", kind="skills_changed")
    mock_stream = _async_iter([event, None])

    mock_ext_registry = MagicMock()
    mock_ext_registry.merge_change_streams.return_value = mock_stream

    session = _make_mock_session(extension_registry=mock_ext_registry)
    session._register_skill_commands = MagicMock()
    session.send_available_commands_update = AsyncMock()

    await session._watch_skill_changes()

    session._register_skill_commands.assert_called_once()
    session.send_available_commands_update.assert_called_once()


@pytest.mark.unit
async def test_watch_skill_changes_ignores_non_skills_changed_events() -> None:
    """T8.2: _watch_skill_changes() ignores non-skills_changed events."""
    tools_event = ChangeEvent(capability_name="test-cap", kind="tools_changed")
    resources_event = ChangeEvent(capability_name="test-cap", kind="resource_list_changed")
    skills_event = ChangeEvent(capability_name="test-cap", kind="skills_changed")
    mock_stream = _async_iter([tools_event, resources_event, skills_event, None])

    mock_ext_registry = MagicMock()
    mock_ext_registry.merge_change_streams.return_value = mock_stream

    session = _make_mock_session(extension_registry=mock_ext_registry)
    session._register_skill_commands = MagicMock()
    session.send_available_commands_update = AsyncMock()

    await session._watch_skill_changes()

    # Only the skills_changed event should trigger registration
    session._register_skill_commands.assert_called_once()
    session.send_available_commands_update.assert_called_once()


@pytest.mark.unit
async def test_watcher_task_cancelled_on_close() -> None:
    """T8.3: Watcher task is cancelled when close() is called."""

    # Create a task that blocks forever (simulating a long-running watcher)
    async def _block_forever() -> None:
        await asyncio.Event().wait()

    session = _make_mock_session()
    session._skill_change_task = asyncio.create_task(_block_forever())

    # close() should cancel and await the task
    await session.close()

    assert session._skill_change_task is None


@pytest.mark.unit
async def test_watch_skill_changes_handles_none_stream_no_crash() -> None:
    """T8.4: _watch_skill_changes() handles merge_change_streams().

    Returns None (no crash, task exits).
    """
    mock_ext_registry = MagicMock()
    mock_ext_registry.merge_change_streams.return_value = None

    session = _make_mock_session(extension_registry=mock_ext_registry)
    session._register_skill_commands = MagicMock()

    # Should return without error
    await session._watch_skill_changes()

    # _register_skill_commands should NOT have been called
    session._register_skill_commands.assert_not_called()


@pytest.mark.unit
async def test_concurrent_register_skill_commands_serialized_by_lock() -> None:
    """T8.5: Concurrent _register_skill_commands() calls are serialized.

    Serialized by lock (no duplicate commands).
    """
    skill = _make_skill(name="concurrent-skill", description="Concurrent skill")
    session = _make_mock_session(skills_list=[skill])

    # Call _register_skill_commands concurrently multiple times
    # The lock in _watch_skill_changes serializes access, but
    # _register_skill_commands itself is sync. We test that calling
    # it multiple times with replace=True doesn't produce duplicates.
    for _ in range(5):
        session._register_skill_commands()

    cmds = [c for c in session.command_store.list_commands() if c.name == "concurrent-skill"]
    assert len(cmds) == 1


# ---------------------------------------------------------------------------
# T9 — Skill command execution
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_execute_slash_command_finds_and_executes_skill() -> None:
    """T9.1: execute_slash_command finds skill in command_store.

    Finds and executes it.
    """
    skill = _make_skill(
        name="my-skill",
        description="My test skill",
        instructions="Do something useful",
    )
    session = _make_mock_session(skills_list=[skill])
    session._register_skill_commands()

    # Mock command_store.execute_command to verify it's called
    session.command_store.execute_command = AsyncMock()

    await session.execute_slash_command("/my-skill some args")

    session.command_store.execute_command.assert_called_once()
    call_args = session.command_store.execute_command.call_args
    command_str = call_args.args[0]
    assert "my-skill" in command_str


@pytest.mark.unit
async def test_skill_executor_loads_instructions_and_injects_into_staged_content() -> None:
    """T9.2: Skill executor loads instructions via skill.load_instructions().

    Injects into staged_content.
    """
    from wolfharness_server.opencode_server.skill_bridge import create_skill_command

    skill = _make_skill(
        name="inject-skill",
        description="Test injection",
        instructions="Run this skill carefully.",
    )
    skill_cmd = _make_skill_command(skill=skill, name="inject-skill")
    slashed_cmd = create_skill_command(skill_cmd)

    # Create mock context with node.staged_content
    staged_content = MagicMock()
    mock_node = MagicMock()
    mock_node.staged_content = staged_content
    mock_ctx = MagicMock()
    mock_ctx.data = MagicMock()
    mock_ctx.data.node = mock_node
    mock_ctx.print = AsyncMock()

    # Execute the skill command's executor
    await slashed_cmd.execute(mock_ctx, ["do", "something"], {})

    # Verify instructions were loaded and injected
    staged_content.add_text.assert_called_once()
    injected_text = staged_content.add_text.call_args.args[0]
    assert "Run this skill carefully." in injected_text
    assert "do something" in injected_text


@pytest.mark.unit
async def test_skill_executor_with_load_instructions_raising_value_error() -> None:
    """T9.3: Skill executor with load_instructions() raising ValueError.

    Sends 'no instructions' message, no injection.
    """
    from wolfharness_server.opencode_server.skill_bridge import create_skill_command

    # Create a virtual skill (PurePosixPath) with instructions=None
    # load_instructions() will raise ValueError for virtual skills without pre-set instructions
    skill = Skill(
        name="virtual-skill",
        description="A virtual skill",
        skill_path=PurePosixPath("skill://virtual-skill"),
        instructions=None,
        user_invocable=True,
    )
    skill_cmd = _make_skill_command(skill=skill, name="virtual-skill")
    slashed_cmd = create_skill_command(skill_cmd)

    # Create mock context
    staged_content = MagicMock()
    mock_node = MagicMock()
    mock_node.staged_content = staged_content
    mock_ctx = MagicMock()
    mock_ctx.data = MagicMock()
    mock_ctx.data.node = mock_node
    mock_ctx.print = AsyncMock()

    # Execute — should not raise, should print "no instructions" message
    await slashed_cmd.execute(mock_ctx, ["arg1"], {})

    # staged_content.add_text should NOT be called (no instructions to inject)
    staged_content.add_text.assert_not_called()

    # print should have been called with "no instructions" message
    mock_ctx.print.assert_called()
    print_call_arg = mock_ctx.print.call_args.args[0]
    assert "no instructions" in print_call_arg.lower()


@pytest.mark.unit
async def test_execute_slash_command_nonexistent_returns_without_error() -> None:
    """T9.4: execute_slash_command('/nonexistent') returns without error."""
    session = _make_mock_session(skills_list=[])

    # Mock command_store.get_command to return None for nonexistent
    session.command_store.get_command = MagicMock(return_value=None)
    session.command_store.execute_command = AsyncMock()
    session.command_store.create_context = MagicMock(return_value=MagicMock())

    # Should not raise
    await session.execute_slash_command("/nonexistent")

    # execute_command should still be called (the command_store handles unknown commands)
    # or it may raise internally which gets caught by the try/except
    # The key is no exception propagates


@pytest.mark.unit
async def test_injected_prompt_format_uses_correct_xml_tags() -> None:
    """T9.5: Injected prompt format.

    uses <skill-instruction> + <user-request> XML tags.
    """
    from wolfharness_server.opencode_server.skill_bridge import create_skill_command

    skill = _make_skill(
        name="format-skill",
        description="Test format",
        instructions="Step 1: Do X\nStep 2: Do Y",
    )
    skill_cmd = _make_skill_command(skill=skill, name="format-skill")
    slashed_cmd = create_skill_command(skill_cmd)

    staged_content = MagicMock()
    mock_node = MagicMock()
    mock_node.staged_content = staged_content
    mock_ctx = MagicMock()
    mock_ctx.data = MagicMock()
    mock_ctx.data.node = mock_node
    mock_ctx.print = AsyncMock()

    user_args = ["analyze", "the", "codebase"]
    await slashed_cmd.execute(mock_ctx, user_args, {})

    staged_content.add_text.assert_called_once()
    injected = staged_content.add_text.call_args.args[0]

    # Verify XML tag structure
    assert "<skill-instruction>" in injected
    assert "</skill-instruction>" in injected
    assert "<user-request>" in injected
    assert "</user-request>" in injected

    # Verify content within tags
    assert "Step 1: Do X" in injected
    assert "Step 2: Do Y" in injected
    assert "analyze the codebase" in injected

    # Verify ordering: skill-instruction comes before user-request
    instr_pos = injected.index("<skill-instruction>")
    request_pos = injected.index("<user-request>")
    assert instr_pos < request_pos
