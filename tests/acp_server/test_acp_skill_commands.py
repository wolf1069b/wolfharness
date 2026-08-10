"""Unit tests for ACPSkillBridge and ACPSession._register_skill_commands().

Covers tasks T5.1-T5.5 (ACPSkillBridge) and T6.1-T6.6 (_register_skill_commands).
"""

from __future__ import annotations

from pathlib import PurePosixPath
from unittest.mock import MagicMock

import pytest
from slashed import Command as SlashedCommand, CommandStore

from wolfharness.skills.command import SkillCommand
from wolfharness.skills.skill import Skill
from wolfharness_server.acp_server.commands.skill_commands import ACPSkillBridge


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_skill(
    name: str = "test-skill",
    description: str = "A test skill",
    *,
    user_invocable: bool = True,
    instructions: str | None = "Test instructions",
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


# ---------------------------------------------------------------------------
# T5 — ACPSkillBridge unit tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_handle_change_adds_slashed_command_to_bridge() -> None:
    """T5.1: handle_change() adds a SlashedCommand (not AvailableCommand) to _commands."""
    bridge = ACPSkillBridge()
    skill_cmd = _make_skill_command(name="my-skill", description="My skill")

    bridge.handle_change("my-skill", skill_cmd)

    commands = bridge.get_commands()
    assert len(commands) == 1
    cmd = commands[0]
    assert isinstance(cmd, SlashedCommand)
    assert cmd.name == "my-skill"


@pytest.mark.unit
def test_handle_change_with_none_removes_command() -> None:
    """T5.2: handle_change(name, None) removes the command from _commands."""
    bridge = ACPSkillBridge()
    skill_cmd = _make_skill_command(name="removable", description="Removable skill")

    bridge.handle_change("removable", skill_cmd)
    assert len(bridge.get_commands()) == 1

    bridge.handle_change("removable", None)

    assert len(bridge.get_commands()) == 0


@pytest.mark.unit
def test_get_commands_returns_correct_count_and_names() -> None:
    """T5.3: get_commands() returns correct count and names after multiple handle_change calls."""
    bridge = ACPSkillBridge()
    cmd_a = _make_skill_command(name="skill-a", description="Skill A")
    cmd_b = _make_skill_command(name="skill-b", description="Skill B")
    cmd_c = _make_skill_command(name="skill-c", description="Skill C")

    bridge.handle_change("skill-a", cmd_a)
    bridge.handle_change("skill-b", cmd_b)
    bridge.handle_change("skill-c", cmd_c)

    commands = bridge.get_commands()
    assert len(commands) == 3
    names = {cmd.name for cmd in commands}
    assert names == {"skill-a", "skill-b", "skill-c"}


@pytest.mark.unit
def test_slashed_command_has_correct_fields() -> None:
    """T5.4: SlashedCommand produced by bridge has correct name, description, category, usage."""
    bridge = ACPSkillBridge()
    skill_cmd = _make_skill_command(
        name="my-awesome-skill",
        description="Does awesome things",
    )
    skill_cmd_with_hint = SkillCommand(
        name="my-awesome-skill",
        description="Does awesome things",
        skill=skill_cmd.skill,
        input_hint="/my-awesome-skill <arg1>",
        skill_uri="skill://my-awesome-skill",
    )

    bridge.handle_change("my-awesome-skill", skill_cmd_with_hint)

    commands = bridge.get_commands()
    assert len(commands) == 1
    cmd = commands[0]
    assert cmd.name == "my-awesome-skill"
    assert cmd.description == "Does awesome things"
    assert cmd.category == "skill"
    assert cmd.usage == "/my-awesome-skill <arg1>"


@pytest.mark.unit
def test_bridge_with_empty_registry_returns_zero_commands() -> None:
    """T5.5: Bridge with empty registry returns zero commands."""
    bridge = ACPSkillBridge()

    commands = bridge.get_commands()

    assert commands == []
    assert len(commands) == 0


# ---------------------------------------------------------------------------
# T6 — ACPSession._register_skill_commands unit tests
# ---------------------------------------------------------------------------


def _make_mock_host_context(
    skills: list[Skill] | None = None,
) -> MagicMock:
    """Create a mock HostContext with a skills_registry."""
    ctx = MagicMock()
    skills_manager = MagicMock()
    skills_manager.list_skills.return_value = skills or []
    ctx.skills_registry = skills_manager
    ctx.manifest = MagicMock()
    ctx.manifest.get_command_configs.return_value = None
    ctx.extension_registry = None
    return ctx


def _make_minimal_session(
    host_context: MagicMock | None = None,
    skill_bridge: ACPSkillBridge | None = None,
) -> MagicMock:
    """Create a mock ACPSession-like object for _register_skill_commands testing.

    We mock the session because ACPSession.__post_init__ requires many real
    dependencies (ACPFileSystem, ACPNotifications, etc.) that are irrelevant
    to unit-testing _register_skill_commands in isolation.
    """
    if host_context is None:
        host_context = _make_mock_host_context()
    if skill_bridge is None:
        skill_bridge = ACPSkillBridge()

    session = MagicMock()
    session.host_context = host_context
    session._skill_bridge = skill_bridge
    session.command_store = CommandStore()
    session.command_store._initialize_sync()
    session._notify_command_update = MagicMock()
    session.log = MagicMock()
    return session


def _call_register_skill_commands(session: MagicMock) -> None:
    """Invoke the real _register_skill_commands method on a mock session."""
    # We need to call the unbound method from ACPSession on our mock
    from wolfharness_server.acp_server.session import ACPSession

    ACPSession._register_skill_commands(session)  # type: ignore[arg-type]


def _skill_commands_from_store(store: CommandStore) -> list[SlashedCommand]:
    """Return only skill-category commands from a CommandStore.

    CommandStore initializes with built-in help/exit commands; we filter
    those out to isolate the commands registered by _register_skill_commands.
    """
    return [cmd for cmd in store.list_commands() if cmd.category == "skill"]


@pytest.mark.unit
def test_register_skill_commands_registers_user_invocable_skills() -> None:
    """T6.1: Pool-level skills with user_invocable=True are registered in command_store."""
    skill_a = _make_skill(name="skill-a", description="Skill A")
    skill_b = _make_skill(name="skill-b", description="Skill B")
    host_context = _make_mock_host_context(skills=[skill_a, skill_b])
    session = _make_minimal_session(host_context=host_context)

    _call_register_skill_commands(session)

    commands = _skill_commands_from_store(session.command_store)
    assert len(commands) == 2
    names = {cmd.name for cmd in commands}
    assert names == {"skill-a", "skill-b"}


@pytest.mark.unit
def test_register_skill_commands_excludes_non_user_invocable() -> None:
    """T6.2: Skills with user_invocable=False are excluded from command_store."""
    invocable = _make_skill(name="invocable", description="Invocable", user_invocable=True)
    non_invocable = _make_skill(
        name="non-invocable", description="Non-invocable", user_invocable=False
    )
    host_context = _make_mock_host_context(skills=[invocable, non_invocable])
    session = _make_minimal_session(host_context=host_context)

    _call_register_skill_commands(session)

    commands = _skill_commands_from_store(session.command_store)
    assert len(commands) == 1
    assert commands[0].name == "invocable"


@pytest.mark.unit
def test_register_skill_commands_is_idempotent() -> None:
    """T6.3: Calling _register_skill_commands twice produces no duplicates."""
    skill_a = _make_skill(name="skill-a", description="Skill A")
    host_context = _make_mock_host_context(skills=[skill_a])
    session = _make_minimal_session(host_context=host_context)

    _call_register_skill_commands(session)
    _call_register_skill_commands(session)

    commands = _skill_commands_from_store(session.command_store)
    assert len(commands) == 1
    assert commands[0].name == "skill-a"


@pytest.mark.unit
def test_register_skill_commands_with_empty_registry() -> None:
    """T6.4: _register_skill_commands with empty skills registry adds no commands."""
    host_context = _make_mock_host_context(skills=[])
    session = _make_minimal_session(host_context=host_context)

    _call_register_skill_commands(session)

    commands = _skill_commands_from_store(session.command_store)
    assert len(commands) == 0
    # _notify_command_update should NOT be called when cmd_count is 0
    session._notify_command_update.assert_not_called()


@pytest.mark.unit
def test_register_skill_commands_calls_notify_command_update() -> None:
    """T6.5: _notify_command_update() is called after registration when commands exist."""
    skill_a = _make_skill(name="skill-a", description="Skill A")
    host_context = _make_mock_host_context(skills=[skill_a])
    session = _make_minimal_session(host_context=host_context)

    _call_register_skill_commands(session)

    session._notify_command_update.assert_called_once()


@pytest.mark.unit
def test_register_skill_commands_handles_empty_skills_manager() -> None:
    """T6.6: _register_skill_commands handles an empty SkillsManager gracefully."""
    # Per the task notes: HostContext.skills_registry is non-optional SkillsManager.
    # Test with an empty SkillsManager (list_skills returns []).
    host_context = _make_mock_host_context(skills=[])
    session = _make_minimal_session(host_context=host_context)

    _call_register_skill_commands(session)

    commands = _skill_commands_from_store(session.command_store)
    assert len(commands) == 0
    session._notify_command_update.assert_not_called()
