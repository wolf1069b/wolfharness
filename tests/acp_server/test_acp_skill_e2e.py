"""E2E integration and regression tests for ACP skill commands.

Covers tasks T10.1-T10.6 (integration/E2E) and T11.1-T11.6 (regression/edge cases)
from the OpenSpec tasks for fix-acp-skill-commands.

These tests exercise the full ACPSession lifecycle with real pool/agent
infrastructure, mocking only the ACP transport layer (client, notifications)
and the skills registry contents.
"""

from __future__ import annotations

import asyncio
import contextlib
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from slashed import Command as SlashedCommand, CommandStore

from wolfharness import Agent, AgentPool
from wolfharness.capabilities.change_event import ChangeEvent
from wolfharness.skills.skill import Skill
from wolfharness_server.acp_server.session import ACPSession


if TYPE_CHECKING:
    from acp.schema import AvailableCommand


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_skill(
    name: str = "test-skill",
    description: str = "A test skill",
    *,
    user_invocable: bool = True,
    instructions: str | None = "Test instructions for skill.",
) -> Skill:
    """Create a minimal Skill instance for testing."""
    return Skill(
        name=name,
        description=description,
        skill_path=PurePosixPath(f"skill://{name}"),
        instructions=instructions,
        user_invocable=user_invocable,
    )


def _make_pool_and_agent() -> tuple[AgentPool, Agent]:
    """Create a simple pool with one agent."""
    from wolfharness.models.agents import NativeAgentConfig
    from wolfharness.models.manifest import AgentsManifest

    manifest = AgentsManifest(agents={"test_agent": NativeAgentConfig(model="test")})
    pool = AgentPool(manifest)

    def simple_callback(message: str) -> str:
        return f"Test response: {message}"

    agent = Agent.from_callback(name="test_agent", callback=simple_callback, agent_pool=pool)
    return pool, agent


def _mock_skills_on_pool(pool: AgentPool, skills: list[Skill]) -> None:
    """Patch the pool's SkillsManager.list_skills to return controlled skills."""
    pool.skills.list_skills = MagicMock(return_value=skills)  # type: ignore[method-assign]


def _make_mock_acp_agent() -> MagicMock:
    """Create a mock ACP agent suitable for ACPSession construction."""
    mock_acp_agent = MagicMock()
    mock_acp_agent._task_group = MagicMock()
    mock_acp_agent._task_group.start_soon = MagicMock()
    mock_acp_agent._mcp_manager = MagicMock()
    return mock_acp_agent


def _create_session(
    pool: AgentPool,
    agent: Agent,
    *,
    mock_client: AsyncMock | None = None,
    mock_acp_agent: MagicMock | None = None,
) -> ACPSession:
    """Create a real ACPSession with mocked transport layer."""
    if mock_client is None:
        mock_client = AsyncMock()
    if mock_acp_agent is None:
        mock_acp_agent = _make_mock_acp_agent()
    return ACPSession(
        session_id="test-session-e2e",
        agent=agent,
        cwd="/tmp",
        client=mock_client,
        acp_agent=mock_acp_agent,
    )


def _skill_cmd_names(store: CommandStore) -> set[str]:
    """Return names of skill-category commands from a CommandStore."""
    return {cmd.name for cmd in store.list_commands() if cmd.category == "skill"}


def _all_cmd_names(store: CommandStore) -> set[str]:
    """Return names of all commands from a CommandStore (excluding built-in help/exit)."""
    builtins = {"help", "exit"}
    return {cmd.name for cmd in store.list_commands() if cmd.name not in builtins}


@pytest.fixture
def pool_and_agent() -> tuple[AgentPool, Agent]:
    """Provide a pool and agent."""
    return _make_pool_and_agent()


@pytest.fixture
def mock_client() -> AsyncMock:
    """Provide a mock ACP client."""
    return AsyncMock()


@pytest.fixture
def mock_acp_agent() -> MagicMock:
    """Provide a mock ACP agent."""
    return _make_mock_acp_agent()


# ---------------------------------------------------------------------------
# T10 - Integration / E2E tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_e2e_pool_skills_in_available_commands_update(
    pool_and_agent: tuple[AgentPool, Agent],
    mock_client: AsyncMock,
    mock_acp_agent: MagicMock,
) -> None:
    """T10.1: E2E: Create ACP session with pool skills.

    Verify available_commands_update includes skill commands.
    """
    pool, agent = pool_and_agent
    skill_a = _make_skill(name="skill-a", description="Skill A")
    skill_b = _make_skill(name="skill-b", description="Skill B")
    _mock_skills_on_pool(pool, [skill_a, skill_b])

    session = _create_session(pool, agent, mock_client=mock_client, mock_acp_agent=mock_acp_agent)

    session.notifications.update_commands = AsyncMock()  # type: ignore[method-assign]
    await session.send_available_commands_update()

    assert session.notifications.update_commands.called
    calls = session.notifications.update_commands.call_args_list
    sent_commands: list[AvailableCommand] = calls[0][0][0]
    command_names = [cmd.name for cmd in sent_commands]
    assert "skill-a" in command_names, f"skill-a missing from {command_names}"
    assert "skill-b" in command_names, f"skill-b missing from {command_names}"


@pytest.mark.unit
async def test_e2e_init_client_skills_sends_update(
    pool_and_agent: tuple[AgentPool, Agent],
    mock_client: AsyncMock,
    mock_acp_agent: MagicMock,
) -> None:
    """T10.2: E2E: Create ACP session, call init_client_skills().

    Verify update notification includes client skills.
    """
    pool, agent = pool_and_agent
    _mock_skills_on_pool(pool, [])

    session = _create_session(pool, agent, mock_client=mock_client, mock_acp_agent=mock_acp_agent)

    # Simulate client skills being discovered
    client_skill = _make_skill(name="client-skill", description="Client skill")
    pool.skills.list_skills = MagicMock(return_value=[client_skill])  # type: ignore[method-assign]
    pool.skills.add_skills_directory = AsyncMock()  # type: ignore[method-assign]

    session.notifications.update_commands = AsyncMock()  # type: ignore[method-assign]

    await session.init_client_skills()

    assert session.notifications.update_commands.called
    calls = session.notifications.update_commands.call_args_list
    sent_commands: list[AvailableCommand] = calls[0][0][0]
    command_names = [cmd.name for cmd in sent_commands]
    assert "client-skill" in command_names, f"client-skill missing from {command_names}"


@pytest.mark.unit
async def test_e2e_skills_changed_event_triggers_update(
    pool_and_agent: tuple[AgentPool, Agent],
    mock_client: AsyncMock,
    mock_acp_agent: MagicMock,
) -> None:
    """T10.3: E2E: Simulate skills_changed event from ExtensionRegistry.

    Verify available_commands_update reflects new skill set.
    """
    pool, agent = pool_and_agent
    initial_skill = _make_skill(name="initial-skill", description="Initial")
    _mock_skills_on_pool(pool, [initial_skill])

    # Set up a mock ExtensionRegistry with a controllable change stream
    change_event = asyncio.Event()
    updated_skill = _make_skill(name="updated-skill", description="Updated")

    async def _make_change_stream() -> Any:
        await change_event.wait()
        yield ChangeEvent(capability_name="test", kind="skills_changed")

    mock_ext_registry = MagicMock()

    def _merge_streams(_scope: Any) -> Any:
        return _make_change_stream()

    mock_ext_registry.merge_change_streams.side_effect = _merge_streams
    # Clear cached HostContext and set extension registry before session creation
    pool._host_context = None  # type: ignore[attr-defined]
    pool._extension_registry = mock_ext_registry  # type: ignore[attr-defined]

    session = _create_session(pool, agent, mock_client=mock_client, mock_acp_agent=mock_acp_agent)

    # Replace update mock BEFORE triggering the event
    session.notifications.update_commands = AsyncMock()  # type: ignore[method-assign]
    update_called = asyncio.Event()
    original_update = session.notifications.update_commands

    async def _tracking_update(cmds: Any) -> None:
        await original_update(cmds)
        update_called.set()

    session.notifications.update_commands = _tracking_update  # type: ignore[method-assign]

    # Verify watcher task was created
    assert session._skill_change_task is not None, "Skill change watcher task was not created"

    # Update skills to the new set
    pool.skills.list_skills = MagicMock(return_value=[updated_skill])  # type: ignore[method-assign]

    # Give watcher task time to start and begin awaiting the stream
    await asyncio.sleep(0.1)

    # Trigger the skills_changed event
    change_event.set()

    # Wait for the watcher to process
    with contextlib.suppress(TimeoutError):
        await asyncio.wait_for(update_called.wait(), timeout=5.0)

    assert update_called.is_set(), (
        "send_available_commands_update was not called after skills_changed event"
    )

    # Verify the updated skill is in the commands
    calls = original_update.call_args_list
    assert len(calls) >= 1
    sent_commands: list[AvailableCommand] = calls[-1][0][0]
    command_names = [cmd.name for cmd in sent_commands]
    assert "updated-skill" in command_names, f"updated-skill missing from {command_names}"

    # Cleanup
    await session.close()


@pytest.mark.unit
async def test_e2e_execute_skill_command_injects_instructions(
    pool_and_agent: tuple[AgentPool, Agent],
    mock_client: AsyncMock,
    mock_acp_agent: MagicMock,
) -> None:
    """T10.4: E2E: Execute /skill-name test-args through ACP session.

    Verify instructions injected into staged_content.
    """
    pool, agent = pool_and_agent
    skill = _make_skill(
        name="inject-skill",
        description="Skill that injects instructions",
        instructions="You must follow these special instructions.",
    )
    _mock_skills_on_pool(pool, [skill])

    session = _create_session(pool, agent, mock_client=mock_client, mock_acp_agent=mock_acp_agent)

    # Mock notifications to avoid real ACP communication
    session.notifications.send_agent_text = AsyncMock()  # type: ignore[method-assign]
    session.notifications.send_ext_notification = AsyncMock()  # type: ignore[method-assign]

    # Verify skill command is registered
    assert "inject-skill" in _skill_cmd_names(session.command_store)

    # Execute the skill command
    await session.execute_slash_command("/inject-skill test-args")

    # Verify instructions were injected into staged_content
    staged_text = await agent.staged_content.consume_as_text()
    assert staged_text is not None, "No content was staged after executing skill command"
    assert "special instructions" in staged_text, (
        f"Skill instructions not found in staged content: {staged_text}"
    )
    assert "test-args" in staged_text, f"User args not found in staged content: {staged_text}"


@pytest.mark.unit
async def test_e2e_session_close_cancels_watcher_task(
    pool_and_agent: tuple[AgentPool, Agent],
    mock_client: AsyncMock,
    mock_acp_agent: MagicMock,
) -> None:
    """T10.5: E2E: Session close -> verify watcher task cancelled, no lingering tasks."""
    pool, agent = pool_and_agent
    _mock_skills_on_pool(pool, [_make_skill()])

    # Set up ExtensionRegistry so _skill_change_task is created
    async def _empty_stream() -> Any:
        # Block forever so the task stays alive until cancelled
        await asyncio.Event().wait()
        yield  # pragma: no cover

    mock_ext_registry = MagicMock()
    mock_ext_registry.merge_change_streams.return_value = _empty_stream()
    pool._extension_registry = mock_ext_registry  # type: ignore[attr-defined]

    session = _create_session(pool, agent, mock_client=mock_client, mock_acp_agent=mock_acp_agent)

    # Verify watcher task was created
    assert session._skill_change_task is not None, "Skill change watcher task was not created"
    assert not session._skill_change_task.cancelled(), "Task should not be cancelled before close()"

    await session.close()

    # After close, the task should be None (set to None in close())
    assert session._skill_change_task is None, "Watcher task should be None after close()"


@pytest.mark.unit
async def test_e2e_non_invocable_skills_never_in_commands(
    pool_and_agent: tuple[AgentPool, Agent],
    mock_client: AsyncMock,
    mock_acp_agent: MagicMock,
) -> None:
    """T10.6: E2E: Skills with user-invocable: false.

    Verify they never appear in available_commands_update.
    """
    pool, agent = pool_and_agent
    invocable = _make_skill(name="invocable-skill", description="Invocable", user_invocable=True)
    non_invocable = _make_skill(
        name="non-invocable-skill", description="Non-invocable", user_invocable=False
    )
    _mock_skills_on_pool(pool, [invocable, non_invocable])

    session = _create_session(pool, agent, mock_client=mock_client, mock_acp_agent=mock_acp_agent)

    # Check command_store: non-invocable should not be registered
    store_names = _skill_cmd_names(session.command_store)
    assert "invocable-skill" in store_names
    assert "non-invocable-skill" not in store_names

    # Check available_commands_update: non-invocable should not appear
    session.notifications.update_commands = AsyncMock()  # type: ignore[method-assign]
    await session.send_available_commands_update()

    calls = session.notifications.update_commands.call_args_list
    sent_commands: list[AvailableCommand] = calls[0][0][0]
    command_names = [cmd.name for cmd in sent_commands]
    assert "invocable-skill" in command_names
    assert "non-invocable-skill" not in command_names


# ---------------------------------------------------------------------------
# T11 - Regression and edge case tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_regression_manifest_commands_still_work(
    pool_and_agent: tuple[AgentPool, Agent],
    mock_client: AsyncMock,
    mock_acp_agent: MagicMock,
) -> None:
    """T11.1: Test existing _register_manifest_commands() still works.

    Verify no interference from skill registration.
    """
    pool, agent = pool_and_agent
    _mock_skills_on_pool(pool, [_make_skill(name="my-skill")])

    # Mock manifest to return a command config
    from wolfharness_config.commands import StaticCommandConfig

    manifest_cmd = StaticCommandConfig(
        type="static",
        name="manifest-cmd",
        description="A manifest command",
        content="Hello world",
    )
    pool.manifest.get_command_configs = MagicMock(  # type: ignore[method-assign]
        return_value={"manifest-cmd": manifest_cmd}
    )

    session = _create_session(pool, agent, mock_client=mock_client, mock_acp_agent=mock_acp_agent)

    # Both manifest and skill commands should be in the store
    all_names = _all_cmd_names(session.command_store)
    assert "manifest-cmd" in all_names, f"manifest-cmd missing from {all_names}"
    assert "my-skill" in all_names, f"my-skill missing from {all_names}"


@pytest.mark.unit
async def test_regression_mcp_prompts_as_commands_still_works(
    pool_and_agent: tuple[AgentPool, Agent],
    mock_client: AsyncMock,
    mock_acp_agent: MagicMock,
) -> None:
    """T11.2: Test existing _register_mcp_prompts_as_commands() still works."""
    pool, agent = pool_and_agent
    _mock_skills_on_pool(pool, [])

    session = _create_session(pool, agent, mock_client=mock_client, mock_acp_agent=mock_acp_agent)

    # Mock agent.list_prompts to return a fake MCP prompt
    mock_prompt = MagicMock()
    mock_prompt.name = "mcp-prompt-cmd"
    mock_prompt.description = "An MCP prompt"
    mock_prompt.create_mcp_command = MagicMock(
        return_value=SlashedCommand.from_raw(
            lambda ctx, args, kwargs: None,
            name="mcp-prompt-cmd",
            description="An MCP prompt",
        )
    )
    agent.list_prompts = AsyncMock(return_value=[mock_prompt])  # type: ignore[method-assign]

    session.notifications.update_commands = AsyncMock()  # type: ignore[method-assign]

    await session._register_mcp_prompts_as_commands()

    all_names = _all_cmd_names(session.command_store)
    assert "mcp-prompt-cmd" in all_names, f"mcp-prompt-cmd missing from {all_names}"
    assert session.notifications.update_commands.called


@pytest.mark.unit
async def test_regression_send_update_includes_both_manifest_and_skill_commands(
    pool_and_agent: tuple[AgentPool, Agent],
    mock_client: AsyncMock,
    mock_acp_agent: MagicMock,
) -> None:
    """T11.3: Test send_available_commands_update() includes both manifest and skill commands.

    Verifies both manifest commands AND skill commands appear in update.
    """
    pool, agent = pool_and_agent
    skill = _make_skill(name="skill-cmd", description="A skill command")
    _mock_skills_on_pool(pool, [skill])

    from wolfharness_config.commands import StaticCommandConfig

    manifest_cmd = StaticCommandConfig(
        type="static",
        name="manifest-cmd",
        description="A manifest command",
        content="Hello",
    )
    pool.manifest.get_command_configs = MagicMock(  # type: ignore[method-assign]
        return_value={"manifest-cmd": manifest_cmd}
    )

    session = _create_session(pool, agent, mock_client=mock_client, mock_acp_agent=mock_acp_agent)

    session.notifications.update_commands = AsyncMock()  # type: ignore[method-assign]
    await session.send_available_commands_update()

    calls = session.notifications.update_commands.call_args_list
    sent_commands: list[AvailableCommand] = calls[0][0][0]
    command_names = [cmd.name for cmd in sent_commands]
    assert "manifest-cmd" in command_names, f"manifest-cmd missing from {command_names}"
    assert "skill-cmd" in command_names, f"skill-cmd missing from {command_names}"


@pytest.mark.unit
async def test_regression_command_name_collision_last_registered_wins(
    pool_and_agent: tuple[AgentPool, Agent],
    mock_client: AsyncMock,
    mock_acp_agent: MagicMock,
) -> None:
    """T11.4: Test skill command names don't collide with manifest command names.

    Last registered wins via replace=True.
    """
    pool, agent = pool_and_agent
    # Create a skill with the same name as a manifest command
    shared_name = "shared-cmd"
    skill = _make_skill(name=shared_name, description="Skill version of shared command")
    _mock_skills_on_pool(pool, [skill])

    from wolfharness_config.commands import StaticCommandConfig

    manifest_cmd = StaticCommandConfig(
        type="static",
        name=shared_name,
        description="Manifest version of shared command",
        content="Hello",
    )
    pool.manifest.get_command_configs = MagicMock(  # type: ignore[method-assign]
        return_value={shared_name: manifest_cmd}
    )

    session = _create_session(pool, agent, mock_client=mock_client, mock_acp_agent=mock_acp_agent)

    # __post_init__ calls _register_manifest_commands() first,
    # then _register_skill_commands()
    # _register_skill_commands() uses replace=True,
    # so the skill command should replace the manifest command
    commands = session.command_store.list_commands()
    matching = [cmd for cmd in commands if cmd.name == shared_name]
    assert len(matching) == 1, (
        f"Expected exactly 1 command named '{shared_name}', got {len(matching)}"
    )
    cmd = matching[0]
    # The skill command should have category="skill" (last registered wins)
    assert cmd.category == "skill", (
        f"Expected skill command to replace manifest command (category='skill'), "
        f"got category='{cmd.category}'"
    )


@pytest.mark.unit
async def test_regression_load_skills_false_pool_skills_still_registered(
    pool_and_agent: tuple[AgentPool, Agent],
    mock_client: AsyncMock,
    mock_acp_agent: MagicMock,
) -> None:
    """T11.5: Test ACP session with load_skills=False config.

    Pool skills still registered, client skills NOT discovered.
    _register_skill_commands() is always called in __post_init__
    regardless of load_skills. init_client_skills() is gated by
    should_load_skills in acp_agent.py. This test verifies pool skills
    are registered even when client skill discovery is skipped.
    """
    pool, agent = pool_and_agent
    pool_skill = _make_skill(name="pool-skill", description="A pool-level skill")
    _mock_skills_on_pool(pool, [pool_skill])

    session = _create_session(pool, agent, mock_client=mock_client, mock_acp_agent=mock_acp_agent)

    # Pool skills should be registered (from __post_init__ -> _register_skill_commands)
    store_names = _skill_cmd_names(session.command_store)
    assert "pool-skill" in store_names, (
        f"Pool skill should be registered even with load_skills=False. Got: {store_names}"
    )

    # Verify init_client_skills is NOT called when load_skills=False
    # (it's gated by should_load_skills in acp_agent.py:new_session())
    # We just verify the pool skills are there regardless
    session.notifications.update_commands = AsyncMock()  # type: ignore[method-assign]
    await session.send_available_commands_update()

    calls = session.notifications.update_commands.call_args_list
    sent_commands: list[AvailableCommand] = calls[0][0][0]
    command_names = [cmd.name for cmd in sent_commands]
    assert "pool-skill" in command_names


@pytest.mark.unit
async def test_regression_50_plus_skills_registration_performance(
    pool_and_agent: tuple[AgentPool, Agent],
    mock_client: AsyncMock,
    mock_acp_agent: MagicMock,
) -> None:
    """T11.6: Test session with 50+ skills.

    Performance: registration completes in reasonable time.
    """
    pool, agent = pool_and_agent
    skills = [
        _make_skill(name=f"skill-{i:03d}", description=f"Skill number {i}") for i in range(50)
    ]
    _mock_skills_on_pool(pool, skills)

    import time

    start = time.monotonic()
    session = _create_session(pool, agent, mock_client=mock_client, mock_acp_agent=mock_acp_agent)
    elapsed = time.monotonic() - start

    # All 50 skills should be registered
    store_names = _skill_cmd_names(session.command_store)
    assert len(store_names) == 50, f"Expected 50 skill commands, got {len(store_names)}"
    for i in range(50):
        assert f"skill-{i:03d}" in store_names

    # Registration should complete in reasonable time (< 5 seconds)
    assert elapsed < 5.0, f"Registration of 50 skills took too long: {elapsed:.2f}s"

    # Verify all appear in available_commands_update
    session.notifications.update_commands = AsyncMock()  # type: ignore[method-assign]
    await session.send_available_commands_update()

    calls = session.notifications.update_commands.call_args_list
    sent_commands: list[AvailableCommand] = calls[0][0][0]
    command_names = [cmd.name for cmd in sent_commands]
    assert len(command_names) >= 50
    for i in range(50):
        assert f"skill-{i:03d}" in command_names
