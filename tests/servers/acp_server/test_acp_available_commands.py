"""Tests for the available commands update flow in ACP server.

These verify that available commands are sent at the right lifecycle points
and that local + remote commands are properly merged.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from acp.schema import AvailableCommand
from wolfharness import Agent, AgentPool
from wolfharness_server.acp_server.acp_agent import AgentPoolACPAgent
from wolfharness_server.acp_server.session import ACPSession


def _make_pool_and_agent() -> tuple[AgentPool, Agent]:
    """Create a simple pool with one agent."""
    from wolfharness.models.agents import NativeAgentConfig
    from wolfharness.models.manifest import AgentsManifest

    manifest = AgentsManifest(agents={"test_agent": NativeAgentConfig(model="test")})

    pool = AgentPool(manifest)

    def simple_callback(message: str) -> str:
        return f"Test response: {message}"

    agent = Agent.from_callback(name="test_agent", callback=simple_callback, agent_pool=pool)
    # pool.register() removed; agent created from callback/config above
    return pool, agent


def _make_mock_agent() -> MagicMock:
    """Create a mock agent suitable for AgentPoolACPAgent lifecycle tests."""
    agent = MagicMock()
    agent.name = "test_agent"
    agent.model_name = "test-model"
    agent.get_available_models = AsyncMock(return_value=[])
    agent.get_modes = AsyncMock(return_value=[])
    agent.config_options = []
    agent.staged_content = MagicMock()
    agent.tools = MagicMock()
    agent.tools.list_prompts = AsyncMock(return_value=[])
    agent.conversation = MagicMock()
    agent.conversation.chat_messages = []
    agent.load_rules = AsyncMock()
    agent.load_session = AsyncMock(return_value=True)
    agent._state = None
    return agent


@pytest.fixture
def pool_and_agent() -> tuple[AgentPool, Agent]:
    """Provide a pool and agent."""
    return _make_pool_and_agent()


@pytest.fixture
def mock_client() -> AsyncMock:
    """Provide a mock ACP client."""
    return AsyncMock()


@pytest.fixture
def mock_acp_agent_for_session(
    mock_client: AsyncMock, pool_and_agent: tuple[AgentPool, Agent]
) -> MagicMock:
    """Provide a mock ACP agent suitable for ACPSession construction."""
    _, _agent = pool_and_agent
    mock_acp_agent = MagicMock()
    mock_acp_agent._task_group = MagicMock()
    mock_acp_agent._task_group.start_soon = MagicMock()
    return mock_acp_agent


# =============================================================================
# ACPSession-level tests
# =============================================================================


@pytest.mark.unit
async def test_session_initialize_does_not_call_send_available_commands_update(
    pool_and_agent: tuple[AgentPool, Agent],
    mock_client: AsyncMock,
    mock_acp_agent_for_session: MagicMock,
) -> None:
    """ACPSession.initialize() should NOT call send_available_commands_update().

    The available_commands_update is now sent by the ACP lifecycle handlers
    (new_session, load_session, resume_session) via create_task after the
    response returns. Sending it during initialize() would cause a duplicate
    notification since initialize() runs during session construction inside
    create_session()/resume_session(), and the handlers also schedule it.
    """
    _, agent = pool_and_agent
    session = ACPSession(
        session_id="test-session",
        agent=agent,
        cwd="/tmp",
        client=mock_client,
        acp_agent=mock_acp_agent_for_session,
    )

    # Mock the env aenter since we don't want real filesystem setup
    session.acp_env.__aenter__ = AsyncMock(return_value=None)  # type: ignore[method-assign]
    session.send_available_commands_update = AsyncMock()  # type: ignore[method-assign]

    await session.initialize()

    session.send_available_commands_update.assert_not_awaited()  # type: ignore[attr-defined]


@pytest.mark.unit
async def test_session_send_available_commands_update_sends_correct_notification(
    pool_and_agent: tuple[AgentPool, Agent],
    mock_client: AsyncMock,
    mock_acp_agent_for_session: MagicMock,
) -> None:
    """send_available_commands_update() should send AvailableCommandsUpdate via notifications."""
    _, agent = pool_and_agent
    session = ACPSession(
        session_id="test-session",
        agent=agent,
        cwd="/tmp",
        client=mock_client,
        acp_agent=mock_acp_agent_for_session,
    )

    session.notifications.update_commands = AsyncMock()  # type: ignore[method-assign]

    await session.send_available_commands_update()

    assert session.notifications.update_commands.called
    calls = session.notifications.update_commands.call_args_list
    sent_commands = calls[0][0][0]
    assert isinstance(sent_commands, list)
    # Should contain at least the commands from get_acp_commands()
    assert all(isinstance(cmd, AvailableCommand) for cmd in sent_commands)


@pytest.mark.unit
async def test_session_merges_local_and_remote_commands(
    pool_and_agent: tuple[AgentPool, Agent],
    mock_client: AsyncMock,
    mock_acp_agent_for_session: MagicMock,
) -> None:
    """send_available_commands_update() should merge local and remote commands."""
    _, agent = pool_and_agent
    session = ACPSession(
        session_id="test-session",
        agent=agent,
        cwd="/tmp",
        client=mock_client,
        acp_agent=mock_acp_agent_for_session,
    )

    # Register a local command
    from slashed import Command

    async def dummy_run(ctx: object, args: list[str], kwargs: dict[str, str]) -> None:
        pass

    local_cmd = Command.from_raw(dummy_run, name="local-cmd", description="Local command")
    session.command_store.register_command(local_cmd)

    # Set remote commands
    remote_cmd = AvailableCommand.create(name="remote-cmd", description="Remote command")
    session._remote_commands = [remote_cmd]

    session.notifications.update_commands = AsyncMock()  # type: ignore[method-assign]

    await session.send_available_commands_update()

    calls = session.notifications.update_commands.call_args_list
    sent_commands: list[AvailableCommand] = calls[0][0][0]
    command_names = [cmd.name for cmd in sent_commands]

    assert "local-cmd" in command_names, f"Local command missing from {command_names}"
    assert "remote-cmd" in command_names, f"Remote command missing from {command_names}"


# =============================================================================
# AgentPoolACPAgent lifecycle tests
# =============================================================================


def _create_initialized_acp_agent(
    pool: AgentPool, agent: Agent, mock_client: AsyncMock
) -> AgentPoolACPAgent:
    """Create and initialize an AgentPoolACPAgent."""
    acp_agent = AgentPoolACPAgent(client=mock_client, default_agent=agent)
    # Mark as initialized to bypass initialize() requirement
    acp_agent._initialized = True
    # Clear protocol handler to avoid event_bus.subscribe issues in tests
    # (the mock agent_pool doesn't have a real EventBus)
    acp_agent._protocol_handler = None
    return acp_agent


def _setup_mock_session(acp_agent: AgentPoolACPAgent, session_id: str) -> MagicMock:
    """Set up a mock session in the session manager and return it."""
    mock_session = MagicMock()
    mock_session.session_id = session_id
    mock_session.send_available_commands_update = AsyncMock()
    mock_session.agent.load_rules = AsyncMock()  # type: ignore[method-assign]
    mock_session._register_prompt_hub_commands = AsyncMock()
    mock_session.agent.load_session = AsyncMock(return_value=True)  # type: ignore[method-assign]
    mock_session.agent.conversation.chat_messages = []
    # Configure mock agent to return valid values for ACP helpers
    mock_session.agent.model_name = "test-model"
    mock_session.agent.get_modes = AsyncMock(return_value=[])
    mock_session.agent.get_available_models = AsyncMock(return_value=None)

    # Inject into session manager
    acp_agent.session_manager._acp_sessions[session_id] = mock_session
    # Replace task group with a mock to capture start_soon calls
    acp_agent._task_group = MagicMock()
    acp_agent._task_group.start_soon = MagicMock()
    return mock_session


async def _run_scheduled_and_verify(acp_agent: object, mock_session: MagicMock) -> bool:
    """Run scheduled tasks and verify send_available_commands_update was called."""
    import inspect

    task_group: Any = acp_agent._task_group  # type: ignore[attr-defined]
    scheduled: list[Any] = []
    for call in task_group.start_soon.call_args_list:
        fn = call.args[0] if call.args else None
        args = call.args[1:]
        if fn is None or not callable(fn):
            continue
        result = fn(*args)
        if inspect.isawaitable(result):
            scheduled.append(result)

    # Execute all scheduled awaitables
    if scheduled:
        await asyncio.gather(*scheduled, return_exceptions=True)
    return mock_session.send_available_commands_update.called


@pytest.mark.unit
async def test_new_session_schedules_send_available_commands_update(
    pool_and_agent: tuple[AgentPool, Agent],
    mock_client: AsyncMock,
) -> None:
    """AgentPoolACPAgent.new_session() should schedule send_available_commands_update()."""
    pool, _ = pool_and_agent
    agent = _make_mock_agent()
    acp_agent = _create_initialized_acp_agent(pool, agent, mock_client)
    session_id = "sess-new-001"
    mock_session = _setup_mock_session(acp_agent, session_id)

    # Mock session manager create_session to return our session ID
    acp_agent.session_manager.create_session = AsyncMock(return_value=session_id)  # type: ignore[method-assign]

    from acp.schema import NewSessionRequest

    request = NewSessionRequest(cwd="/tmp")
    await acp_agent.new_session(request)

    assert await _run_scheduled_and_verify(acp_agent, mock_session), (
        "new_session should schedule send_available_commands_update"
    )


@pytest.mark.unit
async def test_load_session_schedules_send_available_commands_update(
    pool_and_agent: tuple[AgentPool, Agent],
    mock_client: AsyncMock,
) -> None:
    """AgentPoolACPAgent.load_session() should schedule send_available_commands_update()."""
    pool, _ = pool_and_agent
    agent = _make_mock_agent()
    acp_agent = _create_initialized_acp_agent(pool, agent, mock_client)
    session_id = "sess-load-001"
    mock_session = _setup_mock_session(acp_agent, session_id)

    acp_agent.session_manager.create_session = AsyncMock(return_value=session_id)  # type: ignore[method-assign]

    from acp.schema import LoadSessionRequest

    request = LoadSessionRequest(session_id=session_id, cwd="/tmp")
    await acp_agent.load_session(request)

    assert await _run_scheduled_and_verify(acp_agent, mock_session), (
        "load_session should schedule send_available_commands_update"
    )


@pytest.mark.unit
async def test_resume_session_schedules_send_available_commands_update(
    pool_and_agent: tuple[AgentPool, Agent],
    mock_client: AsyncMock,
) -> None:
    """AgentPoolACPAgent.resume_session() should schedule send_available_commands_update()."""
    pool, _ = pool_and_agent
    agent = _make_mock_agent()
    acp_agent = _create_initialized_acp_agent(pool, agent, mock_client)
    session_id = "sess-resume-001"
    mock_session = _setup_mock_session(acp_agent, session_id)

    acp_agent.session_manager.create_session = AsyncMock(return_value=session_id)  # type: ignore[method-assign]

    from acp.schema import ResumeSessionRequest

    request = ResumeSessionRequest(session_id=session_id, cwd="/tmp")
    await acp_agent.resume_session(request)

    assert await _run_scheduled_and_verify(acp_agent, mock_session), (
        "resume_session should schedule send_available_commands_update"
    )


@pytest.mark.unit
async def test_new_session_does_not_call_directly_only_schedules(
    pool_and_agent: tuple[AgentPool, Agent],
    mock_client: AsyncMock,
) -> None:
    """new_session should schedule the update via tasks, not call it directly (async safety)."""
    pool, _ = pool_and_agent
    agent = _make_mock_agent()
    acp_agent = _create_initialized_acp_agent(pool, agent, mock_client)
    session_id = "sess-new-002"
    mock_session = _setup_mock_session(acp_agent, session_id)
    acp_agent.session_manager.create_session = AsyncMock(return_value=session_id)  # type: ignore[method-assign]

    from acp.schema import NewSessionRequest

    request = NewSessionRequest(cwd="/tmp")
    await acp_agent.new_session(request)

    # It should NOT have been called directly on the session before scheduling
    mock_session.send_available_commands_update.assert_not_awaited()
    # Instead it should be scheduled via tasks.create_task and run successfully
    assert await _run_scheduled_and_verify(acp_agent, mock_session)


@pytest.mark.unit
async def test_load_session_does_not_call_directly_only_schedules(
    pool_and_agent: tuple[AgentPool, Agent],
    mock_client: AsyncMock,
) -> None:
    """load_session should schedule the update via tasks, not call it directly."""
    pool, _ = pool_and_agent
    agent = _make_mock_agent()
    acp_agent = _create_initialized_acp_agent(pool, agent, mock_client)
    session_id = "sess-load-002"
    mock_session = _setup_mock_session(acp_agent, session_id)
    acp_agent.session_manager.create_session = AsyncMock(return_value=session_id)  # type: ignore[method-assign]

    from acp.schema import LoadSessionRequest

    request = LoadSessionRequest(session_id=session_id, cwd="/tmp")
    await acp_agent.load_session(request)

    mock_session.send_available_commands_update.assert_not_awaited()
    assert await _run_scheduled_and_verify(acp_agent, mock_session)


@pytest.mark.unit
async def test_resume_session_does_not_call_directly_only_schedules(
    pool_and_agent: tuple[AgentPool, Agent],
    mock_client: AsyncMock,
) -> None:
    """resume_session should schedule the update via tasks, not call it directly."""
    pool, _ = pool_and_agent
    agent = _make_mock_agent()
    acp_agent = _create_initialized_acp_agent(pool, agent, mock_client)
    session_id = "sess-resume-002"
    mock_session = _setup_mock_session(acp_agent, session_id)
    acp_agent.session_manager.create_session = AsyncMock(return_value=session_id)  # type: ignore[method-assign]

    from acp.schema import ResumeSessionRequest

    request = ResumeSessionRequest(session_id=session_id, cwd="/tmp")
    await acp_agent.resume_session(request)

    mock_session.send_available_commands_update.assert_not_awaited()
    assert await _run_scheduled_and_verify(acp_agent, mock_session)


@pytest.mark.unit
async def test_merged_commands_contain_both_sources(
    pool_and_agent: tuple[AgentPool, Agent],
    mock_client: AsyncMock,
) -> None:
    """Verify merged command list contains both local and remote commands in correct format."""
    _, agent = pool_and_agent
    session = ACPSession(
        session_id="test-session",
        agent=agent,
        cwd="/tmp",
        client=mock_client,
        acp_agent=MagicMock(),
    )

    from slashed import Command

    async def local_run(ctx: object, args: list[str], kwargs: dict[str, str]) -> None:
        pass

    local_cmd = Command.from_raw(local_run, name="local", description="Local cmd", usage="[arg]")
    session.command_store.register_command(local_cmd)

    remote_cmd = AvailableCommand.create(
        name="remote", description="Remote cmd", input_hint="[arg]"
    )
    session._remote_commands = [remote_cmd]

    session.notifications.update_commands = AsyncMock()  # type: ignore[method-assign]

    await session.send_available_commands_update()

    calls = session.notifications.update_commands.call_args_list
    assert len(calls) == 1
    sent_commands: list[AvailableCommand] = calls[0][0][0]

    local_names = [c.name for c in sent_commands if c.name == "local"]
    remote_names = [c.name for c in sent_commands if c.name == "remote"]

    assert local_names == ["local"], "Local command should be in merged list"
    assert remote_names == ["remote"], "Remote command should be in merged list"
    assert len(sent_commands) >= 2, "Merged list should have at least local + remote"
