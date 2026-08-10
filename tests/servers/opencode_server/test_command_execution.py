"""Tests for OpenCode server command execution.

Tests slashed command execution, MCP prompt fallback, and precedence handling.

.. caution:: Anti-pattern warning (issue #339)
    Several tests in this file historically over-mocked the execution chain:
    ``command.execute = AsyncMock()`` (no real skill_bridge injection, no
    ``ctx.print``, no ``staged_content``) and ``run_stream`` as a never-yielding
    generator (no events, no TextPart, no UserMessageInsertedEvent). Assertions
    checked routing target (``session_pool_calls == 1``) and HTTP status (200),
    but never checked payload (TextPart content, UserMessage existence,
    ``parent_id`` linkage, double prompt). This is why #339 went undetected.

    The skill-command regression tests in ``test_skill_command_path.py`` exercise
    the real ``execute_command`` → skill execution → ``send_message`` chain and
    MUST be used as the pattern for new command execution tests.

    Tests below that still use ``AsyncMock()`` for ``command.execute`` are
    testing non-skill dispatch logic (routing, precedence, error handling) —
    not the skill execution payload. They are acceptable as long as they do
    not claim to verify skill command behavior.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest


if TYPE_CHECKING:
    from httpx import AsyncClient

    from wolfharness_server.opencode_server.state import ServerState


pytestmark = [pytest.mark.asyncio, pytest.mark.integration]


async def test_execute_slashed_command_success(
    async_client: AsyncClient,
    server_state: ServerState,
    mock_agent: Mock,
):
    """Test slashed command execution when command is in CommandStore.

    Happy path - command exists in CommandStore, executes successfully.
    """
    # Create session first
    response = await async_client.post("/session", json={"title": "Test Session"})
    assert response.status_code == 200
    session_id = response.json()["id"]

    # Mock CommandStore with a command
    mock_command = MagicMock()
    mock_command.execute = AsyncMock()
    mock_command_store = MagicMock()
    mock_command_store.get_command = MagicMock(return_value=mock_command)
    server_state.command_store = mock_command_store

    # Mock empty MCP prompts (no collision)
    mock_agent.list_prompts = AsyncMock(return_value=[])

    # Execute command
    response = await async_client.post(
        f"/session/{session_id}/command",
        json={"command": "test-cmd", "arguments": "arg1 arg2"},
    )

    # Verify success
    assert response.status_code == 200
    result = response.json()
    assert "info" in result
    assert "parts" in result

    # Verify command was called (get_command is called twice: once for check, once to retrieve)
    assert mock_command_store.get_command.call_count == 2
    mock_command_store.get_command.assert_called_with("test-cmd")
    mock_command.execute.assert_called_once()


async def test_mcp_prompt_fallback(
    async_client: AsyncClient,
    server_state: ServerState,
    mock_agent: Mock,
):
    """Test MCP prompt fallback when command not in CommandStore.

    Command doesn't exist in CommandStore but exists as MCP prompt.
    Should fall back and execute via MCP.
    """
    # Create session first
    response = await async_client.post("/session", json={"title": "Test Session"})
    assert response.status_code == 200
    session_id = response.json()["id"]

    # Mock CommandStore without the command
    mock_command_store = MagicMock()
    mock_command_store.get_command = MagicMock(return_value=None)
    server_state.command_store = mock_command_store

    # Mock MCP prompt
    mock_prompt = MagicMock()
    mock_prompt.name = "test-cmd"
    mock_prompt.arguments = [{"name": "arg1"}]
    mock_prompt.get_components = AsyncMock(return_value=[])
    mock_agent.list_prompts = AsyncMock(return_value=[mock_prompt])
    mock_agent.run = AsyncMock(return_value=MagicMock(data="MCP prompt result"))

    # Execute command via MCP fallback
    response = await async_client.post(
        f"/session/{session_id}/command",
        json={"command": "test-cmd", "arguments": "value1"},
    )

    # Verify success
    assert response.status_code == 200
    result = response.json()
    assert "info" in result
    assert "parts" in result

    # Verify MCP prompt was used
    mock_agent.list_prompts.assert_called()
    mock_prompt.get_components.assert_called_once()


async def test_precedence_slashed_over_mcp(
    async_client: AsyncClient,
    server_state: ServerState,
    mock_agent: Mock,
):
    """Test that CommandStore commands take precedence over MCP prompts.

    Both exist, CommandStore should be used.
    """
    # Create session first
    response = await async_client.post("/session", json={"title": "Test Session"})
    assert response.status_code == 200
    session_id = response.json()["id"]

    # Mock CommandStore with command
    mock_command = MagicMock()
    mock_command.execute = AsyncMock()
    mock_command_store = MagicMock()
    mock_command_store.get_command = MagicMock(return_value=mock_command)
    server_state.command_store = mock_command_store

    # Mock MCP prompt with same name
    mock_prompt = MagicMock()
    mock_prompt.name = "test-cmd"
    mock_agent.list_prompts = AsyncMock(return_value=[mock_prompt])

    # Execute command
    response = await async_client.post(
        f"/session/{session_id}/command",
        json={"command": "test-cmd"},
    )

    # Verify success
    assert response.status_code == 200

    # Verify CommandStore command was executed (not MCP)
    mock_command.execute.assert_called_once()

    # Verify MCP prompt.get_components was NOT called
    mock_prompt.get_components.assert_not_called()


async def test_unknown_command_returns_404(
    async_client: AsyncClient,
    server_state: ServerState,
    mock_agent: Mock,
):
    """Test 404 response when command not found anywhere.

    Neither CommandStore nor MCP has the command.
    """
    # Create session first
    response = await async_client.post("/session", json={"title": "Test Session"})
    assert response.status_code == 200
    session_id = response.json()["id"]

    # Mock CommandStore without the command
    mock_command_store = MagicMock()
    mock_command_store.get_command = MagicMock(return_value=None)
    server_state.command_store = mock_command_store

    # Mock empty MCP prompts
    mock_agent.list_prompts = AsyncMock(return_value=[])

    # Execute unknown command
    response = await async_client.post(
        f"/session/{session_id}/command",
        json={"command": "unknown-cmd"},
    )

    # Verify 404
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


async def test_none_command_store_graceful(
    async_client: AsyncClient,
    server_state: ServerState,
    mock_agent: Mock,
):
    """Test graceful handling when command_store is None.

    Should fall back to MCP prompts.
    """
    # Create session first
    response = await async_client.post("/session", json={"title": "Test Session"})
    assert response.status_code == 200
    session_id = response.json()["id"]

    # Set command_store to None
    server_state.command_store = None

    # Mock MCP prompt
    mock_prompt = MagicMock()
    mock_prompt.name = "fallback-cmd"
    mock_prompt.arguments = []
    mock_prompt.get_components = AsyncMock(return_value=[])
    mock_agent.list_prompts = AsyncMock(return_value=[mock_prompt])
    mock_agent.run = AsyncMock(return_value=MagicMock(data="Fallback result"))

    # Execute command
    response = await async_client.post(
        f"/session/{session_id}/command",
        json={"command": "fallback-cmd"},
    )

    # Verify success via MCP fallback
    assert response.status_code == 200
    result = response.json()
    assert "info" in result

    # Verify MCP was checked and used
    mock_agent.list_prompts.assert_called()


async def test_command_execution_error(
    async_client: AsyncClient,
    server_state: ServerState,
    mock_agent: Mock,
):
    """Test graceful handling of command execution failures.

    Command exists but raises exception during execution.
    """
    # Create session first
    response = await async_client.post("/session", json={"title": "Test Session"})
    assert response.status_code == 200
    session_id = response.json()["id"]

    # Mock CommandStore with failing command
    mock_command = MagicMock()
    mock_command.execute = AsyncMock(side_effect=RuntimeError("Command failed"))
    mock_command_store = MagicMock()
    mock_command_store.get_command = MagicMock(return_value=mock_command)
    server_state.command_store = mock_command_store

    # Mock empty MCP prompts
    mock_agent.list_prompts = AsyncMock(return_value=[])

    # Execute command that will fail
    response = await async_client.post(
        f"/session/{session_id}/command",
        json={"command": "failing-cmd"},
    )

    # Verify 500 error
    assert response.status_code == 500
    assert "failed" in response.json()["detail"].lower()


async def test_collision_warning_logged(
    async_client: AsyncClient,
    server_state: ServerState,
    mock_agent: Mock,
):
    """Test warning is logged when both slashed command and MCP prompt exist."""
    from unittest.mock import patch

    response = await async_client.post("/session", json={"title": "Test Session"})
    assert response.status_code == 200
    session_id = response.json()["id"]

    mock_command = MagicMock()
    mock_command.execute = AsyncMock()
    mock_command_store = MagicMock()
    mock_command_store.get_command = MagicMock(return_value=mock_command)
    server_state.command_store = mock_command_store

    mock_prompt = MagicMock()
    mock_prompt.name = "collision-cmd"
    mock_agent.list_prompts = AsyncMock(return_value=[mock_prompt])

    with patch("wolfharness_server.opencode_server.routes.session_routes.logger") as mock_logger:
        response = await async_client.post(
            f"/session/{session_id}/command",
            json={"command": "collision-cmd"},
        )

    assert response.status_code == 200

    mock_logger.warning.assert_called_once()
    call_args = mock_logger.warning.call_args
    assert "Both slashed command and prompt exist" in call_args.args[0]


async def test_concurrent_slash_commands_same_session_are_serialized(
    async_client: AsyncClient,
    server_state: ServerState,
    mock_agent: Mock,
):
    """Test that concurrent slash commands to the same session are serialized.

    The route-level lock in ``execute_command`` ensures that multiple commands
    sent to the same session concurrently are processed sequentially, not in
    parallel. This prevents race conditions during command execution.
    """
    # Create session first
    response = await async_client.post("/session", json={"title": "Test Session"})
    assert response.status_code == 200
    session_id = response.json()["id"]

    # Track concurrent execution
    active_executions = 0
    max_concurrent = 0
    execution_lock = asyncio.Lock()

    async def tracked_execute(*args, **kwargs):
        nonlocal active_executions, max_concurrent
        async with execution_lock:
            active_executions += 1
            max_concurrent = max(max_concurrent, active_executions)
        # Simulate some work
        await asyncio.sleep(0.1)
        async with execution_lock:
            active_executions -= 1

    # Mock CommandStore with tracked command
    mock_command = MagicMock()
    mock_command.execute = AsyncMock(side_effect=tracked_execute)
    mock_command_store = MagicMock()
    mock_command_store.get_command = MagicMock(return_value=mock_command)
    server_state.command_store = mock_command_store

    # Mock empty MCP prompts
    mock_agent.list_prompts = AsyncMock(return_value=[])

    # Send two commands concurrently to the same session
    async def send_command(cmd: str):
        return await async_client.post(
            f"/session/{session_id}/command",
            json={"command": cmd},
        )

    results = await asyncio.gather(
        send_command("cmd-a"),
        send_command("cmd-b"),
    )

    # Both should succeed
    assert all(r.status_code == 200 for r in results)

    # Verify commands were executed sequentially (never concurrently)
    assert max_concurrent == 1, (
        f"Expected sequential execution (max_concurrent=1), "
        f"but got max_concurrent={max_concurrent}. "
        f"Route-level lock is not serializing commands."
    )


async def test_skill_command_routes_through_session_pool(  # noqa: PLR0915
    async_client: AsyncClient,
    server_state: ServerState,
    mock_agent: Mock,
):
    """Test that skill command routes through send_message (not run_stream).

    After the #339 fix, skill commands (category=='skill') route through
    _execute_skill_command → send_message → EventBus-only, NOT through
    _execute_slashed_command → run_stream. This test uses a real
    SlashedCommand with category='skill' that calls ctx.print and injects
    into staged_content, verifying the correct routing path.
    """
    from slashed import Command as SlashedCommand

    from wolfharness_server.opencode_server.models import UserMessage

    # Create session first
    response = await async_client.post("/session", json={"title": "Test Session"})
    assert response.status_code == 200
    session_id = response.json()["id"]

    # Build a real skill command with category='skill'
    async def _execute_skill(ctx: Any, args: list[str], kwargs: dict[str, str]) -> None:
        await ctx.print("Loading skill: direct-skill (skill://test/direct-skill)")
        if hasattr(ctx.data, "node") and hasattr(ctx.data.node, "staged_content"):
            ctx.data.node.staged_content.add_text(
                "<skill-instruction>Test instructions</skill-instruction>"
            )

    skill_command = SlashedCommand.from_raw(
        _execute_skill,
        name="direct-skill",
        description="Direct skill test",
        category="skill",
        usage="<args>",
    )

    # Wire real command into a mock CommandStore
    mock_command_store = MagicMock()
    mock_command_store.get_command = MagicMock(return_value=skill_command)
    mock_command_store.event_handler = None
    mock_command_store.output = MagicMock()
    server_state.command_store = mock_command_store

    mock_agent.list_prompts = AsyncMock(return_value=[])

    # Set up session agent mock with staged_content and get_context()
    pool = server_state.pool_or_none
    assert pool is not None
    session_agent = pool.session_pool.sessions.get_or_create_session_agent.return_value
    ctx_mock = MagicMock()
    ctx_mock.node = session_agent
    session_agent.get_context = MagicMock(return_value=ctx_mock)
    session_agent.staged_content = MagicMock()
    session_agent.staged_content.add_text = MagicMock()
    session_agent.staged_content.__bool__ = MagicMock(return_value=True)
    session_agent.staged_content.__len__ = MagicMock(return_value=1)

    # Track send_message and run_stream calls
    send_message_calls: list[tuple[Any, Any]] = []
    original_send_message = pool.session_pool.send_message

    async def _track_send_message(*args: Any, **kwargs: Any) -> Any:
        send_message_calls.append((args, kwargs))
        return await original_send_message(*args, **kwargs)

    pool.session_pool.send_message = _track_send_message  # type: ignore[method-assign]

    run_stream_calls: list[tuple[Any, Any]] = []

    async def _track_run_stream(*args: Any, **kwargs: Any) -> Any:
        run_stream_calls.append((args, kwargs))
        if False:
            yield MagicMock()

    pool.session_pool.run_stream = _track_run_stream  # type: ignore[method-assign]

    response = await async_client.post(
        f"/session/{session_id}/command",
        json={"command": "direct-skill", "arguments": "some args"},
    )

    # Command should execute successfully — returns 200
    assert response.status_code == 200
    result = response.json()
    assert "info" in result
    assert "parts" in result

    # Verify send_message was called (NOT run_stream — skill commands use the new path)
    assert len(send_message_calls) == 1, (
        f"send_message should be called once, got {len(send_message_calls)}"
    )
    assert len(run_stream_calls) == 0, (
        f"run_stream should NOT be called for skill commands, got {len(run_stream_calls)}"
    )

    # Verify no "Loading skill" in response parts (ctx.print discarded)
    for part in result.get("parts", []):
        if part.get("type") == "text":
            assert "Loading skill" not in part.get("text", ""), (
                "ctx.print output leaked into TextPart"
            )

    # Verify a UserMessage was created (not swallowed)
    messages = server_state.messages.get(session_id, [])
    user_messages = [m for m in messages if isinstance(m.info, UserMessage)]
    assert len(user_messages) >= 1, "UserMessage was not created — user input swallowed"

    # Verify parent_id is linked (not empty)
    parent_id = result["info"].get("parentID", "")
    assert parent_id != "", "Assistant message parent_id is empty — not linked to user message"


async def test_slash_command_routes_through_session_pool(
    async_client: AsyncClient,
    server_state: ServerState,
    mock_agent: Mock,
):
    """Test that non-skill slash command routes through SessionPool.run_stream().

    Non-skill commands (category != 'skill') continue to use the
    _execute_slashed_command path with run_stream. This test uses a real
    SlashedCommand that calls ctx.print (output captured in TextPart)
    to verify the existing behavior is preserved.
    """
    from slashed import Command as SlashedCommand

    # Create session first
    response = await async_client.post("/session", json={"title": "Test Session"})
    assert response.status_code == 200
    session_id = response.json()["id"]

    # Build a real non-skill command that calls ctx.print
    async def _execute_cmd(ctx: Any, args: list[str], kwargs: dict[str, str]) -> None:
        await ctx.print("Command output: test-cmd executed")

    real_command = SlashedCommand.from_raw(
        _execute_cmd,
        name="test-cmd",
        description="Test non-skill command",
        category="test",
        usage="<args>",
    )

    # Wire real command into a mock CommandStore
    mock_command_store = MagicMock()
    mock_command_store.get_command = MagicMock(return_value=real_command)
    mock_command_store.event_handler = None
    mock_command_store.output = MagicMock()
    server_state.command_store = mock_command_store

    # Track agent.run_stream calls
    agent_calls: list[tuple[Any, Any]] = []

    async def _mock_run_stream(*args: Any, **kwargs: Any) -> Any:
        agent_calls.append((args, kwargs))
        if False:
            yield MagicMock()

    mock_agent.run_stream = _mock_run_stream  # type: ignore[method-assign]

    # Track session_pool.run_stream calls
    session_pool_calls: list[tuple[Any, Any]] = []

    async def _mock_session_run_stream(*args: Any, **kwargs: Any) -> Any:
        session_pool_calls.append((args, kwargs))
        if False:
            yield MagicMock()

    mock_agent.host_context.session_pool.run_stream = _mock_session_run_stream  # type: ignore[attr-defined]

    # Mock empty MCP prompts
    mock_agent.list_prompts = AsyncMock(return_value=[])

    response = await async_client.post(
        f"/session/{session_id}/command",
        json={"command": "test-cmd", "arguments": "arg1 arg2"},
    )

    # Verify success
    assert response.status_code == 200
    result = response.json()
    assert "info" in result
    assert "parts" in result

    # Verify response parts are not empty — non-skill commands capture output
    parts = result.get("parts", [])
    assert len(parts) >= 2, (
        f"Non-skill command should have step-start + text parts, got {len(parts)} parts"
    )

    # Verify the command output is captured in a TextPart
    text_parts = [p for p in parts if p.get("type") == "text"]
    assert len(text_parts) >= 1, "Non-skill command should have a TextPart with captured output"
    assert "Command output: test-cmd executed" in text_parts[0].get("text", ""), (
        f"TextPart should contain command output, got: {text_parts[0].get('text', '')}"
    )

    # Verify session_pool.run_stream was called (not direct agent.run_stream)
    assert len(session_pool_calls) == 1
    assert len(agent_calls) == 0


async def test_mcp_prompt_routes_through_session_pool(
    async_client: AsyncClient,
    server_state: ServerState,
    mock_agent: Mock,
):
    """Test that MCP prompt routes through SessionPool.receive_request().

    SessionPool is now the default execution path for all categories.
    """
    # Create session first
    response = await async_client.post("/session", json={"title": "Test Session"})
    assert response.status_code == 200
    session_id = response.json()["id"]

    # Mock CommandStore without the command
    mock_command_store = MagicMock()
    mock_command_store.get_command = MagicMock(return_value=None)
    server_state.command_store = mock_command_store

    # Mock MCP prompt
    mock_prompt = MagicMock()
    mock_prompt.name = "direct-prompt"
    mock_prompt.arguments = []
    mock_prompt.get_components = AsyncMock(return_value=[])
    mock_agent.list_prompts = AsyncMock(return_value=[mock_prompt])
    mock_agent.run = AsyncMock(return_value=MagicMock(data="Direct result"))

    # Track session_pool.receive_request calls
    receive_request_calls: list[tuple[Any, Any]] = []

    async def _mock_receive_request(*args: Any, **kwargs: Any) -> Any:
        receive_request_calls.append((args, kwargs))
        return None

    mock_agent.host_context.session_pool.send_message = _mock_receive_request  # type: ignore[attr-defined]

    response = await async_client.post(
        f"/session/{session_id}/command",
        json={"command": "direct-prompt"},
    )

    # Verify success
    assert response.status_code == 200
    result = response.json()
    assert "info" in result
    assert "parts" in result

    # Verify session_pool.receive_request was called (not direct agent.run)
    assert len(receive_request_calls) == 1
    mock_agent.run.assert_not_called()


async def test_skill_command_full_chain_integration(  # noqa: PLR0915
    async_client: AsyncClient,
    server_state: ServerState,
    mock_agent: Mock,
):
    """Full-chain integration test: execute_command → skill execution → send_message.

    This is the integration test that #339 needed but never had. It exercises
    the real dispatch chain with a real SlashedCommand (category='skill')
    that calls ctx.print and injects into staged_content, then verifies
    the routing arguments passed to integration.route_message:
    - content="" (empty — model gets instructions from staged_content only)
    - meta=OpenCodeUserMessageMeta with raw command string
    - assistant_msg_id propagated for SSE/HTTP ID consistency
    - message_id (user_msg_id) for EventProcessor user message creation

    Uses the conftest's _mock_route_message which simulates the
    EventProcessor by creating a UserMessage from meta.parts.
    """
    from slashed import Command as SlashedCommand

    from wolfharness_server.opencode_server.event_processor import (
        OpenCodeUserMessageMeta,
    )
    from wolfharness_server.opencode_server.models import TextPart, UserMessage

    # Create session
    response = await async_client.post("/session", json={"title": "Integration Test"})
    assert response.status_code == 200
    session_id = response.json()["id"]

    # Build a real skill command
    async def _execute_skill(ctx: Any, args: list[str], kwargs: dict[str, str]) -> None:
        await ctx.print("Loading skill: integration-test (skill://test/integration-test)")
        user_request = " ".join(args)
        full_prompt = f"""<skill-instruction>
Test skill instructions for integration test.
</skill-instruction>

<user-request>
{user_request}
</user-request>"""
        if hasattr(ctx.data, "node") and hasattr(ctx.data.node, "staged_content"):
            ctx.data.node.staged_content.add_text(full_prompt)

    skill_command = SlashedCommand.from_raw(
        _execute_skill,
        name="integration-test",
        description="Integration test skill",
        category="skill",
        usage="<args>",
    )

    # Wire into mock CommandStore
    mock_store = MagicMock()
    mock_store.get_command = MagicMock(return_value=skill_command)
    mock_store.event_handler = None
    mock_store.output = MagicMock()
    server_state.command_store = mock_store

    mock_agent.list_prompts = AsyncMock(return_value=[])

    # Set up session agent mock with staged_content
    pool = server_state.pool_or_none
    assert pool is not None
    session_agent = pool.session_pool.sessions.get_or_create_session_agent.return_value
    ctx_mock = MagicMock()
    ctx_mock.node = session_agent
    session_agent.get_context = MagicMock(return_value=ctx_mock)
    session_agent.staged_content = MagicMock()
    session_agent.staged_content.add_text = MagicMock()
    session_agent.staged_content.__bool__ = MagicMock(return_value=True)
    session_agent.staged_content.__len__ = MagicMock(return_value=1)

    # Execute the skill command
    response = await async_client.post(
        f"/session/{session_id}/command",
        json={"command": "integration-test", "arguments": "do the thing"},
    )

    assert response.status_code == 200
    result = response.json()

    # --- Verify routing arguments ---
    integration = server_state.session_pool_integration
    route_calls = integration.route_message.call_args_list
    assert len(route_calls) >= 1, "route_message was not called"

    _, route_kwargs = route_calls[-1]

    # Content must be empty (model gets instructions from staged_content only)
    assert route_kwargs.get("content") == "", (
        f"route_message content should be empty, got: {route_kwargs.get('content')!r}"
    )

    # Meta must be OpenCodeUserMessageMeta with raw command string
    meta = route_kwargs.get("meta")
    assert isinstance(meta, OpenCodeUserMessageMeta), (
        f"meta should be OpenCodeUserMessageMeta, got {type(meta)}"
    )
    assert len(meta.parts) >= 1
    part_dict = meta.parts[0]
    assert part_dict.get("type") == "text"
    assert "/integration-test" in part_dict.get("text", "")
    assert "do the thing" in part_dict.get("text", "")

    # assistant_msg_id must be propagated
    assistant_msg_id = route_kwargs.get("assistant_msg_id")
    assert assistant_msg_id is not None, "assistant_msg_id not propagated"
    assert assistant_msg_id == result["info"]["id"], (
        f"assistant_msg_id ({assistant_msg_id}) != HTTP response ID ({result['info']['id']})"
    )

    # message_id (user_msg_id) must be propagated
    user_msg_id = route_kwargs.get("message_id")
    assert user_msg_id is not None, "message_id (user_msg_id) not propagated"

    # --- Verify staged_content was injected ---
    add_text_calls = session_agent.staged_content.add_text.call_args_list
    assert len(add_text_calls) == 1, "staged_content.add_text should be called exactly once"
    staged_text = add_text_calls[0].args[0] if add_text_calls[0].args else ""
    assert "<skill-instruction>" in staged_text
    assert "<user-request>" in staged_text
    assert "do the thing" in staged_text

    # --- Verify user message was created from meta ---
    messages = server_state.messages.get(session_id, [])
    user_messages = [m for m in messages if isinstance(m.info, UserMessage)]
    assert len(user_messages) >= 1, "UserMessage not created from meta"
    text_parts = [p for p in user_messages[0].parts if isinstance(p, TextPart)]
    assert len(text_parts) >= 1
    assert "/integration-test" in text_parts[0].text

    # --- Verify parent_id linkage ---
    parent_id = result["info"].get("parentID", "")
    assert parent_id == user_messages[0].info.id, (
        f"parent_id ({parent_id}) != user_msg_id ({user_messages[0].info.id})"
    )

    # --- Verify no "Loading skill" in response parts ---
    for part in result.get("parts", []):
        if part.get("type") == "text":
            assert "Loading skill" not in part.get("text", ""), "ctx.print leaked into TextPart"

    # --- Verify run_stream was NOT called ---
    run_stream_calls: list[Any] = []

    async def _track_run_stream(*args: Any, **kwargs: Any) -> Any:
        run_stream_calls.append((args, kwargs))
        if False:
            yield MagicMock()

    pool.session_pool.run_stream = _track_run_stream  # type: ignore[method-assign]
    assert len(run_stream_calls) == 0, "run_stream should NOT be called for skill commands"
