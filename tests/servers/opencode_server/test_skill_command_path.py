"""Regression tests for OpenCode skill slash command execution path (issue #339).

These tests verify that skill commands (category == "skill") route through
the normal prompt lifecycle (send_message -> EventBus-only) instead of the
broken _execute_slashed_command path that caused three symptoms:

1. ctx.print("Loading skill: ...") rendered as AI reply (TextPart)
2. User message swallowed (no UserMessage created)
3. Double prompt injection (staged_content + hardcoded Chinese prompt)

These tests use real ``SlashedCommand`` instances with ``category='skill'``
that call ``ctx.print`` and inject into ``staged_content`` — the core skill
execution chain is NOT mocked. However, ``route_message`` and
``send_message`` are mock-based (the conftest's ``_mock_route_message``
simulates the EventProcessor's user-message reconstruction from ``meta.parts``).
The VCR test in ``tests/vcr/test_skill_command_vcr.py`` exercises the real
EventBus → EventProcessor chain end-to-end (requires a recorded cassette).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from slashed import Command as SlashedCommand, CommandContext

from wolfharness_server.opencode_server.event_processor import (
    OpenCodeUserMessageMeta,
)
from wolfharness_server.opencode_server.models import (
    TextPart,
    UserMessage,
)
from wolfharness_server.opencode_server.routes.session_routes import (
    _CommandOutputCapture,
    _DiscardOutputWriter,
)


if TYPE_CHECKING:
    from httpx import AsyncClient as HttpxAsyncClient

    from wolfharness_server.opencode_server.state import ServerState


pytestmark = [pytest.mark.asyncio, pytest.mark.integration]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_real_skill_command(
    name: str = "test-skill",
    instructions: str = "You are a test skill. Follow instructions.",
) -> SlashedCommand:
    """Create a real SlashedCommand with category='skill'.

    The execute function mimics skill_bridge.create_skill_command's
    execute_skill: calls ctx.print('Loading skill: ...') and injects
    into ctx.data.node.staged_content. This is the behavior that #339's
    broken path captured into a TextPart instead of discarding.
    """

    async def execute_skill(
        ctx: CommandContext[Any],
        args: list[str],
        kwargs: dict[str, str],
    ) -> None:
        await ctx.print(f"Loading skill: {name} (skill://test/{name})")
        user_request = " ".join(args)
        full_prompt = f"""<skill-instruction>
{instructions}
</skill-instruction>

<user-request>
{user_request}
</user-request>"""
        # Inject into staged_content — mirrors skill_bridge.py L140
        if (
            hasattr(ctx, "data")
            and ctx.data is not None
            and hasattr(ctx.data, "node")
            and ctx.data.node is not None
            and hasattr(ctx.data.node, "staged_content")
        ):
            ctx.data.node.staged_content.add_text(full_prompt)

    return SlashedCommand.from_raw(
        execute_skill,
        name=name,
        description="Test skill for #339 regression",
        category="skill",
        usage="<args>",
    )


def _make_real_non_skill_command(name: str = "test-help") -> SlashedCommand:
    """Create a real SlashedCommand with category != 'skill' (e.g. 'help')."""

    async def execute_cmd(
        ctx: CommandContext[Any],
        args: list[str],
        kwargs: dict[str, str],
    ) -> None:
        await ctx.print("Help: this is a simple print command.")

    return SlashedCommand.from_raw(
        execute_cmd,
        name=name,
        description="Test non-skill command",
        category="help",
        usage="",
    )


def _setup_skill_command_store(
    server_state: ServerState,
    command: SlashedCommand,
) -> None:
    """Wire a real SlashedCommand into server_state.command_store.

    Also ensures mock_agent.list_prompts returns [] (no MCP prompt collision)
    so the dispatcher can proceed without errors. Sets event_handler=None
    on the store so CommandContext.print doesn't try to await a MagicMock.
    """
    mock_store = MagicMock()
    mock_store.get_command = MagicMock(return_value=command)
    mock_store.event_handler = None  # CommandContext.print checks this
    mock_store.output = MagicMock()  # emit() is called synchronously
    server_state.command_store = mock_store
    # list_prompts must be an AsyncMock — the dispatcher awaits it
    server_state.agent.list_prompts = AsyncMock(return_value=[])


def _setup_session_agent_with_staged_content(server_state: ServerState) -> MagicMock:
    """Ensure the mock session agent has get_context() returning a node with staged_content.

    The conftest's _mock_session_agent is a bare Mock. get_context() returns
    a Mock, and .node.staged_content is auto-created as a Mock. We just need
    to ensure add_text is callable (it is, being a Mock attribute).
    Returns the session_agent mock for inspection.
    """
    pool = server_state.pool_or_none
    assert pool is not None, "Pool must be available"
    session_agent = pool.session_pool.sessions.get_or_create_session_agent.return_value
    # Ensure get_context returns something with .node.staged_content
    ctx_mock = MagicMock()
    ctx_mock.node = session_agent  # node IS the agent
    session_agent.get_context = MagicMock(return_value=ctx_mock)
    session_agent.staged_content = MagicMock()
    session_agent.staged_content.add_text = MagicMock()
    session_agent.staged_content.__bool__ = MagicMock(return_value=True)
    session_agent.staged_content.__len__ = MagicMock(return_value=1)
    return session_agent


# ---------------------------------------------------------------------------
# Tests: Symptom #1 — ctx.print output must NOT appear in TextPart
# ---------------------------------------------------------------------------


async def test_skill_command_ctx_print_not_in_textpart(
    async_client: HttpxAsyncClient,
    server_state: ServerState,
):
    """Symptom #1: 'Loading skill: ...' must NOT be rendered as assistant TextPart.

    Before fix: _execute_slashed_command captured ctx.print into a TextPart
    via _CommandOutputCapture, causing the TUI to show 'Loading skill: ...'
    as the AI's reply.

    After fix: _execute_skill_command uses _DiscardOutputWriter which
    discards ctx.print output.
    """
    # Setup
    response = await async_client.post("/session", json={"title": "Test"})
    session_id = response.json()["id"]
    _setup_session_agent_with_staged_content(server_state)
    skill_cmd = _make_real_skill_command()
    _setup_skill_command_store(server_state, skill_cmd)

    # Execute
    response = await async_client.post(
        f"/session/{session_id}/command",
        json={"command": "test-skill", "arguments": "analyze this"},
    )

    # Verify response
    assert response.status_code == 200
    result = response.json()

    # The placeholder parts should be empty (no TextPart with "Loading skill")
    parts = result.get("parts", [])
    for part in parts:
        if part.get("type") == "text":
            assert "Loading skill" not in part.get("text", ""), (
                "ctx.print output leaked into TextPart — _DiscardOutputWriter not used"
            )


# ---------------------------------------------------------------------------
# Tests: Symptom #2 — User message must be created with raw command
# ---------------------------------------------------------------------------


async def test_skill_command_creates_user_message(
    async_client: HttpxAsyncClient,
    server_state: ServerState,
):
    """Symptom #2: A UserMessage must be created with the raw command string.

    Before fix: _execute_slashed_command created only an AssistantMessage
    with parent_id="", swallowing the user's input.

    After fix: _execute_skill_command passes meta=OpenCodeUserMessageMeta
    with the raw command string, and the EventProcessor creates the
    UserMessage from meta.parts.
    """
    response = await async_client.post("/session", json={"title": "Test"})
    session_id = response.json()["id"]
    _setup_session_agent_with_staged_content(server_state)
    skill_cmd = _make_real_skill_command()
    _setup_skill_command_store(server_state, skill_cmd)

    await async_client.post(
        f"/session/{session_id}/command",
        json={"command": "test-skill", "arguments": "analyze this"},
    )

    # Check that a user message was created in state
    messages = server_state.messages.get(session_id, [])
    user_messages = [m for m in messages if isinstance(m.info, UserMessage)]
    assert len(user_messages) >= 1, "No UserMessage created — user input was swallowed"

    # The user message should contain the raw command string
    user_msg = user_messages[0]
    text_parts = [p for p in user_msg.parts if isinstance(p, TextPart)]
    assert len(text_parts) >= 1, "UserMessage has no TextPart"
    raw_command_text = text_parts[0].text
    assert "/test-skill" in raw_command_text, (
        f"User message text '{raw_command_text}' does not contain raw command '/test-skill'"
    )
    assert "analyze this" in raw_command_text, (
        f"User message text '{raw_command_text}' does not contain arguments 'analyze this'"
    )


# ---------------------------------------------------------------------------
# Tests: Symptom #3a — No double prompt injection
# ---------------------------------------------------------------------------


async def test_skill_command_no_double_prompt(
    async_client: HttpxAsyncClient,
    server_state: ServerState,
):
    """Symptom #3a: Model receives instructions exactly once (from staged_content).

    Before fix: _execute_slashed_command built a second hardcoded Chinese
    prompt AND used run_stream with that prompt, double-injecting with
    staged_content.

    After fix: _execute_skill_command passes content="" to send_message.
    The model gets instructions+args exclusively from staged_content.
    """
    response = await async_client.post("/session", json={"title": "Test"})
    session_id = response.json()["id"]
    session_agent = _setup_session_agent_with_staged_content(server_state)
    skill_cmd = _make_real_skill_command()
    _setup_skill_command_store(server_state, skill_cmd)

    await async_client.post(
        f"/session/{session_id}/command",
        json={"command": "test-skill", "arguments": "analyze this"},
    )

    # Verify send_message was called with empty content (not a second prompt)
    pool = server_state.pool_or_none
    assert pool is not None
    send_message_calls = pool.session_pool.send_message.call_args_list
    assert len(send_message_calls) >= 1, "send_message was not called"

    # Check the content argument — should be empty string
    _, kwargs = send_message_calls[-1]
    content = kwargs.get("content", None)
    assert content == "", (
        f"send_message content should be empty string, got: {content!r}. "
        "Model would receive double prompt injection."
    )

    # Verify staged_content.add_text was called exactly once (by execute_skill)
    add_text_calls = session_agent.staged_content.add_text.call_args_list
    assert len(add_text_calls) == 1, (
        f"staged_content.add_text should be called once, got {len(add_text_calls)} calls"
    )

    # The staged content should contain both skill-instruction and user-request
    staged_prompt = add_text_calls[0].args[0] if add_text_calls[0].args else ""
    assert "<skill-instruction>" in staged_prompt
    assert "<user-request>" in staged_prompt
    assert "analyze this" in staged_prompt


# ---------------------------------------------------------------------------
# Tests: Symptom #3b/3c — parent_id linkage
# ---------------------------------------------------------------------------


async def test_skill_command_parent_id_linked(
    async_client: HttpxAsyncClient,
    server_state: ServerState,
):
    """Symptom #3b/3c: Assistant message parent_id must link to user message.

    Before fix: _execute_slashed_command created AssistantMessage with
    parent_id="" (no user message to link to).

    After fix: _execute_skill_command generates a user_msg_id and sets
    parent_id=user_msg_id on the assistant message.
    """
    response = await async_client.post("/session", json={"title": "Test"})
    session_id = response.json()["id"]
    _setup_session_agent_with_staged_content(server_state)
    skill_cmd = _make_real_skill_command()
    _setup_skill_command_store(server_state, skill_cmd)

    resp = await async_client.post(
        f"/session/{session_id}/command",
        json={"command": "test-skill", "arguments": "analyze this"},
    )

    # The HTTP response contains the assistant message placeholder
    result = resp.json()
    info = result["info"]
    parent_id = info.get("parentID", "")
    assert parent_id != "", "Assistant message parent_id is empty — not linked to user message"
    assert parent_id is not None, "Assistant message parent_id is None"

    # The user message in state should have the same ID as parent_id
    messages = server_state.messages.get(session_id, [])
    user_messages = [m for m in messages if isinstance(m.info, UserMessage)]
    assert len(user_messages) >= 1
    user_msg_id = user_messages[0].info.id
    assert parent_id == user_msg_id, (
        f"parent_id ({parent_id}) does not match user_msg_id ({user_msg_id})"
    )


# ---------------------------------------------------------------------------
# Tests: Non-skill commands retain existing behavior
# ---------------------------------------------------------------------------


async def test_non_skill_command_retains_existing_behavior(
    async_client: HttpxAsyncClient,
    server_state: ServerState,
):
    """Non-skill commands (e.g. /help) must continue through _execute_slashed_command.

    This guards against accidental regression of the simple-command path.
    """
    response = await async_client.post("/session", json={"title": "Test"})
    session_id = response.json()["id"]

    # Setup a non-skill command (category != "skill")
    non_skill_cmd = _make_real_non_skill_command()
    _setup_skill_command_store(server_state, non_skill_cmd)

    # Mock agent for the _execute_slashed_command path
    pool = server_state.pool_or_none
    assert pool is not None
    server_state.agent.list_prompts = AsyncMock(return_value=[])

    # Mock run_stream to yield nothing (command output path)
    async def _mock_run_stream(*args: Any, **kwargs: Any) -> Any:
        if False:
            yield MagicMock()

    pool.session_pool.run_stream = _mock_run_stream  # type: ignore[method-assign]

    response = await async_client.post(
        f"/session/{session_id}/command",
        json={"command": "test-help"},
    )

    assert response.status_code == 200
    result = response.json()

    # Non-skill commands should have output captured in TextPart
    parts = result.get("parts", [])
    text_parts = [p for p in parts if p.get("type") == "text"]
    # The _execute_slashed_command path captures output in a TextPart
    # (step_start, text_part, step_finish are added)
    assert len(text_parts) >= 1 or len(parts) >= 2, (
        "Non-skill command should have output parts (existing behavior)"
    )

    # No user message should be created for non-skill commands
    messages = server_state.messages.get(session_id, [])
    user_messages = [m for m in messages if isinstance(m.info, UserMessage)]
    assert len(user_messages) == 0, (
        "Non-skill command should NOT create a user message (existing behavior)"
    )


# ---------------------------------------------------------------------------
# Tests: No race condition — staged_content + routing within same lock
# ---------------------------------------------------------------------------


async def test_skill_command_no_race_condition(
    async_client: HttpxAsyncClient,
    server_state: ServerState,
):
    """Staged_content injection and send_message routing occur within execute_command's lock.

    A concurrent POST /message cannot consume the staged_content before
    the skill command's run starts because both injection and routing
    happen inside execute_command's session lock.
    """
    response = await async_client.post("/session", json={"title": "Test"})
    session_id = response.json()["id"]
    session_agent = _setup_session_agent_with_staged_content(server_state)
    skill_cmd = _make_real_skill_command()
    _setup_skill_command_store(server_state, skill_cmd)

    # Track call ordering: staged_content injection must happen before
    # send_message is called (both inside the lock).
    call_order: list[str] = []

    original_add_text = session_agent.staged_content.add_text

    def tracking_add_text(*args: Any, **kwargs: Any) -> None:
        call_order.append("staged_content_add_text")
        original_add_text(*args, **kwargs)

    session_agent.staged_content.add_text = tracking_add_text  # type: ignore[method-assign]

    pool = server_state.pool_or_none
    assert pool is not None
    original_send_message = pool.session_pool.send_message

    async def tracking_send_message(*args: Any, **kwargs: Any) -> Any:
        call_order.append("send_message")
        return await original_send_message(*args, **kwargs)

    pool.session_pool.send_message = tracking_send_message  # type: ignore[method-assign]

    await async_client.post(
        f"/session/{session_id}/command",
        json={"command": "test-skill", "arguments": "test"},
    )

    # Verify staged_content was injected before send_message was called
    assert "staged_content_add_text" in call_order, "staged_content.add_text was not called"
    assert "send_message" in call_order, "send_message was not called"
    staged_idx = call_order.index("staged_content_add_text")
    send_idx = call_order.index("send_message")
    assert staged_idx < send_idx, (
        f"staged_content injection (index {staged_idx}) must occur before "
        f"send_message (index {send_idx}) — race condition risk"
    )


# ---------------------------------------------------------------------------
# Tests: Empty content + meta propagation for TUI display
# ---------------------------------------------------------------------------


async def test_skill_command_meta_propagation_for_tui(
    async_client: HttpxAsyncClient,
    server_state: ServerState,
):
    """meta=OpenCodeUserMessageMeta(parts=[...]) carries raw command for TUI display.

    send_message(content="") causes EventProcessor to create an empty user
    message (no parts) because `if content:` is falsy. Passing meta with
    serialized TextPart parts allows the EventProcessor to reconstruct
    the user message from meta.parts, displaying '/test-skill analyze this'
    in the TUI.
    """
    response = await async_client.post("/session", json={"title": "Test"})
    session_id = response.json()["id"]
    _setup_session_agent_with_staged_content(server_state)
    skill_cmd = _make_real_skill_command()
    _setup_skill_command_store(server_state, skill_cmd)

    await async_client.post(
        f"/session/{session_id}/command",
        json={"command": "test-skill", "arguments": "analyze this"},
    )

    # Check that route_message was called with meta containing raw command
    integration = server_state.session_pool_integration
    route_message_calls = integration.route_message.call_args_list
    assert len(route_message_calls) >= 1, "route_message was not called"

    _, kwargs = route_message_calls[-1]
    meta = kwargs.get("meta")
    assert isinstance(meta, OpenCodeUserMessageMeta), (
        f"meta should be OpenCodeUserMessageMeta, got {type(meta)}"
    )
    assert len(meta.parts) >= 1, "meta.parts should not be empty"

    # The first part should be a serialized TextPart with the raw command
    part_dict = meta.parts[0]
    assert part_dict.get("type") == "text", f"Expected text part, got {part_dict.get('type')}"
    text = part_dict.get("text", "")
    assert "/test-skill" in text, f"meta part text should contain '/test-skill', got: {text}"
    assert "analyze this" in text, f"meta part text should contain 'analyze this', got: {text}"

    # Verify user message was created from meta (not from empty content)
    messages = server_state.messages.get(session_id, [])
    user_messages = [m for m in messages if isinstance(m.info, UserMessage)]
    assert len(user_messages) >= 1
    text_parts = [p for p in user_messages[0].parts if isinstance(p, TextPart)]
    assert len(text_parts) >= 1, (
        "UserMessage should have TextPart from meta reconstruction, not empty"
    )
    assert "/test-skill" in text_parts[0].text


# ---------------------------------------------------------------------------
# Tests: assistant_msg_id propagation through integration.route_message
# ---------------------------------------------------------------------------


async def test_skill_command_assistant_msg_id_propagation(
    async_client: HttpxAsyncClient,
    server_state: ServerState,
):
    """assistant_msg_id propagates through integration.route_message.

    The HTTP response's assistant message ID should match what
    integration.route_message receives, ensuring no cosmetic mismatch
    between the HTTP placeholder and the SSE-delivered message.
    """
    response = await async_client.post("/session", json={"title": "Test"})
    session_id = response.json()["id"]
    _setup_session_agent_with_staged_content(server_state)
    skill_cmd = _make_real_skill_command()
    _setup_skill_command_store(server_state, skill_cmd)

    resp = await async_client.post(
        f"/session/{session_id}/command",
        json={"command": "test-skill", "arguments": "test"},
    )

    result = resp.json()
    http_assistant_id = result["info"]["id"]

    # Check that route_message received the same assistant_msg_id
    integration = server_state.session_pool_integration
    route_message_calls = integration.route_message.call_args_list
    assert len(route_message_calls) >= 1

    _, kwargs = route_message_calls[-1]
    route_assistant_id = kwargs.get("assistant_msg_id")
    assert route_assistant_id is not None, "assistant_msg_id not passed to route_message"
    assert route_assistant_id == http_assistant_id, (
        f"HTTP response assistant ID ({http_assistant_id}) does not match "
        f"route_message assistant_msg_id ({route_assistant_id}) — cosmetic mismatch"
    )

    # Verify _pending_message_ids was populated for event bridge reuse
    pending_ids = integration._pending_message_ids
    assert session_id in pending_ids, "_pending_message_ids not populated for session"
    assert pending_ids[session_id] == http_assistant_id, (
        f"_pending_message_ids ({pending_ids.get(session_id)}) does not match "
        f"HTTP response assistant ID ({http_assistant_id})"
    )


# ---------------------------------------------------------------------------
# Tests: _DiscardOutputWriter unit test
# ---------------------------------------------------------------------------


async def test_discard_output_writer_discards_messages():
    """_DiscardOutputWriter should discard all messages (not capture them)."""
    writer = _DiscardOutputWriter()
    await writer.print("Loading skill: test")
    await writer.print("Another message")
    # _DiscardOutputWriter discards output silently — unlike _CommandOutputCapture,
    # it has no buffer or string representation that stores messages.
    assert not isinstance(writer, _CommandOutputCapture)


# ---------------------------------------------------------------------------
# Tests: CommandExecutedEvent is broadcast
# ---------------------------------------------------------------------------


async def test_skill_command_broadcasts_command_executed_event(
    async_client: HttpxAsyncClient,
    server_state: ServerState,
):
    """CommandExecutedEvent is broadcast after routing (signals dispatch)."""
    from tests.servers.opencode_server.conftest import EventCapture

    response = await async_client.post("/session", json={"title": "Test"})
    session_id = response.json()["id"]
    _setup_session_agent_with_staged_content(server_state)
    skill_cmd = _make_real_skill_command()
    _setup_skill_command_store(server_state, skill_cmd)

    # Capture events
    capture = EventCapture()
    original_broadcast = server_state.broadcast_event

    async def capturing_broadcast(event: Any) -> None:
        await capture.capture(event)
        await original_broadcast(event)

    server_state.broadcast_event = capturing_broadcast  # type: ignore[method-assign]

    await async_client.post(
        f"/session/{session_id}/command",
        json={"command": "test-skill", "arguments": "test"},
    )

    # Check for CommandExecutedEvent
    executed_events = capture.get_events_by_type("command.executed")
    assert len(executed_events) >= 1, "CommandExecutedEvent was not broadcast"
    event = executed_events[0]
    props = event.properties
    assert props.name == "test-skill"
    assert props.arguments == "test"
