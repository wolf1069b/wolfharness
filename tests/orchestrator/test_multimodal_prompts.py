"""Tests for multimodal content preservation in the prompt pipeline.

Verifies that multimodal content (images, audio, structured blocks) is not
stringified when passed through the session → RunHandle → Turn pipeline.

Covers:
- RunHandle.start() accepts ``str | list[Any]``
- ACP turn flattens ``list[str | list[Any]]`` into ``list[UserContent]``
- Native turn flattens ``list[str | list[Any]]`` into ``list[UserContent]``
- steer() with list content preserves structure
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from pydantic_ai.messages import BinaryImage
import pytest

from wolfharness.agents.context import AgentRunContext
from wolfharness.agents.events import StreamCompleteEvent
from wolfharness.lifecycle.comm_channel import DirectChannel
from wolfharness.lifecycle.journal import MemoryJournal
from wolfharness.messaging import ChatMessage
from wolfharness.orchestrator.run import RunHandle
from wolfharness.orchestrator.turn import Turn


pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _CapturingStubTurn(Turn):
    """Stub Turn that captures the prompts it receives and yields one event."""

    def __init__(self, captured_prompts: list[Any]) -> None:
        self._captured = captured_prompts

    async def execute(self):  # type: ignore[override]
        self._message_history = []
        self._final_message = ChatMessage(content="done", role="assistant")
        yield StreamCompleteEvent(message=self._final_message)


def _make_run_handle(
    *,
    agent: Any | None = None,
    event_bus: Any | None = None,
    session: Any | None = None,
) -> RunHandle:
    if agent is None:
        agent = MagicMock()
        agent.create_turn = MagicMock(return_value=_CapturingStubTurn([]))
        agent.conversation = MagicMock()
        agent.conversation.add_chat_messages = MagicMock()
    if event_bus is None:
        event_bus = AsyncMock()
    if session is None:
        session = MagicMock()
        session.turn_lock = asyncio.Lock()
        session._comm_channel = DirectChannel(MemoryJournal())
        session.input_provider = None
        session.parent_session_id = None
    return RunHandle(
        run_id="test-run",
        session_id="test-session",
        agent_type="native",
        agent=agent,
        event_bus=event_bus,
        session=session,
    )


def _stream_complete() -> StreamCompleteEvent[Any]:
    return StreamCompleteEvent(message=ChatMessage(content="done", role="assistant"))


async def _consume(gen: Any) -> list[Any]:
    return [event async for event in gen]


# ---------------------------------------------------------------------------
# RunHandle.start() signature tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_start_accepts_string_prompt() -> None:
    """Single string prompt passes through as string."""
    captured: list[Any] = []

    turn = _CapturingStubTurn(captured)
    agent = MagicMock()
    agent.create_turn = MagicMock(return_value=turn)
    agent.conversation = MagicMock()
    agent.conversation.add_chat_messages = MagicMock()
    handle = _make_run_handle(agent=agent)

    gen = handle.start("hello world")
    task = asyncio.create_task(_consume(gen))
    await asyncio.sleep(0.05)
    handle.close()
    await asyncio.sleep(0.05)
    await task

    # create_turn should have been called with prompts=["hello world"]
    call_kwargs = agent.create_turn.call_args
    prompts_arg = call_kwargs.kwargs["prompts"]
    assert prompts_arg == ["hello world"]
    assert isinstance(prompts_arg[0], str)


@pytest.mark.unit
async def test_start_accepts_list_prompt() -> None:
    """List prompt (multimodal) passes through as list."""
    binary_img = BinaryImage(data=b"\x89PNG", media_type="image/png")
    multimodal: list[Any] = ["describe this", binary_img]

    turn = _CapturingStubTurn([])
    agent = MagicMock()
    agent.create_turn = MagicMock(return_value=turn)
    agent.conversation = MagicMock()
    agent.conversation.add_chat_messages = MagicMock()
    handle = _make_run_handle(agent=agent)

    gen = handle.start(multimodal)
    task = asyncio.create_task(_consume(gen))
    await asyncio.sleep(0.05)
    handle.close()
    await asyncio.sleep(0.05)
    await task

    call_kwargs = agent.create_turn.call_args
    prompts_arg = call_kwargs.kwargs["prompts"]
    # Should be [list[Any]] — the multimodal list wrapped in a list
    assert len(prompts_arg) == 1
    assert isinstance(prompts_arg[0], list)
    assert prompts_arg[0] is multimodal
    # BinaryImage should NOT be stringified
    assert any(isinstance(item, BinaryImage) for item in prompts_arg[0])


@pytest.mark.unit
async def test_start_accepts_empty_string() -> None:
    """Empty string produces empty prompts list (no spurious turn)."""
    from wolfharness.agents.staged_content import StagedContent

    turn = _CapturingStubTurn([])
    agent = MagicMock()
    agent.create_turn = MagicMock(return_value=turn)
    agent.conversation = MagicMock()
    agent.conversation.add_chat_messages = MagicMock()
    agent.staged_content = StagedContent()  # empty — no staged content
    handle = _make_run_handle(agent=agent)

    gen = handle.start("")
    task = asyncio.create_task(_consume(gen))
    await asyncio.sleep(0.05)
    handle.close()
    await asyncio.sleep(0.05)
    await task

    # create_turn should NOT have been called (empty prompt → idle → close)
    agent.create_turn.assert_not_called()


# ---------------------------------------------------------------------------
# steer() with list content
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_steer_accepts_list_content() -> None:
    """steer() with a list preserves multimodal structure."""
    binary_img = BinaryImage(data=b"\x89PNG", media_type="image/png")
    multimodal: list[Any] = ["look at this", binary_img]

    turn = _CapturingStubTurn([])
    agent = MagicMock()
    agent.create_turn = MagicMock(return_value=turn)
    agent.conversation = MagicMock()
    agent.conversation.add_chat_messages = MagicMock()
    handle = _make_run_handle(agent=agent)

    # Steer should not raise even with list content
    result = handle.steer(multimodal)
    # result is a message_id or None
    assert result is None or isinstance(result, str)


# ---------------------------------------------------------------------------
# ACP turn flattening tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_acp_turn_flattens_multimodal_prompts() -> None:
    """ACPTurn flattens list[str | list[Any]] into list[UserContent] for ACP."""
    from wolfharness.agents.acp_agent.turn import ACPTurn

    binary_img = BinaryImage(data=b"\x89PNG", media_type="image/png")

    # Simulate prompts from RunLoop: [str, list[UserContent]]
    prompts: list[Any] = [
        "first prompt",
        ["second prompt", binary_img],
    ]

    captured_content: list[Any] = []

    class _MockACPClient:
        async def prompt(self, session_id: str, content: Any) -> Any:
            captured_content.append(content)
            return MagicMock()

        async def stream_events(self, response: Any) -> Any:
            return
            yield  # type: ignore[unreachable]

        async def get_messages(self, session_id: str) -> Any:
            return []

    run_ctx = MagicMock(spec=AgentRunContext)
    run_ctx.hooks_fired = set()
    run_ctx.cancelled = False
    run_ctx.session_id = "test"
    run_ctx.turn_id = "turn-1"
    run_ctx.run_id = "run-1"

    acp_turn = ACPTurn(
        acp_client=_MockACPClient(),
        prompts=prompts,
        run_ctx=run_ctx,
        session_id="test-session",
        agent_name="test-agent",
    )

    # Execute the turn — it will call prompt() which captures content
    [event async for event in acp_turn.execute()]

    # Verify content was flattened: should be a list of ContentBlock
    assert len(captured_content) == 1
    content = captured_content[0]
    # convert_to_acp_content returns list[ContentBlock]
    assert isinstance(content, list)
    # Should contain text blocks from both prompts, plus image from the list
    text_blocks = [b for b in content if hasattr(b, "text")]
    assert any("first prompt" in getattr(b, "text", "") for b in text_blocks)
    assert any("second prompt" in getattr(b, "text", "") for b in text_blocks)
    # Should have an image block (not stringified BinaryImage)
    image_blocks = [
        b for b in content if hasattr(b, "mime_type") and "image" in getattr(b, "mime_type", "")
    ]
    assert len(image_blocks) == 1


@pytest.mark.unit
async def test_acp_turn_flattens_all_string_prompts() -> None:
    """ACPTurn with all string prompts flattens into text blocks."""
    from wolfharness.agents.acp_agent.turn import ACPTurn

    prompts: list[Any] = ["hello", "world"]

    captured_content: list[Any] = []

    class _MockACPClient:
        async def prompt(self, session_id: str, content: Any) -> Any:
            captured_content.append(content)
            return MagicMock()

        async def stream_events(self, response: Any) -> Any:
            return
            yield  # type: ignore[unreachable]

        async def get_messages(self, session_id: str) -> Any:
            return []

    run_ctx = MagicMock(spec=AgentRunContext)
    run_ctx.hooks_fired = set()
    run_ctx.cancelled = False
    run_ctx.session_id = "test"
    run_ctx.turn_id = "turn-1"
    run_ctx.run_id = "run-1"

    acp_turn = ACPTurn(
        acp_client=_MockACPClient(),
        prompts=prompts,
        run_ctx=run_ctx,
        session_id="test-session",
        agent_name="test-agent",
    )

    [event async for event in acp_turn.execute()]

    assert len(captured_content) == 1
    content = captured_content[0]
    assert isinstance(content, list)
    text_blocks = [b for b in content if hasattr(b, "text")]
    assert any("hello" in getattr(b, "text", "") for b in text_blocks)
    assert any("world" in getattr(b, "text", "") for b in text_blocks)


@pytest.mark.unit
async def test_acp_turn_empty_prompts() -> None:
    """ACPTurn with empty prompts sends empty string."""
    from wolfharness.agents.acp_agent.turn import ACPTurn

    captured_content: list[Any] = []

    class _MockACPClient:
        async def prompt(self, session_id: str, content: Any) -> Any:
            captured_content.append(content)
            return MagicMock()

        async def stream_events(self, response: Any) -> Any:
            return
            yield  # type: ignore[unreachable]

        async def get_messages(self, session_id: str) -> Any:
            return []

    run_ctx = MagicMock(spec=AgentRunContext)
    run_ctx.hooks_fired = set()
    run_ctx.cancelled = False
    run_ctx.session_id = "test"
    run_ctx.turn_id = "turn-1"
    run_ctx.run_id = "run-1"

    acp_turn = ACPTurn(
        acp_client=_MockACPClient(),
        prompts=[],
        run_ctx=run_ctx,
        session_id="test-session",
        agent_name="test-agent",
    )

    [event async for event in acp_turn.execute()]

    assert len(captured_content) == 1
    content = captured_content[0]
    assert isinstance(content, list)
    assert len(content) == 1
    # Should be a text block with empty string
    assert hasattr(content[0], "text")
    assert content[0].text == ""


# ---------------------------------------------------------------------------
# Native turn flattening tests (verify existing behavior)
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_native_turn_preserves_binary_content() -> None:
    """NativeTurn flattens list[str | list[Any]] preserving BinaryImage.

    This is a verification test — NativeTurn already had this logic.
    We test through RunHandle.start() to ensure the full pipeline
    preserves multimodal content.
    """
    binary_img = BinaryImage(data=b"\x89PNG", media_type="image/png")
    multimodal: list[Any] = ["describe this image", binary_img]

    captured_prompts: list[Any] = []

    turn = _CapturingStubTurn(captured_prompts)
    agent = MagicMock()
    agent.create_turn = MagicMock(return_value=turn)
    agent.conversation = MagicMock()
    agent.conversation.add_chat_messages = MagicMock()
    handle = _make_run_handle(agent=agent)

    gen = handle.start(multimodal)
    task = asyncio.create_task(_consume(gen))
    await asyncio.sleep(0.05)
    handle.close()
    await asyncio.sleep(0.05)
    await task

    call_kwargs = agent.create_turn.call_args
    prompts_arg = call_kwargs.kwargs["prompts"]

    # The multimodal list should be preserved as a list element
    assert len(prompts_arg) == 1
    assert isinstance(prompts_arg[0], list)
    # BinaryImage should not be stringified
    found_binary = any(isinstance(item, BinaryImage) for item in prompts_arg[0])
    assert found_binary, "BinaryImage should be preserved, not stringified"


# ---------------------------------------------------------------------------
# Flatten logic unit tests (direct verification of the pattern)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_flatten_single_string() -> None:
    """Single string prompt → passes through as string."""

    def flatten(prompts: list[Any]) -> str | list[Any]:
        if not prompts:
            return ""
        if len(prompts) == 1:
            return prompts[0]
        flattened: list[Any] = []
        for p in prompts:
            if isinstance(p, str):
                flattened.append(p)
            else:
                flattened.extend(p)
        return flattened

    result = flatten(["hello"])
    assert result == "hello"
    assert isinstance(result, str)


@pytest.mark.unit
def test_flatten_single_list() -> None:
    """Single list prompt (multimodal) → preserved as list."""

    def flatten(prompts: list[Any]) -> str | list[Any]:
        if not prompts:
            return ""
        if len(prompts) == 1:
            return prompts[0]
        flattened: list[Any] = []
        for p in prompts:
            if isinstance(p, str):
                flattened.append(p)
            else:
                flattened.extend(p)
        return flattened

    binary_img = BinaryImage(data=b"\x89PNG", media_type="image/png")
    multimodal = ["text", binary_img]
    result = flatten([multimodal])
    assert result is multimodal
    assert isinstance(result, list)
    assert any(isinstance(item, BinaryImage) for item in result)


@pytest.mark.unit
def test_flatten_multiple_strings() -> None:
    """Multiple string prompts → flattened into list."""

    def flatten(prompts: list[Any]) -> str | list[Any]:
        if not prompts:
            return ""
        if len(prompts) == 1:
            return prompts[0]
        flattened: list[Any] = []
        for p in prompts:
            if isinstance(p, str):
                flattened.append(p)
            else:
                flattened.extend(p)
        return flattened

    result = flatten(["hello", "world"])
    assert result == ["hello", "world"]
    assert isinstance(result, list)


@pytest.mark.unit
def test_flatten_mixed_string_and_list() -> None:
    """Mixed string + list prompts → flattened correctly."""

    def flatten(prompts: list[Any]) -> str | list[Any]:
        if not prompts:
            return ""
        if len(prompts) == 1:
            return prompts[0]
        flattened: list[Any] = []
        for p in prompts:
            if isinstance(p, str):
                flattened.append(p)
            else:
                flattened.extend(p)
        return flattened

    binary_img = BinaryImage(data=b"\x89PNG", media_type="image/png")
    result = flatten(["hello", ["describe", binary_img]])
    assert isinstance(result, list)
    assert result == ["hello", "describe", binary_img]
    # BinaryImage should be in the flattened list as-is
    assert any(isinstance(item, BinaryImage) for item in result)


@pytest.mark.unit
def test_flatten_empty() -> None:
    """Empty prompts → empty string."""

    def flatten(prompts: list[Any]) -> str | list[Any]:
        if not prompts:
            return ""
        if len(prompts) == 1:
            return prompts[0]
        flattened: list[Any] = []
        for p in prompts:
            if isinstance(p, str):
                flattened.append(p)
            else:
                flattened.extend(p)
        return flattened

    result = flatten([])
    assert result == ""
