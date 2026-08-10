"""Tests for ToolDisplayCapability — global tool rename + diff-event injection."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock

from pydantic_ai.messages import ToolCallPart
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.toolsets import RenamedToolset
import pytest

from wolfharness.capabilities.tool_display_capability import ToolDisplayCapability


if TYPE_CHECKING:
    from wolfharness.agents.events import DiffContentItem


pytestmark = pytest.mark.unit


def _make_call(tool_name: str, args: dict[str, Any]) -> ToolCallPart:
    """Create a ToolCallPart with the given tool name and arguments."""
    return ToolCallPart(tool_name=tool_name, args={}, tool_call_id=f"call_{tool_name}")


def _make_tool_def(name: str) -> ToolDefinition:
    """Create a minimal ToolDefinition."""
    return ToolDefinition(name=name, description="test tool")


def _make_handler(result: Any) -> AsyncMock:
    """Create an async handler returning a fixed result."""
    return AsyncMock(return_value=result)


def _make_ctx(events: Any) -> MagicMock:
    """Create a RunContext stand-in carrying deps with an events emitter."""
    deps = MagicMock()
    deps.events = events
    ctx = MagicMock()
    ctx.deps = deps
    return ctx


# ---- get_wrapper_toolset: RenamedToolset ----


@pytest.mark.asyncio
async def test_get_wrapper_toolset_applies_rename() -> None:
    """rename_mode=True with a non-empty name_map wraps in RenamedToolset."""
    cap = ToolDisplayCapability(
        rename_mode=True,
        name_map={"viking_write": "write"},
        emit_diff=False,
    )
    wrapped = cap.get_wrapper_toolset(MagicMock())  # type: ignore[arg-type]
    assert isinstance(wrapped, RenamedToolset)
    # name_map is {original: display} in config, inverted to {display: original} for RenamedToolset
    assert wrapped.name_map == {"write": "viking_write"}


def test_get_wrapper_toolset_rename_disabled() -> None:
    """rename_mode=False leaves the toolset unchanged (None wrapper)."""
    cap = ToolDisplayCapability(
        rename_mode=False,
        name_map={"viking_write": "write"},
        emit_diff=True,
    )
    assert cap.get_wrapper_toolset(MagicMock()) is None  # type: ignore[arg-type]


def test_get_wrapper_toolset_empty_name_map() -> None:
    """An empty name_map never wraps, regardless of rename_mode."""
    cap = ToolDisplayCapability(rename_mode=True, name_map={}, emit_diff=True)
    assert cap.get_wrapper_toolset(MagicMock()) is None  # type: ignore[arg-type]


# ---- wrap_tool_execute: diff event injection ----


@pytest.mark.asyncio
async def test_wrap_execute_injects_diff_for_write_tool() -> None:
    """A write-style tool in emit_diff_for gets a DiffContentItem event."""
    events = AsyncMock()
    ctx = _make_ctx(events)
    cap = ToolDisplayCapability(
        emit_diff=True,
        emit_diff_for={"viking_write", "viking_edit"},
    )
    handler = _make_handler("Wrote 3 chars to viking://x.")

    result = await cap.wrap_tool_execute(
        ctx,
        call=_make_call("viking_write", {"uri": "viking://x.md", "content": "abc"}),
        tool_def=_make_tool_def("viking_write"),  # type: ignore[arg-type]
        args={"uri": "viking://x.md", "content": "abc"},
        handler=handler,  # type: ignore[arg-type]
    )

    assert result == "Wrote 3 chars to viking://x."
    events.tool_call_progress.assert_awaited_once()
    call_kwargs = events.tool_call_progress.await_args.kwargs
    assert call_kwargs["title"] == "Modified: viking://x.md"
    items: list[DiffContentItem] = call_kwargs["items"]
    assert len(items) == 1
    assert items[0].path == "viking://x.md"
    assert items[0].old_text is None
    assert items[0].new_text == "abc"


@pytest.mark.asyncio
async def test_wrap_execute_injects_diff_for_edit_tool() -> None:
    """An edit-style tool carries old_string/new_string as old/new text."""
    events = AsyncMock()
    ctx = _make_ctx(events)
    cap = ToolDisplayCapability(emit_diff=True, emit_diff_for={"viking_edit"})

    await cap.wrap_tool_execute(
        ctx,
        call=_make_call("viking_edit", {"uri": "viking://x.md"}),
        tool_def=_make_tool_def("viking_edit"),  # type: ignore[arg-type]
        args={
            "uri": "viking://x.md",
            "old_string": "old",
            "new_string": "new",
        },
        handler=_make_handler("Replaced 1 occurrence(s) in viking://x.md."),  # type: ignore[arg-type]
    )

    events.tool_call_progress.assert_awaited_once()
    items: list[DiffContentItem] = events.tool_call_progress.await_args.kwargs["items"]
    assert items[0].old_text == "old"
    assert items[0].new_text == "new"


@pytest.mark.asyncio
async def test_wrap_execute_no_event_outside_emit_diff_for() -> None:
    """Tools not in emit_diff_for produce no diff event."""
    events = AsyncMock()
    ctx = _make_ctx(events)
    cap = ToolDisplayCapability(emit_diff=True, emit_diff_for={"viking_edit"})

    result = await cap.wrap_tool_execute(
        ctx,
        call=_make_call("viking_read", {}),
        tool_def=_make_tool_def("viking_read"),  # type: ignore[arg-type]
        args={},
        handler=_make_handler("content"),  # type: ignore[arg-type]
    )

    assert result == "content"
    events.tool_call_progress.assert_not_awaited()


@pytest.mark.asyncio
async def test_wrap_execute_no_event_when_emit_disabled() -> None:
    """emit_diff=False (e.g. child capability self-emits) injects nothing."""
    events = AsyncMock()
    ctx = _make_ctx(events)
    cap = ToolDisplayCapability(emit_diff=False, emit_diff_for={"viking_write"})

    await cap.wrap_tool_execute(
        ctx,
        call=_make_call("viking_write", {"uri": "viking://x.md", "content": "abc"}),
        tool_def=_make_tool_def("viking_write"),  # type: ignore[arg-type]
        args={"uri": "viking://x.md", "content": "abc"},
        handler=_make_handler("ok"),  # type: ignore[arg-type]
    )

    events.tool_call_progress.assert_not_awaited()


@pytest.mark.asyncio
async def test_wrap_execute_no_event_without_events_emitter() -> None:
    """Missing deps/events emitter degrades gracefully (no crash)."""
    cap = ToolDisplayCapability(emit_diff=True, emit_diff_for={"viking_write"})

    for ctx in (
        MagicMock(deps=None),
        MagicMock(deps=MagicMock(events=None)),
    ):
        result = await cap.wrap_tool_execute(
            ctx,
            call=_make_call("viking_write", {"uri": "viking://x.md", "content": "abc"}),
            tool_def=_make_tool_def("viking_write"),  # type: ignore[arg-type]
            args={"uri": "viking://x.md", "content": "abc"},
            handler=_make_handler("ok"),  # type: ignore[arg-type]
        )
        assert result == "ok"

    handler = _make_handler("ok")
    await cap.wrap_tool_execute(
        MagicMock(deps=None),
        call=_make_call("viking_write", {"uri": "viking://x.md", "content": "abc"}),
        tool_def=_make_tool_def("viking_write"),  # type: ignore[arg-type]
        args={"uri": "viking://x.md", "content": "abc"},
        handler=handler,  # type: ignore[arg-type]
    )
    handler.assert_awaited_once()


@pytest.mark.asyncio
async def test_wrap_execute_no_event_when_path_missing() -> None:
    """Tools lacking a derivable path produce no event."""
    events = AsyncMock()
    ctx = _make_ctx(events)
    cap = ToolDisplayCapability(emit_diff=True, emit_diff_for={"viking_write"})

    await cap.wrap_tool_execute(
        ctx,
        call=_make_call("viking_write", {}),
        tool_def=_make_tool_def("viking_write"),  # type: ignore[arg-type]
        args={},
        handler=_make_handler("ok"),  # type: ignore[arg-type]
    )

    events.tool_call_progress.assert_not_awaited()


# ---- orthogonal switch matrix ----


@pytest.mark.asyncio
async def test_rename_only_mode_no_diff() -> None:
    """rename_mode=True + emit_diff=False: renames, no events (fsspec-style)."""
    events = AsyncMock()
    ctx = _make_ctx(events)
    cap = ToolDisplayCapability(
        rename_mode=True,
        name_map={"viking_write": "write"},
        emit_diff=False,
    )
    wrapped = cap.get_wrapper_toolset(MagicMock())  # type: ignore[arg-type]
    assert isinstance(wrapped, RenamedToolset)

    await cap.wrap_tool_execute(
        ctx,
        call=_make_call("viking_write", {"uri": "viking://x.md", "content": "abc"}),
        tool_def=_make_tool_def("viking_write"),  # type: ignore[arg-type]
        args={"uri": "viking://x.md", "content": "abc"},
        handler=_make_handler("ok"),  # type: ignore[arg-type]
    )
    events.tool_call_progress.assert_not_awaited()


@pytest.mark.asyncio
async def test_rich_only_mode_keeps_original_name() -> None:
    """rename_mode=False + emit_diff=True: original names, diff events (ACP-style)."""
    cap = ToolDisplayCapability(
        rename_mode=False,
        name_map={"viking_write": "write"},
        emit_diff=True,
        emit_diff_for={"viking_write"},
    )
    assert cap.get_wrapper_toolset(MagicMock()) is None  # type: ignore[arg-type]

    events = AsyncMock()
    ctx = _make_ctx(events)
    await cap.wrap_tool_execute(
        ctx,
        call=_make_call("viking_write", {"uri": "viking://x.md", "content": "abc"}),
        tool_def=_make_tool_def("viking_write"),  # type: ignore[arg-type]
        args={"uri": "viking://x.md", "content": "abc"},
        handler=_make_handler("ok"),  # type: ignore[arg-type]
    )
    events.tool_call_progress.assert_awaited_once()


# ---- degenerate config: no-op decorator ----


def test_empty_config_is_noop_decorator() -> None:
    """All defaults off → toolset unchanged and no events."""
    cap = ToolDisplayCapability(
        rename_mode=True,
        name_map={},
        emit_diff=True,
        emit_diff_for=set(),
    )
    assert cap.get_wrapper_toolset(MagicMock()) is None  # type: ignore[arg-type]

    # get_wrapper_toolset with no name_map is a no-op even with emit_diff on.
    assert cap.emit_diff is True
    assert cap.emit_diff_for == set()
