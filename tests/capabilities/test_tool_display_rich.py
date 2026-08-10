"""Tests for ToolDisplayCapability emit_rich layer — rich display for read/query tools.

Covers the four-axis switch matrix (rename / emit_diff / emit_rich),
pre-execution ``ToolCallStartEvent`` (kind + locations), post-execution
content items, cross-boundary name resolution, and no-duplicate emission
when a tool is targeted by both rich and diff layers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock

from pydantic_ai.messages import ToolCallPart
from pydantic_ai.tools import ToolDefinition
import pytest

from wolfharness.capabilities.tool_display_capability import (
    ToolDisplayCapability,
    _parse_locations,
    _unwrap_result,
)


if TYPE_CHECKING:
    from wolfharness.agents.events import TextContentItem


pytestmark = pytest.mark.unit


def _make_call(tool_name: str, tool_call_id: str = "call_rich") -> ToolCallPart:
    """Create a ToolCallPart with a real tool_call_id."""
    return ToolCallPart(tool_name=tool_name, args={}, tool_call_id=tool_call_id)


def _make_tool_def(name: str) -> ToolDefinition:
    return ToolDefinition(name=name, description="test tool")


def _make_handler(result: Any) -> AsyncMock:
    return AsyncMock(return_value=result)


class _FakeDeps:
    """Stand-in for AgentContext where tool_wrapping.py never ran."""

    def __init__(self, events: Any = None) -> None:
        self.tool_call_id: str | None = None
        self.tool_name: str | None = None
        self._events = events or AsyncMock()

    @property
    def events(self) -> Any:
        return self._events


def _make_ctx(deps: Any = None) -> Any:
    """Create a minimal ctx object with ``.deps``."""
    ctx = type("_Ctx", (), {})()
    ctx.deps = deps or _FakeDeps()
    return ctx


# ---------------------------------------------------------------------------
# 5.1 Switch matrix: emit_rich on/off, empty set, tool not in set
# ---------------------------------------------------------------------------


class TestRichSwitchMatrix:
    """``emit_rich`` / ``emit_rich_for`` must gate rich injection orthogonally."""

    @pytest.mark.asyncio
    async def test_rich_injects_start_and_progress_for_targeted_tool(self) -> None:
        """A read tool in emit_rich_for gets start + progress events."""
        events = AsyncMock()
        deps = _FakeDeps(events=events)
        ctx = _make_ctx(deps)

        cap = ToolDisplayCapability(
            emit_diff=False,
            emit_rich=True,
            emit_rich_for={"viking_read"},
        )

        result = await cap.wrap_tool_execute(
            ctx,
            call=_make_call("viking_read"),
            tool_def=_make_tool_def("viking_read"),
            args={"uris": ["viking://a.md", "viking://b.md"]},
            handler=_make_handler("=== viking://a.md ===\ncontent a"),
        )

        assert result == "=== viking://a.md ===\ncontent a"
        events.tool_call_start.assert_awaited_once()
        assert events.tool_call_start.await_args.kwargs["kind"] == "read"
        assert events.tool_call_start.await_args.kwargs["locations"] == [
            "viking://a.md",
            "viking://b.md",
        ]
        events.tool_call_progress.assert_awaited_once()
        # deps populated so downstream converters don't drop the event
        assert deps.tool_call_id == "call_rich"

    @pytest.mark.asyncio
    async def test_rich_disabled_emits_nothing(self) -> None:
        """emit_rich=False leaves tools untouched."""
        events = AsyncMock()
        ctx = _make_ctx(_FakeDeps(events=events))
        cap = ToolDisplayCapability(
            emit_rich=False,
            emit_rich_for={"viking_read"},
        )

        result = await cap.wrap_tool_execute(
            ctx,
            call=_make_call("viking_read"),
            tool_def=_make_tool_def("viking_read"),
            args={"uris": ["viking://a.md"]},
            handler=_make_handler("content"),
        )

        assert result == "content"
        events.tool_call_start.assert_not_awaited()
        events.tool_call_progress.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_tool_not_in_rich_set_emits_nothing(self) -> None:
        """A tool outside emit_rich_for gets no rich events."""
        events = AsyncMock()
        ctx = _make_ctx(_FakeDeps(events=events))
        cap = ToolDisplayCapability(
            emit_rich=True,
            emit_rich_for={"viking_read"},
        )

        result = await cap.wrap_tool_execute(
            ctx,
            call=_make_call("viking_search"),
            tool_def=_make_tool_def("viking_search"),
            args={"query": "foo"},
            handler=_make_handler("results"),
        )

        assert result == "results"
        events.tool_call_start.assert_not_awaited()


# ---------------------------------------------------------------------------
# 5.2 / 5.3 Location & content derivation
# ---------------------------------------------------------------------------


class TestRichDerivation:
    """Locations come from recognized path keys (incl. lists); content from result."""

    def test_parse_locations_scalar_and_list(self) -> None:
        assert _parse_locations({"uri": "viking://a.md"}) == ["viking://a.md"]
        assert _parse_locations({"uris": ["viking://a.md", "viking://b.md"]}) == [
            "viking://a.md",
            "viking://b.md",
        ]
        assert _parse_locations({"path": "x.py", "uris": ["y.md"]}) == ["x.py", "y.md"]

    def test_parse_locations_empty(self) -> None:
        assert _parse_locations({"query": "foo"}) == []

    def test_unwrap_result_toolreturn_and_str(self) -> None:
        assert _unwrap_result("plain") == "plain"
        from pydantic_ai.messages import ToolReturn

        assert _unwrap_result(ToolReturn(return_value="wrapped")) == "wrapped"

    @pytest.mark.asyncio
    async def test_rich_progress_carries_content_item(self) -> None:
        """Post-execution: registered extractor wraps result into a content item."""
        events = AsyncMock()
        ctx = _make_ctx(_FakeDeps(events=events))
        cap = ToolDisplayCapability(
            emit_rich=True,
            emit_rich_for={"viking_read"},
        )

        await cap.wrap_tool_execute(
            ctx,
            call=_make_call("viking_read"),
            tool_def=_make_tool_def("viking_read"),
            args={"uris": ["viking://a.md"]},
            handler=_make_handler("line 1\nline 2"),
        )

        items: list[TextContentItem] = events.tool_call_progress.await_args.kwargs["items"]
        assert len(items) == 1
        assert items[0].text == "line 1\nline 2"

    @pytest.mark.asyncio
    async def test_unknown_tool_no_content_but_title(self) -> None:
        """Unregistered tool: start event still emitted (title/kind), no content."""
        events = AsyncMock()
        ctx = _make_ctx(_FakeDeps(events=events))
        cap = ToolDisplayCapability(
            emit_rich=True,
            emit_rich_for={"some_tool"},
        )

        await cap.wrap_tool_execute(
            ctx,
            call=_make_call("some_tool"),
            tool_def=_make_tool_def("some_tool"),
            args={"path": "x.py"},
            handler=_make_handler("result"),
        )

        events.tool_call_start.assert_awaited_once()
        assert events.tool_call_start.await_args.kwargs["locations"] == ["x.py"]
        events.tool_call_progress.assert_not_awaited()


# ---------------------------------------------------------------------------
# 5.4 Rename + rich combination (ACP: rename_mode=False still gets rich)
# ---------------------------------------------------------------------------


class TestRichWithRename:
    """Rich layer is orthogonal to rename."""

    @pytest.mark.asyncio
    async def test_rich_works_with_rename_disabled(self) -> None:
        """rename_mode=False (ACP) still emits rich events with original names."""
        events = AsyncMock()
        deps = _FakeDeps(events=events)
        ctx = _make_ctx(deps)
        cap = ToolDisplayCapability(
            rename_mode=False,
            name_map={"viking_read": "read"},
            emit_rich=True,
            emit_rich_for={"viking_read"},
        )

        await cap.wrap_tool_execute(
            ctx,
            call=_make_call("viking_read"),
            tool_def=_make_tool_def("viking_read"),
            args={"uris": ["viking://a.md"]},
            handler=_make_handler("content"),
        )

        events.tool_call_start.assert_awaited_once()
        assert deps.tool_name == "viking_read"

    @pytest.mark.asyncio
    async def test_rich_matches_across_rename_boundary(self) -> None:
        """Model calls display name 'read'; emit_rich_for={'viking_read'} still matches."""
        events = AsyncMock()
        ctx = _make_ctx(_FakeDeps(events=events))
        cap = ToolDisplayCapability(
            rename_mode=True,
            name_map={"viking_read": "read"},
            emit_rich=True,
            emit_rich_for={"viking_read"},
        )

        await cap.wrap_tool_execute(
            ctx,
            call=_make_call("read", tool_call_id="call_disp"),
            tool_def=_make_tool_def("read"),
            args={"uris": ["viking://a.md"]},
            handler=_make_handler("content"),
        )

        events.tool_call_start.assert_awaited_once()
        assert events.tool_call_start.await_args.kwargs["kind"] == "read"


# ---------------------------------------------------------------------------
# 5.5 No duplicate emission when targeted by both layers
# ---------------------------------------------------------------------------


class TestRichAndDiffIsolation:
    """A tool in both emit_rich_for and emit_diff_for gets exactly one content path."""

    @pytest.mark.asyncio
    async def test_read_tool_gets_rich_not_diff(self) -> None:
        """Read tool in both sets: rich content only, no DiffContentItem."""
        events = AsyncMock()
        ctx = _make_ctx(_FakeDeps(events=events))
        cap = ToolDisplayCapability(
            emit_rich=True,
            emit_rich_for={"viking_read"},
            emit_diff=True,
            emit_diff_for={"viking_read", "viking_write"},
        )

        await cap.wrap_tool_execute(
            ctx,
            call=_make_call("viking_read"),
            tool_def=_make_tool_def("viking_read"),
            args={"uris": ["viking://a.md"]},
            handler=_make_handler("content"),
        )

        # rich target takes precedence: exactly one tool_call_progress (rich content)
        assert events.tool_call_progress.await_count == 1
        from wolfharness.agents.events import TextContentItem

        items = events.tool_call_progress.await_args.kwargs["items"]
        assert len(items) == 1
        assert isinstance(items[0], TextContentItem)

    @pytest.mark.asyncio
    async def test_write_tool_gets_diff_not_rich(self) -> None:
        """Write tool in both sets: diff only, rich path skipped (no extractor)."""
        events = AsyncMock()
        ctx = _make_ctx(_FakeDeps(events=events))
        cap = ToolDisplayCapability(
            emit_rich=True,
            emit_rich_for={"viking_write"},
            emit_diff=True,
            emit_diff_for={"viking_write"},
        )

        await cap.wrap_tool_execute(
            ctx,
            call=_make_call("viking_write"),
            tool_def=_make_tool_def("viking_write"),
            args={"uri": "viking://x.md", "content": "abc"},
            handler=_make_handler("Wrote 3 chars."),
        )

        assert events.tool_call_progress.await_count == 1
        from wolfharness.agents.events import DiffContentItem

        items = events.tool_call_progress.await_args.kwargs["items"]
        assert len(items) == 1
        assert isinstance(items[0], DiffContentItem)


# ---------------------------------------------------------------------------
# 5.9 Post-title must not override pre-execution title (regression)
# ---------------------------------------------------------------------------


class TestRichPostTitlePreserved:
    """Post-execution progress must not clobber the pre-execution title.

    viking_search/glob share the read extractor; a hardcoded post title
    ("Read") would replace the correct "Search for '<query>'" in protocol
    clients. The post event must either carry the same title as the start
    event or omit it (letting clients fall back to the existing title).
    """

    @pytest.mark.asyncio
    async def test_search_post_title_does_not_override_start(self) -> None:
        """viking_search: post progress carries the pre-execution title."""
        events = AsyncMock()
        ctx = _make_ctx(_FakeDeps(events=events))
        cap = ToolDisplayCapability(
            emit_rich=True,
            emit_rich_for={"viking_search"},
        )

        await cap.wrap_tool_execute(
            ctx,
            call=_make_call("viking_search", tool_call_id="call_s"),
            tool_def=_make_tool_def("viking_search"),
            args={"query": "hydraulic pump"},
            handler=_make_handler("match 1\nmatch 2"),
        )

        # start event carries the search title
        events.tool_call_start.assert_awaited_once()
        start_title = events.tool_call_start.await_args.kwargs["title"]
        assert "hydraulic pump" in start_title, f"start title should mention query: {start_title}"

        # post progress must not replace it with a bare "Read"
        events.tool_call_progress.assert_awaited_once()
        post_kwargs = events.tool_call_progress.await_args.kwargs
        post_title = post_kwargs.get("title")
        if post_title is not None:
            assert post_title == start_title, (
                "post title must equal start title or be absent, "
                f"got: {post_title!r} vs {start_title!r}"
            )

    @pytest.mark.asyncio
    async def test_glob_post_title_uses_uri_location(self) -> None:
        """viking_glob: post title matches the start title (pattern-based)."""
        events = AsyncMock()
        ctx = _make_ctx(_FakeDeps(events=events))
        cap = ToolDisplayCapability(
            emit_rich=True,
            emit_rich_for={"viking_glob"},
        )

        await cap.wrap_tool_execute(
            ctx,
            call=_make_call("viking_glob", tool_call_id="call_g"),
            tool_def=_make_tool_def("viking_glob"),
            args={"pattern": "**/*.md", "uri": "viking://wiki"},
            handler=_make_handler("viking://wiki/a.md\nviking://wiki/b.md"),
        )

        events.tool_call_start.assert_awaited_once()
        start_title = events.tool_call_start.await_args.kwargs["title"]
        assert "**/*.md" in start_title, f"glob start title should mention pattern: {start_title}"

        events.tool_call_progress.assert_awaited_once()
        post_title = events.tool_call_progress.await_args.kwargs["title"]
        assert post_title == start_title, (
            f"post title must equal start title, got: {post_title!r} vs {start_title!r}"
        )


# ---------------------------------------------------------------------------
# 5.10 target_uri recognized as a location (search subtree restriction)
# ---------------------------------------------------------------------------


class TestSearchTargetUriLocation:
    """``target_uri`` on search/find tools surfaces as a file location."""

    def test_parse_locations_recognizes_target_uri(self) -> None:
        assert _parse_locations({"target_uri": "viking://wiki/"}) == ["viking://wiki/"]
        assert _parse_locations({"query": "x", "target_uri": "viking://wiki/"}) == [
            "viking://wiki/"
        ]
