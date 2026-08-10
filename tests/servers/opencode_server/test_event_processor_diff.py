"""Tests for DiffContentItem → unified diff text in opencode event_processor.

The opencode TUI Edit component reads ``metadata.diff`` as a unified diff
text string (produced by ``createTwoFilesPatch`` in opencode's edit.ts).
The wolfharness event_processor must convert ``DiffContentItem`` from
``ToolCallProgressEvent`` into that text format and carry it through to
``ToolStateCompleted.metadata.diff`` so the TUI can render the diff view.
"""

from __future__ import annotations

import pytest

from wolfharness.agents.events import DiffContentItem, ToolCallProgressEvent
from wolfharness.agents.events.events import ToolCallCompleteEvent, ToolCallStartEvent
from wolfharness_server.opencode_server.event_processor import EventProcessor
from wolfharness_server.opencode_server.event_processor_context import (
    EventProcessorContext,
)
from wolfharness_server.opencode_server.models.message import (
    MessagePath,
    MessageTime,
    MessageWithParts,
)


def _make_ctx(server_state: object) -> EventProcessorContext:
    """Create a minimal EventProcessorContext for tool diff tests."""
    assistant_msg = MessageWithParts.assistant(
        message_id="msg_001",
        session_id="test-diff-session",
        time=MessageTime(created=0),
        agent_name="test-agent",
        model_id="test-model",
        parent_id="parent-1",
        provider_id="wolfharness",
        path=MessagePath(cwd="/tmp", root="/tmp"),
    )
    return EventProcessorContext(
        session_id="test-diff-session",
        assistant_msg_id="msg_001",
        assistant_msg=assistant_msg,
        state=server_state,  # type: ignore[arg-type]
        working_dir="/tmp",
    )


class TestDiffContentItemToUnifiedDiff:
    """DiffContentItem in ToolCallProgressEvent must produce metadata.diff."""

    @pytest.mark.asyncio
    async def test_write_diff_carries_to_complete(
        self,
        server_state: object,
    ) -> None:
        """DiffContentItem(old=None, new=content) → metadata.diff unified text.

        Simulates viking_write: tool start → progress with diff → complete.
        The final ToolPart.state.metadata.diff must be a unified diff string.
        """
        processor = EventProcessor()
        ctx = _make_ctx(server_state)

        async def _feed(event: object) -> None:
            async for _ in processor.process(event, ctx):  # type: ignore[arg-type]
                pass

        # Step 1: start
        await _feed(
            ToolCallStartEvent(
                tool_call_id="call_diff_001",
                tool_name="write",
                raw_input={"file_path": "viking://test.md", "content": "new content"},
                title="Writing viking://test.md",
            )
        )

        # Step 2: progress with DiffContentItem
        await _feed(
            ToolCallProgressEvent(
                tool_call_id="call_diff_001",
                tool_name="write",
                title="Modified: viking://test.md",
                items=[
                    DiffContentItem(
                        path="viking://test.md",
                        old_text=None,
                        new_text="new content",
                    )
                ],
            )
        )

        # Step 3: complete
        await _feed(
            ToolCallCompleteEvent(
                tool_call_id="call_diff_001",
                tool_name="write",
                tool_input={"file_path": "viking://test.md", "content": "new content"},
                tool_result="Written successfully.",
                agent_name="test-agent",
                message_id="msg_001",
            )
        )

        # Assert: final ToolPart has metadata.diff as a unified diff text string
        tool_part = ctx.get_tool_part("call_diff_001")
        assert tool_part is not None, "ToolPart should exist after complete"
        assert tool_part.state is not None
        metadata = getattr(tool_part.state, "metadata", None)
        assert metadata is not None, "metadata must be set on completed tool part"
        diff = metadata.get("diff")
        assert diff is not None, "metadata.diff must be present"
        assert isinstance(diff, str), f"metadata.diff must be a string, got {type(diff)}"
        assert "viking://test.md" in diff
        assert "new content" in diff
        # diagnostics=[] triggers Write component's code-block branch
        assert metadata.get("diagnostics") == []

    @pytest.mark.asyncio
    async def test_edit_diff_carries_old_and_new(
        self,
        server_state: object,
    ) -> None:
        """DiffContentItem(old_text, new_text) → metadata.diff with both sides."""
        processor = EventProcessor()
        ctx = _make_ctx(server_state)

        async def _feed(event: object) -> None:
            async for _ in processor.process(event, ctx):  # type: ignore[arg-type]
                pass

        await _feed(
            ToolCallStartEvent(
                tool_call_id="call_diff_002",
                tool_name="edit",
                raw_input={
                    "file_path": "viking://doc.md",
                    "old_string": "old",
                    "new_string": "new",
                },
                title="Editing viking://doc.md",
            )
        )

        await _feed(
            ToolCallProgressEvent(
                tool_call_id="call_diff_002",
                tool_name="edit",
                title="Modified: viking://doc.md",
                items=[
                    DiffContentItem(
                        path="viking://doc.md",
                        old_text="old line",
                        new_text="new line",
                    )
                ],
            )
        )

        await _feed(
            ToolCallCompleteEvent(
                tool_call_id="call_diff_002",
                tool_name="edit",
                tool_input={"file_path": "viking://doc.md"},
                tool_result="Edited.",
                agent_name="test-agent",
                message_id="msg_001",
            )
        )

        tool_part = ctx.get_tool_part("call_diff_002")
        assert tool_part is not None
        metadata = getattr(tool_part.state, "metadata", None)
        assert metadata is not None
        diff = metadata.get("diff")
        assert diff is not None
        assert isinstance(diff, str)
        assert "old line" in diff
        assert "new line" in diff
        # Unified diff should have removal and addition markers
        assert "-old line" in diff or "-old" in diff
        assert "+new line" in diff or "+new" in diff

    @pytest.mark.asyncio
    async def test_diff_output_is_parseable_unified_diff(
        self,
        server_state: object,
    ) -> None:
        r"""Diff output must be parseable by strict unified diff parsers.

        The npm ``diff`` package's ``parsePatch`` (used by opencode TUI)
        strictly validates hunk line counts against ``@@ -X,Y +A,B @@``
        headers. Missing ``\\n`` on the last content line causes
        "Added line count did not match for hunk".

        Requirements:
        - Every line properly ``\\n``-terminated (diff ends with ``\\n``)
        - ``---`` and ``+++`` headers both use the file path (no ``(old)`` suffix)
        """
        processor = EventProcessor()
        ctx = _make_ctx(server_state)

        async def _feed(event: object) -> None:
            async for _ in processor.process(event, ctx):  # type: ignore[arg-type]
                pass

        await _feed(
            ToolCallStartEvent(
                tool_call_id="call_diff_003",
                tool_name="edit",
                raw_input={"file_path": "viking://note.md"},
                title="Editing viking://note.md",
            )
        )
        await _feed(
            ToolCallProgressEvent(
                tool_call_id="call_diff_003",
                tool_name="edit",
                title="Modified: viking://note.md",
                items=[
                    DiffContentItem(
                        path="viking://note.md",
                        old_text="old line",  # No trailing \n — triggers the bug
                        new_text="new line",  # No trailing \n
                    )
                ],
            )
        )
        await _feed(
            ToolCallCompleteEvent(
                tool_call_id="call_diff_003",
                tool_name="edit",
                tool_input={"file_path": "viking://note.md"},
                tool_result="Edited.",
                agent_name="test-agent",
                message_id="msg_001",
            )
        )

        tool_part = ctx.get_tool_part("call_diff_003")
        assert tool_part is not None
        metadata = getattr(tool_part.state, "metadata", None)
        assert metadata is not None
        diff = metadata.get("diff")
        assert diff is not None
        assert isinstance(diff, str)

        # Bug 1: diff must end with \n (no dangling last line)
        assert diff.endswith("\n"), f"Diff must end with \\n, got tail: {diff[-20:]!r}"

        # Bug 2: fromfile/tofile must both be the path (no "(old)" suffix)
        assert "(old)" not in diff, f"Diff must not contain '(old)' suffix: {diff!r}"
        assert "--- viking://note.md\n" in diff
        assert "+++ viking://note.md\n" in diff

        # Bug 3: hunk line counts must match — verify by parsing
        # Every non-header line in a hunk must start with ' ', '-', or '+'
        hunk_lines = [
            line
            for line in diff.split("\n")
            if line
            and not line.startswith("@@")
            and not line.startswith("---")
            and not line.startswith("+++")
        ]
        for line in hunk_lines:
            assert line[0] in (" ", "-", "+"), f"Invalid hunk line prefix: {line!r}"


class TestRichProgressCarriesContent:
    """TextContentItem in ToolCallProgressEvent surfaces as tool output.

    The emit_rich layer wraps viking read/search results in
    ``TextContentItem``; the event_processor must append its text to the
    tool's accumulated output so the OpenCode TUI renders the content.
    """

    @pytest.mark.asyncio
    async def test_text_content_becomes_tool_output(
        self,
        server_state: object,
    ) -> None:
        """TextContentItem → appended to ctx.tool_outputs for the tool part."""
        from wolfharness.agents.events import TextContentItem

        processor = EventProcessor()
        ctx = _make_ctx(server_state)

        async def _feed(event: object) -> None:
            async for _ in processor.process(event, ctx):  # type: ignore[arg-type]
                pass

        # Step 1: start (mirrors emit_rich pre-execution event)
        await _feed(
            ToolCallStartEvent(
                tool_call_id="call_rich_001",
                tool_name="viking_read",
                raw_input={"uris": ["viking://test.md"]},
                title="Read viking://test.md",
                kind="read",
            )
        )

        # Step 2: rich progress with content
        await _feed(
            ToolCallProgressEvent(
                tool_call_id="call_rich_001",
                tool_name="viking_read",
                title="Read viking://test.md",
                items=[TextContentItem(text="line 1\nline 2")],
            )
        )

        # Step 3: complete
        await _feed(
            ToolCallCompleteEvent(
                tool_call_id="call_rich_001",
                tool_name="viking_read",
                tool_input={"uris": ["viking://test.md"]},
                tool_result="line 1\nline 2",
                agent_name="test-agent",
                message_id="msg_001",
            )
        )

        output = ctx.get_tool_output("call_rich_001")
        assert "line 1\nline 2" in output, f"Content not in tool output: {output!r}"
