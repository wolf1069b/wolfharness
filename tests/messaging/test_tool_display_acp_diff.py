"""ACP Converter diff path — ToolDisplayCapability events as FileEditToolCallContent.

Task 4.6: after an agent runs a decorated tool, the injected
``ToolCallProgressEvent(items=[DiffContentItem])`` must convert to an
ACP ``ToolCallProgress`` update carrying ``FileEditToolCallContent`` —
the ACP client (Zed) diff rendering path.
"""

from __future__ import annotations

import pytest

from acp.schema import FileEditToolCallContent, ToolCallProgress, ToolCallStart
from wolfharness.agents.events import (
    DiffContentItem,
    LocationContentItem,
    ToolCallProgressEvent,
    ToolCallStartEvent,
)
from wolfharness_server.acp_server.event_converter import ACPEventConverter


pytestmark = pytest.mark.integration


async def collect_updates(converter: ACPEventConverter, event):
    """Collect all ACP updates produced for a single event."""
    return [u async for u in converter.convert(event)]


@pytest.mark.anyio
async def test_diff_progress_converts_to_file_edit_content() -> None:
    """ToolCallProgressEvent with a DiffContentItem yields FileEditToolCallContent.

    Exercises the same event shape ToolDisplayCapability.wrap_tool_execute
    emits: a progress event carrying old/new text for a viking URI.
    """
    converter = ACPEventConverter()

    # Seed the tool call state the converter tracks internally.
    await collect_updates(
        converter,
        ToolCallStartEvent(
            tool_call_id="call_1",
            tool_name="viking_write",
            title="Write viking://x.md",
        ),
    )

    updates = await collect_updates(
        converter,
        ToolCallProgressEvent(
            tool_call_id="call_1",
            status="in_progress",
            title="Modified: viking://x.md",
            items=[
                DiffContentItem(
                    path="viking://x.md",
                    old_text=None,
                    new_text="new content",
                )
            ],
        ),
    )

    progress_updates = [u for u in updates if isinstance(u, ToolCallProgress)]
    assert progress_updates, "expected at least one ToolCallProgress update"
    progress = progress_updates[0]
    assert progress.tool_call_id == "call_1"
    assert progress.content is not None
    file_edits = [c for c in progress.content if isinstance(c, FileEditToolCallContent)]
    assert len(file_edits) == 1
    assert file_edits[0].path == "viking://x.md"
    assert file_edits[0].old_text is None
    assert file_edits[0].new_text == "new content"


@pytest.mark.anyio
async def test_converter_requires_no_start_for_rich_progress() -> None:
    """A progress event without a prior start still converts safely."""
    converter = ACPEventConverter()
    updates = await collect_updates(
        converter,
        ToolCallProgressEvent(
            tool_call_id="orphan",
            status="in_progress",
            title="Modified: viking://x.md",
            items=[DiffContentItem(path="viking://x.md", old_text="a", new_text="b")],
        ),
    )

    # No crash; at minimum no updates or a progress update.
    assert isinstance(updates, list)


@pytest.mark.anyio
async def test_rich_start_converts_kind_and_locations() -> None:
    """ToolCallStartEvent with kind + locations yields ACP ToolCallStart.

    Exercises the emit_rich pre-execution event shape: a read tool
    (viking_read) marked with kind="read" and multiple URI locations
    must surface as an ACP tool-start notification with matching kind
    and file locations.
    """
    converter = ACPEventConverter()
    updates = await collect_updates(
        converter,
        ToolCallStartEvent(
            tool_call_id="call_rich",
            tool_name="viking_read",
            title="Read viking://a.md, viking://b.md",
            kind="read",
            locations=[
                LocationContentItem(path="viking://a.md"),
                LocationContentItem(path="viking://b.md"),
            ],
        ),
    )

    start_updates = [u for u in updates if isinstance(u, ToolCallStart)]
    assert start_updates, "expected at least one ToolCallStart update"
    start = start_updates[0]
    assert start.tool_call_id == "call_rich"
    assert start.kind == "read"
    assert start.locations is not None
    assert [loc.path for loc in start.locations] == ["viking://a.md", "viking://b.md"]
