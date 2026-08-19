"""Tests for ACP prompt metadata helpers."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from acp.schema import (
    AgentThoughtChunk,
    ToolCallProgress,
    ToolCallStart,
    TurnCompleteUpdate,
    UserMessageChunk,
)
from wolfharness_server.acp_server.handler import _content_items_to_text
from wolfharness_server.acp_server.viking_archive import ACPVikingEventArchive


def test_content_items_to_text_keeps_text_prompt_for_capability_metadata() -> None:
    text = _content_items_to_text([
        "/systematic-troubleshooting cy215c 动臂吊臂",
        SimpleNamespace(text="补充: 冷车更明显"),
        SimpleNamespace(uri="file:///tmp/image.png"),
    ])

    assert text == "/systematic-troubleshooting cy215c 动臂吊臂\n补充: 冷车更明显"


@pytest.mark.asyncio
async def test_viking_archive_serializes_acp_user_thought_and_tool_updates() -> None:
    archive = ACPVikingEventArchive(enabled=True, batch_size=100)

    await archive.record_update(
        consumer_session_id="acp-session-1",
        source_session_id="acp-session-1",
        event_id="evt-user",
        update=UserMessageChunk.text("/fta-eval report.md", message_id="msg-user"),
    )
    await archive.record_update(
        consumer_session_id="acp-session-1",
        source_session_id="acp-session-1",
        event_id="evt-think",
        update=AgentThoughtChunk.text("需要先读取报告", message_id="msg-think"),
    )
    await archive.record_update(
        consumer_session_id="acp-session-1",
        source_session_id="acp-session-1",
        event_id="evt-tool-start",
        update=ToolCallStart(
            tool_call_id="tool-1",
            title="Read report",
            status="in_progress",
            kind="read",
        ),
    )
    await archive.record_update(
        consumer_session_id="acp-session-1",
        source_session_id="acp-session-1",
        event_id="evt-tool-update",
        update=ToolCallProgress(tool_call_id="tool-1", status="completed"),
    )

    records = archive._pending["acp-session-1"]
    assert [record["event_type"] for record in records] == [
        "user_message_chunk",
        "agent_thought_chunk",
        "tool_call",
        "tool_call_update",
    ]
    assert records[0]["message_id"] == "msg-user"
    assert records[2]["tool_call_id"] == "tool-1"


@pytest.mark.asyncio
async def test_viking_archive_appends_events_and_writes_memory_context() -> None:
    client = AsyncMock()
    archive = ACPVikingEventArchive(enabled=True, session_prefix="iroot-acp-session")
    archive._client = client

    await archive.record_update(
        consumer_session_id="session/with spaces",
        source_session_id="session/with spaces",
        event_id=123,
        update=UserMessageChunk.text("/systematic-troubleshooting cy215c 动臂吊臂"),
    )
    await archive.flush_session("session/with spaces")

    assert client.write.call_count == 2
    events_call = client.write.call_args_list[0]
    context_call = client.write.call_args_list[1]
    assert events_call.args[0] == "viking://sessions/iroot-acp-session-session-with-spaces/events.jsonl"
    event_record = json.loads(events_call.args[1].splitlines()[0])
    assert event_record["event_id"] == "123"
    assert event_record["event_type"] == "user_message_chunk"
    assert events_call.kwargs["mode"] == "append"
    assert events_call.kwargs["processing_mode"] == "raw"
    assert context_call.args[0] == (
        "viking://sessions/iroot-acp-session-session-with-spaces/memory_context.json"
    )
    assert context_call.kwargs["mode"] == "replace"


@pytest.mark.asyncio
async def test_viking_archive_flush_failure_is_non_blocking() -> None:
    client = AsyncMock()
    client.write = AsyncMock(side_effect=RuntimeError("viking unavailable"))
    archive = ACPVikingEventArchive(enabled=True)
    archive._client = client

    await archive.record_update(
        consumer_session_id="acp-session-1",
        source_session_id="acp-session-1",
        event_id="evt-user",
        update=UserMessageChunk.text("普通 prompt"),
    )

    await archive.flush_session("acp-session-1")

    client.write.assert_awaited_once()


@pytest.mark.asyncio
async def test_viking_archive_turn_complete_flushes_pending_events() -> None:
    client = AsyncMock()
    archive = ACPVikingEventArchive(enabled=True, flush_on_turn_complete=True)
    archive._client = client

    await archive.record_update(
        consumer_session_id="acp-session-1",
        source_session_id="acp-session-1",
        event_id="evt-done",
        update=TurnCompleteUpdate(stop_reason="end_turn"),
    )
    await archive.flush_all()

    assert "acp-session-1" not in archive._pending
    assert client.write.call_count == 2
