"""Tests for ACP prompt metadata helpers."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from acp.schema import (
    AgentMessageChunk,
    AgentThoughtChunk,
    TextContentBlock,
    ToolCallProgress,
    ToolCallStart,
    TurnCompleteUpdate,
    UserMessageChunk,
)
from wolfharness_server.acp_server.handler import _content_items_to_text
from wolfharness_server.acp_server.viking_archive import (
    ACPVikingEventArchive,
    ACPVikingProtocolObserver,
)


async def _drain_archive_tasks(archive: ACPVikingEventArchive) -> None:
    for _ in range(5):
        if not archive._tasks:
            return
        await asyncio.sleep(0)


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
    client.commit_session = AsyncMock(return_value={"archive_uri": "viking://archive/001"})
    archive = ACPVikingEventArchive(
        enabled=True,
        session_prefix="iroot-acp-session",
        user="bin.chen",
    )
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
    assert events_call.args[0] == (
        "viking://user/bin.chen/sessions/iroot-acp-session-session-with-spaces/events.jsonl"
    )
    event_record = json.loads(events_call.args[1].splitlines()[0])
    assert event_record["event_id"] == "123"
    assert event_record["event_type"] == "user_message_chunk"
    assert events_call.kwargs["mode"] == "append"
    assert events_call.kwargs["processing_mode"] == "raw"
    assert context_call.args[0] == (
        "viking://user/bin.chen/sessions/iroot-acp-session-session-with-spaces/memory_context.json"
    )
    assert context_call.kwargs["mode"] == "replace"
    assert context_call.kwargs["processing_mode"] == "raw"
    client.create_session.assert_awaited_once_with(
        session_id="iroot-acp-session-session-with-spaces",
        source_type="iroot-acp-session",
    )
    client.add_message.assert_awaited_once_with(
        "iroot-acp-session-session-with-spaces",
        "user",
        parts=[{"type": "text", "text": "/systematic-troubleshooting cy215c 动臂吊臂"}],
    )
    client.commit_session.assert_awaited_once_with(
        "iroot-acp-session-session-with-spaces",
        keep_recent_count=10,
    )


@pytest.mark.asyncio
async def test_viking_archive_writes_readable_transcript_with_merged_chunks() -> None:
    client = AsyncMock()
    client.commit_session = AsyncMock(return_value={"archive_uri": "viking://archive/001"})
    archive = ACPVikingEventArchive(enabled=True, user="bin.chen")
    archive._client = client

    await archive.record_update(
        consumer_session_id="acp-session-1",
        source_session_id="acp-session-1",
        event_id="evt-user-1",
        update=UserMessageChunk.text("发动机", message_id="msg-user"),
    )
    await archive.record_update(
        consumer_session_id="acp-session-1",
        source_session_id="acp-session-1",
        event_id="evt-user-2",
        update=UserMessageChunk.text("无法启动", message_id="msg-user"),
    )
    await archive.record_update(
        consumer_session_id="acp-session-1",
        source_session_id="acp-session-1",
        event_id="evt-assistant-1",
        update=AgentMessageChunk(
            content=TextContentBlock(text="先检查"),
            message_id="msg-agent",
        ),
    )
    await archive.record_update(
        consumer_session_id="acp-session-1",
        source_session_id="acp-session-1",
        event_id="evt-assistant-2",
        update=AgentMessageChunk(
            content=TextContentBlock(text="蓄电池电压"),
            message_id="msg-agent",
        ),
    )

    await archive.flush_session("acp-session-1")

    assert client.add_message.await_count == 2
    assert client.add_message.await_args_list[0].args == (
        "iroot-acp-session-acp-session-1",
        "user",
    )
    assert client.add_message.await_args_list[0].kwargs["parts"] == [
        {"type": "text", "text": "发动机无法启动"}
    ]
    assert client.add_message.await_args_list[1].args == (
        "iroot-acp-session-acp-session-1",
        "assistant",
    )
    assert client.add_message.await_args_list[1].kwargs["parts"] == [
        {"type": "text", "text": "先检查蓄电池电压"}
    ]


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
    assert len(archive._pending["acp-session-1"]) == 1


@pytest.mark.asyncio
async def test_viking_archive_resolves_user_from_health_for_user_scoped_uri() -> None:
    client = AsyncMock()
    response = SimpleNamespace(json=lambda: {"user_id": "bin.chen"})
    client._request = AsyncMock(return_value=response)

    archive = ACPVikingEventArchive(enabled=True)
    await archive._resolve_user_from_health(client)

    assert archive._events_uri("session/with spaces") == (
        "viking://user/bin.chen/sessions/iroot-acp-session-session-with-spaces/events.jsonl"
    )


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
    await _drain_archive_tasks(archive)

    assert "acp-session-1" not in archive._pending
    assert client.write.call_count == 2


@pytest.mark.asyncio
async def test_viking_protocol_observer_records_raw_frontend_prompt_and_response() -> None:
    client = AsyncMock()
    archive = ACPVikingEventArchive(enabled=True, batch_size=100)
    archive._client = client
    observer = ACPVikingProtocolObserver(archive)

    observer(
        SimpleNamespace(
            direction="incoming",
            message={
                "jsonrpc": "2.0",
                "id": 7,
                "method": "session/prompt",
                "params": {
                    "sessionId": "ses-front-1",
                    "prompt": [{"type": "text", "text": "/fta-eval report.md"}],
                },
            },
        )
    )
    observer(
        SimpleNamespace(
            direction="outgoing",
            message={
                "jsonrpc": "2.0",
                "id": 7,
                "result": {"stopReason": "end_turn"},
            },
        )
    )
    await _drain_archive_tasks(archive)

    assert "ses-front-1" not in archive._pending
    assert client.write.call_count == 2
    records = [json.loads(line) for line in client.write.call_args_list[0].args[1].splitlines()]
    assert [record["event_type"] for record in records] == ["rpc_request", "rpc_response"]
    assert records[0]["direction"] == "incoming"
    assert records[0]["method"] == "session/prompt"
    assert records[0]["protocol"]["params"]["prompt"][0]["text"] == "/fta-eval report.md"
    assert records[1]["direction"] == "outgoing"
    assert records[1]["method"] == "session/prompt"


@pytest.mark.asyncio
async def test_viking_protocol_observer_flushes_prompt_error_response() -> None:
    client = AsyncMock()
    archive = ACPVikingEventArchive(enabled=True, batch_size=100)
    archive._client = client
    observer = ACPVikingProtocolObserver(archive)

    observer(
        SimpleNamespace(
            direction="incoming",
            message={
                "jsonrpc": "2.0",
                "id": 8,
                "method": "session/prompt",
                "params": {
                    "sessionId": "ses-front-error",
                    "prompt": [{"type": "text", "text": "测试 prompt"}],
                },
            },
        )
    )
    observer(
        SimpleNamespace(
            direction="outgoing",
            message={
                "jsonrpc": "2.0",
                "id": 8,
                "error": {
                    "code": -32603,
                    "message": "Internal error",
                    "data": {"details": "prompt_cache_retention is not supported on this model"},
                },
            },
        )
    )
    await _drain_archive_tasks(archive)

    assert "ses-front-error" not in archive._pending
    assert client.write.call_count == 2
    event_record = json.loads(client.write.call_args_list[0].args[1].splitlines()[-1])
    assert event_record["event_type"] == "rpc_error"
    assert event_record["method"] == "session/prompt"
    assert "prompt_cache_retention" in event_record["protocol"]["error"]["data"]["details"]


@pytest.mark.asyncio
async def test_viking_protocol_observer_records_client_bound_elicitation_response() -> None:
    client = AsyncMock()
    archive = ACPVikingEventArchive(enabled=True, batch_size=100)
    archive._client = client
    observer = ACPVikingProtocolObserver(archive)

    observer(
        SimpleNamespace(
            direction="outgoing",
            message={
                "jsonrpc": "2.0",
                "id": 3,
                "method": "elicitation/create",
                "params": {
                    "sessionId": "ses-front-2",
                    "message": "请选择是否继续检查",
                },
            },
        )
    )
    observer(
        SimpleNamespace(
            direction="incoming",
            message={
                "jsonrpc": "2.0",
                "id": 3,
                "result": {"action": "accept", "content": {"confirmed": True}},
            },
        )
    )
    await _drain_archive_tasks(archive)

    assert "ses-front-2" not in archive._pending
    assert client.write.call_count == 2
    records = [json.loads(line) for line in client.write.call_args_list[0].args[1].splitlines()]
    assert [record["event_type"] for record in records] == ["rpc_request", "rpc_response"]
    assert records[0]["direction"] == "outgoing"
    assert records[0]["method"] == "elicitation/create"
    assert records[1]["direction"] == "incoming"
    assert records[1]["protocol"]["result"]["content"]["confirmed"] is True
