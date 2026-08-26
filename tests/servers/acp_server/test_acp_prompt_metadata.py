"""Tests for ACP prompt metadata helpers."""

from __future__ import annotations

import asyncio
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
from wolfharness_server.acp_server.viking_archive import ACPVikingEventArchive


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

    assert archive._pending_update_count["acp-session-1"] == 4
    assert len(archive._transcript_pending["acp-session-1"]) == 4
    assert archive._transcript_pending["acp-session-1"][2]["parts"][0]["tool_id"] == "tool-1"
    assert archive._transcript_pending["acp-session-1"][3]["parts"][0]["tool_status"] == "completed"


@pytest.mark.asyncio
async def test_viking_archive_writes_structured_updates_as_readable_transcript() -> None:
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

    client.write.assert_not_awaited()
    client.create_session.assert_awaited_once_with(
        session_id="iroot-acp-session-session-with-spaces",
        source_type="iroot-acp-session",
    )
    client.add_message.assert_awaited_once_with(
        "iroot-acp-session-session-with-spaces",
        "user",
        parts=[{"type": "text", "text": "/systematic-troubleshooting cy215c 动臂吊臂"}],
    )
    client.commit_session.assert_awaited_once_with("iroot-acp-session-session-with-spaces")


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
async def test_viking_archive_filters_low_signal_ack_messages() -> None:
    client = AsyncMock()
    archive = ACPVikingEventArchive(enabled=True, user="bin.chen")
    archive._client = client

    await archive.record_update(
        consumer_session_id="acp-session-1",
        source_session_id="acp-session-1",
        event_id="evt-user",
        update=UserMessageChunk.text("好的", message_id="msg-user"),
    )
    await archive.flush_session("acp-session-1")

    client.add_message.assert_not_awaited()
    client.commit_session.assert_not_awaited()


@pytest.mark.asyncio
async def test_viking_archive_flush_failure_is_non_blocking() -> None:
    client = AsyncMock()
    client.commit_session = AsyncMock(side_effect=RuntimeError("viking unavailable"))
    archive = ACPVikingEventArchive(enabled=True)
    archive._client = client

    await archive.record_update(
        consumer_session_id="acp-session-1",
        source_session_id="acp-session-1",
        event_id="evt-user",
        update=UserMessageChunk.text("普通 prompt"),
    )

    await archive.flush_session("acp-session-1")

    client.commit_session.assert_awaited_once()
    assert archive._pending_update_count["acp-session-1"] == 1
    assert len(archive._transcript_pending["acp-session-1"]) == 1


@pytest.mark.asyncio
async def test_viking_archive_resolves_user_from_health_for_user_scoped_uri() -> None:
    client = AsyncMock()
    response = SimpleNamespace(json=lambda: {"user_id": "bin.chen"})
    client._request = AsyncMock(return_value=response)

    archive = ACPVikingEventArchive(enabled=True)
    await archive._resolve_user_from_health(client)

    assert archive._archive_base_uri("session/with spaces") == (
        "viking://user/bin.chen/sessions/iroot-acp-session-session-with-spaces"
    )


def test_viking_archive_from_config_requires_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VIKING_MCP_URL", raising=False)

    archive = ACPVikingEventArchive.from_config(
        SimpleNamespace(enabled=True, api_key="test-key", url=None)
    )

    assert archive.enabled is False


def test_viking_archive_from_config_uses_env_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VIKING_MCP_URL", "https://viking.example.com")

    archive = ACPVikingEventArchive.from_config(
        SimpleNamespace(enabled=True, api_key="test-key", url=None)
    )

    assert archive.enabled is True
    assert archive.url == "https://viking.example.com"


@pytest.mark.asyncio
async def test_viking_archive_turn_complete_flushes_pending_events() -> None:
    client = AsyncMock()
    archive = ACPVikingEventArchive(enabled=True, flush_on_turn_complete=True)
    archive._client = client

    await archive.record_update(
        consumer_session_id="acp-session-1",
        source_session_id="acp-session-1",
        event_id="evt-user",
        update=UserMessageChunk.text("普通 prompt"),
    )
    await archive.record_update(
        consumer_session_id="acp-session-1",
        source_session_id="acp-session-1",
        event_id="evt-done",
        update=TurnCompleteUpdate(stop_reason="end_turn"),
    )
    await _drain_archive_tasks(archive)

    assert "acp-session-1" not in archive._pending_update_count
    client.add_message.assert_awaited_once()
    client.commit_session.assert_awaited_once()
