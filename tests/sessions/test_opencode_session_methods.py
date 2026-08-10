"""Tests for OpenCodeStorageProvider session methods.

Covers:
- update_session_title, update_sdk_session_id
- delete_session_messages, get_filtered_conversations
"""

from __future__ import annotations

from datetime import UTC, datetime
import tempfile
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from pathlib import Path

import anyenv
import pytest

from wolfharness_config.storage import OpenCodeStorageConfig
from wolfharness_server.opencode_server.models import Session, TimeCreatedUpdated
from wolfharness_storage.opencode_provider import OpenCodeStorageProvider
from wolfharness_storage.opencode_provider.helpers import compute_project_id


pytestmark = pytest.mark.integration


@pytest.fixture
async def provider():
    """Create an OpenCode provider with temp directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = OpenCodeStorageConfig(path=tmpdir)
        prov = OpenCodeStorageProvider(config)
        async with prov:
            yield prov


def _write_session_json(
    provider: OpenCodeStorageProvider,
    session_id: str,
    *,
    title: str = "Test Session",
    project_id: str | None = None,
) -> Path:
    """Write a session JSON file directly to the provider's sessions_path."""
    pid = project_id or compute_project_id(str(provider.base_path))
    project_dir = provider.sessions_path / pid
    project_dir.mkdir(parents=True, exist_ok=True)

    now_ms = int(datetime.now(UTC).timestamp() * 1000)
    session = Session(
        id=session_id,
        project_id=pid,
        directory=str(provider.base_path),
        title=title,
        time=TimeCreatedUpdated(created=now_ms, updated=now_ms),
    )
    session_path = project_dir / f"{session_id}.json"
    dct = session.model_dump(by_alias=True)
    session_path.write_text(anyenv.dump_json(dct, indent=True), encoding="utf-8")
    return session_path


def _write_message_json(
    provider: OpenCodeStorageProvider,
    session_id: str,
    message_id: str,
    role: str = "user",
) -> Path:
    """Write a minimal message JSON file and return its path."""
    msg_dir = provider.messages_path / session_id
    msg_dir.mkdir(parents=True, exist_ok=True)

    now_ms = int(datetime.now(UTC).timestamp() * 1000)
    data = {
        "id": message_id,
        "sessionID": session_id,
        "role": role,
        "time": {"created": now_ms},
    }
    msg_path = msg_dir / f"{message_id}.json"
    msg_path.write_text(anyenv.dump_json(data, indent=True), encoding="utf-8")
    return msg_path


def _write_part_json(
    provider: OpenCodeStorageProvider,
    message_id: str,
    part_id: str,
) -> Path:
    """Write a minimal part JSON file and return its path."""
    parts_dir = provider.parts_path / message_id
    parts_dir.mkdir(parents=True, exist_ok=True)

    now_ms = int(datetime.now(UTC).timestamp() * 1000)
    data = {
        "id": part_id,
        "messageID": message_id,
        "type": "text",
        "text": "hello",
        "time": {"start": now_ms},
    }
    part_path = parts_dir / f"{part_id}.json"
    part_path.write_text(anyenv.dump_json(data, indent=True), encoding="utf-8")
    return part_path


# --- update_session_title ---


async def test_update_session_title_persists(provider: OpenCodeStorageProvider) -> None:
    """Test that update_session_title modifies the session JSON on disk."""
    session_path = _write_session_json(provider, "sess_title_001", title="Old Title")

    await provider.update_session_title("sess_title_001", "New Title")

    # Read raw JSON from disk to verify title was updated
    content = session_path.read_text(encoding="utf-8")
    data = anyenv.load_json(content, return_type=dict)
    assert data["title"] == "New Title"


async def test_update_session_title_not_found(provider: OpenCodeStorageProvider) -> None:
    """Test that updating a nonexistent session logs a warning but does not raise."""
    # Should not raise
    await provider.update_session_title("nonexistent_session", "Title")


# --- update_sdk_session_id ---


async def test_update_sdk_session_id_persists(provider: OpenCodeStorageProvider) -> None:
    """Test that update_sdk_session_id adds metadata.sdk_session_id to the JSON."""
    session_path = _write_session_json(provider, "sess_sdk_001")

    await provider.update_sdk_session_id("sess_sdk_001", "sdk-sess-abc123")

    # Read raw JSON to verify metadata.sdk_session_id was written
    content = session_path.read_text(encoding="utf-8")
    data = anyenv.load_json(content, return_type=dict)
    assert data["metadata"]["sdk_session_id"] == "sdk-sess-abc123"


async def test_update_sdk_session_id_not_found(provider: OpenCodeStorageProvider) -> None:
    """Test that updating a nonexistent session logs a warning but does not raise."""
    await provider.update_sdk_session_id("nonexistent_session", "sdk-id")


# --- delete_session_messages ---


async def test_delete_session_messages_removes_files(provider: OpenCodeStorageProvider) -> None:
    """Test that delete_session_messages removes message JSON and part JSON files."""
    _write_session_json(provider, "sess_del_001")
    _write_message_json(provider, "sess_del_001", "msg_001")
    _write_message_json(provider, "sess_del_001", "msg_002")
    _write_part_json(provider, "msg_001", "msg_001-0")
    _write_part_json(provider, "msg_002", "msg_002-0")

    # Verify files exist
    assert (provider.messages_path / "sess_del_001" / "msg_001.json").exists()
    assert (provider.messages_path / "sess_del_001" / "msg_002.json").exists()
    assert (provider.parts_path / "msg_001" / "msg_001-0.json").exists()
    assert (provider.parts_path / "msg_002" / "msg_002-0.json").exists()

    count = await provider.delete_session_messages("sess_del_001")

    assert count == 2
    # Message files should be gone
    assert not (provider.messages_path / "sess_del_001" / "msg_001.json").exists()
    assert not (provider.messages_path / "sess_del_001" / "msg_002.json").exists()
    # Part files should be gone
    assert not (provider.parts_path / "msg_001" / "msg_001-0.json").exists()
    assert not (provider.parts_path / "msg_002" / "msg_002-0.json").exists()


async def test_delete_session_messages_nonexistent(provider: OpenCodeStorageProvider) -> None:
    """Test that deleting messages for a nonexistent session returns 0 gracefully."""
    count = await provider.delete_session_messages("nonexistent_session")
    assert count == 0


# --- get_filtered_conversations ---


async def test_get_filtered_conversations_by_name(provider: OpenCodeStorageProvider) -> None:
    """Test that filtering by agent_name only returns matching sessions."""
    # Create two sessions with messages
    _write_session_json(provider, "sess_filter_001", title="Session Alpha")
    _write_session_json(provider, "sess_filter_002", title="Session Beta")

    # Create messages with different roles (agent_name comes from message name)
    msg_dir_1 = provider.messages_path / "sess_filter_001"
    msg_dir_1.mkdir(parents=True, exist_ok=True)
    now_ms = int(datetime.now(UTC).timestamp() * 1000)

    # Message from "agent_alpha"
    alpha_msg = {
        "id": "msg_alpha",
        "sessionID": "sess_filter_001",
        "role": "assistant",
        "parentID": "",
        "modelID": "gpt-4o",
        "providerID": "",
        "path": {"cwd": "", "root": ""},
        "time": {"created": now_ms},
        "tokens": {"input": 10, "output": 20, "cache": {"read": 0, "write": 0}},
        "cost": 0.01,
        "finish": "stop",
    }
    (msg_dir_1 / "msg_alpha.json").write_text(
        anyenv.dump_json(alpha_msg, indent=True), encoding="utf-8"
    )

    # Part for alpha message
    parts_dir = provider.parts_path / "msg_alpha"
    parts_dir.mkdir(parents=True, exist_ok=True)
    alpha_part = {
        "id": "msg_alpha-0",
        "messageID": "msg_alpha",
        "sessionID": "sess_filter_001",
        "type": "text",
        "text": "Hello from alpha",
        "time": {"start": now_ms},
    }
    (parts_dir / "msg_alpha-0.json").write_text(
        anyenv.dump_json(alpha_part, indent=True), encoding="utf-8"
    )

    # Second session with "agent_beta"
    msg_dir_2 = provider.messages_path / "sess_filter_002"
    msg_dir_2.mkdir(parents=True, exist_ok=True)

    beta_msg = {
        "id": "msg_beta",
        "sessionID": "sess_filter_002",
        "role": "assistant",
        "parentID": "",
        "modelID": "claude-sonnet",
        "providerID": "",
        "path": {"cwd": "", "root": ""},
        "time": {"created": now_ms},
        "tokens": {"input": 15, "output": 25, "cache": {"read": 0, "write": 0}},
        "cost": 0.02,
        "finish": "stop",
    }
    (msg_dir_2 / "msg_beta.json").write_text(
        anyenv.dump_json(beta_msg, indent=True), encoding="utf-8"
    )

    beta_parts_dir = provider.parts_path / "msg_beta"
    beta_parts_dir.mkdir(parents=True, exist_ok=True)
    beta_part = {
        "id": "msg_beta-0",
        "messageID": "msg_beta",
        "sessionID": "sess_filter_002",
        "type": "text",
        "text": "Hello from beta",
        "time": {"start": now_ms},
    }
    (beta_parts_dir / "msg_beta-0.json").write_text(
        anyenv.dump_json(beta_part, indent=True), encoding="utf-8"
    )

    # Filter by content query - only alpha matches
    results = await provider.get_filtered_conversations(query="alpha")
    assert len(results) == 1
    assert results[0]["id"] == "sess_filter_001"

    # No filter - should return both
    all_results = await provider.get_filtered_conversations()
    assert len(all_results) == 2


async def test_get_filtered_conversations_empty(provider: OpenCodeStorageProvider) -> None:
    """Test that no sessions returns an empty list."""
    results = await provider.get_filtered_conversations()
    assert results == []
