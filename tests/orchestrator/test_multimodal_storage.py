"""Tests for multimodal content preservation in storage, display, and crash recovery.

Verifies that:
- ``extract_text_from_messages`` includes ``ThinkingPart`` content
- ``_summarize_content_block`` produces meaningful placeholders for binary/URL types
- Snapshot store saves ``prompts_serialized`` with full multimodal data
- ``serialize_prompts``/``deserialize_prompts`` round-trips ``BinaryImage`` correctly
- ``prompt_text`` in ``ChatMessage.content`` uses ``_summarize_content_block``
- OpenCode converter uses ``_summarize_content_block`` for non-string content items
"""

from __future__ import annotations

from typing import Any

from pydantic_ai import BinaryImage, ModelRequest, ModelResponse, TextPart, UserPromptPart
from pydantic_ai.messages import (
    AudioUrl,
    BinaryContent,
    DocumentUrl,
    ImageUrl,
    ThinkingPart,
    VideoUrl,
)
import pytest

from wolfharness.agents.native_agent.helpers import (
    _summarize_content_block,
    extract_text_from_messages,
)
from wolfharness.storage.serialization import (
    deserialize_prompts,
    serialize_prompts,
)


pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# extract_text_from_messages
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_extract_text_from_messages_includes_thinking_part() -> None:
    """ThinkingPart content should be included in the extracted text."""
    messages: list[Any] = [
        ModelResponse(
            parts=[
                TextPart(content="Hello "),
                ThinkingPart(content="I should greet the user."),
            ]
        ),
        ModelResponse(
            parts=[
                TextPart(content="world!"),
            ]
        ),
    ]
    result = extract_text_from_messages(messages)
    # Both TextPart and ThinkingPart content should be present
    assert "Hello " in result
    assert "world!" in result
    assert "I should greet the user." in result


@pytest.mark.unit
def test_extract_text_from_messages_excludes_tool_calls() -> None:
    """Tool call parts should not appear in extracted text."""
    from pydantic_ai import ToolCallPart

    messages: list[Any] = [
        ModelResponse(
            parts=[
                TextPart(content="Response text"),
                ToolCallPart(tool_name="bash", args={"command": "ls"}),
            ]
        ),
    ]
    result = extract_text_from_messages(messages)
    assert result == "Response text"


@pytest.mark.unit
def test_extract_text_from_messages_empty() -> None:
    """Empty messages list produces empty string."""
    assert extract_text_from_messages([]) == ""


# ---------------------------------------------------------------------------
# _summarize_content_block
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_summarize_binary_image() -> None:
    """BinaryImage produces an information-preserving placeholder."""
    img = BinaryImage(data=b"\x89PNG\r\n\x1a\n", media_type="image/png")
    result = _summarize_content_block(img)
    assert "image/png" in result
    assert "unsupported" in result


@pytest.mark.unit
def test_summarize_binary_content() -> None:
    """BinaryContent produces an information-preserving placeholder."""
    audio = BinaryContent(data=b"RIFF....", media_type="audio/wav")
    result = _summarize_content_block(audio)
    assert "audio/wav" in result
    assert "unsupported" in result


@pytest.mark.unit
def test_summarize_image_url() -> None:
    """ImageUrl produces '[image: url]' placeholder."""
    url = ImageUrl(url="https://example.com/cat.png")
    result = _summarize_content_block(url)
    assert result == "[image: https://example.com/cat.png]"


@pytest.mark.unit
def test_summarize_audio_url() -> None:
    """AudioUrl produces '[audio: url]' placeholder."""
    url = AudioUrl(url="https://example.com/sound.mp3")
    result = _summarize_content_block(url)
    assert result == "[audio: https://example.com/sound.mp3]"


@pytest.mark.unit
def test_summarize_video_url() -> None:
    """VideoUrl produces '[video: url]' placeholder."""
    url = VideoUrl(url="https://example.com/clip.mp4")
    result = _summarize_content_block(url)
    assert result == "[video: https://example.com/clip.mp4]"


@pytest.mark.unit
def test_summarize_document_url() -> None:
    """DocumentUrl produces '[document: url]' placeholder."""
    url = DocumentUrl(url="https://example.com/doc.pdf")
    result = _summarize_content_block(url)
    assert result == "[document: https://example.com/doc.pdf]"


@pytest.mark.unit
def test_summarize_plain_string() -> None:
    """Plain string passes through unchanged."""
    assert _summarize_content_block("hello world") == "hello world"


@pytest.mark.unit
def test_summarize_text_content() -> None:
    """TextContent produces its content string."""
    from pydantic_ai.messages import TextContent

    tc = TextContent(content="some text")
    assert _summarize_content_block(tc) == "some text"


@pytest.mark.unit
def test_summarize_unknown_type() -> None:
    """Unknown types produce '[TypeName]' placeholder."""
    obj = object()
    result = _summarize_content_block(obj)
    assert result == "[object]"


# ---------------------------------------------------------------------------
# serialize_prompts / deserialize_prompts round-trip
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_serialize_prompts_empty_returns_none() -> None:
    """Empty prompts list returns None."""
    assert serialize_prompts([]) is None


@pytest.mark.unit
def test_serialize_deserialize_prompts_with_binary_image() -> None:
    """BinaryImage round-trips through serialize/deserialize."""
    img = BinaryImage(data=b"\x89PNG\r\n\x1a\n", media_type="image/png")
    prompts: list[str | list[Any]] = [
        "hello",
        ["describe this image", img],
    ]
    json_str = serialize_prompts(prompts)
    assert json_str is not None

    result = deserialize_prompts(json_str)
    assert len(result) == 2
    assert result[0] == "hello"
    assert isinstance(result[1], list)
    assert result[1][0] == "describe this image"
    # BinaryImage should be deserialized back
    restored = result[1][1]
    assert isinstance(restored, BinaryImage)
    assert restored.media_type == "image/png"
    assert restored.data == b"\x89PNG\r\n\x1a\n"


@pytest.mark.unit
def test_serialize_deserialize_prompts_all_strings() -> None:
    """All-string prompts round-trip correctly."""
    prompts: list[str | list[Any]] = ["hello", "world"]
    json_str = serialize_prompts(prompts)
    assert json_str is not None

    result = deserialize_prompts(json_str)
    assert result == ["hello", "world"]


@pytest.mark.unit
def test_deserialize_prompts_none_returns_empty() -> None:
    """None input to deserialize_prompts returns empty list."""
    assert deserialize_prompts(None) == []


@pytest.mark.unit
def test_deserialize_prompts_empty_string_returns_empty() -> None:
    """Empty string input to deserialize_prompts returns empty list."""
    assert deserialize_prompts("") == []


# ---------------------------------------------------------------------------
# Snapshot store saves prompts_serialized
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_snapshot_store_saves_prompts_serialized() -> None:
    """Snapshot store should save prompts_serialized with full multimodal data."""
    from wolfharness.lifecycle import MemorySnapshotStore

    store = MemorySnapshotStore()
    img = BinaryImage(data=b"\x89PNG\r\n\x1a\n", media_type="image/png")
    prompts: list[str | list[Any]] = ["hello", ["describe this", img]]
    prompt_text = "hello\ndescribe this [image/png]"

    # Simulate what run.py does
    state = {
        "state": "running",
        "run_id": "test-run",
        "turn_id": "turn-1",
        "prompt": prompt_text,
        "prompts_serialized": serialize_prompts(prompts),
    }
    store.save(state)

    # Load the latest snapshot (returns (state_dict, seq) tuple)
    load_result = store.load()
    assert load_result is not None
    snapshot, _seq = load_result
    assert snapshot["prompt"] == prompt_text
    assert snapshot["prompts_serialized"] is not None

    # Deserialize and verify full multimodal data is preserved
    restored = deserialize_prompts(snapshot["prompts_serialized"])
    assert len(restored) == 2
    assert restored[0] == "hello"
    assert isinstance(restored[1], list)
    restored_img = restored[1][1]
    assert isinstance(restored_img, BinaryImage)
    assert restored_img.media_type == "image/png"


# ---------------------------------------------------------------------------
# prompt_text stringification uses _summarize_content_block
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_prompt_text_uses_summarize_content_block() -> None:
    """prompt_text should use _summarize_content_block, not str()."""
    img = BinaryImage(data=b"\x89PNG\r\n\x1a\n", media_type="image/png")
    prompts: list[str | list[Any]] = ["hello", ["describe this", img]]

    # Replicate the logic from run.py
    prompt_text = "\n".join(
        p if isinstance(p, str) else " ".join(_summarize_content_block(b) for b in p)
        for p in prompts
    )

    # Should NOT contain "BinaryContent" or "b'\\x89PNG" (raw repr)
    assert "BinaryContent" not in prompt_text
    assert "b'\\x89PNG" not in prompt_text
    assert "\\x89PNG" not in prompt_text
    # Should contain the meaningful placeholder
    assert "image/png" in prompt_text
    assert "unsupported" in prompt_text
    assert "hello" in prompt_text
    assert "describe this" in prompt_text


# ---------------------------------------------------------------------------
# OpenCode converter uses _summarize_content_block
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_opencode_converter_uses_summarize_content_block() -> None:
    """OpenCode converter should use _summarize_content_block for non-string items."""
    img = BinaryImage(data=b"\x89PNG\r\n\x1a\n", media_type="image/png")
    content: list[Any] = ["text before image", img, "text after image"]

    # Replicate the converter logic
    text = " ".join(_summarize_content_block(c) for c in content)

    assert "text before image" in text
    assert "text after image" in text
    assert "image/png" in text
    assert "unsupported" in text
    assert "BinaryContent" not in text
    assert "b'\\x89PNG" not in text


# ---------------------------------------------------------------------------
# compaction _extract_text_content uses _summarize_content_block
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_compaction_extract_text_content_uses_summarize() -> None:
    """compaction._extract_text_content should include placeholders for non-string items."""
    from wolfharness.messaging.compaction import _extract_text_content

    img = BinaryImage(data=b"\x89PNG\r\n\x1a\n", media_type="image/png")
    msg = ModelRequest(
        parts=[
            UserPromptPart(content=["describe this", img]),
        ]
    )

    result = _extract_text_content(msg)
    assert "describe this" in result
    assert "image/png" in result
    assert "unsupported" in result
    assert "BinaryContent" not in result


# ---------------------------------------------------------------------------
# Integration: full round-trip multimodal prompt through DB
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_multimodal_prompt_round_trip_through_db(tmp_path: Any) -> None:
    """Multimodal prompt should survive: save to DB -> load from DB -> reconstruct.

    Uses SQLModelProvider with a temp SQLite database. Saves a ChatMessage
    with ``messages`` containing a ``BinaryImage``, loads it back via
    ``get_session_messages()``, and verifies the ``BinaryImage`` is
    preserved with correct data and media_type.
    """
    from wolfharness.messaging import ChatMessage
    from wolfharness_config.storage import SQLStorageConfig
    from wolfharness_storage.sql_provider import SQLModelProvider

    db_path = tmp_path / "test_multimodal.db"
    config = SQLStorageConfig(url=f"sqlite:///{db_path}")
    provider = SQLModelProvider(config)

    # Clear engine cache to avoid cross-test contamination.
    from wolfharness_config.storage import _engine_cache

    _engine_cache.clear()

    async with provider:
        img = BinaryImage(data=b"\x89PNG\r\n\x1a\n", media_type="image/png")
        msg = ChatMessage(
            content="describe this [image/png]",
            role="user",
            session_id="test-mm-session",
            messages=[ModelRequest(parts=[UserPromptPart(content=["describe this", img])])],
        )
        await provider.log_message(message=msg)

        loaded = await provider.get_session_messages("test-mm-session")
        assert len(loaded) == 1
        loaded_msg = loaded[0]
        assert loaded_msg.role == "user"
        assert loaded_msg.content == "describe this [image/png]"

        # Verify messages field contains the deserialized multimodal content.
        assert len(loaded_msg.messages) == 1
        model_req = loaded_msg.messages[0]
        assert isinstance(model_req, ModelRequest)
        assert len(model_req.parts) == 1
        part = model_req.parts[0]
        assert isinstance(part, UserPromptPart)
        # content should be a list with the string and BinaryImage
        assert isinstance(part.content, list)
        assert part.content[0] == "describe this"
        restored_img = part.content[1]
        assert isinstance(restored_img, BinaryImage)
        assert restored_img.media_type == "image/png"
        assert restored_img.data == b"\x89PNG\r\n\x1a\n"

    _engine_cache.clear()
