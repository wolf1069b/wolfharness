"""Tests for OpenCode storage provider helper functions.

Covers:
- convert_user_content_to_parts() with various UserContent types
"""

from __future__ import annotations

from pydantic_ai import BinaryContent, ImageUrl, TextContent
import pytest

from wolfharness_storage.opencode_provider.helpers import convert_user_content_to_parts


pytestmark = pytest.mark.unit


@pytest.fixture
def message_id() -> str:
    """Provide a stable message ID for tests."""
    return "msg_test123"


@pytest.fixture
def session_id() -> str:
    """Provide a stable session ID for tests."""
    return "ses_test123"


def test_convert_str_content_returns_single_text_part(
    message_id: str,
    session_id: str,
) -> None:
    """Simple string input produces exactly one TextPart."""
    result = convert_user_content_to_parts(
        content="Hello, world!",
        message_id=message_id,
        session_id=session_id,
        part_counter_start=0,
    )

    assert len(result) == 1
    part = result[0]
    assert part.text == "Hello, world!"
    assert part.message_id == message_id
    assert part.session_id == session_id
    assert part.id.startswith("prt_")


def test_convert_list_with_str_items_returns_text_parts(
    message_id: str,
    session_id: str,
) -> None:
    """List of str items produces one TextPart per item."""
    content: list[str] = ["Hello", "World"]
    result = convert_user_content_to_parts(
        content=content,
        message_id=message_id,
        session_id=session_id,
        part_counter_start=0,
    )

    assert len(result) == 2
    assert result[0].text == "Hello"
    assert result[1].text == "World"
    # Each part gets a unique ID
    assert result[0].id != result[1].id


def test_convert_list_with_binary_content_skips_with_warning(
    message_id: str,
    session_id: str,
) -> None:
    """BinaryContent items are skipped (with warning), no TextPart produced."""
    binary = BinaryContent(data=b"\x89PNG", media_type="image/png")
    result = convert_user_content_to_parts(
        content=[binary],
        message_id=message_id,
        session_id=session_id,
        part_counter_start=0,
    )

    assert result == []


def test_convert_list_with_file_url_skips_with_warning(
    message_id: str,
    session_id: str,
) -> None:
    """FileUrl items (e.g. ImageUrl) are skipped (with warning), no TextPart produced."""
    url = ImageUrl(url="https://example.com/image.png")
    result = convert_user_content_to_parts(
        content=[url],
        message_id=message_id,
        session_id=session_id,
        part_counter_start=0,
    )

    assert result == []


def test_convert_empty_list_returns_empty_list(
    message_id: str,
    session_id: str,
) -> None:
    """Empty list input returns empty list."""
    result = convert_user_content_to_parts(
        content=[],
        message_id=message_id,
        session_id=session_id,
        part_counter_start=0,
    )

    assert result == []


def test_convert_mixed_content_produces_only_text_parts(
    message_id: str,
    session_id: str,
) -> None:
    """Mixed str and binary content: only text items become TextParts."""
    binary = BinaryContent(data=b"\x00\x01", media_type="application/octet-stream")
    url = ImageUrl(url="https://example.com/doc.pdf")
    content = ["First text", binary, "Second text", url]
    result = convert_user_content_to_parts(
        content=content,
        message_id=message_id,
        session_id=session_id,
        part_counter_start=0,
    )

    assert len(result) == 2
    assert result[0].text == "First text"
    assert result[1].text == "Second text"


def test_convert_text_content_produces_text_part(
    message_id: str,
    session_id: str,
) -> None:
    """TextContent items produce TextParts with their content."""
    tc = TextContent(content="Structured text")
    result = convert_user_content_to_parts(
        content=[tc],
        message_id=message_id,
        session_id=session_id,
        part_counter_start=0,
    )

    assert len(result) == 1
    assert result[0].text == "Structured text"
