from __future__ import annotations

import base64

from mcp.types import BlobResourceContents
from pydantic_ai import BinaryContent
import pytest

from wolfharness.mcp_server.conversions import from_mcp_content


pytestmark = pytest.mark.unit


async def test_from_mcp_content_blob_preserves_mime_type() -> None:
    """BlobResourceContents mimeType is propagated, not hardcoded octet-stream."""
    raw = b"\x89PNG\r\n\x1a\n"
    blob = base64.b64encode(raw).decode("ascii")
    result = await from_mcp_content([
        BlobResourceContents(uri="mcp://s/img.png", blob=blob, mimeType="image/png")
    ])
    assert len(result) == 1
    content = result[0]
    assert isinstance(content, BinaryContent)
    assert content.data == raw
    assert content.media_type == "image/png"


async def test_from_mcp_content_blob_falls_back_to_octet_stream() -> None:
    """BlobResourceContents without mimeType still returns bytes."""
    raw = b"\x00\x01\x02"
    blob = base64.b64encode(raw).decode("ascii")
    result = await from_mcp_content([BlobResourceContents(uri="mcp://s/data.bin", blob=blob)])
    assert len(result) == 1
    content = result[0]
    assert isinstance(content, BinaryContent)
    assert content.data == raw
    assert content.media_type == "application/octet-stream"
