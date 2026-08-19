"""Wiring tests for ImageNormalizer at protocol entry points (RFC-0059).

Verify the RFC-0059 normalization is actually applied at the three
user-upload entry points: the OpenCode ``extract_user_prompt_from_parts``
converter, the ACP ``from_acp_content`` converter, and the functional
``run_agent`` wrapper.
"""

from __future__ import annotations

import base64
import io
import random

from PIL import Image
import pytest

from wolfharness.images.normalizer import ImageNormalizer
from wolfharness_config.attachment import AttachmentImageConfig


pytestmark = pytest.mark.unit


def _noisy_png_bytes(width: int, height: int) -> bytes:
    rng = random.Random(42)
    img = Image.new("RGB", (width, height))
    pixels = [
        (rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255))
        for _ in range(width * height)
    ]
    img.putdata(pixels)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _data_uri(data: bytes, mime: str = "image/png") -> str:
    return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"


# =============================================================================
# OpenCode: extract_user_prompt_from_parts
# =============================================================================


async def _extract_prompt(part_input: object, normalizer: ImageNormalizer | None) -> list:
    from wolfharness_server.opencode_server.converters import (
        extract_user_prompt_from_parts,
    )

    parts_list = await extract_user_prompt_from_parts(
        [part_input],
        "wire-test-session",
        agent=None,
        normalizer=normalizer,
    )
    return list(parts_list)


async def test_extract_user_prompt_normalizes_oversized_image() -> None:
    """Oversized data-URI image in FilePartInput is resized by the normalizer."""
    from pydantic_ai import BinaryContent

    from wolfharness_server.opencode_server.models import FilePartInput

    data = _noisy_png_bytes(800, 600)
    part = FilePartInput(
        mime="image/png",
        url=_data_uri(data),
        filename="photo.png",
    )
    normalizer = ImageNormalizer(AttachmentImageConfig(max_base64_bytes=64 * 1024))

    result = await _extract_prompt(part, normalizer)

    assert len(result) == 1
    assert isinstance(result[0], BinaryContent)
    assert result[0].media_type in ("image/png", "image/jpeg")
    assert len(result[0].data) < len(data)


async def test_extract_user_prompt_leaves_small_image_untouched() -> None:
    """Small data-URI image passes through as-is (no resize)."""
    from pydantic_ai import BinaryContent

    from wolfharness_server.opencode_server.models import FilePartInput

    data = _noisy_png_bytes(100, 100)
    part = FilePartInput(
        mime="image/png",
        url=_data_uri(data),
        filename="small.png",
    )
    normalizer = ImageNormalizer()

    result = await _extract_prompt(part, normalizer)

    assert len(result) == 1
    assert isinstance(result[0], BinaryContent)
    assert result[0].data == data


async def test_extract_user_prompt_non_image_not_normalized() -> None:
    """Non-image FilePartInput is not routed through the image normalizer."""
    from pydantic_ai import BinaryContent

    from wolfharness_server.opencode_server.models import FilePartInput

    pdf_bytes = b"%PDF-1.4 test-not-a-real-pdf"
    part = FilePartInput(
        mime="application/pdf",
        url=_data_uri(pdf_bytes, "application/pdf"),
        filename="doc.pdf",
    )
    normalizer = ImageNormalizer(AttachmentImageConfig(auto_resize=False, max_base64_bytes=8))

    result = await _extract_prompt(part, normalizer)

    assert len(result) == 1
    assert isinstance(result[0], BinaryContent)
    assert result[0].data == pdf_bytes


async def test_extract_user_prompt_no_normalizer_passthrough() -> None:
    """Without a normalizer the converter is unchanged (backward compatible)."""
    from pydantic_ai import BinaryContent

    from wolfharness_server.opencode_server.models import FilePartInput

    data = _noisy_png_bytes(800, 600)
    part = FilePartInput(mime="image/png", url=_data_uri(data), filename="p.png")

    result = await _extract_prompt(part, None)

    assert len(result) == 1
    assert isinstance(result[0], BinaryContent)
    assert result[0].data == data


# =============================================================================
# ACP: from_acp_content
# =============================================================================


def _from_acp(block: object, normalizer: ImageNormalizer | None):
    from wolfharness_server.acp_server.converters import from_acp_content

    return from_acp_content(block, fs=None, normalizer=normalizer)


def test_from_acp_content_normalizes_oversized_image() -> None:
    """Oversized ACP ImageContentBlock is resized by the normalizer."""
    from pydantic_ai import BinaryContent

    from acp.schema import ImageContentBlock

    data = _noisy_png_bytes(800, 600)
    block = ImageContentBlock(
        data=base64.b64encode(data).decode("ascii"),
        mime_type="image/png",
    )
    normalizer = ImageNormalizer(AttachmentImageConfig(max_base64_bytes=64 * 1024))

    result = _from_acp(block, normalizer)

    assert isinstance(result, BinaryContent)
    assert len(result.data) < len(data)


def test_from_acp_content_leaves_small_image_untouched() -> None:
    """Small ACP ImageContentBlock passes through as-is."""
    from pydantic_ai import BinaryContent

    from acp.schema import ImageContentBlock

    data = _noisy_png_bytes(100, 100)
    block = ImageContentBlock(
        data=base64.b64encode(data).decode("ascii"),
        mime_type="image/png",
    )
    normalizer = ImageNormalizer()

    result = _from_acp(block, normalizer)

    assert isinstance(result, BinaryContent)
    assert result.data == data


def test_from_acp_content_no_normalizer_passthrough() -> None:
    """Without a normalizer the ACP converter is unchanged."""
    from pydantic_ai import BinaryContent

    from acp.schema import ImageContentBlock

    data = _noisy_png_bytes(800, 600)
    block = ImageContentBlock(
        data=base64.b64encode(data).decode("ascii"),
        mime_type="image/png",
    )

    result = _from_acp(block, None)

    assert isinstance(result, BinaryContent)
    assert result.data == data


def test_from_acp_content_embedded_blob_unknown_mime_passes_through() -> None:
    """Embedded blob with a mime outside the old whitelist reaches the model.

    A generic binary (e.g. application/json, text/yaml) must become
    ``BinaryContent`` with its mime preserved instead of a text placeholder.
    """
    from pydantic_ai import BinaryContent

    from acp.schema import (
        BlobResourceContents,
        EmbeddedResourceContentBlock,
    )

    data = b'{"key": "value"}'
    block = EmbeddedResourceContentBlock(
        resource=BlobResourceContents(
            uri="acp://inner/config.json",
            blob=base64.b64encode(data).decode("ascii"),
            mime_type="application/json",
        )
    )

    result = _from_acp(block, None)

    assert isinstance(result, BinaryContent)
    assert result.data == data
    assert result.media_type == "application/json"


def test_from_acp_content_embedded_blob_missing_mime_uses_octet_stream() -> None:
    """Embedded blob without mime falls back to application/octet-stream."""
    from pydantic_ai import BinaryContent

    from acp.schema import BlobResourceContents, EmbeddedResourceContentBlock

    data = b"\x00\x01\x02"
    block = EmbeddedResourceContentBlock(
        resource=BlobResourceContents(
            uri="acp://inner/raw.bin",
            blob=base64.b64encode(data).decode("ascii"),
        )
    )

    result = _from_acp(block, None)

    assert isinstance(result, BinaryContent)
    assert result.data == data
    assert result.media_type == "application/octet-stream"


def test_from_acp_content_embedded_blob_image_is_binary_image() -> None:
    """Embedded image blob stays BinaryImage, not generic BinaryContent."""
    from pydantic_ai import BinaryImage

    from acp.schema import BlobResourceContents, EmbeddedResourceContentBlock

    data = _noisy_png_bytes(64, 64)
    block = EmbeddedResourceContentBlock(
        resource=BlobResourceContents(
            uri="acp://inner/photo.png",
            blob=base64.b64encode(data).decode("ascii"),
            mime_type="image/png",
        )
    )

    result = _from_acp(block, None)

    assert isinstance(result, BinaryImage)
    assert result.data == data
    assert result.media_type == "image/png"


# =============================================================================
# functional run_agent: image_url normalization
# =============================================================================


def test_make_image_normalizer_defaults() -> None:
    """_make_image_normalizer returns None for None config."""
    from wolfharness.functional.run import _make_image_normalizer

    assert _make_image_normalizer(None) is None


def test_make_image_normalizer_with_config() -> None:
    """_make_image_normalizer builds a normalizer from config."""
    from wolfharness.functional.run import _make_image_normalizer

    n = _make_image_normalizer(AttachmentImageConfig(auto_resize=False))

    assert n is not None
    assert n.config.auto_resize is False


def test_normalize_image_url_resizes_data_uri() -> None:
    """_normalize_image_url normalizes oversized data URI when normalizer set."""
    from wolfharness.functional.run import _normalize_image_url

    data = _noisy_png_bytes(800, 600)
    url = _data_uri(data)
    normalizer = ImageNormalizer(AttachmentImageConfig(max_base64_bytes=64 * 1024))

    normalized = _normalize_image_url(url, normalizer)

    assert normalized != url
    payload = normalized.split(";base64,", 1)[1]
    assert len(payload) <= 64 * 1024


def test_normalize_image_url_no_normalizer_unchanged() -> None:
    """_normalize_image_url returns the URL unchanged without a normalizer."""
    from wolfharness.functional.run import _normalize_image_url

    url = "https://example.com/photo.png"

    assert _normalize_image_url(url, None) == url
