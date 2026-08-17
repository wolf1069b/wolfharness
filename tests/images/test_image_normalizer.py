"""Tests for ImageNormalizer (RFC-0059)."""

from __future__ import annotations

import base64
import io
import random

from PIL import Image
import pytest

from wolfharness.images.normalizer import ImageNormalizer, ImageSizeError
from wolfharness_config.attachment import AttachmentImageConfig


pytestmark = pytest.mark.unit


def _png_bytes(width: int, height: int, color: tuple[int, int, int] = (200, 30, 30)) -> bytes:
    """Create a solid-color PNG image of the given dimensions."""
    img = Image.new("RGB", (width, height), color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _noisy_png_bytes(width: int, height: int) -> bytes:
    """Create a noisy PNG image whose bytes are hard to compress."""
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
    """Build a data URI from image bytes."""
    return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"


def test_normalize_passes_small_image_through():
    """A small image within limits passes through unchanged."""
    data = _png_bytes(100, 100)
    n = ImageNormalizer()

    url, mime = n.normalize(_data_uri(data), "image/png")

    assert url == _data_uri(data)
    assert mime == "image/png"


def test_normalize_resizes_oversized_image():
    """An oversized image is resized to within max_base64_bytes."""
    data = _png_bytes(4000, 3000)
    n = ImageNormalizer(AttachmentImageConfig(max_base64_bytes=64 * 1024))

    url, mime = n.normalize(_data_uri(data), "image/png")

    assert url != _data_uri(data)
    payload = url.split(";base64,", 1)[1]
    assert len(payload) <= 64 * 1024
    assert mime in ("image/png", "image/jpeg")


def test_normalize_resizes_oversized_dimensions():
    """An image with oversized dimensions is resized even when bytes are small."""
    data = _png_bytes(5000, 100)
    n = ImageNormalizer(
        AttachmentImageConfig(max_width=2000, max_height=2000, max_base64_bytes=5 * 1024 * 1024)
    )

    url, _mime = n.normalize(_data_uri(data), "image/png")

    decoded = base64.b64decode(url.split(";base64,", 1)[1])
    with Image.open(io.BytesIO(decoded)) as img:
        assert img.width <= 2000
        assert img.height <= 2000


def test_normalize_non_data_uri_passes_through():
    """A remote URL passes through unchanged (no SSRF)."""
    n = ImageNormalizer()

    url, mime = n.normalize("https://example.com/photo.png", "image/png")

    assert url == "https://example.com/photo.png"
    assert mime == "image/png"


def test_normalize_invalid_base64_passes_through():
    """A malformed base64 data URI passes through unchanged."""
    n = ImageNormalizer()

    url, mime = n.normalize("data:image/png;base64,@@@not-base64@@@", "image/png")

    assert url == "data:image/png;base64,@@@not-base64@@@"
    assert mime == "image/png"


def test_normalize_disabled_raises_for_oversized():
    """With auto_resize=False, an oversized image raises ImageSizeError."""
    data = _noisy_png_bytes(400, 300)
    n = ImageNormalizer(AttachmentImageConfig(auto_resize=False, max_base64_bytes=64 * 1024))

    assert (
        len(data)
        > AttachmentImageConfig(auto_resize=False, max_base64_bytes=64 * 1024).max_base64_bytes
        * 3
        // 4
    )

    with pytest.raises(ImageSizeError):
        n.normalize(_data_uri(data), "image/png")


def test_normalize_disabled_passes_small_through():
    """With auto_resize=False, a small image still passes through."""
    data = _png_bytes(100, 100)
    n = ImageNormalizer(AttachmentImageConfig(auto_resize=False))

    url, mime = n.normalize(_data_uri(data), "image/png")

    assert url == _data_uri(data)
    assert mime == "image/png"


def test_normalize_bytes_returns_new_bytes_on_resize():
    """normalize_bytes returns a new buffer (not the input) when resized."""
    data = _png_bytes(4000, 3000)
    n = ImageNormalizer(AttachmentImageConfig(max_base64_bytes=64 * 1024))

    normalized, _mime = n.normalize_bytes(data, "image/png")

    assert normalized is not data
    assert len(normalized) < len(data)


def test_exports_available():
    """ImageNormalizer and ImageSizeError are exported from wolfharness.images."""
    from wolfharness import images

    assert images.ImageNormalizer is ImageNormalizer
    assert images.ImageSizeError is ImageSizeError
