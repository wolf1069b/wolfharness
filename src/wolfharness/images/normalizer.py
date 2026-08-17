"""Image attachment normalization service (RFC-0059).

Normalizes oversized image attachments on the protocol user-upload path
before they are forwarded to the model. Reuses the existing
``resize_image_if_needed()`` from the fsspec toolset so behavior stays
consistent with the tool-read path.

Failure semantics:

- ``auto_resize: true`` (default): images exceeding dimension or byte
  limits are resized and re-encoded. If normalization itself fails
  (e.g. Pillow unavailable), the original is passed through unchanged —
  the session is never interrupted.
- ``auto_resize: false``: the byte budget is still enforced and an
  over-limit image raises :class:`ImageSizeError`.
- Only ``data:`` base64 URIs are processed. Remote ``http(s)://`` and
  ``file://`` URLs are left untouched (avoids SSRF / filesystem access).
"""

from __future__ import annotations

import base64
import binascii
import io
import logging
from typing import Any

from wolfharness_config.attachment import AttachmentImageConfig


logger = logging.getLogger(__name__)


class ImageSizeError(Exception):
    """Raised when an image attachment cannot be brought within limits.

    Used for the ``auto_resize: false`` path: an over-limit user attachment
    fails with this error instead of silently passing an oversized image.
    """


class ImageNormalizer:
    """Normalize oversized image attachments.

    Args:
        config: Image attachment normalization configuration. When omitted,
            falls back to ``AttachmentImageConfig()`` defaults.
    """

    def __init__(self, config: AttachmentImageConfig | None = None) -> None:
        self._config = config if config is not None else AttachmentImageConfig()

    def normalize(self, url: str, mime: str) -> tuple[str, str]:
        """Normalize an image attachment URL.

        Args:
            url: Image URL. Only ``data:`` base64 URIs are processed.
            mime: MIME type of the image (e.g. ``image/png``).

        Returns:
            Tuple of ``(possibly_normalized_url, mime)``. Passes the inputs
            through unchanged for non-``data:`` URLs and for images already
            within limits.

        Raises:
            ImageSizeError: When ``auto_resize`` is disabled and the image
                exceeds configured limits.
        """
        if not url.startswith("data:"):
            return url, mime

        payload = _data_uri_payload(url)
        if payload is None:
            return url, mime

        try:
            data = base64.b64decode(payload)
        except (binascii.Error, ValueError):
            logger.warning("Invalid base64 in image data URI; passing through unchanged")
            return url, mime

        normalized_data, new_mime = self.normalize_bytes(data, mime)
        if normalized_data is data:
            return url, mime
        return _make_data_uri(new_mime, _b64encode(normalized_data)), new_mime

    def normalize_bytes(self, data: bytes, mime: str) -> tuple[bytes, str]:
        """Normalize raw image bytes.

        Args:
            data: Raw image bytes.
            mime: MIME type of the image (e.g. ``image/png``).

        Returns:
            Tuple of ``(possibly_normalized_bytes, mime)``. Returns the
            original input unchanged when already within limits or not
            normalizable.

        Raises:
            ImageSizeError: When ``auto_resize`` is disabled and the image
                exceeds configured limits.
        """
        target_bytes = self._config.max_base64_bytes * 3 // 4
        max_size = min(self._config.max_width, self._config.max_height)

        if len(data) <= target_bytes and not self._config.auto_resize:
            return data, mime
        if len(data) <= target_bytes and _fits_dimensions(data, max_size):
            return data, mime

        if not self._config.auto_resize:
            return self._normalize_disabled(data, mime, max_size, target_bytes)

        return self._normalize_enabled(data, mime, max_size, target_bytes)

    def _normalize_enabled(
        self,
        data: bytes,
        mime: str,
        max_size: int,
        target_bytes: int,
    ) -> tuple[bytes, str]:
        """Normalize an image with ``auto_resize: true``."""
        from wolfharness_toolsets.fsspec_toolset.image_utils import (
            resize_image_if_needed,
        )

        try:
            resized, new_mime, note = resize_image_if_needed(
                data,
                mime,
                max_size=max_size,
                max_bytes=target_bytes,
            )
        except Exception:
            logger.warning("Image normalization failed; passing through unchanged", exc_info=True)
            return data, mime

        if note is None:
            return data, mime

        if len(resized) > target_bytes * 4 // 3:
            logger.warning(
                "Re-encoded image still exceeds max_base64_bytes; passing original through"
            )
            return data, mime

        return resized, new_mime

    def _normalize_disabled(
        self,
        data: bytes,
        mime: str,
        max_size: int,
        target_bytes: int,
    ) -> tuple[bytes, str]:
        """Raise :class:`ImageSizeError` for over-limit images."""
        try:
            with _open_image(data) as img:
                width, height = img.size
        except Exception:  # noqa: BLE001
            logger.warning("Image decode failed with auto_resize disabled; passing through")
            return data, mime

        if width > max_size or height > max_size or len(data) > target_bytes:
            raise ImageSizeError(
                f"Image attachment {width}x{height} exceeds configured limits "
                f"({max_size}x{max_size} px, {target_bytes} bytes) and auto_resize is disabled"
            )
        return data, mime

    @property
    def config(self) -> AttachmentImageConfig:
        """The underlying normalization configuration."""
        return self._config


def _open_image(data: bytes) -> Any:
    """Open an image from bytes as a context manager."""
    from PIL import Image

    return Image.open(io.BytesIO(data))


def _fits_dimensions(data: bytes, max_size: int) -> bool:
    """Return whether an image's dimensions are within ``max_size``."""
    try:
        with _open_image(data) as img:
            width, height = img.size
    except Exception:  # noqa: BLE001
        return True
    return int(width) <= max_size and int(height) <= max_size


def _data_uri_payload(url: str) -> str | None:
    """Extract the base64 payload from a data URI, or None if not base64."""
    marker = ";base64,"
    index = url.find(marker)
    if index < 0:
        return None
    return url[index + len(marker) :]


def _make_data_uri(mime: str, encoded: str) -> str:
    """Build a ``data:`` URI from a MIME type and base64 payload."""
    return f"data:{mime};base64,{encoded}"


def _b64encode(data: bytes) -> str:
    """Base64-encode bytes to ASCII (no newlines)."""
    return base64.b64encode(data).decode("ascii")
