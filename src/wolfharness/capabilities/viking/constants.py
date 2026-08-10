"""Constants for the Viking capability — image extension detection.

Kept in a dedicated module (not ``__init__``) so ``tools.py`` can import
them at runtime without importing the full capability package (avoiding
import cycles, since the capability module lazily imports tools).
"""

from __future__ import annotations


# Image extensions recognized by the openviking server parser layer
# (``parse/parsers/media/constants.py``). MUST be kept in sync manually with
# the server's ``IMAGE_EXTENSIONS`` — extension is the authoritative signal
# since the server's ``stat`` API exposes no MIME field.
#
# ``.svg`` IS included (matching the server), but is a vector format most
# vision APIs reject, so ``viking_read`` downgrades SVG URIs to a text hint
# and never returns bytes for them. See ``_should_return_image_bytes``.
IMAGE_EXTENSIONS: frozenset[str] = frozenset({
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".bmp",
    ".webp",
    ".svg",
    ".tiff",
    ".tif",
    ".ico",
    ".jp2",
})

# MIME mapping for the byte-return image extensions above. Unknown
# extensions fall back to ``application/octet-stream``.
IMAGE_MIME_TYPES: dict[str, str] = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".bmp": "image/bmp",
    ".webp": "image/webp",
    ".tiff": "image/tiff",
    ".tif": "image/tiff",
    ".ico": "image/x-icon",
    ".jp2": "image/jp2",
}
