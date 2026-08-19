"""Shared utilities for multimodal content description and binary classification.

Provides ``describe_multimodal_content`` and ``classify_binary_content`` so
that multiple call sites (helpers, MCP conversions, storage) share a single
source of truth for text placeholders and modality classification.
"""

from __future__ import annotations

import re
from typing import Literal, assert_never

from pydantic_ai import BinaryContent, BinaryImage
from pydantic_ai.messages import (
    AudioUrl,
    DocumentUrl,
    ImageUrl,
    MultiModalContent,
    UploadedFile,
    VideoUrl,
)


type BinaryCategory = Literal["image", "audio", "video", "document", "unknown"]


# Control characters that could break prompt formatting if a caller-supplied
# filename were interpolated verbatim (RFC-0061 Security considerations).
_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f]")

# pydantic-ai falls back to a short content hash (sha1[:6]) when no explicit
# identifier is available.  A bare hash is not a retrievable reference, so it
# must not be presented to the model as a "File:" it could open.
_HASH_IDENTIFIER = re.compile(r"^[0-9a-fA-F]{4,64}$")


def _safe_ref(value: str) -> str:
    r"""Sanitize a caller-influenced identifier (filename/path) for interpolation.

    Replaces control characters (newlines, tabs, NUL, ...) with a visible
    ``\xNN`` escape so a malicious or malformed filename cannot inject
    prompt structure or break log lines. Non-control characters pass through.
    """

    def _escape(match: re.Match[str]) -> str:
        return f"\\x{ord(match.group()):02x}"

    return _CONTROL_CHARS.sub(_escape, value)


def _describe_binary(binary: BinaryImage | BinaryContent) -> str:
    """Build an information-preserving placeholder for binary content.

    Uses the underlying ``_identifier`` field, which is only set when a
    caller explicitly provided one (e.g. the tool that produced the bytes
    knew the source path). The public ``.identifier`` property is avoided
    because it falls back to a content hash that is not a retrievable
    reference. Hash-shaped identifiers are likewise suppressed — a bare
    digest is not a path the model could open.
    """
    media = binary.media_type
    identifier = binary._identifier
    if identifier and not _HASH_IDENTIFIER.fullmatch(identifier):
        safe_id = _safe_ref(identifier)
        return (
            f"[User supplied {media} — direct model processing is "
            f"unsupported by the active model. File: {safe_id} "
            f"(may be opened by a vision-capable subagent or file tool)]"
        )
    return (
        f"[User supplied {media} — direct model processing is "
        f"unsupported by the active model (content not inlined; "
        f"no file reference available)]"
    )


def describe_multimodal_content(content: MultiModalContent) -> str:
    """Produce a short text placeholder for a pydantic-ai multimodal content item.

    Handles all ``MultiModalContent`` variant types, producing meaningful
    placeholders instead of raw ``repr()`` output for binary/URL types.

    For URL types the URL itself is returned (it is already a retrievable
    reference). For ``UploadedFile`` the file id is returned. For binary
    content (``BinaryImage`` / ``BinaryContent``) the placeholder is
    *information-preserving* (RFC-0061): it states the media type, whether an
    ``identifier`` is available, and that direct model processing is
    unsupported, rather than emitting a bare ``[image/png]`` token that
    misleads the model into inventing content it cannot see.

    Args:
        content: A ``MultiModalContent`` variant — ``BinaryImage``,
            ``BinaryContent``, ``ImageUrl``, ``AudioUrl``, ``VideoUrl``,
            ``DocumentUrl``, or ``UploadedFile``.

    Returns:
        An information-preserving placeholder string suitable for logs,
        ``ChatMessage.content``, and other text-only contexts.
    """
    match content:
        case BinaryImage() | BinaryContent() as binary:
            return _describe_binary(binary)
        case ImageUrl(url=url):
            return f"[image: {url}]"
        case AudioUrl(url=url):
            return f"[audio: {url}]"
        case VideoUrl(url=url):
            return f"[video: {url}]"
        case DocumentUrl(url=url):
            return f"[document: {url}]"
        case UploadedFile(file_id=file_id):
            return f"[uploaded_file: {file_id}]"
        case _ as unreachable:
            assert_never(unreachable)


def classify_binary_content(content: BinaryContent) -> BinaryCategory:
    """Classify a ``BinaryContent`` instance by its ``media_type`` prefix.

    Args:
        content: A ``BinaryContent`` (or ``BinaryImage``) instance.

    Returns:
        One of ``"image"``, ``"audio"``, ``"video"``, ``"document"``, or
        ``"unknown"``.
    """
    media = content.media_type
    if media.startswith("image/"):
        return "image"
    if media.startswith("audio/"):
        return "audio"
    if media.startswith("video/"):
        return "video"
    if media in ("application/pdf", "text/pdf"):
        return "document"
    return "unknown"
