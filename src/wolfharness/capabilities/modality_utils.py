"""Shared utilities for multimodal content description and binary classification.

Provides ``describe_multimodal_content`` and ``classify_binary_content`` so
that multiple call sites (helpers, MCP conversions, storage) share a single
source of truth for text placeholders and modality classification.
"""

from __future__ import annotations

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


def describe_multimodal_content(content: MultiModalContent) -> str:
    """Produce a short text placeholder for a pydantic-ai multimodal content item.

    Handles all ``MultiModalContent`` variant types, producing meaningful
    placeholders instead of raw ``repr()`` output for binary/URL types.

    Args:
        content: A ``MultiModalContent`` variant — ``BinaryImage``,
            ``BinaryContent``, ``ImageUrl``, ``AudioUrl``, ``VideoUrl``,
            ``DocumentUrl``, or ``UploadedFile``.

    Returns:
        A lowercase placeholder string suitable for logs, ``ChatMessage.content``,
        and other text-only contexts.
    """
    match content:
        case BinaryImage(media_type=media) | BinaryContent(media_type=media):
            return f"[{media}]"
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
