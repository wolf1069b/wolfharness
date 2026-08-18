"""Image attachment configuration models.

Controls how user-uploaded image attachments are normalized before they
are forwarded to the model (RFC-0059). The defaults mirror opencode's
``attachment.image`` limits (2000x2000 px, 5 MB base64 payload).
"""

from __future__ import annotations

from pydantic import ConfigDict, Field
from schemez import Schema


class AttachmentImageConfig(Schema):
    """Configuration for image attachment normalization.

    Controls automatic resizing/re-encoding of oversized image attachments
    on the protocol user-upload path (Python API, ACP, OpenCode server).

    Values are aligned with opencode's ``attachment.image`` defaults so
    cross-project expectations stay consistent.
    """

    auto_resize: bool = Field(default=True, title="Auto-resize images")
    """Whether to automatically resize/re-encode oversized images.

    When ``True`` (default), image attachments exceeding ``max_width`` /
    ``max_height`` / ``max_base64_bytes`` are resized and re-encoded before
    being forwarded to the model. When ``False``, a byte-budget check is
    retained but no resizing is performed.
    """

    max_width: int = Field(default=2000, ge=1, title="Max image width")
    """Maximum image width in pixels (default 2000)."""

    max_height: int = Field(default=2000, ge=1, title="Max image height")
    """Maximum image height in pixels (default 2000)."""

    max_base64_bytes: int = Field(
        default=5 * 1024 * 1024,
        ge=1,
        title="Max base64 payload bytes",
    )
    """Maximum base64 payload size in bytes (default 5 MB).

    The base64 payload is the portion after ``data:<mime>;base64,`` in a
    data URI, measured in UTF-8 bytes — matching opencode's semantics.
    """

    model_config = ConfigDict(frozen=True)
