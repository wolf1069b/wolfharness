"""Common/shared models used across multiple domains."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Self

from pydantic import Field

from wolfharness.utils.time_utils import now_ms
from wolfharness_server.opencode_server.models.base import OpenCodeBaseModel


if TYPE_CHECKING:
    from pydantic_ai.usage import UsageBase

    from wolfharness.utils.streams import FileChange

FileDiffStatus = Literal["added", "deleted", "modified"]


class TimeCreatedUpdated(OpenCodeBaseModel):
    """Timestamp with created and updated fields (milliseconds)."""

    created: int
    updated: int


class TimeCreated(OpenCodeBaseModel):
    """Timestamp with created field only (milliseconds)."""

    created: int

    @classmethod
    def now(cls) -> Self:
        return cls(created=now_ms())


class TimeStartEnd(OpenCodeBaseModel):
    """Timestamp with start and optional end (milliseconds)."""

    start: int
    end: int | None = None


class ModelRef(OpenCodeBaseModel):
    """Reference to a provider model with optional variant.

    OpenCode v1.4.0+ nests ``variant`` inside the ``model`` object:
    ``{ providerID?, modelID?, variant? }``.

    All fields are optional to support partial references — e.g. a message
    that only specifies a ``variant`` (thinking effort level) without
    changing the provider or model.
    """

    provider_id: str | None = None
    model_id: str | None = None
    variant: str | None = None
    """Reasoning/thinking variant for this model (e.g. 'low', 'medium', 'high', 'max')."""


class TokenCache(OpenCodeBaseModel):
    """Prompt-cache token counts."""

    read: int = 0
    write: int = 0


class Tokens(OpenCodeBaseModel):
    """Token usage for one assistant message.

    The TUI computes context-window fill as
    ``input + output + reasoning + cache.read + cache.write``.
    """

    input: int = 0
    output: int = 0
    reasoning: int = 0
    cache: TokenCache = Field(default_factory=TokenCache)
    total: int | None = None

    @classmethod
    def from_pydantic_ai(cls, usage: UsageBase) -> Tokens:
        """Create from a pydantic-ai Usage object.

        Args:
            usage: pydantic-ai request usage with token counts.
        """
        reasoning = usage.details.get("reasoning_tokens", 0)
        return cls(
            input=usage.input_tokens,
            output=usage.output_tokens,
            reasoning=reasoning,
            cache=TokenCache(read=usage.cache_read_tokens, write=usage.cache_write_tokens),
            total=usage.total_tokens + reasoning,
        )


class TextSpan(OpenCodeBaseModel):
    """A text span in user input (value + start/end offsets)."""

    value: str
    start: int
    end: int


class FileDiff(OpenCodeBaseModel):
    """A file diff entry.

    Matches the OpenCode v1.4.0+ SnapshotFileDiff schema:
    ``{ file, patch, additions, deletions, status? }``
    """

    file: str
    patch: str | None = None
    additions: int
    deletions: int
    status: FileDiffStatus | None = None

    @classmethod
    def from_file_change(cls, change: FileChange) -> Self:
        """Create a FileDiff from a FileChange."""
        diff_text = change.to_unified_diff()
        match change.operation:
            case "create":
                status: FileDiffStatus | None = "added"
            case "delete":
                status = "deleted"
            case "edit" | "write":
                status = "modified"
            case _:
                status = None
        return cls(
            file=change.path,
            patch=diff_text,
            additions=diff_text.count("\n+"),
            deletions=diff_text.count("\n-"),
            status=status,
        )
