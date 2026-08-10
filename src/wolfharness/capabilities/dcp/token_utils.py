"""Token estimation and context pressure utilities for DCP.

Provides character-based heuristic token estimation with optional
``tiktoken`` integration, plus a context pressure ratio calculator.

The heuristic handles mixed ASCII/CJK content by checking Unicode ranges:
CJK characters are estimated at ``chars / 2`` tokens (roughly 1 token
per character for Chinese/Japanese/Korean), while ASCII/English content
uses ``chars / 4`` (roughly 1 token per 4 characters, matching OpenAI's
rule of thumb).
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from pydantic_ai.messages import ModelMessage

# CJK Unicode ranges for character-level token estimation.
# Each CJK character roughly maps to ~0.5 tokens (2 chars per token),
# compared to ~0.25 tokens for ASCII (4 chars per token).
_CJK_RANGES: tuple[tuple[int, int], ...] = (
    (0x4E00, 0x9FFF),  # CJK Unified Ideographs
    (0x3400, 0x4DBF),  # CJK Unified Ideographs Extension A
    (0x20000, 0x2A6DF),  # CJK Unified Ideographs Extension B
    (0x2A700, 0x2B73F),  # CJK Unified Ideographs Extension C
    (0x2B740, 0x2B81F),  # CJK Unified Ideographs Extension D
    (0x2B820, 0x2CEAF),  # CJK Unified Ideographs Extension E
    (0x2CEB0, 0x2EBEF),  # CJK Unified Ideographs Extension F
    (0x30000, 0x3134F),  # CJK Unified Ideographs Extension G
    (0x3040, 0x309F),  # Hiragana
    (0x30A0, 0x30FF),  # Katakana
    (0xAC00, 0xD7AF),  # Hangul Syllables
    (0x1100, 0x11FF),  # Hangul Jamo
    (0xFF00, 0xFFEF),  # Fullwidth Forms (CJK punctuation/compatibility)
)


def _is_cjk(char: str) -> bool:
    """Check if a character is in a CJK Unicode range.

    Args:
        char: A single character string.

    Returns:
        ``True`` if the character falls within any CJK Unicode range.
    """
    code = ord(char)
    return any(low <= code <= high for low, high in _CJK_RANGES)


def _estimate_text_tokens(text: str) -> int:
    """Estimate token count for a text string using char-based heuristic.

    CJK characters are estimated at ``chars / 2`` tokens.
    ASCII/other characters are estimated at ``chars / 4`` tokens.

    Args:
        text: The input string to estimate.

    Returns:
        Estimated token count (minimum 0).
    """
    if not text:
        return 0

    cjk_count = 0
    other_count = 0
    for char in text:
        if _is_cjk(char):
            cjk_count += 1
        else:
            other_count += 1

    # CJK: ~2 chars per token; ASCII/other: ~4 chars per token
    return (cjk_count + 1) // 2 + (other_count + 3) // 4


def _estimate_tokens_with_tiktoken(messages: list[ModelMessage]) -> int | None:
    """Try to estimate tokens using ``tiktoken`` if available.

    Args:
        messages: List of model messages to estimate.

    Returns:
        Total token count, or ``None`` if ``tiktoken`` is not available.
    """
    try:
        import tiktoken
    except ImportError:
        return None

    enc = tiktoken.get_encoding("cl100k_base")
    total = 0
    for msg in messages:
        parts = getattr(msg, "parts", [])
        for part in parts:
            # Extract text content from each part type
            content = getattr(part, "content", None)
            if isinstance(content, str):
                total += len(enc.encode(content))
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, str):
                        total += len(enc.encode(item))
            elif isinstance(content, dict):
                total += len(enc.encode(json.dumps(content, ensure_ascii=False)))

            # Handle ToolCallPart.args which may be str or dict
            args = getattr(part, "args", None)
            if args is not None and not isinstance(args, str):
                # args is a dict — estimate its JSON representation
                total += len(enc.encode(json.dumps(args, ensure_ascii=False)))
            elif isinstance(args, str):
                total += len(enc.encode(args))
    return total


def _estimate_tokens_heuristic(messages: list[ModelMessage]) -> int:
    """Estimate token count using char-based heuristic.

    Iterates over all parts of all messages, extracting text content
    from ``TextPart``, ``UserPromptPart``, ``ToolReturnPart``, and
    ``ToolCallPart``.

    Args:
        messages: List of model messages to estimate.

    Returns:
        Estimated total token count.
    """
    total = 0
    for msg in messages:
        parts = getattr(msg, "parts", [])
        for part in parts:
            part_kind = getattr(part, "part_kind", None)

            if part_kind == "text":
                # TextPart.content is always str
                total += _estimate_text_tokens(getattr(part, "content", ""))
            elif part_kind == "user-prompt":
                # UserPromptPart.content is str | Sequence[UserContent]
                content = getattr(part, "content", "")
                if isinstance(content, str):
                    total += _estimate_text_tokens(content)
                elif isinstance(content, list):
                    for item in content:
                        if isinstance(item, str):
                            total += _estimate_text_tokens(item)
            elif part_kind == "tool-return":
                # ToolReturnPart.content is str, list, or other
                content = getattr(part, "content", "")
                if isinstance(content, str):
                    total += _estimate_text_tokens(content)
                elif isinstance(content, list):
                    for item in content:
                        if isinstance(item, str):
                            total += _estimate_text_tokens(item)
                        else:
                            total += _estimate_text_tokens(str(item))
                else:
                    total += _estimate_text_tokens(str(content))
            elif part_kind == "tool-call":
                # ToolCallPart.args is str | dict | None
                args = getattr(part, "args", None)
                if isinstance(args, str):
                    total += _estimate_text_tokens(args)
                elif isinstance(args, dict):
                    total += _estimate_text_tokens(json.dumps(args, ensure_ascii=False))
            elif part_kind == "retry-prompt":
                # RetryPromptPart.content is list[ErrorDetails] | str
                content = getattr(part, "content", "")
                if isinstance(content, str):
                    total += _estimate_text_tokens(content)
                elif isinstance(content, list):
                    for item in content:
                        total += _estimate_text_tokens(str(item))

    return total


def estimate_tokens(messages: list[ModelMessage]) -> int:
    """Estimate total token count for a list of model messages.

    Tries ``tiktoken`` first for accurate BPE-based counting.  If
    ``tiktoken`` is not installed, falls back to a char-based heuristic
    that handles CJK characters differently from ASCII.

    Args:
        messages: List of model messages (``ModelRequest`` or
            ``ModelResponse``) to estimate.

    Returns:
        Estimated total token count.  Returns ``0`` for empty input.
    """
    if not messages:
        return 0

    # Try tiktoken for accurate estimation
    tiktoken_count = _estimate_tokens_with_tiktoken(messages)
    if tiktoken_count is not None:
        return tiktoken_count

    # Fall back to char-based heuristic
    return _estimate_tokens_heuristic(messages)


def calculate_context_pressure(estimated_tokens: int, max_context_tokens: int) -> float:
    """Calculate context pressure as a ratio of used to maximum tokens.

    Args:
        estimated_tokens: The estimated number of tokens in use.
        max_context_tokens: The maximum context window size in tokens.

    Returns:
        Pressure ratio from ``0.0`` (empty) to ``1.0`` (full) or
        above if over capacity.  Returns ``0.0`` if
        ``max_context_tokens`` is zero (division guard).
    """
    if max_context_tokens <= 0:
        return 0.0
    return estimated_tokens / max_context_tokens
