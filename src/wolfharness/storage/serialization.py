"""Serialization utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import ConfigDict, TypeAdapter
from pydantic_ai import ModelMessage, ModelRequestPart, ModelResponsePart
from pydantic_ai.messages import UserContent

from wolfharness.log import get_logger
from wolfharness.sessions.models import PendingDeferredCall


if TYPE_CHECKING:
    from collections.abc import Sequence


logger = get_logger(__name__)

# Type adapter for serializing ModelResponsePart sequences (also accepts ModelRequestPart)
parts_adapter = TypeAdapter(
    list[ModelResponsePart | ModelRequestPart],
    config=ConfigDict(ser_json_bytes="base64", val_json_bytes="base64"),
)

# Type adapter for serializing ModelMessage sequences
messages_adapter = TypeAdapter(
    list[ModelMessage],
    config=ConfigDict(ser_json_bytes="base64", val_json_bytes="base64"),
)

# Type adapter for serializing PendingDeferredCall sequences
deferred_calls_adapter = TypeAdapter(list[PendingDeferredCall])

# Type adapter for serializing prompts (list[str | list[UserContent]])
prompts_adapter = TypeAdapter(
    list[str | list[UserContent]],
    config=ConfigDict(ser_json_bytes="base64", val_json_bytes="base64"),
)


def deserialize_parts(parts_json: str | None) -> Sequence[ModelResponsePart | ModelRequestPart]:
    """Deserialize pydantic-ai message parts from JSON string.

    Args:
        parts_json: JSON string representation of parts or None if empty

    Returns:
        Sequence of ModelResponsePart objects, empty if deserialization fails
    """
    if not parts_json:
        return []

    try:
        # Deserialize using pydantic's JSON deserialization
        return parts_adapter.validate_json(parts_json.encode())
    except Exception as e:  # noqa: BLE001
        logger.warning("Failed to deserialize message parts", error=e)
        return []  # Return empty list on failure


def serialize_parts(parts: Sequence[ModelResponsePart | ModelRequestPart]) -> str | None:
    """Serialize pydantic-ai message parts from ChatMessage.

    Args:
        parts: Sequence of ModelResponsePart from ChatMessage.parts

    Returns:
        JSON string representation of parts or None if empty
    """
    if not parts:
        return None

    try:
        # Convert parts to serializable format
        serializable_parts = []
        for part in parts:
            # Handle RetryPromptPart context serialization issues
            from pydantic_ai import RetryPromptPart

            if isinstance(part, RetryPromptPart) and isinstance(part.content, list):
                for content in part.content:
                    if isinstance(content, dict) and "ctx" in content:
                        content["ctx"] = {k: str(v) for k, v in content["ctx"].items()}
            serializable_parts.append(part)

        # Serialize using pydantic's JSON serialization
        return parts_adapter.dump_json(serializable_parts).decode()
    except Exception as e:  # noqa: BLE001
        logger.warning("Failed to serialize message parts", error=e)
        return None  # Don't return str() — it's garbage data


def deserialize_messages(messages_json: str | None) -> list[ModelMessage]:
    """Deserialize pydantic-ai ModelMessage list from JSON string.

    Args:
        messages_json: JSON string representation of messages or None if empty

    Returns:
        List of ModelMessage objects, empty if deserialization fails
    """
    if not messages_json:
        return []

    try:
        # Deserialize using pydantic's JSON deserialization
        return messages_adapter.validate_json(messages_json.encode())
    except Exception as e:  # noqa: BLE001
        logger.warning("Failed to deserialize model messages", error=e)
        return []  # Return empty list on failure


def serialize_messages(messages: Sequence[ModelMessage]) -> str | None:
    """Serialize pydantic-ai ModelMessage list to JSON string.

    Args:
        messages: Sequence of ModelMessage objects from ChatMessage.messages

    Returns:
        JSON string representation of messages or None if empty
    """
    if not messages:
        return None

    try:
        # Convert messages to serializable format
        serializable_messages = []
        for message in messages:
            # Handle RetryPromptPart context serialization issues
            from pydantic_ai import ModelRequest, RetryPromptPart

            if isinstance(message, ModelRequest):
                for part in message.parts:
                    if isinstance(part, RetryPromptPart) and isinstance(part.content, list):
                        for content in part.content:
                            if isinstance(content, dict) and "ctx" in content:
                                content["ctx"] = {k: str(v) for k, v in content["ctx"].items()}
            serializable_messages.append(message)

        # Serialize using pydantic's JSON serialization
        return messages_adapter.dump_json(serializable_messages).decode()
    except Exception as e:  # noqa: BLE001
        logger.warning("Failed to serialize model messages", error=e)
        return None  # Don't return str() — it's garbage data


def serialize_pending_calls(calls: list[PendingDeferredCall]) -> str:
    """Serialize PendingDeferredCall list to JSON string.

    Args:
        calls: List of PendingDeferredCall objects to serialize.

    Returns:
        JSON string representation of the calls. Empty list returns "[]".
    """
    try:
        return deferred_calls_adapter.dump_json(calls).decode()
    except Exception as e:  # noqa: BLE001
        logger.warning("Failed to serialize pending deferred calls", error=e)
        return "[]"


def deserialize_pending_calls(json_str: str | None) -> list[PendingDeferredCall]:
    """Deserialize PendingDeferredCall list from JSON string.

    Args:
        json_str: JSON string representation of pending calls or None if empty.

    Returns:
        List of PendingDeferredCall objects, empty if deserialization fails.
    """
    if not json_str:
        return []

    try:
        return deferred_calls_adapter.validate_json(json_str.encode())
    except Exception as e:  # noqa: BLE001
        logger.warning("Failed to deserialize pending deferred calls", error=e)
        return []


def serialize_prompts(prompts: list[str | list[Any]]) -> str | None:
    """Serialize a prompts list to JSON string with base64 for binary content.

    Each prompt is either a plain ``str`` or a ``list[UserContent]`` containing
    text, images, audio, video, etc. Binary content is base64-encoded.

    Args:
        prompts: List of prompts from the RunLoop — each item is a string
            or a list of UserContent blocks.

    Returns:
        JSON string representation of the prompts, or ``None`` if empty or
        serialization fails.
    """
    if not prompts:
        return None

    try:
        return prompts_adapter.dump_json(prompts).decode()
    except Exception as e:  # noqa: BLE001
        logger.warning("Failed to serialize prompts", error=e)
        return None


def deserialize_prompts(json_str: str | None) -> list[str | list[Any]]:
    """Deserialize a prompts list from JSON string.

    Reconstructs the full prompts list with BinaryContent objects restored
    from base64-encoded data.

    Args:
        json_str: JSON string representation of prompts or ``None`` if empty.

    Returns:
        List of prompts (each a ``str`` or ``list[UserContent]``), empty if
        deserialization fails.
    """
    if not json_str:
        return []

    try:
        result = prompts_adapter.validate_json(json_str.encode())
        return list(result)
    except Exception as e:  # noqa: BLE001
        logger.warning("Failed to deserialize prompts", error=e)
        return []
