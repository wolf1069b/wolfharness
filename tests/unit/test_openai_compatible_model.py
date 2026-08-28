"""Unit tests for :class:`OpenAICompatibleModel`.

Verifies that list-type ``ToolReturnPart`` content is mapped to native
``list[ChatCompletionContentPartTextParam]`` instead of a JSON string,
and that non-list content falls back to the parent's string serialization.

See: https://github.com/wolf1069b/wolfharness/issues/112
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

from wolfharness.models.openai_compatible import OpenAICompatibleModel


pytestmark = pytest.mark.unit


def _make_model() -> OpenAICompatibleModel:
    """Create an ``OpenAICompatibleModel`` with a mock provider.

    We avoid real network connections by mocking the provider's client.
    The ``_map_user_message`` method only accesses ``self.profile``
    (inherited from ``OpenAIChatModel``), which is initialised in
    ``__init__`` from the provider's profile resolution.
    """
    from pydantic_ai.providers.openai import OpenAIProvider

    # OpenAIProvider with a fake base_url and api_key to avoid reading
    # real environment variables.
    provider = OpenAIProvider(base_url="http://localhost:9999/v1", api_key="test-key")
    return OpenAICompatibleModel(model_name="test-model", provider=provider)


async def _collect_mapped_messages(
    model: OpenAICompatibleModel, parts: list[Any]
) -> list[dict[str, Any]]:
    """Run ``_map_user_message`` with the given parts and collect results.

    Returns a list of dicts with ``role``, ``content``, and optional
    ``tool_call_id`` keys for easy assertion.
    """
    from pydantic_ai.messages import ModelRequest

    message = ModelRequest(parts=parts)
    results: list[dict[str, Any]] = []
    async for msg in model._map_user_message(message):
        d: dict[str, Any] = {"role": msg["role"]}
        if "content" in msg:
            d["content"] = msg["content"]
        if "tool_call_id" in msg:
            d["tool_call_id"] = msg["tool_call_id"]
        results.append(d)
    return results


class TestToolReturnListContent:
    """Tests for list-type tool return content mapping."""

    async def test_list_content_becomes_native_list(self) -> None:
        """List tool return with string elements → list[ChatCompletionContentPartTextParam]."""
        from pydantic_ai.messages import ToolReturnPart

        model = _make_model()
        part = ToolReturnPart(
            tool_name="search",
            content=["result1", "result2", "result3"],
            tool_call_id="call_001",
        )

        results = await _collect_mapped_messages(model, [part])

        assert len(results) == 1
        msg = results[0]
        assert msg["role"] == "tool"
        assert isinstance(msg["content"], list)
        assert len(msg["content"]) == 3
        for item in msg["content"]:
            assert item["type"] == "text"
        assert msg["content"][0]["text"] == "result1"
        assert msg["content"][1]["text"] == "result2"
        assert msg["content"][2]["text"] == "result3"

    async def test_list_content_with_non_string_elements(self) -> None:
        """List tool return with mixed types → each element serialized to text."""
        from pydantic_ai.messages import ToolReturnPart

        model = _make_model()
        part = ToolReturnPart(
            tool_name="query",
            content=[{"key": "value"}, 42, "plain"],
            tool_call_id="call_002",
        )

        results = await _collect_mapped_messages(model, [part])

        assert len(results) == 1
        msg = results[0]
        assert isinstance(msg["content"], list)
        assert len(msg["content"]) == 3
        assert msg["content"][0]["text"] == '{"key":"value"}'
        assert msg["content"][1]["text"] == "42"
        assert msg["content"][2]["text"] == "plain"

    async def test_single_element_list_still_uses_list(self) -> None:
        """Single-element list content → still native list (was_list=True)."""
        from pydantic_ai.messages import ToolReturnPart

        model = _make_model()
        part = ToolReturnPart(
            tool_name="single",
            content=["only-result"],
            tool_call_id="call_003",
        )

        results = await _collect_mapped_messages(model, [part])

        assert len(results) == 1
        msg = results[0]
        assert isinstance(msg["content"], list)
        assert len(msg["content"]) == 1
        assert msg["content"][0]["text"] == "only-result"

    async def test_empty_list_falls_back_to_string(self) -> None:
        """Empty list content → falls back to string (no text items to emit)."""
        from pydantic_ai.messages import ToolReturnPart

        model = _make_model()
        part = ToolReturnPart(
            tool_name="empty",
            content=[],
            tool_call_id="call_004",
        )

        results = await _collect_mapped_messages(model, [part])

        assert len(results) == 1
        msg = results[0]
        # Empty list → content_items returns [] → no list_content → fallback to string
        assert isinstance(msg["content"], str)


class TestFallbackToStringContent:
    """Tests verifying non-list content falls back to string serialization."""

    async def test_string_content_uses_string(self) -> None:
        """String tool return → string content (parent behavior)."""
        from pydantic_ai.messages import ToolReturnPart

        model = _make_model()
        part = ToolReturnPart(
            tool_name="echo",
            content="hello world",
            tool_call_id="call_010",
        )

        results = await _collect_mapped_messages(model, [part])

        assert len(results) == 1
        msg = results[0]
        assert msg["role"] == "tool"
        assert isinstance(msg["content"], str)
        assert msg["content"] == "hello world"

    async def test_dict_content_uses_string(self) -> None:
        """Dict tool return → JSON string content (parent behavior)."""
        from pydantic_ai.messages import ToolReturnPart

        model = _make_model()
        part = ToolReturnPart(
            tool_name="get_data",
            content={"status": "ok", "count": 5},
            tool_call_id="call_011",
        )

        results = await _collect_mapped_messages(model, [part])

        assert len(results) == 1
        msg = results[0]
        assert isinstance(msg["content"], str)
        assert '"status"' in msg["content"]
        assert '"ok"' in msg["content"]

    async def test_integer_content_uses_string(self) -> None:
        """Integer tool return → string content (parent behavior)."""
        from pydantic_ai.messages import ToolReturnPart

        model = _make_model()
        part = ToolReturnPart(
            tool_name="count",
            content=42,
            tool_call_id="call_012",
        )

        results = await _collect_mapped_messages(model, [part])

        assert len(results) == 1
        msg = results[0]
        assert isinstance(msg["content"], str)

    async def test_failed_tool_return_falls_back_to_string(self) -> None:
        """Failed tool return with list content → string content (error wrapping)."""
        from pydantic_ai.messages import ToolReturnPart

        model = _make_model()
        part = ToolReturnPart(
            tool_name="failing",
            content=["error detail 1", "error detail 2"],
            tool_call_id="call_013",
        )
        part.outcome = "failed"

        results = await _collect_mapped_messages(model, [part])

        assert len(results) == 1
        msg = results[0]
        # Failed returns fall back to string for error wrapping
        assert isinstance(msg["content"], str)


class TestMultiplePartsInMessage:
    """Tests for messages with multiple parts (system, user, tool returns)."""

    async def test_mixed_parts_preserve_order(self) -> None:
        """System + tool return + user prompt → correct ordering and types."""
        from pydantic_ai.messages import (
            SystemPromptPart,
            ToolReturnPart,
            UserPromptPart,
        )

        model = _make_model()
        parts = [
            SystemPromptPart(content="You are helpful."),
            ToolReturnPart(
                tool_name="search",
                content=["result1", "result2"],
                tool_call_id="call_020",
            ),
            UserPromptPart(content="What did you find?"),
        ]

        results = await _collect_mapped_messages(model, parts)

        assert len(results) == 3
        # System message
        assert results[0]["role"] == "system"
        assert results[0]["content"] == "You are helpful."
        # Tool message with list content
        assert results[1]["role"] == "tool"
        assert isinstance(results[1]["content"], list)
        # User message
        assert results[2]["role"] == "user"

    async def test_multiple_tool_returns(self) -> None:
        """Multiple tool returns in one message → each gets its own mapping."""
        from pydantic_ai.messages import ToolReturnPart

        model = _make_model()
        parts = [
            ToolReturnPart(
                tool_name="tool_a",
                content=["a1", "a2"],
                tool_call_id="call_030",
            ),
            ToolReturnPart(
                tool_name="tool_b",
                content="simple string",
                tool_call_id="call_031",
            ),
        ]

        results = await _collect_mapped_messages(model, list(parts))

        assert len(results) == 2
        # First tool return: list content
        assert results[0]["role"] == "tool"
        assert isinstance(results[0]["content"], list)
        assert len(results[0]["content"]) == 2
        # Second tool return: string content
        assert results[1]["role"] == "tool"
        assert isinstance(results[1]["content"], str)


class TestModelHelperIntegration:
    """Tests verifying model_helpers uses OpenAICompatibleModel."""

    def test_get_openai_based_model_returns_compatible(self) -> None:
        """``_get_openai_based_model`` returns ``OpenAICompatibleModel``."""
        from wolfharness.utils.model_helpers import _get_openai_based_model

        model = _get_openai_based_model(
            "deepseek:test-model",
            base_url="http://localhost:9999/v1",
            api_key="test-key",
        )
        assert isinstance(model, OpenAICompatibleModel)

    def test_deepseek_prefix_uses_compatible(self) -> None:
        """``deepseek:`` prefix creates ``OpenAICompatibleModel``."""
        from wolfharness.utils.model_helpers import _infer_single_model

        with patch.dict("os.environ", {"DEEPSEEK_API_KEY": "test"}):
            model = _infer_single_model("deepseek:test-model")
        assert isinstance(model, OpenAICompatibleModel)

    def test_is_subclass_of_openai_chat_model(self) -> None:
        """``OpenAICompatibleModel`` is a subclass of ``OpenAIChatModel``."""
        from pydantic_ai.models.openai import OpenAIChatModel

        assert issubclass(OpenAICompatibleModel, OpenAIChatModel)
