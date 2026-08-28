"""OpenAI-compatible model with native list tool return support.

Subclasses :class:`OpenAIChatModel` to send ``ToolReturnPart`` list content
as native ``list[ChatCompletionContentPartTextParam]`` instead of a JSON
string, which is required by OpenAI-compatible models (e.g. GLM-5) whose
chat templates branch on ``tool`` role ``content`` being a string vs list.

See: https://github.com/wolf1069b/wolfharness/issues/112
"""

from __future__ import annotations

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import AsyncIterable

from openai.types import chat
from openai.types.chat import ChatCompletionContentPartTextParam
from pydantic_ai._utils import guard_tool_call_id as _guard_tool_call_id
from pydantic_ai.messages import (
    RetryPromptPart,
    SystemPromptPart,
    ToolReturnPart,
    UserContent,
    UserPromptPart,
    is_multi_modal_content,
)
from pydantic_ai.models.openai import OpenAIChatModel


if TYPE_CHECKING:
    from pydantic_ai.messages import ModelRequest
    from pydantic_ai.providers.openai import OpenAIProvider


class OpenAICompatibleModel(OpenAIChatModel):
    """``OpenAIChatModel`` subclass for OpenAI-compatible providers.

    Unlike the base class, when a :class:`ToolReturnPart` has ``list``
    content and no multimodal files, the tool message ``content`` is sent
    as ``list[ChatCompletionContentPartTextParam]`` (one text part per
    list element) instead of a single JSON-serialized string.

    This ensures models with list-aware chat templates (e.g. GLM-5) can
    render each list element as a separate ``<|tool_return|>`` block.

    Non-list content, multimodal file content, and failed tool returns
    fall back to the parent's string serialization.
    """

    async def _map_user_message(
        self, message: ModelRequest
    ) -> AsyncIterable[chat.ChatCompletionMessageParam]:
        """Map a model request to OpenAI chat completion message params.

        Identical to the parent implementation except for the
        ``ToolReturnPart`` branch: when content is a ``list`` with no
        files, yields a tool message with native list content.
        """
        file_content: list[UserContent] = []
        for part in message.parts:
            if isinstance(part, SystemPromptPart):
                system_prompt_role = self.profile.get("openai_system_prompt_role", None)
                if system_prompt_role == "developer":
                    yield chat.ChatCompletionDeveloperMessageParam(
                        role="developer", content=part.content
                    )
                elif system_prompt_role == "user":
                    yield chat.ChatCompletionUserMessageParam(role="user", content=part.content)
                else:
                    yield chat.ChatCompletionSystemMessageParam(role="system", content=part.content)
            elif isinstance(part, UserPromptPart):
                yield await self._map_user_prompt(part)
            elif isinstance(part, ToolReturnPart):
                tool_text, tool_file_content = part.model_response_str_and_user_content()
                file_content.extend(tool_file_content)

                # Use native list content when:
                # 1. Content is a list (not scalar/dict)
                # 2. No multimodal files were extracted
                # 3. Tool return was not a failure (errors need wrapping)
                if (
                    not tool_file_content
                    and isinstance(part.content, list)
                    and part.outcome != "failed"
                ):
                    text_items = part.content_items(mode="str", wrap_if_error=False)
                    list_content: list[ChatCompletionContentPartTextParam] = [
                        ChatCompletionContentPartTextParam(type="text", text=item)
                        for item in text_items
                        if not is_multi_modal_content(item) and isinstance(item, str)
                    ]
                    if list_content:
                        yield chat.ChatCompletionToolMessageParam(
                            role="tool",
                            tool_call_id=_guard_tool_call_id(t=part),
                            content=list_content,
                        )
                    else:
                        yield chat.ChatCompletionToolMessageParam(
                            role="tool",
                            tool_call_id=_guard_tool_call_id(t=part),
                            content=tool_text,
                        )
                else:
                    yield chat.ChatCompletionToolMessageParam(
                        role="tool",
                        tool_call_id=_guard_tool_call_id(t=part),
                        content=tool_text,
                    )
            elif isinstance(part, RetryPromptPart):
                if part.tool_name is None:
                    yield chat.ChatCompletionUserMessageParam(
                        role="user", content=part.model_response()
                    )
                else:
                    yield chat.ChatCompletionToolMessageParam(
                        role="tool",
                        tool_call_id=_guard_tool_call_id(t=part),
                        content=part.model_response(),
                    )
            else:
                from typing import assert_never

                assert_never(part)
        if file_content:
            yield await self._map_user_prompt(UserPromptPart(content=file_content))


def create_openai_compatible_model(
    model_name: str,
    *,
    provider: OpenAIProvider,
) -> OpenAICompatibleModel:
    """Factory for creating an ``OpenAICompatibleModel`` instance.

    Args:
        model_name: The model identifier (without provider prefix).
        provider: An ``OpenAIProvider`` configured with the appropriate
            base_url and api_key.

    Returns:
        An ``OpenAICompatibleModel`` instance.
    """
    return OpenAICompatibleModel(model_name=model_name, provider=provider)
