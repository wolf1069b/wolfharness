"""Utility functions for model inference.

Replacement for llmling_models.models.helpers and llmling_models.models.function_to_model.
"""

from __future__ import annotations

import functools
import inspect
import logging
import os
from typing import TYPE_CHECKING, Any

import anyenv
from pydantic import BaseModel
from pydantic_ai import ModelResponse, TextPart, ToolCallPart, messages
from pydantic_ai.models import Model, infer_model as infer_model_
from pydantic_ai.models.function import (
    AgentInfo,
    BuiltinToolCallsReturns,
    DeltaThinkingCalls,
    DeltaToolCall,
    DeltaToolCalls,
    FunctionModel,
)
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIResponsesModel
from pydantic_ai.providers.openai import OpenAIProvider


logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable

    from pydantic_ai import ModelMessage, ModelResponsePart


def _get_openai_based_model(
    model: str, base_url: str | None = None, api_key: str | None = None
) -> Model:
    """Get model instance with appropriate implementation based on environment."""
    model_name = model

    if ":" in model:
        _, model_name = model.split(":", 1)

    provider = OpenAIProvider(base_url=base_url, api_key=api_key)
    return OpenAIChatModel(model_name=model_name, provider=provider)


def infer_model(model: str | Model) -> Model:
    """Extended infer_model from pydantic-ai with fallback support.

    For fallback models, use comma-separated model names.
    Example: "openai:gpt-4,openai:gpt-3.5-turbo"
    """
    from pydantic_ai.models.fallback import FallbackModel

    # If model is already a Model instance or something else not string
    if not isinstance(model, str):
        return model

    # Check for comma-separated model list (fallback case)
    if "," in model:
        model_names = [name.strip() for name in model.split(",")]
        if len(model_names) <= 1:
            return _infer_single_model(model)

        # Create fallback model chain
        default_model = _infer_single_model(model_names[0])
        fallback_models = [_infer_single_model(m) for m in model_names[1:]]
        return FallbackModel(default_model, *fallback_models)

    # Regular single model case
    return _infer_single_model(model)


def _infer_single_model(model: str | Model) -> Model:  # noqa: PLR0911
    """Extended infer_model from pydantic-ai."""
    if not isinstance(model, str):
        return model

    if model.startswith("openrouter:"):
        key = os.getenv("OPENROUTER_API_KEY")
        return _get_openai_based_model(model, base_url="https://openrouter.ai/api/v1", api_key=key)
    if model.startswith("grok:"):
        key = os.getenv("X_AI_API_KEY") or os.getenv("GROK_API_KEY")
        return _get_openai_based_model(model, base_url="https://api.x.ai/v1", api_key=key)
    if model.startswith("deepseek:"):
        key = os.getenv("DEEPSEEK_API_KEY")
        return _get_openai_based_model(model, base_url="https://api.deepseek.com", api_key=key)
    if model.startswith("perplexity:"):
        key = os.getenv("PERPLEXITY_API_KEY")
        return _get_openai_based_model(model, base_url="https://api.perplexity.ai", api_key=key)
    if model.startswith("lm-studio:"):
        return _get_openai_based_model(
            model, base_url="http://localhost:1234/v1/", api_key="lm-studio"
        )
    if model.startswith("openai:"):
        # Use OpenAIResponsesModel (Responses API) to match pydantic-ai's default.
        # Users who need the Chat Completions API can use the "openai-chat:" prefix,
        # which falls through to pydantic-ai's built-in infer_model.
        model_name = model.split(":", 1)[1]
        openai_provider = OpenAIProvider()
        return OpenAIResponsesModel(model_name=model_name, provider=openai_provider)
    if model.startswith("zen:"):
        return _get_openai_based_model(model)
    if model.startswith("anthropic-max:"):
        from pydantic_ai.models.anthropic import AnthropicModel

        from wolfharness.auth.anthropic_auth import AnthropicMaxProvider

        provider = AnthropicMaxProvider()
        model_name = model.removeprefix("anthropic-max:")
        return AnthropicModel(model_name=model_name, provider=provider)  # type: ignore[arg-type]

    if model.startswith("copilot:"):
        from httpx import AsyncClient

        token = os.getenv("GITHUB_COPILOT_API_KEY")
        headers = {
            "Authorization": f"Bearer {token}",
            "editor-version": "Neovim/0.9.0",
            "Copilot-Integration-Id": "vscode-chat",
        }
        client = AsyncClient(headers=headers)
        base_url = "https://api.githubcopilot.com"
        prov = OpenAIProvider(base_url=base_url, api_key=token, http_client=client)
        model_name = model.removeprefix("copilot:")
        return OpenAIChatModel(model_name=model_name, provider=prov)

    if model.startswith("import:"):
        from wolfharness.utils.importing import import_callable

        imported = import_callable(model.removeprefix("import:"))
        return imported() if isinstance(imported, type) else imported  # type: ignore[return-value]
    if model == "test":
        from pydantic_ai.models.test import TestModel

        return TestModel()
    if model.startswith("test:"):
        from pydantic_ai.models.test import TestModel

        return TestModel(custom_output_text=model.removeprefix("test:"))
    if model.startswith("gemini:"):
        model = model.replace("gemini:", "google-gla:")
    return infer_model_(model)


def format_part(  # noqa: PLR0911
    response: str | messages.ModelRequestPart | messages.ModelResponsePart,
) -> str:
    """Format any kind of response part in a readable way.

    Args:
        response: Response part to format

    Returns:
        A human-readable string representation
    """
    from pydantic_ai import BinaryContent, FileUrl

    match response:
        case str():
            return response
        case messages.ToolCallPart(args=args, tool_name=tool_name):
            return f"Tool call: {tool_name}\nArgs: {args}"
        case messages.ToolReturnPart(tool_name=tool_name, content=content):
            return f"Tool {tool_name} returned: {content}"
        case messages.RetryPromptPart(content=content) if isinstance(content, str):
            return f"Retry needed: {content}"
        case messages.RetryPromptPart(content=content):
            return f"Validation errors:\n{content}"
        case messages.UserPromptPart(content=content) if not isinstance(content, str):
            texts = []
            for item in content:
                match item:
                    case str():
                        texts.append(f"{item}")
                    case FileUrl(url=url):
                        texts.append(f"{url}")
                    case BinaryContent(identifier=identifier):
                        texts.append(f"Binary content: <{identifier}>")
            return "\n".join(texts)
        case (
            messages.SystemPromptPart(content=content)
            | messages.UserPromptPart(content=content)
            | messages.TextPart(content=content)
        ) if isinstance(content, str):
            return content
        case _:
            return str(response)


def function_to_model(
    callback: Callable[..., Any],
    streamable: bool = True,
) -> FunctionModel:
    """Factory to get a text model for Callables with "simpler" signatures.

    This function serves as a helper to allow creating FunctionModels which take either
    no arguments or a single argument in form of a prompt.
    """
    sig = inspect.signature(callback)
    # Count required parameters (those without defaults)
    required_params = sum(
        1 for param in sig.parameters.values() if param.default is inspect.Parameter.empty
    )
    takes_prompt = required_params > 0

    @functools.wraps(callback)
    async def callback_wrapper(
        messages: list[ModelMessage], agent_info: AgentInfo
    ) -> ModelResponse:
        try:
            if takes_prompt:
                prompt = format_part(messages[-1].parts[-1])
                if inspect.iscoroutinefunction(callback):
                    result = await callback(prompt)
                else:
                    result = callback(prompt)
            elif inspect.iscoroutinefunction(callback):
                result = await callback()
            else:
                result = callback()

            if isinstance(result, str):
                part: ModelResponsePart = TextPart(result)
            # For structured responses, check if agent expects structured output
            elif agent_info.allow_text_output:
                # Agent expects text - serialize the structured result
                serialized = (
                    anyenv.dump_json(result.model_dump())
                    if isinstance(result, BaseModel)
                    else str(result)
                )
                part = TextPart(serialized)
            else:
                # Agent expects structured output - return as ToolCallPart
                part = ToolCallPart(tool_name="final_result", args=result.model_dump())
            return ModelResponse(parts=[part])
        except Exception as e:
            logger.exception("Processor callback failed")
            name = getattr(callback, "__name__", str(callback))
            msg = f"Processor error in {name!r}: {e}"
            raise RuntimeError(msg) from e

    async def stream_function(
        messages: list[ModelMessage], agent_info: AgentInfo
    ) -> AsyncIterator[str | DeltaToolCalls | DeltaThinkingCalls | BuiltinToolCallsReturns]:
        result = await callback_wrapper(messages, agent_info)
        part = result.parts[0]
        match part:
            case TextPart():
                yield part.content
            case ToolCallPart():
                args_json = anyenv.dump_json(part.args) if part.args else "{}"
                yield {0: DeltaToolCall(name=part.tool_name, json_args=args_json)}
            case _:
                msg = f"Unexpected part type: {type(part)}"
                raise ValueError(msg)

    kwargs: dict[str, Any] = {"stream_function": stream_function} if streamable else {}
    return FunctionModel(function=callback_wrapper, **kwargs)
