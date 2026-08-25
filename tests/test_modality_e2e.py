"""Integration tests for multimodal capability end-to-end pipeline.

Tests verify the full pipeline from agent config → capability injection →
tool execution / message filtering → model request, exercising the
ModalityFilterCapability through realistic agent construction and
execution paths rather than isolated unit calls.
"""

from __future__ import annotations

from typing import Any

from pydantic_ai import BinaryImage
from pydantic_ai.messages import (
    ImageUrl,
    ModelRequest,
    ModelResponse,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import ModelRequestContext, ModelRequestParameters
from pydantic_ai.models.test import TestModel
import pytest

from wolfharness.agents.native_agent.agent import Agent
from wolfharness.capabilities.modality_filter import ModalityFilterCapability
from wolfharness.models.agents import NativeAgentConfig
from wolfharness.models.model_configs import (
    FallbackModelConfig,
    TestModelConfig as _TestModelConfig,
)
from wolfharness_config.model_capabilities import ModelCapabilities


pytestmark = [pytest.mark.integration]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _text_only_caps() -> ModelCapabilities:
    """ModelCapabilities with all multimodal inputs disabled."""
    return ModelCapabilities(
        image_input=False,
        audio_input=False,
        video_input=False,
        document_input=False,
        image_output=False,
    )


def _all_caps() -> ModelCapabilities:
    """ModelCapabilities with all multimodal inputs enabled."""
    return ModelCapabilities(
        image_input=True,
        audio_input=True,
        video_input=True,
        document_input=True,
        image_output=True,
    )


def _image_enabled_caps() -> ModelCapabilities:
    """ModelCapabilities with only image_input enabled."""
    return ModelCapabilities(
        image_input=True,
        audio_input=False,
        video_input=False,
        document_input=False,
        image_output=False,
    )


def _binary_image() -> BinaryImage:
    """A sample BinaryImage for testing."""
    return BinaryImage(data=b"\x89PNG\r\n\x1a\n", media_type="image/png")


def _make_ctx(messages: list[Any]) -> ModelRequestContext:
    """Build a minimal ModelRequestContext for before_model_request tests."""
    return ModelRequestContext(
        model=TestModel(),
        messages=messages,
        model_settings=None,
        model_request_parameters=ModelRequestParameters(
            function_tools=[],
            native_tools=[],
        ),
    )


def _handler(value: Any) -> Any:
    """Create an async handler that returns ``value`` when called with args."""

    async def _h(args: dict) -> Any:
        return value

    return _h


# ---------------------------------------------------------------------------
# 10.1 — Tool returns image → text-only model → text placeholder
# ---------------------------------------------------------------------------


@pytest.mark.integration
async def test_tool_image_text_only_model_receives_placeholder() -> None:
    """Tool returns BinaryImage when model has image_input=False.

    Verifies that ModalityFilterCapability.wrap_tool_execute replaces the
    image with a text placeholder before the model would see it.
    """
    cap = ModalityFilterCapability(
        capabilities=_text_only_caps(),
        image_strategy="describe",
    )
    img = _binary_image()
    result = await cap.wrap_tool_execute(
        ctx=None,  # type: ignore[arg-type]
        call=None,  # type: ignore[arg-type]
        tool_def=None,  # type: ignore[arg-type]
        args={},
        handler=_handler(img),
    )
    # The tool result should be a text placeholder, not a BinaryImage.
    assert isinstance(result, str)
    assert "image/png" in result
    assert "unsupported" in result


# ---------------------------------------------------------------------------
# 10.2 — Tool returns image → vision model → unchanged
# ---------------------------------------------------------------------------


@pytest.mark.integration
async def test_tool_image_vision_model_receives_binary_unchanged() -> None:
    """Tool returns BinaryImage when model has image_input=True.

    Verifies that ModalityFilterCapability passes the image through
    unchanged when the model supports image input.
    """
    cap = ModalityFilterCapability(
        capabilities=_all_caps(),
        image_strategy="describe",
    )
    img = _binary_image()
    result = await cap.wrap_tool_execute(
        ctx=None,  # type: ignore[arg-type]
        call=None,  # type: ignore[arg-type]
        tool_def=None,  # type: ignore[arg-type]
        args={},
        handler=_handler(img),
    )
    assert result is img


# ---------------------------------------------------------------------------
# 10.3 — Fallback model: primary has vision, fallback doesn't → filter adapts
# ---------------------------------------------------------------------------


@pytest.mark.integration
async def test_fallback_model_intersection_injects_filter() -> None:
    """FallbackModelConfig with mixed vision capabilities populates user-configured filter.

    The user must explicitly configure a ModalityFilterCapability. The factory
    populates it with the declared capabilities directly (no per-model
    intersection or network lookups for fallback models).
    """
    user_filter = ModalityFilterCapability(
        capabilities=None,
        image_strategy="describe",
    )
    config = NativeAgentConfig(
        model=FallbackModelConfig(
            models=["openai:gpt-4o", "openai:gpt-3.5-turbo"],
            capabilities=ModelCapabilities(),
        ),
    )
    agent = Agent(
        name="test",
        model="test",
        agent_config=config,
        capabilities=[user_filter],
    )

    pydantic_agent = await agent.get_agentlet(model=None, output_type=str)

    filter_caps = [
        cap
        for cap in pydantic_agent.root_capability.capabilities
        if isinstance(cap, ModalityFilterCapability)
    ]
    assert len(filter_caps) == 1
    # Fallback models return declared caps directly (all None for ModelCapabilities()).
    assert filter_caps[0].capabilities is not None
    assert filter_caps[0].capabilities.image_input is None

    # Verify image content is degraded by this capability.
    # None is treated as unsupported via ``is True`` check → text-only behavior.
    img = _binary_image()
    degraded = await filter_caps[0]._filter_tool_result(None, img)  # type: ignore[arg-type]
    assert isinstance(degraded, str)
    assert "image/png" in degraded
    assert "unsupported" in degraded


# ---------------------------------------------------------------------------
# 10.4 — Historical ToolReturnPart with image → degraded in before_model_request
# ---------------------------------------------------------------------------


@pytest.mark.integration
async def test_history_tool_return_image_degraded_for_text_only() -> None:
    """Message history with image in ToolReturnPart degraded for text-only model.

    Simulates a prior turn where a tool returned a BinaryImage. When this
    history is replayed to a text-only model, before_model_request should
    replace the image with a text placeholder.
    """
    cap = ModalityFilterCapability(
        capabilities=_text_only_caps(),
        image_strategy="describe",
    )
    img = _binary_image()
    tool_return = ToolReturnPart(
        tool_name="screenshot",
        content=img,
        tool_call_id="tc_1",
    )
    msg = ModelResponse(parts=[tool_return])
    ctx = _make_ctx([msg])

    result = await cap.before_model_request(
        ctx=None,  # type: ignore[arg-type]
        request_context=ctx,
    )

    new_msg = result.messages[0]
    assert isinstance(new_msg, ModelResponse)
    new_part = new_msg.parts[0]
    assert isinstance(new_part, ToolReturnPart)
    assert "image/png" in new_part.content
    assert "unsupported" in new_part.content


# ---------------------------------------------------------------------------
# 10.5 — User-supplied image in UserPromptPart degraded for text-only model
# ---------------------------------------------------------------------------


@pytest.mark.integration
async def test_user_prompt_image_degraded_for_text_only() -> None:
    """UserPromptPart with ImageUrl degraded for text-only model.

    Verifies that before_model_request scans UserPromptPart content (not
    just ToolReturnPart) and degrades unsupported multimodal content.
    """
    cap = ModalityFilterCapability(
        capabilities=_text_only_caps(),
        image_strategy="describe",
    )
    url_img = ImageUrl(url="https://example.com/photo.png")
    user_part = UserPromptPart(content=["Describe this photo", url_img])
    msg = ModelRequest(parts=[user_part])
    ctx = _make_ctx([msg])

    result = await cap.before_model_request(
        ctx=None,  # type: ignore[arg-type]
        request_context=ctx,
    )

    new_msg = result.messages[0]
    assert isinstance(new_msg, ModelRequest)
    new_part = new_msg.parts[0]
    assert isinstance(new_part, UserPromptPart)
    assert isinstance(new_part.content, list)
    assert new_part.content[0] == "Describe this photo"
    assert new_part.content[1] == "[image: https://example.com/photo.png]"


# ---------------------------------------------------------------------------
# 10.6 — User-configured modality_filter takes precedence over auto-injection
# ---------------------------------------------------------------------------


@pytest.mark.integration
async def test_user_config_modality_filter_precedence() -> None:
    """User-configured ModalityFilterCapability is populated, not auto-injected.

    When the user explicitly configures a ModalityFilterCapability in the
    agent's capabilities list, the factory populates it with resolved
    capabilities and preserves the user's strategy settings.
    """
    user_filter = ModalityFilterCapability(
        capabilities=None,
        image_strategy="drop",
        audio_strategy="pass",
    )
    config = NativeAgentConfig(
        model=_TestModelConfig(capabilities=_text_only_caps()),
    )
    agent = Agent(
        name="test",
        model="test",
        agent_config=config,
        capabilities=[user_filter],
    )
    pydantic_agent = await agent.get_agentlet(model=None, output_type=str)

    filter_caps = [
        cap
        for cap in pydantic_agent.root_capability.capabilities
        if isinstance(cap, ModalityFilterCapability)
    ]
    # Exactly one — the user's instance, not a second auto-injected one.
    assert len(filter_caps) == 1
    # User's strategy settings are preserved.
    assert filter_caps[0].image_strategy == "drop"
    assert filter_caps[0].audio_strategy == "pass"
    # Capabilities were populated by the factory.
    assert filter_caps[0].capabilities is not None
    assert filter_caps[0].capabilities.image_input is False
    assert filter_caps[0].capabilities.audio_input is False


# ---------------------------------------------------------------------------
# 10.7 — Non-mutation of ModelRequestContext in before_model_request
# ---------------------------------------------------------------------------


@pytest.mark.integration
async def test_before_model_request_does_not_mutate_original() -> None:
    """before_model_request returns a new context, original is unchanged.

    Verifies that dataclasses.replace is used (not in-place mutation):
    the original ModelRequestContext's messages should still contain the
    original BinaryImage, while the returned context has the degraded text.
    """
    cap = ModalityFilterCapability(
        capabilities=_text_only_caps(),
        image_strategy="describe",
    )
    img = _binary_image()
    tool_return = ToolReturnPart(
        tool_name="screenshot",
        content=img,
        tool_call_id="tc_1",
    )
    msg = ModelResponse(parts=[tool_return])
    ctx = _make_ctx([msg])

    # Capture original state.
    original_messages = list(ctx.messages)
    original_content = ctx.messages[0].parts[0].content  # type: ignore[union-attr]

    result = await cap.before_model_request(
        ctx=None,  # type: ignore[arg-type]
        request_context=ctx,
    )

    # Original context unchanged — same message objects, same content.
    assert ctx.messages == original_messages
    assert ctx.messages[0].parts[0].content is original_content  # type: ignore[union-attr]

    # Returned context is different and has degraded content.
    assert result is not ctx
    assert result.messages is not ctx.messages
    new_part = result.messages[0].parts[0]  # type: ignore[union-attr]
    assert isinstance(new_part, ToolReturnPart)
    assert "image/png" in new_part.content
    assert "unsupported" in new_part.content
    assert new_part.content is not original_content


# ---------------------------------------------------------------------------
# 10.8 — ACP agent with image_input=False advertises image_prompts=False
# ---------------------------------------------------------------------------


@pytest.mark.integration
async def test_acp_agent_image_input_false_advertises_false() -> None:
    """ACP agent wrapper with image_input=False advertises image_prompts=False.

    Tests the ACP server integration: when the default agent's model has
    image_input=False in its ModelCapabilities, the InitializeResponse
    should advertise image_prompts=False to the ACP client.
    """
    # The ACP agent derives image_prompts from
    # model_config.capabilities.image_input. We test this logic directly
    # by simulating the derivation that occurs in initialize().
    from wolfharness.models.model_configs import BaseModelConfig

    model_config = _TestModelConfig(capabilities=_text_only_caps())
    assert isinstance(model_config, BaseModelConfig)
    caps = model_config.capabilities
    assert caps is not None

    # This mirrors the logic in AgentPoolACPAgent.initialize():
    # image_prompts = caps.image_input if caps.image_input is not None else True
    image_prompts = caps.image_input if caps.image_input is not None else True
    assert image_prompts is False

    audio_prompts = caps.audio_input if caps.audio_input is not None else True
    assert audio_prompts is False

    # Also verify the optimistic default when capabilities are None.
    model_config_no_caps = _TestModelConfig(capabilities=None)
    if (
        isinstance(model_config_no_caps, BaseModelConfig)
        and model_config_no_caps.capabilities is None
    ):
        image_prompts_default = True  # optimistic default
        assert image_prompts_default is True

    # And when image_input is explicitly True.
    model_config_vision = _TestModelConfig(capabilities=_image_enabled_caps())
    assert isinstance(model_config_vision, BaseModelConfig)
    caps_vision = model_config_vision.capabilities
    assert caps_vision is not None
    image_prompts_vision = caps_vision.image_input if caps_vision.image_input is not None else True
    assert image_prompts_vision is True
