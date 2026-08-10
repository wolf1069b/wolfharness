"""Unit tests for ModalityFilterCapability (Group 4, tasks 4.6 + 4.7)."""

from __future__ import annotations

from typing import Any

from pydantic_ai import BinaryContent, BinaryImage
from pydantic_ai.capabilities import CapabilityOrdering
from pydantic_ai.messages import (
    AudioUrl,
    ImageUrl,
    ModelRequest,
    ModelResponse,
    ToolReturnPart,
    UserPromptPart,
    VideoUrl,
)
from pydantic_ai.models import ModelRequestContext, ModelRequestParameters
from pydantic_ai.models.test import TestModel
import pytest

from wolfharness.capabilities.modality_filter import ModalityFilterCapability
from wolfharness_config.model_capabilities import ModelCapabilities


pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _handler(value: Any):
    """Create an async handler that returns ``value``."""

    async def _h(args: dict) -> Any:
        return value

    return _h


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


def _image_only_caps() -> ModelCapabilities:
    """ModelCapabilities with only image input enabled."""
    return ModelCapabilities(
        image_input=True,
        audio_input=False,
        video_input=False,
        document_input=False,
        image_output=False,
    )


def _make_ctx(messages: list) -> ModelRequestContext:
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


def _binary_image() -> BinaryImage:
    return BinaryImage(data=b"\x89PNG\r\n\x1a\n", media_type="image/png")


def _binary_audio() -> BinaryContent:
    return BinaryContent(data=b"RIFF....", media_type="audio/wav")


def _binary_video() -> BinaryContent:
    return BinaryContent(data=b"\x00\x00\x00", media_type="video/mp4")


def _binary_pdf() -> BinaryContent:
    return BinaryContent(data=b"%PDF-1.4", media_type="application/pdf")


# ---------------------------------------------------------------------------
# 4.6 — wrap_tool_execute tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_wrap_tool_image_describe_strategy() -> None:
    """Image modality with 'describe' strategy replaces with text placeholder."""
    cap = ModalityFilterCapability(capabilities=_text_only_caps(), image_strategy="describe")
    img = _binary_image()
    result = await cap.wrap_tool_execute(
        ctx=None,  # type: ignore[arg-type]
        call=None,  # type: ignore[arg-type]
        tool_def=None,  # type: ignore[arg-type]
        args={},
        handler=_handler(img),
    )
    assert result == "[image/png]"


@pytest.mark.unit
async def test_wrap_tool_image_drop_strategy() -> None:
    """Image modality with 'drop' strategy returns fallback text when sole content."""
    cap = ModalityFilterCapability(capabilities=_text_only_caps(), image_strategy="drop")
    img = _binary_image()
    result = await cap.wrap_tool_execute(
        ctx=None,  # type: ignore[arg-type]
        call=None,  # type: ignore[arg-type]
        tool_def=None,  # type: ignore[arg-type]
        args={},
        handler=_handler(img),
    )
    assert result == "[Tool returned only unsupported multimodal content]"


@pytest.mark.unit
async def test_wrap_tool_image_pass_strategy() -> None:
    """Image modality with 'pass' strategy leaves content unchanged."""
    cap = ModalityFilterCapability(capabilities=_text_only_caps(), image_strategy="pass")
    img = _binary_image()
    result = await cap.wrap_tool_execute(
        ctx=None,  # type: ignore[arg-type]
        call=None,  # type: ignore[arg-type]
        tool_def=None,  # type: ignore[arg-type]
        args={},
        handler=_handler(img),
    )
    assert result is img


@pytest.mark.unit
async def test_wrap_tool_audio_describe() -> None:
    """Audio modality is degraded to text placeholder."""
    cap = ModalityFilterCapability(capabilities=_text_only_caps(), audio_strategy="describe")
    audio = _binary_audio()
    result = await cap.wrap_tool_execute(
        ctx=None,  # type: ignore[arg-type]
        call=None,  # type: ignore[arg-type]
        tool_def=None,  # type: ignore[arg-type]
        args={},
        handler=_handler(audio),
    )
    assert result == "[audio/wav]"


@pytest.mark.unit
async def test_wrap_tool_video_describe() -> None:
    """Video modality is degraded to text placeholder."""
    cap = ModalityFilterCapability(capabilities=_text_only_caps(), video_strategy="describe")
    video = _binary_video()
    result = await cap.wrap_tool_execute(
        ctx=None,  # type: ignore[arg-type]
        call=None,  # type: ignore[arg-type]
        tool_def=None,  # type: ignore[arg-type]
        args={},
        handler=_handler(video),
    )
    assert result == "[video/mp4]"


@pytest.mark.unit
async def test_wrap_tool_document_describe() -> None:
    """Document modality (PDF) is degraded to text placeholder."""
    cap = ModalityFilterCapability(capabilities=_text_only_caps(), document_strategy="describe")
    pdf = _binary_pdf()
    result = await cap.wrap_tool_execute(
        ctx=None,  # type: ignore[arg-type]
        call=None,  # type: ignore[arg-type]
        tool_def=None,  # type: ignore[arg-type]
        args={},
        handler=_handler(pdf),
    )
    assert result == "[application/pdf]"


@pytest.mark.unit
async def test_wrap_tool_mixed_content_list() -> None:
    """Mixed list with supported and unsupported content: only unsupported is degraded."""
    cap = ModalityFilterCapability(
        capabilities=_image_only_caps(),
        image_strategy="describe",
        audio_strategy="describe",
    )
    img = _binary_image()
    audio = _binary_audio()
    text = "some text"
    result = await cap.wrap_tool_execute(
        ctx=None,  # type: ignore[arg-type]
        call=None,  # type: ignore[arg-type]
        tool_def=None,  # type: ignore[arg-type]
        args={},
        handler=_handler([img, text, audio]),
    )
    assert isinstance(result, list)
    assert result[0] is img  # image is supported, passes through
    assert result[1] == text
    assert result[2] == "[audio/wav]"  # audio is unsupported, described


@pytest.mark.unit
async def test_wrap_tool_supported_modality_passthrough() -> None:
    """Supported modality passes through unchanged."""
    cap = ModalityFilterCapability(capabilities=_all_caps())
    img = _binary_image()
    result = await cap.wrap_tool_execute(
        ctx=None,  # type: ignore[arg-type]
        call=None,  # type: ignore[arg-type]
        tool_def=None,  # type: ignore[arg-type]
        args={},
        handler=_handler(img),
    )
    assert result is img


@pytest.mark.unit
async def test_wrap_tool_drop_all_content_fallback() -> None:
    """When drop removes ALL content from a list, fallback text is returned."""
    cap = ModalityFilterCapability(
        capabilities=_text_only_caps(),
        image_strategy="drop",
        audio_strategy="drop",
    )
    img = _binary_image()
    audio = _binary_audio()
    result = await cap.wrap_tool_execute(
        ctx=None,  # type: ignore[arg-type]
        call=None,  # type: ignore[arg-type]
        tool_def=None,  # type: ignore[arg-type]
        args={},
        handler=_handler([img, audio]),
    )
    assert result == "[Tool returned only unsupported multimodal content]"


@pytest.mark.unit
async def test_wrap_tool_binary_content_audio_wav() -> None:
    """BinaryContent with audio/wav is classified as audio."""
    cap = ModalityFilterCapability(capabilities=_text_only_caps(), audio_strategy="describe")
    audio = BinaryContent(data=b"RIFF", media_type="audio/wav")
    result = await cap.wrap_tool_execute(
        ctx=None,  # type: ignore[arg-type]
        call=None,  # type: ignore[arg-type]
        tool_def=None,  # type: ignore[arg-type]
        args={},
        handler=_handler(audio),
    )
    assert result == "[audio/wav]"


@pytest.mark.unit
async def test_wrap_tool_binary_content_video_mp4() -> None:
    """BinaryContent with video/mp4 is classified as video."""
    cap = ModalityFilterCapability(capabilities=_text_only_caps(), video_strategy="describe")
    video = BinaryContent(data=b"\x00\x00", media_type="video/mp4")
    result = await cap.wrap_tool_execute(
        ctx=None,  # type: ignore[arg-type]
        call=None,  # type: ignore[arg-type]
        tool_def=None,  # type: ignore[arg-type]
        args={},
        handler=_handler(video),
    )
    assert result == "[video/mp4]"


@pytest.mark.unit
async def test_wrap_tool_binary_content_pdf() -> None:
    """BinaryContent with application/pdf is classified as document."""
    cap = ModalityFilterCapability(capabilities=_text_only_caps(), document_strategy="describe")
    pdf = BinaryContent(data=b"%PDF", media_type="application/pdf")
    result = await cap.wrap_tool_execute(
        ctx=None,  # type: ignore[arg-type]
        call=None,  # type: ignore[arg-type]
        tool_def=None,  # type: ignore[arg-type]
        args={},
        handler=_handler(pdf),
    )
    assert result == "[application/pdf]"


@pytest.mark.unit
async def test_wrap_tool_binary_image_always_image() -> None:
    """BinaryImage is always classified as image regardless of media_type check."""
    # Even with an unusual media_type, BinaryImage is image.
    img = BinaryImage(data=b"\x89PNG", media_type="image/png")
    cap = ModalityFilterCapability(
        capabilities=_text_only_caps(),
        image_strategy="describe",
        audio_strategy="describe",
    )
    result = await cap.wrap_tool_execute(
        ctx=None,  # type: ignore[arg-type]
        call=None,  # type: ignore[arg-type]
        tool_def=None,  # type: ignore[arg-type]
        args={},
        handler=_handler(img),
    )
    # Should use image strategy, not audio strategy.
    assert result == "[image/png]"


@pytest.mark.unit
async def test_wrap_tool_string_passthrough() -> None:
    """String results pass through unchanged."""
    cap = ModalityFilterCapability(capabilities=_text_only_caps())
    result = await cap.wrap_tool_execute(
        ctx=None,  # type: ignore[arg-type]
        call=None,  # type: ignore[arg-type]
        tool_def=None,  # type: ignore[arg-type]
        args={},
        handler=_handler("hello world"),
    )
    assert result == "hello world"


@pytest.mark.unit
async def test_wrap_tool_image_url_describe() -> None:
    """ImageUrl is degraded via describe strategy."""
    cap = ModalityFilterCapability(capabilities=_text_only_caps(), image_strategy="describe")
    url_img = ImageUrl(url="https://example.com/img.png")
    result = await cap.wrap_tool_execute(
        ctx=None,  # type: ignore[arg-type]
        call=None,  # type: ignore[arg-type]
        tool_def=None,  # type: ignore[arg-type]
        args={},
        handler=_handler(url_img),
    )
    assert result == "[image: https://example.com/img.png]"


@pytest.mark.unit
async def test_wrap_tool_audio_url_drop() -> None:
    """AudioUrl with drop strategy returns fallback text."""
    cap = ModalityFilterCapability(capabilities=_text_only_caps(), audio_strategy="drop")
    url_audio = AudioUrl(url="https://example.com/audio.wav")
    result = await cap.wrap_tool_execute(
        ctx=None,  # type: ignore[arg-type]
        call=None,  # type: ignore[arg-type]
        tool_def=None,  # type: ignore[arg-type]
        args={},
        handler=_handler(url_audio),
    )
    assert result == "[Tool returned only unsupported multimodal content]"


@pytest.mark.unit
async def test_wrap_tool_video_url_pass() -> None:
    """VideoUrl with pass strategy is unchanged."""
    cap = ModalityFilterCapability(capabilities=_text_only_caps(), video_strategy="pass")
    url_video = VideoUrl(url="https://example.com/video.mp4")
    result = await cap.wrap_tool_execute(
        ctx=None,  # type: ignore[arg-type]
        call=None,  # type: ignore[arg-type]
        tool_def=None,  # type: ignore[arg-type]
        args={},
        handler=_handler(url_video),
    )
    assert result is url_video


# ---------------------------------------------------------------------------
# 4.7 — before_model_request tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_before_model_request_tool_return_image_degraded() -> None:
    """Historical ToolReturnPart with image replayed to text-only model is degraded."""
    cap = ModalityFilterCapability(capabilities=_text_only_caps(), image_strategy="describe")
    img = _binary_image()
    tool_return = ToolReturnPart(
        tool_name="screenshot",
        content=img,
        tool_call_id="tc_1",
    )
    msg = ModelResponse(parts=[tool_return])
    ctx = _make_ctx([msg])

    result = await cap.before_model_request(ctx=None, request_context=ctx)  # type: ignore[arg-type]

    new_msg = result.messages[0]
    assert isinstance(new_msg, ModelResponse)
    new_part = new_msg.parts[0]
    assert isinstance(new_part, ToolReturnPart)
    assert new_part.content == "[image/png]"


@pytest.mark.unit
async def test_before_model_request_user_prompt_image_degraded() -> None:
    """UserPromptPart with image for text-only model is degraded."""
    cap = ModalityFilterCapability(capabilities=_text_only_caps(), image_strategy="describe")
    img = _binary_image()
    user_part = UserPromptPart(content=["Describe this", img])
    msg = ModelRequest(parts=[user_part])
    ctx = _make_ctx([msg])

    result = await cap.before_model_request(ctx=None, request_context=ctx)  # type: ignore[arg-type]

    new_msg = result.messages[0]
    assert isinstance(new_msg, ModelRequest)
    new_part = new_msg.parts[0]
    assert isinstance(new_part, UserPromptPart)
    assert isinstance(new_part.content, list)
    assert new_part.content[0] == "Describe this"
    assert new_part.content[1] == "[image/png]"


@pytest.mark.unit
async def test_before_model_request_no_multimodal_fast_path() -> None:
    """No multimodal content → fast path, returns same context unchanged."""
    cap = ModalityFilterCapability(capabilities=_text_only_caps())
    msg = ModelRequest(parts=[UserPromptPart(content="hello")])
    ctx = _make_ctx([msg])

    result = await cap.before_model_request(ctx=None, request_context=ctx)  # type: ignore[arg-type]

    assert result is ctx


@pytest.mark.unit
async def test_before_model_request_original_not_mutated() -> None:
    """Original ModelRequestContext.messages is not mutated."""
    cap = ModalityFilterCapability(capabilities=_text_only_caps(), image_strategy="describe")
    img = _binary_image()
    tool_return = ToolReturnPart(
        tool_name="screenshot",
        content=img,
        tool_call_id="tc_1",
    )
    msg = ModelResponse(parts=[tool_return])
    ctx = _make_ctx([msg])
    original_messages = list(ctx.messages)
    original_part_content = ctx.messages[0].parts[0].content  # type: ignore[union-attr]

    await cap.before_model_request(ctx=None, request_context=ctx)  # type: ignore[arg-type]

    # Original context unchanged.
    assert ctx.messages == original_messages
    assert ctx.messages[0].parts[0].content is original_part_content  # type: ignore[union-attr]


@pytest.mark.unit
async def test_before_model_request_mixed_message_types() -> None:
    """Both ModelRequest and ModelResponse with multimodal content are filtered."""
    cap = ModalityFilterCapability(
        capabilities=_text_only_caps(),
        image_strategy="describe",
        audio_strategy="describe",
    )
    img = _binary_image()
    audio = _binary_audio()

    user_part = UserPromptPart(content=["text", img])
    req = ModelRequest(parts=[user_part])

    tool_return = ToolReturnPart(
        tool_name="recorder",
        content=audio,
        tool_call_id="tc_1",
    )
    resp = ModelResponse(parts=[tool_return])

    ctx = _make_ctx([req, resp])

    result = await cap.before_model_request(ctx=None, request_context=ctx)  # type: ignore[arg-type]

    new_req = result.messages[0]
    assert isinstance(new_req, ModelRequest)
    new_user_part = new_req.parts[0]
    assert isinstance(new_user_part, UserPromptPart)
    assert isinstance(new_user_part.content, list)
    assert new_user_part.content[1] == "[image/png]"

    new_resp = result.messages[1]
    assert isinstance(new_resp, ModelResponse)
    new_tool_part = new_resp.parts[0]
    assert isinstance(new_tool_part, ToolReturnPart)
    assert new_tool_part.content == "[audio/wav]"


@pytest.mark.unit
async def test_before_model_request_supported_modality_passthrough() -> None:
    """Supported modality in message history passes through unchanged."""
    cap = ModalityFilterCapability(capabilities=_all_caps())
    img = _binary_image()
    tool_return = ToolReturnPart(
        tool_name="screenshot",
        content=img,
        tool_call_id="tc_1",
    )
    msg = ModelResponse(parts=[tool_return])
    ctx = _make_ctx([msg])

    result = await cap.before_model_request(ctx=None, request_context=ctx)  # type: ignore[arg-type]

    # Should return same context (content is supported, no change).
    assert result is ctx


# ---------------------------------------------------------------------------
# Capability metadata tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_has_wrap_node_run_false() -> None:
    """has_wrap_node_run returns False."""
    cap = ModalityFilterCapability(capabilities=_text_only_caps())
    assert cap.has_wrap_node_run is False


@pytest.mark.unit
def test_get_ordering_wrapped_by_budget_and_dcp() -> None:
    """get_ordering declares wrapped_by ToolOutputBudgetCapability and DCP."""
    cap = ModalityFilterCapability(capabilities=_text_only_caps())
    ordering = cap.get_ordering()
    assert ordering is not None
    assert isinstance(ordering, CapabilityOrdering)
    # wrapped_by means those capabilities are OUTER, self is INNER.
    from wolfharness.capabilities.dcp.capability import DynamicContextPruningCapability
    from wolfharness.capabilities.tool_output_budget import ToolOutputBudgetCapability

    assert ToolOutputBudgetCapability in ordering.wrapped_by
    assert DynamicContextPruningCapability in ordering.wrapped_by


@pytest.mark.unit
async def test_for_run_returns_new_instance() -> None:
    """for_run returns a fresh instance with same config."""
    cap = ModalityFilterCapability(
        capabilities=_text_only_caps(),
        image_strategy="drop",
        audio_strategy="pass",
    )
    new_cap = await cap.for_run(ctx=None)  # type: ignore[arg-type]
    assert new_cap is not cap
    assert new_cap.image_strategy == "drop"
    assert new_cap.audio_strategy == "pass"
    assert new_cap.capabilities == cap.capabilities
