"""Modality filter capability — degrade unsupported multimodal content.

Intercepts tool returns (via ``wrap_tool_execute``) and message history
(via ``before_model_request``) to degrade multimodal content that the
target model cannot handle.  For each unsupported modality, the
configured strategy is applied:

- ``describe``: replace the content with a text placeholder via
  ``describe_multimodal_content``.
- ``drop``: remove the content entirely.
- ``pass``: leave the content unchanged (no filtering).

When ``drop`` removes ALL content from a list, a fallback text string is
returned so the model still receives a meaningful tool result.

Ordering
========

``ModalityFilterCapability`` declares ``wrapped_by`` both
``ToolOutputBudgetCapability`` and ``DynamicContextPruningCapability``.
This makes the modality filter INNER to both — it post-processes tool
returns BEFORE budget truncation and BEFORE DCP pruning.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import TYPE_CHECKING, Any, Literal, assert_never

from pydantic_ai import BinaryContent, BinaryImage
from pydantic_ai.capabilities import AbstractCapability, CapabilityOrdering
from pydantic_ai.messages import (
    AudioUrl,
    DocumentUrl,
    ImageUrl,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    MultiModalContent,
    ToolReturnPart,
    UploadedFile,
    UserPromptPart,
    VideoUrl,
)

from wolfharness.capabilities.modality_utils import (
    BinaryCategory,
    classify_binary_content,
    describe_multimodal_content,
)


if TYPE_CHECKING:
    from pydantic_ai import RunContext
    from pydantic_ai.capabilities import WrapToolExecuteHandler
    from pydantic_ai.messages import ToolCallPart
    from pydantic_ai.models import ModelRequestContext
    from pydantic_ai.tools import ToolDefinition

    from wolfharness_config.model_capabilities import ModelCapabilities


logger = logging.getLogger(__name__)


type ModalityStrategy = Literal["describe", "drop", "pass"]


_FALLBACK_DROP_TEXT = "[Tool returned only unsupported multimodal content]"

_MULTIMODAL_TYPES: tuple[type, ...] = (
    BinaryContent,
    BinaryImage,
    ImageUrl,
    AudioUrl,
    VideoUrl,
    DocumentUrl,
    UploadedFile,
)


@dataclasses.dataclass
class ModalityFilterCapability(AbstractCapability[Any]):
    """Degrade unsupported multimodal content for text-only models.

    Inspects tool returns and message history, replacing or removing
    multimodal content that the model's ``ModelCapabilities`` does not
    support.

    Attributes:
        capabilities: Resolved model capabilities (all fields are
            ``bool``, not ``None``).  May be ``None`` when the
            capability is created from YAML config and the agent
            factory has not yet populated it.
        image_strategy: Strategy for unsupported image content.
        audio_strategy: Strategy for unsupported audio content.
        video_strategy: Strategy for unsupported video content.
        document_strategy: Strategy for unsupported document content.
    """

    capabilities: ModelCapabilities | None = None
    image_strategy: ModalityStrategy = "describe"
    audio_strategy: ModalityStrategy = "describe"
    video_strategy: ModalityStrategy = "describe"
    document_strategy: ModalityStrategy = "describe"

    @property
    def has_wrap_node_run(self) -> bool:
        return False

    # ---- Ordering ----

    def get_ordering(self) -> CapabilityOrdering | None:
        """Declare position in the capability chain.

        ``wrapped_by`` makes this capability INNER to both
        ``ToolOutputBudgetCapability`` and
        ``DynamicContextPruningCapability``.  Tool returns are
        modality-filtered BEFORE budget truncation and BEFORE DCP
        pruning.
        """
        from wolfharness.capabilities.dcp.capability import (
            DynamicContextPruningCapability,
        )
        from wolfharness.capabilities.tool_output_budget import (
            ToolOutputBudgetCapability,
        )

        return CapabilityOrdering(
            wrapped_by=[ToolOutputBudgetCapability, DynamicContextPruningCapability],
        )

    # ---- Per-run isolation ----

    async def for_run(self, ctx: RunContext[Any]) -> ModalityFilterCapability:
        return ModalityFilterCapability(
            capabilities=self.capabilities,
            image_strategy=self.image_strategy,
            audio_strategy=self.audio_strategy,
            video_strategy=self.video_strategy,
            document_strategy=self.document_strategy,
        )

    # ---- Tool execution wrapping ----

    async def wrap_tool_execute(
        self,
        ctx: RunContext[Any],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: Any,
        handler: WrapToolExecuteHandler,
    ) -> Any:
        result = await handler(args)
        return self._filter_tool_result(result)

    # ---- Pre-request message filtering ----

    async def before_model_request(
        self,
        ctx: RunContext[Any],
        request_context: ModelRequestContext,
    ) -> ModelRequestContext:
        """Filter unsupported multimodal content from message history.

        Scans ``ModelRequest`` and ``ModelResponse`` messages for
        multimodal content in ``UserPromptPart`` and ``ToolReturnPart``,
        degrades unsupported content, and returns a new
        ``ModelRequestContext`` via ``dataclasses.replace()``.

        Fast-path: if no multimodal content types are detected, the
        original ``request_context`` is returned unchanged.
        """
        messages = request_context.messages

        # Fast-path: quick scan for multimodal content.
        if not _has_multimodal_content(messages):
            return request_context

        new_messages: list[ModelMessage] = []
        changed = False
        for msg in messages:
            match msg:
                case ModelRequest():
                    filtered_req = self._filter_model_request(msg)
                    if filtered_req is not msg:
                        changed = True
                    new_messages.append(filtered_req)
                case ModelResponse():
                    filtered_resp = self._filter_model_response(msg)
                    if filtered_resp is not msg:
                        changed = True
                    new_messages.append(filtered_resp)
                case _:
                    new_messages.append(msg)

        if not changed:
            return request_context

        return dataclasses.replace(request_context, messages=new_messages)

    # ---- Internal: tool result filtering ----

    def _filter_tool_result(self, result: Any) -> Any:
        """Degrade multimodal content in a tool result.

        Handles ``str``, ``list``, and direct ``MultiModalContent``.
        """
        match result:
            case str():
                return result
            case list():
                return self._filter_content_list(result)
            case _:
                if isinstance(result, _MULTIMODAL_TYPES):
                    filtered = self._filter_single_content(result)  # type: ignore[arg-type]
                    match filtered:
                        case str():
                            return filtered
                        case None:
                            return _FALLBACK_DROP_TEXT
                        case _:
                            return filtered
                return result

    def _filter_content_list(self, items: list[Any]) -> Any:
        """Filter a list of content items.

        Returns the original list if nothing changed, a new list if
        items were modified, or a fallback string if all items were
        dropped.
        """
        new_items: list[Any] = []
        changed = False
        for item in items:
            if isinstance(item, _MULTIMODAL_TYPES):
                filtered = self._filter_single_content(item)  # type: ignore[arg-type]
                match filtered:
                    case None:
                        changed = True
                    case str():
                        new_items.append(filtered)
                        changed = True
                    case _:
                        new_items.append(filtered)
            else:
                new_items.append(item)

        if not changed:
            return items

        # If everything was dropped, return fallback text.
        if not new_items:
            return _FALLBACK_DROP_TEXT

        return new_items

    def _filter_single_content(
        self,
        content: (
            BinaryContent
            | BinaryImage
            | ImageUrl
            | AudioUrl
            | VideoUrl
            | DocumentUrl
            | UploadedFile
        ),
    ) -> str | MultiModalContent | None:
        """Filter a single ``MultiModalContent`` item.

        Returns:
            - ``str`` — replaced with a description placeholder.
            - ``None`` — content was dropped.
            - ``MultiModalContent`` — content was passed through
              unchanged.
        """
        category = self._classify_content(content)

        if not self._is_modality_supported(category):
            strategy = self._get_strategy(category)
            match strategy:
                case "describe":
                    return describe_multimodal_content(content)
                case "drop":
                    return None
                case "pass":
                    return content
                case _ as unreachable:
                    assert_never(unreachable)

        return content

    # ---- Internal: message filtering ----

    def _filter_model_request(self, msg: ModelRequest) -> ModelRequest:
        """Filter multimodal content in a ``ModelRequest``.

        Returns the original message if nothing changed, or a new
        ``ModelRequest`` via ``dataclasses.replace()``.
        """
        new_parts: list[Any] = []
        changed = False
        for part in msg.parts:
            match part:
                case UserPromptPart():
                    new_part = self._filter_user_prompt_part(part)
                    if new_part is not part:
                        changed = True
                    new_parts.append(new_part)
                case _:
                    new_parts.append(part)

        if not changed:
            return msg
        return dataclasses.replace(msg, parts=new_parts)

    def _filter_model_response(self, msg: ModelResponse) -> ModelResponse:
        """Filter multimodal content in a ``ModelResponse``.

        Returns the original message if nothing changed, or a new
        ``ModelResponse`` via ``dataclasses.replace()``.
        """
        new_parts: list[Any] = []
        changed = False
        for part in msg.parts:
            match part:
                case ToolReturnPart():
                    new_part = self._filter_tool_return_part(part)
                    if new_part is not part:
                        changed = True
                    new_parts.append(new_part)
                case _:
                    new_parts.append(part)

        if not changed:
            return msg
        return dataclasses.replace(msg, parts=new_parts)

    def _filter_user_prompt_part(self, part: UserPromptPart) -> UserPromptPart:
        """Filter multimodal content in a ``UserPromptPart``.

        ``UserPromptPart.content`` can be ``str`` or
        ``Sequence[UserContent]``.
        """
        content = part.content
        match content:
            case str():
                return part
            case list():
                new_items = self._filter_content_list(content)
                match new_items:
                    case list():
                        return dataclasses.replace(part, content=new_items)
                    case str():
                        # All content was dropped — return fallback text.
                        return dataclasses.replace(part, content=new_items)
                    case _:
                        return part
            case _:
                return part

    def _filter_tool_return_part(self, part: ToolReturnPart) -> ToolReturnPart:
        """Filter multimodal content in a ``ToolReturnPart``.

        ``ToolReturnPart.content`` can be ``str``, ``MultiModalContent``,
        or ``list[str | MultiModalContent]``.
        """
        content = part.content
        new_content: Any = content
        match content:
            case str():
                pass
            case list():
                new_items = self._filter_content_list(content)
                match new_items:
                    case list() | str():
                        new_content = new_items
                    case _:
                        pass
            case _:
                if isinstance(content, _MULTIMODAL_TYPES):
                    filtered = self._filter_single_content(content)  # type: ignore[arg-type]
                    match filtered:
                        case str():
                            new_content = filtered
                        case None:
                            new_content = _FALLBACK_DROP_TEXT
                        case _:
                            pass

        if new_content is content:
            return part
        return dataclasses.replace(part, content=new_content)

    # ---- Internal: classification & strategy ----

    def _classify_content(
        self,
        content: (
            BinaryContent
            | BinaryImage
            | ImageUrl
            | AudioUrl
            | VideoUrl
            | DocumentUrl
            | UploadedFile
        ),
    ) -> BinaryCategory:
        """Classify a ``MultiModalContent`` item into a modality category.

        For ``BinaryImage``, always returns ``"image"`` regardless of
        ``media_type`` (checked via ``isinstance`` before
        ``classify_binary_content``).

        For ``BinaryContent``, uses ``classify_binary_content()``.
        Returns ``"unknown"`` for unclassified binary content.

        For URL types, returns the corresponding category directly.
        ``UploadedFile`` returns ``"unknown"`` since the modality is
        not known from the file ID alone.
        """
        match content:
            case BinaryImage() | ImageUrl():
                return "image"
            case BinaryContent():
                return classify_binary_content(content)
            case AudioUrl():
                return "audio"
            case VideoUrl():
                return "video"
            case DocumentUrl():
                return "document"
            case UploadedFile():
                return "unknown"
            case _ as unreachable:
                assert_never(unreachable)

    def _is_modality_supported(self, category: BinaryCategory) -> bool:
        """Check whether the model supports a given modality category.

        ``"unknown"`` content is always treated as supported — we
        cannot degrade what we cannot classify.

        When ``capabilities`` is ``None`` (unresolved), all modalities
        are treated as supported (no filtering).
        """
        if self.capabilities is None:
            return True
        match category:
            case "image":
                return self.capabilities.image_input is True
            case "audio":
                return self.capabilities.audio_input is True
            case "video":
                return self.capabilities.video_input is True
            case "document":
                return self.capabilities.document_input is True
            case "unknown":
                return True
            case _ as unreachable:
                assert_never(unreachable)

    def _get_strategy(self, category: BinaryCategory) -> ModalityStrategy:
        """Get the degradation strategy for a modality category.

        ``"unknown"`` always returns ``"pass"`` since we cannot
        determine the modality to apply the correct strategy.
        """
        match category:
            case "image":
                return self.image_strategy
            case "audio":
                return self.audio_strategy
            case "video":
                return self.video_strategy
            case "document":
                return self.document_strategy
            case "unknown":
                return "pass"
            case _ as unreachable:
                assert_never(unreachable)


# ---- Fast-path scan ----


def _has_multimodal_content(messages: list[ModelMessage]) -> bool:
    """Quick isinstance scan to detect multimodal content in messages.

    Returns ``True`` if any ``UserPromptPart`` or ``ToolReturnPart``
    in the message list contains ``MultiModalContent`` instances.
    """
    multimodal_types = (
        BinaryContent,
        BinaryImage,
        ImageUrl,
        AudioUrl,
        VideoUrl,
        DocumentUrl,
        UploadedFile,
    )
    for msg in messages:
        match msg:
            case ModelRequest():
                for part in msg.parts:
                    match part:
                        case UserPromptPart():
                            content = part.content
                            if isinstance(content, list):
                                for item in content:
                                    if isinstance(item, multimodal_types):
                                        return True
                        case _:
                            pass
            case ModelResponse():
                for resp_part in msg.parts:
                    match resp_part:
                        case ToolReturnPart():
                            content = resp_part.content
                            if isinstance(content, multimodal_types):
                                return True
                            if isinstance(content, list):
                                for item in content:
                                    if isinstance(item, multimodal_types):
                                        return True
                        case _:
                            pass
            case _:
                pass
    return False
