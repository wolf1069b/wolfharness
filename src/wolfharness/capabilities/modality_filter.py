"""Modality filter capability — degrade unsupported multimodal content.

Intercepts tool returns (via ``wrap_tool_execute``) and message history
(via ``before_model_request``) to degrade multimodal content that the
target model cannot handle.  For each unsupported modality, the
configured strategy is applied:

- ``describe``: replace the content with a text placeholder via
  ``describe_multimodal_content``.
- ``understand``: replace the content with a real text description
  produced by a vision LLM (image strategy only).  Requires
  ``vision_model``; without one it falls back to ``describe``.
- ``reference``: persist the binary content to a per-run scratch
  directory and replace it with a ``[file: <path>]`` reference that a
  vision-capable subagent or file tool can open (RFC-0061).  URL and
  ``UploadedFile`` content has no local bytes, so it falls back to
  ``describe``.
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
import hashlib
import logging
import mimetypes
from pathlib import Path
import shutil
import tempfile
from typing import TYPE_CHECKING, Any, Literal, assert_never

import anyio
import logfire
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

from wolfharness.agents.context import AgentRunContext
from wolfharness.capabilities.modality_utils import (
    BinaryCategory,
    classify_binary_content,
    describe_multimodal_content,
)


if TYPE_CHECKING:
    from pydantic_ai import RunContext
    from pydantic_ai.capabilities import (
        AgentNode,
        NodeResult,
        WrapToolExecuteHandler,
    )
    from pydantic_ai.messages import ToolCallPart
    from pydantic_ai.models import ModelRequestContext
    from pydantic_ai.tools import ToolDefinition

    from wolfharness_config.model_capabilities import ModelCapabilities


logger = logging.getLogger(__name__)


type ModalityStrategy = Literal["describe", "reference", "drop", "pass", "understand"]


_FALLBACK_DROP_TEXT = "[Tool returned only unsupported multimodal content]"

# Maximum image payload the vision LLM will be asked to analyze; larger
# payloads fall back to the ``reference`` strategy instead.
_VISION_MAX_BYTES = 10_000_000

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
        vision_model: Model variant name or namespaced string used by
            the ``"understand"`` image strategy.  ``None`` falls back to
            ``"describe"``.
    """

    capabilities: ModelCapabilities | None = None
    image_strategy: ModalityStrategy = "describe"
    audio_strategy: ModalityStrategy = "describe"
    video_strategy: ModalityStrategy = "describe"
    document_strategy: ModalityStrategy = "describe"
    vision_model: str | None = None

    _scratch_dirs: set[Path] = dataclasses.field(default_factory=set, init=False, repr=False)
    _vision_cache: dict[bytes, str] = dataclasses.field(
        default_factory=dict, init=False, repr=False
    )

    @property
    def has_wrap_node_run(self) -> bool:
        return False

    # ---- Scratch root (RFC-0061 `reference` strategy) ----

    def _scratch_dir(self, ctx: RunContext[Any]) -> Path:
        """Return the per-run scratch directory for ``reference`` writes.

        Rooted at ``tempfile.gettempdir()/wolfharness-modality/{session_id}``
        so different runs never collide and the OS can reclaim stale
        directories.  Falls back to a shared run-id-scoped directory when
        no session id is available (standalone ``agent.run()``).
        """
        base = Path(tempfile.gettempdir()) / "wolfharness-modality"
        session_id = self._session_id(ctx)
        scratch = base / session_id
        scratch.mkdir(parents=True, exist_ok=True)
        self._scratch_dirs.add(scratch)
        return scratch

    def _session_id(self, ctx: RunContext[Any]) -> str:
        """Get the current run's session id, or a safe fallback."""
        run_ctx = ctx.deps.run_ctx if hasattr(ctx.deps, "run_ctx") else None
        if isinstance(run_ctx, AgentRunContext):
            return run_ctx.session_id or "default"
        return str(getattr(run_ctx, "run_id", None) or "default")

    def _reference_content(self, content: MultiModalContent, ctx: RunContext[Any]) -> str:
        """Persist binary content and return a ``[file: <path>]`` reference.

        URL types and ``UploadedFile`` have no local bytes to persist, so
        they fall back to ``describe_multimodal_content``.
        """
        if not isinstance(content, (BinaryContent, BinaryImage)):
            return describe_multimodal_content(content)

        scratch = self._scratch_dir(ctx)
        ext = mimetypes.guess_extension(content.media_type) or ".bin"
        # Hash-based name so identical payloads dedupe and filenames stay
        # filesystem-safe regardless of any caller-supplied identifier.
        digest = hashlib.sha256(content.data, usedforsecurity=False).hexdigest()[:16]
        path = scratch / f"content-{digest}{ext}"
        path.write_bytes(content.data)
        return f"[file: {path}]"

    async def after_node_run(
        self,
        ctx: RunContext[Any],
        *,
        node: AgentNode[Any],
        result: NodeResult[Any],
    ) -> NodeResult[Any]:
        """Remove scratch directories written by this run.

        ``reference`` strategy persists degraded content to disk; the
        session can be resumed later via restart, so cleanup happens only
        when the run itself ends.  Per-instance tracking avoids deleting
        directories another concurrent run may still need.
        """
        for scratch in self._scratch_dirs:
            try:
                shutil.rmtree(scratch, ignore_errors=True)
            except OSError:
                logger.warning("Failed to remove modality scratch dir: %s", scratch)
        self._scratch_dirs.clear()
        return result

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
            vision_model=self.vision_model,
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
        return await self._filter_tool_result(ctx, result)

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
                    filtered_req = await self._filter_model_request(ctx, msg)
                    if filtered_req is not msg:
                        changed = True
                    new_messages.append(filtered_req)
                case ModelResponse():
                    filtered_resp = await self._filter_model_response(ctx, msg)
                    if filtered_resp is not msg:
                        changed = True
                    new_messages.append(filtered_resp)
                case _:
                    new_messages.append(msg)

        if not changed:
            return request_context

        return dataclasses.replace(request_context, messages=new_messages)

    # ---- Internal: tool result filtering ----

    async def _filter_tool_result(
        self, ctx: RunContext[Any], result: Any, *, allow_vision: bool = True
    ) -> Any:
        """Degrade multimodal content in a tool result.

        Handles ``str``, ``list``, and direct ``MultiModalContent``.
        """
        match result:
            case str():
                return result
            case list():
                return await self._filter_content_list(ctx, result, allow_vision=allow_vision)
            case _:
                if isinstance(result, _MULTIMODAL_TYPES):
                    filtered = await self._filter_single_content(
                        ctx,
                        result,  # type: ignore[arg-type]
                        allow_vision=allow_vision,
                    )
                    match filtered:
                        case str():
                            return filtered
                        case None:
                            return _FALLBACK_DROP_TEXT
                        case _:
                            return filtered
                return result

    async def _filter_content_list(
        self, ctx: RunContext[Any], items: list[Any], *, allow_vision: bool = True
    ) -> Any:
        """Filter a list of content items.

        Returns the original list if nothing changed, a new list if
        items were modified, or a fallback string if all items were
        dropped.
        """
        new_items: list[Any] = []
        changed = False
        for item in items:
            if isinstance(item, _MULTIMODAL_TYPES):
                filtered = await self._filter_single_content(
                    ctx,
                    item,  # type: ignore[arg-type]
                    allow_vision=allow_vision,
                )
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

    async def _filter_single_content(  # noqa: PLR0911
        self,
        ctx: RunContext[Any],
        content: (
            BinaryContent
            | BinaryImage
            | ImageUrl
            | AudioUrl
            | VideoUrl
            | DocumentUrl
            | UploadedFile
        ),
        *,
        allow_vision: bool = True,
    ) -> str | MultiModalContent | None:
        """Filter a single ``MultiModalContent`` item.

        Args:
            ctx: The pydantic-ai run context.
            content: The multimodal content item to filter.
            allow_vision: Whether the ``"understand"`` strategy is
                permitted to call the vision LLM.  ``False`` falls back
                to ``describe`` for determinism (used on the
                ``before_model_request`` history rebuild path).

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
                case "reference":
                    return self._reference_content(content, ctx)
                case "drop":
                    return None
                case "pass":
                    return content
                case "understand":
                    if not allow_vision or self.vision_model is None:
                        return describe_multimodal_content(content)
                    return await self._understand_image(ctx, content)
                case _ as unreachable:
                    assert_never(unreachable)

        return content

    # ---- Internal: vision understanding ----

    @logfire.instrument("Understanding image via vision LLM")
    async def _understand_image(  # noqa: PLR0911
        self,
        ctx: RunContext[Any],
        content: (
            BinaryContent
            | BinaryImage
            | ImageUrl
            | AudioUrl
            | VideoUrl
            | DocumentUrl
            | UploadedFile
        ),
    ) -> str:
        """Describe an image via a vision LLM.

        Replaces the image binary with a real text description produced
        by the configured ``vision_model``.  Never raises — all failure
        paths fall back to ``describe`` or ``reference`` so the agent
        turn cannot break because of a vision model failure.
        """
        # Only binary image bytes can be understood — URL types and
        # non-image binary content fall back to describe.
        if not isinstance(content, (BinaryImage, BinaryContent)):
            return describe_multimodal_content(content)
        if not isinstance(content, BinaryImage) and not content.media_type.startswith("image/"):
            return describe_multimodal_content(content)

        # Normalize image bytes before vision call — resize/re-encode to
        # keep the request fast and within model resolution limits. Reuses
        # the same ImageNormalizer as the user-attachment path (RFC-0059)
        # so all images go through one unified processing pipeline.
        from wolfharness.images.normalizer import ImageNormalizer

        normalizer = ImageNormalizer()
        normalized_data, normalized_mime = normalizer.normalize_bytes(
            content.data, content.media_type
        )
        if normalized_data is not content.data:
            content = BinaryImage(data=normalized_data, media_type=normalized_mime)

        # Byte-size guard: reject payloads still too large after normalization.
        if len(content.data) > _VISION_MAX_BYTES:
            return self._reference_content(content, ctx)

        # Per-instance dedup cache keyed by (normalized) content hash.
        key = hashlib.sha256(content.data, usedforsecurity=False).digest()
        if key in self._vision_cache:
            return f"[Image analysis: {self._vision_cache[key]}]"

        model = self._resolve_vision_model(ctx)
        if model is None:
            return describe_multimodal_content(content)

        from pydantic_ai import Agent

        vision_agent = Agent(
            model=model,
            system_prompt=(
                "You are a vision assistant. Describe the image concisely, "
                "focusing on visible content, text, and notable details. "
                "Keep the description under 500 characters."
            ),
        )

        try:
            with anyio.fail_after(30):
                result = await vision_agent.run(["Describe this image.", content])
            description = result.output
        except TimeoutError:
            # Fall back to reference (save to disk).
            return self._reference_content(content, ctx)
        except Exception:
            logger.warning("Vision LLM call failed, falling back to describe", exc_info=True)
            return describe_multimodal_content(content)

        self._vision_cache[key] = description
        return f"[Image analysis: {description}]"

    def _resolve_vision_model(self, ctx: RunContext[Any]) -> Any | None:
        """Resolve the vision model instance for the ``understand`` strategy.

        Tries the agent manifest first (model variant names), then falls
        back to ``infer_model`` for namespaced strings (standalone runs).
        Returns ``None`` when resolution fails so the caller can fall
        back to ``describe``.
        """
        if not isinstance(self.vision_model, str):
            return None
        try:
            from wolfharness.capabilities.agent_context import (
                resolve_agent_context_from_deps,
            )

            agent_ctx = resolve_agent_context_from_deps(ctx.deps)
            return agent_ctx.host.manifest.resolve_model(self.vision_model).get_model()
        except Exception:  # noqa: BLE001
            pass
        try:
            from wolfharness.utils.model_helpers import infer_model

            return infer_model(self.vision_model)
        except Exception:  # noqa: BLE001
            return None

    # ---- Internal: message filtering ----

    async def _filter_model_request(self, ctx: RunContext[Any], msg: ModelRequest) -> ModelRequest:
        """Filter multimodal content in a ``ModelRequest``.

        Returns the original message if nothing changed, or a new
        ``ModelRequest`` via ``dataclasses.replace()``.
        """
        new_parts: list[Any] = []
        changed = False
        for part in msg.parts:
            match part:
                case UserPromptPart():
                    new_part = await self._filter_user_prompt_part(ctx, part)
                    if new_part is not part:
                        changed = True
                    new_parts.append(new_part)
                case _:
                    new_parts.append(part)

        if not changed:
            return msg
        return dataclasses.replace(msg, parts=new_parts)

    async def _filter_model_response(
        self, ctx: RunContext[Any], msg: ModelResponse
    ) -> ModelResponse:
        """Filter multimodal content in a ``ModelResponse``.

        Returns the original message if nothing changed, or a new
        ``ModelResponse`` via ``dataclasses.replace()``.
        """
        new_parts: list[Any] = []
        changed = False
        for part in msg.parts:
            match part:
                case ToolReturnPart():
                    new_part = await self._filter_tool_return_part(ctx, part)
                    if new_part is not part:
                        changed = True
                    new_parts.append(new_part)
                case _:
                    new_parts.append(part)

        if not changed:
            return msg
        return dataclasses.replace(msg, parts=new_parts)

    async def _filter_user_prompt_part(
        self, ctx: RunContext[Any], part: UserPromptPart
    ) -> UserPromptPart:
        """Filter multimodal content in a ``UserPromptPart``.

        ``UserPromptPart.content`` can be ``str`` or
        ``Sequence[UserContent]``.
        """
        content = part.content
        match content:
            case str():
                return part
            case list():
                new_items = await self._filter_content_list(ctx, content, allow_vision=False)
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

    async def _filter_tool_return_part(
        self, ctx: RunContext[Any], part: ToolReturnPart
    ) -> ToolReturnPart:
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
                new_items = await self._filter_content_list(ctx, content, allow_vision=False)
                match new_items:
                    case list() | str():
                        new_content = new_items
                    case _:
                        pass
            case _:
                if isinstance(content, _MULTIMODAL_TYPES):
                    filtered = await self._filter_single_content(
                        ctx,
                        content,  # type: ignore[arg-type]
                        allow_vision=False,
                    )
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
                return "describe" if self.audio_strategy == "understand" else self.audio_strategy
            case "video":
                return "describe" if self.video_strategy == "understand" else self.video_strategy
            case "document":
                if self.document_strategy == "understand":
                    return "describe"
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
