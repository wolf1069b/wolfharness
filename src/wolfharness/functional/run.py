"""Functional wrappers for Agent usage."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Unpack, overload

from anyenv import run_sync
from pydantic_ai import ImageUrl

from wolfharness import Agent


if TYPE_CHECKING:
    from wolfharness.agents.native_agent import AgentKwargs
    from wolfharness.common_types import PromptCompatible
    from wolfharness.images.normalizer import ImageNormalizer


def _make_image_normalizer(
    attachment_image: Any | None,
) -> ImageNormalizer | None:
    """Build an ``ImageNormalizer`` from an explicit config (RFC-0059)."""
    if attachment_image is None:
        return None
    from wolfharness.images.normalizer import ImageNormalizer

    return ImageNormalizer(attachment_image)


def _normalize_image_url(url: str, normalizer: ImageNormalizer | None) -> str:
    """Normalize a data-URI image URL if a normalizer is available."""
    if normalizer is None:
        return url
    normalized, _mime = normalizer.normalize(url, "image/*")
    return normalized


@overload
async def run_agent[TResult](
    prompt: PromptCompatible,
    image_url: str | None = None,
    *,
    output_type: type[TResult],
    attachment_image: Any | None = None,
    **kwargs: Unpack[AgentKwargs],
) -> TResult: ...


@overload
async def run_agent(
    prompt: PromptCompatible,
    image_url: str | None = None,
    *,
    output_type: None = None,
    attachment_image: Any | None = None,
    **kwargs: Unpack[AgentKwargs],
) -> str: ...


async def run_agent(
    prompt: PromptCompatible,
    image_url: str | None = None,
    *,
    output_type: type[Any] | None = None,
    attachment_image: Any | None = None,
    **kwargs: Unpack[AgentKwargs],
) -> Any:
    """Run prompt through agent and return result.

    Args:
        prompt: The user prompt.
        image_url: Optional image URL or ``data:`` URI. ``data:`` URIs are
            normalized when they exceed configured limits (RFC-0059).
        output_type: Optional structured output type.
        attachment_image: Optional ``AttachmentImageConfig`` for image
            normalization. When omitted, defaults apply.
        **kwargs: Additional agent constructor kwargs.
    """
    async with Agent[Any, str](**kwargs) as agent:
        # Convert to structured output agent if output_type specified
        final = agent.to_structured(output_type) if output_type is not None else agent

        if image_url:
            normalized = _normalize_image_url(image_url, _make_image_normalizer(attachment_image))
            image = ImageUrl(url=normalized)
            result = await final.run(prompt, image)
        else:
            result = await final.run(prompt)
        return result.content


@overload
def run_agent_sync[TResult](
    prompt: PromptCompatible,
    image_url: str | None = None,
    *,
    output_type: type[TResult],
    attachment_image: Any | None = None,
    **kwargs: Unpack[AgentKwargs],
) -> TResult: ...


@overload
def run_agent_sync(
    prompt: PromptCompatible,
    image_url: str | None = None,
    *,
    attachment_image: Any | None = None,
    **kwargs: Unpack[AgentKwargs],
) -> str: ...


def run_agent_sync(
    prompt: PromptCompatible,
    image_url: str | None = None,
    *,
    output_type: type[Any] | None = None,
    attachment_image: Any | None = None,
    **kwargs: Unpack[AgentKwargs],
) -> Any:
    """Sync wrapper for run_agent."""

    async def _run() -> Any:
        if output_type is None:
            return await run_agent(
                prompt,
                image_url,
                attachment_image=attachment_image,
                **kwargs,
            )
        return await run_agent(
            prompt,
            image_url,
            output_type=output_type,
            attachment_image=attachment_image,
            **kwargs,
        )

    return run_sync(_run())
