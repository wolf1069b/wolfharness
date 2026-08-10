"""Integration tests for model variant resolution in event processing.

Verifies that StreamCompleteEvent preserves variant model_id/provider_id
instead of overwriting with raw pydantic-ai API response names.

Regression test for: OpenCode TUI context length becomes 0% after agent run.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import Mock

import pytest

from wolfharness.agents.events import StreamCompleteEvent
from wolfharness.models.model_configs import AnyModelConfig, StringModelConfig
from wolfharness_server.opencode_server.event_adapter import OpenCodeEventAdapter
from wolfharness_server.opencode_server.event_processor_context import (
    EventProcessorContext,
)
from wolfharness_server.opencode_server.models import (
    MessagePath,
    MessageTime,
    MessageUpdatedEvent,
    MessageWithParts,
)


pytestmark = pytest.mark.integration


def _make_model_variants() -> dict[str, StringModelConfig]:
    """Create model_variants matching the bug report config."""
    return {
        "kimi-k2": StringModelConfig(
            identifier="openai-chat:svc/kimi-k2",
            provider="wolf-ai",
            context_length=262144,
        ),
    }


def _make_context(
    model_variants: dict[str, AnyModelConfig] | None = None,
    initial_model_id: str = "kimi-k2",
    initial_provider_id: str = "wolf-ai",
) -> EventProcessorContext:
    """Create an EventProcessorContext with real model_variants.

    The assistant message is pre-populated with variant names (as
    _create_assistant_message would do in the real flow).
    """
    session_id = "test-session"
    assistant_msg_id = "msg-001"
    assistant_msg = MessageWithParts.assistant(
        message_id=assistant_msg_id,
        session_id=session_id,
        time=MessageTime(created=0),
        agent_name="test-agent",
        model_id=initial_model_id,
        provider_id=initial_provider_id,
        path=MessagePath(cwd="/tmp", root="/tmp"),
        parent_id="msg-000",
    )

    state = Mock()
    state.messages = {}
    state.messages.setdefault(session_id, [])
    state.ensure_session = Mock()
    state.storage = Mock()
    state.storage.log_message = Mock()
    state.model_variants = model_variants if model_variants is not None else {}

    return EventProcessorContext(
        session_id=session_id,
        assistant_msg_id=assistant_msg_id,
        assistant_msg=assistant_msg,
        state=state,
        working_dir="/tmp",
    )


def _make_stream_complete_msg(
    model_name: str = "svc/kimi-k2",
    provider_name: str = "openai",
) -> Mock:
    """Create a mock ChatMessage simulating a pydantic-ai API response.

    In the real flow, model_name = result.response.model_name (e.g. "svc/kimi-k2")
    and provider_name = result.response.provider_name (e.g. "openai" — the
    canonicalized system name, since pydantic-ai canonicalizes "openai-chat" → "openai").
    """
    msg = Mock()
    msg.content = "Done"
    msg.usage = None
    msg.cost_info = None
    msg.model_name = model_name
    msg.provider_name = provider_name
    return msg


async def _collect_events(gen: Any) -> list[Any]:
    """Collect all events from an async generator."""
    return [event async for event in gen]


class TestStreamCompleteVariantResolution:
    """StreamCompleteEvent must preserve variant model_id/provider_id."""

    @pytest.mark.asyncio
    async def test_stream_complete_preserves_variant_model_id(self) -> None:
        """StreamComplete does NOT overwrite variant model_id with raw API name.

        The assistant message starts with variant names (set by _create_assistant_message).
        After StreamCompleteEvent, model_id should still be the variant name.
        """
        variants = _make_model_variants()
        ctx = _make_context(model_variants=variants)
        adapter = OpenCodeEventAdapter(context=ctx)

        msg = _make_stream_complete_msg(model_name="svc/kimi-k2", provider_name="openai")
        await _collect_events(adapter.convert_event(StreamCompleteEvent(message=msg)))

        assert ctx.assistant_msg.info.model_id == "kimi-k2"
        assert ctx.assistant_msg.info.provider_id == "wolf-ai"

    @pytest.mark.asyncio
    async def test_no_message_updated_event_when_variant_matches(self) -> None:
        """MessageUpdatedEvent is NOT emitted when resolved variant matches existing values."""
        variants = _make_model_variants()
        ctx = _make_context(model_variants=variants)
        adapter = OpenCodeEventAdapter(context=ctx)

        msg = _make_stream_complete_msg(model_name="svc/kimi-k2", provider_name="openai")
        events = await _collect_events(adapter.convert_event(StreamCompleteEvent(message=msg)))

        message_updated = [e for e in events if isinstance(e, MessageUpdatedEvent)]
        assert len(message_updated) == 0

    @pytest.mark.asyncio
    async def test_message_updated_event_when_model_actually_changed(self) -> None:
        """MessageUpdatedEvent IS emitted when the model actually changed.

        Simulates a fallback model scenario: assistant message starts with variant "kimi-k2"
        but the API response carries a different model that resolves to a different variant.
        """
        variants = {
            "kimi-k2": StringModelConfig(
                identifier="openai-chat:svc/kimi-k2",
                provider="wolf-ai",
            ),
            "ds-v4": StringModelConfig(
                identifier="openai-chat:ds-v4-flash",
                provider="wolf-ai",
            ),
        }
        # Start with kimi-k2 variant
        ctx = _make_context(
            model_variants=variants,
            initial_model_id="kimi-k2",
            initial_provider_id="wolf-ai",
        )
        adapter = OpenCodeEventAdapter(context=ctx)

        # API response carries ds-v4-flash model (fallback scenario)
        msg = _make_stream_complete_msg(model_name="ds-v4-flash", provider_name="openai")
        events = await _collect_events(adapter.convert_event(StreamCompleteEvent(message=msg)))

        assert ctx.assistant_msg.info.model_id == "ds-v4"
        assert ctx.assistant_msg.info.provider_id == "wolf-ai"
        message_updated = [e for e in events if isinstance(e, MessageUpdatedEvent)]
        assert len(message_updated) == 1

    @pytest.mark.asyncio
    async def test_stream_complete_falls_back_to_raw_when_no_variants(self) -> None:
        """StreamComplete with empty model_variants falls back to raw API names."""
        ctx = _make_context(
            model_variants={},
            initial_model_id="kimi-k2",
            initial_provider_id="wolf-ai",
        )
        adapter = OpenCodeEventAdapter(context=ctx)

        msg = _make_stream_complete_msg(model_name="svc/kimi-k2", provider_name="openai")
        events = await _collect_events(adapter.convert_event(StreamCompleteEvent(message=msg)))

        # With no variants, raw names are used → they differ from initial → event emitted
        assert ctx.assistant_msg.info.model_id == "svc/kimi-k2"
        assert ctx.assistant_msg.info.provider_id == "openai"
        message_updated = [e for e in events if isinstance(e, MessageUpdatedEvent)]
        assert len(message_updated) == 1

    @pytest.mark.asyncio
    async def test_stream_complete_falls_back_when_no_variant_match(self) -> None:
        """StreamComplete falls back to raw names when model is not in any variant."""
        variants = _make_model_variants()
        ctx = _make_context(model_variants=variants)
        adapter = OpenCodeEventAdapter(context=ctx)

        # A model that doesn't match any variant
        msg = _make_stream_complete_msg(model_name="gpt-4o", provider_name="openai")
        events = await _collect_events(adapter.convert_event(StreamCompleteEvent(message=msg)))

        # No variant match → raw names used → differs from initial → event emitted
        assert ctx.assistant_msg.info.model_id == "gpt-4o"
        assert ctx.assistant_msg.info.provider_id == "openai"
        message_updated = [e for e in events if isinstance(e, MessageUpdatedEvent)]
        assert len(message_updated) == 1
