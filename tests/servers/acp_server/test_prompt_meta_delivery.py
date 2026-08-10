"""Unit tests for _meta.delivery extraction at acp_agent.py:prompt().

Tests the ACTUAL extraction logic in ``AgentPoolACPAgent.prompt()`` —
not just the downstream ``handle_prompt(delivery=...)`` forwarding. Verifies
that various ``_meta`` shapes are correctly parsed and invalid values are
filtered out.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from acp.schema.client_requests import PromptRequest
from acp.schema.content_blocks import TextContentBlock


pytestmark = [pytest.mark.unit, pytest.mark.anyio]


def _make_agent() -> MagicMock:
    """Create a minimal AgentPoolACPAgent double for testing prompt().

    Bypasses __init__ and sets only the attributes that prompt() reads.
    """
    agent = MagicMock()
    agent._initialized = True
    agent._protocol_handler = MagicMock()
    agent._protocol_handler.handle_prompt = AsyncMock(return_value=MagicMock())
    return agent


async def _call_prompt(agent: MagicMock, meta: dict | None) -> None:
    """Call agent.prompt() with a PromptRequest carrying the given _meta."""
    from wolfharness_server.acp_server.acp_agent import AgentPoolACPAgent

    request = PromptRequest(
        session_id="test-session",
        prompt=[TextContentBlock(type="text", text="hello")],
        field_meta=meta,  # type: ignore[arg-type]
    )
    # Bind prompt() as an unbound method on the mock
    await AgentPoolACPAgent.prompt(agent, request)  # type: ignore[arg-type]


async def test_meta_delivery_steer_extracted() -> None:
    """_meta={"delivery": "steer"} → handle_prompt(delivery="steer")."""
    agent = _make_agent()
    await _call_prompt(agent, {"delivery": "steer"})
    agent._protocol_handler.handle_prompt.assert_called_once()
    assert agent._protocol_handler.handle_prompt.call_args.kwargs["delivery"] == "steer"


async def test_meta_delivery_followup_extracted() -> None:
    """_meta={"delivery": "followup"} → handle_prompt(delivery="followup")."""
    agent = _make_agent()
    await _call_prompt(agent, {"delivery": "followup"})
    agent._protocol_handler.handle_prompt.assert_called_once()
    assert agent._protocol_handler.handle_prompt.call_args.kwargs["delivery"] == "followup"


async def test_meta_delivery_absent_defaults_to_none() -> None:
    """_meta without "delivery" key → handle_prompt(delivery=None)."""
    agent = _make_agent()
    await _call_prompt(agent, {"traceparent": "00-..."})
    agent._protocol_handler.handle_prompt.assert_called_once()
    assert agent._protocol_handler.handle_prompt.call_args.kwargs["delivery"] is None


async def test_meta_none_defaults_to_none() -> None:
    """_meta=None → handle_prompt(delivery=None)."""
    agent = _make_agent()
    await _call_prompt(agent, None)
    agent._protocol_handler.handle_prompt.assert_called_once()
    assert agent._protocol_handler.handle_prompt.call_args.kwargs["delivery"] is None


async def test_meta_delivery_invalid_string_filtered() -> None:
    """_meta={"delivery": "invalid"} → handle_prompt(delivery=None).

    Only "steer" and "followup" are valid values; anything else is filtered.
    """
    agent = _make_agent()
    await _call_prompt(agent, {"delivery": "invalid"})
    agent._protocol_handler.handle_prompt.assert_called_once()
    assert agent._protocol_handler.handle_prompt.call_args.kwargs["delivery"] is None


async def test_meta_delivery_non_string_filtered() -> None:
    """_meta={"delivery": 123} → handle_prompt(delivery=None).

    Non-string values are filtered by the isinstance check.
    """
    agent = _make_agent()
    await _call_prompt(agent, {"delivery": 123})
    agent._protocol_handler.handle_prompt.assert_called_once()
    assert agent._protocol_handler.handle_prompt.call_args.kwargs["delivery"] is None


async def test_prompt_raises_when_not_initialized() -> None:
    """prompt() raises RuntimeError when agent is not initialized."""
    agent = _make_agent()
    agent._initialized = False
    with pytest.raises(RuntimeError, match="not initialized"):
        await _call_prompt(agent, {"delivery": "steer"})


async def test_prompt_raises_when_no_protocol_handler() -> None:
    """prompt() raises RuntimeError when no protocol handler is configured."""
    agent = _make_agent()
    agent._protocol_handler = None
    with pytest.raises(RuntimeError, match="No protocol handler"):
        await _call_prompt(agent, {"delivery": "steer"})
