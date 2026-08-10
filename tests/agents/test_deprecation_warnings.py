"""Tests for deprecation warnings on ``inject_prompt`` and ``queue_prompt``.

Pooled native agents (those with ``agent_pool is not None``) should emit
a ``DeprecationWarning`` when calling ``inject_prompt()`` or ``queue_prompt()``,
guiding users toward ``SessionPool.steer()`` and ``SessionPool.followup()``.

Standalone native agents and non-native agents should NOT emit any warning.


# TODO: L2 migration — test requires complex mock pool dependencies that
# cannot be easily replaced with a real pool. Needs investigation.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
import warnings

from pydantic_ai.models.test import TestModel
import pytest

from wolfharness import Agent, AgentPool


pytestmark = pytest.mark.integration


TEST_RESPONSE = "I am a test response"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pooled_native_agent() -> Agent:
    """Create a native Agent with a mocked agent_pool / session_pool.

    ``task_manager.fire_and_forget()`` schedules the async coroutines
    (``steer`` / ``followup``) without error.
    """
    agent = Agent(name="test-agent", model=TestModel(custom_output_text=TEST_RESPONSE))
    session_pool = MagicMock()
    session_pool.steer = AsyncMock(return_value=True)
    session_pool.followup = AsyncMock(return_value=True)

    mock_pool = MagicMock(spec=AgentPool)
    # host_context calls _agent_pool.get_context(); configure it to return
    # a context with the session_pool mock.
    mock_context = MagicMock()
    mock_context.session_pool = session_pool
    mock_pool.get_context.return_value = mock_context
    mock_pool.session_pool = session_pool

    agent.agent_pool = mock_pool
    agent._events.session_id = "test-session-id"
    return agent


# ---------------------------------------------------------------------------
# Pooled native agents — DeprecationWarning expected
# ---------------------------------------------------------------------------


async def test_pooled_native_inject_prompt_deprecation_warning() -> None:
    """Pooled native inject_prompt() emits DeprecationWarning."""
    agent = _make_pooled_native_agent()

    with pytest.warns(DeprecationWarning, match="inject_prompt"):
        agent.inject_prompt("test message")


async def test_pooled_native_queue_prompt_deprecation_warning() -> None:
    """Pooled native queue_prompt() emits DeprecationWarning."""
    agent = _make_pooled_native_agent()

    with pytest.warns(DeprecationWarning, match="queue_prompt"):
        agent.queue_prompt("follow-up message")


# ---------------------------------------------------------------------------
# Standalone native agent — no warning
# ---------------------------------------------------------------------------


def test_standalone_native_inject_prompt_no_deprecation() -> None:
    """Standalone native inject_prompt() does NOT emit DeprecationWarning."""
    agent = Agent(name="test-agent", model=TestModel(custom_output_text=TEST_RESPONSE))
    # agent_pool is None by default for standalone agents

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        # Should complete without raising DeprecationWarning
        agent.inject_prompt("test message")


# ---------------------------------------------------------------------------
# Non-native agent — no warning
# ---------------------------------------------------------------------------


def test_non_native_inject_prompt_no_deprecation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Non-native inject_prompt() does NOT emit DeprecationWarning.

    Uses a native Agent instance with ``AGENT_TYPE`` temporarily overridden to
    ``"acp"`` to simulate an ACP agent without spawning a subprocess.
    """
    agent = Agent(name="test-agent", model=TestModel(custom_output_text=TEST_RESPONSE))
    monkeypatch.setattr(agent, "AGENT_TYPE", "acp")

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        # Should complete without raising DeprecationWarning
        agent.inject_prompt("test message")
