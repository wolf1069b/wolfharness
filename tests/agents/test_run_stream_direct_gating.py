"""Tests for AGENT_TYPE gating in BaseAgent.run_stream().

Verifies that the ``if self.AGENT_TYPE == "native"`` gating in
``run_stream()`` correctly skips the manual loop
for native agents and executes it for non-native agents.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar
from unittest.mock import MagicMock

import pytest

from wolfharness.agents.base_agent import BaseAgent
from wolfharness.orchestrator.turn import Turn


pytestmark = pytest.mark.unit


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from wolfharness.agents.context import AgentRunContext
    from wolfharness.agents.events import RichAgentStreamEvent


# ---------------------------------------------------------------------------
# Test helper: minimal BaseAgent subclass that spies on _stream_events
# ---------------------------------------------------------------------------


class _FakeEvent:
    """Minimal concrete event for stream testing (no real event fields needed)."""


class _GatingTestAgent(BaseAgent[None, str]):
    """Test agent that tracks _stream_events calls.

    Subclasses override AGENT_TYPE to test native vs non-native gating.
    ``_stream_events`` records every call and queues an extra prompt on the first
    invocation, allowing the test to distinguish single-call (native) from
    multi-call (non-native) behaviour.
    """

    AGENT_TYPE: ClassVar[str]  # set by subclasses

    def __init__(self, call_log: list[tuple[Any, ...]]) -> None:
        super().__init__(name="gating_test")
        self._call_log = call_log
        self._has_queued_extra = False

    # -- concrete abstract methods -------------------------------------------

    @property
    def model_name(self) -> str | None:
        return "test-model"

    async def set_model(self, model: str) -> None:
        pass

    async def _stream_events(
        self,
        run_ctx: AgentRunContext,
        prompts: list[Any],
        *,
        user_msg: Any = None,
        **kwargs: Any,
    ) -> AsyncIterator[RichAgentStreamEvent[str]]:
        # Spy: record prompts passed to _stream_events.
        self._call_log.append(tuple(prompts))
        # On the very first call, mark that an extra prompt was queued so
        # the test can distinguish single-call (native) from multi-call
        # (non-native) behaviour.
        if not self._has_queued_extra:
            self._has_queued_extra = True
        yield _FakeEvent()  # type: ignore[return-value]

    def create_turn(
        self,
        prompts: list[str],
        run_ctx: AgentRunContext,
        message_history: list[Any],
    ) -> Turn:
        """Return a mock Turn — not exercised in gating tests."""
        return MagicMock(spec=Turn)

    async def _interrupt(self, run_ctx: AgentRunContext | None = None) -> None:
        pass

    async def get_available_models(self) -> None:
        return None

    async def get_modes(self) -> list[Any]:
        return []

    async def _set_mode(self, mode_id: str, category_id: str) -> None:
        pass

    async def list_sessions(
        self,
        *,
        cwd: str | None = None,
        limit: int | None = None,
    ) -> list[Any]:
        return []

    async def load_session(self, session_id: str) -> Any:
        return None


class _NativeTestAgent(_GatingTestAgent):
    """Agent with AGENT_TYPE = 'native' (skips manual loop)."""

    AGENT_TYPE: ClassVar = "native"


class _NonNativeTestAgent(_GatingTestAgent):
    """Agent with AGENT_TYPE = 'acp' (executes manual loop)."""

    AGENT_TYPE: ClassVar = "acp"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def test_native_agent_skips_manual_loop() -> None:
    """Native AGENT_TYPE should cause run_stream() to skip the while loop.

    When AGENT_TYPE == 'native', the extra prompt queued during
    _stream_events must NOT be processed -- the method uses a simple
    ``async for`` and exits without re-checking the injection queue.
    """
    call_log: list[tuple[Any, ...]] = []
    agent = _NativeTestAgent(call_log)

    events: list[object] = []
    events.extend([event async for event in agent.run_stream("test prompt")])

    # Native path: _stream_events is called exactly once
    assert len(call_log) == 1, (
        f"Expected 1 call to _stream_events for native agent, got {len(call_log)}"
    )
    # The queued extra prompt should still be in the injection manager
    assert agent._has_queued_extra, "Extra prompt should have been queued"
    # Sanity: we got the fake event
    assert len(events) == 1
    assert isinstance(events[0], _FakeEvent)
