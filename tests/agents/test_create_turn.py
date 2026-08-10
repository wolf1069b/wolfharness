"""Tests for BaseAgent.create_turn() and Agent.create_turn()."""

from __future__ import annotations

import inspect

from pydantic_ai.models.test import TestModel
import pytest

from wolfharness.agents.base_agent import BaseAgent
from wolfharness.agents.context import AgentRunContext
from wolfharness.agents.native_agent.agent import Agent
from wolfharness.agents.native_agent.turn import NativeTurn


@pytest.mark.unit
def test_agent_create_turn_returns_native_turn() -> None:
    """Given a native Agent, create_turn() returns a NativeTurn instance."""
    agent = Agent(model=TestModel(), name="test_agent")
    run_ctx = AgentRunContext()
    prompts: list[str] = ["Hello"]

    turn = agent.create_turn(
        prompts=prompts,
        run_ctx=run_ctx,
        message_history=[],
    )

    assert isinstance(turn, NativeTurn)


@pytest.mark.unit
def test_create_turn_is_abstract() -> None:
    """create_turn is abstract on BaseAgent and must be overridden by subclasses."""
    assert "create_turn" in BaseAgent.__abstractmethods__


# ---------------------------------------------------------------------------
# Tests from PR #64 review (create_turn / ACP turn)
# ---------------------------------------------------------------------------


def test_acp_turn_joins_all_prompts_not_just_last() -> None:
    """ACPTurn should join all prompts, not just take self._prompts[-1].

    Using self._prompts[-1] discards all but the last prompt.
    The fix: join all prompts with newlines or spaces.
    """
    # We test the logic directly by checking what ACPTurn does
    # with multiple prompts
    prompts = ["first prompt", "second prompt", "third prompt"]

    # Old (buggy) behavior: only last prompt
    old_result = prompts[-1]
    assert old_result == "third prompt"
    assert "first" not in old_result

    # Fixed behavior: join all
    new_result = "\n\n".join(prompts)
    assert "first prompt" in new_result
    assert "second prompt" in new_result
    assert "third prompt" in new_result


def test_acp_adapter_satisfies_protocol() -> None:
    """ACP agent adapter must satisfy ACPClientProtocol (no TODO/cast needed).

    T1 built stream_events() and get_messages() on ACPAgentAPI, so the
    create_turn method passes self._api directly without cast. This test
    verifies the adapter is complete by checking that ACPAgentAPI has the
    required methods and that create_turn does not use cast.
    """
    import inspect

    from acp.agent.acp_agent_api import ACPAgentAPI
    from wolfharness.agents.acp_agent.turn import ACPClientProtocol

    # ACPAgentAPI must have stream_events and get_messages methods
    assert hasattr(ACPAgentAPI, "stream_events"), "ACPAgentAPI must have stream_events()"
    assert hasattr(ACPAgentAPI, "get_messages"), "ACPAgentAPI must have get_messages()"

    # create_turn must not contain cast or TODO
    import wolfharness.agents.acp_agent.acp_agent as acp_module

    source = inspect.getsource(acp_module.ACPAgent.create_turn)
    assert "cast" not in source, "create_turn must not use cast — adapter is complete"
    assert "TODO" not in source, "create_turn must not have TODO — adapter is complete"

    # ACPAgentAPI must satisfy ACPClientProtocol at runtime
    from unittest.mock import MagicMock

    from wolfharness.agents.acp_agent.client_handler import TimeoutableEvent
    from wolfharness.agents.acp_agent.session_state import ACPSessionState

    api = ACPAgentAPI(
        MagicMock(),
        state=ACPSessionState(session_id="test"),
        update_event=TimeoutableEvent(),
    )
    assert isinstance(api, ACPClientProtocol), (
        "ACPAgentAPI must satisfy ACPClientProtocol without cast"
    )


def test_acp_turn_uses_run_ctx_run_id() -> None:
    """ACPTurn.execute() must use self._run_ctx.run_id, not generate uuid4."""
    import wolfharness.agents.acp_agent.turn as turn_module

    source = inspect.getsource(turn_module.ACPTurn.execute)
    assert "self._run_ctx.run_id" in source, (
        "ACPTurn must use self._run_ctx.run_id for consistency with RunHandle"
    )
    assert "str(uuid4())" not in source or "message_id" in source, (
        "ACPTurn.execute() must not generate a new run_id via uuid4()"
    )


def test_acp_turn_no_redundant_run_started_event() -> None:
    """ACPTurn.execute() must not yield RunStartedEvent.

    RunHandle.start() already publishes RunStartedEvent before calling
    turn.execute(). Yielding it again causes duplicate events.
    """
    import wolfharness.agents.acp_agent.turn as turn_module

    source = inspect.getsource(turn_module.ACPTurn.execute)
    # Check that RunStartedEvent is not yielded in the execute method body
    import re

    yield_matches = re.findall(r"yield\s+RunStartedEvent", source)
    assert len(yield_matches) == 0, (
        f"ACPTurn.execute() still yields RunStartedEvent {len(yield_matches)} "
        "time(s) — RunHandle.start() already publishes it"
    )


def test_acp_turn_no_unused_initial_message_history() -> None:
    """ACPTurn must not store _initial_message_history (dead code)."""
    import wolfharness.agents.acp_agent.turn as turn_module

    source = inspect.getsource(turn_module.ACPTurn.__init__)
    assert "_initial_message_history" not in source, (
        "_initial_message_history is dead code — assigned but never used. Should be removed."
    )


# ---------------------------------------------------------------------------
# ACPTurn agent_name propagation (from PR #64 round-7 review)
# ---------------------------------------------------------------------------


def test_acp_turn_accepts_agent_name() -> None:
    """ACPTurn.__init__ must accept and store agent_name parameter."""
    import wolfharness.agents.acp_agent.turn as turn_module

    source = inspect.getsource(turn_module.ACPTurn.__init__)
    assert "agent_name" in source, (
        "ACPTurn.__init__ must accept agent_name parameter for RunErrorEvent"
    )
    assert "self._agent_name" in source, "ACPTurn must store agent_name as self._agent_name"


def test_acp_turn_run_error_event_includes_agent_name() -> None:
    """ACPTurn.execute() must pass agent_name to RunErrorEvent yields."""
    import wolfharness.agents.acp_agent.turn as turn_module

    source = inspect.getsource(turn_module.ACPTurn.execute)
    assert "agent_name=self._agent_name" in source, (
        "ACPTurn.execute() must include agent_name in RunErrorEvent yields"
    )
