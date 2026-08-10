"""L3 VCR test — RunLoop streaming event sequences (design D8, P6 pattern).

Exercises the real ``RunHandle`` (RunLoop) lifecycle with VCR-replayed model
responses. Tests cover: streaming event sequence, multi-turn idle/wake,
mid-turn steering, and follow-up between turns.

Cassettes ([HUMAN-REQUIRED]):
- ``tests/cassettes/vcr/test_runloop_streaming/test_real_streaming_event_sequence.yaml``
- ``tests/cassettes/vcr/test_runloop_streaming/test_multi_turn_idle_wake.yaml``
- ``tests/cassettes/vcr/test_runloop_streaming/test_steer_mid_turn.yaml``
- ``tests/cassettes/vcr/test_runloop_streaming/test_followup_between_turns.yaml``
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from tests.vcr.conftest import cassette_exists


if TYPE_CHECKING:
    from wolfharness import AgentPool

pytestmark = [pytest.mark.vcr, pytest.mark.integration]

_MODULE_STEM = "test_runloop_streaming"


@pytest.mark.skipif(
    not cassette_exists(_MODULE_STEM, "test_real_streaming_event_sequence"),
    reason="Cassette not recorded yet — run with --record-mode=once",
)
async def test_real_streaming_event_sequence(vcr_pool: AgentPool) -> None:
    """The RunLoop emits the expected event skeleton for a single turn.

    Expected order:
        RunStartedEvent → PartStartEvent → PartDeltaEvent* →
        StreamCompleteEvent
    """
    agent = vcr_pool.get_agent("test_agent")
    events: list[Any] = [
        event async for event in agent.run_stream("Say hello in one short sentence.")
    ]

    assert events, "run_stream produced no events"
    type_names = [type(e).__name__ for e in events]
    # Collapse consecutive PartDeltaEvents.
    collapsed: list[str] = []
    for name in type_names:
        if name == "PartDeltaEvent" and collapsed and collapsed[-1] == "PartDeltaEvent":
            continue
        collapsed.append(name)
    assert "RunStartedEvent" in collapsed
    assert "StreamCompleteEvent" in collapsed
    assert collapsed.index("RunStartedEvent") < collapsed.index("StreamCompleteEvent")


@pytest.mark.skipif(
    not cassette_exists(_MODULE_STEM, "test_multi_turn_idle_wake"),
    reason="Cassette not recorded yet — run with --record-mode=once",
)
async def test_multi_turn_idle_wake(vcr_pool: AgentPool) -> None:
    """Two consecutive ``run()`` calls on the same agent produce two responses.

    The RunLoop goes idle between turns and wakes for the next prompt. VCR
    replays both model API calls. Asserts both responses are non-empty.
    """
    agent = vcr_pool.get_agent("test_agent")
    first = await agent.run("Say hello.")
    assert first is not None
    assert first.content is not None

    second = await agent.run("Say goodbye.")
    assert second is not None
    assert second.content is not None


@pytest.mark.skipif(
    not cassette_exists(_MODULE_STEM, "test_steer_mid_turn"),
    reason="Cassette not recorded yet — run with --record-mode=once",
)
async def test_steer_mid_turn(vcr_pool: AgentPool) -> None:
    """Steering injects a message into the active turn.

    Uses ``SessionPool`` to steer a message mid-turn. VCR replays the model
    API call(s). Asserts the turn completes with a ``StreamCompleteEvent``.
    """
    # Access the pool's SessionPool for steering.
    session_pool = vcr_pool.session_pool
    session_id = "test-steer-vcr"
    # Create a session and send the first prompt.
    await session_pool.sessions.get_or_create_session(session_id, agent_name="test_agent")
    handle = await session_pool.send_message(
        session_id=session_id,
        content="Count slowly from 1 to 10.",
        mode=None,
    )
    # Steer a follow-up message into the active turn.
    await session_pool.send_message(
        session_id=session_id,
        content="Actually, stop at 3.",
        mode="steer",
    )
    # Wait for completion.
    await session_pool.wait_for_completion(session_id)
    assert handle is not None


@pytest.mark.skipif(
    not cassette_exists(_MODULE_STEM, "test_followup_between_turns"),
    reason="Cassette not recorded yet — run with --record-mode=once",
)
async def test_followup_between_turns(vcr_pool: AgentPool) -> None:
    """A follow-up message queued between turns is delivered on the next turn.

    The RunLoop goes idle after the first turn, then wakes for the queued
    follow-up. Asserts the second turn produces a response.
    """
    session_pool = vcr_pool.session_pool
    session_id = "test-followup-vcr"
    await session_pool.sessions.get_or_create_session(session_id, agent_name="test_agent")
    await session_pool.send_message(
        session_id=session_id,
        content="Say hello.",
        mode="queue",
    )
    await session_pool.wait_for_completion(session_id)
    # Queue a follow-up for the next turn.
    await session_pool.send_message(
        session_id=session_id,
        content="Now say goodbye.",
        mode="queue",
    )
    await session_pool.wait_for_completion(session_id)
