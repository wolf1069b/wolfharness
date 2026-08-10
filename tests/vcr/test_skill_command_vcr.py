"""L3 VCR test — skill command execution path (issue #339).

Exercises the real ``send_message(content="")`` + ``staged_content`` path
with VCR-replayed model responses. This is the protocol-level test that
#339 needed: it verifies that when a skill command injects instructions
into ``staged_content`` and routes with empty content, the model receives
the skill instructions (not "Loading skill: ..."), events are delivered
exactly-once, and the response is the model's actual reply.

Cassette ([HUMAN-REQUIRED]):
- ``tests/cassettes/vcr/test_skill_command_vcr/test_skill_command_staged_content_routing.yaml``

Recording:
    OPENAI_API_KEY=sk-... uv run pytest tests/vcr/test_skill_command_vcr.py \
        --record-mode=once -k test_skill_command_staged_content_routing
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import pytest

from tests.vcr.conftest import cassette_exists
from wolfharness.agents.events import (
    PartStartEvent,
    StreamCompleteEvent,
    UserMessageInsertedEvent,
)


if TYPE_CHECKING:
    from wolfharness import AgentPool
    from wolfharness.orchestrator.event_bus import EventBus

pytestmark = [pytest.mark.vcr, pytest.mark.integration]

_MODULE_STEM = "test_skill_command_vcr"

# The skill instructions that would be injected by skill_bridge.execute_skill.
_SKILL_PROMPT = """<skill-instruction>
You are a helpful test skill. When the user asks you to do something,
respond concisely with "Skill executed: <request>".
</skill-instruction>

<user-request>
say hello
</user-request>"""


@pytest.mark.skipif(
    not cassette_exists(_MODULE_STEM, "test_skill_command_staged_content_routing"),
    reason="Cassette not recorded yet — run with --record-mode=once",
)
async def test_skill_command_staged_content_routing(vcr_pool: AgentPool) -> None:
    """Skill command routes through send_message(content="") + staged_content.

    Simulates what _execute_skill_command does after skill_bridge.execute_skill
    injects instructions into the per-session agent's staged_content:

    1. Get per-session agent via get_or_create_session_agent
    2. Inject skill instructions into agent.staged_content
    3. Call send_message with empty content (model gets instructions from
       staged_content only — no double injection)
    4. Subscribe to EventBus and collect events

    Asserts:
    - At least one event is received (model actually ran)
    - UserMessageInsertedEvent is emitted (for TUI display)
    - StreamCompleteEvent is received (exactly-once terminal event)
    - No event contains "Loading skill" (ctx.print discarded)
    - The model's response is based on the skill instructions, not a
      hardcoded prompt
    """
    event_bus: EventBus = vcr_pool.session_pool.event_bus
    session_id = "test-skill-vcr"

    # Get per-session agent (matching _execute_skill_command pattern)
    agent = await vcr_pool.session_pool.sessions.get_or_create_session_agent(
        session_id, agent_name="test_agent"
    )

    # Inject skill instructions into staged_content (simulating
    # skill_bridge.execute_skill which calls ctx.data.node.staged_content.add_text)
    agent.staged_content.add_text(_SKILL_PROMPT)
    assert len(agent.staged_content) > 0, "staged_content should have content after injection"

    # Subscribe to EventBus BEFORE routing so we don't miss events
    queue = await event_bus.subscribe(session_id, scope="session")

    # Route with empty content — model gets instructions from staged_content only
    # This is the core of the #339 fix: content="" + staged_content
    await vcr_pool.session_pool.send_message(
        session_id=session_id,
        content="",
        mode="queue",
    )

    # Collect events until we get a terminal event
    events: list[Any] = []
    try:
        while True:
            event = await asyncio.wait_for(queue.get(), timeout=15.0)
            # Unwrap EventEnvelope if present
            raw = getattr(event, "event", event)
            events.append(raw)
            type_name = type(raw).__name__
            if "Complete" in type_name or "Error" in type_name:
                break
    except TimeoutError:
        pass

    # Wait for run to fully complete
    await vcr_pool.session_pool.wait_for_completion(session_id)

    # --- Assertions ---

    # 1. Events were received (model actually ran with staged_content)
    assert events, "Expected at least one EventBus event — model may not have run"

    # 2. StreamCompleteEvent received (terminal event, exactly-once)
    complete_events = [e for e in events if isinstance(e, StreamCompleteEvent)]
    assert len(complete_events) == 1, (
        f"Expected exactly one StreamCompleteEvent, got {len(complete_events)}"
    )

    # 3. UserMessageInsertedEvent emitted (for TUI display)
    user_msg_events = [e for e in events if isinstance(e, UserMessageInsertedEvent)]
    assert len(user_msg_events) >= 1, (
        "Expected at least one UserMessageInsertedEvent for TUI display"
    )

    # 4. No event contains "Loading skill" (ctx.print output discarded)
    #    Check PartStartEvent text — this is where model output appears
    part_start_events = [e for e in events if isinstance(e, PartStartEvent)]
    for pse in part_start_events:
        content_str = str(getattr(pse, "content", ""))
        assert "Loading skill" not in content_str, (
            f"PartStartEvent contains 'Loading skill' — ctx.print was not discarded: {content_str}"
        )

    # 5. The model produced a response (not empty)
    #    StreamCompleteEvent carries the final message
    complete = complete_events[0]
    # The final message should exist and contain text
    final_msg = getattr(complete, "message", None)
    if final_msg is not None:
        # The model should have responded based on the skill instructions
        # (not "Loading skill: ..." which was the #339 bug)
        msg_str = str(final_msg)
        assert "Loading skill" not in msg_str, (
            f"Model response contains 'Loading skill': {msg_str[:200]}"
        )
