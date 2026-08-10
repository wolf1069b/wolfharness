"""Red-flag test: Verify conversation history is preserved after run failure.

This test reproduces the user-reported bug:
1. Send prompt → succeeds (builds conversation)
2. Send prompt → model fails (RunFailedEvent)
3. Send prompt → should have conversation from steps 1 and 2
"""

from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest

from wolfharness import AgentPool, AgentsManifest, NativeAgentConfig
from wolfharness.agents.native_agent.turn import NativeTurn
from wolfharness.lifecycle.types import DeliveryMode


pytestmark = pytest.mark.integration


@pytest.fixture
def manifest() -> AgentsManifest:
    """Create a minimal manifest with a native agent."""
    agent_config = NativeAgentConfig(
        name="test_agent",
        model="test",
        system_prompt="You are a helpful assistant.",
    )
    return AgentsManifest(agents={"test_agent": agent_config})


@pytest.mark.anyio
async def test_conversation_preserved_after_run_failure(
    manifest: AgentsManifest,
) -> None:
    """Red-flag: Conversation history is preserved when a run fails and a new prompt is sent.

    Steps:
    1. Run first prompt → succeeds → conversation has 2 messages (user + assistant)
    2. Run second prompt → fails with exception → conversation has 3 messages (no assistant added
    for failed run)
    3. Run third prompt → succeeds → should have at least 4 messages (including step 1's history)
    """
    async with AgentPool(manifest) as pool:
        session_pool = pool.session_pool
        assert session_pool is not None

        session_id = "test-redflag-session"

        # Create session first (required by SessionController.receive_request)
        await session_pool.create_session(session_id, agent_name="test_agent")

        # --- Step 1: Successful first prompt ---
        msg_id = await session_pool.send_message(
            session_id,
            "Hello, what is 2+2?",
            mode=DeliveryMode.QUEUE,
        )
        assert msg_id is not None, "First prompt should start a run"
        await session_pool.wait_for_completion(session_id)

        # Get the per-session agent and check conversation
        agent1 = await session_pool.sessions.get_or_create_session_agent(session_id)
        conv = agent1.conversation
        msgs_after_step1 = list(conv.chat_messages)
        agent1_id = id(agent1)

        assert len(msgs_after_step1) >= 2, (
            f"Expected at least 2 messages after first prompt (user + assistant), "
            f"got {len(msgs_after_step1)}"
        )
        # First message should be user, second should be assistant
        assert msgs_after_step1[0].role == "user", (
            f"Expected user role, got {msgs_after_step1[0].role}"
        )
        assert msgs_after_step1[1].role == "assistant", (
            f"Expected assistant role, got {msgs_after_step1[1].role}"
        )

        # --- Step 2: Failed second prompt ---
        # Patch NativeTurn.execute to simulate a model failure that occurs AFTER
        # the user message is saved to the conversation (matching real scenario).
        # RunHandle.start() saves user msg, then calls turn.execute() → FAILS
        with patch.object(
            NativeTurn, "execute", side_effect=RuntimeError("Simulated model API error")
        ):
            msg_id2 = await session_pool.send_message(
                session_id,
                "What is 3+3?",
                mode=DeliveryMode.QUEUE,
            )
            assert msg_id2 is not None, "Second prompt should start a run"
            await session_pool.wait_for_completion(session_id)

        # Give time for cleanup
        await asyncio.sleep(0.1)

        # Verify the same agent instance is used
        agent2 = await session_pool.sessions.get_or_create_session_agent(session_id)
        assert id(agent2) == agent1_id, (
            f"Expected same agent instance ({agent1_id}), got different ({id(agent2)}). "
            "Agent was recreated — conversation history would be lost."
        )

        # Check conversation after failed run
        msgs_after_step2 = list(agent2.conversation.chat_messages)
        # After failure: should have step1 messages + step2 user message (no assistant)
        assert len(msgs_after_step2) >= len(msgs_after_step1) + 1, (
            f"Expected at least {len(msgs_after_step1) + 1} messages after failed run "
            f"(step1 messages + failed user prompt), got {len(msgs_after_step2)}"
        )
        # The last message should be the user message from step 2
        assert msgs_after_step2[-1].role == "user", (
            f"Last message should be user (from failed run), got role={msgs_after_step2[-1].role}"
        )

        # --- Step 3: Third prompt (should have full context) ---
        msg_id3 = await session_pool.send_message(
            session_id,
            "What was the first question I asked?",
            mode=DeliveryMode.QUEUE,
        )
        assert msg_id3 is not None, "Third prompt should start a run"
        await session_pool.wait_for_completion(session_id)

        # Check conversation after third run
        agent3 = await session_pool.sessions.get_or_create_session_agent(session_id)
        msgs_after_step3 = list(agent3.conversation.chat_messages)

        # Should have all previous messages + new user + new assistant
        assert len(msgs_after_step3) >= len(msgs_after_step2) + 2, (
            f"Expected at least {len(msgs_after_step2) + 2} messages after third prompt "
            f"(previous + new user + assistant), got {len(msgs_after_step3)}. "
            "This is a RED FLAG — conversation history was lost after a failed run!"
        )

        # The conversation should still contain the first question
        first_user_msg = msgs_after_step1[0]
        assert first_user_msg in msgs_after_step3 or any(
            "2+2" in str(m.content) for m in msgs_after_step3
        ), (
            "RED FLAG: First prompt's user message not found in conversation after third run. "
            "Conversation history was lost!"
        )


@pytest.mark.anyio
async def test_agent_identity_preserved_after_failure(
    manifest: AgentsManifest,
) -> None:
    """Red-flag: The same agent instance is used after a run failure.

    If get_or_create_session_agent returns a different agent instance,
    the conversation history stored on the old agent is lost.
    """
    async with AgentPool(manifest) as pool:
        session_pool = pool.session_pool
        assert session_pool is not None

        session_id = "test-identity-session"

        # Create session first
        await session_pool.create_session(session_id, agent_name="test_agent")

        # First successful run
        msg_id = await session_pool.send_message(session_id, "Hello")
        assert msg_id is not None
        await session_pool.wait_for_completion(session_id)

        agent_before = await session_pool.sessions.get_or_create_session_agent(session_id)
        agent_id_before = id(agent_before)
        msg_count_before = len(list(agent_before.conversation.chat_messages))

        # Second run — simulate failure during turn execution (after user msg saved)
        with patch.object(NativeTurn, "execute", side_effect=RuntimeError("Simulated failure")):
            msg_id2 = await session_pool.send_message(session_id, "Fail me")
            assert msg_id2 is not None
            await session_pool.wait_for_completion(session_id)

        await asyncio.sleep(0.1)

        agent_after = await session_pool.sessions.get_or_create_session_agent(session_id)
        agent_id_after = id(agent_after)

        assert agent_id_after == agent_id_before, (
            f"RED FLAG: Agent identity changed! Before={agent_id_before}, After={agent_id_after}. "
            "If the agent was recreated, all conversation history is lost."
        )

        msg_count_after = len(list(agent_after.conversation.chat_messages))
        assert msg_count_after > msg_count_before, (
            f"RED FLAG: Message count did not increase after failed run. "
            f"Before={msg_count_before}, After={msg_count_after}. "
            "The failed run's user message should have been added to the conversation."
        )
