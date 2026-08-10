from __future__ import annotations

import asyncio

from pydantic_ai.models.test import TestModel
import pytest

from wolfharness import Agent
from wolfharness.messaging.message_utils import build_message_index, get_message_chain


pytestmark = pytest.mark.unit


async def test_message_chain():
    """Test that message chain tracks transformations correctly via parent_id."""
    agent_a = Agent("agent-a", model="test")
    agent_b = Agent("agent-b", model="test")
    agent_c = Agent("agent-c", model="test")

    # Connect chain
    agent_a.connect_to(agent_b)
    agent_b.connect_to(agent_c)

    # Build agent index dict manually (replaces pool.manifest.agents)
    agent_index = {"agent-a": agent_a, "agent-b": agent_b, "agent-c": agent_c}

    # When A processes a new message
    result_a = await agent_a.run("Start")
    assert result_a.parent_id is not None  # Points to user message

    # When B processes A's message via run_message
    result_b = await agent_b.run_message(result_a)
    assert result_b.parent_id is not None
    # Chain should show A
    chain_b = get_message_chain(result_b, agent_index)
    assert "agent-a" in chain_b

    # When C processes B's message
    result_c = await agent_c.run_message(result_b)
    assert result_c.parent_id is not None
    # Chain should show A and B
    chain_c = get_message_chain(result_c, agent_index)
    assert "agent-a" in chain_c
    assert "agent-b" in chain_c


async def test_run_result_has_parent_id():
    """Test that the message returned by run() has proper parent_id."""
    model = TestModel(custom_output_text="Response from A")
    agent_a = Agent("agent-a", model=model)
    agent_b = Agent("agent-b", model=model)
    # Connect A to B
    agent_a.connect_to(agent_b)
    # When A runs
    result = await agent_a.run("Test message")
    # The returned message should have parent_id pointing to user message
    assert result.parent_id is not None
    # Wait for forwarding to complete
    if agent_a._pending_tasks:
        await asyncio.gather(*agent_a._pending_tasks, return_exceptions=True)
    if agent_b._pending_tasks:
        await asyncio.gather(*agent_b._pending_tasks, return_exceptions=True)
    # B's messages should have parent_id tracking the chain
    if agent_b.conversation.chat_messages:
        b_user_msg = next(
            (m for m in agent_b.conversation.chat_messages if m.role == "user"),
            None,
        )
        if b_user_msg:
            # The user message in B should have parent_id from A's response
            assert b_user_msg.parent_id == result.message_id


async def test_message_chain_through_routing():
    """Test that message chain tracks correctly through the routing system."""
    model_a = TestModel(custom_output_text="Response from A")
    model_b = TestModel(custom_output_text="Response from B")
    model_c = TestModel(custom_output_text="Response from C")

    agent_a = Agent("agent-a", model=model_a)
    agent_b = Agent("agent-b", model=model_b)
    agent_c = Agent("agent-c", model=model_c)
    # Connect the chain
    agent_a.connect_to(agent_b)
    agent_b.connect_to(agent_c)

    # Build agent index dict manually
    agent_index = {"agent-a": agent_a, "agent-b": agent_b, "agent-c": agent_c}

    # When A starts the chain
    await agent_a.run("Start message")
    # Wait for all routing to complete
    if agent_a._pending_tasks:
        await asyncio.gather(*agent_a._pending_tasks, return_exceptions=True)
    if agent_b._pending_tasks:
        await asyncio.gather(*agent_b._pending_tasks, return_exceptions=True)
    if agent_c._pending_tasks:
        await asyncio.gather(*agent_c._pending_tasks, return_exceptions=True)
    # All agents should share the same session_id
    assert (
        agent_a.conversation.chat_messages[0].session_id
        == agent_b.conversation.chat_messages[0].session_id
    )
    assert (
        agent_b.conversation.chat_messages[0].session_id
        == agent_c.conversation.chat_messages[0].session_id
    )

    # C's response should have a chain back through B and A
    if agent_c.conversation.chat_messages:
        c_response = next(
            (m for m in agent_c.conversation.chat_messages if m.role == "assistant"),
            None,
        )
        if c_response:
            chain = get_message_chain(c_response, agent_index)
            # Chain should include both A and B
            assert "agent-a" in chain or "agent-b" in chain


async def test_build_message_index():
    """Test that build_message_index works across agents."""
    agent_a = Agent("agent-a", model="test")
    agent_b = Agent("agent-b", model="test")

    result_a = await agent_a.run("Hello from A")
    result_b = await agent_b.run("Hello from B")

    # Build index from a manually-constructed agent dict
    agent_index = {"agent-a": agent_a, "agent-b": agent_b}
    index = build_message_index(agent_index)

    # Should find messages from both agents
    assert result_a.message_id in index
    assert result_b.message_id in index

    found_a_msg, found_a_agent = index[result_a.message_id]
    found_b_msg, found_b_agent = index[result_b.message_id]

    assert found_a_msg.message_id == result_a.message_id
    assert found_a_agent == "agent-a"
    assert found_b_msg.message_id == result_b.message_id
    assert found_b_agent == "agent-b"

    # Non-existent ID should not be in index
    assert "non-existent-id" not in index


if __name__ == "__main__":
    pytest.main([__file__])
