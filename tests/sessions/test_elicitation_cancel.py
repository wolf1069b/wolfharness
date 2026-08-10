"""Tests for elicitation cancellation and conversation history preservation.

This test verifies the fix for the bug where conversation history was lost
when elicitation (question tool) was cancelled by the user.

Bug: When user pressed ESC to cancel a question, and then sent a new message,
the agent would think it's the first message in the conversation.

Root cause: user_msg was only saved to conversation history when final_message
was not None. If elicitation was cancelled and no assistant response was generated,
user_msg was never saved.
"""

from __future__ import annotations

from pydantic_ai.models.test import TestModel
import pytest

from wolfharness import Agent
from wolfharness.agents.events import StreamCompleteEvent


pytestmark = pytest.mark.unit


async def test_user_message_preserved_when_no_assistant_response():
    """Test that user message is saved even when no assistant response is generated.

    This reproduces the bug scenario where:
    1. User sends message
    2. Agent starts processing but doesn't generate a response (e.g., elicitation cancelled)
    3. User sends another message
    4. Agent should remember the first message
    """
    # Use TestModel which will return an empty response
    test_model = TestModel()

    async with Agent(
        model=test_model,
        name="test_agent",
    ) as agent:
        # Send first message
        user_msg_1 = "Help me with something"

        # Collect events from the stream
        final_message = None
        async for event in agent.run_stream(user_msg_1):
            if isinstance(event, StreamCompleteEvent):
                final_message = event.message

        # Verify final message was received
        assert final_message is not None

        # Verify user message was saved to conversation history
        # This was the bug: user_msg was not saved if final_message handling failed
        assert len(agent.conversation.chat_messages) >= 1

        first_saved_msg = agent.conversation.chat_messages[0]
        assert first_saved_msg.role == "user"
        assert user_msg_1 in str(first_saved_msg.content)

        # Now send second message
        user_msg_2 = "What did I just ask?"

        final_message_2 = None
        async for event in agent.run_stream(user_msg_2):
            if isinstance(event, StreamCompleteEvent):
                final_message_2 = event.message

        assert final_message_2 is not None

        # Verify conversation history now has both user messages
        user_messages = [msg for msg in agent.conversation.chat_messages if msg.role == "user"]
        assert len(user_messages) == 2
        assert user_msg_1 in str(user_messages[0].content)
        assert user_msg_2 in str(user_messages[1].content)


async def test_conversation_history_accumulates_across_runs():
    """Test that conversation history accumulates correctly across multiple runs."""
    test_model = TestModel()

    async with Agent(
        model=test_model,
        name="test_agent",
    ) as agent:
        # Track initial state
        initial_count = len(agent.conversation.chat_messages)

        # Run 1
        async for _event in agent.run_stream("First message"):
            pass

        # Should have added user + assistant messages
        after_first = len(agent.conversation.chat_messages)
        assert after_first >= initial_count + 2

        # Run 2
        async for _event in agent.run_stream("Second message"):
            pass

        # Should have added more messages
        after_second = len(agent.conversation.chat_messages)
        assert after_second >= after_first + 2

        # Run 3 - with interruption simulation
        async for _event in agent.run_stream("Third message"):
            pass

        after_third = len(agent.conversation.chat_messages)
        assert after_third >= after_second + 2

        # Verify all user messages are present
        user_contents = [
            msg.content for msg in agent.conversation.chat_messages if msg.role == "user"
        ]
        assert any("First message" in str(c) for c in user_contents)
        assert any("Second message" in str(c) for c in user_contents)
        assert any("Third message" in str(c) for c in user_contents)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
