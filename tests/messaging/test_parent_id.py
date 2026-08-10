"""Tests for parent_id tree structure in conversations."""

from __future__ import annotations

from pydantic_ai.models.test import TestModel
import pytest

from wolfharness import Agent, ChatMessage


pytestmark = pytest.mark.unit


TEST_RESPONSE = "I am a test response"


@pytest.fixture
def test_agent() -> Agent[None]:
    """Create an agent with TestModel for testing."""
    model = TestModel(custom_output_text=TEST_RESPONSE)
    return Agent(name="test-agent", model=model)


class TestParentIdBasic:
    """Tests for basic parent_id linking in conversations."""

    async def test_first_user_message_has_no_parent(self, test_agent: Agent[None]):
        """First user message in a conversation should have no parent_id."""
        async for _event in test_agent.run_stream("Hello"):
            pass

        history = test_agent.conversation.get_history()
        assert len(history) == 2  # user + response

        user_msg = history[0]
        assert user_msg.role == "user"
        assert user_msg.parent_id is None  # First message has no parent

    async def test_response_has_user_message_as_parent(self, test_agent: Agent[None]):
        """Response message should have the user message as parent."""
        async for _event in test_agent.run_stream("Hello"):
            pass

        history = test_agent.conversation.get_history()
        user_msg = history[0]
        response_msg = history[1]

        assert response_msg.role == "assistant"
        assert response_msg.parent_id == user_msg.message_id

    async def test_multi_turn_parent_chain(self, test_agent: Agent[None]):
        """Multiple turns should form a proper parent chain."""
        # First turn
        async for _event in test_agent.run_stream("First message"):
            pass

        # Second turn
        async for _event in test_agent.run_stream("Second message"):
            pass

        history = test_agent.conversation.get_history()
        assert len(history) == 4  # 2 user + 2 response

        user_1 = history[0]
        response_1 = history[1]
        user_2 = history[2]
        response_2 = history[3]

        # First user message has no parent
        assert user_1.parent_id is None

        # First response points to first user message
        assert response_1.parent_id == user_1.message_id

        # Second user message points to first response
        assert user_2.parent_id == response_1.message_id

        # Second response points to second user message
        assert response_2.parent_id == user_2.message_id

    async def test_parent_chain_is_linear(self, test_agent: Agent[None]):
        """Verify the parent chain forms a linear sequence."""
        # Run 3 turns
        for i in range(3):
            async for _event in test_agent.run_stream(f"Message {i}"):
                pass

        history = test_agent.conversation.get_history()
        assert len(history) == 6  # 3 user + 3 response

        # Walk the chain from last to first
        current = history[-1]
        chain = [current]

        while current.parent_id is not None:
            # Find parent in history
            parent = next(m for m in history if m.message_id == current.parent_id)
            chain.append(parent)
            current = parent

        # Chain should contain all messages
        assert len(chain) == 6

        # First message (end of chain) should have no parent
        assert chain[-1].parent_id is None
        assert chain[-1].role == "user"


class TestParentIdWithRun:
    """Tests for parent_id with non-streaming run()."""

    async def test_run_sets_parent_id(self, test_agent: Agent[None]):
        """Non-streaming run() should also set parent_id correctly."""
        await test_agent.run("First")
        await test_agent.run("Second")

        history = test_agent.conversation.get_history()
        assert len(history) == 4

        user_1 = history[0]
        response_1 = history[1]
        user_2 = history[2]
        response_2 = history[3]

        # Check the chain
        assert user_1.parent_id is None
        assert response_1.parent_id == user_1.message_id
        assert user_2.parent_id == response_1.message_id
        assert response_2.parent_id == user_2.message_id


class TestChatMessageParentId:
    """Tests for ChatMessage.parent_id field directly."""

    def test_user_prompt_accepts_parent_id(self):
        """ChatMessage.user_prompt should accept parent_id parameter."""
        msg = ChatMessage.user_prompt("Hello", parent_id="parent-123")
        assert msg.parent_id == "parent-123"

    def test_user_prompt_defaults_to_none(self):
        """ChatMessage.user_prompt should default parent_id to None."""
        msg = ChatMessage.user_prompt("Hello")
        assert msg.parent_id is None

    def test_from_pydantic_ai_accepts_parent_id(self):
        """ChatMessage.from_pydantic_ai should accept parent_id parameter."""
        from pydantic_ai import ModelRequest

        request = ModelRequest(parts=[])
        msg = ChatMessage.from_pydantic_ai(
            content="Hello",
            message=request,
            parent_id="parent-456",
        )
        assert msg.parent_id == "parent-456"

    def test_from_pydantic_ai_response_accepts_parent_id(self):
        """ChatMessage.from_pydantic_ai should accept parent_id for responses."""
        from pydantic_ai import ModelResponse, TextPart

        response = ModelResponse(parts=[TextPart(content="Hi")])
        msg = ChatMessage.from_pydantic_ai(
            content="Hi",
            message=response,
            parent_id="parent-789",
        )
        assert msg.parent_id == "parent-789"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
