from __future__ import annotations

import inspect

from pydantic_ai.models.test import TestModel
import pytest

from wolfharness import Agent
from wolfharness.models.agents import NativeAgentConfig
from wolfharness.utils.inspection import get_fn_name
from wolfharness_config.session import MemoryConfig


pytestmark = pytest.mark.unit


# Helper to access the validation method from Agent class
def _validate_processor_signature(processor):
    """Validate processor signature using Agent's validation method."""
    # Create a minimal agent instance to access the method
    from wolfharness.agents.native_agent.agent import Agent as NativeAgent

    # Access as unbound method and call with self=None (internal use only)
    return NativeAgent._validate_processor_signature(None, processor)


@pytest.fixture
def mock_model():
    return TestModel(custom_output_text="Response")


def test_config_validation_empty():
    """Test MemoryConfig with empty processors list."""
    config = MemoryConfig(history_processors=[])
    assert config.history_processors == []


def test_config_validation_none():
    """Test MemoryConfig with None processors."""
    config = MemoryConfig(history_processors=None)
    assert config.history_processors is None


def test_validate_signature_too_many_args():
    """Test validation rejects processors with too many args."""

    def bad(a, b, c):
        return a

    with pytest.raises(ValueError, match="must take 1 or 2 arguments"):
        _validate_processor_signature(bad)


def test_validate_signature_wrong_name():
    """Test validation rejects processors with wrong second param name."""

    def bad(ctx, extra_arg):
        return ctx

    with pytest.raises(ValueError, match="must be messages/msgs/history"):
        _validate_processor_signature(bad)


def test_validate_signature_sync_no_ctx():
    """Test validation accepts sync processor without context."""

    def good(messages):
        return messages

    _validate_processor_signature(good)  # should not raise


def test_validate_signature_sync_ctx():
    """Test validation accepts sync processor with context."""

    def good(ctx, messages):
        return messages

    _validate_processor_signature(good)  # should not raise


def test_validate_signature_async():
    """Test validation accepts async processor."""

    async def good(messages):
        return messages

    _validate_processor_signature(good)  # should not raise


def test_config_resolution_invalid_path():
    """Test resolution with invalid import path."""
    config = NativeAgentConfig(
        model="test",
        session=MemoryConfig(history_processors=["invalid.module:func"]),
    )
    with pytest.raises(ValueError, match="Failed to resolve history processor"):
        config.get_history_processors()


def test_config_resolution_invalid_signature_too_many():
    """Test resolution with invalid signature (too many args)."""
    config = NativeAgentConfig(
        model="test",
        session=MemoryConfig(
            history_processors=["tests.test_processors:invalid_processor_too_many"]
        ),
    )
    with pytest.raises(ValueError, match="must take 1 or 2 arguments"):
        config.get_history_processors()


def test_config_resolution_second_param_name_allowed():
    """Second parameter name is not restricted (see NativeAgentConfig.get_history_processors)."""
    config = NativeAgentConfig(
        model="test",
        session=MemoryConfig(
            history_processors=["tests.test_processors:invalid_processor_wrong_name"]
        ),
    )
    processors = config.get_history_processors()
    assert len(processors) == 1
    assert processors[0].__name__ == "invalid_processor_wrong_name"


def test_config_resolution_sync_no_ctx():
    """Test resolution of sync processor without context."""
    config = NativeAgentConfig(
        model="test",
        session=MemoryConfig(history_processors=["tests.test_processors:keep_recent"]),
    )
    processors = config.get_history_processors()
    assert len(processors) == 1
    assert get_fn_name(processors[0]) == "keep_recent"


def test_config_resolution_async_no_ctx():
    """Test resolution of async processor without context."""
    config = NativeAgentConfig(
        model="test",
        session=MemoryConfig(history_processors=["tests.test_processors:filter_thinking_async"]),
    )
    processors = config.get_history_processors()
    assert len(processors) == 1
    assert get_fn_name(processors[0]) == "filter_thinking_async"
    assert inspect.iscoroutinefunction(processors[0])


def test_config_resolution_sync_ctx():
    """Test resolution of sync processor with context."""
    config = NativeAgentConfig(
        model="test",
        session=MemoryConfig(history_processors=["tests.test_processors:context_aware_sync"]),
    )
    processors = config.get_history_processors()
    assert len(processors) == 1
    assert get_fn_name(processors[0]) == "context_aware_sync"


def test_config_resolution_async_ctx():
    """Test resolution of async processor with context."""
    config = NativeAgentConfig(
        model="test",
        session=MemoryConfig(history_processors=["tests.test_processors:context_aware_async"]),
    )
    processors = config.get_history_processors()
    assert len(processors) == 1
    assert get_fn_name(processors[0]) == "context_aware_async"


def test_config_resolution_empty():
    """Test resolution with no processors configured."""
    config = NativeAgentConfig(model="test")
    assert config.get_history_processors() == []


def test_config_resolution_empty_list():
    """Test resolution with empty processors list."""
    config = NativeAgentConfig(
        model="test",
        session=MemoryConfig(history_processors=[]),
    )
    assert config.get_history_processors() == []


async def test_integration_processors_called(mock_model):
    """Integration test: Verify processor is actually called during run."""
    called = False

    def my_processor(messages):
        nonlocal called
        called = True
        return messages

    async with Agent(name="test", model=mock_model, history_processors=[my_processor]) as agent:
        await agent.run("Hello")
        assert called is True


async def test_multiple_processors_sequential(mock_model):
    """Verify multiple processors run in sequence."""
    order: list[int] = []

    def proc1(messages):
        order.append(1)
        return messages

    def proc2(messages):
        order.append(2)
        return messages

    async with Agent(name="test", model=mock_model, history_processors=[proc1, proc2]) as agent:
        await agent.run("Hello")
        assert order == [1, 2]


async def test_compatibility_no_processors(mock_model):
    """Verify agent works fine without processors."""
    async with Agent(name="test", model=mock_model) as agent:
        agentlet = await agent.get_agentlet(None, str)
        assert getattr(agentlet, "history_processors", []) == []
        result = await agent.run("Hello")
        assert result.data == "Response"


@pytest.mark.skip(
    reason="Run/Turn separation duplicates user prompt in message history;"
    " needs investigation of _message_history propagation in RunHandle"
)
async def test_compaction_and_processors_interaction(mock_model):
    """Test interaction between CompactionPipeline and history processors.

    Order should be:
    1. CompactionPipeline (filters/truncates)
    2. History Processors (receives already compacted messages)
    """
    from pydantic_ai import ModelResponse, TextPart

    from wolfharness.messaging import ChatMessage

    history = [
        ChatMessage.user_prompt("M1"),
        ChatMessage(
            role="assistant",
            content="R1",
            messages=[ModelResponse(parts=[TextPart(content="R1")])],
        ),
        ChatMessage.user_prompt("M2"),
        ChatMessage(
            role="assistant",
            content="R2",
            messages=[ModelResponse(parts=[TextPart(content="R2")])],
        ),
    ]

    seen_messages: list[object] = []

    def my_processor(messages):
        nonlocal seen_messages
        seen_messages = messages
        return messages

    async with Agent(name="test", model=mock_model, history_processors=[my_processor]) as agent:
        agent.conversation.set_history(history)
        await agent.run("Hello")

        # Processor should see all 4 history messages + 1 new user message = 5 total
        assert len(seen_messages) == 5
        assert "M1" in str(seen_messages[0])
        assert "R1" in str(seen_messages[1])
        assert "M2" in str(seen_messages[2])
        assert "R2" in str(seen_messages[3])
        assert "Hello" in str(seen_messages[4])
