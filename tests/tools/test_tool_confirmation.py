from __future__ import annotations

from pydantic_ai.models.test import TestModel
import pytest

from wolfharness import Agent, Tool
from wolfharness.ui.mock_provider import MockInputProvider


pytestmark = pytest.mark.unit


async def test_tool_confirmation():
    # Create two tools - one requiring confirmation, one not
    def tool_with_confirm(text: str) -> str:
        """Tool requiring confirmation."""
        return f"Confirmed tool got: {text}"

    def tool_without_confirm(text: str) -> str:
        """Tool not requiring confirmation."""
        return f"Regular tool got: {text}"

    tool_info_with = Tool.from_callable(
        tool_with_confirm,
        requires_confirmation=True,
    )
    tool_info_without = Tool.from_callable(
        tool_without_confirm,
        requires_confirmation=False,
    )

    mock = MockInputProvider(tool_confirmation="allow")
    model = TestModel(call_tools=[tool_info_with.name])

    agent = Agent("test-agent", model=model, input_provider=mock)
    agent._builtin_provider.register_tool(tool_info_with)

    # Run agent - should trigger confirmation for the tool
    await agent.run("test")

    # Verify confirmation was requested
    assert len(mock.calls) == 1
    call = mock.calls[0]
    assert call.method == "get_tool_confirmation"
    context = call.args[0]
    assert context.tool_name == tool_info_with.name

    # Test tool without confirmation requirement
    mock = MockInputProvider(tool_confirmation="allow")
    model = TestModel(call_tools=[tool_info_without.name])

    agent = Agent("test-agent", model=model, input_provider=mock)
    agent._builtin_provider.register_tool(tool_info_without)

    # Run agent - should NOT trigger confirmation
    await agent.run("test")

    # Verify no confirmation was requested
    assert len(mock.calls) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-vv"])
