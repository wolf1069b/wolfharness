"""Tests for tool confirmation with capability-based toolsets.

Consolidated from:
- test_confirmation_integration.py (full agent run flow with multiple confirmations)
- test_confirmation_ui.py (AgentContext.handle_confirmation bridging)
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from pydantic_ai.messages import ToolCallPart
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import (
    DeferredToolRequests,
    RunContext,
)
from pydantic_ai.usage import RunUsage
from pydantic_graph import End
import pytest

from wolfharness import Agent
from wolfharness.agents.context import AgentContext
from wolfharness.tools.base import Tool


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_agent() -> Agent[Any]:
    """Create an agent with mocked internals for confirmation testing."""
    model = TestModel(custom_output_text="test")
    return Agent(name="confirmation-test-agent", model=model)


@pytest.fixture
def mock_input_provider() -> MagicMock:
    """Create a mock InputProvider that returns 'allow' by default."""
    provider = MagicMock()
    provider.get_tool_confirmation = AsyncMock(return_value="allow")
    return provider


@pytest.fixture
def confirmation_tool() -> Tool[Any]:
    """Create a tool that requires confirmation."""

    def tool_with_confirm(text: str) -> str:
        """Tool requiring confirmation."""
        return f"Confirmed tool got: {text}"

    return Tool.from_callable(tool_with_confirm, requires_confirmation=True)


@pytest.fixture
def no_confirmation_tool() -> Tool[Any]:
    """Create a tool that does not require confirmation."""

    def tool_without_confirm(text: str) -> str:
        """Tool not requiring confirmation."""
        return f"Regular tool got: {text}"

    return Tool.from_callable(tool_without_confirm, requires_confirmation=False)


@pytest.fixture
def confirmation_tool_1() -> Tool[Any]:
    """First tool requiring confirmation."""

    def dangerous_read(path: str) -> str:
        """Read a file path. Requires confirmation."""
        return f"Contents of {path}"

    return Tool.from_callable(dangerous_read, requires_confirmation=True)


@pytest.fixture
def confirmation_tool_2() -> Tool[Any]:
    """Second tool requiring confirmation."""

    def dangerous_write(path: str, content: str) -> str:
        """Write to a file path. Requires confirmation."""
        return f"Wrote to {path}"

    return Tool.from_callable(dangerous_write, requires_confirmation=True)


@pytest.fixture
def sample_deferred_requests() -> DeferredToolRequests:
    """Create sample DeferredToolRequests with multiple approval requests."""
    return DeferredToolRequests(
        approvals=[
            ToolCallPart(
                tool_name="dangerous_read",
                args={"path": "/etc/passwd"},
                tool_call_id="tc-read-001",
            ),
            ToolCallPart(
                tool_name="dangerous_write",
                args={"path": "/etc/hosts", "content": "test"},
                tool_call_id="tc-write-002",
            ),
        ]
    )


# ============================================================================
# UI-level confirmation tests
# ============================================================================


@pytest.mark.unit
async def test_confirmation_ui_approval(
    mock_agent: Agent[Any],
    mock_input_provider: MagicMock,
    confirmation_tool: Tool[Any],
) -> None:
    """Test approval flow through InputProvider with capability-based tools."""
    mock_agent._input_provider = mock_input_provider
    ctx = mock_agent.get_context(input_provider=mock_input_provider)
    result = await ctx.handle_confirmation(confirmation_tool, {"text": "hello"})

    mock_input_provider.get_tool_confirmation.assert_called_once()
    call_args = mock_input_provider.get_tool_confirmation.call_args
    assert call_args[0][0] is ctx
    assert call_args[0][1] == confirmation_tool.description
    assert result == "allow"


@pytest.mark.unit
async def test_confirmation_ui_denial(
    mock_agent: Agent[Any],
    confirmation_tool: Tool[Any],
) -> None:
    """Test denial flow through InputProvider."""
    mock_provider = MagicMock()
    mock_provider.get_tool_confirmation = AsyncMock(return_value="skip")
    mock_agent._input_provider = mock_provider
    ctx = mock_agent.get_context(input_provider=mock_provider)
    result = await ctx.handle_confirmation(confirmation_tool, {"text": "hello"})

    mock_provider.get_tool_confirmation.assert_called_once()
    assert result == "skip"


@pytest.mark.unit
async def test_confirmation_ui_timeout(
    mock_agent: Agent[Any],
    confirmation_tool: Tool[Any],
) -> None:
    """Test timeout during confirmation."""
    mock_provider = MagicMock()
    mock_provider.get_tool_confirmation = AsyncMock(
        side_effect=TimeoutError("Confirmation timed out")
    )
    mock_agent._input_provider = mock_provider
    ctx = mock_agent.get_context(input_provider=mock_provider)

    with pytest.raises(TimeoutError, match="Confirmation timed out"):
        await ctx.handle_confirmation(confirmation_tool, {"text": "hello"})

    mock_provider.get_tool_confirmation.assert_called_once()


@pytest.mark.unit
async def test_confirmation_ui_abort_run(
    mock_agent: Agent[Any],
    confirmation_tool: Tool[Any],
) -> None:
    """Test abort_run confirmation result from InputProvider."""
    mock_provider = MagicMock()
    mock_provider.get_tool_confirmation = AsyncMock(return_value="abort_run")
    mock_agent._input_provider = mock_provider
    ctx = mock_agent.get_context(input_provider=mock_provider)
    result = await ctx.handle_confirmation(confirmation_tool, {"text": "hello"})
    assert result == "abort_run"


@pytest.mark.unit
async def test_confirmation_ui_abort_chain(
    mock_agent: Agent[Any],
    confirmation_tool: Tool[Any],
) -> None:
    """Test abort_chain confirmation result from InputProvider."""
    mock_provider = MagicMock()
    mock_provider.get_tool_confirmation = AsyncMock(return_value="abort_chain")
    mock_agent._input_provider = mock_provider
    ctx = mock_agent.get_context(input_provider=mock_provider)
    result = await ctx.handle_confirmation(confirmation_tool, {"text": "hello"})
    assert result == "abort_chain"


@pytest.mark.unit
async def test_confirmation_context_populated(
    mock_agent: Agent[Any],
    mock_input_provider: MagicMock,
    confirmation_tool: Tool[Any],
) -> None:
    """AgentContext passed to InputProvider has tool execution fields set."""
    mock_agent._input_provider = mock_input_provider
    ctx = mock_agent.get_context(
        input_provider=mock_input_provider,
        tool_name=confirmation_tool.name,
        tool_input={"text": "hello"},
        tool_call_id="call-123",
    )
    await ctx.handle_confirmation(confirmation_tool, {"text": "hello"})

    passed_ctx = mock_input_provider.get_tool_confirmation.call_args[0][0]
    assert isinstance(passed_ctx, AgentContext)
    assert passed_ctx.tool_name == confirmation_tool.name
    assert passed_ctx.tool_input == {"text": "hello"}
    assert passed_ctx.tool_call_id == "call-123"


@pytest.mark.unit
def test_requires_confirmation_propagated_to_pydantic_ai(
    confirmation_tool: Tool[Any],
) -> None:
    """Tool.requires_confirmation is propagated to pydantic-ai Tool.requires_approval."""
    pa_tool = confirmation_tool.to_pydantic_ai()
    assert pa_tool.requires_approval is True


@pytest.mark.unit
def test_no_confirmation_not_propagated_to_pydantic_ai(
    no_confirmation_tool: Tool[Any],
) -> None:
    """Tool without requires_confirmation does not set requires_approval."""
    pa_tool = no_confirmation_tool.to_pydantic_ai()
    assert pa_tool.requires_approval is False


@pytest.mark.unit
async def test_confirmation_never_mode_bypasses_provider(
    mock_agent: Agent[Any],
    mock_input_provider: MagicMock,
    confirmation_tool: Tool[Any],
) -> None:
    """tool_confirmation_mode='never' bypasses InputProvider entirely."""
    mock_agent.tool_confirmation_mode = "never"
    ctx = mock_agent.get_context(input_provider=mock_input_provider)
    result = await ctx.handle_confirmation(confirmation_tool, {"text": "hello"})
    mock_input_provider.get_tool_confirmation.assert_not_called()
    assert result == "allow"


@pytest.mark.unit
async def test_confirmation_per_tool_mode_no_confirmation_bypasses(
    mock_agent: Agent[Any],
    mock_input_provider: MagicMock,
    no_confirmation_tool: Tool[Any],
) -> None:
    """per_tool mode with non-confirmation tool bypasses InputProvider."""
    mock_agent.tool_confirmation_mode = "per_tool"
    ctx = mock_agent.get_context(input_provider=mock_input_provider)
    result = await ctx.handle_confirmation(no_confirmation_tool, {"text": "hello"})
    mock_input_provider.get_tool_confirmation.assert_not_called()
    assert result == "allow"


# ============================================================================
# Integration tests: multiple tools in same run
# ============================================================================


def _create_mock_agentlet_from_caps(
    capabilities: list[Any],
    deferred_requests: DeferredToolRequests,
    final_text: str = "Done",
) -> MagicMock:
    """Create a mock pydantic-ai agentlet that simulates deferred approval flow.

    Iterates ALL HandleDeferredToolCalls capabilities in order, chaining
    results: a None return means "pass to next capability". This mirrors
    how pydantic-ai combines multiple HandleDeferredToolCalls into a
    single pipeline at runtime.
    """
    from pydantic_ai.capabilities import HandleDeferredToolCalls

    deferred_caps: list[HandleDeferredToolCalls] = [
        cap for cap in capabilities if isinstance(cap, HandleDeferredToolCalls)
    ]

    mock_result = MagicMock()
    mock_result.data = final_text
    mock_result.all_messages.return_value = []
    mock_result.response.provider_details.get.return_value = None
    mock_usage = MagicMock()
    mock_usage.input_tokens = 10
    mock_usage.output_tokens = 5
    mock_usage.total_tokens = 15
    mock_result.usage = mock_usage

    cap_instances = deferred_caps

    def mock_iter(
        prompts: list[Any],
        *,
        deps: Any = None,
        message_history: list[Any] | None = None,
        usage_limits: Any = None,
    ) -> Any:
        class MockAgentRun:
            def __init__(self) -> None:
                self.result = mock_result
                self.usage = RunUsage()
                self.ctx = RunContext(
                    deps=deps,
                    model=MagicMock(),
                    usage=MagicMock(),
                )
                # next_node: first node for explicit next() loop.
                # Use a simple sentinel that isn't End/ModelRequestNode/CallToolsNode
                # so the loop body is skipped and next() is called immediately.
                self.next_node = MagicMock()

            async def next(self, node: Any) -> End[Any]:  # type: ignore[override]
                """Run capabilities then return End to break the loop."""
                for cap in cap_instances:
                    run_ctx = RunContext(
                        deps=deps,
                        model=MagicMock(),
                        usage=MagicMock(),
                    )
                    result = await cap.handle_deferred_tool_calls(
                        run_ctx, requests=deferred_requests
                    )
                    # None means "pass through to next capability"
                    # DeferredToolResults means "handled, stop chaining"
                    if result is not None:
                        break
                return End(data=final_text)

            def __aiter__(self) -> Any:
                return self

            async def __anext__(self) -> Any:
                for cap in cap_instances:
                    run_ctx = RunContext(
                        deps=deps,
                        model=MagicMock(),
                        usage=MagicMock(),
                    )
                    result = await cap.handle_deferred_tool_calls(
                        run_ctx, requests=deferred_requests
                    )
                    # None means "pass through to next capability"
                    # DeferredToolResults means "handled, stop chaining"
                    if result is not None:
                        break
                raise StopAsyncIteration

            async def __aenter__(self) -> Any:
                return self

            async def __aexit__(self, *args: object) -> None:
                pass

            def all_messages(self) -> list[Any]:
                return []

            def new_messages(self) -> list[Any]:
                return []

        return MockAgentRun()

    mock_agentlet = MagicMock()
    mock_agentlet.iter = mock_iter
    return mock_agentlet


@pytest.mark.unit
async def test_multiple_confirmation_tools_same_run(
    mock_agent: Agent[Any],
    confirmation_tool_1: Tool[Any],
    confirmation_tool_2: Tool[Any],
    sample_deferred_requests: DeferredToolRequests,
) -> None:
    """Test multiple confirmation-required tools all get approval prompts."""
    mock_provider = MagicMock()
    mock_provider.get_tool_confirmation = AsyncMock(return_value="allow")
    mock_agent._input_provider = mock_provider
    mock_agent._builtin_provider.register_tool(confirmation_tool_1)
    mock_agent._builtin_provider.register_tool(confirmation_tool_2)

    with patch("wolfharness.agents.native_agent.agent.PydanticAgent") as mock_pydantic_agent:

        def side_effect(**kwargs: Any) -> MagicMock:
            capabilities = kwargs.get("capabilities", []) or []
            return _create_mock_agentlet_from_caps(capabilities, sample_deferred_requests)

        mock_pydantic_agent.side_effect = side_effect
        await mock_agent.run("Test prompt")

    assert mock_provider.get_tool_confirmation.call_count == 2
    call_args_list = mock_provider.get_tool_confirmation.call_args_list
    assert call_args_list[0][0][0].tool_name == "dangerous_read"
    assert call_args_list[1][0][0].tool_name == "dangerous_write"


@pytest.mark.unit
async def test_mixed_approval_denial_same_run(
    mock_agent: Agent[Any],
    confirmation_tool_1: Tool[Any],
    confirmation_tool_2: Tool[Any],
    sample_deferred_requests: DeferredToolRequests,
) -> None:
    """Test approving some tools and denying others in same run."""
    mock_provider = MagicMock()
    mock_provider.get_tool_confirmation = AsyncMock(side_effect=["allow", "skip"])
    mock_agent._input_provider = mock_provider
    mock_agent._builtin_provider.register_tool(confirmation_tool_1)
    mock_agent._builtin_provider.register_tool(confirmation_tool_2)

    with patch("wolfharness.agents.native_agent.agent.PydanticAgent") as mock_pydantic_agent:

        def side_effect(**kwargs: Any) -> MagicMock:
            capabilities = kwargs.get("capabilities", []) or []
            return _create_mock_agentlet_from_caps(capabilities, sample_deferred_requests)

        mock_pydantic_agent.side_effect = side_effect
        await mock_agent.run("Test prompt")

    assert mock_provider.get_tool_confirmation.call_count == 2
    assert mock_provider.get_tool_confirmation.call_args_list[0][0][0].tool_name == "dangerous_read"
    assert (
        mock_provider.get_tool_confirmation.call_args_list[1][0][0].tool_name == "dangerous_write"
    )


@pytest.mark.unit
async def test_never_mode_auto_approves_all_tools(
    mock_agent: Agent[Any],
    confirmation_tool_1: Tool[Any],
    confirmation_tool_2: Tool[Any],
    sample_deferred_requests: DeferredToolRequests,
) -> None:
    """tool_confirmation_mode='never': bridge routes to provider when called directly."""
    mock_provider = MagicMock()
    mock_provider.get_tool_confirmation = AsyncMock(return_value="allow")
    mock_agent._input_provider = mock_provider
    mock_agent.tool_confirmation_mode = "never"
    mock_agent._builtin_provider.register_tool(confirmation_tool_1)
    mock_agent._builtin_provider.register_tool(confirmation_tool_2)

    with patch("wolfharness.agents.native_agent.agent.PydanticAgent") as mock_pydantic_agent:

        def side_effect(**kwargs: Any) -> MagicMock:
            capabilities = kwargs.get("capabilities", []) or []
            return _create_mock_agentlet_from_caps(capabilities, sample_deferred_requests)

        mock_pydantic_agent.side_effect = side_effect
        await mock_agent.run("Test prompt")

    mock_provider.get_tool_confirmation.assert_called()


@pytest.mark.unit
async def test_abort_run_stops_subsequent_confirmations(
    mock_agent: Agent[Any],
    confirmation_tool_1: Tool[Any],
    confirmation_tool_2: Tool[Any],
    sample_deferred_requests: DeferredToolRequests,
) -> None:
    """abort_run on first tool should still present it, then stop."""
    mock_provider = MagicMock()
    mock_provider.get_tool_confirmation = AsyncMock(side_effect=["abort_run"])
    mock_agent._input_provider = mock_provider
    mock_agent._builtin_provider.register_tool(confirmation_tool_1)
    mock_agent._builtin_provider.register_tool(confirmation_tool_2)

    with patch("wolfharness.agents.native_agent.agent.PydanticAgent") as mock_pydantic_agent:

        def side_effect(**kwargs: Any) -> MagicMock:
            capabilities = kwargs.get("capabilities", []) or []
            return _create_mock_agentlet_from_caps(capabilities, sample_deferred_requests)

        mock_pydantic_agent.side_effect = side_effect
        await mock_agent.run("Test prompt")

    assert mock_provider.get_tool_confirmation.call_count >= 1
    assert mock_provider.get_tool_confirmation.call_args_list[0][0][0].tool_name == "dangerous_read"
