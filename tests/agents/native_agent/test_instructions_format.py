"""Test pydantic-ai compatible instruction format conversion.

Tests that AgentPool instruction functions can be passed directly to
PydanticAgent(instructions=[...]) and that SystemPrompts correctly
converts to pydantic-ai format.
"""

from __future__ import annotations

from typing import Any

from pydantic_ai import Agent as PydanticAgent, RunContext
import pytest

from wolfharness.agents.context import AgentContext
from wolfharness.agents.native_agent import Agent
from wolfharness.agents.sys_prompts import SystemPrompts
from wolfharness.capabilities.function_toolset import FunctionToolsetCapability
from wolfharness.prompts.instructions import (
    InstructionFunc,
    PydanticAIInstruction,
)
from wolfharness.utils.context_wrapping import wrap_instruction


pytestmark = pytest.mark.unit


class TestPydanticAIInstructionType:
    """Test PydanticAIInstruction protocol and type compatibility."""

    def test_pydantic_ai_instruction_isinstance(self):
        """PydanticAIInstruction should support isinstance checks."""

        def with_pydantic_ai_ctx(ctx: RunContext[AgentContext[Any]]) -> str:
            return f"Model: {ctx.deps.model_name}"

        assert isinstance(with_pydantic_ai_ctx, PydanticAIInstruction)

    def test_pydantic_ai_instruction_in_union(self):
        """PydanticAIInstruction should be assignable to InstructionFunc."""

        def with_pydantic_ai_ctx(ctx: RunContext[AgentContext[Any]]) -> str:
            return "test"

        func: InstructionFunc = with_pydantic_ai_ctx
        assert callable(func)

    def test_async_pydantic_ai_instruction_isinstance(self):
        """Async PydanticAIInstruction should support isinstance checks."""

        async def async_with_ctx(ctx: RunContext[AgentContext[Any]]) -> str:
            return "test"

        assert isinstance(async_with_ctx, PydanticAIInstruction)


class TestWrapInstructionWithPydanticAISignature:
    """Test wrap_instruction with RunContext[AgentContext[Any]] signatures."""

    async def test_wrap_instruction_passes_through_pydantic_ai_signature(self):
        """Functions already accepting RunContext[AgentContext] pass through."""

        def pydantic_ai_instruction(ctx: RunContext[AgentContext[Any]]) -> str:
            return f"Model: {ctx.deps.model_name}"

        wrapped = wrap_instruction(pydantic_ai_instruction)

        # Create a mock RunContext with AgentContext as deps
        mock_agent_ctx = AgentContext(
            node=None,  # type: ignore[arg-type]
            pool=None,
            input_provider=None,
            data=None,
            model_name="openai:gpt-4o-mini",
        )
        mock_run_ctx = RunContext(
            deps=mock_agent_ctx,
            model=None,  # type: ignore[arg-type]
            usage=None,  # type: ignore[arg-type]
            prompt=None,  # type: ignore[arg-type]
            retry=0,
            messages=[],
        )

        result = await wrapped(mock_run_ctx)
        assert result == "Model: openai:gpt-4o-mini"

    async def test_wrap_instruction_wraps_agent_context_function(self):
        """Old AgentContext-only functions are wrapped correctly."""

        def agent_context_instruction(ctx: AgentContext[Any]) -> str:
            return f"Model: {ctx.model_name}"

        wrapped = wrap_instruction(agent_context_instruction)

        mock_agent_ctx = AgentContext(
            node=None,  # type: ignore[arg-type]
            pool=None,
            input_provider=None,
            data=None,
            model_name="openai:gpt-4o-mini",
        )
        mock_run_ctx = RunContext(
            deps=mock_agent_ctx,
            model=None,  # type: ignore[arg-type]
            usage=None,  # type: ignore[arg-type]
            prompt=None,  # type: ignore[arg-type]
            retry=0,
            messages=[],
        )

        result = await wrapped(mock_run_ctx)
        assert result == "Model: openai:gpt-4o-mini"

    async def test_wrap_instruction_wraps_simple_function(self):
        """Simple no-arg functions are wrapped correctly."""

        def simple_instruction() -> str:
            return "Be helpful"

        wrapped = wrap_instruction(simple_instruction)

        mock_agent_ctx = AgentContext(
            node=None,  # type: ignore[arg-type]
            pool=None,
            input_provider=None,
            data=None,
        )
        mock_run_ctx = RunContext(
            deps=mock_agent_ctx,
            model=None,  # type: ignore[arg-type]
            usage=None,  # type: ignore[arg-type]
            prompt=None,  # type: ignore[arg-type]
            retry=0,
            messages=[],
        )

        result = await wrapped(mock_run_ctx)
        assert result == "Be helpful"


@pytest.mark.real_model
class TestSystemPromptsPydanticAIConversion:
    """Test SystemPrompts.to_pydantic_ai_instructions()."""

    async def test_static_string_passes_through(self):
        """Static string prompts are returned as string instructions."""
        sys_prompts = SystemPrompts("You are a helpful assistant.")

        # Create a minimal agent for formatting
        agent = Agent(name="test", model="openai:gpt-4o-mini")
        async with agent:
            instructions = await sys_prompts.to_pydantic_ai_instructions(agent)

        assert len(instructions) >= 1
        assert isinstance(instructions[0], str)
        assert "You are a helpful assistant." in instructions[0]

    async def test_callable_prompt_wrapped(self):
        """No-arg callable prompts are rendered into static system prompt."""

        def dynamic_prompt() -> str:
            return "Dynamic instruction"

        sys_prompts = SystemPrompts(dynamic_prompt)

        agent = Agent(name="test", model="openai:gpt-4o-mini")
        async with agent:
            instructions = await sys_prompts.to_pydantic_ai_instructions(agent)

        # No-arg callable is rendered into the formatted system prompt
        assert len(instructions) >= 1
        assert isinstance(instructions[0], str)
        assert "Dynamic instruction" in instructions[0]

    async def test_callable_with_args_prompt_wrapped(self):
        """Callable prompts with arguments are wrapped as dynamic instructions."""

        def dynamic_prompt(ctx: AgentContext[Any]) -> str:
            return f"Dynamic: {ctx.model_name}"

        sys_prompts = SystemPrompts(dynamic_prompt)

        agent = Agent(name="test", model="openai:gpt-4o-mini")
        async with agent:
            instructions = await sys_prompts.to_pydantic_ai_instructions(agent)

        # Should have formatted system prompt (without the callable) + wrapped callable
        assert len(instructions) >= 2
        # First is the formatted string (without the callable)
        assert isinstance(instructions[0], str)
        # Second is the wrapped callable
        assert callable(instructions[1])

    async def test_pydantic_ai_compatible_function_passes_through(self):
        """RunContext[AgentContext] functions are wrapped and callable."""

        def pydantic_ai_prompt(ctx: RunContext[AgentContext[Any]]) -> str:
            return f"Using model: {ctx.deps.model_name}"

        sys_prompts = SystemPrompts(pydantic_ai_prompt)

        agent = Agent(name="test", model="openai:gpt-4o-mini")
        async with agent:
            instructions = await sys_prompts.to_pydantic_ai_instructions(agent)

        # The callable should be wrapped and executable
        assert len(instructions) >= 2
        wrapped = instructions[1]
        assert callable(wrapped)

        # Test that it can be called with a RunContext
        mock_agent_ctx = AgentContext(
            node=agent,  # type: ignore[arg-type]
            pool=None,
            input_provider=None,
            data=None,
            model_name="openai:gpt-4o-mini",
        )
        mock_run_ctx = RunContext(
            deps=mock_agent_ctx,
            model=None,  # type: ignore[arg-type]
            usage=None,  # type: ignore[arg-type]
            prompt=None,  # type: ignore[arg-type]
            retry=0,
            messages=[],
        )

        result = await wrapped(mock_run_ctx)  # type: ignore[operator]
        assert result == "Using model: openai:gpt-4o-mini"


class PydanticAIInstructionProvider(FunctionToolsetCapability):
    """Provider that returns pydantic-ai compatible instructions."""

    def __init__(self) -> None:
        super().__init__(
            name="pydantic_ai_provider",
            instructions="PydanticAI instruction provider",
        )


@pytest.mark.real_model
class TestNativeAgentPydanticAIInstructions:
    """Test NativeAgent integration with pydantic-ai compatible instructions."""

    async def test_agentlet_accepts_pydantic_ai_instruction_functions(self):
        """Test that get_agentlet works with pydantic-ai signature instructions."""
        provider = PydanticAIInstructionProvider()

        agent = Agent(
            name="test_agent",
            model="openai:gpt-4o-mini",
            system_prompt="You are an AI assistant.",
        )
        agent._add_capability(provider)

        async with agent:
            agentlet: PydanticAgent[Any, str] = await agent.get_agentlet(None, None, None)

            assert isinstance(agentlet, PydanticAgent)
            # Should have system prompt + provider instruction
            assert len(agentlet._instructions) >= 2  # type: ignore[arg-type]

    async def test_pydantic_ai_instruction_executed_at_runtime(self):
        """Test that pydantic-ai instruction functions are evaluated at runtime."""
        call_count = 0

        def counting_instruction(ctx: RunContext[AgentContext[Any]]) -> str:
            nonlocal call_count
            call_count += 1
            return f"Call count: {call_count}"

        provider = PydanticAIInstructionProvider()

        agent = Agent(
            name="test_agent",
            model="openai:gpt-4o-mini",
            system_prompt="You are an AI assistant.",
        )
        agent._add_capability(provider)

        async with agent:
            agentlet = await agent.get_agentlet(None, None, None)

            # Instructions should be present but not yet executed
            assert len(agentlet._instructions) >= 2  # type: ignore[arg-type]

    async def test_mixed_instruction_signatures_work(self):
        """Test that old and new instruction signatures work together."""

        class MixedProvider(FunctionToolsetCapability):
            def __init__(self) -> None:
                super().__init__(name="mixed_provider", instructions="Mixed instruction provider")

        agent = Agent(
            name="mixed_agent",
            model="openai:gpt-4o-mini",
            system_prompt="You are an AI assistant.",
        )
        agent._add_capability(MixedProvider())

        async with agent:
            agentlet = await agent.get_agentlet(None, None, None)

            assert isinstance(agentlet, PydanticAgent)
            # System prompt + provider instruction
            assert len(agentlet._instructions) >= 2  # type: ignore[arg-type]
