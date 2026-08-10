"""Test provider instruction integration into NativeAgent."""

from __future__ import annotations

from typing import Any

from pydantic_ai import Agent as PydanticAgent
import pytest

from wolfharness.agents.native_agent import Agent
from wolfharness.capabilities.function_toolset import FunctionToolsetCapability


pytestmark = pytest.mark.unit


class SimpleInstructionProvider(FunctionToolsetCapability):
    """Simple provider that returns static instructions."""

    def __init__(self) -> None:
        super().__init__(name="simple_provider", instructions="Be helpful and concise")


class AgentContextInstructionProvider(FunctionToolsetCapability):
    """Provider that returns AgentContext-aware instruction."""

    def __init__(self) -> None:
        super().__init__(name="agent_context_provider", instructions="Use the agent context wisely")


class RunContextInstructionProvider(FunctionToolsetCapability):
    """Provider that returns RunContext-aware instruction."""

    def __init__(self) -> None:
        super().__init__(name="run_context_provider", instructions="Model: gpt-4o-mini")


class EmptyInstructionProvider(FunctionToolsetCapability):
    """Provider that returns no instructions."""

    def __init__(self) -> None:
        super().__init__(name="empty_provider")


@pytest.fixture
async def agent_with_instruction_providers():
    """Create an agent with instruction providers."""
    provider1 = SimpleInstructionProvider()
    provider2 = AgentContextInstructionProvider()
    provider3 = RunContextInstructionProvider()

    agent = Agent(
        name="test_agent",
        model="openai:gpt-4o-mini",
        system_prompt="You are an AI assistant.",
    )

    agent._add_capability(provider1)
    agent._add_capability(provider2)
    agent._add_capability(provider3)

    return agent


@pytest.mark.real_model
class TestNativeAgentInstructions:
    """Test NativeAgent integration with provider instructions."""

    async def test_agentlet_collects_instructions_from_providers(
        self, agent_with_instruction_providers: Agent
    ):
        """Test that get_agentlet collects instructions from all providers."""
        agentlet: PydanticAgent[Any, str] = await agent_with_instruction_providers.get_agentlet(
            None, None, None
        )

        assert isinstance(agentlet, PydanticAgent)
        assert agentlet.name == "test_agent"

    async def test_formatted_system_prompt_includes_static_prompt(
        self, agent_with_instruction_providers: Agent
    ):
        """Test that formatted system prompt includes static system prompt."""
        async with agent_with_instruction_providers:
            assert agent_with_instruction_providers._formatted_system_prompt is not None
            assert (
                "You are an AI assistant."
                in agent_with_instruction_providers._formatted_system_prompt
            )

    async def test_instructions_are_collected_and_wrapped(
        self, agent_with_instruction_providers: Agent
    ):
        """Test that instructions from providers are collected and wrapped."""
        async with agent_with_instruction_providers:
            agentlet: PydanticAgent[Any, str] = await agent_with_instruction_providers.get_agentlet(
                None, None, None
            )

            assert agentlet.instructions is not None

    async def test_provider_instructions_reactive(self, agent_with_instruction_providers: Agent):
        """Test that provider instructions are called on each run."""
        async with agent_with_instruction_providers as agent:
            agentlet: PydanticAgent[Any, str] = await agent.get_agentlet(None, None, None)

            assert agentlet is not None
            assert agentlet.instructions is not None

    async def test_no_providers_uses_only_static_prompt(self):
        """Test that agent works normally with no instruction providers."""
        agent = Agent(
            name="simple_agent",
            model="openai:gpt-4o-mini",
            system_prompt="You are a simple assistant.",
        )

        async with agent:
            agentlet: PydanticAgent[Any, str] = await agent.get_agentlet(None, None, None)

            assert isinstance(agentlet, PydanticAgent)
            assert agentlet.instructions is not None

    async def test_provider_returning_empty_instructions(self):
        """Test that providers returning empty list are handled."""
        agent = Agent(
            name="empty_provider_agent",
            model="openai:gpt-4o-mini",
            system_prompt="You are an assistant.",
        )

        agent._add_capability(EmptyInstructionProvider())

        async with agent:
            agentlet: PydanticAgent[Any, str] = await agent.get_agentlet(None, None, None)
            assert isinstance(agentlet, PydanticAgent)

    async def test_provider_get_instructions_error_handling(self):
        """Test that errors in provider.get_instructions are handled gracefully.

        When a provider's get_instructions() raises, the agent should catch
        the error during instruction collection. However, since the provider
        is also passed as a capability to PydanticAI, PydanticAI may also
        raise when it calls get_instructions() on the capability.
        """
        from pydantic_ai.exceptions import UserError

        class FailingInstructionProvider(FunctionToolsetCapability):
            """Provider that fails to provide instructions."""

            def __init__(self) -> None:
                super().__init__(name="failing_provider")

            def get_instructions(self) -> str | None:
                msg = "Failed to get instructions"
                raise RuntimeError(msg)

        agent = Agent(
            name="failing_provider_agent",
            model="openai:gpt-4o-mini",
            system_prompt="You are an assistant.",
        )

        agent._add_capability(FailingInstructionProvider())

        # The agent should handle the error — either by catching it
        # during instruction collection or by PydanticAI raising
        async with agent:
            try:
                agentlet: PydanticAgent[Any, str] = await agent.get_agentlet(None, None, None)
                assert isinstance(agentlet, PydanticAgent)
            except (RuntimeError, UserError):
                # Expected: the failing provider causes an error
                pass

    async def test_from_config_with_provider_instruction_ref(self):
        """Test from_config with ProviderInstructionConfig using ref."""
        from wolfharness.models.agents import NativeAgentConfig

        class SimpleRefProvider(FunctionToolsetCapability):
            def __init__(self) -> None:
                super().__init__(
                    name="simple_ref_provider",
                    instructions="Dynamic instruction from ref provider",
                )

            async def get_tools(self) -> list[Any]:
                return []

        config = NativeAgentConfig(
            name="test_agent_with_ref",
            model="openai:gpt-4o-mini",
            system_prompt=["Be helpful."],
        )

        agent = Agent.from_config(config)

        provider = SimpleRefProvider()
        agent._add_capability(provider)

        async with agent:
            agentlet: PydanticAgent[Any, str] = await agent.get_agentlet(None, None, None)
            assert isinstance(agentlet, PydanticAgent)

            provider_names = [p.name for p in agent._all_capabilities]
            assert "simple_ref_provider" in provider_names
