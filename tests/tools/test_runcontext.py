from __future__ import annotations

from typing import Any

from pydantic_ai import RunContext
from pydantic_ai.models.test import TestModel
import pytest

from wolfharness import Agent, AgentContext
from wolfharness.delegation import AgentPool
from wolfharness.models.agents import NativeAgentConfig
from wolfharness.models.manifest import AgentsManifest
from wolfharness_config.toolsets import SubagentToolsetConfig


async def run_ctx_tool(ctx: RunContext) -> str:
    """Tool expecting RunContext."""
    assert isinstance(ctx, RunContext)
    return "RunContext tool"


async def agent_ctx_tool(ctx: AgentContext) -> str:
    """Tool expecting AgentContext."""
    assert isinstance(ctx, AgentContext)
    return "AgentContext tool"


async def data_with_run_ctx(ctx: RunContext[Any]) -> str:
    """Tool accessing data through RunContext."""
    return f"Data from RunContext: {ctx.deps.data}"


async def data_with_agent_ctx(ctx: AgentContext) -> str:
    """Tool accessing data through AgentContext."""
    # When a tool requests AgentContext, it gets RunContext.deps
    # RunContext.deps is AgentContext, and the user data is in AgentContext.data
    # But ctx here is AgentContext, so ctx.data contains the user data directly
    data_value = ctx.data.data if isinstance(ctx.data, AgentContext) else ctx.data
    return f"Data from AgentContext: {data_value}"


async def no_ctx_tool() -> str:
    """Tool without any context."""
    return "No context tool"


async def dual_ctx_tool(run_ctx: RunContext, agent_ctx: AgentContext) -> str:
    """Tool expecting both RunContext and AgentContext."""
    assert isinstance(run_ctx, RunContext)
    assert isinstance(agent_ctx, AgentContext)
    return f"Dual context tool (agent: {agent_ctx.node_name})"


async def test_tool_context_injection():
    """Test that tools receive correct context."""
    context_received = None
    deps_received = None

    async def test_tool(ctx: RunContext[Any]) -> str:
        """Test tool that captures its context."""
        nonlocal context_received, deps_received
        context_received = ctx
        deps_received = ctx.deps
        return "Called"

    async with Agent(model=TestModel(call_tools=["test_tool"]), deps_type=bool) as agent:
        # Register our test tool
        agent._builtin_provider.register_tool(test_tool, enabled=True)
        # Run agent which should trigger tool
        await agent.run("Test", deps=True)
        assert context_received is not None, "Tool did not receive context"
        assert isinstance(context_received, RunContext), "Wrong context type"

        # Verify dependencies
        assert deps_received is not None, "Tool did not receive dependencies"


async def test_plain_tool_no_context():
    """Test that plain tools work without context."""
    count = 0

    async def plain_tool() -> str:
        """Tool without context parameter."""
        nonlocal count
        count += 1
        return "Got arg"

    async with Agent(model=TestModel(call_tools=["plain_tool"])) as agent:
        agent._builtin_provider.register_tool(plain_tool, enabled=True)
        # Should work without error
        await agent.run("Test")
        assert count == 1


@pytest.mark.integration
@pytest.mark.xfail(
    reason="Test passes toolsets via Agent() constructor, but session_pool path "
    "recreates agent from manifest config (without toolsets). SubagentTools "
    "capability is lost. Fix: configure SubagentToolsetConfig in manifest.tools.",
    strict=False,
)
async def test_capability_tools(default_model: str):
    """Test that capability tools work with AgentContext via manifest config."""
    manifest = AgentsManifest(
        agents={
            "test": NativeAgentConfig(model=default_model),
            "test_2": NativeAgentConfig(model=default_model),
            "helper": NativeAgentConfig(model=default_model, system_prompt="You help with tasks"),
        }
    )
    async with AgentPool(manifest) as pool:
        subagent = SubagentToolsetConfig()
        providers = [subagent.get_provider()]
        agent = Agent(name="test", model=default_model, toolsets=providers, agent_pool=pool)
        prompt = "Execute task 'say hello' on agent with name `helper`."
        result = await agent.run(prompt)
        assert result.get_tool_calls()
        assert result.get_tool_calls()[0].tool_name == "task"


async def test_context_compatibility():
    """Test that both context types work in tools."""
    model = TestModel(call_tools=["run_ctx_tool", "agent_ctx_tool", "no_ctx_tool"])
    async with Agent(model=model) as agent:
        agent._builtin_provider.register_tool(run_ctx_tool)
        agent._builtin_provider.register_tool(agent_ctx_tool)
        agent._builtin_provider.register_tool(no_ctx_tool)

        # All should work
        result = await agent.run("Test")
        assert any(call.result == "RunContext tool" for call in result.get_tool_calls())
        assert any(call.result == "AgentContext tool" for call in result.get_tool_calls())
        assert any(call.result == "No context tool" for call in result.get_tool_calls())


async def test_context_sharing():
    """Test that both context types access same data."""
    shared_data = {"key": "value"}
    model = TestModel(call_tools=["data_with_run_ctx", "data_with_agent_ctx"])
    agent = Agent[dict[str, str]](name="test", model=model, deps_type=dict)
    agent._builtin_provider.register_tool(data_with_run_ctx)
    agent._builtin_provider.register_tool(data_with_agent_ctx)

    async with agent:
        result = await agent.run("Test", deps=shared_data)

        assert any(
            call.result == "Data from RunContext: {'key': 'value'}"
            for call in result.get_tool_calls()
        )
        assert any(
            call.result == "Data from AgentContext: {'key': 'value'}"
            for call in result.get_tool_calls()
        )


async def test_dual_context_tool():
    """Test tool that requires both RunContext and AgentContext."""
    async with Agent(model=TestModel(call_tools=["dual_ctx_tool"]), name="dual-agent") as agent:
        agent._builtin_provider.register_tool(dual_ctx_tool)
        # This should work if dual context injection is implemented
        result = await agent.run("Test")
        # Should successfully call the tool with both contexts
        tool_calls = result.get_tool_calls()
        assert len(tool_calls) > 0
        expected_result = "Dual context tool (agent: dual-agent)"
        assert any(call.result == expected_result for call in tool_calls)


if __name__ == "__main__":
    pytest.main([__file__, "-vv", "--log-level", "debug"])
