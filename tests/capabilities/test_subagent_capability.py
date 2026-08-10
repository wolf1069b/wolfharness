"""Tests for SubagentCapability — native capability for subagent delegation."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.toolsets import FunctionToolset
import pytest

from wolfharness.capabilities.agent_context import AgentContextDeps
from wolfharness.capabilities.delegation import AgentNotFoundError, DelegationService
from wolfharness.capabilities.subagent_capability import SubagentCapability
from wolfharness.host.context import RunScope


pytestmark = pytest.mark.unit


# =============================================================================
# Test fixtures
# =============================================================================


class FakeDelegationService:
    """Minimal DelegationService implementation for testing."""

    def __init__(
        self,
        *,
        agents: list[str] | None = None,
        spawn_output: list[str] | None = None,
    ) -> None:
        self._agents = agents if agents is not None else ["analyzer", "reviewer"]
        self._spawn_output = spawn_output if spawn_output is not None else ["chunk_1", "chunk_2"]
        self.spawn_calls: list[tuple[str, str]] = []

    async def create_child_session(
        self,
        agent_name: str,
        *,
        parent_session_id: str | None = None,
        description: str = "",
        **metadata: Any,
    ) -> str:
        return "child_session_001"

    def spawn_subagent(self, name: str, prompt: str) -> Any:
        """Yield test chunks, recording the call."""
        self.spawn_calls.append((name, prompt))

        if name not in self._agents:
            raise AgentNotFoundError(name)

        async def _gen() -> Any:
            for chunk in self._spawn_output:
                yield chunk

        return _gen()

    def get_available_agents(self) -> list[str]:
        return list(self._agents)


def _make_ctx(delegation: DelegationService) -> Any:
    """Create a RunContext-like object with AgentContextDeps as deps.

    Sets ``host.session_pool = None`` so the fallback delegation path
    is exercised (matching pre-migration behavior). The
    ``agent_registry.list_names`` is wired to the delegation service's
    ``get_available_agents()`` so ``get_available_agents`` tests work.
    """
    host = MagicMock()
    host.session_pool = None
    agent_registry = MagicMock()
    agent_registry.list_names = MagicMock(return_value=delegation.get_available_agents())
    ctx = MagicMock()
    ctx.deps = AgentContextDeps(
        agent_registry=agent_registry,
        delegation=delegation,
        session=MagicMock(),
        scope=RunScope(),
        host=host,
    )
    return ctx


# =============================================================================
# Tests
# =============================================================================


def test_is_abstract_capability() -> None:
    """SubagentCapability is an instance of AbstractCapability."""
    cap = SubagentCapability()
    assert isinstance(cap, AbstractCapability)


def test_get_toolset_returns_function_toolset() -> None:
    """get_toolset() returns a FunctionToolset with both tools."""
    cap = SubagentCapability()
    toolset = cap.get_toolset()
    assert toolset is not None
    assert isinstance(toolset, FunctionToolset)


def test_toolset_exposes_spawn_subagent() -> None:
    """Toolset contains the spawn_subagent tool."""
    cap = SubagentCapability()
    toolset = cap.get_toolset()
    assert isinstance(toolset, FunctionToolset)
    assert "spawn_subagent" in toolset.tools


def test_toolset_exposes_get_available_agents() -> None:
    """Toolset contains the get_available_agents tool."""
    cap = SubagentCapability()
    toolset = cap.get_toolset()
    assert isinstance(toolset, FunctionToolset)
    assert "get_available_agents" in toolset.tools


async def test_spawn_subagent_calls_delegation() -> None:
    """spawn_subagent tool delegates to DelegationService.spawn_subagent."""
    delegation = FakeDelegationService(spawn_output=["result_a", "result_b"])
    ctx = _make_ctx(delegation)

    result = await SubagentCapability.spawn_subagent(ctx, "analyzer", "do stuff")

    assert delegation.spawn_calls == [("analyzer", "do stuff")]
    assert "result_a" in result
    assert "result_b" in result


async def test_spawn_subagent_unknown_agent_raises() -> None:
    """AgentNotFoundError is propagated when agent name is unknown."""
    delegation = FakeDelegationService(agents=["analyzer"])
    ctx = _make_ctx(delegation)

    with pytest.raises(AgentNotFoundError, match="nonexistent"):
        await SubagentCapability.spawn_subagent(ctx, "nonexistent", "prompt")


async def test_get_available_agents_returns_registry_names() -> None:
    """get_available_agents tool returns the agent list from DelegationService."""
    delegation = FakeDelegationService(agents=["alpha", "beta", "gamma"])
    ctx = _make_ctx(delegation)

    result = await SubagentCapability.get_available_agents(ctx)

    assert result == ["alpha", "beta", "gamma"]


async def test_get_available_agents_returns_empty_list() -> None:
    """get_available_agents returns empty list when no agents registered."""
    delegation = FakeDelegationService(agents=[])
    ctx = _make_ctx(delegation)

    result = await SubagentCapability.get_available_agents(ctx)

    assert result == []


def test_get_instructions_returns_description() -> None:
    """get_instructions() returns a non-None description."""
    cap = SubagentCapability()
    instructions = cap.get_instructions()
    assert instructions is not None
    assert "spawn_subagent" in instructions
    assert "get_available_agents" in instructions


def test_no_agent_pool_reference() -> None:
    """SubagentCapability does not import or reference AgentPool."""
    cap = SubagentCapability()
    # The capability should have no pool/AgentPool attribute
    assert not hasattr(cap, "pool")
    assert not hasattr(cap, "_pool")
    assert not hasattr(cap, "agent_pool")


async def test_aenter_returns_self() -> None:
    """__aenter__ returns the capability itself (no-op lifecycle)."""
    cap = SubagentCapability[Any]()
    result = await cap.__aenter__()
    assert result is cap


async def test_aexit_is_noop() -> None:
    """__aexit__ is a no-op that returns None."""
    cap = SubagentCapability[Any]()
    result = await cap.__aexit__(None, None, None)
    assert result is None


def test_toolset_id_customizable() -> None:
    """Toolset ID can be customized via constructor."""
    cap = SubagentCapability(toolset_id="custom_subagent")
    toolset = cap.get_toolset()
    assert isinstance(toolset, FunctionToolset)
    assert toolset.id == "custom_subagent"


def test_default_toolset_id() -> None:
    """Default toolset ID is 'subagent'."""
    cap = SubagentCapability()
    toolset = cap.get_toolset()
    assert isinstance(toolset, FunctionToolset)
    assert toolset.id == "subagent"


async def test_spawn_subagent_empty_output() -> None:
    """spawn_subagent returns empty string when delegation yields no chunks."""
    delegation = FakeDelegationService(spawn_output=[])
    ctx = _make_ctx(delegation)

    result = await SubagentCapability.spawn_subagent(ctx, "analyzer", "prompt")

    assert result == ""


async def test_delegation_service_protocol_isinstance() -> None:
    """FakeDelegationService satisfies the runtime_checkable DelegationService Protocol."""
    delegation = FakeDelegationService()
    assert isinstance(delegation, DelegationService)


def test_agent_not_found_error_message() -> None:
    """AgentNotFoundError stores and reports the agent name."""
    err = AgentNotFoundError("my_agent")
    assert err.agent_name == "my_agent"
    assert "my_agent" in str(err)


# =============================================================================
# Regression tests: _resolve_agent_context unwrapping
# =============================================================================


async def test_resolve_agent_context_from_runtime_context() -> None:
    """_resolve_agent_context unwraps AgentContextDeps from RuntimeAgentContext.data.

    In production, PydanticAI wraps our AgentContextDeps inside
    agents.context.AgentContext.data. The tool functions receive
    ctx.deps = agents.context.AgentContext, and our
    capabilities.agent_context.AgentContextDeps is at ctx.deps.data.
    """
    from wolfharness.agents.context import AgentContext as RuntimeAgentContext

    delegation = FakeDelegationService(agents=["alpha", "beta"])
    host = MagicMock()
    host.session_pool = None
    agent_registry = MagicMock()
    agent_registry.list_names = MagicMock(return_value=delegation.get_available_agents())

    cap_ctx = AgentContextDeps(
        agent_registry=agent_registry,
        delegation=delegation,
        session=MagicMock(),
        scope=RunScope(),
        host=host,
    )
    runtime_ctx = RuntimeAgentContext(node=MagicMock())
    runtime_ctx.data = cap_ctx

    ctx = MagicMock()
    ctx.deps = runtime_ctx

    result = await SubagentCapability.get_available_agents(ctx)

    assert result == ["alpha", "beta"]


async def test_resolve_agent_context_none_deps() -> None:
    """_resolve_agent_context raises RuntimeError when deps is None."""
    ctx = MagicMock()
    ctx.deps = None

    with pytest.raises(
        RuntimeError, match=r"SubagentCapability requires AgentContextDeps as deps\. Got: None"
    ):
        await SubagentCapability.get_available_agents(ctx)


async def test_resolve_agent_context_runtime_ctx_none_data() -> None:
    """_resolve_agent_context raises RuntimeError when RuntimeAgentContext.data is None."""
    from wolfharness.agents.context import AgentContext as RuntimeAgentContext

    runtime_ctx = RuntimeAgentContext(node=MagicMock())
    runtime_ctx.data = None

    ctx = MagicMock()
    ctx.deps = runtime_ctx

    with pytest.raises(
        RuntimeError,
        match=r"SubagentCapability requires AgentContextDeps at deps\.data\. Got: None",
    ):
        await SubagentCapability.get_available_agents(ctx)


async def test_resolve_agent_context_neither_type() -> None:
    """_resolve_agent_context raises RuntimeError for unknown deps type."""
    ctx = MagicMock()
    ctx.deps = object()

    with pytest.raises(
        RuntimeError, match=r"SubagentCapability requires AgentContextDeps as deps\. Got: object"
    ):
        await SubagentCapability.get_available_agents(ctx)
