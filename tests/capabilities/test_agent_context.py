"""Tests for AgentContextDeps, RunScope, and DelegationService protocol.

Covers M3 Wave 1 task group 10: RunScope stub, AgentContextDeps dataclass,
and DelegationService Protocol definition.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock

import pytest


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

from wolfharness.capabilities.agent_context import AgentContextDeps
from wolfharness.capabilities.delegation import AgentNotFoundError, DelegationService
from wolfharness.host.context import HostContext, RunScope


pytestmark = pytest.mark.unit


# =============================================================================
# Test doubles
# =============================================================================


class _StubDelegationService:
    """Minimal implementation of DelegationService protocol for testing."""

    async def create_child_session(
        self,
        agent_name: str,
        *,
        parent_session_id: str | None = None,
        description: str = "",
        **metadata: Any,
    ) -> str:
        return "child_session_001"

    async def spawn_subagent(self, name: str, prompt: str) -> AsyncIterator[Any]:
        if name == "missing":
            raise AgentNotFoundError(name)
        yield f"result from {name}: {prompt}"

    def get_available_agents(self) -> list[str]:
        return ["coder", "reviewer"]


def _make_host_context() -> HostContext:
    """Build a HostContext with minimal stubs for testing."""
    manifest = MagicMock()
    storage = MagicMock()
    vfs_registry = MagicMock()
    connection_registry = MagicMock()
    mcp = MagicMock()
    skills_registry = MagicMock()
    prompt_manager = MagicMock()
    process_manager = MagicMock()
    file_ops = MagicMock()
    todos = MagicMock()

    return HostContext(
        manifest=manifest,
        storage=storage,
        vfs_registry=vfs_registry,
        connection_registry=connection_registry,
        mcp=mcp,
        skills_registry=skills_registry,
        prompt_manager=prompt_manager,
        process_manager=process_manager,
        file_ops=file_ops,
        todos=todos,
        session_pool=None,
        config_file_path=None,
    )


def _make_session_state() -> Any:
    """Build a SessionState-like object for testing.

    Uses a simple stand-in since SessionState from orchestrator requires
    many runtime dependencies. AgentContextDeps only needs the reference.
    """
    return MagicMock()


def _make_agent_registry() -> Any:
    """Build an AgentRegistry-like object for testing."""
    from wolfharness.host.registry import AgentRegistry

    return AgentRegistry()


def _make_agent_context() -> AgentContextDeps:
    """Build an AgentContextDeps with test doubles."""
    return AgentContextDeps(
        agent_registry=_make_agent_registry(),
        delegation=_StubDelegationService(),
        session=_make_session_state(),
        scope=RunScope(),
        host=_make_host_context(),
        extension_registry=None,
    )


# =============================================================================
# RunScope tests
# =============================================================================


def test_run_scope_defaults() -> None:
    """RunScope has correct default values."""
    scope = RunScope()
    assert scope.config_id == "default"
    assert scope.tenant_id == "default"
    assert scope.user_id == "anonymous"
    assert scope.session_id == ""


def test_run_scope_frozen() -> None:
    """RunScope is immutable."""
    scope = RunScope()
    with pytest.raises(FrozenInstanceError):
        scope.config_id = "other"  # type: ignore[misc]


def test_run_scope_custom_values() -> None:
    """RunScope accepts custom values."""
    scope = RunScope(
        config_id="my_config",
        tenant_id="my_tenant",
        user_id="user123",
        session_id="sess-abc",
    )
    assert scope.config_id == "my_config"
    assert scope.tenant_id == "my_tenant"
    assert scope.user_id == "user123"
    assert scope.session_id == "sess-abc"


# =============================================================================
# AgentContextDeps tests
# =============================================================================


def test_agent_context_frozen() -> None:
    """AgentContextDeps is immutable."""
    ctx = _make_agent_context()
    with pytest.raises(FrozenInstanceError):
        ctx.extension_registry = MagicMock()  # type: ignore[misc]


def test_agent_context_all_fields_accessible() -> None:
    """All six fields of AgentContextDeps are accessible."""
    ctx = _make_agent_context()
    assert ctx.agent_registry is not None
    assert ctx.delegation is not None
    assert ctx.session is not None
    assert ctx.scope is not None
    assert ctx.host is not None
    # extension_registry is separately tested for default


def test_agent_context_extension_registry_defaults_none() -> None:
    """AgentContextDeps.extension_registry defaults to None."""
    ctx = _make_agent_context()
    assert ctx.extension_registry is None


# =============================================================================
# DelegationService protocol tests
# =============================================================================


def test_delegation_protocol_isinstance() -> None:
    """DelegationService protocol isinstance checks work."""
    stub = _StubDelegationService()
    assert isinstance(stub, DelegationService)


@pytest.mark.asyncio
async def test_delegation_spawn_subagent_success() -> None:
    """spawn_subagent yields results for known agents."""
    stub = _StubDelegationService()
    results: list[Any] = [item async for item in stub.spawn_subagent("coder", "write code")]
    assert len(results) == 1
    assert "coder" in results[0]


@pytest.mark.asyncio
async def test_delegation_agent_not_found() -> None:
    """Unknown agent raises AgentNotFoundError."""
    stub = _StubDelegationService()

    with pytest.raises(AgentNotFoundError):
        async for _ in stub.spawn_subagent("missing", "prompt"):
            pass


def test_delegation_get_available_agents() -> None:
    """get_available_agents returns list of agent names."""
    stub = _StubDelegationService()
    agents = stub.get_available_agents()
    assert "coder" in agents
    assert "reviewer" in agents


def test_agent_not_found_error_message() -> None:
    """AgentNotFoundError includes agent name in message."""
    err = AgentNotFoundError("nonexistent")
    assert "nonexistent" in str(err)


# =============================================================================
# TeamModeConfig integration tests
# =============================================================================


@pytest.mark.unit
def test_agent_context_team_mode_config_defaults_none() -> None:
    """AgentContextDeps.team_mode_config defaults to None when not provided."""
    ctx = _make_agent_context()
    assert ctx.team_mode_config is None


@pytest.mark.unit
def test_agent_context_team_mode_config_accepts_instance() -> None:
    """AgentContextDeps accepts a TeamModeConfig instance."""
    from wolfharness_config.team_mode import TeamModeConfig

    config = TeamModeConfig(enabled=True, member_eligible=["translator"])
    ctx = AgentContextDeps(
        agent_registry=_make_agent_registry(),
        delegation=_StubDelegationService(),
        session=_make_session_state(),
        scope=RunScope(),
        host=_make_host_context(),
        extension_registry=None,
        team_mode_config=config,
    )
    assert ctx.team_mode_config is not None
    assert ctx.team_mode_config.enabled is True
    assert "translator" in ctx.team_mode_config.member_eligible


# =============================================================================
# resolve_agent_context_from_deps utility tests
# =============================================================================


def test_resolve_agent_context_from_deps_direct() -> None:
    """resolve_agent_context_from_deps returns deps when it is directly an AgentContextDeps."""
    from wolfharness.capabilities.agent_context import resolve_agent_context_from_deps

    agent_ctx = _make_agent_context()
    result = resolve_agent_context_from_deps(agent_ctx)
    assert result is agent_ctx


def test_resolve_agent_context_from_deps_runtime_context() -> None:
    """resolve_agent_context_from_deps unwraps .data from RuntimeAgentContext."""
    from wolfharness.agents.context import AgentContext as RuntimeAgentContext
    from wolfharness.capabilities.agent_context import resolve_agent_context_from_deps

    agent_ctx = _make_agent_context()
    runtime_ctx = RuntimeAgentContext(node=MagicMock())
    runtime_ctx.data = agent_ctx

    result = resolve_agent_context_from_deps(runtime_ctx)
    assert result is agent_ctx


def test_resolve_agent_context_from_deps_none() -> None:
    """resolve_agent_context_from_deps raises RuntimeError when deps is None."""
    from wolfharness.capabilities.agent_context import resolve_agent_context_from_deps

    with pytest.raises(
        RuntimeError, match=r"TestCap requires AgentContextDeps as deps\. Got: None"
    ):
        resolve_agent_context_from_deps(None, capability_name="TestCap")


def test_resolve_agent_context_from_deps_runtime_ctx_none_data() -> None:
    """resolve_agent_context_from_deps raises RuntimeError when .data is None."""
    from wolfharness.agents.context import AgentContext as RuntimeAgentContext
    from wolfharness.capabilities.agent_context import resolve_agent_context_from_deps

    runtime_ctx = RuntimeAgentContext(node=MagicMock())
    runtime_ctx.data = None

    with pytest.raises(
        RuntimeError, match=r"TestCap requires AgentContextDeps at deps\.data\. Got: None"
    ):
        resolve_agent_context_from_deps(runtime_ctx, capability_name="TestCap")


def test_resolve_agent_context_from_deps_neither_type() -> None:
    """resolve_agent_context_from_deps raises RuntimeError for unknown deps type."""
    from wolfharness.capabilities.agent_context import resolve_agent_context_from_deps

    with pytest.raises(RuntimeError, match=r"TestCap requires AgentContextDeps as deps\. Got: str"):
        resolve_agent_context_from_deps("not a context", capability_name="TestCap")
