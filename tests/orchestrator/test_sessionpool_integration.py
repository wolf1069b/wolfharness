"""Integration tests for subagent session agent MCP behavior.

This file documents and enforces the MCP inheritance rules for child
sessions (subagents).  Each test has a docstring explaining the expected
behavior and the regression it prevents.

## MCP Inheritance Rules (as of 2026-06-30)

### Rule 1: Pool-level MCP servers are accessible to ALL agents
Pool-level MCP servers (defined in the top-level ``mcp_servers:`` YAML
section) are stored in ``pool.mcp``.  Every agent that does NOT have its
own ``mcp_servers`` gets ``self.mcp = pool.mcp`` (shared), so
``get_agentlet()`` calls ``pool.mcp.get_capabilities()`` and the subagent
sees pool-level tools.

### Rule 2: Agent-level MCP servers are NOT inherited by child sessions
When an agent defines its own ``mcp_servers:``, a dedicated
``MCPManager`` is created (``self._mcp_shared = False``).  Child
sessions created from this agent do NOT inherit this dedicated manager.
The child gets its own agent config, and if the child's config has no
``mcp_servers``, ``self.mcp = pool.mcp``.

### Rule 3: Pool-level ACP providers ARE added to child sessions
``agent.tools.add_provider(pool.mcp.get_aggregating_provider())`` is
called on BOTH main session and child session paths.  This ensures ACP
subagents can use pool-level MCP-over-ACP servers.

### Rule 4: Parent's external providers are NOT inherited
Providers added to ``parent_agent.tools`` at runtime (e.g. mock MCP
providers, static tool providers) are NOT forwarded to child session
agents.  Only pool-level providers (added by the orchestrator) and the
child's own config-level providers are present.

### Rule 5: Child session is_per_session_agent=False
``close_session()`` on a child session does NOT call
``agent.__aexit__()``.  The parent owns the lifecycle.

### Rule 6: No cross-task CancelScope sharing
Each ``get_capabilities()`` call creates a fresh ``MCPToolset`` (no
``_toolset_cache``).  This prevents the ``RuntimeError: Attempted to
exit cancel scope in a different task`` when a parent agent (task A)
and subagent (task B) share the same MCPManager.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from pydantic_ai.messages import PartStartEvent, ThinkingPart
from pydantic_ai.models.test import TestModel
import pytest

from wolfharness import Agent, AgentPool, AgentsManifest, NativeAgentConfig
from wolfharness.agents.context import AgentRunContext
from wolfharness.agents.events import SpawnSessionStart, StreamCompleteEvent
from wolfharness.capabilities.function_toolset import FunctionToolsetCapability
from wolfharness.hooks import AgentHooks, CallableHook, HookResult
from wolfharness.lifecycle.types import DeliveryMode
from wolfharness.messaging import ChatMessage
from wolfharness.orchestrator.core import EventBus, SessionState
from wolfharness.orchestrator.run import RunHandle
from wolfharness.tools.base import Tool
from wolfharness_server.opencode_server.models.parts import (
    ToolPart,
    ToolStateCompleted,
    ToolStateRunning,
)
from wolfharness_server.opencode_server.session_pool_integration import (
    OpenCodeSessionPoolIntegration,
)


if TYPE_CHECKING:
    from collections.abc import Sequence

    from wolfharness.skills.skill import Skill


class MockServerState:
    """Mock OpenCode ServerState for testing (consolidated)."""

    def __init__(self) -> None:
        self.messages: dict[str, list[Any]] = {}
        self.events: list[Any] = []
        self.working_dir = "/tmp"
        self.agent: Any = None
        self.pool: Any = None
        self.pool_or_none: Any = None
        self.session_controller: Any = None
        self.session_status: dict[str, Any] = {}
        self.sessions: dict[str, Any] = {}
        self.session_locks: dict[str, Any] = {}

    async def broadcast_event(self, event: Any) -> None:
        self.events.append(event)

    def resolve_default_model_info(self) -> tuple[str, str]:
        return "default", "wolfharness"


class MockMCPCapability(FunctionToolsetCapability):
    """Mock MCP provider for testing inheritance."""

    kind = "mcp"

    def __init__(
        self,
        name: str = "mock_mcp",
        skills: list[Skill] | None = None,
        tools: list[Tool] | None = None,
        prompts: list[Any] | None = None,
        resources: list[Any] | None = None,
    ) -> None:
        super().__init__(name=name)
        self._skills = skills or []
        self._tools = tools or []
        self._prompts = prompts or []
        self._resources = resources or []

    async def get_skills(self) -> list[Skill]:
        """Get mock skills."""
        return self._skills

    async def get_tools(self) -> Sequence[Tool]:
        """Get mock tools."""
        return self._tools

    async def get_prompts(self) -> list[Any]:
        """Get mock prompts."""
        return self._prompts

    async def get_resources(self) -> list[Any]:
        """Get mock resources."""
        return self._resources


def _mock_tool() -> str:
    """A mock tool for testing provider inheritance."""
    return "mock_result"


# ---------------------------------------------------------------------------
# Rule 1: Pool-level MCP servers are accessible to ALL agents
# ---------------------------------------------------------------------------


@pytest.mark.integration
async def test_pool_level_mcp_accessible_to_child_without_own_mcp_servers() -> None:
    """Rule 1: Child agent without own mcp_servers shares pool.mcp.

    When the child agent's config has no ``mcp_servers``, the child's
    ``self.mcp`` is set to ``pool.mcp`` during ``MessageNode.__init__``.
    This means ``get_agentlet()`` → ``self.mcp.get_capabilities()`` returns
    pool-level MCP capabilities.

    Regression: If someone adds ``agent.mcp = parent_agent.mcp`` back,
    the child would get the parent's dedicated MCPManager instead of
    pool.mcp, and pool-level tools would be lost when the parent has
    its own mcp_servers.
    """
    agent_config = NativeAgentConfig(
        name="test_agent",
        model="test",
        system_prompt="You are a test agent",
    )
    manifest = AgentsManifest(agents={"test_agent": agent_config})

    async with AgentPool(manifest) as pool:
        session_pool = pool.session_pool
        assert session_pool is not None
        await session_pool.start()

        parent_session_id = "parent-rule1-pool-access"
        child_session_id = "child-rule1-pool-access"

        await session_pool.create_session(parent_session_id, agent_name="test_agent")
        await session_pool.sessions.get_or_create_session_agent(parent_session_id)

        await session_pool.create_session(
            child_session_id,
            parent_session_id=parent_session_id,
            agent_name="test_agent",
        )
        child_agent = await session_pool.sessions.get_or_create_session_agent(child_session_id)

        # EXPECT: child_agent.mcp IS pool.mcp (shared manager)
        assert child_agent.mcp is pool.mcp, (
            "Child agent without own mcp_servers must share pool.mcp. "
            "If this fails, someone broke the messagenode.py shared MCP logic."
        )

        await session_pool.shutdown()


# ---------------------------------------------------------------------------
# Rule 2: Agent-level MCP servers are NOT inherited by child sessions
# ---------------------------------------------------------------------------


@pytest.mark.integration
async def test_agent_level_mcp_not_inherited_by_child() -> None:
    """Rule 2: Parent's agent-level MCPManager is NOT shared with children.

    When the parent agent has its own ``mcp_servers``, it gets a
    dedicated ``MCPManager`` (``self._mcp_shared = False``).  The child
    session agent does NOT inherit this dedicated manager.  Instead,
    the child gets its own config, and if the child has no
    ``mcp_servers``, ``self.mcp = pool.mcp``.

    Regression: The old code had ``agent.mcp = parent_agent.mcp`` which
    caused cross-task CancelScope errors when the parent and child ran
    in different asyncio tasks.
    """
    # Use StdioMCPServerConfig with a command that doesn't exist.
    # We test at the config level — we don't need to actually connect.
    # The key assertion is that the parent's MCPManager is dedicated
    # (not pool.mcp), and the child's IS pool.mcp.
    from wolfharness_config.mcp_server import StdioMCPServerConfig

    parent_config = NativeAgentConfig(
        name="parent_agent",
        model="test",
        system_prompt="Parent with own MCP servers",
        mcp_servers=[
            StdioMCPServerConfig(
                name="fake_agent_mcp",
                command="/nonexistent/fake-mcp-server",
                args=[],
            ),
        ],
    )
    child_config = NativeAgentConfig(
        name="child_agent",
        model="test",
        system_prompt="Child without own MCP servers",
    )
    manifest = AgentsManifest(agents={"parent_agent": parent_config, "child_agent": child_config})

    async with AgentPool(manifest) as pool:
        # Test at config level — create agents without entering them
        # (entering would try to connect the fake MCP server)
        parent_agent_obj = parent_config.get_agent(pool=pool)
        child_agent_obj = child_config.get_agent(pool=pool)

        # EXPECT: parent has its own dedicated MCPManager (NOT pool.mcp)
        assert parent_agent_obj.mcp is not pool.mcp, (
            "Parent with own mcp_servers must have a dedicated MCPManager, not pool.mcp."
        )
        assert parent_agent_obj._mcp_shared is False, (
            "Parent with own mcp_servers must have _mcp_shared=False."
        )

        # EXPECT: child uses pool.mcp (NOT parent's dedicated MCPManager)
        assert child_agent_obj.mcp is pool.mcp, (
            "Child must use pool.mcp, NOT parent's dedicated MCPManager. "
            "If this fails, the old agent.mcp = parent_agent.mcp sharing "
            "was reintroduced."
        )
        assert child_agent_obj.mcp is not parent_agent_obj.mcp, (
            "Child's MCPManager must NOT be the same object as parent's."
        )


# ---------------------------------------------------------------------------
# Rule 3: Pool-level ACP providers ARE added to child sessions
# ---------------------------------------------------------------------------


@pytest.mark.integration
async def test_pool_level_acp_provider_added_to_child_session() -> None:
    """Rule 3: Child session gets pool.mcp.get_aggregating_provider().

    The orchestrator adds ``pool.mcp.get_aggregating_provider()`` to
    BOTH main session and child session agents.  This ensures ACP
    subagents can access pool-level MCP-over-ACP servers.

    Regression: Commit 5609a125d removed this line from the child
    session path, causing subagents to lose access to pool-level ACP
    MCP servers.  The fix re-added it.
    """
    agent_config = NativeAgentConfig(
        name="test_agent",
        model="test",
        system_prompt="You are a test agent",
    )
    manifest = AgentsManifest(agents={"test_agent": agent_config})

    async with AgentPool(manifest) as pool:
        session_pool = pool.session_pool
        assert session_pool is not None
        await session_pool.start()

        parent_session_id = "parent-rule3-acp-provider"
        child_session_id = "child-rule3-acp-provider"

        await session_pool.create_session(parent_session_id, agent_name="test_agent")
        parent_agent = await session_pool.sessions.get_or_create_session_agent(parent_session_id)

        await session_pool.create_session(
            child_session_id,
            parent_session_id=parent_session_id,
            agent_name="test_agent",
        )
        child_agent = await session_pool.sessions.get_or_create_session_agent(child_session_id)

        # EXPECT: Both parent and child have the pool's aggregating provider
        pool_agg = pool.mcp.get_aggregating_provider()
        parent_has_agg = any(
            p is pool_agg or p.name == pool_agg.name for p in parent_agent._external_capabilities
        )
        child_has_agg = any(
            p is pool_agg or p.name == pool_agg.name for p in child_agent._external_capabilities
        )

        assert parent_has_agg, "Parent session must have pool.mcp.get_aggregating_provider()."
        assert child_has_agg, (
            "Child session must have pool.mcp.get_aggregating_provider(). "
            "If this fails, the orchestrator child session path is missing "
            "agent.tools.add_provider(self.pool.mcp.get_aggregating_provider())."
        )

        await session_pool.shutdown()


# ---------------------------------------------------------------------------
# Rule 4: Parent's external providers are NOT inherited
# ---------------------------------------------------------------------------


@pytest.mark.integration
async def test_parent_external_providers_not_inherited_by_child() -> None:
    """Rule 4: Runtime providers added to parent are NOT forwarded to child.

    When code adds a provider to ``parent_agent.tools`` at runtime
    (e.g. via ``parent_agent.tools.add_provider(...)``), the child
    session agent does NOT inherit it.  Only pool-level providers
    (added by the orchestrator) and the child's own config-level
    providers are present.

    Regression: The old code had a loop that copied parent's
    ``kind=='mcp'`` providers to the child.  This caused stale
    providers to leak across sessions.
    """
    agent_config = NativeAgentConfig(
        name="test_agent",
        model="test",
        system_prompt="You are a test agent",
    )
    manifest = AgentsManifest(agents={"test_agent": agent_config})

    async with AgentPool(manifest) as pool:
        session_pool = pool.session_pool
        assert session_pool is not None
        await session_pool.start()

        parent_session_id = "parent-rule4-no-inherit"
        child_session_id = "child-rule4-no-inherit"

        await session_pool.create_session(parent_session_id, agent_name="test_agent")
        parent_agent = await session_pool.sessions.get_or_create_session_agent(parent_session_id)

        # Add runtime providers to parent
        mock_tool = Tool.from_callable(_mock_tool, name_override="mock_tool")
        mcp_provider = MockMCPCapability(
            name="parent_mcp_provider",
            tools=[mock_tool],
        )
        static_provider = FunctionToolsetCapability(
            name="parent_static_provider",
            tools=[mock_tool],
        )
        parent_agent._add_capability(mcp_provider)
        parent_agent._add_capability(static_provider)

        await session_pool.create_session(
            child_session_id,
            parent_session_id=parent_session_id,
            agent_name="test_agent",
        )
        child_agent = await session_pool.sessions.get_or_create_session_agent(child_session_id)

        # EXPECT: Neither MCP nor non-MCP providers are inherited
        assert mcp_provider not in child_agent._external_capabilities, (
            "Child must NOT inherit parent's kind=='mcp' providers."
        )
        assert static_provider not in child_agent._external_capabilities, (
            "Child must NOT inherit parent's non-MCP providers."
        )

        await session_pool.shutdown()


# ---------------------------------------------------------------------------
# Rule 5: Child session is_per_session_agent=False
# ---------------------------------------------------------------------------


@pytest.mark.integration
async def test_child_session_is_not_per_session_agent() -> None:
    """Rule 5: Child session has is_per_session_agent=False.

    close_session() on a child session does NOT call agent.__aexit__().
    The parent owns the lifecycle.  This prevents premature MCPManager
    cleanup when a subagent finishes.
    """
    agent_config = NativeAgentConfig(
        name="test_agent",
        model="test",
        system_prompt="You are a test agent",
    )
    manifest = AgentsManifest(agents={"test_agent": agent_config})

    async with AgentPool(manifest) as pool:
        session_pool = pool.session_pool
        assert session_pool is not None
        await session_pool.start()

        parent_session_id = "parent-rule5-cleanup"
        child_session_id = "child-rule5-cleanup"

        await session_pool.create_session(parent_session_id, agent_name="test_agent")
        await session_pool.sessions.get_or_create_session_agent(parent_session_id)

        await session_pool.create_session(
            child_session_id,
            parent_session_id=parent_session_id,
            agent_name="test_agent",
        )
        await session_pool.sessions.get_or_create_session_agent(child_session_id)

        child_state = session_pool.sessions._sessions.get(child_session_id)
        assert child_state is not None
        assert child_state.is_per_session_agent is False, (
            "Child session must have is_per_session_agent=False so "
            "close_session() does not call agent.__aexit__()."
        )

        # Close child — parent should survive
        await session_pool.close_session(child_session_id)

        parent_state = session_pool.sessions._sessions.get(parent_session_id)
        assert parent_state is not None, "Parent session should still exist after child close."

        await session_pool.shutdown()


# ---------------------------------------------------------------------------
# MCPToolset caching by client_id
# ---------------------------------------------------------------------------


@pytest.mark.integration
async def test_toolset_cache_exists_on_mcp_manager() -> None:
    """MCPManager has a _toolset_cache attribute for connection reuse.

    ``get_capabilities()`` caches ``MCPToolset`` instances by ``client_id``
    so repeated calls reuse the same underlying connection. The ``MCP``
    wrapper instances remain distinct.
    """
    from wolfharness.mcp_server.manager import MCPManager

    manager = MCPManager(name="test")
    assert hasattr(manager, "_toolset_cache")

    await manager.cleanup()


@pytest.mark.integration
async def test_get_capabilities_caches_toolset_by_client_id() -> None:
    """get_capabilities() caches MCPToolset by client_id.

    Two calls to ``manager.get_capabilities()`` return MCP capabilities
    with the same underlying ``MCPToolset`` instance (cached by
    ``client_id``). The MCP wrappers themselves are distinct.
    """
    from wolfharness.mcp_server.manager import MCPManager
    from wolfharness_config.mcp_server import StdioMCPServerConfig

    manager = MCPManager(
        servers=[
            StdioMCPServerConfig(
                name="test_server",
                command="python",
                args=["server.py"],
            ),
        ],
    )

    caps1 = await manager.get_capabilities()
    caps2 = await manager.get_capabilities()

    assert len(caps1) == 1
    assert len(caps2) == 1

    assert caps1[0].id == caps2[0].id
    # MCP wrappers are distinct
    assert caps1[0] is not caps2[0]
    # Underlying MCPToolset is cached (shared)
    assert caps1[0].local is caps2[0].local

    await manager.cleanup()


# ---------------------------------------------------------------------------
# Rule 7: Skills providers ARE added to child sessions
# ---------------------------------------------------------------------------


@pytest.mark.integration
async def test_skills_providers_added_to_child_session() -> None:
    """Rule 7: Skills are available to child sessions via SkillManagerCap.

    After the unify-skill-loading change, load_skill/list_skills are owned
    by SkillManagerCap (registered at pool scope). _inject_pool_providers
    no longer injects skills_tools_provider. Skills are accessible via
    the ExtensionRegistry.
    """
    agent_config = NativeAgentConfig(
        name="test_agent",
        model="test",
        system_prompt="You are a test agent",
    )
    manifest = AgentsManifest(agents={"test_agent": agent_config})

    async with AgentPool(manifest) as pool:
        session_pool = pool.session_pool
        assert session_pool is not None
        await session_pool.start()

        parent_session_id = "parent-rule7-skills"
        child_session_id = "child-rule7-skills"

        await session_pool.create_session(parent_session_id, agent_name="test_agent")
        await session_pool.sessions.get_or_create_session_agent(parent_session_id)

        await session_pool.create_session(
            child_session_id,
            parent_session_id=parent_session_id,
            agent_name="test_agent",
        )

        # SkillManagerCap is registered at pool scope; agents access it via
        # ExtensionRegistry. Verify the pool has skill_capabilities.
        assert len(pool.skill_capabilities) > 0

        await session_pool.shutdown()


# ---------------------------------------------------------------------------
# Rule 8: env and _internal_fs ARE inherited from parent
# ---------------------------------------------------------------------------


@pytest.mark.integration
async def test_env_and_internal_fs_inherited_from_parent() -> None:
    """Rule 8: env and _internal_fs are inherited by child sessions.

    While MCP managers are NOT shared, ``agent.env`` and
    ``agent._internal_fs`` ARE inherited from the parent agent.  This
    ensures environment variables and filesystem access are consistent.
    """
    agent_config = NativeAgentConfig(
        name="test_agent",
        model="test",
        system_prompt="You are a test agent",
    )
    manifest = AgentsManifest(agents={"test_agent": agent_config})

    async with AgentPool(manifest) as pool:
        session_pool = pool.session_pool
        assert session_pool is not None
        await session_pool.start()

        parent_session_id = "parent-rule8-env"
        child_session_id = "child-rule8-env"

        await session_pool.create_session(parent_session_id, agent_name="test_agent")
        parent_agent = await session_pool.sessions.get_or_create_session_agent(parent_session_id)

        await session_pool.create_session(
            child_session_id,
            parent_session_id=parent_session_id,
            agent_name="test_agent",
        )
        child_agent = await session_pool.sessions.get_or_create_session_agent(child_session_id)

        # EXPECT: env and _internal_fs are inherited
        if parent_agent.env is not None:
            assert child_agent.env is parent_agent.env, (
                "Child must inherit parent's env (same object)."
            )
        assert child_agent._internal_fs is parent_agent._internal_fs, (
            "Child must inherit parent's _internal_fs (same object)."
        )

        await session_pool.shutdown()


# ---------------------------------------------------------------------------
# Rule 9: Child agent is a NEW object (not parent's agent)
# ---------------------------------------------------------------------------


@pytest.mark.integration
async def test_child_agent_is_new_object() -> None:
    """Rule 9: Child session agent is a distinct object from parent's.

    The child must NOT be the same Python object as the parent agent.
    This ensures chat history and agent state are isolated per session.
    """
    agent_config = NativeAgentConfig(
        name="test_agent",
        model="test",
        system_prompt="You are a test agent",
    )
    manifest = AgentsManifest(agents={"test_agent": agent_config})

    async with AgentPool(manifest) as pool:
        session_pool = pool.session_pool
        assert session_pool is not None
        await session_pool.start()

        parent_session_id = "parent-rule9-new-obj"
        child_session_id = "child-rule9-new-obj"

        await session_pool.create_session(parent_session_id, agent_name="test_agent")
        parent_agent = await session_pool.sessions.get_or_create_session_agent(parent_session_id)

        await session_pool.create_session(
            child_session_id,
            parent_session_id=parent_session_id,
            agent_name="test_agent",
        )
        child_agent = await session_pool.sessions.get_or_create_session_agent(child_session_id)

        assert child_agent is not parent_agent, (
            "Child must have its own per-session agent object, not share parent's."
        )

        await session_pool.shutdown()


# ---------------------------------------------------------------------------
# Rule 10: base_agent is NOT mutated by child session creation
# ---------------------------------------------------------------------------


@pytest.mark.integration
async def test_base_agent_not_mutated_by_child_creation() -> None:
    """Rule 10: Creating a child session does NOT mutate the base agent.

    The base agent (from ``manifest.agents[name].get_agent()``) must
    not accumulate providers from child session creation.  Each child
    gets its own per-session agent with only pool-level providers.
    """
    agent_config = NativeAgentConfig(
        name="test_agent",
        model="test",
        system_prompt="You are a test agent",
    )
    manifest = AgentsManifest(agents={"test_agent": agent_config})

    async with AgentPool(manifest) as pool:
        session_pool = pool.session_pool
        assert session_pool is not None
        await session_pool.start()

        parent_session_id = "parent-rule10-no-mutate"
        child_session_id = "child-rule10-no-mutate"

        base_agent = pool.manifest.agents["test_agent"].get_agent(pool=pool)
        base_provider_count = len(base_agent._external_capabilities)

        await session_pool.create_session(parent_session_id, agent_name="test_agent")
        await session_pool.sessions.get_or_create_session_agent(parent_session_id)

        await session_pool.create_session(
            child_session_id,
            parent_session_id=parent_session_id,
            agent_name="test_agent",
        )
        await session_pool.sessions.get_or_create_session_agent(child_session_id)

        # EXPECT: base_agent provider count unchanged
        assert len(base_agent._external_capabilities) == base_provider_count, (
            "base_agent must NOT be mutated by child session creation. "
            f"Before: {base_provider_count}, "
            f"after: {len(base_agent._external_capabilities)}"
        )

        await session_pool.shutdown()


# ---------------------------------------------------------------------------
# Merged from test_sessionpool_e2e_integration.py (suffix: e2e)
# ---------------------------------------------------------------------------


@pytest.mark.integration
async def test_e2e_reasoning_events_through_sessionpool() -> None:
    """End-to-end: AgentPool -> SessionPool -> OpenCode events.

    Verifies that when a model produces reasoning output, the events flow
    through the entire pipeline and reach the SSE broadcast layer.
    """
    agent_config = NativeAgentConfig(
        name="test_agent", model="test", system_prompt="You are a test agent"
    )
    manifest = AgentsManifest(agents={"test_agent": agent_config})
    async with AgentPool(manifest) as pool:
        session_pool = pool.session_pool
        assert session_pool is not None
        server_state = MockServerState()
        server_state.pool = pool
        integration = OpenCodeSessionPoolIntegration(
            session_pool=session_pool, server_state=server_state
        )
        session_id = "test-session"
        message_id = await integration.route_message(
            session_id=session_id, content="hello", priority="when_idle"
        )
        if message_id is not None:
            await session_pool.wait_for_completion(session_id)
            await asyncio.sleep(0.2)
        await integration._stop_event_consumer(session_id)
        assert len(server_state.events) > 0, (
            f"Expected SSE events, got {len(server_state.events)}."
            " Event consumer may not have been started."
        )
        event_types = [type(e).__name__ for e in server_state.events]
        print(f"Broadcast events: {event_types}")
        from wolfharness_server.opencode_server.models import PartUpdatedEvent

        part_events = [e for e in server_state.events if isinstance(e, PartUpdatedEvent)]
        assert len(part_events) > 0, (
            f"Expected PartUpdatedEvent in broadcast, got: {event_types}."
            " Events may not be flowing through EventProcessor."
        )


@pytest.mark.integration
@pytest.mark.slow
async def test_e2e_pre_existing_session_consumer_started() -> None:
    """Consumer must start even when session already exists in SessionPool."""
    agent_config = NativeAgentConfig(
        name="test_agent", model="test", system_prompt="You are a test agent"
    )
    manifest = AgentsManifest(agents={"test_agent": agent_config})
    async with AgentPool(manifest) as pool:
        session_pool = pool.session_pool
        assert session_pool is not None
        server_state = MockServerState()
        server_state.pool = pool
        integration = OpenCodeSessionPoolIntegration(
            session_pool=session_pool, server_state=server_state
        )
        session_id = "pre-existing-session"
        await session_pool.create_session(session_id, agent_name="test_agent")
        message_id = await integration.route_message(
            session_id=session_id, content="hello", priority="when_idle"
        )
        if message_id is not None:
            await session_pool.wait_for_completion(session_id)
            await asyncio.sleep(0.2)
        await integration._stop_event_consumer(session_id)
        assert len(server_state.events) > 0, (
            f"Expected SSE events for pre-existing session, got {len(server_state.events)}"
        )


# ---------------------------------------------------------------------------
# Merged from test_sessionpool_end_to_end_redflag.py (suffix: etr)
# ---------------------------------------------------------------------------


class MockSessionPool:
    """Mock SessionPool for testing."""

    def __init__(self):
        self.event_bus = EventBus()
        self.sessions = MockSessions()

    async def send_message(
        self, session_id, content, mode=DeliveryMode.QUEUE, input_provider=None, **kwargs
    ):
        return None


class MockSessions:
    """Mock Sessions manager."""

    def __init__(self):
        self._sessions = {}
        self._session_agents = {}

    async def get_or_create_session(self, session_id, agent_name=None, **metadata):
        if session_id not in self._sessions:
            from wolfharness.orchestrator.core import SessionState

            state = SessionState(session_id=session_id, agent_name=agent_name or "default")
            self._sessions[session_id] = state
            return (state, True)
        return (self._sessions[session_id], False)

    def get_session(self, session_id):
        return self._sessions.get(session_id)


@pytest.mark.asyncio
async def test_send_message_async_does_not_start_consumer():
    """Red-flag: send_message_async calls session_pool.receive_request directly.

    Without going through integration.route_message, so the event consumer
    is never started for new sessions.
    """
    server_state = MockServerState()
    session_pool = MockSessionPool()
    OpenCodeSessionPoolIntegration(session_pool=session_pool, server_state=server_state)
    session_id = "test_session"
    await session_pool.sessions.get_or_create_session(session_id, agent_name="default")
    thinking_event = PartStartEvent(index=0, part=ThinkingPart(content="Let me think..."))
    await session_pool.event_bus.publish(session_id, thinking_event)
    await asyncio.sleep(0.1)
    assert len(server_state.events) == 0, (
        f"Expected NO OpenCode events (consumer not started), got: {server_state.events}"
    )


@pytest.mark.asyncio
async def test_integration_route_message_starts_consumer():
    """Verify that integration.route_message starts the event consumer.

    And events are broadcast.
    """
    server_state = MockServerState()
    session_pool = MockSessionPool()
    integration = OpenCodeSessionPoolIntegration(
        session_pool=session_pool, server_state=server_state
    )
    session_id = "test_session"
    await integration.create_session(session_id, agent_name="default")
    thinking_event = PartStartEvent(index=0, part=ThinkingPart(content="Let me think..."))
    await session_pool.event_bus.publish(session_id, thinking_event)
    await asyncio.sleep(0.1)
    assert len(server_state.events) > 0, (
        f"Expected OpenCode events (consumer started), got: {server_state.events}"
    )
    await integration._stop_event_consumer(session_id)


@pytest.mark.asyncio
async def test_integration_route_message_starts_consumer_for_existing_session():
    """Red-flag: route_message must start consumer even for pre-existing sessions.

    Sessions created via other paths (e.g. get_or_load_session) don't have
    the consumer started, which would leave EventBus events unconsumed.
    """
    server_state = MockServerState()
    session_pool = MockSessionPool()
    integration = OpenCodeSessionPoolIntegration(
        session_pool=session_pool, server_state=server_state
    )
    session_id = "test_session"
    await session_pool.sessions.get_or_create_session(session_id, agent_name="default")
    assert session_id not in integration._session_groups
    await integration.route_message(
        session_id=session_id, content="test prompt", mode=DeliveryMode.QUEUE
    )
    await asyncio.sleep(0)
    thinking_event = PartStartEvent(index=0, part=ThinkingPart(content="Let me think..."))
    await session_pool.event_bus.publish(session_id, thinking_event)
    await asyncio.sleep(0.1)
    assert len(server_state.events) > 0, (
        f"Expected OpenCode events (consumer started), got: {server_state.events}"
    )
    await integration._stop_event_consumer(session_id)


@pytest.mark.asyncio
async def test_consumer_restarted_after_crash():
    """Red-flag: If consumer loop crashes, _start_event_consumer should restart it.

    By cleaning up the old task and starting a new one.
    """
    server_state = MockServerState()
    session_pool = MockSessionPool()
    integration = OpenCodeSessionPoolIntegration(
        session_pool=session_pool, server_state=server_state
    )
    session_id = "test_session"
    await integration._start_event_consumer(session_id)
    assert session_id in integration._session_groups
    assert session_id in integration._consumer_streams
    await integration.stop_event_consumer(session_id)
    await integration._start_event_consumer(session_id)
    assert session_id in integration._session_groups
    assert session_id in integration._consumer_streams
    await asyncio.sleep(0)
    thinking_event = PartStartEvent(index=0, part=ThinkingPart(content="Let me think..."))
    await session_pool.event_bus.publish(session_id, thinking_event)
    await asyncio.sleep(0.1)
    assert len(server_state.events) > 0, (
        f"Expected events after restart, got: {server_state.events}"
    )
    await integration._stop_event_consumer(session_id)


# ---------------------------------------------------------------------------
# Merged from test_sessionpool_subagent_e2e.py (suffix: se)
# ---------------------------------------------------------------------------


def _get_last_assistant_message(state: MockServerState, session_id: str) -> Any | None:
    """Get the last assistant message for a session."""
    messages = state.messages.get(session_id, [])
    for msg in reversed(messages):
        if hasattr(msg, "info") and hasattr(msg.info, "role") and (msg.info.role == "assistant"):
            return msg
    return None


def _get_tool_part_for_child(msg: Any, child_session_id: str) -> ToolPart | None:
    """Find the ToolPart representing a child session."""
    for part in msg.parts:
        if (
            isinstance(part, ToolPart)
            and part.state is not None
            and hasattr(part.state, "metadata")
            and isinstance(part.state.metadata, dict)
            and (part.state.metadata.get("sessionId") == child_session_id)
        ):
            return part
    return None


@pytest.mark.integration
async def test_subagent_toolpart_transitions_running_to_completed() -> None:
    """Full lifecycle: SpawnSessionStart -> StreamCompleteEvent -> ToolPart Completed.

    This is an end-to-end test that exercises _event_consumer_loop, not just
    EventProcessor in isolation. The bug only appeared because _event_consumer_loop
    handled SpawnSessionStart with 'continue' before EventProcessor saw the event.
    """
    agent_config = NativeAgentConfig(
        name="test_agent", model="test", system_prompt="You are a test agent"
    )
    manifest = AgentsManifest(agents={"test_agent": agent_config})
    async with AgentPool(manifest) as pool:
        session_pool = pool.session_pool
        assert session_pool is not None
        await session_pool.start()
        server_state = MockServerState()
        server_state.pool = pool
        integration = OpenCodeSessionPoolIntegration(
            session_pool=session_pool, server_state=server_state
        )
        parent_session_id = "parent-e2e-test"
        child_session_id = "child-e2e-test"
        await session_pool.create_session(parent_session_id, agent_name="test_agent")
        await session_pool.create_session(
            child_session_id, parent_session_id=parent_session_id, agent_name="worker"
        )
        await integration._start_event_consumer(parent_session_id)
        await asyncio.sleep(0.05)
        spawn_event = SpawnSessionStart(
            child_session_id=child_session_id,
            parent_session_id=parent_session_id,
            tool_call_id="tc-1",
            spawn_mechanism="task",
            source_name="worker",
            source_type="agent",
            depth=1,
            description="Test subagent task",
            metadata={"prompt": "do something"},
            model_id="test-model",
        )
        await session_pool.event_bus.publish(parent_session_id, spawn_event)
        await asyncio.sleep(0.1)
        assistant_msg = _get_last_assistant_message(server_state, parent_session_id)
        assert assistant_msg is not None, "No assistant message found after SpawnSessionStart"
        tool_part = _get_tool_part_for_child(assistant_msg, child_session_id)
        assert tool_part is not None, (
            f"No ToolPart found for child session {child_session_id}."
            " SpawnSessionStart handling may have failed to create it."
        )
        assert isinstance(tool_part.state, ToolStateRunning), (
            f"Expected ToolStateRunning, got {type(tool_part.state).__name__}"
        )
        assert tool_part.state.time.start is not None, "ToolPart should have start time"
        complete_event = StreamCompleteEvent(
            message=ChatMessage(role="assistant", content="Task completed successfully"),
            session_id=child_session_id,
        )
        await session_pool.event_bus.publish(child_session_id, complete_event)
        await asyncio.sleep(0.1)
        assistant_msg = _get_last_assistant_message(server_state, parent_session_id)
        assert assistant_msg is not None
        tool_part = _get_tool_part_for_child(assistant_msg, child_session_id)
        assert tool_part is not None, (
            f"ToolPart for child {child_session_id} disappeared after StreamCompleteEvent"
        )
        assert isinstance(tool_part.state, ToolStateCompleted), (
            f"Expected ToolStateCompleted, got {type(tool_part.state).__name__}."
            " The ToolPart is stuck. This usually means _event_consumer_loop"
            " or _update_parent_toolpart failed."
        )
        assert tool_part.state.time.end is not None, "Completed ToolPart should have end time set"
        assert tool_part.state.output == "Task completed successfully", (
            f"ToolPart output mismatch: {tool_part.state.output}"
        )
        await integration._stop_event_consumer(parent_session_id)
        await session_pool.shutdown()


@pytest.mark.integration
async def test_subagent_toolpart_handles_multiple_child_events() -> None:
    """Verify ToolPart transitions correctly even with intermediate child events."""
    agent_config = NativeAgentConfig(
        name="test_agent", model="test", system_prompt="You are a test agent"
    )
    manifest = AgentsManifest(agents={"test_agent": agent_config})
    async with AgentPool(manifest) as pool:
        session_pool = pool.session_pool
        assert session_pool is not None
        await session_pool.start()
        server_state = MockServerState()
        server_state.pool = pool
        integration = OpenCodeSessionPoolIntegration(
            session_pool=session_pool, server_state=server_state
        )
        parent_session_id = "parent-multi-test"
        child_session_id = "child-multi-test"
        await session_pool.create_session(parent_session_id, agent_name="test_agent")
        await session_pool.create_session(
            child_session_id, parent_session_id=parent_session_id, agent_name="analyzer"
        )
        await integration._start_event_consumer(parent_session_id)
        await asyncio.sleep(0.05)
        spawn_event = SpawnSessionStart(
            child_session_id=child_session_id,
            parent_session_id=parent_session_id,
            tool_call_id="tc-2",
            spawn_mechanism="task",
            source_name="analyzer",
            source_type="agent",
            depth=1,
            description="Analysis task",
            metadata={"prompt": "analyze this"},
        )
        await session_pool.event_bus.publish(parent_session_id, spawn_event)
        await asyncio.sleep(0.1)
        assistant_msg = _get_last_assistant_message(server_state, parent_session_id)
        assert assistant_msg is not None
        tool_part = _get_tool_part_for_child(assistant_msg, child_session_id)
        assert tool_part is not None, "ToolPart should exist after SpawnSessionStart"
        assert isinstance(tool_part.state, ToolStateRunning)
        complete_event = StreamCompleteEvent(
            message=ChatMessage(role="assistant", content="Analysis done"),
            session_id=child_session_id,
        )
        await session_pool.event_bus.publish(child_session_id, complete_event)
        await asyncio.sleep(0.1)
        assistant_msg = _get_last_assistant_message(server_state, parent_session_id)
        assert assistant_msg is not None
        tool_part = _get_tool_part_for_child(assistant_msg, child_session_id)
        assert tool_part is not None, "ToolPart should still exist after completion"
        assert isinstance(tool_part.state, ToolStateCompleted), (
            f"ToolPart stuck in {type(tool_part.state).__name__} after completion"
        )
        await integration._stop_event_consumer(parent_session_id)
        await session_pool.shutdown()


# ---------------------------------------------------------------------------
# Merged from test_session_pool_hooks.py (suffix: hk)
# ---------------------------------------------------------------------------


hook_calls: list[tuple[str, dict[str, Any]]] = []


def _reset_calls() -> None:
    hook_calls.clear()


def _make_recorder(event: str) -> CallableHook:

    def _fn(**kwargs: Any) -> HookResult:
        hook_calls.append((event, kwargs))
        return {"decision": "allow"}

    return CallableHook(event=event, fn=_fn)


def _make_run_handle(agent: Agent[Any, Any], run_ctx: AgentRunContext) -> RunHandle:
    """Create a RunHandle wired for the SessionPool path."""
    from wolfharness.lifecycle import DirectChannel, MemoryJournal

    event_bus = EventBus()
    session = SessionState(session_id="test-session", agent_name="test-agent")
    # Set up lifecycle dimensions required by RunHandle._execute_turn()
    session._comm_channel = DirectChannel(MemoryJournal())
    return RunHandle(
        run_id="test-run",
        session_id="test-session",
        agent_type="native",
        agent=agent,
        event_bus=event_bus,
        session=session,
        run_ctx=run_ctx,
    )


@pytest.mark.integration
async def test_hooks_fire_through_run_handle_start() -> None:
    """Given a RunHandle.start() path, hooks fire during turn execution.

    This is the regression test for the bug where hooks didn't fire
    when going through the SessionPool/RunHandle path.
    """
    _reset_calls()
    hooks = AgentHooks(
        pre_turn=[_make_recorder("pre_turn")], post_turn=[_make_recorder("post_turn")]
    )
    agent = Agent(
        name="test-pool-hooks", model=TestModel(custom_output_text="pool response"), hooks=hooks
    )
    async with agent:
        run_ctx = AgentRunContext(session_id="test-session")
        handle = _make_run_handle(agent, run_ctx)
        events: list[Any] = []
        gen = handle.start("hello")

        async def _consume() -> None:
            events.extend([event async for event in gen])

        consumer_task = asyncio.create_task(_consume())
        await asyncio.sleep(0.1)
        handle.close()
        await asyncio.sleep(0.1)
        await consumer_task
    event_names = [name for name, _ in hook_calls]
    assert "pre_turn" in event_names, "pre_turn hook must fire through RunHandle.start()"
    assert any(isinstance(e, StreamCompleteEvent) for e in events)


@pytest.mark.integration
async def test_hooks_fired_cleared_between_turns() -> None:
    """Given two sequential turns, turn 2 hooks still fire.

    Previously, ``hooks_fired`` on ``AgentRunContext`` needed to be cleared
    between turns to prevent turn 1's guard keys from blocking turn 2. With
    the ``hooks_fired`` guard removed (replaced by per-Turn ``_logged_tools``
    set), a new Turn instance is created for each turn with a fresh set,
    so no explicit clearing is needed.
    """
    _reset_calls()
    hooks = AgentHooks(
        pre_turn=[_make_recorder("pre_turn")], post_turn=[_make_recorder("post_turn")]
    )
    agent = Agent(
        name="test-multi-turn-hooks", model=TestModel(custom_output_text="response"), hooks=hooks
    )
    async with agent:
        run_ctx = AgentRunContext(session_id="test-session")
        handle = _make_run_handle(agent, run_ctx)
        gen = handle.start("first prompt")

        async def _consume() -> None:
            _ = [event async for event in gen]

        consumer_task = asyncio.create_task(_consume())
        await asyncio.sleep(0.1)
        handle.close()
        await asyncio.sleep(0.1)
        await consumer_task
    pre_turn_count = sum((1 for name, _ in hook_calls if name == "pre_turn"))
    assert pre_turn_count == 1, "pre_turn must fire in turn 1"
    from wolfharness.agents.native_agent.turn import NativeTurn

    turn = NativeTurn(
        agent=agent, prompts=["second prompt"], run_ctx=run_ctx, message_history=[], hooks=hooks
    )
    _ = [event async for event in turn.execute()]
    pre_turn_count = sum((1 for name, _ in hook_calls if name == "pre_turn"))
    assert pre_turn_count == 2, "pre_turn must fire again in turn 2"


@pytest.mark.integration
async def test_hooks_fire_in_direct_turn_execute() -> None:
    """Given a NativeTurn created via agent.create_turn(), hooks fire.

    This is a simplified version of the SessionPool path that verifies
    the create_turn() → turn.execute() pipeline fires hooks correctly.
    """
    _reset_calls()
    hooks = AgentHooks(
        pre_turn=[_make_recorder("pre_turn")], post_turn=[_make_recorder("post_turn")]
    )
    agent = Agent(
        name="test-create-turn-hooks", model=TestModel(custom_output_text="response"), hooks=hooks
    )
    async with agent:
        run_ctx = AgentRunContext(session_id="test-session")
        turn = agent.create_turn(prompts=["hello"], run_ctx=run_ctx, message_history=[])
        events = [event async for event in turn.execute()]
    event_names = [name for name, _ in hook_calls]
    assert "pre_turn" in event_names
    assert "post_turn" in event_names
    assert any(isinstance(e, StreamCompleteEvent) for e in events)
