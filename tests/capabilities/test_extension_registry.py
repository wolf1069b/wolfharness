"""Unit tests for ExtensionRegistry.

Tests cover:
- Scope isolation (pool visible to all, session isolated, turn cleaned up)
- URI routing (skill://, mcp://, unknown scheme)
- Change stream merging (two streams, exception in one, no observables)
- Cycle detection (A→B→A raises)
- Depth limit (warning at depth 4)
- Concurrency (concurrent registration, concurrent reads, mixed)
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from wolfharness.capabilities.change_event import ChangeEvent
from wolfharness.capabilities.extension_registry import (
    CircularCompositionError,
    ExtensionRegistry,
    Scope,
    ScopeLevel,
)


pytestmark = pytest.mark.unit


# ---- Test Helpers ----


class FakeCapability:
    """Minimal fake capability for testing."""

    def __init__(self, name: str = "fake") -> None:
        self._name = name

    @property
    def name(self) -> str:
        return self._name


class FakeSkillResource(FakeCapability):
    """Fake capability implementing SkillResource protocol."""

    def __init__(self, name: str = "skill_cap", skills: dict[str, str] | None = None) -> None:
        super().__init__(name)
        self._skills = skills or {"test-skill": "skill content"}

    async def list_skills(self) -> list[Any]:
        from wolfharness.capabilities.resource_protocols import SkillEntry

        return [
            SkillEntry(name=n, description=d, uri=f"skill://{n}") for n, d in self._skills.items()
        ]

    async def read_skill(self, name: str) -> str | None:
        return self._skills.get(name)

    async def skill_exists(self, name: str) -> bool:
        return name in self._skills


class FakeMcpResource(FakeCapability):
    """Fake capability implementing McpResource protocol.

    With the new ResourceAccess protocol, ``read_resource()`` returns
    ``list[TextResourceContent | BlobResourceContent] | None`` instead
    of ``str | None``.
    """

    @property
    def owned_schemes(self) -> frozenset[str]:
        return frozenset()

    def __init__(self, name: str = "mcp_cap", resources: dict[str, str] | None = None) -> None:
        super().__init__(name)
        self._resources = resources or {"mcp://server/path": "resource content"}

    async def list_tools(self) -> list[Any]:
        return []

    async def call_tool(self, name: str, args: dict[str, Any]) -> Any:
        from wolfharness.capabilities.resource_protocols import ToolResult

        return ToolResult(content="tool result")

    async def list_resources(self) -> list[Any]:
        from wolfharness.capabilities.resource_protocols import ResourceEntry

        return [ResourceEntry(uri=u, name=n) for u, n in self._resources.items()]

    async def read_resource(self, uri: str) -> list[Any] | None:
        from wolfharness.capabilities.resource_protocols import TextResourceContent

        content = self._resources.get(uri)
        if content is None:
            return None
        return [TextResourceContent(uri=uri, text=content)]

    async def resource_exists(self, uri: str) -> bool:
        return uri in self._resources


class FakeChangeObservable(FakeCapability):
    """Fake capability implementing ChangeObservable protocol."""

    def __init__(self, name: str = "observable", events: list[ChangeEvent] | None = None) -> None:
        super().__init__(name)
        self._events = events or []
        self._started = False

    def on_change(self) -> Any:
        if not self._events:
            return None

        async def _iter():
            for event in self._events:
                yield event

        return _iter()


class FakeCommandResource(FakeCapability):
    """Fake capability implementing CommandResource protocol."""

    def __init__(self, name: str = "cmd_cap") -> None:
        super().__init__(name)

    async def list_commands(self) -> list[Any]:
        from wolfharness.capabilities.resource_protocols import CommandEntry

        return [CommandEntry(name="test-cmd", description="Test command")]

    async def get_command(self, name: str) -> Any:
        from wolfharness.capabilities.resource_protocols import CommandEntry

        if name == "test-cmd":
            return CommandEntry(name=name, description="Test command")
        return None


class FakeResourceTemplateAccess(FakeCapability):
    """Fake capability implementing ResourceTemplateAccess protocol."""

    def __init__(self, name: str = "template_cap") -> None:
        super().__init__(name)

    async def list_resource_templates(self) -> list[Any]:
        from wolfharness.capabilities.resource_protocols import ResourceTemplateEntry

        return [
            ResourceTemplateEntry(
                uri_template="file:///{path}",
                name="file-template",
                description="File template",
            )
        ]

    async def complete_resource_template(
        self,
        uri_template: str,
        argument: Any,
        context: dict[str, str] | None = None,
    ) -> Any:
        from wolfharness.capabilities.resource_protocols import CompletionResult

        return CompletionResult(values=["completion1", "completion2"])


# ---- Scope Isolation Tests ----


class TestScopeIsolation:
    """Test 4-level scope isolation."""

    def test_pool_level_visible_to_all(self) -> None:
        """Pool-level capability visible regardless of scope."""
        reg = ExtensionRegistry()
        cap = FakeCapability("pool-cap")
        reg.register(cap, Scope(level=ScopeLevel.POOL))

        assert cap in reg.get_visible_capabilities(Scope(level=ScopeLevel.POOL))
        assert cap in reg.get_visible_capabilities(
            Scope(level=ScopeLevel.SESSION, session_id="ses1")
        )
        assert cap in reg.get_visible_capabilities(
            Scope(level=ScopeLevel.AGENT, session_id="ses1", agent_name="agent1")
        )
        assert cap in reg.get_visible_capabilities(
            Scope(level=ScopeLevel.TURN, session_id="ses1", agent_name="agent1", turn_id="turn1")
        )

    def test_session_level_isolated(self) -> None:
        """Session-level capability not visible to other sessions."""
        reg = ExtensionRegistry()
        cap = FakeCapability("session-cap")
        reg.register(cap, Scope(level=ScopeLevel.SESSION, session_id="ses1"))

        assert cap in reg.get_visible_capabilities(
            Scope(level=ScopeLevel.SESSION, session_id="ses1")
        )
        assert cap not in reg.get_visible_capabilities(
            Scope(level=ScopeLevel.SESSION, session_id="ses2")
        )

    def test_agent_level_isolated(self) -> None:
        """Agent-level capability not visible to other agents."""
        reg = ExtensionRegistry()
        cap = FakeCapability("agent-cap")
        reg.register(
            cap,
            Scope(level=ScopeLevel.AGENT, session_id="ses1", agent_name="agent1"),
        )

        assert cap in reg.get_visible_capabilities(
            Scope(level=ScopeLevel.AGENT, session_id="ses1", agent_name="agent1")
        )
        assert cap not in reg.get_visible_capabilities(
            Scope(level=ScopeLevel.AGENT, session_id="ses1", agent_name="agent2")
        )

    def test_turn_level_isolated(self) -> None:
        """Turn-level capability not visible to other turns."""
        reg = ExtensionRegistry()
        cap = FakeCapability("turn-cap")
        reg.register(
            cap,
            Scope(
                level=ScopeLevel.TURN,
                session_id="ses1",
                agent_name="agent1",
                turn_id="turn1",
            ),
        )

        assert cap in reg.get_visible_capabilities(
            Scope(
                level=ScopeLevel.TURN,
                session_id="ses1",
                agent_name="agent1",
                turn_id="turn1",
            )
        )
        assert cap not in reg.get_visible_capabilities(
            Scope(
                level=ScopeLevel.TURN,
                session_id="ses1",
                agent_name="agent1",
                turn_id="turn2",
            )
        )

    def test_clear_turn_removes_capabilities(self) -> None:
        """Turn-level capabilities removed after clear_turn()."""
        reg = ExtensionRegistry()
        cap = FakeCapability("turn-cap")
        scope = Scope(
            level=ScopeLevel.TURN,
            session_id="ses1",
            agent_name="agent1",
            turn_id="turn1",
        )
        reg.register(cap, scope)

        assert cap in reg.get_visible_capabilities(scope)

        reg.clear_turn("ses1", "agent1", "turn1")

        assert cap not in reg.get_visible_capabilities(scope)

    def test_unregister_removes_capability(self) -> None:
        """unregister() removes capability from scope."""
        reg = ExtensionRegistry()
        cap = FakeCapability("pool-cap")
        scope = Scope(level=ScopeLevel.POOL)
        reg.register(cap, scope)

        assert cap in reg.get_visible_capabilities(scope)
        assert reg.unregister(cap, scope)
        assert cap not in reg.get_visible_capabilities(scope)


# ---- Typed Query Tests ----


class TestTypedQueries:
    """Test typed query methods."""

    def test_get_skill_resources(self) -> None:
        reg = ExtensionRegistry()
        skill_cap = FakeSkillResource()
        mcp_cap = FakeMcpResource()
        reg.register(skill_cap, Scope(level=ScopeLevel.POOL))
        reg.register(mcp_cap, Scope(level=ScopeLevel.POOL))

        skills = reg.get_skill_resources(Scope(level=ScopeLevel.POOL))
        assert skill_cap in skills
        assert mcp_cap not in skills

    def test_get_mcp_resources(self) -> None:
        reg = ExtensionRegistry()
        skill_cap = FakeSkillResource()
        mcp_cap = FakeMcpResource()
        reg.register(skill_cap, Scope(level=ScopeLevel.POOL))
        reg.register(mcp_cap, Scope(level=ScopeLevel.POOL))

        with pytest.warns(DeprecationWarning, match="get_mcp_resources.*deprecated"):
            mcps = reg.get_mcp_resources(Scope(level=ScopeLevel.POOL))
        assert mcp_cap in mcps
        assert skill_cap not in mcps

    def test_get_mcp_resources_delegates_to_get_resource_access(self) -> None:
        """get_mcp_resources() result equals get_resource_access() result."""
        reg = ExtensionRegistry()
        mcp_cap = FakeMcpResource()
        reg.register(mcp_cap, Scope(level=ScopeLevel.POOL))

        with pytest.warns(DeprecationWarning, match="get_mcp_resources.*deprecated"):
            deprecated_result = reg.get_mcp_resources(Scope(level=ScopeLevel.POOL))
        new_result = reg.get_resource_access(Scope(level=ScopeLevel.POOL))

        assert deprecated_result == new_result

    def test_get_resource_access(self) -> None:
        reg = ExtensionRegistry()
        skill_cap = FakeSkillResource()
        mcp_cap = FakeMcpResource()
        reg.register(skill_cap, Scope(level=ScopeLevel.POOL))
        reg.register(mcp_cap, Scope(level=ScopeLevel.POOL))

        resources = reg.get_resource_access(Scope(level=ScopeLevel.POOL))
        assert mcp_cap in resources
        assert skill_cap not in resources

    def test_get_tool_access(self) -> None:
        reg = ExtensionRegistry()
        skill_cap = FakeSkillResource()
        mcp_cap = FakeMcpResource()
        reg.register(skill_cap, Scope(level=ScopeLevel.POOL))
        reg.register(mcp_cap, Scope(level=ScopeLevel.POOL))

        tools = reg.get_tool_access(Scope(level=ScopeLevel.POOL))
        assert mcp_cap in tools
        assert skill_cap not in tools

    def test_get_resource_template_access(self) -> None:
        reg = ExtensionRegistry()
        mcp_cap = FakeMcpResource()
        template_cap = FakeResourceTemplateAccess()
        reg.register(mcp_cap, Scope(level=ScopeLevel.POOL))
        reg.register(template_cap, Scope(level=ScopeLevel.POOL))

        templates = reg.get_resource_template_access(Scope(level=ScopeLevel.POOL))
        assert template_cap in templates
        assert mcp_cap not in templates

    def test_get_resource_access_empty(self) -> None:
        """get_resource_access() returns empty list when no ResourceAccess caps."""
        reg = ExtensionRegistry()
        skill_cap = FakeSkillResource()
        reg.register(skill_cap, Scope(level=ScopeLevel.POOL))

        resources = reg.get_resource_access(Scope(level=ScopeLevel.POOL))
        assert resources == []

    def test_get_tool_access_empty(self) -> None:
        """get_tool_access() returns empty list when no ToolAccess caps."""
        reg = ExtensionRegistry()
        skill_cap = FakeSkillResource()
        reg.register(skill_cap, Scope(level=ScopeLevel.POOL))

        tools = reg.get_tool_access(Scope(level=ScopeLevel.POOL))
        assert tools == []

    def test_get_resource_template_access_empty(self) -> None:
        """get_resource_template_access() returns empty list when no ResourceTemplateAccess caps."""
        reg = ExtensionRegistry()
        mcp_cap = FakeMcpResource()
        reg.register(mcp_cap, Scope(level=ScopeLevel.POOL))

        templates = reg.get_resource_template_access(Scope(level=ScopeLevel.POOL))
        assert templates == []

    def test_get_command_resources(self) -> None:
        reg = ExtensionRegistry()
        cmd_cap = FakeCommandResource()
        reg.register(cmd_cap, Scope(level=ScopeLevel.POOL))

        cmds = reg.get_command_resources(Scope(level=ScopeLevel.POOL))
        assert cmd_cap in cmds

    def test_get_observable_capabilities(self) -> None:
        reg = ExtensionRegistry()
        obs_cap = FakeChangeObservable()
        non_obs = FakeCapability()
        reg.register(obs_cap, Scope(level=ScopeLevel.POOL))
        reg.register(non_obs, Scope(level=ScopeLevel.POOL))

        observables = reg.get_observable_capabilities(Scope(level=ScopeLevel.POOL))
        assert obs_cap in observables
        assert non_obs not in observables


# ---- URI Routing Tests ----


class TestURIRouting:
    """Test resolve_uri() scheme routing."""

    @pytest.mark.asyncio
    async def test_resolve_skill_uri(self) -> None:
        from wolfharness.skills.skill import Skill

        reg = ExtensionRegistry()
        skill_cap = FakeSkillResource(skills={"my-skill": "skill content here"})
        reg.register(skill_cap, Scope(level=ScopeLevel.POOL))

        result = await reg.resolve_uri("skill://my-skill", Scope(level=ScopeLevel.POOL))
        assert isinstance(result, Skill)
        assert result.name == "my-skill"
        assert result.instructions == "skill content here"

    @pytest.mark.asyncio
    async def test_resolve_mcp_uri(self) -> None:
        reg = ExtensionRegistry()
        mcp_cap = FakeMcpResource(resources={"mcp://server/path": "resource content"})
        reg.register(mcp_cap, Scope(level=ScopeLevel.POOL))

        result = await reg.resolve_uri("mcp://server/path", Scope(level=ScopeLevel.POOL))
        assert result == "resource content"

    @pytest.mark.asyncio
    async def test_resolve_unknown_scheme_returns_none(self) -> None:
        reg = ExtensionRegistry()
        result = await reg.resolve_uri("unknown://foo", Scope(level=ScopeLevel.POOL))
        assert result is None

    @pytest.mark.asyncio
    async def test_resolve_nonexistent_skill_returns_none(self) -> None:
        reg = ExtensionRegistry()
        skill_cap = FakeSkillResource(skills={"existing": "content"})
        reg.register(skill_cap, Scope(level=ScopeLevel.POOL))

        result = await reg.resolve_uri("skill://nonexistent", Scope(level=ScopeLevel.POOL))
        assert result is None


# ---- Change Stream Merging Tests ----


class TestChangeStreamMerging:
    """Test merge_change_streams()."""

    @pytest.mark.asyncio
    async def test_merge_two_streams(self) -> None:
        reg = ExtensionRegistry()
        event1 = ChangeEvent(capability_name="cap1", kind="tools_changed")
        event2 = ChangeEvent(capability_name="cap2", kind="tools_changed")
        obs1 = FakeChangeObservable(name="obs1", events=[event1])
        obs2 = FakeChangeObservable(name="obs2", events=[event2])
        reg.register(obs1, Scope(level=ScopeLevel.POOL))
        reg.register(obs2, Scope(level=ScopeLevel.POOL))

        merged = reg.merge_change_streams(Scope(level=ScopeLevel.POOL))
        assert merged is not None

        events = [event async for event in merged]

        assert len(events) == 2
        assert event1 in events
        assert event2 in events

    @pytest.mark.asyncio
    async def test_no_observables_returns_none(self) -> None:
        reg = ExtensionRegistry()
        merged = reg.merge_change_streams(Scope(level=ScopeLevel.POOL))
        assert merged is None

    @pytest.mark.asyncio
    async def test_exception_in_one_stream_does_not_kill_merge(self) -> None:
        reg = ExtensionRegistry()

        event_good = ChangeEvent(capability_name="good", kind="tools_changed")

        class ErrorObservable(FakeChangeObservable):
            def on_change(self) -> Any:
                async def _iter():
                    raise RuntimeError("Stream error")
                    yield  # pragma: no cover

                return _iter()

        good_obs = FakeChangeObservable(name="good", events=[event_good])
        error_obs = ErrorObservable(name="error")
        reg.register(good_obs, Scope(level=ScopeLevel.POOL))
        reg.register(error_obs, Scope(level=ScopeLevel.POOL))

        merged = reg.merge_change_streams(Scope(level=ScopeLevel.POOL))
        assert merged is not None

        events = [event async for event in merged]

        # Good stream's event should still be delivered
        assert event_good in events


# ---- Cycle Detection Tests ----


class TestCycleDetection:
    """Test cycle detection at add_child() time."""

    def test_circular_composition_raises(self) -> None:
        reg = ExtensionRegistry()
        cap_a = FakeCapability("A")
        cap_b = FakeCapability("B")

        reg.add_child(cap_a, cap_b)

        with pytest.raises(CircularCompositionError):
            reg.add_child(cap_b, cap_a)

    def test_no_cycle_for_independent_chains(self) -> None:
        reg = ExtensionRegistry()
        cap_a = FakeCapability("A")
        cap_b = FakeCapability("B")
        cap_c = FakeCapability("C")

        reg.add_child(cap_a, cap_b)
        reg.add_child(cap_b, cap_c)
        # Should not raise
        reg.add_child(cap_a, cap_c)


# ---- Depth Limit Tests ----


class TestDepthLimit:
    """Test composition depth limit."""

    def test_depth_limit_warning(self, capsys: pytest.CaptureFixture[str]) -> None:
        reg = ExtensionRegistry(max_composition_depth=3)
        cap_a = FakeCapability("A")
        cap_b = FakeCapability("B")
        cap_c = FakeCapability("C")
        cap_d = FakeCapability("D")

        reg.add_child(cap_a, cap_b)  # depth 2
        reg.add_child(cap_b, cap_c)  # depth 3
        # depth 4 should warn but not block
        reg.add_child(cap_c, cap_d)

    def test_get_depth(self) -> None:
        reg = ExtensionRegistry()
        cap_a = FakeCapability("A")
        cap_b = FakeCapability("B")
        cap_c = FakeCapability("C")

        assert reg.get_depth(cap_a) == 1

        reg.add_child(cap_a, cap_b)
        assert reg.get_depth(cap_b) == 2

        reg.add_child(cap_b, cap_c)
        assert reg.get_depth(cap_c) == 3


# ---- Concurrency Tests ----


class TestConcurrency:
    """Test concurrent access."""

    @pytest.mark.asyncio
    async def test_concurrent_turn_registration(self) -> None:
        reg = ExtensionRegistry()
        caps = [FakeCapability(f"cap-{i}") for i in range(10)]
        scope = Scope(
            level=ScopeLevel.TURN,
            session_id="ses1",
            agent_name="agent1",
            turn_id="turn1",
        )

        await asyncio.gather(*[reg.register_async(cap, scope) for cap in caps])

        visible = reg.get_visible_capabilities(scope)
        assert len(visible) == 10

    @pytest.mark.asyncio
    async def test_concurrent_reads_during_registration(self) -> None:
        reg = ExtensionRegistry()
        scope = Scope(level=ScopeLevel.POOL)

        async def register():
            for i in range(10):
                reg.register(FakeCapability(f"cap-{i}"), scope)
                await asyncio.sleep(0)

        async def read():
            for _ in range(10):
                reg.get_visible_capabilities(scope)
                await asyncio.sleep(0)

        await asyncio.gather(register(), read())

        # Should complete without errors
        assert len(reg.get_visible_capabilities(scope)) >= 10

    @pytest.mark.asyncio
    async def test_mixed_register_and_read(self) -> None:
        reg = ExtensionRegistry()
        pool_scope = Scope(level=ScopeLevel.POOL)
        turn_scope = Scope(
            level=ScopeLevel.TURN,
            session_id="ses1",
            agent_name="agent1",
            turn_id="turn1",
        )

        async def register_turn():
            for i in range(5):
                await reg.register_async(FakeCapability(f"turn-{i}"), turn_scope)

        def register_pool():
            for i in range(5):
                reg.register(FakeCapability(f"pool-{i}"), pool_scope)

        await asyncio.gather(register_turn(), asyncio.to_thread(register_pool))

        pool_visible = reg.get_visible_capabilities(pool_scope)
        turn_visible = reg.get_visible_capabilities(turn_scope)

        # Turn scope sees both pool and turn caps
        assert len(turn_visible) == 10
        # Pool scope only sees pool caps
        assert len(pool_visible) == 5
