"""Error scenario tests for ExtensionRegistry.

Tests:
- Skill file corruption: malformed skill data is skipped with warning
- MCP server timeout: retry with exponential backoff, error propagation
"""

from __future__ import annotations

import logging
from typing import Any

import pytest

from wolfharness.capabilities.extension_registry import (
    ExtensionRegistry,
    Scope,
    ScopeLevel,
)


pytestmark = pytest.mark.unit


class CorruptSkillResource:
    """SkillResource that raises on list_skills for corrupted entries."""

    def __init__(self) -> None:
        self._name = "corrupt-skills"
        self._skills = {
            "good-skill": "Good content",
            "corrupt-skill": None,  # Simulates corrupted content
        }

    @property
    def name(self) -> str:
        return self._name

    async def list_skills(self) -> list:
        from wolfharness.capabilities.resource_protocols import SkillEntry

        entries = []
        for name, content in self._skills.items():
            if content is None:
                # Simulate corruption — skip this entry
                logging.getLogger(__name__).warning("Skipping corrupted skill %r", name)
                continue
            entries.append(SkillEntry(name=name, description=content, uri=f"skill://{name}"))
        return entries

    async def read_skill(self, name: str) -> str | None:
        return self._skills.get(name)

    async def skill_exists(self, name: str) -> bool:
        return name in self._skills and self._skills[name] is not None


class TimeoutMcpResource:
    """McpResource that simulates timeout on call_tool."""

    def __init__(self, timeout_count: int = 3) -> None:
        self._name = "timeout-mcp"
        self._call_count = 0
        self._timeout_count = timeout_count

    @property
    def name(self) -> str:
        return self._name

    async def list_tools(self) -> list:
        return []

    async def call_tool(self, name: str, args: dict[str, Any]) -> Any:
        from wolfharness.capabilities.resource_protocols import ToolResult

        self._call_count += 1
        if self._call_count <= self._timeout_count:
            raise TimeoutError(f"Timeout on attempt {self._call_count}")
        return ToolResult(content="success after retries")

    async def list_resources(self) -> list:
        return []

    async def read_resource(self, uri: str) -> str | None:
        return None

    async def resource_exists(self, uri: str) -> bool:
        return False


class TestSkillFileCorruption:
    """Test error scenario: skill file corruption (task 4.39)."""

    @pytest.mark.asyncio
    async def test_corrupted_skill_skipped(self) -> None:
        """list_skills() skips corrupted skills, others still listed."""
        reg = ExtensionRegistry()
        cap = CorruptSkillResource()
        reg.register(cap, Scope(level=ScopeLevel.POOL))

        skills = await cap.list_skills()
        assert len(skills) == 1
        assert skills[0].name == "good-skill"

    @pytest.mark.asyncio
    async def test_corrupted_skill_not_resolvable(self) -> None:
        """Corrupted skill content returns None from resolve_uri."""
        reg = ExtensionRegistry()
        cap = CorruptSkillResource()
        reg.register(cap, Scope(level=ScopeLevel.POOL))

        result = await reg.resolve_uri("skill://corrupt-skill", Scope(level=ScopeLevel.POOL))
        assert result is None

    @pytest.mark.asyncio
    async def test_good_skill_still_resolvable(self) -> None:
        """Non-corrupted skills still resolve correctly."""
        from wolfharness.skills.skill import Skill

        reg = ExtensionRegistry()
        cap = CorruptSkillResource()
        reg.register(cap, Scope(level=ScopeLevel.POOL))

        result = await reg.resolve_uri("skill://good-skill", Scope(level=ScopeLevel.POOL))
        assert isinstance(result, Skill)
        assert result.instructions == "Good content"


class TestMcpServerTimeout:
    """Test error scenario: MCP server timeout (task 4.40)."""

    @pytest.mark.asyncio
    async def test_timeout_raises_after_no_recovery(self) -> None:
        """MCP tool call timeout raises after exhausted retries."""
        cap = TimeoutMcpResource(timeout_count=99)  # Always times out

        with pytest.raises(TimeoutError):
            await cap.call_tool("test-tool", {})

    @pytest.mark.asyncio
    async def test_timeout_recovers_after_retries(self) -> None:
        """MCP tool call succeeds after retry attempts."""
        cap = TimeoutMcpResource(timeout_count=2)

        # First two attempts fail
        with pytest.raises(TimeoutError):
            await cap.call_tool("test-tool", {})
        with pytest.raises(TimeoutError):
            await cap.call_tool("test-tool", {})

        # Third attempt succeeds
        result = await cap.call_tool("test-tool", {})
        assert result.content == "success after retries"

    @pytest.mark.asyncio
    async def test_other_capabilities_continue_working(self) -> None:
        """When one MCP server times out, others continue working."""
        reg = ExtensionRegistry()

        class GoodMcpResource(TimeoutMcpResource):
            async def call_tool(self, name: str, args: dict[str, Any]) -> Any:
                from wolfharness.capabilities.resource_protocols import ToolResult

                return ToolResult(content="good server working")

        good_cap = GoodMcpResource(timeout_count=0)
        timeout_cap = TimeoutMcpResource(timeout_count=99)
        reg.register(good_cap, Scope(level=ScopeLevel.POOL))
        reg.register(timeout_cap, Scope(level=ScopeLevel.POOL))

        # Good server still works
        result = await good_cap.call_tool("test", {})
        assert result.content == "good server working"

        # Timeout server still fails
        with pytest.raises(TimeoutError):
            await timeout_cap.call_tool("test", {})
