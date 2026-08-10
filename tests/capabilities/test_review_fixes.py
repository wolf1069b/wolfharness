"""TDD tests for PR #143 review fixes.

Tests for:
1. CRITICAL: __aexit__ should not close shared pool transport
4. HIGH: matcher_fn backward compat with 2-arg signature
5+6. HIGH: McpServerCap read_skill/skill_exists parse skill:// URI
7. HIGH: resolve_uri provider-aware routing
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock

import pytest

from wolfharness.capabilities.mcp_server_cap import McpServerCap


pytestmark = pytest.mark.unit


if TYPE_CHECKING:
    from typing import Self

    from wolfharness.capabilities.resource_protocols import SkillEntry


# ---------------------------------------------------------------------------
# Shared test doubles (copied from test_mcp_server_cap.py)
# ---------------------------------------------------------------------------


@dataclass
class FakeMCPClient:
    """Async mock of MCPClient for testing McpServerCap."""

    _tools: list[Any] = field(default_factory=list)
    _resources: list[Any] = field(default_factory=list)
    _read_results: dict[str, list[Any]] = field(default_factory=dict)
    _connected: bool = False
    _tool_change_callback: Any = None
    _exited: bool = False
    config: Any = None

    async def __aenter__(self) -> Self:
        self._connected = True
        return self

    async def __aexit__(self, *args: object) -> None:
        self._connected = False
        self._exited = True

    async def list_tools(self) -> list[Any]:
        if not self._connected:
            raise RuntimeError("Not connected")
        return list(self._tools)

    async def list_resources(self) -> list[Any]:
        if not self._connected:
            raise RuntimeError("Not connected")
        return list(self._resources)

    async def read_resource(self, uri: str) -> list[Any]:
        if not self._connected:
            raise RuntimeError("Not connected")
        if uri not in self._read_results:
            raise RuntimeError(f"Resource not found: {uri}")
        return self._read_results[uri]

    async def call_tool(self, name: str, *args: Any, **kwargs: Any) -> str:
        return f"called:{name}"

    def convert_tool(self, tool: Any) -> Any:
        return tool


class FakeSessionPool:
    """Fake SessionConnectionPool that returns a FakeMCPClient."""

    def __init__(self, client: FakeMCPClient) -> None:
        self._client = client
        self.get_client_call_count = 0

    async def get_client(self, config: Any, skill_name: str | None = None) -> FakeMCPClient:
        self.get_client_call_count += 1
        self._client._connected = True
        return self._client


def _make_resource(
    uri: str, name: str = "", description: str = "", mime_type: str = ""
) -> MagicMock:
    res = MagicMock()
    res.uri = uri
    res.name = name
    res.description = description
    res.mimeType = mime_type
    return res


def _make_text_content(text: str) -> MagicMock:
    content = MagicMock()
    content.text = text
    return content


def _make_config(client_id: str = "test_server") -> MagicMock:
    config = MagicMock()
    config.client_id = client_id
    return config


# ---------------------------------------------------------------------------
# Fix 1: __aexit__ should not close shared pool transport
# ---------------------------------------------------------------------------


class TestAexitGuardsPoolTransport:
    """CRITICAL: __aexit__ must not close client from session pool."""

    @pytest.mark.asyncio
    async def test_aexit_does_not_close_pooled_client(self) -> None:
        """When client comes from session pool, __aexit__ must not call client.__aexit__."""
        client = FakeMCPClient()
        pool = FakeSessionPool(client)
        cap = McpServerCap(config=_make_config(), session_pool=pool)

        # Trigger lazy init via session pool
        await cap.list_tools()
        assert cap._client is not None
        assert not client._exited

        # Exit the cap — should NOT close the client since it came from pool
        await cap.__aexit__(None, None, None)
        assert cap._client is None
        assert not client._exited, "Pooled client was closed by __aexit__!"

    @pytest.mark.asyncio
    async def test_aexit_closes_pre_created_client(self) -> None:
        """When client was pre-created (no session pool), __aexit__ should close it."""
        client = FakeMCPClient()
        client._connected = True  # Pre-connect since _ensure_client returns as-is
        cap = McpServerCap(config=_make_config(), session_pool=None, client=client)

        # Trigger init — returns the pre-created client directly
        await cap.list_tools()
        assert cap._client is not None

        # Exit the cap — SHOULD close the client since it was pre-created
        await cap.__aexit__(None, None, None)
        assert cap._client is None
        assert client._exited, "Pre-created client was NOT closed by __aexit__!"


# ---------------------------------------------------------------------------
# Fix 5+6: McpServerCap read_skill/skill_exists parse skill:// URI
# ---------------------------------------------------------------------------


class TestMcpServerCapUriParsing:
    """McpServerCap should parse skill:// URIs and extract the short name."""

    @pytest.mark.asyncio
    async def test_read_skill_with_skill_uri(self) -> None:
        """read_skill should extract short name from skill://provider/name URI."""
        client = FakeMCPClient(
            _resources=[_make_resource(uri="file:///skills/my-skill", name="my-skill")],
            _read_results={"file:///skills/my-skill": [_make_text_content("skill content")]},
        )
        client._connected = True
        cap = McpServerCap(config=_make_config(), session_pool=None, client=client)

        # Pass full skill:// URI
        result = await cap.read_skill("skill://test_server/my-skill")
        assert result == "skill content"

    @pytest.mark.asyncio
    async def test_read_skill_with_plain_name_still_works(self) -> None:
        """read_skill should still work with plain skill names (no URI)."""
        client = FakeMCPClient(
            _resources=[_make_resource(uri="file:///skills/my-skill", name="my-skill")],
            _read_results={"file:///skills/my-skill": [_make_text_content("skill content")]},
        )
        client._connected = True
        cap = McpServerCap(config=_make_config(), session_pool=None, client=client)

        result = await cap.read_skill("my-skill")
        assert result == "skill content"

    @pytest.mark.asyncio
    async def test_skill_exists_with_skill_uri(self) -> None:
        """skill_exists should extract short name from skill:// URI."""
        client = FakeMCPClient(
            _resources=[_make_resource(uri="file:///skills/my-skill", name="my-skill")],
        )
        client._connected = True
        cap = McpServerCap(config=_make_config(), session_pool=None, client=client)

        assert await cap.skill_exists("skill://test_server/my-skill") is True
        assert await cap.skill_exists("skill://test_server/nonexistent") is False

    @pytest.mark.asyncio
    async def test_skill_exists_with_plain_name_still_works(self) -> None:
        """skill_exists should still work with plain names."""
        client = FakeMCPClient(
            _resources=[_make_resource(uri="file:///skills/my-skill", name="my-skill")],
        )
        client._connected = True
        cap = McpServerCap(config=_make_config(), session_pool=None, client=client)

        assert await cap.skill_exists("my-skill") is True
        assert await cap.skill_exists("nonexistent") is False


# ---------------------------------------------------------------------------
# Fix 4: matcher_fn backward compat with 2-arg signature
# ---------------------------------------------------------------------------


class TestMatcherFnBackwardCompat:
    """SkillManagerCap.get_instructions callable should support 2-arg matcher_fn."""

    @pytest.mark.asyncio
    async def test_two_arg_matcher_fn_receives_skill_names(self) -> None:
        """Matcher functions expecting (messages, skill_names) should work."""
        from wolfharness.capabilities.skill_manager_cap import SkillManagerCap
        from wolfharness.skills.skill import Skill

        # Create a minimal skill
        skill = MagicMock(spec=Skill)
        skill.name = "test-skill"
        skill.description = "Test skill"
        skill.disable_model_invocation = False
        skill.load_instructions.return_value = "instructions here"
        skill.mcp_servers = None
        skill.tools = None

        received_args: list[Any] = []

        def matcher_2arg(messages: Any, skill_names: list[str]) -> list[str]:
            received_args.append((messages, skill_names))
            return ["test-skill"]

        cap = SkillManagerCap(
            local_skills={"test-skill": skill},
            matcher_fn=matcher_2arg,
            inject_mode="matcher",
        )

        # get_instructions returns [metadata, callable]
        result = cap.get_instructions()
        assert result is not None
        assert isinstance(result, list)
        dynamic_fn = result[1]

        # Call the dynamic callable with a mock RunContext
        messages: list[Any] = []
        run_ctx = MagicMock()
        run_ctx.messages = messages
        await dynamic_fn(run_ctx)

        # Verify matcher was called with 2 args
        assert len(received_args) == 1
        assert received_args[0][0] is messages
        assert received_args[0][1] == ["test-skill"]

    @pytest.mark.asyncio
    async def test_one_arg_matcher_fn_still_works(self) -> None:
        """Single-arg matcher functions should continue to work."""
        from wolfharness.capabilities.skill_manager_cap import SkillManagerCap

        skill = MagicMock()
        skill.name = "test-skill"
        skill.description = "Test skill"
        skill.disable_model_invocation = False
        skill.load_instructions.return_value = "instructions here"
        skill.mcp_servers = None
        skill.tools = None

        received_messages: list[Any] = []

        def matcher_1arg(messages: Any) -> list[str]:
            received_messages.append(messages)
            return ["test-skill"]

        cap = SkillManagerCap(
            local_skills={"test-skill": skill},
            matcher_fn=matcher_1arg,
            inject_mode="matcher",
        )

        # get_instructions returns [metadata, callable]
        result = cap.get_instructions()
        assert result is not None
        assert isinstance(result, list)
        dynamic_fn = result[1]

        # Call the dynamic callable with a mock RunContext
        messages: list[Any] = []
        run_ctx = MagicMock()
        run_ctx.messages = messages
        await dynamic_fn(run_ctx)

        assert len(received_messages) == 1
        assert received_messages[0] is messages


# ---------------------------------------------------------------------------
# Fix 7: resolve_uri provider-aware routing
# ---------------------------------------------------------------------------


class TestResolveUriProviderRouting:
    """ExtensionRegistry.resolve_uri routes skill:// URIs (D9 flat format)."""

    @pytest.mark.asyncio
    async def test_resolve_uri_with_provider_routes_to_correct_cap(self) -> None:
        """skill://name with flat URI resolves to first matching cap.

        With D9, provider segment is removed. skill://shared-name
        iterates all caps and returns the first match.
        """
        from wolfharness.capabilities.extension_registry import (
            ExtensionRegistry,
            Scope,
            ScopeLevel,
        )
        from wolfharness.capabilities.resource_protocols import SkillEntry
        from wolfharness.skills.skill import Skill

        class NamedSkillCap:
            """Fake SkillResource with a configurable serialization name."""

            def __init__(self, ser_name: str, skills: dict[str, str]) -> None:
                self._ser_name = ser_name
                self._skills = skills
                self.skill_exists_calls: list[str] = []

            def get_serialization_name(self) -> str:
                return self._ser_name

            async def list_skills(self) -> list[SkillEntry]:
                return [
                    SkillEntry(name=n, description=d, uri=f"skill://{n}")
                    for n, d in self._skills.items()
                ]

            async def read_skill(self, name: str) -> str | None:
                return self._skills.get(name)

            async def skill_exists(self, name: str) -> bool:
                self.skill_exists_calls.append(name)
                return name in self._skills

        cap_a = NamedSkillCap("providerA", {"shared-name": "content from A"})
        cap_b = NamedSkillCap("providerB", {"shared-name": "content from B"})

        reg = ExtensionRegistry()
        reg.register(cap_a, Scope(level=ScopeLevel.POOL))
        reg.register(cap_b, Scope(level=ScopeLevel.POOL))

        # Flat URI (D9): skill://shared-name resolves to first matching cap.
        result = await reg.resolve_uri("skill://shared-name", Scope(level=ScopeLevel.POOL))
        assert isinstance(result, Skill)
        assert result.instructions == "content from A"

    @pytest.mark.asyncio
    async def test_resolve_uri_without_provider_iterates_all(self) -> None:
        """skill://name without provider should iterate all (backward compat)."""
        from wolfharness.capabilities.extension_registry import (
            ExtensionRegistry,
            Scope,
            ScopeLevel,
        )
        from wolfharness.capabilities.resource_protocols import SkillEntry
        from wolfharness.skills.skill import Skill

        class NamedSkillCap:
            def __init__(self, ser_name: str, skills: dict[str, str]) -> None:
                self._ser_name = ser_name
                self._skills = skills

            def get_serialization_name(self) -> str:
                return self._ser_name

            async def list_skills(self) -> list[SkillEntry]:
                return [
                    SkillEntry(name=n, description=d, uri=f"skill://{n}")
                    for n, d in self._skills.items()
                ]

            async def read_skill(self, name: str) -> str | None:
                return self._skills.get(name)

            async def skill_exists(self, name: str) -> bool:
                return name in self._skills

        cap_a = NamedSkillCap("providerA", {"my-skill": "content from A"})

        reg = ExtensionRegistry()
        reg.register(cap_a, Scope(level=ScopeLevel.POOL))

        result = await reg.resolve_uri("skill://my-skill", Scope(level=ScopeLevel.POOL))
        assert isinstance(result, Skill)
        assert result.instructions == "content from A"
