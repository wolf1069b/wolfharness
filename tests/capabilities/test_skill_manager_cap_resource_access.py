"""Unit tests for ``SkillManagerCap`` ResourceAccess delegation (RFC-0058).

Verifies that ``SkillManagerCap`` (which now inherits ``ResourceAccess``)
aggregates resource listing/reading/existence from its per-skill MCP
children, without needing top-level MCP providers as ``children``.
"""

from __future__ import annotations

from typing import Any

import pytest

from wolfharness.capabilities.resource_protocols import ResourceAccess, ResourceEntry
from wolfharness.capabilities.skill_manager_cap import SkillManagerCap
from wolfharness.skills.skill import Skill
from wolfharness_config.skills import SkillMcpServerConfig


pytestmark = pytest.mark.unit


class FakeResourceChild:
    """Stand-in for a per-skill ``McpServerCap`` implementing ``ResourceAccess``."""

    @property
    def owned_schemes(self) -> frozenset[str]:
        return frozenset()

    def __init__(
        self,
        name: str,
        entries: list[ResourceEntry],
        read_results: dict[str, Any],
    ) -> None:
        self.name = name
        self._entries = entries
        self._read_results = read_results
        self.read_calls: list[str] = []
        self.exists_calls: list[str] = []

    async def list_resources(self) -> list[ResourceEntry]:
        return list(self._entries)

    async def read_resource(self, uri: str) -> Any:
        self.read_calls.append(uri)
        return self._read_results.get(uri)

    async def resource_exists(self, uri: str) -> bool:
        self.exists_calls.append(uri)
        return uri in self._read_results


def _entry(uri: str, name: str = "") -> ResourceEntry:
    return ResourceEntry(uri=uri, name=name or uri, description="", mime_type="text/plain")


def _skill_with_mcp(name: str = "alpha", server_name: str = "remote") -> Skill:
    return Skill(
        name=name,
        description=name,
        skill_path=f"/tmp/{name}",
        mcp_servers={
            server_name: SkillMcpServerConfig(url="http://localhost:9999/mcp"),
        },
    )


def _cap_with_children(children: list[FakeResourceChild]) -> SkillManagerCap:
    cap = SkillManagerCap(local_skills={"alpha": _skill_with_mcp()}, name="pool-skills")
    cap._skill_mcp_children = {"alpha": children}  # type: ignore[assignment]
    return cap


async def test_skill_manager_cap_is_resource_access() -> None:
    cap = SkillManagerCap(local_skills={}, name="pool-skills")
    assert isinstance(cap, ResourceAccess)


async def test_list_resources_aggregates_child_entries() -> None:
    child_a = FakeResourceChild("a", [_entry("mcp://a/one"), _entry("mcp://a/two")], {})
    child_b = FakeResourceChild("b", [_entry("mcp://b/three")], {})
    cap = _cap_with_children([child_a, child_b])

    entries = await cap.list_resources()

    uris = sorted(e.uri for e in entries)
    assert uris == ["mcp://a/one", "mcp://a/two", "mcp://b/three"]


async def test_read_resource_returns_child_result() -> None:
    result = [{"uri": "mcp://a/one", "text": "content"}]
    child = FakeResourceChild("a", [], {"mcp://a/one": result})
    cap = _cap_with_children([child])

    got = await cap.read_resource("mcp://a/one")

    assert got == result
    assert child.read_calls == ["mcp://a/one"]


async def test_read_resource_in_turn_skips_none() -> None:
    child_hit = FakeResourceChild("hit", [], {"mcp://file/one": {"text": "x"}})
    child_miss = FakeResourceChild("miss", [], {})
    cap = _cap_with_children([child_hit, child_miss])

    got = await cap.read_resource("mcp://file/one")

    assert got == {"text": "x"}
    # The child that has the resource is consulted first and its result
    # returned immediately; the remaining children are not queried.
    assert child_hit.read_calls == ["mcp://file/one"]
    assert child_miss.read_calls == []


async def test_read_resource_returns_none_when_absent_everywhere() -> None:
    child = FakeResourceChild("a", [], {})
    cap = _cap_with_children([child])

    assert await cap.read_resource("mcp://nowhere") is None


async def test_read_resource_skips_failing_child() -> None:
    class ExplodingChild(FakeResourceChild):
        async def read_resource(self, uri: str) -> Any:
            raise ConnectionError("boom")

    good = FakeResourceChild("good", [], {"mcp://x": {"text": "ok"}})
    bad = ExplodingChild("bad", [], {})
    cap = _cap_with_children([bad, good])

    got = await cap.read_resource("mcp://x")

    assert got == {"text": "ok"}


async def test_resource_exists_returns_true_if_any_child_has_it() -> None:
    child_hit = FakeResourceChild("hit", [], {"mcp://file/one": {"text": "x"}})
    child_miss = FakeResourceChild("miss", [], {})
    cap = _cap_with_children([child_hit, child_miss])

    assert await cap.resource_exists("mcp://file/one") is True


async def test_resource_exists_false_when_none_have_it() -> None:
    child = FakeResourceChild("a", [], {})
    cap = _cap_with_children([child])

    assert await cap.resource_exists("mcp://nowhere") is False


async def test_resource_exists_skips_failing_child() -> None:
    class ExplodingChild(FakeResourceChild):
        async def resource_exists(self, uri: str) -> bool:
            raise ConnectionError("boom")

    good = FakeResourceChild("good", [], {"mcp://x": {"text": "ok"}})
    bad = ExplodingChild("bad", [], {})
    cap = _cap_with_children([bad, good])

    assert await cap.resource_exists("mcp://x") is True
