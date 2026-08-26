"""Unit tests for the ResourceAccess + ResourceTemplateAccess surface of WikiBuildCapability.

OPA/OPS/OPL tickets are exposed as MCP resources.

``xeno_adp_agentic`` is a separate package that is not installed in the
agentpool environment, so the lazy imports inside ``_ensure_tools()`` are
faked via ``sys.modules`` — the tested surface is the capability's resource
protocol mapping, not the wiki storage layer.
"""

from __future__ import annotations

from pathlib import Path
import types
from typing import Any, ClassVar

import pytest

from wolfharness.capabilities.resource_protocols import (
    CompletionArgument,
    ResourceAccess,
    ResourceTemplateAccess,
    TextResourceContent,
)
from wolfharness.capabilities.viking.wiki_build import WikiBuildCapability


pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Fake xeno_adp_agentic surface (mirrors the real WikiBuildTools contract)
# ---------------------------------------------------------------------------


def _parse_frontmatter(content: str) -> dict[str, str]:
    """Read ``key: value`` lines from a leading YAML frontmatter block."""
    frontmatter: dict[str, str] = {}
    if not content.startswith("---"):
        return frontmatter
    end = content.find("\n---", 3)
    if end < 0:
        return frontmatter
    for line in content[3:end].splitlines():
        key, separator, value = line.partition(":")
        if separator:
            frontmatter[key.strip()] = value.strip()
    return frontmatter


class FakeWikiBuildTools:
    """In-process stand-in for ``WikiBuildTools`` backed by a temp wiki root.

    Mirrors the record/URI shapes of ``get_opas`` / ``get_ops`` / ``get_opls``
    / ``read_resource`` from ``xeno_adp_agentic.wiki.serve.opa``.
    """

    root_uri = "viking://resources/test_ns"
    _DIRS: ClassVar[dict[str, str]] = {"OPA": "OP/OpA", "OPS": "OP/OpS", "OPL": "OP/OpL"}

    def __init__(
        self,
        wiki_root: str,
        library_root: str,
        *,
        case_root: str | None = None,
        faultannotated_root: str | None = None,
        bom_root: str | None = None,
        build_logger: Any = None,
    ) -> None:
        """Create the fake, preparing the ``OP/`` directory tree."""
        del case_root, faultannotated_root, bom_root, build_logger
        self.root = Path(wiki_root)
        self.library = Path(library_root)
        for rel_dir in self._DIRS.values():
            (self.root / rel_dir).mkdir(parents=True, exist_ok=True)

    def write_ticket(
        self,
        kind: str,
        record_id: str,
        *,
        title: str,
        status: str = "pending",
        category: str = "",
    ) -> str:
        """Seed one ticket markdown file and return its URI."""
        rel_dir = Path(self._DIRS[kind])
        if category:
            rel_dir = rel_dir / category
        path = self.root / rel_dir
        path.mkdir(parents=True, exist_ok=True)
        target = path / f"{record_id}.md"
        category_line = f"category: {category}\n" if category else ""
        target.write_text(
            f"---\nid: {record_id}\ntitle: {title}\nstatus: {status}\n{category_line}---\n"
            f"# {title}\n\n正文内容。\n",
            encoding="utf-8",
        )
        return f"{self.root_uri}/{target.relative_to(self.root).as_posix()}"

    def update_ops(
        self,
        ops_id: str,
        *,
        title: str | None = None,
        analysis: str | None = None,
        solution: str | None = None,
        evidence_uris: list[str] | None = None,
        related_uris: list[str] | None = None,
        candidate_content: str | None = None,
        candidate_operations: list[dict[str, object]] | None = None,
        expected_sha256: str | None = None,
        status: str | None = None,
        reviewed_by: str = "",
        review_notes: str = "",
    ) -> dict[str, str]:
        """Record a patch call and return a shaped OPS result."""
        self.last_update_ops: dict[str, object] = {
            "ops_id": ops_id,
            "title": title,
            "analysis": analysis,
            "solution": solution,
            "evidence_uris": evidence_uris,
            "related_uris": related_uris,
            "candidate_content": candidate_content,
            "candidate_operations": candidate_operations,
            "expected_sha256": expected_sha256,
            "status": status,
            "reviewed_by": reviewed_by,
            "review_notes": review_notes,
        }
        return {
            "ops_id": ops_id,
            "uri": f"{self.root_uri}/OP/OpS/{ops_id}.md",
            "parent_opa": "opa-001",
            "target_uri": f"{self.root_uri}/Component/001",
            "status": status or "unconfirmed",
        }

    def _scan(self, rel_dir: str, recursive: bool) -> list[tuple[Path, str]]:
        base = self.root / rel_dir
        if not base.exists():
            return []
        pattern = "**/*.md" if recursive else "*.md"
        return [(p, p.read_text(encoding="utf-8")) for p in sorted(base.glob(pattern))]

    def get_opas(
        self,
        *,
        target_uri: str = "",
        status: str = "",
        category: str = "",
        limit: int = 50,
        **_: object,
    ) -> list[dict[str, str]]:
        """Return OPA records matching the filters, mirroring the real tool."""
        records: list[dict[str, str]] = []
        for path, content in self._scan("OP/OpA", recursive=True):
            fm = _parse_frontmatter(content)
            if status and fm.get("status", "") != status:
                continue
            if category and fm.get("category", "") != category:
                continue
            if target_uri and fm.get("target_uri", "") != target_uri:
                continue
            records.append(
                {
                    "opa_id": str(fm.get("id", path.stem)),
                    "title": str(fm.get("title", "")),
                    "status": str(fm.get("status", "pending")),
                    "category": str(fm.get("category", "")),
                    "uri": f"{self.root_uri}/{path.relative_to(self.root).as_posix()}",
                },
            )
            if len(records) >= limit:
                break
        return records

    def get_ops(
        self,
        *,
        parent_opa: str = "",
        status: str = "",
        limit: int = 50,
        **_: object,
    ) -> list[dict[str, str]]:
        """Return OPS records, mirroring the real tool."""
        del parent_opa
        records: list[dict[str, str]] = []
        for path, content in self._scan("OP/OpS", recursive=False):
            fm = _parse_frontmatter(content)
            if status and fm.get("status", "") != status:
                continue
            records.append(
                {
                    "ops_id": str(fm.get("id", path.stem)),
                    "title": str(fm.get("title", "")),
                    "status": str(fm.get("status", "unconfirmed")),
                    "uri": f"{self.root_uri}/{path.relative_to(self.root).as_posix()}",
                },
            )
            if len(records) >= limit:
                break
        return records

    def get_opls(
        self,
        *,
        parent_opa: str = "",
        status: str = "",
        limit: int = 50,
        **_: object,
    ) -> list[dict[str, str]]:
        """Return OPL records, mirroring the real tool."""
        del parent_opa
        records: list[dict[str, str]] = []
        for path, content in self._scan("OP/OpL", recursive=False):
            fm = _parse_frontmatter(content)
            if status and fm.get("status", "") != status:
                continue
            records.append(
                {
                    "opl_id": str(fm.get("id", path.stem)),
                    "title": str(fm.get("title", "")),
                    "status": str(fm.get("status", "unconfirmed")),
                    "uri": f"{self.root_uri}/{path.relative_to(self.root).as_posix()}",
                },
            )
            if len(records) >= limit:
                break
        return records

    def read_resource(self, uri: str, line_numbers: bool = False) -> str | None:
        """Read a ticket file by its ``viking://`` URI."""
        del line_numbers
        relative = uri.removeprefix(self.root_uri + "/")
        path = self.root / relative
        if path.exists() and path.is_file():
            return path.read_text(encoding="utf-8")
        return None


def _fake_xeno_modules() -> dict[str, types.ModuleType]:
    """Build the ``xeno_adp_agentic`` module chain exposing the fakes.

    ``_ensure_tools()`` imports ``WikiBuildLogger`` and ``WikiBuildTools``
    from these paths; the parent packages must be present so the leaf
    ``from ... import X`` statements resolve.
    """

    def _package(name: str) -> types.ModuleType:
        module = types.ModuleType(name)
        module.__path__ = []  # type: ignore[attr-defined]
        return module

    build_logger = types.ModuleType("xeno_adp_agentic.wiki.build.build_logger")
    build_logger.__dict__["WikiBuildLogger"] = type(  # type: ignore[attr-defined]
        "WikiBuildLogger",
        (),
        {"__init__": lambda self, log_dir: setattr(self, "log_dir", log_dir)},
    )
    build_tools = types.ModuleType("xeno_adp_agentic.wiki.serve.build_tools")
    build_tools.__dict__["WikiBuildTools"] = FakeWikiBuildTools  # type: ignore[attr-defined]
    return {
        "xeno_adp_agentic": _package("xeno_adp_agentic"),
        "xeno_adp_agentic.wiki": _package("xeno_adp_agentic.wiki"),
        "xeno_adp_agentic.wiki.build": _package("xeno_adp_agentic.wiki.build"),
        "xeno_adp_agentic.wiki.serve": _package("xeno_adp_agentic.wiki.serve"),
        "xeno_adp_agentic.wiki.build.build_logger": build_logger,
        "xeno_adp_agentic.wiki.serve.build_tools": build_tools,
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def wiki_cap(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> WikiBuildCapability:
    """Build a WikiBuildCapability whose lazy tools resolve to the fakes."""
    monkeypatch.setenv("WIKI_STORAGE_BACKEND", "local")
    cap = WikiBuildCapability(
        wiki_root=str(tmp_path / "wiki"),
        library_root=str(tmp_path / "library"),
    )
    cap._tools = FakeWikiBuildTools(
        wiki_root=str(tmp_path / "wiki"),
        library_root=str(tmp_path / "library"),
    )
    assert isinstance(cap.tools, FakeWikiBuildTools)
    return cap


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


def test_implements_resource_protocols(wiki_cap: WikiBuildCapability) -> None:
    """The capability satisfies both runtime-checkable resource protocols."""
    assert isinstance(wiki_cap, ResourceAccess)
    assert isinstance(wiki_cap, ResourceTemplateAccess)


# ---------------------------------------------------------------------------
# ResourceAccess
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_resources_returns_opa_ops_opl_entries(
    wiki_cap: WikiBuildCapability,
) -> None:
    """list_resources surfaces one entry per seeded ticket, all kinds."""
    tools = wiki_cap.tools
    assert isinstance(tools, FakeWikiBuildTools)
    tools.write_ticket("OPA", "opa-001", title="压力判据缺失", category="conflict")
    tools.write_ticket("OPA", "opa-002", title="散热器检查", category="gap")
    tools.write_ticket("OPS", "ops-001", title="补充压力上限说明", status="unconfirmed")
    tools.write_ticket("OPL", "opl-001", title="提交共轨压力提案", status="unconfirmed")

    entries = await wiki_cap.list_resources()

    assert len(entries) == 4
    by_uri = {e.uri: e for e in entries}
    opa_uri = "viking://resources/test_ns/OP/OpA/conflict/opa-001.md"
    assert opa_uri in by_uri
    opa = by_uri[opa_uri]
    # name is a short, display-width-safe label (title-first), not raw record id
    assert opa.name == "OPA 压力判据缺失"
    assert "status=pending" in opa.description
    assert "category=conflict" in opa.description
    assert opa.mime_type == "text/markdown"
    assert by_uri["viking://resources/test_ns/OP/OpS/ops-001.md"].name == "OPS 补充压力上限说明"
    assert by_uri["viking://resources/test_ns/OP/OpL/opl-001.md"].name == "OPL 提交共轨压力提案"


@pytest.mark.asyncio
async def test_list_resources_bounded_per_category(wiki_cap: WikiBuildCapability) -> None:
    """list_resources caps each ticket kind at 50 entries."""
    tools = wiki_cap.tools
    assert isinstance(tools, FakeWikiBuildTools)
    for index in range(60):
        tools.write_ticket("OPA", f"opa-{index:03d}", title=f"ticket {index}")

    entries = await wiki_cap.list_resources()

    assert len(entries) == 50


@pytest.mark.asyncio
async def test_read_resource_returns_ticket_content(
    wiki_cap: WikiBuildCapability,
) -> None:
    """read_resource returns a TextResourceContent for a seeded ticket URI."""
    tools = wiki_cap.tools
    assert isinstance(tools, FakeWikiBuildTools)
    uri = tools.write_ticket("OPA", "opa-001", title="压力判据缺失")

    contents = await wiki_cap.read_resource(uri)

    assert contents is not None
    assert len(contents) == 1
    item = contents[0]
    assert isinstance(item, TextResourceContent)
    assert item.uri == uri
    assert item.mime_type == "text/markdown"
    assert "压力判据缺失" in item.text


@pytest.mark.asyncio
async def test_read_resource_unknown_uri_returns_none(
    wiki_cap: WikiBuildCapability,
) -> None:
    """read_resource returns None for a URI that is not a ticket."""
    contents = await wiki_cap.read_resource(
        "viking://resources/test_ns/OP/OpA/does-not-exist.md",
    )
    assert contents is None


@pytest.mark.asyncio
async def test_resource_exists_true_and_false(wiki_cap: WikiBuildCapability) -> None:
    """resource_exists mirrors read_resource success."""
    tools = wiki_cap.tools
    assert isinstance(tools, FakeWikiBuildTools)
    uri = tools.write_ticket("OPS", "ops-001", title="提案", status="unconfirmed")

    assert await wiki_cap.resource_exists(uri) is True
    assert await wiki_cap.resource_exists("viking://resources/test_ns/OP/OpS/nope.md") is False


# ---------------------------------------------------------------------------
# ResourceTemplateAccess
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_resource_templates_declares_tree_templates(
    wiki_cap: WikiBuildCapability,
) -> None:
    """list_resource_templates declares OPA/OPS/OPL id templates."""
    templates = await wiki_cap.list_resource_templates()

    assert len(templates) == 3
    by_template = {t.uri_template: t for t in templates}
    assert by_template["viking://resources/{namespace}/OP/OpA/{id}"].title == "OPA ticket by id"
    assert by_template["viking://resources/{namespace}/OP/OpS/{id}"].title == "OPS ticket by id"
    assert by_template["viking://resources/{namespace}/OP/OpL/{id}"].title == "OPL ticket by id"
    assert all(t.mime_type == "text/markdown" for t in templates)


@pytest.mark.asyncio
async def test_complete_resource_template_returns_pending_opas(
    wiki_cap: WikiBuildCapability,
) -> None:
    """Completion on the OPA template suggests pending OPA ids, not resolved ones."""
    tools = wiki_cap.tools
    assert isinstance(tools, FakeWikiBuildTools)
    tools.write_ticket("OPA", "opa-101", title="共轨压力判据", status="pending")
    tools.write_ticket("OPA", "opa-102", title="已闭环问题", status="resolved")

    result = await wiki_cap.complete_resource_template(
        "viking://resources/{namespace}/OP/OpA/{id}",
        CompletionArgument("id", ""),
        context={"namespace": "test_ns"},
    )

    assert result.values == ["opa-101 共轨压力判据"]
    assert result.total == 1
    assert result.has_more is False


@pytest.mark.asyncio
async def test_complete_resource_template_filters_by_argument_value(
    wiki_cap: WikiBuildCapability,
) -> None:
    """Completion narrows suggestions by the current argument value."""
    tools = wiki_cap.tools
    assert isinstance(tools, FakeWikiBuildTools)
    tools.write_ticket("OPA", "opa-201", title="主泵压力异常", status="pending")
    tools.write_ticket("OPA", "opa-202", title="液压油温过高", status="pending")

    result = await wiki_cap.complete_resource_template(
        "viking://resources/{namespace}/OP/OpA/{id}",
        CompletionArgument("id", "202"),
    )

    assert result.values == ["opa-202 液压油温过高"]


@pytest.mark.asyncio
async def test_complete_resource_template_ops_and_opl_suggest_own_ids(
    wiki_cap: WikiBuildCapability,
) -> None:
    """OPS/OPL templates complete their own record ids."""
    tools = wiki_cap.tools
    assert isinstance(tools, FakeWikiBuildTools)
    tools.write_ticket("OPS", "ops-301", title="建议一", status="unconfirmed")
    tools.write_ticket("OPL", "opl-301", title="提案一", status="unconfirmed")

    ops_result = await wiki_cap.complete_resource_template(
        "viking://resources/{namespace}/OP/OpS/{id}",
        CompletionArgument("id", ""),
    )
    opl_result = await wiki_cap.complete_resource_template(
        "viking://resources/{namespace}/OP/OpL/{id}",
        CompletionArgument("id", ""),
    )

    assert ops_result.values == ["ops-301 建议一"]
    assert opl_result.values == ["opl-301 提案一"]


@pytest.mark.asyncio
async def test_complete_resource_template_unsupported_raises(
    wiki_cap: WikiBuildCapability,
) -> None:
    """Completion raises NotImplementedError for unknown templates/arguments."""
    with pytest.raises(NotImplementedError):
        await wiki_cap.complete_resource_template(
            "viking://resources/{namespace}/something/{id}",
            CompletionArgument("id", ""),
        )
    with pytest.raises(NotImplementedError):
        await wiki_cap.complete_resource_template(
            "viking://resources/{namespace}/OP/OpA/{id}",
            CompletionArgument("status", ""),
        )


@pytest.mark.asyncio
async def test_update_ops_ticket_patches_in_place(
    wiki_cap: WikiBuildCapability,
) -> None:
    """update_ops_ticket is a patch: only passed fields reach engines.update_ops.

    Untouched fields (title/solution/evidence/…) are forwarded as ``None``
    so the engine preserves their current values, and the ops id is derived
    from the given ``viking://`` URI.
    """
    from wolfharness.capabilities.viking.ticket import build_ticket_tools

    tools = wiki_cap.tools
    assert isinstance(tools, FakeWikiBuildTools)
    tools.write_ticket("OPA", "opa-001", title="主泵压力判据")
    ops_uri = tools.write_ticket("OPS", "ops-001", title="原始建议", status="unconfirmed")

    update_ops_ticket = next(
        fn for fn in build_ticket_tools(wiki_cap) if fn.__name__ == "update_ops_ticket"
    )

    result = await update_ops_ticket(
        object(),
        ops_uri=ops_uri,
        status="confirmed",
        reviewed_by="expert-7",
    )

    assert result["ops_id"] == "ops-001"
    assert result["status"] == "confirmed"
    assert tools.last_update_ops["ops_id"] == "ops-001"
    assert tools.last_update_ops["status"] == "confirmed"
    assert tools.last_update_ops["reviewed_by"] == "expert-7"
    # patch semantics — untouched fields stay None so the engine keeps them.
    assert tools.last_update_ops["title"] is None
    assert tools.last_update_ops["solution"] is None
    assert tools.last_update_ops["candidate_operations"] is None
