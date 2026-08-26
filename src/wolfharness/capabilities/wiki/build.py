"""WikiBuildCapability — wiki knowledge construction tools for WolfHarness.

Exposes the wiki construction toolkit (entity materialization, OPA
records, build lifecycle, external expert OP flow) as a pydantic-ai
capability so agents can drive manual → wiki entity builds in-process
instead of through an external MCP subprocess.

Tool functions live in :mod:`wolfharness.capabilities.wiki.tools`
(following the same pattern as :mod:`wolfharness.capabilities.viking.tools`).
The capability class handles configuration, lazy ``WikiBuildTools``
creation, role-based filtering, and first-turn index injection.
"""

from __future__ import annotations

import asyncio
import copy
import os
from pathlib import Path
import re
from typing import TYPE_CHECKING, Any, ClassVar

import logfire
from pydantic import BaseModel
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.toolsets import AgentToolset, FunctionToolset

from wolfharness.capabilities.resource_protocols import (
    BlobResourceContent,
    CompletionArgument,
    CompletionResult,
    ResourceAccess,
    ResourceEntry,
    ResourceTemplateAccess,
    ResourceTemplateEntry,
    TextResourceContent,
)

# Re-export tool inventory and role types from wiki_build_tools so existing
# imports from ``wiki_build`` keep working.
from wolfharness.capabilities.wiki.tools import (  # noqa: F401
    ALL_WIKI_TOOLS,
    ROLE_TOOLS,
    WIKI_AGENT_ROLES,
    RoleFilter,
    _build_method_wrappers as _build_tool_fns,
)
from wolfharness.log import get_logger
from wolfharness.tools.base import FunctionTool


if TYPE_CHECKING:
    from collections.abc import Sequence

    from pydantic_ai.models import ModelRequestContext
    from pydantic_ai.tools import RunContext

logger = get_logger(__name__)


def _is_viking_backend() -> bool:
    return os.environ.get("WIKI_STORAGE_BACKEND", "viking") == "viking"


class WikiBuildConfig(BaseModel):
    """Configuration for :class:`WikiBuildCapability`.

    All roots default to ``None`` and resolve from the corresponding
    environment variables at tool-creation time.
    """

    model_config = {"arbitrary_types_allowed": True}

    wiki_root: str | None = None
    library_root: str | None = None
    case_root: str | None = None
    faultannotated_root: str | None = None
    bom_root: str | None = None
    tool_names: tuple[str, ...] = ()
    include_helpers: bool = True
    role: str | None = None
    index_enabled: bool = False
    index_max_tokens: int = 1000
    index_limit: int = 20
    build_log_dir: str | None = None
    sync_after_apply: bool = False
    """When True, push patched wiki page to remote Viking after
    applying a knowledge proposal (OPL).  Honoured by the ticket
    ``apply_opl_ticket`` tool; intended for ``local_viking`` storage
    mode."""


class WikiBuildCapability(
    AbstractCapability[Any],
    ResourceAccess,
    ResourceTemplateAccess,
):
    """Capability exposing wiki construction tools to an agent.

    Config fields mirror :class:`WikiBuildConfig` as constructor kwargs
    so the entry-point ``build()`` path (``cls(**args)``) can construct
    the capability directly from YAML ``args``.

    Implements the ``ResourceAccess`` and ``ResourceTemplateAccess``
    protocols so OPA/OPS/OPL wiki tickets surface as MCP resources —
    they show up in resource listings and support ``@``-completion in
    external agents (``ExtensionRegistry.get_resource_access``,
    ``opencode_server`` resource endpoint).
    """

    _FIRST_TURN_MAX_MESSAGES = 2

    #: URI template per ticket kind — matches the ``OP/`` storage layout.
    _OP_TICKET_TEMPLATES: ClassVar[dict[str, str]] = {
        "OPA": "viking://resources/{namespace}/OP/OpA/{id}",
        "OPS": "viking://resources/{namespace}/OP/OpS/{id}",
        "OPL": "viking://resources/{namespace}/OP/OpL/{id}",
    }

    _OP_TEMPLATE_RE = re.compile(
        r"^viking://resources/\{namespace\}/OP/(OpA|OpS|OpL)/\{id\}$",
    )

    _LIST_RESOURCE_LIMIT = 50
    _COMPLETION_LIMIT = 200

    def __init__(
        self,
        config: WikiBuildConfig | None = None,
        *,
        wiki_root: str | None = None,
        library_root: str | None = None,
        case_root: str | None = None,
        faultannotated_root: str | None = None,
        bom_root: str | None = None,
        include_helpers: bool = True,
        role: str | None = None,
        index_enabled: bool = False,
        index_max_tokens: int = 1000,
        index_limit: int = 20,
        build_log_dir: str | None = None,
        sync_after_apply: bool = False,
        tool_names: tuple[str, ...] = (),
    ) -> None:
        self._config = config or WikiBuildConfig(
            wiki_root=wiki_root,
            library_root=library_root,
            case_root=case_root,
            faultannotated_root=faultannotated_root,
            bom_root=bom_root,
            include_helpers=include_helpers,
            role=role,
            index_enabled=index_enabled,
            index_max_tokens=index_max_tokens,
            index_limit=index_limit,
            build_log_dir=build_log_dir,
            sync_after_apply=sync_after_apply,
            tool_names=tool_names,
        )
        self._tools: Any | None = None
        self._build_logger: Any | None = None
        self._tool_fns: list[Any] = []
        self._index_injected: bool = False

    @property
    def config(self) -> WikiBuildConfig:
        return self._config

    @property
    def tools(self) -> Any | None:
        return self._tools

    def _ensure_tools(self) -> None:
        """Lazily create the ticket tools instance.

        Prefers the full ``WikiBuildTools`` from ``xeno_adp_agentic`` when
        available (complete entity-write + apply path). Falls back to the
        standalone ``TicketEngine`` in ``wolfharness.wiki`` so the ticket
        capability works without the ``xeno_adp_agentic`` dependency.
        """
        if self._tools is not None:
            return

        wiki_root = self._config.wiki_root or os.environ.get("WIKI_ROOT") or "output/wiki_newbuild"

        try:
            from wolfharness.capabilities.wiki.build_logger import WikiBuildLogger

            from .wiki_build_tools import WikiBuildTools

            library_root = (
                self._config.library_root or os.environ.get("LIBRARY_ROOT") or "output/library"
            )
            fault_root = self._config.faultannotated_root or os.environ.get("FAULTANNOTATED_ROOT")
            log_dir = self._config.build_log_dir or os.environ.get("WIKI_BUILD_LOG_DIR")
            if not log_dir:
                log_dir = "logs" if "://" in str(wiki_root) else str(Path(wiki_root) / "logs")
            self._build_logger = WikiBuildLogger(log_dir)
            self._tools = WikiBuildTools(
                wiki_root,
                library_root,
                case_root=self._config.case_root or os.environ.get("CASE_ROOT"),
                faultannotated_root=fault_root,
                bom_root=self._config.bom_root or os.environ.get("WIKI_BOM_ROOT"),
                build_logger=self._build_logger,
            )
        except ImportError:
            from wolfharness.capabilities.wiki.storage import create_wiki_store
            from wolfharness.capabilities.wiki.tickets.ticket_engine import TicketEngine

            store = create_wiki_store(wiki_root)
            self._tools = TicketEngine(store)

    def get_instructions(self) -> str | None:
        from wolfharness.capabilities.wiki.tools import get_instructions

        return get_instructions(self._config.role)

    # ---- ResourceAccess ----

    async def list_resources(self) -> Sequence[ResourceEntry]:
        """List OPA/OPS/OPL wiki tickets as MCP resources.

        Returns:
            Sequence of ``ResourceEntry`` descriptors — at most 50 per
            ticket kind.
        """
        self._ensure_tools()
        assert self._tools is not None
        opas = await asyncio.to_thread(self._tools.get_opas, limit=self._LIST_RESOURCE_LIMIT)
        ops = await asyncio.to_thread(self._tools.get_ops, limit=self._LIST_RESOURCE_LIMIT)
        opls = await asyncio.to_thread(self._tools.get_opls, limit=self._LIST_RESOURCE_LIMIT)
        return [
            *[self._ticket_entry("OPA", record) for record in opas],
            *[self._ticket_entry("OPS", record) for record in ops],
            *[self._ticket_entry("OPL", record) for record in opls],
        ]

    async def read_resource(
        self, uri: str
    ) -> list[TextResourceContent | BlobResourceContent] | None:
        """Read a wiki ticket by its ``viking://`` URI.

        Delegates to ``WikiBuildTools.read_resource``, which resolves the
        ``OP/`` subtree under the active wiki store.

        Args:
            uri: Resource URI to read (e.g. ``viking://resources/<ns>/OP/OpA/<id>.md``).

        Returns:
            List containing the ticket markdown as ``TextResourceContent``,
            or ``None`` when the URI is not an existing ticket.
        """
        self._ensure_tools()
        assert self._tools is not None
        content = await asyncio.to_thread(self._tools.read_resource, uri)
        if content is None:
            return None
        return [TextResourceContent(uri=uri, mime_type="text/markdown", text=content)]

    async def resource_exists(self, uri: str) -> bool:
        """Check whether a wiki ticket resource exists.

        Args:
            uri: Resource URI to check.

        Returns:
            ``True`` when the ticket is readable, ``False`` otherwise.
        """
        contents = await self.read_resource(uri)
        return contents is not None

    # ---- ResourceTemplateAccess ----

    async def list_resource_templates(self) -> Sequence[ResourceTemplateEntry]:
        """Declare the OPA/OPS/OPL ticket URI templates.

        Returns:
            Sequence of ``ResourceTemplateEntry`` descriptors, one per
            ticket kind.
        """
        return [
            ResourceTemplateEntry(
                uri_template=template,
                name="Wiki tickets",
                title=f"{kind} ticket by id",
                description=(
                    f"{kind} wiki construction ticket under a namespace's "
                    f"OP/{subdir}/ tree ({kind_label})."
                ),
                mime_type="text/markdown",
            )
            for template, (kind, subdir, kind_label) in (
                (self._OP_TICKET_TEMPLATES["OPA"], ("OPA", "OpA", "working problem analysis")),
                (self._OP_TICKET_TEMPLATES["OPS"], ("OPS", "OpS", "solution suggestion")),
                (self._OP_TICKET_TEMPLATES["OPL"], ("OPL", "OpL", "knowledge proposal link")),
            )
        ]

    async def complete_resource_template(
        self,
        uri_template: str,
        argument: CompletionArgument,
        context: dict[str, str] | None = None,
    ) -> CompletionResult:
        """Suggest ticket ids for a resource template's ``{id}`` argument.

        Args:
            uri_template: The URI template to complete.
            argument: The argument being completed — must be ``id``.
            context: Optional context arguments (may carry ``namespace``).

        Returns:
            ``CompletionResult`` with matching ``<id> <title>`` suggestions.

        Raises:
            NotImplementedError: If the template is unsupported or the
                argument is not ``id``.
        """
        kind = self._template_kind(uri_template)
        if kind is None or argument.name != "id":
            raise NotImplementedError(
                f"Completion not supported for template {uri_template!r}",
            )
        self._ensure_tools()
        assert self._tools is not None
        namespace = (context or {}).get("namespace", "")
        if namespace and not namespace.startswith("viking://"):
            namespace = f"viking://resources/{namespace}/"
        if kind == "OPA":
            rows = await asyncio.to_thread(
                self._tools.get_opas,
                status="pending",
                limit=self._COMPLETION_LIMIT,
            )
            id_key = "opa_id"
        elif kind == "OPS":
            rows = await asyncio.to_thread(self._tools.get_ops, limit=self._COMPLETION_LIMIT)
            id_key = "ops_id"
        else:
            rows = await asyncio.to_thread(self._tools.get_opls, limit=self._COMPLETION_LIMIT)
            id_key = "opl_id"
        needle = argument.value.strip().lower()
        values: list[str] = []
        for record in rows:
            record_uri = str(record.get("uri", ""))
            if namespace and not record_uri.startswith(namespace):
                continue
            record_id = str(record.get(id_key, ""))
            title = str(record.get("title", ""))
            if needle and needle not in record_id.lower() and needle not in title.lower():
                continue
            values.append(f"{record_id} {title}".strip())
        return CompletionResult(values=values, total=len(values), has_more=False)

    # ---- Helpers ----

    @classmethod
    def _template_kind(cls, uri_template: str) -> str | None:
        """Map a supported ticket template to its kind (OPA/OPS/OPL).

        Args:
            uri_template: The URI template to classify.

        Returns:
            ``"OPA"`` / ``"OPS"`` / ``"OPL"``, or ``None`` when unsupported.
        """
        normalized = uri_template.replace("${", "{")
        match = cls._OP_TEMPLATE_RE.match(normalized)
        if match is None:
            return None
        return {"OpA": "OPA", "OpS": "OPS", "OpL": "OPL"}[match.group(1)]

    @staticmethod
    def _ticket_entry(kind: str, record: dict[str, object]) -> ResourceEntry:
        """Map one OPA/OPS/OPL record dict to a resource descriptor.

        Args:
            kind: Ticket kind — ``"OPA"``, ``"OPS"`` or ``"OPL"``.
            record: Record dict as returned by the ``get_*`` tools.

        Returns:
            The corresponding ``ResourceEntry``.
        """
        record_id = str(record.get(f"{kind.lower()}_id", ""))
        title = str(record.get("title", ""))
        name = f"{kind} {record_id}: {title}" if title else f"{kind} {record_id}"
        detail = f"status={record.get('status', '')}"
        category = str(record.get("category", ""))
        if category:
            detail = f"{detail}, category={category}"
        return ResourceEntry(
            uri=str(record.get("uri", "")),
            name=name,
            description=detail,
            mime_type="text/markdown",
        )

    def get_toolset(self) -> AgentToolset[Any] | None:
        from wolfharness.capabilities.wiki.tools import (
            build_tools,
        )

        tool_fns = build_tools(self)
        if not tool_fns:
            return None
        toolset: FunctionToolset[Any] = FunctionToolset(tool_fns, id="wiki_build")
        if self._config.role is not None:
            return RoleFilter(self._config.role).get_wrapper_toolset(toolset)
        return toolset

    async def get_tools(self) -> Sequence[FunctionTool]:
        self._ensure_tools()
        return [FunctionTool.from_callable(fn) for fn in self._tool_fns]

    def _resolve_index_roots(self) -> list[tuple[str, str]]:
        candidates: list[tuple[str, str | None]] = [
            ("wiki", self._config.wiki_root or os.environ.get("WIKI_ROOT")),
            ("raw", self._config.library_root or os.environ.get("LIBRARY_ROOT")),
            ("bom", self._config.bom_root or os.environ.get("WIKI_BOM_ROOT")),
        ]
        resolved: list[tuple[str, str]] = []
        for label, uri in candidates:
            if uri is None:
                continue
            if uri.startswith("viking://"):
                resolved.append((label, uri))
            elif _is_viking_backend() and label == "wiki" and os.environ.get("VIKING_NAMESPACE"):
                resolved.append(
                    (label, f"viking://resources/{os.environ['VIKING_NAMESPACE']}"),
                )
            elif _is_viking_backend() and label == "raw" and os.environ.get("VIKING_RAW_NAMESPACE"):
                resolved.append(
                    (label, f"viking://resources/{os.environ['VIKING_RAW_NAMESPACE']}"),
                )
        return resolved

    async def before_model_request(
        self,
        ctx: RunContext[Any],
        request_context: ModelRequestContext,
    ) -> ModelRequestContext:
        if not self._config.index_enabled or self._index_injected:
            return request_context

        if len(request_context.messages) > self._FIRST_TURN_MAX_MESSAGES:
            self._index_injected = True
            return request_context

        self._index_injected = True

        with logfire.span("wiki_build.before_model_request"):
            try:
                from wolfharness.capabilities.wiki.tickets.index import (
                    _format_index_block,
                )

                index_block = _format_index_block(
                    self._resolve_index_roots(),
                    max_tokens=self._config.index_max_tokens,
                    limit=self._config.index_limit,
                )
                if not index_block.strip():
                    return request_context

                from dataclasses import replace

                from pydantic_ai.messages import (
                    ModelRequest,
                    UserPromptPart,
                )

                messages = list(request_context.messages)
                insert_idx: int | None = None
                for i in range(len(messages) - 1, -1, -1):
                    msg = messages[i]
                    if isinstance(msg, ModelRequest) and any(
                        isinstance(p, UserPromptPart) for p in msg.parts
                    ):
                        insert_idx = i
                        break

                if insert_idx is None:
                    return request_context

                system_msg = ModelRequest(parts=[UserPromptPart(content=index_block)])
                new_messages = [*messages[:insert_idx], system_msg, *messages[insert_idx:]]
                return replace(request_context, messages=new_messages)
            except Exception:
                logger.warning("Wiki index injection failed", exc_info=True)
                return request_context

    async def for_run(self, ctx: RunContext[Any]) -> WikiBuildCapability:
        run_copy = copy.copy(self)
        run_copy._index_injected = False
        return run_copy
