"""WikiBuildCapability — wiki knowledge construction tools for WolfHarness.

Exposes the wiki construction toolkit (entity materialization, OPA
records, build lifecycle, external expert OP flow) as a pydantic-ai
capability so agents can drive manual → wiki entity builds in-process
instead of through an external MCP subprocess.

Tool functions live in :mod:`wolfharness.capabilities.viking.wiki_build_tools`
(following the same pattern as :mod:`wolfharness.capabilities.viking.tools`).
The capability class handles configuration, lazy ``WikiBuildTools``
creation, role-based filtering, and first-turn index injection.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import logfire
from pydantic import BaseModel
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.toolsets import AgentToolset, FunctionToolset

from wolfharness.log import get_logger

# Re-export tool inventory and role types from wiki_build_tools so existing
# imports from ``wiki_build`` keep working.
from wolfharness.capabilities.viking.wiki_build_tools import (  # noqa: F401
    ALL_WIKI_TOOLS,
    ROLE_TOOLS,
    RoleFilter,
    WIKI_AGENT_ROLES,
    _build_method_wrappers as _build_tool_fns,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pydantic_ai.models import ModelRequestContext
    from pydantic_ai.tools import RunContext

    from wolfharness.tools.base import FunctionTool

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
    ``apply_external_opl``. Intended for ``local_viking`` storage mode."""
    include_external_ops: bool = False
    """When True, supplement the role's method-wrapper tools with the
    three external OP closures. Automatically True for
    ``wiki_external_expert`` role."""


class WikiBuildCapability(AbstractCapability[Any]):
    """Capability exposing wiki construction tools to an agent.

    Config fields mirror :class:`WikiBuildConfig` as constructor kwargs
    so the entry-point ``build()`` path (``cls(**args)``) can construct
    the capability directly from YAML ``args``.
    """

    _FIRST_TURN_MAX_MESSAGES = 2

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
        include_external_ops: bool = False,
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
            include_external_ops=include_external_ops,
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
        """Lazily create the host ``WikiBuildTools`` instance."""
        if self._tools is not None:
            return

        from xeno_adp_agentic.wiki.build.build_logger import WikiBuildLogger
        from xeno_adp_agentic.wiki.serve.build_tools import WikiBuildTools

        wiki_root = self._config.wiki_root or os.environ.get("WIKI_ROOT") or "output/wiki_newbuild"
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

    def get_instructions(self) -> str | None:
        from wolfharness.capabilities.viking.wiki_build_tools import get_instructions

        return get_instructions(self._config.role)

    def get_toolset(self) -> AgentToolset[Any] | None:
        from wolfharness.capabilities.viking.wiki_build_tools import (
            RoleFilter,
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
                from wolfharness.capabilities.viking.wiki_index import (
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
                    SystemPromptPart,
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

                system_msg = ModelRequest(parts=[SystemPromptPart(content=index_block)])
                new_messages = [*messages[:insert_idx], system_msg, *messages[insert_idx:]]
                return replace(request_context, messages=new_messages)
            except Exception:
                logger.warning("Wiki index injection failed", exc_info=True)
                return request_context

    async def for_run(self, ctx: RunContext[Any]) -> WikiBuildCapability:
        run_copy = copy.copy(self)
        run_copy._index_injected = False
        return run_copy
