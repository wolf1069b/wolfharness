"""ResourceCapability — unified resource access via 5 agent-facing tools.

Provides a single ``AbstractCapability`` that aggregates resource access
across all visible ``ResourceAccess``, ``SkillResource``, and
``ResourceTemplateAccess`` providers registered in the
``ExtensionRegistry``. The capability is stateless — it reads
``ctx.deps`` (an ``AgentContextDeps``) at runtime to resolve providers.

Tools exposed:
    - ``list_resources``: Aggregate resources from MCP + skills
    - ``read_resource``: Read content by URI (skill:// or other)
    - ``resource_exists``: Check if a resource exists
    - ``list_resource_templates``: List URI templates for dynamic discovery
    - ``complete_resource_template``: Get completion suggestions for template params
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Annotated

import logfire
from pydantic import Field
from pydantic_ai import ToolReturn
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai.toolsets import AgentToolset, FunctionToolset

from wolfharness.capabilities.extension_registry import Scope, ScopeLevel
from wolfharness.capabilities.resource_protocols import (
    CompletionArgument,
    CompletionResult,
    ResourceEntry,
    ResourceTemplateEntry,
)


if TYPE_CHECKING:
    from wolfharness.capabilities.agent_context import AgentContextDeps


# Number of header lines (header + separator) before data rows.
_HEADER_LINE_COUNT = 2

# Default pagination limits.
_DEFAULT_LIST_LIMIT = 50
_DEFAULT_READ_TEXT_LIMIT = 10_000
_MAX_COMPLETION_SUGGESTIONS = 100

# Max wall-clock time a single provider may take to answer a listing request.
# Providers that time out are skipped with a warning instead of blocking the
# whole listing.
_LIST_PROVIDER_TIMEOUT = 10.0  # seconds


class ResourceCapability(AbstractCapability[AgentDepsT]):
    """Unified resource access capability providing 5 agent-facing tools.

    Aggregates resources from all visible providers (MCP servers, local
    skills) via the ``ExtensionRegistry`` on ``AgentContextDeps``. The
    capability is stateless — no resources are held between turns.

    Tools route by URI scheme to ``ResourceAccess`` providers. ``skill://``
    resolution is retained as a silent, non-advertised fallback (see
    ``resource_resolver.resolve_resource_content``) for protocol consumers.
    """

    def __init__(self, *, toolset_id: str = "resource_access") -> None:
        """Initialize the resource capability.

        Args:
            toolset_id: Identifier for the produced ``FunctionToolset``.
        """
        self._toolset_id = toolset_id

    @property
    def name(self) -> str:
        """Return the capability name."""
        return "resource_capability"

    async def __aenter__(self) -> ResourceCapability[AgentDepsT]:
        """Enter async context — no-op (stateless capability)."""
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        """Exit async context — no-op (stateless capability)."""

    def get_instructions(self) -> str | None:
        """Return brief system prompt instructions about resource tools.

        Returns:
            A short instruction string describing available resource
            management tools and supported URI schemes.
        """
        return (
            "You have access to resource management tools:\n"
            "- list_resources: List available resources from connected MCP "
            "servers and local files (paginated, use offset to page through)\n"
            "- read_resource: Read content from a resource URI (e.g. mcp://, "
            "file://, viking://, optionally http(s) for servers that use web "
            "URIs; web pages are usually better read with a web fetch tool)\n"
            "- resource_exists: Check if a resource exists\n"
            "- list_resource_templates: List URI templates for dynamic "
            "resource discovery (paginated, use offset to page through)\n"
            "- complete_resource_template: Get completion suggestions for "
            "template parameters\n\n"
            "URI schemes: mcp:// for MCP server resources, file:// for "
            "file-based resources"
        )

    @logfire.instrument("capability.resource_capability.get_toolset")
    def get_toolset(self) -> AgentToolset[AgentDepsT] | None:
        """Return a ``FunctionToolset`` with all 5 resource tools.

        The tools access ``ctx.deps`` at runtime, which must be an
        ``AgentContextDeps`` with an ``extension_registry`` field.
        """
        return FunctionToolset(
            [
                self.list_resources,
                self.read_resource,
                self.resource_exists,
                self.list_resource_templates,
                self.complete_resource_template,
            ],
            id=self._toolset_id,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_agent_context(ctx: RunContext[AgentDepsT]) -> AgentContextDeps:
        """Extract the ``AgentContextDeps`` from the run context deps.

        Delegates to the shared ``resolve_agent_context_from_deps`` utility
        which handles both the production path (``RuntimeAgentContext.data``)
        and the test path (direct ``AgentContextDeps``).

        Args:
            ctx: The pydantic-ai run context.

        Returns:
            The ``AgentContextDeps`` instance from ``ctx.deps`` (or ``ctx.deps.data``).

        Raises:
            RuntimeError: If deps is None or AgentContextDeps is not found.
        """
        from wolfharness.capabilities.agent_context import resolve_agent_context_from_deps

        return resolve_agent_context_from_deps(ctx.deps, capability_name="ResourceCapability")

    @staticmethod
    def _make_scope(agent_ctx: AgentContextDeps) -> Scope:
        """Build a ``Scope`` from ``AgentContextDeps`` fields.

        Uses SESSION level to get the complete view (POOL + AGENT + SESSION).

        Args:
            agent_ctx: The per-turn agent context.

        Returns:
            A ``Scope`` at SESSION level with agent and session identifiers.
        """
        session_id = agent_ctx.session.session_id if agent_ctx.session else ""
        return Scope(
            level=ScopeLevel.SESSION,
            agent_name=agent_ctx.agent_name,
            session_id=session_id,
        )

    @staticmethod
    def _extract_skill_name(uri: str) -> str:
        """Extract the skill name from a ``skill://`` URI.

        Takes the first path segment after ``skill://``.

        Args:
            uri: A ``skill://`` URI.

        Returns:
            The skill name (first path segment).
        """
        path = uri[len("skill://") :]
        return path.split("/")[0] if path else ""

    # ------------------------------------------------------------------
    # Tool implementations
    # ------------------------------------------------------------------

    @logfire.instrument("capability.resource_capability.list_resources")
    async def list_resources(
        self,
        ctx: RunContext[AgentDepsT],
        limit: Annotated[
            int,
            Field(description="Maximum number of resources to return (default: 50)"),
        ] = _DEFAULT_LIST_LIMIT,
        offset: Annotated[
            int,
            Field(description="Number of resources to skip for pagination"),
        ] = 0,
    ) -> str:
        """List available resources from connected MCP servers and local files.

        Results are paginated. Use ``offset`` to page through large result sets.

        Args:
            ctx: The run context providing agent dependencies.
            limit: Maximum number of resources to return.
            offset: Number of resources to skip.
        """
        agent_ctx = self._resolve_agent_context(ctx)
        registry = agent_ctx.extension_registry
        if registry is None:
            return "No resources available."

        scope = self._make_scope(agent_ctx)
        provider_caps = registry.get_resource_access(scope)
        if not provider_caps:
            return "No resources available."

        # Query all providers concurrently; a slow or unreachable provider
        # must not block the rest. Per-provider timeout bounds the total wait.
        gathered = await asyncio.gather(
            *[
                asyncio.wait_for(cap.list_resources(), timeout=_LIST_PROVIDER_TIMEOUT)
                for cap in provider_caps
            ],
            return_exceptions=True,
        )

        source_entries: list[tuple[str, ResourceEntry]] = []
        for cap, result in zip(provider_caps, gathered, strict=True):
            if isinstance(result, BaseException):
                logfire.warning(
                    "Failed to list resources from {source}",
                    source=type(cap).__name__,
                )
                continue
            source = type(cap).__name__
            source_entries.extend((source, entry) for entry in result)

        total = len(source_entries)
        page = source_entries[offset : offset + limit]

        if not page:
            if offset > 0:
                return f"No resources at offset {offset}. Total: {total} resource(s)."
            return "No resources available."

        header = f"{'Source':<25} {'URI':<45} {'Name':<20} {'Description':<30} {'MIME Type':<15}"
        lines = [header, "-" * len(header)]
        lines.extend(
            f"{source:<25} {entry.uri:<45} {entry.name:<20} "
            f"{entry.description:<30} {entry.mime_type:<15}"
            for source, entry in page
        )

        remaining = total - offset - len(page)
        if remaining > 0:
            lines.append(
                f"\n... {remaining} more resources. "
                f"Call list_resources with offset={offset + len(page)} to see more."
            )

        return "\n".join(lines)

    @logfire.instrument("capability.resource_capability.read_resource")
    async def read_resource(
        self,
        ctx: RunContext[AgentDepsT],
        uri: Annotated[
            str,
            Field(
                description=(
                    "Resource URI to read, e.g. 'mcp://server/resource' or 'file://path/to/file'"
                )
            ),
        ],
    ) -> ToolReturn:
        """Read content from a resource by URI.

        Supports text and binary content. Routes by URI scheme; ``skill://``
        URIs are additionally resolved as a non-advertised fallback via
        ``resource_resolver.resolve_resource_content``.

        Args:
            ctx: The run context providing agent dependencies.
            uri: Resource URI to read.
        """
        from wolfharness.capabilities.resource_resolver import resolve_resource_content

        agent_ctx = self._resolve_agent_context(ctx)
        registry = agent_ctx.extension_registry
        if registry is None:
            return ToolReturn(return_value=f"Resource not found: {uri}")

        scope = self._make_scope(agent_ctx)
        resource_caps = registry.get_resource_access(scope)
        skill_caps = registry.get_skill_resources(scope)

        content = await resolve_resource_content(uri, resource_caps, skill_caps)
        if content is None:
            return ToolReturn(return_value=f"Resource not found: {uri}")

        # Bridge list[UserContent] → ToolReturn
        text_parts = [p for p in content if isinstance(p, str)]
        return_value = "\n".join(text_parts) if text_parts else ""
        return ToolReturn(return_value=return_value, content=content)

    @logfire.instrument("capability.resource_capability.resource_exists")
    async def resource_exists(
        self,
        ctx: RunContext[AgentDepsT],
        uri: Annotated[str, Field(description="Resource URI to check")],
    ) -> bool:
        """Check if a resource exists.

        Routes by URI scheme:
        ``skill://`` → skill providers, other URIs → resource providers.

        Args:
            ctx: The run context providing agent dependencies.
            uri: Resource URI to check.

        Returns:
            True if any provider has the resource, False otherwise.
        """
        agent_ctx = self._resolve_agent_context(ctx)
        registry = agent_ctx.extension_registry
        if registry is None:
            return False

        scope = self._make_scope(agent_ctx)

        if uri.startswith("skill://"):
            from wolfharness.skills.uri_resolver import ResolvedSkillURI

            resolved = ResolvedSkillURI.parse(uri)
            skill_name = resolved.skill_name

            # If URI contains a reference path, check if the reference file exists.
            # Exception: "SKILL.md" (case-insensitive) is the skill's main file —
            # use skill_exists() for backward compatibility and virtual skill support.
            if (
                resolved.reference_path is not None
                and resolved.reference_path.upper() != "SKILL.MD"
            ):
                from wolfharness.capabilities.resource_resolver import _resolve_skill_reference

                try:
                    ref_content = await _resolve_skill_reference(
                        registry.get_skill_resources(scope),
                        skill_name,
                        resolved.reference_path,
                    )
                    return ref_content is not None
                except Exception:  # noqa: BLE001
                    return False

            # No reference path — check if the skill itself exists
            for skill_cap in registry.get_skill_resources(scope):
                try:
                    if await skill_cap.skill_exists(skill_name):
                        return True
                except Exception:  # noqa: BLE001
                    continue
            return False

        for resource_cap in registry.get_resource_access(scope):
            try:
                if await resource_cap.resource_exists(uri):
                    return True
            except Exception:  # noqa: BLE001
                continue
        return False

    @logfire.instrument("capability.resource_capability.list_resource_templates")
    async def list_resource_templates(
        self,
        ctx: RunContext[AgentDepsT],
        limit: Annotated[
            int,
            Field(description="Maximum number of templates to return (default: 50)"),
        ] = _DEFAULT_LIST_LIMIT,
        offset: Annotated[
            int,
            Field(description="Number of templates to skip for pagination"),
        ] = 0,
    ) -> str:
        """List URI templates for dynamic resource discovery.

        Results are paginated. Use ``offset`` to page through large result sets.

        Args:
            ctx: The run context providing agent dependencies.
            limit: Maximum number of templates to return.
            offset: Number of templates to skip.
        """
        agent_ctx = self._resolve_agent_context(ctx)
        registry = agent_ctx.extension_registry
        if registry is None:
            return "No resource templates available."

        scope = self._make_scope(agent_ctx)
        provider_caps = registry.get_resource_template_access(scope)
        if not provider_caps:
            return "No resource templates available."

        gathered = await asyncio.gather(
            *[
                asyncio.wait_for(cap.list_resource_templates(), timeout=_LIST_PROVIDER_TIMEOUT)
                for cap in provider_caps
            ],
            return_exceptions=True,
        )

        source_entries: list[tuple[str, ResourceTemplateEntry]] = []
        for cap, result in zip(provider_caps, gathered, strict=True):
            if isinstance(result, BaseException):
                logfire.warning(
                    "Failed to list resource templates from {source}",
                    source=type(cap).__name__,
                )
                continue
            source = type(cap).__name__
            source_entries.extend((source, entry) for entry in result)

        total = len(source_entries)
        page = source_entries[offset : offset + limit]

        if not page:
            if offset > 0:
                return f"No resource templates at offset {offset}. Total: {total} template(s)."
            return "No resource templates available."

        header = (
            f"{'Source':<25} {'URI Template':<40} {'Name':<20} "
            f"{'Title':<15} {'Description':<30} {'MIME Type':<15}"
        )
        lines = [header, "-" * len(header)]
        lines.extend(
            f"{source:<25} {entry.uri_template:<40} {entry.name:<20} "
            f"{entry.title:<15} {entry.description:<30} {entry.mime_type:<15}"
            for source, entry in page
        )

        remaining = total - offset - len(page)
        if remaining > 0:
            lines.append(
                f"\n... {remaining} more templates. "
                f"Call list_resource_templates with offset={offset + len(page)} to see more."
            )

        return "\n".join(lines)

    @logfire.instrument("capability.resource_capability.complete_resource_template")
    async def complete_resource_template(
        self,
        ctx: RunContext[AgentDepsT],
        uri_template: Annotated[str, Field(description="The URI template to complete")],
        argument_name: Annotated[str, Field(description="The parameter name being completed")],
        argument_value: Annotated[str, Field(description="The current value of the parameter")],
    ) -> str:
        """Get completion suggestions for a resource template parameter.

        Args:
            ctx: The run context providing agent dependencies.
            uri_template: The URI template to complete.
            argument_name: The parameter name being completed.
            argument_value: The current value of the parameter.
        """
        agent_ctx = self._resolve_agent_context(ctx)
        registry = agent_ctx.extension_registry
        if registry is None:
            return "No resource template providers available."

        scope = self._make_scope(agent_ctx)
        argument = CompletionArgument(name=argument_name, value=argument_value)

        for cap in registry.get_resource_template_access(scope):
            try:
                templates = await cap.list_resource_templates()
            except Exception:  # noqa: BLE001
                continue
            matching = any(t.uri_template == uri_template for t in templates)
            if not matching:
                continue
            try:
                result: CompletionResult = await cap.complete_resource_template(
                    uri_template,
                    argument,
                )
            except NotImplementedError:
                return f"Completion not supported for template: {uri_template}"
            return self._format_completion_result(result)

        return f"Completion not supported for template: {uri_template}"

    @staticmethod
    def _truncate_text(
        text: str,
        limit: int = _DEFAULT_READ_TEXT_LIMIT,
    ) -> str:
        """Truncate text content if it exceeds the limit.

        Args:
            text: The text to potentially truncate.
            limit: Maximum number of characters to keep.

        Returns:
            The original text if within limit, or a truncated version
            with a suffix indicating the total length.
        """
        if len(text) <= limit:
            return text
        return text[:limit] + f"\n\n... [truncated: {len(text)} chars total, showing first {limit}]"

    @staticmethod
    def _format_completion_result(result: CompletionResult) -> str:
        """Format a ``CompletionResult`` into a human-readable string.

        Args:
            result: The completion result to format.

        Returns:
            A formatted string with completion suggestions.
        """
        lines: list[str] = ["Completion suggestions:"]
        values = result.values[:_MAX_COMPLETION_SUGGESTIONS]
        lines.extend(f"  - {value}" for value in values)
        if len(result.values) > _MAX_COMPLETION_SUGGESTIONS:
            lines.append(
                f"  ... ({len(result.values)} total, showing first {_MAX_COMPLETION_SUGGESTIONS})"
            )
        elif result.has_more:
            lines.append(f"  ... ({result.total} total, more available)")
        elif result.total is not None and result.total > len(result.values):
            lines.append(f"  ... ({result.total} total)")
        return "\n".join(lines)


__all__ = ["ResourceCapability"]
