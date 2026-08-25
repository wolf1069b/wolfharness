"""ResourceCapability — unified MCP Resource access via three agent tools.

Provides a single ``AbstractCapability`` that aggregates resource access
across all visible ``ResourceAccess``, ``SkillResource``, and
``ResourceTemplateAccess`` providers registered in the
``ExtensionRegistry``. The capability is stateless — it reads
``ctx.deps`` (an ``AgentContextDeps``) at runtime to resolve providers.

Only the three ``list_mcp_*``/``read_mcp_resource`` methods are model-facing.
The older five methods remain below as internal compatibility helpers.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import json
from typing import TYPE_CHECKING, Annotated

import logfire
from pydantic import Field
from pydantic_ai import BinaryContent, ToolReturn
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai.toolsets import AgentToolset, FunctionToolset

from wolfharness.capabilities.extension_registry import Scope, ScopeLevel
from wolfharness.capabilities.resource_protocols import (
    BlobResourceContent,
    CompletionArgument,
    CompletionResult,
    McpResourceListResult,
    McpResourceReadResult,
    McpResourceTemplateListResult,
    ResourceEntry,
    ResourceError,
    ResourceTemplateEntry,
    TextResourceContent,
)


if TYPE_CHECKING:
    from wolfharness.capabilities.agent_context import AgentContextDeps
    from wolfharness.capabilities.resource_protocols import McpResourceProvider
    from wolfharness.common_types import JsonObject


# Number of header lines (header + separator) before data rows.
_HEADER_LINE_COUNT = 2

# Default pagination limits.
_DEFAULT_LIST_LIMIT = 50
_DEFAULT_READ_TEXT_LIMIT = 10_000
_MAX_LIST_LIMIT = 100
_CURSOR_VERSION = 2
_MAX_COMPLETION_SUGGESTIONS = 100
_MAX_BLOB_BYTES = 10 * 1024 * 1024
_SUPPORTED_BLOB_MIME_TYPES = frozenset({
    "application/pdf",
    "image/gif",
    "image/jpeg",
    "image/png",
    "image/webp",
})

# Max wall-clock time a single provider may take to answer a listing request.
# Providers that time out are skipped with a warning instead of blocking the
# whole listing.
_LIST_PROVIDER_TIMEOUT = 10.0  # seconds


class ResourceCapability(AbstractCapability[AgentDepsT]):
    """Unified MCP Resource capability providing three agent-facing tools.

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
            "Use MCP Resource tools progressively: list_mcp_resources or "
            "list_mcp_resource_templates first, then read_mcp_resource with "
            "the exact server and opaque URI returned by the provider. "
            "Do not infer URI values from templates."
        )

    @logfire.instrument("capability.resource_capability.get_toolset")
    def get_toolset(self) -> AgentToolset[AgentDepsT] | None:
        """Return a ``FunctionToolset`` with the three MCP Resource tools.

        The tools access ``ctx.deps`` at runtime, which must be an
        ``AgentContextDeps`` with an ``extension_registry`` field.
        """
        return FunctionToolset(
            [
                self.list_mcp_resources,
                self.list_mcp_resource_templates,
                self.read_mcp_resource,
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

    @staticmethod
    def _encode_cursor(
        *,
        server: str | None,
        current_server: str | None,
        provider_index: int,
        upstream_cursor: str | None,
        offset: int,
    ) -> str:
        payload = {
            "version": _CURSOR_VERSION,
            "server": server,
            "current_server": current_server,
            "provider_index": provider_index,
            "upstream_cursor": upstream_cursor,
            "offset": offset,
        }
        raw = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
        return base64.urlsafe_b64encode(raw).decode().rstrip("=")

    @staticmethod
    def _decode_cursor(cursor: str) -> dict[str, object]:
        try:
            padding = "=" * (-len(cursor) % 4)
            value = json.loads(base64.urlsafe_b64decode((cursor + padding).encode()))
        except (
            ValueError,
            TypeError,
            binascii.Error,
            UnicodeDecodeError,
            json.JSONDecodeError,
        ) as exc:
            raise ValueError("cursor is not a valid host cursor") from exc
        if not isinstance(value, dict) or value.get("version") != _CURSOR_VERSION:
            raise ValueError("cursor version is unsupported")
        if not isinstance(value.get("provider_index"), int) or not isinstance(
            value.get("offset"), int
        ):
            raise TypeError("cursor position is invalid")
        if value["provider_index"] < 0 or value["offset"] < 0:
            raise ValueError("cursor position is invalid")
        if value.get("server") is not None and not isinstance(value.get("server"), str):
            raise ValueError("cursor server is invalid")
        if value.get("current_server") is not None and not isinstance(
            value.get("current_server"), str
        ):
            raise ValueError("cursor current server is invalid")
        if value.get("upstream_cursor") is not None and not isinstance(
            value.get("upstream_cursor"), str
        ):
            raise ValueError("cursor upstream value is invalid")
        return value

    @staticmethod
    def _cursor_state(value: dict[str, object]) -> tuple[int, str | None, int]:
        provider_index = value.get("provider_index")
        offset = value.get("offset")
        upstream_cursor = value.get("upstream_cursor")
        if not isinstance(provider_index, int) or not isinstance(offset, int):
            raise TypeError("cursor position is invalid")
        if upstream_cursor is not None and not isinstance(upstream_cursor, str):
            raise ValueError("cursor upstream value is invalid")
        return provider_index, upstream_cursor, offset

    @staticmethod
    def _resource_entry_dict(entry: ResourceEntry) -> dict[str, object]:
        return {
            "server": entry.server,
            "uri": entry.uri,
            "name": entry.name,
            "title": entry.title,
            "description": entry.description,
            "mime_type": entry.mime_type,
            "size": entry.size,
            "annotations": entry.annotations,
            "meta": entry.meta,
        }

    @staticmethod
    def _template_entry_dict(entry: ResourceTemplateEntry) -> dict[str, object]:
        return {
            "server": entry.server,
            "uri_template": entry.uri_template,
            "name": entry.name,
            "title": entry.title,
            "description": entry.description,
            "mime_type": entry.mime_type,
            "annotations": entry.annotations,
            "meta": entry.meta,
        }

    @staticmethod
    def _error_dict(error: ResourceError) -> dict[str, object]:
        return {
            "code": error.code,
            "message": error.message,
            "retryable": error.retryable,
            "suggestion": error.suggestion,
        }

    @staticmethod
    def _validate_limit(limit: int) -> None:
        if not 1 <= limit <= _MAX_LIST_LIMIT:
            raise ValueError("limit must be between 1 and 100")

    async def _mcp_providers(self, ctx: RunContext[AgentDepsT]) -> list[McpResourceProvider]:
        agent_ctx = self._resolve_agent_context(ctx)
        registry = agent_ctx.extension_registry
        if registry is None:
            return []
        from wolfharness.capabilities.resource_protocols import McpResourceProvider

        return sorted(
            (
                cap
                for cap in registry.get_visible_capabilities(self._make_scope(agent_ctx))
                if isinstance(cap, McpResourceProvider)
            ),
            key=lambda provider: provider.server_name,
        )

    @logfire.instrument("capability.resource_capability.list_mcp_resources")
    async def list_mcp_resources(  # noqa: PLR0915
        self,
        ctx: RunContext[AgentDepsT],
        server: Annotated[
            str | None, Field(description="Optional configured MCP server name")
        ] = None,
        cursor: Annotated[
            str | None, Field(description="Opaque cursor from the previous page")
        ] = None,
        limit: Annotated[
            int,
            Field(
                ge=1,
                le=_MAX_LIST_LIMIT,
                description="Number of resources to return, from 1 to 100",
            ),
        ] = _DEFAULT_LIST_LIMIT,
    ) -> McpResourceListResult:
        """List one host-aggregated page of MCP resources."""
        self._validate_limit(limit)
        providers = await self._mcp_providers(ctx)
        if server is not None:
            providers = [provider for provider in providers if provider.server_name == server]
            if not providers:
                return self._resource_list_result(
                    "No MCP server is registered with that name.",
                    errors=[
                        ResourceError(
                            "unknown_server",
                            f"Unknown MCP server: {server}",
                            False,
                            "Use list_mcp_resources without server to discover names.",
                        )
                    ],
                )
        provider_index = 0
        upstream_cursor: str | None = None
        offset = 0
        if cursor:
            try:
                decoded = self._decode_cursor(cursor)
            except (ValueError, TypeError) as exc:
                return self._resource_list_result(
                    "The supplied resource cursor is invalid.",
                    errors=[
                        ResourceError(
                            "invalid_cursor",
                            str(exc),
                            False,
                            "Restart pagination without a cursor.",
                        )
                    ],
                )
            if decoded.get("server") != server:
                return self._resource_list_result(
                    "The cursor does not belong to the requested server.",
                    errors=[
                        ResourceError(
                            "invalid_cursor",
                            "cursor/server mismatch",
                            False,
                            "Restart pagination with the original server argument.",
                        )
                    ],
                )
            provider_index, upstream_cursor, offset = self._cursor_state(decoded)
            if provider_index >= len(providers):
                return self._resource_list_result(
                    "The supplied resource cursor is no longer valid.",
                    errors=[
                        ResourceError(
                            "invalid_cursor",
                            "cursor provider position is outside the current provider set",
                            False,
                            "Restart pagination without a cursor.",
                        )
                    ],
                )
            current_server = decoded.get("current_server")
            if current_server != providers[provider_index].server_name:
                return self._resource_list_result(
                    "The supplied resource cursor is no longer valid.",
                    errors=[
                        ResourceError(
                            "invalid_cursor",
                            "cursor current server does not match provider position",
                            False,
                            "Restart pagination without a cursor.",
                        )
                    ],
                )

        entries: list[ResourceEntry] = []
        errors: list[ResourceError] = []
        next_cursor: str | None = None
        while provider_index < len(providers) and len(entries) < limit:
            provider = providers[provider_index]
            try:
                if not await provider.supports_resources():
                    errors.append(
                        ResourceError(
                            "resources_not_supported",
                            f"Server {provider.server_name} did not declare resources capability",
                            False,
                            "Use its tools or configure a server with resources capability.",
                        )
                    )
                    provider_index += 1
                    upstream_cursor = None
                    offset = 0
                    continue
                page = await provider.list_resources_page(
                    upstream_cursor if isinstance(upstream_cursor, str) else None
                )
            except (OSError, RuntimeError, TimeoutError, ValueError) as exc:
                errors.append(
                    ResourceError(
                        "provider_unavailable",
                        f"Failed to list resources from {provider.server_name}: {exc}",
                        True,
                        "Retry the same request later.",
                    )
                )
                provider_index += 1
                upstream_cursor = None
                offset = 0
                continue
            available = page.entries[offset:]
            take = min(limit - len(entries), len(available))
            entries.extend(available[:take])
            offset += take
            if offset < len(page.entries):
                next_cursor = self._encode_cursor(
                    server=server,
                    current_server=provider.server_name,
                    provider_index=provider_index,
                    upstream_cursor=upstream_cursor if isinstance(upstream_cursor, str) else None,
                    offset=offset,
                )
                break
            if page.next_cursor:
                next_cursor = self._encode_cursor(
                    server=server,
                    current_server=provider.server_name,
                    provider_index=provider_index,
                    upstream_cursor=page.next_cursor,
                    offset=0,
                )
                if len(entries) >= limit:
                    break
                upstream_cursor = page.next_cursor
                offset = 0
                continue
            provider_index += 1
            upstream_cursor = None
            offset = 0
            if len(entries) >= limit:
                break
        if next_cursor is None and provider_index < len(providers):
            next_cursor = self._encode_cursor(
                server=server,
                current_server=providers[provider_index].server_name,
                provider_index=provider_index,
                upstream_cursor=upstream_cursor if isinstance(upstream_cursor, str) else None,
                offset=offset,
            )
        summary = f"Returned {len(entries)} MCP resource(s)"
        return self._resource_list_result(
            summary, resources=entries, next_cursor=next_cursor, errors=errors
        )

    @logfire.instrument("capability.resource_capability.list_mcp_resource_templates")
    async def list_mcp_resource_templates(  # noqa: PLR0915
        self,
        ctx: RunContext[AgentDepsT],
        server: Annotated[
            str | None, Field(description="Optional configured MCP server name")
        ] = None,
        cursor: Annotated[
            str | None, Field(description="Opaque cursor from the previous page")
        ] = None,
        limit: Annotated[
            int,
            Field(
                ge=1,
                le=_MAX_LIST_LIMIT,
                description="Number of templates to return, from 1 to 100",
            ),
        ] = _DEFAULT_LIST_LIMIT,
    ) -> McpResourceTemplateListResult:
        """List one host-aggregated page of MCP resource templates."""
        self._validate_limit(limit)
        providers = await self._mcp_providers(ctx)
        if server is not None:
            providers = [provider for provider in providers if provider.server_name == server]
            if not providers:
                return self._template_list_result(
                    "No MCP server is registered with that name.",
                    templates=[],
                    errors=[
                        ResourceError(
                            "unknown_server",
                            f"Unknown MCP server: {server}",
                            False,
                            "Use list_mcp_resource_templates without server to discover names.",
                        )
                    ],
                )
        provider_index = 0
        upstream_cursor = None
        offset = 0
        if cursor:
            try:
                decoded = self._decode_cursor(cursor)
            except (ValueError, TypeError) as exc:
                return self._template_list_result(
                    "The supplied resource cursor is invalid.",
                    templates=[],
                    errors=[
                        ResourceError(
                            "invalid_cursor",
                            str(exc),
                            False,
                            "Restart pagination without a cursor.",
                        )
                    ],
                )
            if decoded.get("server") != server:
                return self._template_list_result(
                    "The cursor does not belong to the requested server.",
                    templates=[],
                    errors=[
                        ResourceError(
                            "invalid_cursor",
                            "cursor/server mismatch",
                            False,
                            "Restart pagination with the original server argument.",
                        )
                    ],
                )
            provider_index, upstream_cursor, offset = self._cursor_state(decoded)
            if provider_index >= len(providers):
                return self._template_list_result(
                    "The supplied resource cursor is no longer valid.",
                    templates=[],
                    errors=[
                        ResourceError(
                            "invalid_cursor",
                            "cursor provider position is outside the current provider set",
                            False,
                            "Restart pagination without a cursor.",
                        )
                    ],
                )
            current_server = decoded.get("current_server")
            if current_server != providers[provider_index].server_name:
                return self._template_list_result(
                    "The supplied resource cursor is no longer valid.",
                    templates=[],
                    errors=[
                        ResourceError(
                            "invalid_cursor",
                            "cursor current server does not match provider position",
                            False,
                            "Restart pagination without a cursor.",
                        )
                    ],
                )
        entries: list[ResourceTemplateEntry] = []
        errors: list[ResourceError] = []
        next_cursor: str | None = None
        while provider_index < len(providers) and len(entries) < limit:
            provider = providers[provider_index]
            try:
                if not await provider.supports_resources():
                    errors.append(
                        ResourceError(
                            "resources_not_supported",
                            f"Server {provider.server_name} did not declare resources capability",
                            False,
                            "Use its tools or configure a server with resources capability.",
                        )
                    )
                    provider_index += 1
                    upstream_cursor = None
                    offset = 0
                    continue
                page = await provider.list_resource_templates_page(
                    upstream_cursor if isinstance(upstream_cursor, str) else None
                )
            except (OSError, RuntimeError, TimeoutError, ValueError) as exc:
                errors.append(
                    ResourceError(
                        "provider_unavailable",
                        f"Failed to list resource templates from {provider.server_name}: {exc}",
                        True,
                        "Retry the same request later.",
                    )
                )
                provider_index += 1
                upstream_cursor = None
                offset = 0
                continue
            available = page.entries[offset:]
            take = min(limit - len(entries), len(available))
            entries.extend(available[:take])
            offset += take
            if offset < len(page.entries):
                next_cursor = self._encode_cursor(
                    server=server,
                    current_server=provider.server_name,
                    provider_index=provider_index,
                    upstream_cursor=upstream_cursor if isinstance(upstream_cursor, str) else None,
                    offset=offset,
                )
                break
            if page.next_cursor:
                next_cursor = self._encode_cursor(
                    server=server,
                    current_server=provider.server_name,
                    provider_index=provider_index,
                    upstream_cursor=page.next_cursor,
                    offset=0,
                )
                if len(entries) >= limit:
                    break
                upstream_cursor = page.next_cursor
                offset = 0
                continue
            provider_index += 1
            upstream_cursor = None
            offset = 0
        if next_cursor is None and provider_index < len(providers):
            next_cursor = self._encode_cursor(
                server=server,
                current_server=providers[provider_index].server_name,
                provider_index=provider_index,
                upstream_cursor=upstream_cursor if isinstance(upstream_cursor, str) else None,
                offset=offset,
            )
        return self._template_list_result(
            f"Returned {len(entries)} MCP resource template(s)",
            templates=entries,
            next_cursor=next_cursor,
            errors=errors,
        )

    @staticmethod
    def _resource_list_result(
        summary: str,
        *,
        resources: list[ResourceEntry] | None = None,
        next_cursor: str | None = None,
        errors: list[ResourceError] | None = None,
    ) -> McpResourceListResult:
        return McpResourceListResult(
            summary=summary,
            resources=resources or [],
            next_cursor=next_cursor,
            errors=errors or [],
        )

    @staticmethod
    def _template_list_result(
        summary: str,
        *,
        templates: list[ResourceTemplateEntry] | None = None,
        next_cursor: str | None = None,
        errors: list[ResourceError] | None = None,
    ) -> McpResourceTemplateListResult:
        return McpResourceTemplateListResult(
            summary=summary,
            templates=templates or [],
            next_cursor=next_cursor,
            errors=errors or [],
        )

    @logfire.instrument("capability.resource_capability.read_mcp_resource")
    async def read_mcp_resource(  # noqa: PLR0915
        self,
        ctx: RunContext[AgentDepsT],
        server: Annotated[str, Field(description="Configured MCP server name")],
        uri: Annotated[
            str, Field(description="Opaque resource URI copied from MCP list/search output")
        ],
    ) -> ToolReturn[McpResourceReadResult]:
        """Read one resource from the named MCP server without URI rewriting."""
        providers = await self._mcp_providers(ctx)
        provider = next((item for item in providers if item.server_name == server), None)
        if provider is None:
            error = ResourceError(
                "unknown_server",
                f"Unknown MCP server: {server}",
                False,
                "Use list_mcp_resources to discover server names.",
            )
            return self._read_error(uri, error)
        try:
            if not await provider.supports_resources():
                error = ResourceError(
                    "resources_not_supported",
                    f"Server {server} did not declare resources capability",
                    False,
                    "Use the server's tools instead.",
                )
                return self._read_error(uri, error)
            contents = await provider.read_mcp_resource(uri)
        except PermissionError as exc:
            error = ResourceError(
                "permission_denied",
                str(exc),
                False,
                "Request a URI permitted by the upstream server.",
            )
            return self._read_error(uri, error)
        except TimeoutError as exc:
            error = ResourceError("timeout", str(exc), True, "Retry the read later.")
            return self._read_error(uri, error)
        except (OSError, RuntimeError, ValueError) as exc:
            error = ResourceError(
                "provider_unavailable",
                str(exc),
                True,
                "Retry the read or inspect the provider status.",
            )
            return self._read_error(uri, error)
        if not contents:
            error = ResourceError(
                "resource_not_found",
                f"Resource not found: {uri}",
                False,
                "Copy the URI exactly from the server's list or search result.",
            )
            return self._read_error(uri, error)

        model_contents: list[JsonObject] = []
        visible_content: list[str | BinaryContent] = []
        errors: list[ResourceError] = []
        truncated = False
        original_char_count = 0
        for content in contents:
            if isinstance(content, TextResourceContent):
                original_char_count += len(content.text)
                text = content.text
                if len(text) > _DEFAULT_READ_TEXT_LIMIT:
                    text = text[:_DEFAULT_READ_TEXT_LIMIT]
                    truncated = True
                content_entry = self._content_entry_with_meta(
                    {
                        "type": "text",
                        "uri": content.uri,
                        "mime_type": content.mime_type,
                        "text": text,
                    },
                    content.meta,
                )
                model_contents.append(content_entry)
                visible_content.append(text)
            elif isinstance(content, BlobResourceContent):
                try:
                    blob = base64.b64decode(content.blob, validate=True)
                except (binascii.Error, ValueError):
                    error = ResourceError(
                        "unsupported_mime_type",
                        "Provider returned invalid base64 content",
                        False,
                        "Ask the provider for a supported binary resource.",
                    )
                    return self._read_error(uri, error)
                mime_type = content.mime_type or ""
                if mime_type not in _SUPPORTED_BLOB_MIME_TYPES:
                    content_entry = self._content_entry_with_meta(
                        {
                            "type": "blob",
                            "uri": content.uri,
                            "mime_type": mime_type,
                            "size": len(blob),
                            "attached": False,
                            "omission_reason": "unsupported_mime_type",
                        },
                        content.meta,
                    )
                    model_contents.append(content_entry)
                    errors.append(
                        ResourceError(
                            "unsupported_mime_type",
                            f"Binary MIME type is not supported: {mime_type or 'unknown'}",
                            False,
                            "Request a PDF, GIF, JPEG, PNG, or WebP resource.",
                        )
                    )
                    continue
                if len(blob) > _MAX_BLOB_BYTES:
                    content_entry = self._content_entry_with_meta(
                        {
                            "type": "blob",
                            "uri": content.uri,
                            "mime_type": mime_type,
                            "size": len(blob),
                            "attached": False,
                            "omission_reason": "content_too_large",
                        },
                        content.meta,
                    )
                    model_contents.append(content_entry)
                    errors.append(
                        ResourceError(
                            "content_too_large",
                            f"Binary resource exceeds {_MAX_BLOB_BYTES} bytes",
                            False,
                            "Read a smaller resource or request metadata only.",
                        )
                    )
                    continue
                content_entry = self._content_entry_with_meta(
                    {
                        "type": "blob",
                        "uri": content.uri,
                        "mime_type": mime_type,
                        "size": len(blob),
                        "attached": True,
                    },
                    content.meta,
                )
                model_contents.append(content_entry)
                visible_content.append(BinaryContent(data=blob, media_type=mime_type))
        return ToolReturn(
            return_value=McpResourceReadResult(
                summary=f"Read MCP resource {uri} from {server}",
                uri=uri,
                contents=model_contents,
                truncated=truncated,
                original_char_count=original_char_count or None,
                errors=errors,
            ),
            content=visible_content,
        )

    @staticmethod
    def _read_error(uri: str, error: ResourceError) -> ToolReturn[McpResourceReadResult]:
        result = McpResourceReadResult(
            summary=error.message,
            uri=uri,
            errors=[error],
        )
        return ToolReturn(return_value=result, content=error.message)

    @staticmethod
    def _content_entry_with_meta(
        entry: JsonObject,
        meta: JsonObject | None,
    ) -> JsonObject:
        """Attach upstream content metadata without changing other fields."""
        if meta is not None:
            entry["meta"] = meta
        return entry

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

        # Aggregate, deduplicate by URI, and identify sources.
        seen_uris: set[str] = set()
        source_entries: list[tuple[str, ResourceEntry]] = []
        for cap, result in zip(provider_caps, gathered, strict=True):
            if isinstance(result, BaseException):
                logfire.warning(
                    "Failed to list resources from {source}",
                    source=type(cap).__name__,
                )
                continue
            # Use server_name if available, otherwise fall back to the
            # first owned scheme or the class name.
            source = (
                getattr(cap, "server_name", None)
                or (next(iter(cap.owned_schemes)) if cap.owned_schemes else "")
                or type(cap).__name__
            )
            for entry in result:
                if entry.uri in seen_uris:
                    logfire.warning(
                        "Duplicate resource URI '{uri}' from {source} (skipped)",
                        uri=entry.uri,
                        source=source,
                    )
                    continue
                seen_uris.add(entry.uri)
                source_entries.append((source, entry))

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

        content = await resolve_resource_content(
            uri, resource_caps, skill_caps, scheme_registry=registry.scheme_registry
        )
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
